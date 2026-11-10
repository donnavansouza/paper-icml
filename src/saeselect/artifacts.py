"""SAEs and J-lens per model config, Hub listings without downloads, and the disk check.

One entry point per artifact so every stage loads the same files the same way:
``load_sae_for`` and ``load_lens_for`` resolve a configs/models/*.yaml mapping
(``source: hf`` at pinned revisions, or ``source: tiny-random`` for tests).

``plan_downloads`` sizes every file a run will need that is not yet in the
Hugging Face cache and compares the total with the free disk space, so a run
fails before downloading anything that would not fit (the GPU machine had
33 GB free on 2026-09-28).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saeselect.lens import JLens, load_lens_file
from saeselect.provenance import EngineeringFailed
from saeselect.sae import JumpReLUSAE, folder_name, load_gemma_scope, load_llama_scope


def sae_configs(model_cfg: dict[str, Any], *, include_fallback: bool) -> list[dict[str, Any]]:
    """The candidate SAE configurations of a model: primary width and L0 at every
    candidate layer, plus the declared fallback if asked."""
    sae = model_cfg["sae"]
    out = [{"layer": layer, "width": sae["width"], "l0": sae.get("l0"), "suffix": ""} for layer in sae["candidate_layers"]]
    if include_fallback and sae.get("fallback"):
        fb = sae["fallback"]
        out += [{"layer": layer, "width": fb["width"], "l0": fb["l0"], "suffix": fb.get("suffix", ""), "fallback": True}
                for layer in sae["candidate_layers"]]
    return out


def sae_format(model_cfg: dict[str, Any]) -> str:
    """``gemma_scope`` (Gemma Scope 2, the default) or ``llama_scope`` (X2)."""
    return model_cfg["sae"].get("format", "gemma_scope")


def sae_folder(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> str:
    sae = model_cfg["sae"]
    if sae_format(model_cfg) == "llama_scope":
        return sae["folder_pattern"].format(layer=cfg["layer"])
    return folder_name(sae["site"], cfg["layer"], cfg["width"], cfg["l0"], cfg.get("suffix", ""))


def sae_files(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    folder = sae_folder(model_cfg, cfg)
    if sae_format(model_cfg) == "llama_scope":
        return [f"{folder}/hyperparams.json", f"{folder}/checkpoints/final.safetensors"]
    return [f"{folder}/config.json", f"{folder}/params.safetensors"]


def load_sae_for(model_cfg: dict[str, Any], *, layer: int, width: Any, l0: str | None, d_model: int, suffix: str = "", device: str = "cpu",
                 convention: dict[str, Any] | None = None, repo_id: str | None = None, revision: str | None = None) -> JumpReLUSAE:
    """``convention`` applies to Llama Scope SAEs only: the one Stage 0 decided (stage0_decisions ``sae.<layer>.convention``).
    ``repo_id`` and ``revision`` load a Gemma Scope 2 SAE from another release than the model's own (the SAE swap)."""
    sae = model_cfg["sae"]
    if sae.get("source") == "tiny-random" or model_cfg.get("source") == "tiny-random":
        from saeselect import tiny

        return tiny.basis_sae(d_model, int(width), seed=1000 + layer, layer=layer).to(device)
    if sae_format(model_cfg) == "llama_scope":
        if convention is None:
            raise EngineeringFailed("sae_convention", f"layer {layer}: a Llama Scope SAE needs the convention Stage 0 decided")
        return load_llama_scope(repo_id=sae["repo_id"], revision=sae["revision"], folder=sae_folder(model_cfg, {"layer": layer}), layer=layer,
                                d_model=d_model, convention=convention, device=device, local_dir=sae.get("local_dir"))
    return load_gemma_scope(repo_id=repo_id or sae["repo_id"], revision=revision or sae["revision"], site=sae["site"], layer=layer, width=width,
                            l0=l0, d_model=d_model, suffix=suffix, device=device)


def lens_path_for(model_cfg: dict[str, Any]) -> str:
    from huggingface_hub import hf_hub_download

    lens = model_cfg["lens"]
    return hf_hub_download(lens["repo_id"], lens["filename"], revision=lens["revision"])


def load_lens_for(model_cfg: dict[str, Any], *, d_model: int, n_layers: int) -> JLens:
    if model_cfg.get("source") == "tiny-random":
        from saeselect import tiny

        return tiny.random_lens(d_model, list(range(n_layers - 1)), seed=77)
    lens = model_cfg["lens"]
    return load_lens_file(lens_path_for(model_cfg), meta={"repo_id": lens["repo_id"], "revision": lens["revision"], "filename": lens["filename"]})


# --------------------------------------------------------------------------
# Hub listings (no downloads)
# --------------------------------------------------------------------------


def list_sae_folders(model_cfg: dict[str, Any]) -> list[str]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    sae = model_cfg["sae"]
    if sae_format(model_cfg) == "llama_scope":  # one folder per layer at the repository root
        return sorted(e.path for e in HfApi().list_repo_tree(sae["repo_id"], revision=sae["revision"]) if isinstance(e, RepoFolder))
    entries = HfApi().list_repo_tree(sae["repo_id"], path_in_repo=sae["site"], revision=sae["revision"])
    return sorted(e.path.split("/", 1)[1] for e in entries if isinstance(e, RepoFolder))


def sae_header(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Tensor names, shapes and dtypes of one SAE file, read from its header by range request."""
    from huggingface_hub import HfApi

    sae = model_cfg["sae"]
    folder = sae_folder(model_cfg, cfg)
    weights = sae_files(model_cfg, cfg)[1]
    meta = HfApi().parse_safetensors_file_metadata(sae["repo_id"], weights, revision=sae["revision"])
    tensors = {name: {"dtype": info.dtype, "shape": list(info.shape)} for name, info in meta.tensors.items()}
    return {"folder": folder, "tensors": tensors, "threshold_present": "threshold" in tensors}


# --------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------


@dataclass
class NeededFile:
    repo_id: str
    filename: str
    revision: str
    size: int
    cached: bool
    repo_type: str = "model"


def _cached(repo_id: str, filename: str, revision: str, repo_type: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo_id, filename, revision=revision, repo_type=repo_type)
    return isinstance(hit, str) and os.path.isfile(hit)


def file_sizes(repo_id: str, filenames: list[str], revision: str, repo_type: str = "model") -> dict[str, int]:
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(repo_id, filenames, revision=revision, repo_type=repo_type)
    return {info.path: int(getattr(info, "size", 0) or 0) for info in infos}


def model_weight_files(model_cfg: dict[str, Any]) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(model_cfg["repo_id"], revision=model_cfg["revision"])
    return [f for f in files if f.endswith(".safetensors") or f.endswith(".json") or f.startswith("tokenizer")]


def plan_downloads(model_cfg: dict[str, Any], *, sae_cfgs: list[dict[str, Any]], include_model: bool, include_lens: bool) -> list[NeededFile]:
    needed: list[NeededFile] = []
    if include_model:
        files = model_weight_files(model_cfg)
        for name, size in file_sizes(model_cfg["repo_id"], files, model_cfg["revision"]).items():
            needed.append(NeededFile(model_cfg["repo_id"], name, model_cfg["revision"], size, _cached(model_cfg["repo_id"], name, model_cfg["revision"], "model")))
    if include_lens:
        lens = model_cfg["lens"]
        for name, size in file_sizes(lens["repo_id"], [lens["filename"]], lens["revision"]).items():
            needed.append(NeededFile(lens["repo_id"], name, lens["revision"], size, _cached(lens["repo_id"], name, lens["revision"], "model")))
    sae = model_cfg.get("sae") or {}  # a checkpoint used only for Stage 1 has no SAE
    by_repo: dict[tuple[str, str], list[str]] = {}
    for cfg in sae_cfgs:  # a variant may name another release
        key = (cfg.get("repo_id") or sae["repo_id"], cfg.get("revision") or sae["revision"])
        by_repo.setdefault(key, []).extend(sae_files(model_cfg, cfg))
    for (repo, rev), names in by_repo.items():
        for name, size in file_sizes(repo, names, rev).items():
            needed.append(NeededFile(repo, name, rev, size, _cached(repo, name, rev, "model")))
    return needed


def hf_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def check_disk(needed: list[NeededFile], *, margin_bytes: int) -> dict[str, Any]:
    """Refuse (EngineeringFailed) if the uncached files would leave less than
    ``margin_bytes`` free on the disk that holds the Hugging Face cache."""
    cache = hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    to_download = sum(f.size for f in needed if not f.cached)
    report = {
        "cache_dir": cache.as_posix(),
        "free_bytes": free,
        "to_download_bytes": to_download,
        "margin_bytes": margin_bytes,
        "files": [{"repo": f.repo_id, "file": f.filename, "size": f.size, "cached": f.cached} for f in needed],
    }
    if free - to_download < margin_bytes:
        missing = [f"{f.repo_id}/{f.filename} ({f.size / 1e9:.2f} GB)" for f in needed if not f.cached]
        raise EngineeringFailed(
            "disk",
            f"downloading {to_download / 1e9:.1f} GB would leave {(free - to_download) / 1e9:.1f} GB free "
            f"(< {margin_bytes / 1e9:.0f} GB margin) on {cache.as_posix()}. Not cached: {missing}",
        )
    return report
