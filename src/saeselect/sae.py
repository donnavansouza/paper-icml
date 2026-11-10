"""Gemma Scope 2 JumpReLU SAEs: loading, encode, decode, error.

File layout [L, checked 2026-09-28 at google/gemma-scope-2-12b-it revision
4c419f1b, folder resid_post/layer_31_width_65k_l0_medium]:
- ``config.json``: ``architecture: jump_relu``, ``hf_hook_point_in:
  model.layers.{L}.output`` (the output of decoder block L, our layer index L),
  ``width``, ``l0``, ``affine_connection: false``.
- ``params.safetensors``: ``w_enc [d_model, width]``, ``w_dec [width, d_model]``,
  ``b_enc [width]``, ``b_dec [d_model]``, ``threshold [width]``, all F32.
- ``examples.safetensors`` (about 1.8 GB) is never downloaded.

Encoder, as SAELens loads these files (``apply_b_dec_to_input: False``, no
activation normalization): pre = h W_enc + b_enc; a = relu(pre) * [pre > threshold].
Decoder: h_hat = a W_dec + b_dec. Error: eps(h) = h - h_hat. Everything is FP32.

Orientation is asserted from shapes against the model's d_model and the
declared width; check N13 fails if the SAE does not
reconstruct the layer it claims to read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from saeselect.provenance import EngineeringFailed

TENSOR_NAMES = ("w_enc", "w_dec", "b_enc", "b_dec", "threshold")
WIDTHS = {"16k": 16384, "65k": 65536, "262k": 262144, "1m": 1048576}


@dataclass
class JumpReLUSAE:
    W_enc: torch.Tensor  # [d, n]
    b_enc: torch.Tensor  # [n]
    W_dec: torch.Tensor  # [n, d]
    b_dec: torch.Tensor  # [d]
    threshold: torch.Tensor  # [n]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        d, n = self.W_enc.shape
        expected = {"W_dec": (n, d), "b_enc": (n,), "b_dec": (d,), "threshold": (n,)}
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise EngineeringFailed(
                    "sae_shapes", f"{name} has shape {tuple(getattr(self, name).shape)}, expected {shape}"
                )
        for name in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"):
            setattr(self, name, getattr(self, name).float())

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def width(self) -> int:
        return int(self.W_enc.shape[1])

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    def to(self, device: torch.device | str) -> "JumpReLUSAE":
        return JumpReLUSAE(
            W_enc=self.W_enc.to(device),
            b_enc=self.b_enc.to(device),
            W_dec=self.W_dec.to(device),
            b_dec=self.b_dec.to(device),
            threshold=self.threshold.to(device),
            meta=dict(self.meta),
        )

    def pre_activation(self, h: torch.Tensor) -> torch.Tensor:
        return h.to(self.device, torch.float32) @ self.W_enc + self.b_enc

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = self.pre_activation(h)
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(h))

    def error(self, h: torch.Tensor) -> torch.Tensor:
        return h.to(self.device, torch.float32) - self.reconstruct(h)

    def decoder_norms(self) -> torch.Tensor:
        return self.W_dec.norm(dim=1)


def from_tensors(tensors: dict[str, torch.Tensor], *, d_model: int, width: int, meta: dict[str, Any]) -> JumpReLUSAE:
    """Build an SAE from Gemma Scope 2 tensor names, asserting orientation."""
    missing = [name for name in TENSOR_NAMES if name not in tensors]
    if missing:
        raise EngineeringFailed("sae_tensors", f"missing tensors {missing}; found {sorted(tensors)}")
    w_enc, w_dec = tensors["w_enc"], tensors["w_dec"]
    if tuple(w_enc.shape) != (d_model, width):
        raise EngineeringFailed("sae_orientation", f"w_enc shape {tuple(w_enc.shape)}, expected {(d_model, width)}")
    if tuple(w_dec.shape) != (width, d_model):
        raise EngineeringFailed("sae_orientation", f"w_dec shape {tuple(w_dec.shape)}, expected {(width, d_model)}")
    return JumpReLUSAE(
        W_enc=w_enc,
        b_enc=tensors["b_enc"],
        W_dec=w_dec,
        b_dec=tensors["b_dec"],
        threshold=tensors["threshold"],
        meta=meta,
    )


def folder_name(site: str, layer: int, width: str, l0: str, suffix: str = "") -> str:
    """``resid_post/layer_{L}_width_{W}_l0_{S}[_suffix]``."""
    name = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
    return f"{name}_{suffix}" if suffix else name


def load_gemma_scope(
    *,
    repo_id: str,
    revision: str,
    site: str,
    layer: int,
    width: str,
    l0: str,
    d_model: int,
    suffix: str = "",
    device: str = "cpu",
) -> JumpReLUSAE:
    """Download (or reuse the cache of) one Gemma Scope 2 SAE at a pinned revision."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if len(revision) != 40:
        raise EngineeringFailed("sae_revision", f"revision must be a pinned 40-character SHA, got {revision!r}")
    folder = folder_name(site, layer, width, l0, suffix)
    config_path = hf_hub_download(repo_id, f"{folder}/config.json", revision=revision)
    params_path = hf_hub_download(repo_id, f"{folder}/params.safetensors", revision=revision)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("architecture") != "jump_relu":
        raise EngineeringFailed("sae_config", f"{folder}: architecture {config.get('architecture')!r}, expected jump_relu")
    if config.get("affine_connection"):
        raise EngineeringFailed("sae_config", f"{folder}: affine_connection is set; not a plain SAE")
    expected_hook = f"model.layers.{layer}.output"
    if config.get("hf_hook_point_in") != expected_hook:
        raise EngineeringFailed(
            "sae_config", f"{folder}: hook point {config.get('hf_hook_point_in')!r}, expected {expected_hook!r}"
        )
    n = WIDTHS[width]
    if int(config.get("width", n)) != n:
        raise EngineeringFailed("sae_config", f"{folder}: width {config.get('width')} != {n}")
    tensors = load_file(params_path, device=device)
    meta = {
        "repo_id": repo_id,
        "revision": revision,
        "folder": folder,
        "layer": layer,
        "width": width,
        "l0": l0,
        "config": config,
        "params_path": str(params_path),
    }
    return from_tensors(tensors, d_model=d_model, width=n, meta=meta)


# --------------------------------------------------------------------------
# Llama Scope (X2: Llama 3.1 8B base; fnlp/Llama3_1-8B-Base-LXR-8x)
# --------------------------------------------------------------------------
# File layout [L, checked 2026-10-02 at revision 8dbc1d85]: one folder per layer,
# ``Llama3_1-8B-Base-L{L}R-8x/hyperparams.json`` (``act_fn: jumprelu`` with one
# scalar ``jump_relu_threshold``, ``hook_point_in: blocks.{L}.hook_resid_post``,
# ``norm_activation: dataset-wise`` with ``dataset_average_activation_norm``,
# ``top_k``) and ``checkpoints/final.safetensors`` (nn.Linear layout:
# ``encoder.weight [n, d]``, ``encoder.bias [n]``, ``decoder.weight [d, n]``,
# ``decoder.bias [d]``; SAELens reads the same names).
#
# How the stored threshold and weights relate to raw residuals is not settled by
# the sources: the Llama Scope report says the released weights were rescaled for
# the input normalization and have unit-norm decoder columns, while SAELens
# folds sqrt(d) / avg_norm into the weights and multiplies the threshold by it.
# The loader therefore takes a declared ``convention`` = (input_scaled, p):
#   input_scaled: the stored encoder expects s * h and the stored decoder
#     produces s * h, with s = sqrt(d) / dataset_average_activation_norm.in;
#   p: the threshold on the pre-activation is jump_relu_threshold * s**p.
# Stage 0 decides it from neutral text by a rule declared in the model config:
# one convention for the whole release, over every input form and p, the one
# whose mean L0 is nearest the report's average K on average over the layers
# (ties to the higher variance explained); every later stage loads it.

LLAMA_SCOPE_TENSORS = ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias")


def llama_scope_scale(hyperparams: dict[str, Any]) -> float:
    return float(hyperparams["d_model"]) ** 0.5 / float(hyperparams["dataset_average_activation_norm"]["in"])


def llama_scope_sae(tensors: dict[str, torch.Tensor], hyperparams: dict[str, Any], *, input_scaled: bool, p: int,
                    meta: dict[str, Any]) -> JumpReLUSAE:
    """A raw-space JumpReLU SAE from Llama Scope tensors under the convention (input_scaled, p)."""
    missing = [name for name in LLAMA_SCOPE_TENSORS if name not in tensors]
    if missing:
        raise EngineeringFailed("sae_tensors", f"missing tensors {missing}; found {sorted(tensors)}")
    d, n = int(hyperparams["d_model"]), int(hyperparams["d_sae"])
    enc, dec = tensors["encoder.weight"].float(), tensors["decoder.weight"].float()
    if tuple(enc.shape) != (n, d) or tuple(dec.shape) != (d, n):
        raise EngineeringFailed("sae_orientation", f"encoder {tuple(enc.shape)} / decoder {tuple(dec.shape)}, expected {(n, d)} / {(d, n)}")
    if p not in (-1, 0, 1):
        raise EngineeringFailed("sae_config", f"threshold power {p} not in (-1, 0, 1)")
    s = llama_scope_scale(hyperparams)
    w_in = s if input_scaled else 1.0
    threshold = float(hyperparams["jump_relu_threshold"]) * s ** p
    return JumpReLUSAE(
        W_enc=enc.T * w_in, b_enc=tensors["encoder.bias"].float(),
        W_dec=dec.T / w_in, b_dec=tensors["decoder.bias"].float() / w_in,
        threshold=torch.full((n,), threshold, dtype=torch.float32, device=enc.device),
        meta={**meta, "convention": {"input_scaled": input_scaled, "p": p}, "scale": s, "threshold_used": threshold},
    )


def load_llama_scope(*, repo_id: str, revision: str, folder: str, layer: int, d_model: int, convention: dict[str, Any],
                     device: str = "cpu", local_dir: str | None = None) -> JumpReLUSAE:
    """One Llama Scope SAE at a pinned revision (or from ``local_dir`` in tests), under ``convention``."""
    from safetensors.torch import load_file

    def fetch(name: str) -> str:
        if local_dir:
            return str(Path(local_dir) / folder / name)
        from huggingface_hub import hf_hub_download

        if len(revision) != 40:
            raise EngineeringFailed("sae_revision", f"revision must be a pinned 40-character SHA, got {revision!r}")
        return hf_hub_download(repo_id, f"{folder}/{name}", revision=revision)

    hyperparams = json.loads(Path(fetch("hyperparams.json")).read_text(encoding="utf-8"))
    if hyperparams.get("act_fn") != "jumprelu":
        raise EngineeringFailed("sae_config", f"{folder}: act_fn {hyperparams.get('act_fn')!r}, expected jumprelu")
    expected_hook = f"blocks.{layer}.hook_resid_post"
    if hyperparams.get("hook_point_in") != expected_hook or hyperparams.get("hook_point_out") != expected_hook:
        raise EngineeringFailed("sae_config", f"{folder}: hook points {hyperparams.get('hook_point_in')!r}/{hyperparams.get('hook_point_out')!r}, expected {expected_hook!r}")
    if int(hyperparams["d_model"]) != d_model:
        raise EngineeringFailed("sae_config", f"{folder}: d_model {hyperparams['d_model']} != the model's {d_model}")
    if hyperparams.get("apply_decoder_bias_to_pre_encoder") or hyperparams.get("use_glu_encoder"):
        raise EngineeringFailed("sae_config", f"{folder}: pre-encoder decoder bias or GLU encoder is set; not a plain SAE")
    tensors = load_file(fetch("checkpoints/final.safetensors"), device=device)
    meta = {"repo_id": repo_id, "revision": revision, "folder": folder, "layer": layer, "format": "llama_scope",
            "hyperparams": hyperparams, "top_k": hyperparams.get("top_k")}
    return llama_scope_sae(tensors, hyperparams, input_scaled=bool(convention["input_scaled"]), p=int(convention["p"]), meta=meta)
