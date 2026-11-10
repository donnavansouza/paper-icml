"""Stage 0: inventory and numerics.

Writes ``artifacts.json`` (model, lens, SAEs, tokenizer), one entry per check
N1 to N13 (value, rule, pass) in ``report.json``, feature statistics and kappa
per usable SAE (``features/*.safetensors``), ``chat_snapshots.md``, and
``stage0_decisions.json`` (usable SAE per layer, precision mode, whether
batching and suffix caching are allowed, tau_tie per task family). Every
derived value also goes to ``derived.json`` / ``decisions_entry.md``.

``--inventory-only`` reads configs, tokenizer, lens listing and SAE tensor
headers from the Hub without downloading weights (the 12B-PT pass).

``scope: stage1_screen`` (a config with ``base:`` and overrides) runs
only what Stage 1 needs on a checkpoint: the model weights, N11, N1, N4, N5 and
N6 (tau_tie). No lens or SAE is downloaded; the other checks run when a stage
that needs them comes to this checkpoint (X3 for 12B-PT).

``scope: sae_variants`` (X4) measures the model config's declared robustness
SAEs at ``sae_variant_layers``: N13/N3 values, feature statistics and kappa,
which Stage 2's extension mode reads.

Failures: every check runs and is reported; if any fails, the run ends with
ENGINEERING_FAILED naming the first failing check, after all files are written.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from saeselect import artifacts as A
from saeselect import lens as L
from saeselect import stats
from saeselect.hooks import EditSpec, ResidualEdits, ResidualRecorder, hook_count
from saeselect.interventions import candidate_pool, transfer_activation_delta
from saeselect.models import load_model, precision_mode
from saeselect.patching import PatchEvaluator, as_ids, clean_run, margin_gradient
from saeselect.provenance import EngineeringFailed
from saeselect.readout import AnswerSets, answer_margin, logits_fp32, margin_bf16_head
from saeselect.runner import Outcome, RunContext
from saeselect.tasks.records import hash_unit, load_records
from saeselect.tasks.tokenization import encode_prompt, render_chat, tokenizer_fingerprint

GB = 1e9


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _check(value: Any, rule: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"value": value, "rule": rule, "pass": bool(passed), **details}


class _SavedChecks(dict):
    """Check results, written to ``checks.json`` each time one is recorded, so a
    crash later in the run (the first 12B run ran out of memory) keeps them."""

    def __init__(self, run):
        super().__init__()
        self._run = run

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        super().__setitem__(key, value)
        self._run.write_json("checks.json", dict(self))


def _answers(record) -> AnswerSets:
    return AnswerSets(plus=tuple(record.answer_sets["A_plus"]["ids"]), minus=tuple(record.answer_sets["A_minus"]["ids"]))


def _sample(items: list, n: int, *, key: str, seed: int) -> list:
    return sorted(items, key=lambda r: (hash_unit(f"{key}:{getattr(r, 'id', r)}", seed), str(getattr(r, "id", r))))[:n]


def _load_family(data_dir: Path, family: str):
    card = json.loads((data_dir / family / "benchmark_card.json").read_text(encoding="utf-8"))
    records = load_records(data_dir / family / card["records_file"], phase="dev", expected_sha256=card["records_sha256"])
    return card, records


def _neutral_windows(data_dir: Path, n: int) -> list[list[int]]:
    card = json.loads((data_dir / "neutral" / "benchmark_card.json").read_text(encoding="utf-8"))
    path = data_dir / "neutral" / card["items_file"]
    from saeselect.provenance import sha256_file

    if sha256_file(path) != card["records_sha256"]:
        raise EngineeringFailed("neutral_corpus", f"{path} does not match its benchmark card")
    windows = [json.loads(line)["token_ids"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return windows[:n]


def _trials(records) -> list[tuple[Any, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        groups.setdefault(r.matched_group_id, {})[r.role] = r
    return [(g["source"], g["donor"]) for _, g in sorted(groups.items()) if "source" in g and "donor" in g]


def _site(record, site: str) -> list[int]:
    return [record.meta["readout_position"]] if site == "readout" else list(record.meta["concept_positions"])


def _cuda_mem(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    return {"max_allocated_gb": torch.cuda.max_memory_allocated(device) / GB, "max_reserved_gb": torch.cuda.max_memory_reserved(device) / GB}


def _reset_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _free(*_unused: Any) -> None:
    """Collect garbage and return cached GPU memory; callers ``del`` their own references first."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def inventory(model_cfg: dict[str, Any], tokenizer, *, hub: bool) -> dict[str, Any]:
    inv: dict[str, Any] = {"model": {k: model_cfg.get(k) for k in ("name", "source", "repo_id", "revision", "dtype", "checkpoint_kind")}}
    inv["tokenizer"] = tokenizer_fingerprint(tokenizer)
    if getattr(tokenizer, "chat_template", None):
        probe = render_chat(tokenizer, [{"role": "system", "content": "SYSTEM-PROBE"}, {"role": "user", "content": "USER-PROBE"}])
        inv["chat_template"] = {
            "present": True,
            "separate_system_turn": "system" in probe.split("SYSTEM-PROBE")[0].split("<start_of_turn>")[-1],
            "system_merged_into_first_user_turn": probe.find("SYSTEM-PROBE") < probe.find("USER-PROBE") and "<start_of_turn>user" in probe.split("SYSTEM-PROBE")[0],
            "rendered_probe": probe,
            "bos_in_rendered_text": probe.startswith(tokenizer.bos_token or "\0"),
        }
    else:  # pretrained checkpoints may ship without a chat template; chat records then come from the IT template text
        inv["chat_template"] = {"present": False, "separate_system_turn": None, "rendered_probe": ""}
    if hub and model_cfg.get("source") == "hf" and model_cfg.get("lens"):  # none on a checkpoint used only for Stage 1
        lens = model_cfg["lens"]
        inv["lens"] = {**lens, "size_bytes": A.file_sizes(lens["repo_id"], [lens["filename"]], lens["revision"]).get(lens["filename"])}
    if hub and model_cfg.get("source") == "hf" and (model_cfg.get("sae") or {}).get("repo_id"):
        folders = A.list_sae_folders(model_cfg)
        inv["sae"] = {"repo_id": model_cfg["sae"]["repo_id"], "revision": model_cfg["sae"]["revision"], "site": model_cfg["sae"]["site"],
                      "folders": folders, "candidates": [A.sae_header(model_cfg, c) for c in A.sae_configs(model_cfg, include_fallback=True)]}
    return inv


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    if "base" in cfg:  # a scoped config: the base declared values with its overrides
        from saeselect.config import load_config

        cfg = {**load_config(cfg["base"]), **{k: v for k, v in cfg.items() if k != "base"}}
    model_cfg = ctx.model_config
    if model_cfg is None:
        raise EngineeringFailed("config", "stage0 needs --model configs/models/<model>.yaml")
    screen = cfg.get("scope", "full") == "stage1_screen"
    checks_cfg = cfg["checks"]
    seed = cfg["seed"]
    data_dir = Path(cfg["data_dir"])
    limit = ctx.limit
    lim = (lambda n: n if limit is None else min(n, limit))

    if ctx.options.get("inventory_only"):
        return _inventory_only(ctx, model_cfg)
    if cfg.get("scope") == "sae_variants":
        return _sae_variants(ctx, cfg, model_cfg)

    device = torch.device(ctx.device)
    hub = model_cfg.get("source") == "hf"
    sae_cfgs = [] if screen else A.sae_configs(model_cfg, include_fallback=False)
    disk = None
    if hub:
        needed = A.plan_downloads(model_cfg, sae_cfgs=sae_cfgs, include_model=True, include_lens=not screen)
        disk = A.check_disk(needed, margin_bytes=int(cfg["disk"]["margin_gb"] * GB))
    t0 = time.time()
    handle = load_model(model_cfg, device=ctx.device)
    load_seconds = time.time() - t0
    tok = handle.tokenizer
    # the SAE layers; a checkpoint without SAEs (Stage 1 only) declares the layer of the N4 sham edit instead
    layers = list((model_cfg.get("sae") or {}).get("candidate_layers") or model_cfg["probe_layers"])
    checks: dict[str, dict[str, Any]] = _SavedChecks(ctx.run)
    report: dict[str, Any] = {"load_seconds": load_seconds, "disk": disk}

    inv = inventory(model_cfg, tok, hub=hub)
    inv["model"].update(handle.info)
    ctx.run.write_json("artifacts.json", inv)

    lens, missing_layers = None, []
    if not screen:
        lens = A.load_lens_for(model_cfg, d_model=handle.d_model, n_layers=handle.n_layers)
        inv["lens_file"] = {"layers_present": lens.layers, "stored_dtype": lens.stored_dtype, "n_prompts": lens.n_prompts, "d_model": lens.d_model}
        missing_layers = [layer for layer in layers if layer not in lens.layers]
        ctx.run.write_json("artifacts.json", inv)

    windows = _neutral_windows(data_dir, lim(checks_cfg["N13"]["n_windows"]))
    families = {}
    for family in cfg["families"]:
        families[family] = _load_family(data_dir, family)

    # ---- N11 tokenization -------------------------------------------------
    n11 = {}
    for family, (card, records) in families.items():
        same_vocab = card["tokenizer"]["vocab_sha256"] == inv["tokenizer"]["vocab_sha256"]
        sample = _sample(list(records), lim(checks_cfg["N6"]["n_prompts_per_family"]), key=f"N11:{family}", seed=seed)
        bos_ok = True
        for r in sample:
            try:
                ids = encode_prompt(tok, r.prompt, r.prompt_format)
                bos_ok &= len(ids) == r.meta["n_tokens"]
            except EngineeringFailed:
                bos_ok = False
        n11[family] = {"same_vocab_as_data_build": same_vocab, "one_bos_and_lengths_match": bos_ok}
    snapshots = ["# Chat rendering snapshots (N11)", "", "## System-turn probe", "", "```", inv["chat_template"]["rendered_probe"], "```", ""]
    ctx.run.write_text("chat_snapshots.md", "\n".join(snapshots))
    n11_pass = all(v["same_vocab_as_data_build"] and v["one_bos_and_lengths_match"] for v in n11.values())
    checks["N11"] = _check(n11, "exactly one BOS, encodings match the data build, same vocabulary", n11_pass)

    # ---- N1 HF forward vs jlens-wrapped forward ---------------------------------
    try:
        import jlens

        lens_model = jlens.from_hf(handle.hf_model, tok)
        diffs = []
        for window in windows[: lim(checks_cfg["N1"]["n_prompts"])]:
            ids = as_ids(window, device)
            last = handle.n_layers - 1
            with torch.no_grad(), ResidualRecorder(handle.layers, at=[last]) as rec:
                handle.hf_model(input_ids=ids)
            hf_final = rec.activations[last][0]
            with torch.no_grad(), ResidualRecorder(handle.layers, at=[last]) as rec:
                lens_model.forward(ids)
            jl_final = rec.activations[last][0]
            diffs.append(float((logits_fp32(hf_final, handle.head) - logits_fp32(jl_final, handle.head)).abs().max()))
        v = max(diffs)
        checks["N1"] = _check(v, f"max |logit difference| <= {checks_cfg['N1']['max_abs_logit_diff']}", v <= checks_cfg["N1"]["max_abs_logit_diff"])
    except ImportError:
        lens_model = None
        checks["N1"] = _check(None, "jlens must be installed", False)

    # ---- N2 our lens readout vs lens.apply --------------------------------------
    if screen:
        pass  # no lens on a Stage 1 screen
    elif lens_model is not None and not missing_layers:
        import jlens

        ref = jlens.JacobianLens(jacobians={layer: lens.jacobians[layer] for layer in layers}, n_prompts=lens.n_prompts, d_model=lens.d_model)
        worst, top_ok = 0.0, True
        k = checks_cfg["N2"]["top_k"]
        for window in windows[: lim(checks_cfg["N2"]["n_prompts"])]:
            text = tok.decode(window[1:])
            lens_logits, _, input_ids = ref.apply(lens_model, text, layers=layers)
            acts, _ = clean_run(handle, input_ids.to(device), layers, mode="bf16")
            for layer in layers:
                ours = L.jlens_unembed(L.transport(acts[layer], lens.J(layer, device)), handle.final_norm, handle.lm_head, handle.head.softcap).float().cpu()
                worst = max(worst, float((ours - lens_logits[layer]).abs().max()))
                top_ok &= bool(torch.equal(ours.topk(k).indices, lens_logits[layer].topk(k).indices))
        tol = checks_cfg["N2"]["max_abs_logit_diff"]
        checks["N2"] = _check({"max_abs_logit_diff": worst, "top_k_identical": top_ok}, f"identical top-{k}; max |diff| <= {tol}", top_ok and worst <= tol)
    else:
        checks["N2"] = _check({"missing_layers": missing_layers}, "lens has every candidate layer and jlens is installed", False)

    # ---- N4 sham, N5 cleanup ----------------------------------------------------
    rel_family = next(f for f in cfg["families"] if f.startswith("R"))
    trials_all = _trials(families[rel_family][1])
    src, _don = trials_all[0]
    ids = as_ids(encode_prompt(tok, src.prompt, src.prompt_format), device)
    last = handle.n_layers - 1
    with torch.no_grad(), ResidualRecorder(handle.layers, at=[last]) as rec:
        handle.forward(ids)
    clean_final = rec.activations[last].clone()
    zero = torch.zeros(len(src.meta["concept_positions"]), handle.d_model)
    with torch.no_grad(), ResidualRecorder(handle.layers, at=[last]) as rec, ResidualEdits(handle.layers, [EditSpec(layer=layers[0], positions=src.meta["concept_positions"], delta=zero)]):
        handle.forward(ids)
    checks["N4"] = _check(bool(torch.equal(rec.activations[last], clean_final)), "output bit-identical", torch.equal(rec.activations[last], clean_final))

    def boom(h):
        raise RuntimeError("N5 probe")

    try:
        with torch.no_grad(), ResidualRecorder(handle.layers, at=[last]), ResidualEdits(handle.layers, [EditSpec(layer=layers[0], positions=[1], delta_fn=boom)]):
            handle.forward(ids)
    except RuntimeError:
        pass
    left = hook_count(handle.hf_model)
    checks["N5"] = _check(left, "no hooks remain registered", left == 0)

    # ---- N6 FP32 vs BF16 head -> tau_tie per family -----------------------------------
    tau: dict[str, float] = {}
    n6 = {}
    for family, (_, records) in families.items():
        diffs = []
        for r in _sample(list(records), lim(checks_cfg["N6"]["n_prompts_per_family"]), key=f"N6:{family}", seed=seed):
            _, h_final = clean_run(handle, as_ids(encode_prompt(tok, r.prompt, r.prompt_format), device), [], mode="bf16")
            row = h_final[r.meta["readout_position"]]
            ans = _answers(r)
            diffs.append(abs(float(answer_margin(row, handle.head, ans)) - float(margin_bf16_head(row, handle.head, ans))))
        q = float(np.quantile(diffs, checks_cfg["N6"]["quantile"]))
        tau[family] = ctx.derive(f"tau_tie.{family}", q, procedure=f"{checks_cfg['N6']['quantile']} quantile of |m_FP32 - m_BF16 head| at the readout of dev prompts (5.2, N6)",
                                 inputs={"family": family, "n_prompts": len(diffs), "model": model_cfg.get("name")},
                                 numbers={"median": float(np.median(diffs)), "max": float(max(diffs))})
        n6[family] = {"n": len(diffs), "tau_tie": q, "median": float(np.median(diffs)), "max": float(max(diffs))}
    checks["N6"] = _check(n6, "recorded; defines tau_tie per family", True)
    if screen:
        return _screen_outcome(ctx, checks, tau, model_cfg, inv)

    # ---- N3, N13 SAE alignment and fidelity; feature statistics; kappa ----------------
    # Per-layer policy: the D2 fallback replaces a
    # primary SAE below the N13 floor; if the fallback is also below it, the one of the two
    # that explains more variance is kept and every result at
    # that layer is flagged low-fidelity. A layer whose kept SAE fails N3 (margin over its
    # neighbors < the declared 0.05) is excluded from SAE work on this model. Stage 0
    # continues while at least one layer is usable. One SAE is on the GPU at a time.
    floor = checks_cfg["N13"]["min_variance_explained"]
    margin_rule = checks_cfg["N3"]["min_margin_over_neighbors"]
    usable: dict[int, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    low_fidelity: list[int] = []
    n3, n13 = {}, {}
    features_dir = ctx.run.file("features")
    features_dir.mkdir(exist_ok=True)
    def load(option):
        return A.load_sae_for(model_cfg, layer=layer, width=option["width"], l0=option["l0"], d_model=handle.d_model,
                              suffix=option.get("suffix", ""), device=ctx.device, convention=option.get("convention"))

    conventions: dict[str, Any] = {}
    release_convention = None
    if A.sae_format(model_cfg) == "llama_scope" and model_cfg.get("source") == "hf":  # X2: decided before N13 (sae.py)
        if hub:
            A.check_disk(A.plan_downloads(model_cfg, sae_cfgs=A.sae_configs(model_cfg, include_fallback=True), include_model=False,
                                          include_lens=False), margin_bytes=int(cfg["disk"]["margin_gb"] * GB))
        conventions = _llama_scope_convention(handle, model_cfg, layers, windows, device)
        release_convention = conventions["convention"]
        ctx.derive(f"llama_scope_convention.{model_cfg.get('name')}", release_convention,
                   procedure="One convention per release: over input form x threshold power, the convention "
                             "minimizing the mean over candidate layers of |mean L0 on neutral text - K|; ties to the higher mean variance "
                             "explained, then the smaller |p| (sae.convention_rule)",
                   inputs={"model": model_cfg.get("name"), "layers": layers, "n_windows": len(windows)}, numbers=conventions)
    for layer in layers:
        options = [c for c in A.sae_configs(model_cfg, include_fallback=True) if c["layer"] == layer]
        if release_convention is not None:
            options = [{**c, "convention": release_convention} for c in options]
        tried = []  # (option, key, fidelity statistics on the CPU, matched VE, N3 margin)
        sae = None
        for option in options:
            if hub:
                A.check_disk(A.plan_downloads(model_cfg, sae_cfgs=[option], include_model=False, include_lens=False), margin_bytes=int(cfg["disk"]["margin_gb"] * GB))
            sae = load(option)
            fidelity = _sae_fidelity(handle, sae, layer, windows, device)
            fidelity.update({k: fidelity[k].cpu() for k in ("mean", "sd", "firing_rate")})
            key = f"L{layer}_{option['width']}_{option['l0']}{('_' + option['suffix']) if option.get('suffix') else ''}"
            ve_by_layer = fidelity["variance_explained"]
            matched = ve_by_layer[layer]
            neighbors = [ve_by_layer[x] for x in (layer - 1, layer + 1) if x in ve_by_layer]
            margin = min(matched - n for n in neighbors) if neighbors else None
            n3[key] = {"variance_explained_by_layer": {str(x): v for x, v in ve_by_layer.items()}, "beats_neighbors_by": margin}
            n13[key] = {"variance_explained": matched, "decoder_norms": fidelity["decoder_norms"], "mean_l0": fidelity["mean_l0"]}
            ctx.run.write_json("sae_progress.json", {"N3": n3, "N13": n13})  # kept if a later step fails
            tried.append((option, key, fidelity, matched, margin))
            if matched >= floor:
                break  # this one is kept (the primary, or D2's fallback)
            del sae  # D2: try the declared fallback, one SAE on the GPU at a time
            sae = None
            _free()
        option, key, fidelity, matched, margin = tried[-1] if tried[-1][3] >= floor else max(tried, key=lambda t: t[3])
        entry = {**option, "variance_explained": matched, "fallback_used": bool(option.get("fallback")), "key": key,
                 "low_fidelity": matched < floor, "n3_margin": margin}
        if matched < floor:
            low_fidelity.append(layer)
            others = ", ".join(f"{t[1]} {t[3]:.3f}" for t in tried if t[1] != key)
            ctx.flags.append(f"N13: layer {layer} kept with {key} below the {floor} floor (VE {matched:.3f}; also tried: {others}); "
                             "results there are low-fidelity")
        if margin is None or margin < margin_rule:
            excluded[str(layer)] = f"N3: {key} beats its neighbors by {margin if margin is None else round(margin, 4)} < {margin_rule}"
            ctx.flags.append(f"N3: layer {layer} excluded from SAE work on this model ({excluded[str(layer)]})")
        else:
            usable[layer] = entry
            if sae is None:
                sae = load(option)  # the better of primary and fallback was released after its fidelity check
            sae.W_enc = None  # kappa needs the decoder only; frees 4 GB for a 262k SAE on 12B
            _free()
            J = lens.J(layer, device)
            kappa_j = L.kappa(sae.W_dec, handle.head, J)
            kappa_i = L.kappa(sae.W_dec, handle.head, None)
            from safetensors.torch import save_file

            save_file({"mean": fidelity["mean"], "sd": fidelity["sd"], "firing_rate": fidelity["firing_rate"],
                       "kappa_jlens": kappa_j.cpu(), "kappa_logit_lens": kappa_i.cpu()},
                      str(features_dir / f"{key}.safetensors"), metadata={"layer": str(layer), "config": json.dumps(option), "n_tokens": str(fidelity["n_tokens"])})
            del J, kappa_j, kappa_i
        del sae
        _free()
    checks["N3"] = _check(n3, f"matched layer beats layers l-1 and l+1 by >= {margin_rule}; a failing layer is excluded", bool(usable),
                          outcome={str(layer): "excluded" if str(layer) in excluded else "ok" for layer in layers}, excluded=excluded)
    checks["N13"] = _check({"per_config": n13, "usable": {str(k): v["key"] for k, v in usable.items()}},
                           f"variance explained >= {floor}, else the declared fallback; if both are below the floor, the one explaining "
                           "more is kept, flagged", bool(usable),
                           outcome={str(layer): "low_fidelity" if layer in low_fidelity else "ok" for layer in layers}, low_fidelity=low_fidelity)
    if not usable:
        raise EngineeringFailed("N3", f"no candidate layer has a usable SAE: {excluded}")

    # ---- N7 precision audit ------------------------------------------------------
    n7_cfg = checks_cfg["N7"]
    active = [f for f in cfg["families"] if f.startswith("R")]
    trial_pool = [t for f in active for t in _trials(families[f][1])]
    chosen = _sample(trial_pool, lim(n7_cfg["n_trials"]), key="N7", seed=seed)
    usable_layers = sorted(usable)
    by_layer: dict[int, list[int]] = {}
    for i, (s, d) in enumerate(chosen):
        by_layer.setdefault(usable_layers[i % len(usable_layers)], []).append(i)
    plan: dict[int, dict[str, Any]] = {}
    for layer, idxs in by_layer.items():
        u = usable[layer]
        sae = A.load_sae_for(model_cfg, layer=layer, width=u["width"], l0=u["l0"], d_model=handle.d_model, suffix=u.get("suffix", ""), device=ctx.device,
                             convention=u.get("convention"))
        for i in idxs:
            s, d = chosen[i]
            site = n7_cfg["sites"][(i // len(usable_layers)) % len(n7_cfg["sites"])]
            pos = _site(s, site)
            ids_s = as_ids(encode_prompt(tok, s.prompt, s.prompt_format), device)
            ids_d = as_ids(encode_prompt(tok, d.prompt, d.prompt_format), device)
            acts_s, _ = clean_run(handle, ids_s, [layer], mode="bf16")
            acts_d, _ = clean_run(handle, ids_d, [layer], mode="bf16")
            h_s, h_d = acts_s[layer][pos].float(), acts_d[layer][pos].float()
            da = transfer_activation_delta(sae, h_s, h_d)
            pool = candidate_pool(da).tolist()
            feats = _sample(pool, lim(n7_cfg["n_features_per_trial"]), key=f"N7:{s.id}", seed=seed)
            deltas = [(da[:, [f]] @ sae.W_dec[[f]]).cpu() for f in feats]
            plan[i] = {"layer": layer, "site": site, "positions": pos, "ids": ids_s, "answers": _answers(s), "features": feats, "deltas": deltas}
            if i == 0:  # N10 picks its features by |ATP| from this trial's whole candidate pool
                plan[i]["pool_features"] = pool
                plan[i]["pool_deltas"] = [(da[:, [f]] @ sae.W_dec[[f]]).cpu() for f in pool]
        del sae
        _free()
    n7 = _precision_audit(handle, model_cfg, plan, n7_cfg, ctx)
    precision = "bf16" if n7["bf16_passes"] else "fp32_residual"
    checks["N7"] = _check(n7["summary"], f">= {n7_cfg['min_share_in_band']:.0%} of realized/intended in {n7_cfg['ratio_band']} and Spearman >= {n7_cfg['min_spearman']} "
                          "(else FP32-residual mode)", True, decision=precision)
    ctx.derive("precision_mode", precision, procedure="N7 rule (6.1): BF16 residual if its norm ratios and effect ranking match the FP32 reference, else FP32-residual mode",
               inputs={"model": model_cfg.get("name"), "reference": model_cfg.get("precision_reference")}, numbers=n7["summary"])
    if precision == "fp32_residual":
        checks["N1_fp32_residual"], checks["N4_fp32_residual"] = _fp32_residual_rechecks(handle, windows, src, layers, device, checks_cfg, lim)
    # A model config may declare the gradient mode (fp32_grad on Llama after its second N10
    # failure); otherwise gradients run in N7's mode. N10 and N12 test the mode that gradients will use.
    gradient = model_cfg.get("gradient_mode") or precision
    # Causal measurements run in exact FP32; N7 still sets the mode of
    # gradient computations (ATP), which fp32_full cannot run. Sham check in the causal mode.
    causal = cfg.get("causal_mode", precision)
    if causal == "fp32_full":
        checks["N4_fp32_full"] = _sham_check(handle, src, layers, device, "fp32_full")
    checkpointed = checks_cfg["N12"].get("backward") == "checkpointed"

    # ---- N10 (gradient mode), N9 and N8 (causal mode) ----------------------------------------
    p = plan[0]
    checks["N10"] = _atp_check(handle, p, gradient, checks_cfg["N10"], checkpointed=checkpointed)
    evaluator_sets = [[EditSpec(layer=p["layer"], positions=p["positions"], delta=delta)] for delta in p["deltas"][: lim(checks_cfg["N9"]["n_patches"])]]
    serial = PatchEvaluator(handle, p["ids"], p["answers"], mode=causal, strategy="serial").evaluate(evaluator_sets)
    batched = PatchEvaluator(handle, p["ids"], p["answers"], mode=causal, strategy="batched", batch_size=checks_cfg["N9"]["batch_size"]).evaluate(evaluator_sets)
    suffix = PatchEvaluator(handle, p["ids"], p["answers"], mode=causal, strategy="suffix").evaluate(evaluator_sets[: lim(checks_cfg["N8"]["n_patches"])])
    n9 = max(abs(a.margin - b.margin) for a, b in zip(serial, batched))
    n8 = max(abs(a.margin - b.margin) for a, b in zip(serial, suffix))
    checks["N9"] = _check(n9, f"max |margin difference| <= {checks_cfg['N9']['max_abs_margin_diff']} in {causal} mode (else serial only)", True,
                          allowed=n9 <= checks_cfg["N9"]["max_abs_margin_diff"], site=p["site"], mode=causal)
    checks["N8"] = _check(n8, f"max |margin difference| <= {checks_cfg['N8']['max_abs_margin_diff']} in {causal} mode (else suffix caching disabled)", True,
                          allowed=n8 <= checks_cfg["N8"]["max_abs_margin_diff"], site=p["site"], mode=causal)

    # ---- N12 memory and time -------------------------------------------------------------
    longest = max((c.get("prompt_tokens") or {}).get("max", 0) for c, _ in families.values())
    checks["N12"] = _memory_time(handle, windows, plan, gradient, causal, layers, checks_cfg["N12"], device, lim,
                                 longest_prompt=longest if limit is None else None)
    if "longest_prompt_oom" in checks["N12"]["value"]:
        ctx.flags.append(f"N12: out of memory at the longest data prompt ({longest} tokens); a declared truncation rule is needed (10.2)")

    decisions = {
        "model": model_cfg.get("name"),
        "smoke": ctx.smoke,
        "precision_mode": precision,  # N7's decision
        "gradient_mode": gradient,  # gradients (ATP, local-gradient diagnostic): N7's mode, or the model config's declared one (X2)
        "causal_mode": causal,  # every causal measurement (single-feature and set effects, references)
        "backward": "checkpointed" if checkpointed else "plain",
        "batching_allowed": checks["N9"]["allowed"],
        "batch_size": checks_cfg["N9"]["batch_size"],
        "suffix_caching_allowed": checks["N8"]["allowed"],
        "sae": {str(k): v for k, v in usable.items()},
        "excluded_layers": excluded,
        "low_fidelity_layers": low_fidelity,
        "tau_tie": tau,
        "sae_conventions": conventions,  # X2: Llama Scope only (empty for Gemma Scope)
        "features_dir": "features",
        "code_commit": ctx.git.get("commit"),
    }
    ctx.run.write_json("stage0_decisions.json", decisions)
    for key in ("batching_allowed", "suffix_caching_allowed"):
        ctx.derive(key, decisions[key], procedure=f"{'N9' if 'batch' in key else 'N8'} parity rule (6.1)", inputs={"model": model_cfg.get("name")},
                   numbers={"max_abs_margin_diff": checks["N9" if "batch" in key else "N8"]["value"]})
    ctx.run.write_json("artifacts.json", inv)
    report.update(checks=checks, decisions=decisions)

    summary = _summary(checks, decisions, model_cfg)
    failed = [name for name, c in checks.items() if not c["pass"]]
    if failed:
        ctx.run.write_json("checks.json", checks)
        raise EngineeringFailed(failed[0], f"checks failed: {failed}", [summary])
    return Outcome(status="PASS", summary=summary, report=report,
                   send_back=["artifacts.json", "stage0_decisions.json", "derived.json", "decisions_entry.md", "chat_snapshots.md"])


# --------------------------------------------------------------------------
# Check implementations
# --------------------------------------------------------------------------


def choose_convention(tables: dict[str, dict[str, dict[str, Any]]], targets: dict[str, float], powers: list[int]) -> dict[str, Any]:
    """One convention per release: among input form x threshold power, the convention minimizing
    the mean over layers of |mean L0 - K|; ties to the higher mean variance explained, then the smaller |p|.
    ``tables[layer]["input_scaled=<bool>,p=<int>"]`` holds ``mean_l0`` and ``variance_explained``."""
    grid = [(s, p) for s in (True, False) for p in powers]
    pooled = {}
    for s, p in grid:
        key = f"input_scaled={s},p={p}"
        gaps = [abs(tables[layer][key]["mean_l0"] - targets[layer]) for layer in tables]
        ves = [tables[layer][key]["variance_explained"] for layer in tables]
        pooled[key] = {"mean_abs_l0_gap": float(np.mean(gaps)), "mean_variance_explained": float(np.mean(ves))}
    best = min(grid, key=lambda sp: (pooled[f"input_scaled={sp[0]},p={sp[1]}"]["mean_abs_l0_gap"],
                                     -pooled[f"input_scaled={sp[0]},p={sp[1]}"]["mean_variance_explained"], abs(sp[1])))
    return {"convention": {"input_scaled": best[0], "p": best[1]}, "pooled": pooled}


@torch.no_grad()
def _llama_scope_convention(handle, model_cfg: dict[str, Any], layers: list[int], windows: list[list[int]], device: torch.device) -> dict[str, Any]:
    """X2 (sae.py): how the stored Llama Scope tensors map to raw residuals, one convention for the whole release
    (per-layer picks can disagree): at every candidate layer, mean L0
    and variance explained on neutral text (BOS excluded) under every input form (scaled or not) x threshold power p
    of the model config's ``sae.convention_rule``; then :func:`choose_convention` with K = each file's ``top_k``."""
    rule = model_cfg["sae"]["convention_rule"]
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    targets: dict[str, float] = {}
    for layer in layers:
        residuals = []
        for window in windows:
            acts, _ = clean_run(handle, as_ids(window, device), [layer], mode="bf16")
            residuals.append(acts[layer][1:].float())
        table = {}
        for s in (True, False):
            for p in rule["powers"]:
                sae = A.load_sae_for(model_cfg, layer=layer, width=None, l0=None, d_model=handle.d_model, device=str(device),
                                     convention={"input_scaled": s, "p": p})
                sse = sq = l0 = 0.0
                total = torch.zeros(handle.d_model, dtype=torch.float64, device=device)
                n = 0
                for h in residuals:
                    a = sae.encode(h)
                    sse += float((h - sae.decode(a)).pow(2).sum())
                    sq += float(h.pow(2).sum())
                    total += h.double().sum(0)
                    l0 += float((a > 0).sum())
                    n += h.shape[0]
                mean = total / n
                table[f"input_scaled={s},p={p}"] = {"variance_explained": 1.0 - sse / (sq - n * float(mean.pow(2).sum())),
                                                    "mean_l0": l0 / n, "threshold": sae.meta["threshold_used"], "top_k": sae.meta.get("top_k")}
                del sae
                _free()
        tables[str(layer)] = table
        targets[str(layer)] = next(iter(table.values()))["top_k"] if rule["target"] == "top_k" else rule["target"]
        del residuals
        _free()
    chosen = choose_convention(tables, targets, rule["powers"])
    return {**chosen, "target_l0": targets, "tables": tables}


@torch.no_grad()
def _sae_fidelity(handle, sae, layer: int, windows: list[list[int]], device: torch.device) -> dict[str, Any]:
    """Variance explained by ``sae`` on the residuals of layers l-1, l, l+1
    (BOS excluded), and feature statistics at layer l. VE = 1 - SSE / TSS,
    SSE = sum over tokens of ||h - h_hat||^2, TSS = sum ||h - mean(h)||^2
    (both squared norms; VE itself is a ratio, not squared again)."""
    test_layers = [x for x in (layer - 1, layer, layer + 1) if 0 <= x < handle.n_layers]
    sse = {x: 0.0 for x in test_layers}
    sq = {x: 0.0 for x in test_layers}
    total = {x: torch.zeros(handle.d_model, dtype=torch.float64, device=device) for x in test_layers}
    n_tokens = 0
    s1 = torch.zeros(sae.width, dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    fired = torch.zeros_like(s1)
    l0 = 0.0
    for window in windows:
        acts, _ = clean_run(handle, as_ids(window, device), test_layers, mode="bf16")
        for x in test_layers:
            h = acts[x][1:].float()  # BOS excluded
            a = sae.encode(h)
            sse[x] += float((h - sae.decode(a)).pow(2).sum())
            sq[x] += float(h.pow(2).sum())
            total[x] += h.double().sum(0)
            if x == layer:
                a64 = a.double()
                s1 += a64.sum(0)
                s2 += (a64 * a64).sum(0)
                fired += (a > 0).double().sum(0)
                l0 += float((a > 0).sum())
        n_tokens += len(window) - 1
    ve = {}
    for x in test_layers:
        mean = total[x] / n_tokens
        tss = sq[x] - n_tokens * float(mean.pow(2).sum())
        ve[x] = 1.0 - sse[x] / tss
    mean_a = s1 / n_tokens
    sd = (s2 / n_tokens - mean_a * mean_a).clamp_min(0).sqrt()
    norms = sae.decoder_norms()
    return {"variance_explained": ve, "mean": mean_a.float(), "sd": sd.float(), "firing_rate": (fired / n_tokens).float(),
            "n_tokens": n_tokens, "mean_l0": l0 / n_tokens,
            "decoder_norms": {"min": float(norms.min()), "median": float(norms.median()), "max": float(norms.max())}}


def _single_feature_effects(handle, plan, mode: str) -> tuple[list[list[float]], list[float]]:
    """Per trial, the effect m(T({i})) - m_clean of each planned single-feature
    transfer in ``mode``; and every realized/intended norm ratio."""
    effects, ratios = [], []
    for _, p in sorted(plan.items()):
        ev = PatchEvaluator(handle, p["ids"], p["answers"], mode=mode)
        clean = ev.clean_margin()
        res = ev.evaluate([[EditSpec(layer=p["layer"], positions=p["positions"], delta=d)] for d in p["deltas"]])
        effects.append([r.margin - clean for r in res])
        ratios += [r.realized_norm / r.intended_norm for r in res if r.intended_norm > 0]
    return effects, ratios


def _precision_audit(handle, model_cfg, plan, n7_cfg, ctx) -> dict[str, Any]:
    """N7: BF16-mode effects and norm ratios against the declared reference
    (4B: a full-FP32 load; 12B: FP32-residual mode). Also reported, not gating: FP32-residual effects against exact FP32 (the full-FP32 load on 4B,
    the layer-wise ``fp32_full`` mode on 12B), since FP32-residual mode is the
    precision later stages use when BF16 fails."""
    per_trial = {i: {"layer": p["layer"], "site": p["site"], "n_features": len(p["deltas"])} for i, p in sorted(plan.items())}
    bf16_effects, ratios = _single_feature_effects(handle, plan, "bf16")
    reference = model_cfg.get("precision_reference", "fp32_residual")
    if reference == "full_fp32":
        fp32_model = load_model(model_cfg, device=str(handle.device), dtype="float32")
        ref_effects, _ = _single_feature_effects(fp32_model, plan, "bf16")  # "bf16" = as loaded, here FP32
        del fp32_model  # the full-FP32 copy (4B) leaves the GPU here
        _free()
        exact_effects = ref_effects
        residual_effects, _ = _single_feature_effects(handle, plan, "fp32_residual")
    else:
        ref_effects, _ = _single_feature_effects(handle, plan, "fp32_residual")
        residual_effects = ref_effects
        exact_effects, _ = _single_feature_effects(handle, plan, "fp32_full")

    def flat(x: list[list[float]]) -> list[float]:
        return [e for effects in x for e in effects]

    def rho(a, b) -> float:
        return stats.spearman(flat(a), flat(b)) if len(flat(a)) > 1 else float("nan")

    lo, hi = n7_cfg["ratio_band"]
    in_band = float(np.mean([lo <= r <= hi for r in ratios])) if ratios else 0.0
    rho_ref = rho(bf16_effects, ref_effects)
    passes = in_band >= n7_cfg["min_share_in_band"] and rho_ref >= n7_cfg["min_spearman"]
    diff = np.abs(np.array(flat(residual_effects)) - np.array(flat(exact_effects)))
    # Within-trial agreement is what Stage 2's M1 (Spearman over one trial's pool) depends on.
    within = [stats.spearman(a, b) for a, b in zip(residual_effects, exact_effects) if len(a) > 2]
    within = [w for w in within if not np.isnan(w)]
    summary = {"n_patches": len(ratios), "share_ratio_in_band": in_band, "ratio_quantiles": np.quantile(ratios, [0.01, 0.5, 0.99]).tolist() if ratios else None,
               "spearman_bf16_vs_reference": rho_ref, "reference": reference, "bf16_passes": passes,
               "mean_abs_effect_bf16": float(np.mean(np.abs(flat(bf16_effects)))) if ratios else None,
               "descriptive": {"spearman_fp32_residual_vs_exact_fp32": rho(residual_effects, exact_effects),
                               "within_trial_spearman_fp32_residual_vs_exact_fp32": {
                                   "median": float(np.median(within)), "q10": float(np.quantile(within, 0.1)), "n_trials": len(within)} if within else None,
                               "spearman_bf16_vs_exact_fp32": rho(bf16_effects, exact_effects),
                               "mean_abs_effect_exact_fp32": float(np.mean(np.abs(flat(exact_effects)))) if ratios else None,
                               "abs_diff_fp32_residual_vs_exact": {"median": float(np.median(diff)), "p99": float(np.quantile(diff, 0.99)), "max": float(diff.max())} if diff.size else None}}
    ctx.run.write_json("n7_precision_audit.json", {"summary": summary, "trials": per_trial, "bf16_effects": bf16_effects, "reference_effects": ref_effects,
                                                   "fp32_residual_effects": residual_effects, "exact_fp32_effects": exact_effects})
    return {"bf16_passes": passes, "summary": summary}


def _fp32_residual_rechecks(handle, windows, src_record, layers, device, checks_cfg, lim):
    """6.1: FP32-residual mode must pass N1 (against the plain forward in that mode) and N4."""
    last = handle.n_layers - 1
    ids = as_ids(windows[0], device)
    with torch.no_grad(), precision_mode(handle, "fp32_residual"), ResidualRecorder(handle.layers, at=[last]) as rec:
        handle.hf_model(input_ids=ids)
    a = logits_fp32(rec.activations[last][0], handle.head)
    with torch.no_grad(), precision_mode(handle, "fp32_residual"), ResidualRecorder(handle.layers, at=[last]) as rec:
        handle.forward(ids)
    b = logits_fp32(rec.activations[last][0], handle.head)
    d = float((a - b).abs().max())
    n1 = _check(d, f"max |logit difference| <= {checks_cfg['N1']['max_abs_logit_diff']}", d <= checks_cfg["N1"]["max_abs_logit_diff"])
    ids = as_ids(encode_prompt(handle.tokenizer, src_record.prompt, src_record.prompt_format), device)
    with torch.no_grad(), precision_mode(handle, "fp32_residual"), ResidualRecorder(handle.layers, at=[last]) as rec:
        handle.forward(ids)
    clean = rec.activations[last].clone()
    zero = torch.zeros(len(src_record.meta["concept_positions"]), handle.d_model)
    with torch.no_grad(), precision_mode(handle, "fp32_residual"), ResidualRecorder(handle.layers, at=[last]) as rec, \
            ResidualEdits(handle.layers, [EditSpec(layer=layers[0], positions=src_record.meta["concept_positions"], delta=zero)]):
        handle.forward(ids)
    same = bool(torch.equal(rec.activations[last], clean))
    return n1, _check(same, "output bit-identical", same)


def _sham_check(handle, src_record, layers, device, mode) -> dict[str, Any]:
    """N4 in ``mode``: a zero delta leaves the output bit-identical."""
    last = handle.n_layers - 1
    ids = as_ids(encode_prompt(handle.tokenizer, src_record.prompt, src_record.prompt_format), device)
    with torch.no_grad(), precision_mode(handle, mode), ResidualRecorder(handle.layers, at=[last]) as rec:
        handle.forward(ids)
    clean = rec.activations[last].clone()
    zero = torch.zeros(len(src_record.meta["concept_positions"]), handle.d_model)
    with torch.no_grad(), precision_mode(handle, mode), ResidualRecorder(handle.layers, at=[last]) as rec, \
            ResidualEdits(handle.layers, [EditSpec(layer=layers[0], positions=src_record.meta["concept_positions"], delta=zero)]):
        handle.forward(ids)
    same = bool(torch.equal(rec.activations[last], clean))
    return _check(same, "output bit-identical", same)


def _atp_check(handle, p, mode, n10_cfg, *, checkpointed: bool = False) -> dict[str, Any]:
    """N10: the ATP prediction g . delta_i, with g computed in the
    production precision ``mode``, against a central finite difference along
    delta_i computed in exact FP32 (``reference_mode``), where margins are a smooth
    function of the input. A finite difference in BF16-matmul precision is
    dominated by rounding. The features are the
    ``n_features`` with the largest |ATP| in the first N7 trial's whole candidate
    pool. A half-step finite difference is reported to show convergence."""
    m0, grad = margin_gradient(handle, p["ids"], p["layer"], p["answers"], readout=p["ids"].shape[1] - 1, mode=mode, checkpointed=checkpointed)
    pos = p["positions"]
    deltas, features = p["pool_deltas"], p["pool_features"]
    preds = [float((grad[pos] * d.to(grad.device)).sum()) for d in deltas]
    order = sorted(range(len(preds)), key=lambda j: -abs(preds[j]))[: n10_cfg["n_features"]]
    s = n10_cfg["fd_scale"]
    ev = PatchEvaluator(handle, p["ids"], p["answers"], mode=n10_cfg["reference_mode"])
    rows = []
    for j in order:
        edits = [[EditSpec(layer=p["layer"], positions=pos, delta=deltas[j], scale=sign * step)] for step in (s, s / 2) for sign in (1, -1)]
        plus, minus, plus_half, minus_half = ev.evaluate(edits)
        fd = (plus.margin - minus.margin) / (2 * s)
        fd_half = (plus_half.margin - minus_half.margin) / s
        rel = abs(fd - preds[j]) / max(abs(fd), abs(preds[j]), 1e-12)
        rows.append({"feature": features[j], "atp": preds[j], "finite_difference": fd, "finite_difference_half_step": fd_half, "rel_error": rel})
    worst = max(r["rel_error"] for r in rows) if rows else float("nan")
    return _check(worst, f"relative error <= {n10_cfg['max_rel_error']} on the {n10_cfg['n_features']} features with the largest |ATP| "
                  f"(ATP in {mode}; finite difference in {n10_cfg['reference_mode']}, step {s})",
                  bool(rows) and worst <= n10_cfg["max_rel_error"], features=rows, gradient_mode=mode, checkpointed=checkpointed,
                  reference_mode=n10_cfg["reference_mode"], pool_size=len(features))


def _memory_time(handle, windows, plan, mode, causal, layers, n12_cfg, device, lim, longest_prompt: int | None = None) -> dict[str, Any]:
    """N12 at the declared length and, as information, at the longest prompt in
    the data build if that is longer. Pass rule: the forward and the
    backward in the declared form (``backward: checkpointed``, which recomputes
    layers and gives the same gradient) fit at the declared length. The plain
    backward is measured too, after the checkpointed one so that its
    out-of-memory state cannot inflate the other reading. Each step is caught
    separately. Patch costs: in the causal mode serial, batched and
    suffix-cached; serial in the gradient mode for comparison."""
    out: dict[str, Any] = {}
    body = [t for w in windows for t in w[1:]]
    distinct = sorted(set(body))
    probe = AnswerSets(plus=(distinct[0],), minus=(distinct[1],))  # any two tokens: N12 times the backward pass, not a margin
    ok = True
    lengths = [("declared", n12_cfg["tokens"])]
    if longest_prompt and longest_prompt > n12_cfg["tokens"]:
        lengths.append(("longest_prompt", longest_prompt))

    def measure(key: str, fn) -> bool:
        try:
            _reset_mem(device)
            t = time.time()
            fn()
            out[key] = {"seconds": time.time() - t, "memory": _cuda_mem(device)}
            return True
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - GPU only
            out[key] = {"oom": True, "error": str(exc)[:300]}
            _free()
            return False

    for label, n in lengths:
        tokens = [windows[0][0]] + (body * (n // max(len(body), 1) + 1))[: n - 1]  # BOS + neutral text, repeated to n tokens
        ids = as_ids(tokens, device)
        prefix = "" if label == "declared" else "longest_prompt_"
        out[f"{prefix}tokens"] = n
        fwd = measure(f"{prefix}forward", lambda: clean_run(handle, ids, [], mode=mode))
        ckpt = measure(f"{prefix}forward_backward_checkpointed", lambda: margin_gradient(handle, ids, min(layers), probe, readout=n - 1, mode=mode, checkpointed=True))
        plain = measure(f"{prefix}forward_backward_to_layer", lambda: margin_gradient(handle, ids, min(layers), probe, readout=n - 1, mode=mode))
        fits = fwd and (ckpt if n12_cfg.get("backward") == "checkpointed" else plain)
        if label == "declared" and not fits:
            ok = False
            out["oom"] = {"tokens": n, "step": f"forward or {n12_cfg.get('backward', 'plain')} backward at the declared length"}
        if label == "longest_prompt" and not fits:
            out["longest_prompt_oom"] = {"tokens": n}
        del ids
        _free()
    p = plan[next(iter(sorted(plan)))]
    sets = [[EditSpec(layer=p["layer"], positions=p["positions"], delta=d)] for d in p["deltas"]]
    sets = (sets * (n12_cfg["n_patches"] // max(len(sets), 1) + 1))[: lim(n12_cfg["n_patches"])]
    batch = n12_cfg.get("batch_size", 8)
    sets = sets[: max(batch, len(sets) // batch * batch)]  # whole batches, so the batched timing compares like with like
    for label, strategy, patch_mode in ((f"serial_{causal}", "serial", causal), (f"batched_{causal}", "batched", causal),
                                        (f"suffix_{causal}", "suffix", causal), (f"serial_{mode}", "serial", mode)):
        _reset_mem(device)
        t = time.time()
        PatchEvaluator(handle, p["ids"], p["answers"], mode=patch_mode, strategy=strategy, batch_size=batch).evaluate(sets)
        out[f"patches_{label}"] = {"n": len(sets), "seconds": time.time() - t, "prompt_tokens": int(p["ids"].shape[1]), "mode": patch_mode,
                                   "memory": _cuda_mem(device)}
    return _check(out, f"forward and {n12_cfg.get('backward', 'plain')} backward fit at {n12_cfg['tokens']} tokens; "
                  "an out-of-memory error fails with the size that failed", ok)


# --------------------------------------------------------------------------
# Inventory only, the Stage 1 screen, and the summary
# --------------------------------------------------------------------------


def _screen_outcome(ctx: RunContext, checks, tau: dict[str, float], model_cfg: dict[str, Any], inv: dict[str, Any]) -> Outcome:
    """The checks a Stage 1 screen needs, and stage0_decisions.json with tau_tie."""
    decisions = {"model": model_cfg.get("name"), "smoke": ctx.smoke, "scope": "stage1_screen", "tau_tie": tau,
                 "code_commit": ctx.git.get("commit"),
                 "not_run": "N2, N3, N7 to N10, N12, N13, feature statistics and kappa: they run when a stage needs SAEs on this checkpoint"}
    ctx.run.write_json("stage0_decisions.json", decisions)
    ctx.run.write_json("artifacts.json", inv)
    lines = ["## What ran", "", f"Stage 0 scoped to the Stage 1 screen for `{model_cfg.get('name')}`: N11, N1, N4, N5, N6. "
             "No lens or SAE was downloaded.", "", "## Checks", "", "| Check | Value | Rule | Pass |", "|---|---|---|---|"]
    for name, c in checks.items():
        lines.append(f"| {name} | {_fmt(name, c)} | {c['rule'].replace('|', '&#124;')} | {'yes' if c['pass'] else 'NO'} |")
    lines += ["", "## Decisions (stage0_decisions.json)", "", "- tau_tie per family: " + ", ".join(f"{k} {v:.3g}" for k, v in tau.items()),
              "", "## What this permits", "", "Stage 1 (raw completions, then the dev phase) may run on this checkpoint. Values derived here "
              "bind only this checkpoint.", "", "## Next step", "", "Send back the files listed in the status line."]
    summary = "\n".join(lines) + "\n"
    failed = [name for name, c in checks.items() if not c["pass"]]
    if failed:
        raise EngineeringFailed(failed[0], f"checks failed: {failed}", [summary])
    return Outcome(status="PASS", summary=summary, report={"checks": dict(checks), "decisions": decisions},
                   send_back=["artifacts.json", "stage0_decisions.json", "derived.json", "decisions_entry.md", "chat_snapshots.md"])


def variant_key(option: dict[str, Any]) -> str:
    """L<layer>_<width>_<l0>[_<suffix>][_<tag>]: the tag names a variant from another release (e.g. pt_sae)."""
    key = f"L{option['layer']}_{option['width']}_{option['l0']}"
    for part in (option.get("suffix"), option.get("tag")):
        if part:
            key += f"_{part}"
    return key


def _sae_variants(ctx: RunContext, cfg: dict[str, Any], model_cfg: dict[str, Any]) -> Outcome:
    """X4 (7.2): the declared SAE variants (the model config's ``sae.robustness``
    list) at ``sae_variant_layers``, with N13 and N3 values and the feature
    statistics and kappa Stage 2 reads, so Stage 2 can rerun with only the SAE
    changed. Fidelity is reported per variant; a variant below the N13 floor is
    flagged, not dropped (X4 measures robustness to the dictionary)."""
    from safetensors.torch import save_file

    checks_cfg = cfg["checks"]
    lim = (lambda n: n if ctx.limit is None else min(n, ctx.limit))
    device = torch.device(ctx.device)
    variants = [{"layer": int(layer), "width": v["width"], "l0": v.get("l0"), "suffix": v.get("suffix", ""),
                 **{k: v[k] for k in ("repo_id", "revision", "tag") if v.get(k)}}
                for layer in cfg["sae_variant_layers"] for v in model_cfg["sae"][cfg.get("sae_variant_list", "robustness")]]
    disk = None
    if model_cfg.get("source") == "hf":
        needed = A.plan_downloads(model_cfg, sae_cfgs=variants, include_model=True, include_lens=True)
        disk = A.check_disk(needed, margin_bytes=int(cfg["disk"]["margin_gb"] * GB))
    handle = load_model(model_cfg, device=ctx.device)
    lens = A.load_lens_for(model_cfg, d_model=handle.d_model, n_layers=handle.n_layers)
    windows = _neutral_windows(Path(cfg["data_dir"]), lim(checks_cfg["N13"]["n_windows"]))
    floor = checks_cfg["N13"]["min_variance_explained"]
    features_dir = ctx.run.file("features")
    features_dir.mkdir(exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for option in variants:
        layer = option["layer"]
        sae = A.load_sae_for(model_cfg, layer=layer, width=option["width"], l0=option["l0"], d_model=handle.d_model,
                             suffix=option["suffix"], device=ctx.device, repo_id=option.get("repo_id"), revision=option.get("revision"))
        fidelity = _sae_fidelity(handle, sae, layer, windows, device)
        key = variant_key(option)
        ve = fidelity["variance_explained"]
        matched = ve[layer]
        neighbors = [ve[x] for x in (layer - 1, layer + 1) if x in ve]
        margin = min(matched - n for n in neighbors) if neighbors else None
        sae.W_enc = None  # kappa needs the decoder only
        _free()
        J = lens.J(layer, device)
        kappa_j = L.kappa(sae.W_dec, handle.head, J)
        kappa_i = L.kappa(sae.W_dec, handle.head, None)
        save_file({"mean": fidelity["mean"].cpu(), "sd": fidelity["sd"].cpu(), "firing_rate": fidelity["firing_rate"].cpu(),
                   "kappa_jlens": kappa_j.cpu(), "kappa_logit_lens": kappa_i.cpu()},
                  str(features_dir / f"{key}.safetensors"), metadata={"layer": str(layer), "config": json.dumps(option), "n_tokens": str(fidelity["n_tokens"])})
        entries[key] = {**option, "key": key, "variance_explained": matched, "n3_margin": margin, "low_fidelity": matched < floor,
                        "mean_l0": fidelity["mean_l0"], "variance_explained_by_layer": {str(x): v for x, v in ve.items()}}
        if matched < floor:
            ctx.flags.append(f"N13: {key} explains {matched:.3f} < {floor} (reported, not dropped: X4)")
        ctx.run.write_json("sae_progress.json", entries)
        del sae, J, kappa_j, kappa_i
        _free()
    decisions = {"model": model_cfg.get("name"), "smoke": ctx.smoke, "scope": "sae_variants", "sae_variants": entries,
                 "features_dir": "features", "code_commit": ctx.git.get("commit")}
    ctx.run.write_json("stage0_decisions.json", decisions)
    lines = ["## What ran", "", f"Stage 0 scoped to SAE variants (X4) for `{model_cfg.get('name')}`: N13 and N3 values, feature statistics "
             f"and kappa for each variant, on {len(windows)} neutral windows.", "",
             "| Variant | Variance explained | Margin over neighbors | Mean L0 | Below the N13 floor |", "|---|---|---|---|---|"]
    for k, e in entries.items():
        margin_txt = "n/a" if e["n3_margin"] is None else f"{e['n3_margin']:.3f}"
        lines.append(f"| {k} | {e['variance_explained']:.3f} | {margin_txt} | {e['mean_l0']:.1f} | {'yes' if e['low_fidelity'] else 'no'} |")
    lines += ["", "## Next step", "", "Send back the files in the status line."]
    return Outcome(status="PASS", summary="\n".join(lines) + "\n", report={"sae_variants": entries, "disk": disk},
                   send_back=["stage0_decisions.json"])


def _inventory_only(ctx: RunContext, model_cfg: dict[str, Any]) -> Outcome:
    from saeselect.tasks.tokenization import load_tokenizer

    tok = load_tokenizer(model_cfg)
    hub = model_cfg.get("source") == "hf"
    inv = inventory(model_cfg, tok, hub=hub)
    if hub:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_cfg["repo_id"], "config.json", revision=model_cfg["revision"])
        text_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        inv["model_config_json"] = text_cfg
        needed = A.plan_downloads(model_cfg, sae_cfgs=A.sae_configs(model_cfg, include_fallback=False), include_model=True, include_lens=True)
        inv["downloads_needed_for_full_stage0"] = {"bytes_not_cached": sum(f.size for f in needed if not f.cached),
                                                   "files": [{"file": f"{f.repo_id}/{f.filename}", "size": f.size, "cached": f.cached} for f in needed]}
    ctx.run.write_json("artifacts.json", inv)
    data_cards = Path(ctx.config["data_dir"])
    same = None
    card_path = data_cards / "R1_country_capital" / "benchmark_card.json"
    if card_path.is_file():
        same = json.loads(card_path.read_text(encoding="utf-8"))["tokenizer"]["vocab_sha256"] == inv["tokenizer"]["vocab_sha256"]
    lines = ["## What ran", "", f"Stage 0 inventory only for `{model_cfg.get('name')}` (no weights downloaded).", "",
             "## Results", "", f"- Tokenizer vocabulary identical to the data build's: {same}",
             f"- Chat template separate system turn: {inv['chat_template']['separate_system_turn']}"]
    if hub:
        lines += [f"- SAE folders listed: {len(inv['sae']['folders'])}",
                  f"- Not yet cached for a full Stage 0: {inv['downloads_needed_for_full_stage0']['bytes_not_cached'] / GB:.1f} GB"]
    flags = [] if same in (True, None) else ["tokenizer vocabulary differs from the data build: records must not be reused for this checkpoint"]
    return Outcome(status="PASS", summary="\n".join(lines) + "\n", report={"inventory": {k: v for k, v in inv.items() if k != "sae"}},
                   send_back=["artifacts.json"], flags=flags)


def _fmt(name: str, c: dict[str, Any]) -> str:
    """The number that decides each check, in plain words (details in report.json)."""
    v = c["value"]
    try:
        if name == "N2":
            return f"max diff {v['max_abs_logit_diff']:.2g}; top-k identical {v['top_k_identical']}"
        if name == "N3":
            margins = ", ".join(f"{k} {x['beats_neighbors_by']:.3f}" for k, x in v.items() if x["beats_neighbors_by"] is not None)
            dropped = ", ".join(f"layer {k}" for k in c.get("excluded", {}))
            return f"margins {margins}" + (f"; excluded: {dropped}" if dropped else "")
        if name == "N13":
            low = ", ".join(f"layer {k}" for k in c.get("low_fidelity", []))
            return "VE " + ", ".join(f"{k} {x['variance_explained']:.3f}" for k, x in v["per_config"].items()) + (f"; low-fidelity: {low}" if low else "")
        if name == "N10":
            return f"worst relative error {v:.3g} ({c.get('gradient_mode')} gradient vs {c.get('reference_mode')} finite difference)"
        if name == "N6":
            return "tau_tie " + ", ".join(f"{k.split('_')[0]} {x['tau_tie']:.2g}" for k, x in v.items())
        if name == "N7":
            d = v.get("descriptive") or {}
            within = (d.get("within_trial_spearman_fp32_residual_vs_exact_fp32") or {}).get("median", float("nan"))
            return (f"{v['share_ratio_in_band']:.1%} of {v['n_patches']} ratios in band; Spearman {v['spearman_bf16_vs_reference']:.3f} "
                    f"vs {v['reference']}; decision {c.get('decision')}. Descriptive: FP32-residual vs exact FP32 Spearman "
                    f"{d.get('spearman_fp32_residual_vs_exact_fp32', float('nan')):.3f} pooled, {within:.3f} median within trial")
        if name == "N11":
            bad = [k for k, x in v.items() if not (x["same_vocab_as_data_build"] and x["one_bos_and_lengths_match"] and not x["multi_token_answers"])]
            return "all families pass" if not bad else f"fails for {bad}"
        if name == "N12":
            per_patch = {k[len("patches_"):]: x["seconds"] / max(x["n"], 1) for k, x in v.items() if k.startswith("patches_")}

            def step(key: str) -> str:
                x = v.get(key) or {}
                if x.get("oom"):
                    return "OUT OF MEMORY"
                peak = (x.get("memory") or {}).get("max_reserved_gb")
                return f"{x.get('seconds', float('nan')):.2f}s" + (f" ({peak:.1f} GB)" if peak else "")

            text = (f"{v.get('tokens')} tok: forward {step('forward')}, backward {step('forward_backward_to_layer')}, "
                    f"checkpointed backward {step('forward_backward_checkpointed')}")
            if "longest_prompt_tokens" in v:
                text += (f"; {v['longest_prompt_tokens']} tok: backward {step('longest_prompt_forward_backward_to_layer')}, "
                         f"checkpointed {step('longest_prompt_forward_backward_checkpointed')}")
            return text + "; s per patch " + ", ".join(f"{k} {s:.3f}" for k, s in per_patch.items())
        if name in ("N8", "N9"):
            return f"{v:.2g} (allowed: {c.get('allowed')})"
    except (KeyError, TypeError, ValueError):
        return "see report.json"
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        return "see report.json"
    return str(v)


def _summary(checks: dict[str, dict[str, Any]], decisions: dict[str, Any], model_cfg: dict[str, Any]) -> str:
    lines = ["## What ran", "", f"Stage 0 inventory and numerics for `{model_cfg.get('name')}`.", "",
             "## Checks", "", "| Check | Value | Rule | Pass |", "|---|---|---|---|"]
    for name, c in sorted(checks.items(), key=lambda kv: (int("".join(ch for ch in kv[0].split("_")[0] if ch.isdigit())), kv[0])):
        policy = c.get("pass") and any(x != "ok" for x in (c.get("outcome") or {}).values())
        verdict = "yes, with the layer policy" if policy else ("yes" if c["pass"] else "NO")
        lines.append(f"| {name} | {_fmt(name, c)} | {c['rule'].replace('|', '&#124;')} | {verdict} |")
    lines += ["", "## Decisions (stage0_decisions.json)", "",
              f"- Causal measurements: **{decisions.get('causal_mode', decisions['precision_mode'])}**; gradient computations (N7): "
              f"**{decisions.get('gradient_mode', decisions['precision_mode'])}**; backward: {decisions.get('backward', 'plain')}",
              f"- In the causal mode: batching allowed {decisions['batching_allowed']}; suffix caching allowed {decisions['suffix_caching_allowed']}",
              "- Usable SAE per layer: " + ", ".join(f"{k}: {v['key']} (VE {v['variance_explained']:.3f}{', low-fidelity' if v.get('low_fidelity') else ''})"
                                                     for k, v in decisions["sae"].items()),
              "- Excluded layers (N3): " + (", ".join(f"{k} ({why})" for k, why in decisions.get("excluded_layers", {}).items()) or "none"),
              "- tau_tie per family: " + ", ".join(f"{k} {v:.3g}" for k, v in decisions["tau_tie"].items()), "",
              "## What this permits", "", "The Stage 0 gate (6.1) passes when every check passes; the decisions above then bind every later stage "
              "on this model. Values derived on 4B are never reused on 12B.", "", "## Next step", "",
              "Send back the files listed in the status line; then raw completions."]
    return "\n".join(lines) + "\n"
