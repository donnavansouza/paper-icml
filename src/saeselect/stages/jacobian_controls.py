"""``python -m saeselect jacobian-controls``: J-variant controls for the ranking advantage (extension X10). M1 only.

    python -m saeselect jacobian-controls --config configs/extensions/jacobian_variants.yaml --model configs/models/<model>.yaml --phase dev --out runs/jacobian-variants-dev-<m>-01 --device cuda
    python -m saeselect jacobian-controls --config configs/extensions/jacobian_variants.yaml --phase test --frozen runs/jacobian-variants-dev-<m>-01/frozen_protocol.json \
        --out runs/jacobian-variants-test-<m>-01 --device cuda

For every trial row of the parent Stage 2 run (its dev run in the dev phase, its test run in the test phase), the JL-OUT
score is recomputed on the same support prompts with J_l replaced by each variant; nothing else changes:

- JL-RAND: U diag(sigma) V^T, sigma the singular values of J_l, U and V Haar-random orthogonal (declared seed and draws);
- JL-PERM: P J_l, P a random permutation of the output coordinates (declared seed and draws);
- JL-OTHER: the J of another layer (declared map: the model's other concept layer).

Each variant's M1 is the Spearman correlation (unsquared) of its support score with the parent's stored single-feature
effects over the trial's evaluation pool, exactly as the parent computed JL-OUT's M1 (no new patching); with several draws,
the trial's M1 is the mean over draws. Parity: the recomputed JL-OUT and LL-OUT M1 must equal the stored ones within the
declared tolerance on every trial, or the run fails (EngineeringFailed:parity); with ``parity_swap_bound`` a trial
may also differ by one near-tie swap, 12 / (n (n + 1)) for a pool of n. Analysis: JL-OUT minus each variant on M1
with the parent's P2 procedure (pair means, cluster bootstrap within relation, three-way against the parent's M1 SESOI),
Holm across the three variants.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from saeselect import artifacts as A
from saeselect import lens as L
from saeselect import signals as S
from saeselect import stats
from saeselect.analysis import rebuild as R
from saeselect.stats import sub_seed as _seed
from saeselect.analysis.posthoc import reading
from saeselect.interventions import candidate_pool
from saeselect.models import load_model
from saeselect.patching import as_ids
from saeselect.provenance import EngineeringFailed, RunContractError, json_hash, seal_protocol, utc_now, verify_sealed_protocol
from saeselect.readout import output_direction
from saeselect.results import write_results_zip
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2

VARIANTS = ("JL-RAND", "JL-PERM", "JL-OTHER")
TWINS = ("LL-OUT", "JL-OUT")


# --------------------------------------------------------------------------
# J variants
# --------------------------------------------------------------------------


def haar(d: int, gen: torch.Generator) -> torch.Tensor:
    """A Haar-random orthogonal matrix (QR of a Gaussian with the sign of R's diagonal fixed), float64 on the CPU."""
    q, r = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def spectrum_matched(J: torch.Tensor, seed: int) -> torch.Tensor:
    """U diag(sigma_J) V^T with U, V Haar-random: J's singular values, random singular vectors."""
    sigma = torch.linalg.svdvals(J.double().cpu())
    gen = torch.Generator().manual_seed(seed)
    u, v = haar(J.shape[0], gen), haar(J.shape[1], gen)
    return ((u * sigma.unsqueeze(0)) @ v.T).float()


def permuted(J: torch.Tensor, seed: int) -> torch.Tensor:
    """P J: the output coordinates (rows; J is applied as J v) permuted at random."""
    perm = torch.randperm(J.shape[0], generator=torch.Generator().manual_seed(seed))
    return J.cpu()[perm].clone()


def variant_names(cfg: dict[str, Any]) -> list[str]:
    """The variants a config declares: the three of the J-variant check, or with ``design: controls`` the declared ones in declared order, with
    ``JL-INTERP`` expanded to one variant per t."""
    if cfg.get("design") != "controls":
        return list(VARIANTS)
    names: list[str] = []
    for key, v in cfg["variants"].items():
        names += [f"JL-T{t}" for t in v["t"]] if key == "JL-INTERP" else [key]
    return names


def variant_matrices(lens, layer: int, other_layer: int | None, spec: dict[str, Any], seed: int,
                     names: list[str] | tuple[str, ...] = VARIANTS) -> dict[str, list[torch.Tensor | None]]:
    """name -> one matrix per draw (None is the identity, the logit-lens twin), on the CPU. The control variants keep the
    identity path that J_l contains through the skip connections and perturb only R = J_l - I:
    JL-RES = I + (R's singular values, Haar-random singular vectors); JL-RESPERM = I + R with its output coordinates
    permuted; JL-T<t> = (1 - t) I + t J_l."""
    J = lens.J(layer, "cpu")
    eye = torch.eye(J.shape[0], dtype=J.dtype)
    out: dict[str, list[torch.Tensor | None]] = {"LL-OUT": [None], "JL-OUT": [J]}
    for name in names:
        if name == "JL-RAND":
            out[name] = [spectrum_matched(J, _seed(seed, "JL-RAND", layer, i)) for i in range(int(spec["JL-RAND"]["draws"]))]
        elif name == "JL-PERM":
            out[name] = [permuted(J, _seed(seed, "JL-PERM", layer, i)) for i in range(int(spec["JL-PERM"]["draws"]))]
        elif name == "JL-OTHER":
            out[name] = [lens.J(other_layer, "cpu")]
        elif name == "JL-RES":
            out[name] = [eye + spectrum_matched(J - eye, _seed(seed, "JL-RES", layer, i)) for i in range(int(spec["JL-RES"]["draws"]))]
        elif name == "JL-RESPERM":
            out[name] = [eye + permuted(J - eye, _seed(seed, "JL-RESPERM", layer, i)) for i in range(int(spec["JL-RESPERM"]["draws"]))]
        elif name.startswith("JL-T"):
            t = float(name[len("JL-T"):])
            out[name] = [(1 - t) * eye + t * J]
        else:
            raise RunContractError(f"unknown J variant {name!r}")
    return out


def j_diagnostics(J: torch.Tensor) -> dict[str, float]:
    """How much of J_l is the identity path: relative Frobenius norm of R = J - I, cosine of J with I, top singular values."""
    Jd = J.double().cpu()
    eye = torch.eye(Jd.shape[0], dtype=Jd.dtype)
    R = Jd - eye
    sj, sr = torch.linalg.svdvals(Jd), torch.linalg.svdvals(R)
    return {"rel_frobenius_R": float(R.norm() / Jd.norm()), "cos_J_I": float((Jd * eye).sum() / (Jd.norm() * eye.norm())),
            "mean_diag_J": float(Jd.diagonal().mean()), "top_sv_J": float(sj[0]), "top_sv_R": float(sr[0]),
            "median_sv_J": float(sj.median()), "median_sv_R": float(sr.median())}


# --------------------------------------------------------------------------
# Scores and M1
# --------------------------------------------------------------------------


def prompt_scores(sae, cache: S2.PromptCache, t: S2.Trial, layer: int, site: str, head, mats: dict[str, list]) -> dict[str, list[dict[int, float]]]:
    """Every variant's (per draw) score of the pool features on one prompt, with Stage 2's own arithmetic
    (``S2.trial_pass`` and ``signals.in_context_scores``)."""
    src, don = cache.get(t.source), cache.get(t.donor)
    dev = sae.device
    positions = src["sites"][site]
    h_src, h_don = src["acts"][(layer, site)].to(dev), don["acts"][(layer, site)].to(dev)
    da_full = sae.encode(h_don) - sae.encode(h_src)
    pool = candidate_pool(da_full)
    da, W = da_full.index_select(1, pool), sae.W_dec.index_select(0, pool)
    h_f = src["h_final"].to(dev)
    w_out = output_direction(h_f, head, t.answers)
    ids = pool.tolist()
    out: dict[str, list[dict[int, float]]] = {}
    for name, draws in mats.items():
        if name == "NORM":  # Stage 2's NORM on the same prompt, the control of JL-OUT's partial correlation
            norm = S.in_context_scores(da=da, W_dec_pool=W, sd_pool=torch.ones(len(ids), device=dev), lens_s={},
                                       grad_site=torch.zeros(len(positions), W.shape[1], device=dev))["NORM"]
            out[name] = [dict(zip(ids, norm.tolist()))]
            continue
        out[name] = []
        for J in draws:
            s = L.lens_sensitivity(w_out, h_f, head, J)
            sc = S.in_context_scores(da=da, W_dec_pool=W, sd_pool=torch.ones(len(ids), device=dev), lens_s={name: s},
                                     grad_site=torch.zeros(len(positions), W.shape[1], device=dev))[name]
            out[name].append(dict(zip(ids, sc.tolist())))
    return out


def trial_row(r: dict[str, Any], support: list[dict[str, list[dict[int, float]]]]) -> dict[str, Any]:
    """M1 of every variant for one parent row, the draws' mean as the variant's M1, and the twins' parity."""
    effects = {int(f): v for f, v in r["effects"].items()}
    m1, draws = {}, {}
    for name in support[0]:
        per = [S.ranking_metrics(S.support_scores(s[name][i] for s in support), effects, ks=[])["M1"] for i in range(len(support[0][name]))]
        draws[name] = per
        ok = [x for x in per if x is not None]
        m1[name] = {"M1": float(np.mean(ok)) if ok else None}
    parity = {}
    for name in [*TWINS, *(["NORM"] if "NORM" in support[0] else [])]:
        stored, now = r["metrics"][name]["M1"], m1[name]["M1"]
        parity[name] = 0.0 if stored is None and now is None else (float("inf") if stored is None or now is None else abs(stored - now))
    if "NORM" in support[0]:  # does JL-OUT carry information about the effects beyond NORM (partial Spearman, M6)?
        m1["JL-OUT"]["M6_given_NORM"] = S.ranking_metrics(S.support_scores(s["JL-OUT"][0] for s in support), effects, ks=[],
                                                          control=S.support_scores(s["NORM"][0] for s in support))["M6"]
    return {"trial": r["trial"], "relation": r["relation"], "pair": r["pair"], "layer": r["layer"], "site": r["site"],
            "metrics": m1, "draws": draws, "parity": parity, "pool_size": len(effects)}


def parity_tolerance(tol: float, pool_size: int, swap_bound: bool) -> float:
    """A trial's M1 parity tolerance. On a GPU other than the parent's, two near-tied scores can swap order, which
    moves a Spearman over a pool of n by at most 12 / (n (n + 1)); the bound admits one such swap and nothing larger."""
    return max(tol, 12.0 / (pool_size * (pool_size + 1))) if swap_bound and pool_size > 1 else tol


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def analyze(rows: list[dict[str, Any]], cfg: dict[str, Any], *, seed: int, holm_alpha: float,
            contrasts: list[tuple[str, str]] | None = None, names: list[str] | tuple[str, ...] = VARIANTS) -> dict[str, Any]:
    """The declared contrasts on M1 (the parent's P2 procedure; the J-variant check: JL-OUT minus each variant), Holm across them; per
    cell; mean M1 per signal; with NORM scored, JL-OUT's partial Spearman given NORM (pair means, three-way above 0)."""
    contrasts = contrasts or [("JL-OUT", v) for v in VARIANTS]
    label = (lambda a, b: b) if contrasts == [("JL-OUT", v) for v in VARIANTS] else (lambda a, b: f"{a} - {b}")
    family = {label(a, b): S2.contrast(rows, a, b, "M1", k=None, cfg=cfg, seed=_seed(seed, "family", b) if a == "JL-OUT" else _seed(seed, "family", a, b))
              for a, b in contrasts}
    testable = {v: c["_verdict"] for v, c in family.items() if c is not None}
    if testable:
        stats.apply_holm(testable, alpha=holm_alpha)
        for v, verdict in testable.items():
            family[v].update(p_holm=verdict.p_adjusted, better_after_holm=verdict.better_after_holm)
    per_cell = {}
    for layer, site in sorted({(r["layer"], r["site"]) for r in rows}):
        sub = [r for r in rows if r["layer"] == layer and r["site"] == site]
        per_cell[f"{layer}/{site}"] = {label(a, b): S2.contrast(sub, a, b, "M1", k=None, cfg=cfg,
                                                                seed=_seed(seed, "cell", layer, site, b) if a == "JL-OUT" else _seed(seed, "cell", layer, site, a, b))
                                       for a, b in contrasts}
    means = {}
    for name in (*TWINS, *names, *(["NORM"] if rows and "NORM" in rows[0]["metrics"] else [])):
        vals, _, _ = S2._cluster_means(rows, lambda r, n=name: r["metrics"][n]["M1"])
        means[name] = float(np.mean(vals)) if vals else None
    out: dict[str, Any] = {"family": family, "per_cell": per_cell, "mean_M1": means}
    if rows and "M6_given_NORM" in rows[0]["metrics"].get("JL-OUT", {}):
        vals, _, strata = S2._cluster_means(rows, lambda r: r["metrics"]["JL-OUT"].get("M6_given_NORM"))
        b = cfg["analysis"]["bootstrap"]
        res = stats.cluster_bootstrap(vals, n_resamples=b["n_resamples"], seed=_seed(seed, "M6 NORM"), strata=strata)
        v = stats.verdict(res, sesoi=cfg["analysis"]["sesoi"]["M1"], predicted_sign=1, level=b["level"], tost_level=b["tost_level"])
        out["JL-OUT partial given NORM"] = {"estimate": v.estimate, "ci95": list(v.ci95), "ci90": list(v.ci90), "p": v.p_value,
                                            "better": v.better, "practically_better": v.practically_better, "equivalent": v.equivalent,
                                            "worse": v.worse, "n_pairs": len(vals)}
    return _strip(out)


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "_verdict"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


def _parent(ctx: RunContext) -> tuple[str, dict[str, Any], dict[str, Any], Path]:
    """The model, its parent entry, the parent's sealed frozen protocol and the parent run of this phase.

    The model is the parent's frozen model config, the one that produced the stored effects: model config files gained
    fields after some parents froze (the SAE swap's ``sae.swap``), so ``--model`` only names the checkpoint, and its weights
    (repository and revision) must be the parent's."""
    named = ctx.model_config if ctx.phase == "dev" else (ctx.frozen or {}).get("model")
    if named is None:
        raise RunContractError("jvariants needs --model configs/models/<model>.yaml on dev (the test takes it from the frozen protocol)")
    name = named["name"]
    entry = (ctx.config.get("parents") or {}).get(name)
    if entry is None:
        raise RunContractError(f"configs: no parent for {name}")
    frozen = ctx.load_upstream(entry["frozen"])
    verify_sealed_protocol(frozen)
    for key in ("name", "repo_id", "revision"):
        if named.get(key) != frozen["model"].get(key):
            raise RunContractError(f"{name}: --model {key} {named.get(key)!r} is not the parent's {frozen['model'].get(key)!r} ({entry['frozen']})")
    return name, entry, frozen, Path(entry["dev_run"] if ctx.phase == "dev" else entry["test_run"])


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    if ctx.phase not in ("dev", "test"):
        raise RunContractError("jvariants needs --phase dev or --phase test")
    if ctx.phase == "test" and json_hash(cfg) != json_hash(ctx.frozen["config"]):
        raise RunContractError("--config differs from the frozen protocol; a change needs a new protocol")
    name, entry, parent_frozen, parent_dir = _parent(ctx)
    data = R.read_run(parent_dir)
    prov, flags = R.provenance(data, parent_frozen, Path(entry["frozen"]))
    if data["protocol"].get("phase") != ctx.phase:
        raise RunContractError(f"{parent_dir} is a {data['protocol'].get('phase')} run; the {ctx.phase} phase reads the parent's {ctx.phase} run")
    ctx.flags.extend(f"parent {parent_dir.as_posix()}: {f}" for f in flags if "a dev run" not in f)
    pcfg = parent_frozen.get("config_resolved") or S2._resolve(parent_frozen["config"])
    model_cfg = parent_frozen["model"]
    s0 = ctx.load_upstream(Path(parent_frozen["stage0_run"]) / "stage0_decisions.json")
    causal = s0.get("causal_mode", s0["precision_mode"])
    rows_in = data["rows"]
    if ctx.limit is not None:
        keep = set(sorted({r["pair"] for r in rows_in})[: ctx.limit])
        rows_in = [r for r in rows_in if r["pair"] in keep]
    cells = sorted({(r["layer"], r["site"]) for r in rows_in})
    names = variant_names(cfg)
    other = {int(k): int(v) for k, v in (entry.get("other_layer") or {}).items()}
    missing = sorted({layer for layer, _ in cells} - set(other))
    if missing and "JL-OTHER" in names:
        raise RunContractError(f"{name}: no JL-OTHER layer declared for {missing}")

    # trials of the parent's relations in this phase, keyed as the parent's rows
    trials: dict[str, S2.Trial] = {}
    data_sha = {}
    for family in parent_frozen["relations"]:
        card, records = S2._load(Path(pcfg["data_dir"]), family, ctx.phase)
        data_sha[family] = card["records_sha256"]
        for t in S2.build_trials(records, family):
            trials[t.key] = t
    unknown = sorted({r["trial"] for r in rows_in} - set(trials))
    if unknown:
        raise RunContractError(f"{len(unknown)} parent trials are not in this phase's records, e.g. {unknown[:3]}")
    by_group: dict[tuple[str, str], dict[str, S2.Trial]] = defaultdict(dict)
    for t in trials.values():
        by_group[(t.pair, t.direction)][t.template] = t

    handle = load_model(model_cfg, device=ctx.device)
    lens = A.load_lens_for(model_cfg, d_model=handle.d_model, n_layers=handle.n_layers)
    cache = S2.PromptCache(handle, sorted({layer for layer, _ in cells}), causal)
    seed = int(cfg["seed"])
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    timing: dict[str, Any] | None = None
    for layer in sorted({layer for layer, _ in cells}):
        u = s0["sae"][str(layer)]
        sae = A.load_sae_for(model_cfg, layer=layer, width=u["width"], l0=u["l0"], d_model=handle.d_model, suffix=u.get("suffix", ""),
                             device=ctx.device, convention=u.get("convention"), repo_id=u.get("repo_id"), revision=u.get("revision"))
        if cfg.get("j_diagnostics"):
            diagnostics[str(layer)] = j_diagnostics(lens.J(layer, "cpu"))
        mats = {n: [None if J is None else J.to(ctx.device) for J in draws]  # on the device once per layer
                for n, draws in variant_matrices(lens, layer, other.get(layer), cfg["variants"], seed, names).items()}
        if cfg.get("partial_given_NORM"):
            mats["NORM"] = [None]
        if cfg.get("timing") and timing is None:
            site0 = next(s for (lay, s) in cells if lay == layer)
            sample = [trials[r["trial"]] for r in rows_in if r["layer"] == layer and r["site"] == site0][: int(cfg["timing"]["n_prompts"])]
            timing = benchmark(handle, causal, s0, sae, lens, layer, site0, sample, ctx.device)
        for site in [s for (lay, s) in cells if lay == layer]:
            scored: dict[str, dict[str, list[dict[int, float]]]] = {}
            for r in [x for x in rows_in if x["layer"] == layer and x["site"] == site]:
                t = trials[r["trial"]]
                group = by_group[(t.pair, t.direction)]
                support = []
                for tpl in r["support_templates"]:
                    s_t = group[tpl]
                    if s_t.key not in scored:
                        scored[s_t.key] = prompt_scores(sae, cache, s_t, layer, site, handle.head, mats)
                    support.append(scored[s_t.key])
                rows.append(trial_row(r, support))
        del sae, mats
        S2._free()

    tol = float(cfg["parity_max_abs_m1"])
    swap_bound = bool(cfg.get("parity_swap_bound"))  # near-tie swaps widen the parity tolerance
    tols = [parity_tolerance(tol, r["pool_size"], swap_bound) for r in rows]
    checked = sorted({n for r in rows for n in r["parity"]})
    worst = {n: max((r["parity"][n] for r in rows), default=0.0) for n in checked}
    over_base = sum(any(v > tol for v in r["parity"].values()) for r in rows)
    write_results_zip(ctx.run.path, {"trials.jsonl": "".join(json.dumps(r, sort_keys=True, default=S2._jsonable) + "\n" for r in rows)})
    design = cfg.get("design", "variants")
    report: dict[str, Any] = {"phase": ctx.phase, "design": design, "model": name, "parent_run": parent_dir.as_posix(), "parent_provenance": prov,
                              "parent_frozen_protocol_hash": parent_frozen["protocol_hash"], "cells": [f"{a}/{b}" for a, b in cells],
                              "variants": names, "other_layer": {str(k): v for k, v in other.items()}, "n_trials": len(rows),
                              "n_pairs": len({r["pair"] for r in rows}), "parity_max_abs_m1": worst, "parity_tolerance": tol,
                              "lens_stored_dtype": lens.stored_dtype}
    if swap_bound:
        report["parity_swap_bound"] = {"trials_over_base_tolerance": over_base, "n_trials": len(rows)}
    if diagnostics:
        report["J_diagnostics"] = diagnostics
    if timing:
        report["timing"] = timing
    failing = [(r, t) for r, t in zip(rows, tols) if any(v > t for v in r["parity"].values())]
    if failing:
        bad = [f"{r['trial']} L{r['layer']}/{r['site']} (tolerance {t:.2g}): {r['parity']}" for r, t in failing][:20]
        bound = " or one near-tie swap, 12/(n(n+1)) for a pool of n" if swap_bound else ""
        raise EngineeringFailed("parity", f"recomputed M1 differs from the parent's stored M1 by up to {worst} (tolerance {tol}{bound})",
                                bad, report=report)
    contrasts = [tuple(c) for c in cfg["contrasts"]] if cfg.get("contrasts") else None
    report["analysis"] = analyze(rows, pcfg, seed=seed, holm_alpha=float(cfg["holm_alpha"]), contrasts=contrasts, names=names)
    report = json.loads(json.dumps(report, default=S2._jsonable))
    summary = _summary(ctx, report, pcfg)
    send_back = ["results.zip"]
    if ctx.phase == "dev":
        protocol = {"stage": "jvariants", "design": design, "smoke": ctx.smoke, "code_commit": ctx.git.get("commit"),
                    "source_run": ctx.run.path.as_posix(), "created_at": utc_now(), "config": cfg, "model": model_cfg,
                    "parent_protocol_hash": parent_frozen["protocol_hash"], "cells": report["cells"], "data": data_sha}
        frozen_protocol = seal_protocol(protocol)
        ctx.run.write_json("frozen_protocol.json", frozen_protocol)
        report["frozen_protocol_hash"] = frozen_protocol["protocol_hash"]
        return Outcome(status="FROZEN_AWAITING_APPROVAL", summary=summary, report=report, send_back=["frozen_protocol.json", *send_back])
    if ctx.frozen.get("parent_protocol_hash") != parent_frozen["protocol_hash"] or ctx.frozen.get("cells") != report["cells"]:
        raise RunContractError(f"the parent protocol or the cells differ from the frozen protocol")
    return Outcome(status="PASS", summary=summary, report=report, send_back=send_back)


def benchmark(handle, causal: str, s0: dict[str, Any], sae, lens, layer: int, site: str, sample: list[S2.Trial], device: str) -> dict[str, Any]:
    """Cost: wall-clock per support prompt, median over the sample, of the clean forward that every
    signal needs, the J-lens score of the pool given that forward, and the backward pass attribution patching adds."""
    from saeselect.patching import margin_gradient

    sync = torch.cuda.synchronize if device.startswith("cuda") and torch.cuda.is_available() else (lambda: None)
    grad_mode = s0.get("gradient_mode", s0["precision_mode"])
    checkpointed = s0.get("backward") == "checkpointed"
    J = lens.J(layer, device)
    fwd, jl, bwd = [], [], []
    for t in sample:
        cache = S2.PromptCache(handle, [layer], causal)
        sync()
        t0 = time.perf_counter()
        src = cache.get(t.source)
        cache.get(t.donor)
        sync()
        t1 = time.perf_counter()
        prompt_scores(sae, cache, t, layer, site, handle.head, {"JL-OUT": [J]})
        sync()
        t2 = time.perf_counter()
        margin_gradient(handle, as_ids(src["ids"], handle.device), layer, t.answers, readout=t.source.meta["readout_position"],
                        mode=grad_mode, checkpointed=checkpointed)
        sync()
        t3 = time.perf_counter()
        fwd.append(t1 - t0)
        jl.append(t2 - t1)
        bwd.append(t3 - t2)
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    return {"n_prompts": len(sample), "layer": layer, "site": site, "device": device,
            "median_seconds": {"clean forwards (source and donor)": med(fwd), "J-lens score of the pool": med(jl),
                               "attribution backward pass": med(bwd)}}


def _ci(c: dict[str, Any] | None, n_resamples: int = 10_000) -> str:
    """Estimate, 95% interval and the verdict with its flags in words (the post hoc analysis's reading)."""
    if not c:
        return "not testable"
    p = c.get("p_holm")
    holm = "" if p is None else (f", Holm p {p:.3g}" if p > 0 else f", Holm p < {2 / n_resamples:.1g}")
    return f"{c['estimate']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {reading(c)}{holm}"


def _summary_a35(ctx: RunContext, report: dict[str, Any], pcfg: dict[str, Any]) -> str:
    a = report["analysis"]
    n = pcfg["analysis"]["bootstrap"]["n_resamples"]
    lines = ["## What ran", "",
             f"The J-variant controls, {ctx.phase} phase, on `{report['model']}`: for every trial row of the parent Stage 2 "
             f"{ctx.phase} run `{report['parent_run']}`, the JL-OUT score recomputed on the same support prompts with J_l replaced by each "
             f"declared variant ({', '.join(report['variants'])}). JL-RES = I + (R's singular values, Haar-random singular vectors), JL-RESPERM "
             "= I + R with its output coordinates permuted, R = J_l - I; JL-T<t> = (1 - t) I + t J_l. M1 = Spearman (unsquared) of the support "
             "score with the parent's stored single-feature effects; draws averaged per trial. No new patching.", "",
             f"Trials: {report['n_trials']} in {report['n_pairs']} pairs; cells {', '.join(report['cells'])}.", "",
             "## Parity", "", "Recomputed against stored M1, worst absolute difference: "
             + ", ".join(f"{k} {v:.2g}" for k, v in report["parity_max_abs_m1"].items()) + f" (tolerance {report['parity_tolerance']:.2g}"
             + (f", or one near-tie swap 12/(n(n+1)) per trial, the swap bound; {report['parity_swap_bound']['trials_over_base_tolerance']} of "
                f"{report['parity_swap_bound']['n_trials']} trials above {report['parity_tolerance']:.2g}" if report.get("parity_swap_bound") else "")
             + "): passed.", "",
             "## Declared contrasts on M1 (Holm across them)", "", "| Contrast | Estimate [95%], verdict |", "|---|---|"]
    lines += [f"| {k} | {_ci(c, n)} |" for k, c in a["family"].items()]
    lines += ["", "Mean M1 (pair means): " + ", ".join(f"{k} {v:.3f}" if v is not None else f"{k} n/a" for k, v in a["mean_M1"].items()), ""]
    if a.get("JL-OUT partial given NORM"):
        lines += [f"JL-OUT's partial Spearman given NORM: {_ci(a['JL-OUT partial given NORM'], n)}", ""]
    keys = list(a["family"])
    lines += ["Per cell (no correction):", "", "| Cell | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {cell} | " + " | ".join(_ci(per[k]) for k in keys) + " |" for cell, per in a["per_cell"].items()]
    if report.get("J_diagnostics"):
        lines += ["", "## How much of J_l is the identity path", "", "| Layer | ||J - I|| / ||J|| | cos(J, I) | mean diag J | top sv J, R |", "|---|---|---|---|---|"]
        lines += [f"| {lay} | {d['rel_frobenius_R']:.3f} | {d['cos_J_I']:.3f} | {d['mean_diag_J']:.3f} | {d['top_sv_J']:.2f}, {d['top_sv_R']:.2f} |"
                  for lay, d in report["J_diagnostics"].items()]
    if report.get("timing"):
        t = report["timing"]
        lines += ["", f"## Cost per support prompt (median of {t['n_prompts']}, layer {t['layer']}, {t['device']})", ""]
        lines += [f"- {k}: {v:.4f} s" for k, v in t["median_seconds"].items() if v is not None]
    lines += ["", "## What this permits", ""]
    lines += (["Dev estimates, not claims. The frozen protocol holds the config, the parent protocol and the cells; the test run needs the "
               "explicit approval and runs once on the parent's test run."] if ctx.phase == "dev" else
              ["Pre-specified secondary analyses on the held-out pairs; three-way verdicts; inconclusive is never "
               "'no difference'."])
    lines += ["", "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"


def _summary(ctx: RunContext, report: dict[str, Any], pcfg: dict[str, Any]) -> str:
    if report.get("design") == "controls":
        return _summary_a35(ctx, report, pcfg)
    a = report["analysis"]
    lines = ["## What ran", "",
             f"The J-variant check, {ctx.phase} phase, on `{report['model']}`: for every trial row of the parent Stage 2 "
             f"{ctx.phase} run `{report['parent_run']}` (parent protocol {report['parent_frozen_protocol_hash'][:12]}...), the JL-OUT score "
             "recomputed on the same support prompts with J replaced by JL-RAND (J's singular values, Haar-random singular vectors), JL-PERM "
             "(J's output coordinates permuted) and JL-OTHER (the J of layer " + ", ".join(f"{k} -> {v}" for k, v in report["other_layer"].items())
             + "). M1 = Spearman (unsquared) of the support score with the parent's stored single-feature effects over the evaluation pool; "
             "several draws are averaged per trial. No new patching.", "",
             f"Trials: {report['n_trials']} in {report['n_pairs']} pairs; cells {', '.join(report['cells'])}.", "",
             "## Parity", "",
             f"Recomputed against stored M1, worst absolute difference: LL-OUT {report['parity_max_abs_m1']['LL-OUT']:.2g}, JL-OUT "
             f"{report['parity_max_abs_m1']['JL-OUT']:.2g} (tolerance {report['parity_tolerance']:.2g}): passed.", "",
             "## JL-OUT minus each variant on M1 (Holm across the three)", "",
             f"Pair means, cluster bootstrap within relation, three-way against the parent's M1 SESOI {pcfg['analysis']['sesoi']['M1']}.", "",
             "| Contrast | Estimate [95%], verdict |", "|---|---|"]
    lines += [f"| JL-OUT - {v} | {_ci(c, pcfg['analysis']['bootstrap']['n_resamples'])} |" for v, c in a["family"].items()]
    lines += ["", "Mean M1 (pair means): " + ", ".join(f"{n} {v:.3f}" if v is not None else f"{n} n/a" for n, v in a["mean_M1"].items()), "",
              "Per cell (no correction):", "", "| Cell | " + " | ".join(f"JL-OUT - {v}" for v in VARIANTS) + " |", "|---|---|---|---|"]
    for cell, per in a["per_cell"].items():
        lines.append(f"| {cell} | " + " | ".join(_ci(per[v]) for v in VARIANTS) + " |")
    lines += ["", "## What this permits", ""]
    if ctx.phase == "dev":
        lines += ["Dev estimates, not claims. The frozen protocol holds the config, the parent protocol and the cells; the test run needs the "
                  "explicit approval and runs once on the parent's test run."]
    else:
        lines += ["JL-OUT better than JL-RAND and JL-PERM means the advantage needs J's directions, not only its "
                  "spectrum; JL-OUT equivalent to JL-OTHER means any of this model's Jacobians serves; better than JL-OTHER means the layer "
                  "matters. Three-way verdicts; inconclusive is never 'no difference'."]
    lines += ["", "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
