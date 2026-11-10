"""``python -m saeselect posthoc``: post hoc re-analyses of the paper's test runs.

    python -m saeselect posthoc --config configs/analysis/posthoc.yaml --out analysis/posthoc

CPU only. Reads Stage 2 test runs' ``results.zip`` (each checked against its sealed frozen protocol, as compare-sets does)
and the Stage 0 decisions those runs used. Each run is analysed with its own frozen config. Pair means, cluster bootstrap
within relation with the frozen resamples, 95% intervals; trials below the frozen M2 full-gain floor are left out
wherever F or a set margin enters. Unsquared throughout. Every result is post hoc or descriptive.

1. SESOI sensitivity: the primary runs' families and the declared cross-run families (the cross-model comparison on X3) recomputed with
   their own procedure and seeds at each SESOI of the grid. At the declared SESOI every value must equal the reports
   (GATE_FAILED:reproduction otherwise).
2. Absolute endpoints per selector: donor-switch share (m(S) > 0; ties |m(S)| < tau left out and counted separately,
   rule 5 and ``readout.is_tie``), tie share, and Delta m = m(S) - m_clean.
3. Recovery against perturbation size: rho(S) = realized |delta h(S)| / realized |delta h_FULL|; F at matched rho along
   each selector's k path.
4. The SAE swap's four contrasts per cell (descriptive, no Holm).
5. JL-OUT - LL-OUT per (run, cell, SAE) against the SAE's variance explained on its host.
6. P2.1 and P2.2 with each relation left out in turn.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from saeselect import stats
from saeselect.analysis import compare_sets as C
from saeselect.analysis import rebuild as R
from saeselect.stats import sub_seed as _seed
from saeselect.provenance import GateFailed, RunContractError, verify_sealed_protocol
from saeselect.readout import is_tie
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2

BRIEF = ("estimate", "ci95", "ci90", "p", "label", "better", "practically_better", "equivalent", "worse", "p_holm", "better_after_holm")


def _flag(flag: str) -> str:
    return flag.replace("(reproduction is still checked)", "(reanalyze checks the analyses it recomputes against the reports)")


def load(entry: dict[str, Any], cache: dict[tuple[str, str], dict[str, Any]], ctx: RunContext | None = None, *,
         keep_effects: bool = False) -> dict[str, Any]:
    """Rows, frozen config, report, provenance and Stage 0 decisions of one Stage 2 run; each run is read once.
    ``keep_effects`` keeps the per-feature single effects (the additivity analysis's superadditivity needs them)."""
    key = (entry["run"], entry["frozen"])
    if key not in cache:
        run_dir, frozen_path = Path(entry["run"]), Path(entry["frozen"])
        data = R.read_run(run_dir)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        verify_sealed_protocol(frozen)
        prov, flags = R.provenance(data, frozen, frozen_path)
        if data["protocol"].get("phase") != "test":
            raise RunContractError(f"{run_dir} is a {data['protocol'].get('phase')} run; the post hoc analysis reads test runs only")
        s0_dir = data["report"].get("stage0_run")
        s0_path = Path(s0_dir) / "stage0_decisions.json" if s0_dir else None
        if s0_path is None or not s0_path.exists():
            raise RunContractError(f"{run_dir}: the Stage 0 decisions it used ({s0_path}) are not here")
        for r in data["rows"]:  # per-feature effects and lens scores are the bulk of a row and no analysis here reads them
            if not keep_effects:
                r.pop("effects", None)
            r.pop("in_context_lens", None)
        cache[key] = {"rows": data["rows"], "cfg": frozen.get("config_resolved") or S2._resolve(frozen["config"]), "report": data["report"],
                      "provenance": {**prov, "stage0_decisions": str(s0_path).replace("\\", "/")},
                      "s0": json.loads(s0_path.read_text(encoding="utf-8"))}
        if ctx is not None:
            ctx.flags.extend(f"{entry['run']}: {_flag(f)}" for f in flags)
    return cache[key]


def reading(c: dict[str, Any] | None) -> str:
    """The three-way verdict with its flags in words."""
    if not c:
        return "not testable"
    if c["better"]:
        if c.get("practically_better"):  # P2.3's slope result carries no such flag
            return "practically better"
        return "above 0, within the margin" if c.get("equivalent") else "better"
    if c["worse"]:
        return "below 0, within the margin" if c.get("equivalent") else "worse"
    return "equivalent" if c.get("equivalent") else "inconclusive"


def _brief(c: dict[str, Any] | None) -> dict[str, Any] | None:
    if not c:
        return None
    out = {k: c[k] for k in BRIEF if k in c}
    out["reading"] = reading(c)
    return out


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=S2._jsonable))


def _with_sesoi(cfg: dict[str, Any], sesoi: float) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["analysis"]["sesoi"] = {**out["analysis"]["sesoi"], "M1": sesoi, "M2": sesoi}
    return out


# --------------------------------------------------------------------------
# 1. SESOI sensitivity
# --------------------------------------------------------------------------


def run_family(run: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """The run's primary and secondary contrasts with its own procedure and seed (``S2.analyze``)."""
    rep = run["report"]
    divergence = {int(k): v for k, v in rep["divergence"].items()}
    return _jsonable(S2.analyze(run["rows"], cfg, pk=int(rep["primary_k"]), divergence=divergence, seed=cfg["seed"]))


def _contrasts(analysis: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    out = {k: v for k, v in analysis["primary"].items()}
    out.update(analysis["secondary"])
    for layer, per in analysis["H3_per_layer_site"].items():
        for site, v in per.items():
            if v is not None:
                out[f"JL-OUT - LL-OUT (M1), L{layer}/{site}"] = v
    return out


def cross_family(xcfg: dict[str, Any], cache: dict, ctx: RunContext | None, *, sesoi: float | None) -> dict[str, Any]:
    """A compare-sets family (``configs/analysis/compare_sets_gemma_base.yaml``, ``configs/analysis/sae_swap.yaml``) with its own seeds, at ``sesoi``."""
    loaded = {}
    for name, entry in xcfg["models"].items():
        run = load(entry, cache, ctx)
        loaded[name] = (C.matched(run["rows"], entry["matched_cells"], xcfg["matched_relations"]), run["cfg"])
    contrasts = xcfg.get("contrasts") or [{"name": f"{n} - {xcfg['reference']}", "model": n, "reference": xcfg["reference"]}
                                          for n in loaded if n != xcfg["reference"]]
    out: dict[str, Any] = {}
    for c in contrasts:
        (sub, cfg_m), (ref, cfg_r) = loaded[c["model"]], loaded[c["reference"]]
        seed_name = c["model"] if not xcfg.get("contrasts") else c["name"]
        out[c["name"]] = {str(k): C.cross(sub, cfg_m, ref, cfg_r, k=k, seed=_seed(xcfg["seed"], "cross", seed_name), sesoi=sesoi)
                          for k in xcfg["cross_model_k"]}
    if xcfg.get("holm_k") is not None:
        family = {name: by_k[str(xcfg["holm_k"])]["_verdict"] for name, by_k in out.items() if by_k.get(str(xcfg["holm_k"]))}
        stats.apply_holm(family, alpha=xcfg["holm_alpha"])
        for name, v in family.items():
            out[name][str(xcfg["holm_k"])].update(p_holm=v.p_adjusted, better_after_holm=v.better_after_holm)
    for by_k in out.values():
        for c in by_k.values():
            if c:
                c.pop("_verdict", None)
    return _jsonable(out)


REBUILD_ONLY = ": in the rebuild only"


def split_mismatches(bad: list[str]) -> tuple[list[str], list[str]]:
    """Reproduction failures, and fields the report never had (an older report than the code that adds them). Every value a
    report holds must be reproduced exactly; a field it lacks contradicts nothing it reported, so it is listed, not failed."""
    return [b for b in bad if not b.endswith(REBUILD_ONLY)], [b[: -len(REBUILD_ONLY)] for b in bad if b.endswith(REBUILD_ONLY)]


def sesoi_sensitivity(cfg: dict[str, Any], cache: dict, ctx: RunContext | None) -> tuple[dict[str, Any], list[str]]:
    """Readings at each SESOI of the grid; mismatches with the reports at the declared SESOI."""
    grid = [float(s) for s in cfg["sesoi"]["grid"]]
    out: dict[str, Any] = {"grid": grid, "runs": {}, "cross": {}, "fields_not_in_reports": []}
    mismatches: list[str] = []
    for name, entry in cfg["models"].items():
        run = load(entry, cache, ctx)
        declared = run["cfg"]["analysis"]["sesoi"]
        if declared["M1"] not in grid or declared["M2"] not in grid:
            raise RunContractError(f"{name}: the declared SESOI {declared} must be in the grid {grid}")
        _, bad = R.compare(run["report"]["analysis"], run_family(run, run["cfg"]))
        bad, extra = split_mismatches(bad)
        mismatches += [f"{name}: {b}" for b in bad]
        out["fields_not_in_reports"] += [f"{name}: {b}" for b in extra]
        per: dict[str, dict[str, Any]] = {}
        for s in grid:
            for cname, c in _contrasts(run_family(run, _with_sesoi(run["cfg"], s))).items():
                per.setdefault(cname, {})[str(s)] = _brief(c)
        out["runs"][name] = per
    for x in cfg["sesoi"].get("cross") or []:
        xcfg = yaml.safe_load(Path(x["config"]).read_text(encoding="utf-8"))
        tag = Path(x["config"]).stem
        _, bad = R.compare(json.loads(Path(x["report"]).read_text(encoding="utf-8"))["cross_model"], cross_family(xcfg, cache, ctx, sesoi=None))
        bad, extra = split_mismatches(bad)
        mismatches += [f"{tag}: {b}" for b in bad]
        out["fields_not_in_reports"] += [f"{tag}: {b}" for b in extra]
        per = {}
        for s in grid:
            for cname, by_k in cross_family(xcfg, cache, ctx, sesoi=s).items():
                for k, c in by_k.items():
                    per.setdefault(f"{cname}, k = {k}", {})[str(s)] = _brief(c)
        out["cross"][tag] = per
    return out, mismatches


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def mean_ci(rows: list[dict[str, Any]], value, cfg: dict[str, Any], seed: int) -> dict[str, Any] | None:
    """Pair mean of a per-trial value, cluster bootstrap within relation, 95% interval (no verdict)."""
    vals, _, strata = S2._cluster_means(rows, value)
    if len(vals) < 2:
        return {"estimate": float(np.mean(vals)), "ci95": None, "n_pairs": len(vals)} if vals else None
    b = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(vals, n_resamples=b["n_resamples"], seed=seed, strata=strata)
    return {"estimate": res.estimate, "ci95": list(res.ci(b["level"])), "n_pairs": len(vals)}


def _above_floor(r: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return r["full_gain"] >= cfg["m2_min_full_gain"]


# --------------------------------------------------------------------------
# 2. Absolute endpoints
# --------------------------------------------------------------------------


def switch(m: float, tau: float) -> float | None:
    """1 if the donor answer wins, 0 if the source answer does, None for a tie (|m| < tau_tie; rule 5, 5.2)."""
    return None if is_tie(m, tau) else float(m > 0)


def margins(r: dict[str, Any], name: str, k: int) -> list[float]:
    """The set margin(s) of one selector at k: one value, the random draws' values, or a reference margin."""
    if name in ("clean", "FULL"):
        return [r["m"][name]]
    if name == "RANDOM":
        return [d["margin"] for d in r["random"][str(k)]]
    return [r["sets"][name][str(k)]["margin"]]


def _switch_share(ms: list[float], tau: float) -> float | None:
    s = [x for x in (switch(m, tau) for m in ms) if x is not None]
    return float(np.mean(s)) if s else None


def absolute_endpoints(run: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    rows, cfg = run["rows"], run["cfg"]
    tau_of = run["s0"]["tau_tie"]
    missing = sorted({r["relation"] for r in rows} - set(tau_of))
    if missing:
        raise RunContractError(f"no tau_tie for {missing} in {run['provenance']['stage0_decisions']}")
    pk = int(run["report"]["primary_k"])
    ks = sorted({*map(int, spec["k"]), pk})
    names = ["clean", "FULL", *spec["signals"], "RANDOM"]
    out: dict[str, Any] = {"primary_k": pk, "tau_tie": {rel: tau_of[rel] for rel in sorted({r["relation"] for r in rows})}, "by_k": {}}
    for k in ks:
        entry: dict[str, Any] = {"selectors": {}, "contrasts": {}}
        for name in names:
            def sw(r, name=name, k=k):
                return _switch_share(margins(r, name, k), tau_of[r["relation"]]) if _above_floor(r, cfg) else None

            def tie(r, name=name, k=k):
                ms = margins(r, name, k)
                return float(np.mean([is_tie(m, tau_of[r["relation"]]) for m in ms])) if _above_floor(r, cfg) else None

            def dm(r, name=name, k=k):
                return float(np.mean(margins(r, name, k))) - r["m"]["clean"] if _above_floor(r, cfg) else None

            entry["selectors"][name] = {"switch": mean_ci(rows, sw, cfg, _seed(seed, "abs", name, k, "switch")),
                                        "tie": mean_ci(rows, tie, cfg, _seed(seed, "abs", name, k, "tie")),
                                        "delta_m": mean_ci(rows, dm, cfg, _seed(seed, "abs", name, k, "dm"))}
        for a, b in spec["contrasts"]:
            def dsw(r, a=a, b=b, k=k):
                if not _above_floor(r, cfg):
                    return None
                tau = tau_of[r["relation"]]
                x, y = switch(margins(r, a, k)[0], tau), switch(margins(r, b, k)[0], tau)
                return None if x is None or y is None else x - y

            def ddm(r, a=a, b=b, k=k):
                return margins(r, a, k)[0] - margins(r, b, k)[0] if _above_floor(r, cfg) else None

            entry["contrasts"][f"{a} - {b}"] = {"switch": mean_ci(rows, dsw, cfg, _seed(seed, "abs", a, b, k, "switch")),
                                                "delta_m": mean_ci(rows, ddm, cfg, _seed(seed, "abs", a, b, k, "dm"))}
        out["by_k"][str(k)] = entry
    return out


# --------------------------------------------------------------------------
# 3. Recovery against perturbation size
# --------------------------------------------------------------------------


def rho(r: dict[str, Any], name: str, k: int) -> float | None:
    """Realized norm of the set's delta over the FULL patch's realized norm."""
    full = r["ref_norms"]["FULL"][1]
    if not full:
        return None
    if name == "RANDOM":
        return float(np.mean([d["norms"][1] for d in r["random"][str(k)]])) / full
    return r["sets"][name][str(k)]["norms"][1] / full


def path(r: dict[str, Any], name: str, ks: list[int]) -> list[tuple[float, float]] | None:
    """(0, 0) then (rho_k, F_k) over the k grid; None if a point is undefined."""
    pts = [(0.0, 0.0)]
    for k in ks:
        x, f = rho(r, name, k), r["sets"][name][str(k)]["F"]
        if x is None or f is None:
            return None
        pts.append((x, f))
    return pts


def alpha_path(r: dict[str, Any], name: str) -> list[tuple[float, float]] | None:
    """(0, 0) then (rho, F) of the same set scaled by each alpha of the stored M4 curve at the primary k (5.5); None if absent."""
    curve = (r.get("alpha") or {}).get(name)
    full = r["ref_norms"]["FULL"][1]
    if not curve or not full or not r["full_gain"]:
        return None
    return [(0.0, 0.0)] + [(d["realized_norm"] / full, d["effect"] / r["full_gain"]) for d in sorted(curve, key=lambda d: d["alpha"])]


def at_rho(pts: list[tuple[float, float]], target: float) -> float | None:
    """F at the path's first crossing of ``target`` by linear interpolation; None if the path never reaches it."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(x0, x1) <= target <= max(x0, x1):
            return y0 if x1 == x0 else y0 + (y1 - y0) * (target - x0) / (x1 - x0)
    return None


def size_analysis(run: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    rows, cfg = run["rows"], run["cfg"]
    pk = int(run["report"]["primary_k"])
    ks = sorted(int(k) for k in cfg["k_grid"])
    out: dict[str, Any] = {"primary_k": pk, "at_primary_k": {}, "matched": {"k": {}, "alpha": {}}}
    for name in [*spec["signals"], "RANDOM"]:
        def x(r, name=name):
            return rho(r, name, pk) if _above_floor(r, cfg) else None

        def f(r, name=name):
            if not _above_floor(r, cfg):
                return None
            if name == "RANDOM":
                v = [d["F"] for d in r["random"][str(pk)] if d["F"] is not None]
                return float(np.mean(v)) if v else None
            return r["sets"][name][str(pk)]["F"]

        out["at_primary_k"][name] = {"rho": mean_ci(rows, x, cfg, _seed(seed, "size", name, "rho")),
                                     "F": mean_ci(rows, f, cfg, _seed(seed, "size", name, "F"))}
    # two ways to reach a size: more features along the k grid, or the same primary-k set scaled by alpha (M4)
    paths = {"k": lambda r, name: path(r, name, ks), "alpha": alpha_path}
    for along, get in paths.items():
        for a, b in spec["pairs"]:
            per: dict[str, Any] = {}
            for target in spec["rho"]:
                not_reached = 0

                def diff(r, target=target, get=get, a=a, b=b):
                    nonlocal not_reached
                    if not _above_floor(r, cfg):
                        return None
                    pa, pb = get(r, a), get(r, b)
                    fa = at_rho(pa, target) if pa else None
                    fb = at_rho(pb, target) if pb else None
                    if fa is None or fb is None:
                        not_reached += 1
                        return None
                    return fa - fb

                c = mean_ci(rows, diff, cfg, _seed(seed, "size", along, a, b, target))
                per[str(target)] = {"difference": c, "trials_not_reached": not_reached,
                                    "trials_above_floor": sum(1 for r in rows if _above_floor(r, cfg))}
            out["matched"][along][f"{a} - {b}"] = per
    return out


# --------------------------------------------------------------------------
# 4. The swap per cell
# --------------------------------------------------------------------------


def swap_cells(cfg: dict[str, Any], cache: dict, ctx: RunContext | None, seed: int) -> dict[str, Any]:
    xcfg = yaml.safe_load(Path(cfg["swap"]["config"]).read_text(encoding="utf-8"))
    runs = {name: load(entry, cache, ctx) for name, entry in xcfg["models"].items()}
    cells = sorted({c for entry in xcfg["models"].values() for c in entry["matched_cells"]})
    out: dict[str, Any] = {"cells": {}, "swapped_saes": {}}
    for name, run in runs.items():
        variants = ((run["report"].get("extension") or {}).get("variants")) or {}
        for tag, v in variants.items():
            layers = sorted({r["layer"] for r in run["rows"] if r.get("sae") == tag})
            out["swapped_saes"][f"{name}: {tag}"] = {"layers": layers, "variance_explained": v.get("variance_explained"),
                                                    "n3_margin": v.get("n3_margin"), "mean_l0": v.get("mean_l0"),
                                                    "low_fidelity": v.get("low_fidelity")}
    for cell in cells:
        sub = {name: C.matched(run["rows"], [cell], xcfg["matched_relations"]) for name, run in runs.items()}
        per: dict[str, Any] = {}
        for c in xcfg["contrasts"]:
            m, ref = c["model"], c["reference"]
            per[c["name"]] = {str(k): _brief_cross(C.cross(sub[m], runs[m]["cfg"], sub[ref], runs[ref]["cfg"], k=k,
                                                           seed=_seed(seed, "swap", cell, c["name"], k)))
                              for k in cfg["swap"]["k"]}
        out["cells"][cell] = per
    return _jsonable(out)


def _brief_cross(c: dict[str, Any] | None) -> dict[str, Any] | None:
    if not c:
        return None
    c.pop("_verdict", None)
    out = _brief(c)
    out.update(model_mean=c["model_mean"], reference_mean=c["reference_mean"], pairs=c["pairs"])
    return out


# --------------------------------------------------------------------------
# 5. JL - LL against SAE fidelity
# --------------------------------------------------------------------------


def fidelity_combos(name: str, entry: dict[str, Any], run: dict[str, Any], ks: list[int], seed: int) -> list[dict[str, Any]]:
    rows, cfg, rep = run["rows"], run["cfg"], run["report"]
    variants = ((rep.get("extension") or {}).get("variants")) or {}
    combos = []
    for layer, site in sorted({(r["layer"], r["site"]) for r in rows}):
        tags = sorted({r.get("sae") for r in rows if r["layer"] == layer and r["site"] == site}, key=str)
        for tag in tags:
            sub = [r for r in rows if r["layer"] == layer and r["site"] == site and r.get("sae") == tag]
            if tag is None:
                ve = run["s0"]["sae"][str(layer)]["variance_explained"]
            elif tag in variants:
                ve = variants[tag]["variance_explained"]
            else:
                raise RunContractError(f"{name}: SAE variant {tag} has no Stage 0 entry in the run's report")
            item = {"run": name, "cell": f"{layer}/{site}", "sae": tag or "stage0", "family": entry["family"], "release": entry["release"],
                    "variance_explained": ve, "M1": _brief(_pop(S2.contrast(sub, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg,
                                                                            seed=_seed(seed, "fid", name, layer, site, tag, "M1"))))}
            for k in ks:
                item[f"M2_k{k}"] = _brief(_pop(S2.contrast(sub, "JL-OUT", "LL-OUT", "M2", k=k, cfg=cfg,
                                                           seed=_seed(seed, "fid", name, layer, site, tag, "M2", k))))
            combos.append(item)
    return combos


def _pop(c: dict[str, Any] | None) -> dict[str, Any] | None:
    if c is not None:
        c.pop("_verdict", None)
    return c


def fidelity_correlations(combos: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset, keep in (("all", lambda c: True), ("gemma", lambda c: c["family"] == "gemma")):
        sel = [c for c in combos if keep(c)]
        per = {}
        for key in ["M1", *(f"M2_k{k}" for k in ks)]:
            pts = [(c["variance_explained"], c[key]["estimate"]) for c in sel if c[key]]
            rho_ = stats.spearman([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 else None
            per[key] = {"spearman": None if rho_ is None or not np.isfinite(rho_) else float(rho_), "n": len(pts)}
        out[subset] = per
    return out


# --------------------------------------------------------------------------
# 6. Leave one relation out
# --------------------------------------------------------------------------


def leave_one_out(run: dict[str, Any], seed: int) -> dict[str, Any]:
    rows, cfg = run["rows"], run["cfg"]
    pk = int(run["report"]["primary_k"])
    out = {}
    for rel in sorted({r["relation"] for r in rows}):
        sub = [r for r in rows if r["relation"] != rel]
        out[rel] = {"pairs": len({r["pair"] for r in sub}),
                    "P2.1": _brief(_pop(S2.contrast(sub, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=_seed(seed, "loro", rel, "M1")))),
                    "P2.2": _brief(_pop(S2.contrast(sub, "JL-OUT", "LL-OUT", "M2", k=pk, cfg=cfg, seed=_seed(seed, "loro", rel, "M2"))))}
    return out


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    for key in ("models", "sesoi", "absolute", "size", "swap", "fidelity", "seed"):
        if key not in cfg:
            raise RunContractError(f"reanalyze needs `{key}` (configs/analysis/posthoc.yaml)")
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    seed = int(cfg["seed"])
    report: dict[str, Any] = {"status_of_results": "post hoc or descriptive; no verdict of record changes"}
    sens, mismatches = sesoi_sensitivity(cfg, cache, ctx)
    if mismatches:
        raise GateFailed("reproduction", f"{len(mismatches)} values recomputed at the declared SESOI differ from the reports", mismatches[:20],
                         report={"mismatches": mismatches})
    report["sesoi"] = sens
    report["absolute"], report["size"], report["leave_one_relation_out"] = {}, {}, {}
    for name, entry in cfg["models"].items():
        r = load(entry, cache, ctx)
        report["absolute"][name] = absolute_endpoints(r, cfg["absolute"], _seed(seed, name))
        report["size"][name] = size_analysis(r, cfg["size"], _seed(seed, name))
        report["leave_one_relation_out"][name] = leave_one_out(r, _seed(seed, name))
    report["swap_cells"] = swap_cells(cfg, cache, ctx, seed)
    ks = [int(k) for k in cfg["fidelity"]["k"]]
    combos = []
    for name, entry in {**cfg["models"], **(cfg.get("sae_runs") or {})}.items():
        combos += fidelity_combos(name, entry, load(entry, cache, ctx), ks, _seed(seed, "fidelity"))
    report["fidelity"] = {"combinations": combos, "spearman": fidelity_correlations(combos, ks)}
    report["provenance"] = {f"{run_key[0]}": v["provenance"] for run_key, v in cache.items()}
    report = _jsonable(report)
    return Outcome(status="PASS", summary=_summary(report, cfg), report=report)


# --------------------------------------------------------------------------
# SUMMARY.md
# --------------------------------------------------------------------------


def _f(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def _mci(c: dict[str, Any] | None, signed: bool = False) -> str:
    if not c:
        return "n/a"
    if not c.get("ci95"):
        return f"{_f(c['estimate'], signed)} (one pair)"
    return f"{_f(c['estimate'], signed)} [{_f(c['ci95'][0], signed)}, {_f(c['ci95'][1], signed)}]"


def _cv(c: dict[str, Any] | None) -> str:
    if not c:
        return "not testable"
    return f"{c['estimate']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {c['reading']}"


def _summary(report: dict[str, Any], cfg: dict[str, Any]) -> str:
    grid = report["sesoi"]["grid"]
    lines = ["## What ran", "",
             "The post hoc analysis: post hoc re-analyses of the paper's test runs, CPU, from their results.zip and the Stage 0 decisions they "
             "used. Pair means; cluster bootstrap within relation; 95% intervals; trials below the M2 full-gain floor left out wherever F or "
             "a set margin enters. Unsquared throughout. Every result here is post hoc or descriptive; no verdict of record changes.", "",
             "## 1. SESOI sensitivity", "",
             "Recomputed with each source's own procedure and seeds. At the declared SESOI every value equals the run reports and the "
             "The cross-model comparison (X3) and the SAE swap analysis reports (checked). Intervals do not depend on the SESOI; the readings do. Rows whose reading "
             "changes across the grid are marked *.", ""]
    extra = report["sesoi"].get("fields_not_in_reports") or []
    if extra:
        lines += [f"Fields the recomputation has and an older report lacks (listed, not failed; every value the reports hold is "
                  f"reproduced): {len(extra)}. " + "; ".join(extra[:8]) + ("; ..." if len(extra) > 8 else ""), ""]
    head = "| Contrast | Estimate [95%] | 90% | " + " | ".join(f"SESOI {s}" for s in grid) + " |"
    sep = "|---|---|---|" + "---|" * len(grid)
    for group, per_source in (("run", report["sesoi"]["runs"]), ("cross", report["sesoi"]["cross"])):
        for source, per in per_source.items():
            lines += [f"### {source}", "", head, sep]
            for cname, by_s in per.items():
                any_c = next((c for c in by_s.values() if c), None)
                if not any_c:
                    lines.append(f"| {cname} | not testable | | " + " | ".join("" for _ in grid) + " |")
                    continue
                readings = [by_s[str(s)]["reading"] if by_s.get(str(s)) else "n/a" for s in grid]
                mark = " *" if len(set(readings)) > 1 else ""
                ci90 = f"[{any_c['ci90'][0]:+.3f}, {any_c['ci90'][1]:+.3f}]" if any_c.get("ci90") else "n/a"
                lines.append(f"| {cname}{mark} | {any_c['estimate']:+.3f} [{any_c['ci95'][0]:+.3f}, {any_c['ci95'][1]:+.3f}] | "
                             f"{ci90} | " + " | ".join(readings) + " |")
            lines.append("")
    lines += ["## 2. Absolute endpoints", "",
              "Donor-switch share: per trial m(S) > 0 counts 1, m(S) < 0 counts 0, and |m(S)| < tau is a tie and is left out (rule 5, 5.2); "
              "tau = tau_tie of the trial's relation (run's Stage 0). Delta m = m(S) - m_clean (log-sum-exp margin, log-odds between the "
              "answer sets). RANDOM: per trial the mean over its draws. Descriptive: no SESOI is declared for these endpoints.", ""]
    for name, a in report["absolute"].items():
        lines += [f"### {name} (primary k {a['primary_k']}; tau_tie " + ", ".join(f"{k} {v:.3g}" for k, v in a["tau_tie"].items()) + ")", "",
                  "| k | Selector | Switch share [95%] | Tie share | Delta m [95%] | Pairs |", "|---|---|---|---|---|---|"]
        for k, e in a["by_k"].items():
            for sel, v in e["selectors"].items():
                lines.append(f"| {k} | {sel} | {_mci(v['switch'])} | {_f((v['tie'] or {}).get('estimate'))} | {_mci(v['delta_m'], True)} | "
                             f"{(v['delta_m'] or {}).get('n_pairs', 0)} |")
            for cname, v in e["contrasts"].items():
                lines.append(f"| {k} | {cname} | {_mci(v['switch'], True)} | | {_mci(v['delta_m'], True)} | {(v['delta_m'] or {}).get('n_pairs', 0)} |")
        lines.append("")
    lines += ["## 3. Recovery against perturbation size", "",
              "rho = realized norm of the set's delta / realized norm of the FULL patch's delta. Matched: F at rho* on each selector's "
              "path, first crossing, linear interpolation; difference on trials where both paths reach rho*. Path k: (0, 0), then "
              "(rho_k, F_k) over the k grid (more features). Path alpha: (0, 0), then the primary-k set scaled by each alpha of the stored "
              "M4 curve (the same features, larger or smaller). Descriptive; not an off-target measure.", ""]
    for name, s in report["size"].items():
        lines += [f"### {name} (primary k {s['primary_k']})", "", "| Selector | rho [95%] | F [95%] |", "|---|---|---|"]
        for sel, v in s["at_primary_k"].items():
            lines.append(f"| {sel} | {_mci(v['rho'])} | {_mci(v['F'])} |")
        lines += ["", "| Path | Pair | rho* | F difference [95%] | Pairs | Trials not reached / above floor |", "|---|---|---|---|---|---|"]
        for along, by_pair in s["matched"].items():
            for pair, per in by_pair.items():
                for target, v in per.items():
                    lines.append(f"| {along} | {pair} | {target} | {_mci(v['difference'], True)} | {(v['difference'] or {}).get('n_pairs', 0)} | "
                                 f"{v['trials_not_reached']} / {v['trials_above_floor']} |")
        lines.append("")
    sw = report["swap_cells"]
    lines += ["## 4. The swap per cell", "",
              "The SAE swap's contrasts in each concept cell separately; three-way against the declared M2 SESOI; no Holm (descriptive). The SAE swap "
              "family stays the result of record.", "", "| Cell | Contrast | k | Difference [95%], reading | Model / reference JL - LL | Pairs |",
              "|---|---|---|---|---|---|"]
    for cell, per in sw["cells"].items():
        for cname, by_k in per.items():
            for k, c in by_k.items():
                lines.append(f"| {cell} | {cname} | {k} | {_cv(c)} | " + (f"{c['model_mean']:+.3f} / {c['reference_mean']:+.3f} | "
                             f"{c['pairs'][0]} / {c['pairs'][1]} |" if c else "n/a | n/a |"))
    lines += ["", "Swapped SAEs on their host (Stage 0, from the X7 reports):", "", "| Arm: SAE | Layers | Variance explained | Layer-match margin | Mean L0 |",
              "|---|---|---|---|---|"]
    for key, v in sw["swapped_saes"].items():
        l0 = "n/a" if v["mean_l0"] is None else f"{v['mean_l0']:.1f}"
        lines.append(f"| {key} | {', '.join(map(str, v['layers']))} | {_f(v['variance_explained'])} | {_f(v['n3_margin'])} | {l0} |")
    ks = [int(k) for k in cfg["fidelity"]["k"]]
    lines += ["", "## 5. JL-OUT - LL-OUT against SAE fidelity", "",
              "Per (run, cell, SAE): the contrast with the P2 procedure; variance explained of the SAE on its host (Stage 0). Spearman across "
              "combinations, descriptive (combinations share pairs and differ in depth, cell and model; no p-value).", "",
              "| Run | Cell | SAE | Release | VE | M1 | " + " | ".join(f"M2 k = {k}" for k in ks) + " |", "|---|---|---|---|---|---|" + "---|" * len(ks)]
    for c in report["fidelity"]["combinations"]:
        lines.append(f"| {c['run']} | {c['cell']} | {c['sae']} | {c['release']} | {c['variance_explained']:.3f} | {_cv(c['M1'])} | "
                     + " | ".join(_cv(c[f'M2_k{k}']) for k in ks) + " |")
    lines += ["", "| Combinations | " + " | ".join(report["fidelity"]["spearman"]["all"]) + " |", "|---|" + "---|" * len(report["fidelity"]["spearman"]["all"])]
    for subset, per in report["fidelity"]["spearman"].items():
        lines.append(f"| {subset} | " + " | ".join(f"{_f(v['spearman'], True)} (n {v['n']})" for v in per.values()) + " |")
    lines += ["", "## 6. Leave one relation out", "", "| Model | Left out | Pairs | P2.1 (M1) | P2.2 (M2, primary k) |", "|---|---|---|---|---|"]
    for name, per in report["leave_one_relation_out"].items():
        for rel, v in per.items():
            lines.append(f"| {name} | {rel} | {v['pairs']} | {_cv(v['P2.1'])} | {_cv(v['P2.2'])} |")
    lines += ["", "## What this permits", "",
              "Everything here is post hoc or descriptive and sits next to the "
              "results of record; none replaces a verdict. Verdicts are three-way; inconclusive is never 'no difference'.", "",
              "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
