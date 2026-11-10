"""``python -m saeselect additivity``: post hoc re-analyses of the paper's test runs.

    python -m saeselect additivity --config configs/analysis/additivity.yaml --out analysis/additivity

CPU only. Reads the three primary Stage 2 test runs (each checked against its sealed frozen protocol, as the post hoc analysis does), their
development runs (for the SAE-ALL cell rule and the candidate cells) and the X8 test runs (off-target readout). Pair means,
cluster bootstrap within relation with each run's frozen resamples, 95% intervals; trials below the frozen M2 floor are left
out wherever F or a set margin enters. Unsquared. Every result is post hoc or descriptive; no verdict of record changes.

1. Ceilings: per cell, the share of the full patch that SAE-ALL (every active feature) and ERR (the SAE error term) recover,
   each selector's F at every k, and its recovery relative to SAE-ALL (ratio of pair means).
2. The primary contrasts (P2.1, P2.2, with the runs' own seeds) restricted to the cells where the SAE carries the transfer:
   the declared eligibility rule with SAE-ALL in place of FULL (median SAE-ALL share on the development trials at least
   the declared threshold).
3. Superadditivity per cell, selector and k: the additive prediction (summed single effects of the set's features over the
   full gain) against the recovered share.
4. F(rho) curves per cell and selector (the k path and the alpha path); the share of trials whose set must be scaled past the
   donor (alpha > 1) to reach each target size; matched-size contrasts on the alpha path restricted to interpolation
   (alpha <= 1 for both selectors).
5. JL-OUT against NORM on M2 at each declared k and on M1-abs, pooled and per cell, three-way against the run's SESOI.
6. Off-target change per unit of recovery (X8 test runs): O1 / F as a ratio of pair means, and O3, per selector and k.
7. Candidate cells: the development median R_FULL of every candidate cell, and the cells that entered the test.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from saeselect import stats
from saeselect.analysis import rebuild as R
from saeselect.analysis import posthoc as PH
from saeselect.stats import sub_seed as _seed
from saeselect.provenance import GateFailed
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2


def _cell(r: dict[str, Any]) -> str:
    return f"{r['layer']}/{r['site']}"


def ref_share(r: dict[str, Any], name: str) -> float | None:
    """Share of the full patch recovered by a reference transfer (SAE-ALL or ERR)."""
    gain = r["m"]["FULL"] - r["m"]["clean"]
    return (r["m"][name] - r["m"]["clean"]) / gain if gain else None


def pair_matrix(rows: list[dict[str, Any]], fns: list) -> tuple[np.ndarray, list[str]]:
    """Per concept pair, the mean of each per-trial value over trials where every value is defined; relation strata."""
    acc: dict[str, list[list[float]]] = defaultdict(list)
    rel: dict[str, str] = {}
    for r in rows:
        vals = [f(r) for f in fns]
        if all(v is not None and math.isfinite(v) for v in vals):
            acc[r["pair"]].append(vals)
            rel[r["pair"]] = r["relation"]
    pairs = sorted(acc)
    return np.array([np.mean(acc[p], axis=0) for p in pairs]).reshape(len(pairs), len(fns)), [rel[p] for p in pairs]


def ratio_ci(rows: list[dict[str, Any]], num, den, cfg: dict[str, Any], seed: int) -> dict[str, Any] | None:
    """Ratio of pair means, num / den, with a cluster bootstrap within relation."""
    m, strata = pair_matrix(rows, [num, den])
    if len(m) < 2 or m[:, 1].sum() == 0:
        return None
    b = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(m, n_resamples=b["n_resamples"], seed=seed, strata=strata,
                                  statistic=lambda a: float(a[:, 0].mean() / a[:, 1].mean()) if a[:, 1].mean() else float("nan"))
    samples = res.samples[np.isfinite(res.samples)]
    lo, hi = np.percentile(samples, [(1 - b["level"]) / 2 * 100, (1 + b["level"]) / 2 * 100])
    return {"estimate": res.estimate, "ci95": [float(lo), float(hi)], "n_pairs": len(m)}


def verdict_ci(rows: list[dict[str, Any]], value, cfg: dict[str, Any], seed: int, *, sesoi: float) -> dict[str, Any] | None:
    """A per-trial contrast, pair means, cluster bootstrap within relation, three-way verdict (S2's procedure)."""
    vals, _, strata = S2._cluster_means(rows, value)
    if len(vals) < 2:
        return None
    b = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(vals, n_resamples=b["n_resamples"], seed=seed, strata=strata)
    v = stats.verdict(res, sesoi=sesoi, predicted_sign=1, level=b["level"], tost_level=b["tost_level"])
    out = {"estimate": v.estimate, "ci95": list(v.ci95), "ci90": list(v.ci90), "n_pairs": len(vals), "better": v.better,
           "practically_better": v.practically_better, "equivalent": v.equivalent, "worse": v.worse}
    out["reading"] = PH.reading(out)
    return out


# --------------------------------------------------------------------------
# 1. Ceilings
# --------------------------------------------------------------------------


def ceilings(rows: list[dict[str, Any]], cfg: dict[str, Any], signals: list[str], ks: list[int], pk: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    above = [r for r in rows if PH._above_floor(r, cfg)]
    for cell in sorted({_cell(r) for r in rows}):
        sub = [r for r in above if _cell(r) == cell]
        pools = [r["pool_size"] for r in rows if _cell(r) == cell]
        c: dict[str, Any] = {"trials_above_floor": len(sub),
                             "pool_size": {"median": float(np.median(pools)), "q25": float(np.percentile(pools, 25)),
                                           "q75": float(np.percentile(pools, 75))} if pools else None}
        for ref in ("SAE-ALL", "ERR"):
            vals = [v for v in (ref_share(r, ref) for r in sub) if v is not None]
            c[ref] = {"mean_ci": PH.mean_ci(sub, lambda r, ref=ref: ref_share(r, ref), cfg, _seed(seed, "ceil", cell, ref)),
                      "median": float(np.median(vals)) if vals else None}
        c["F"] = {name: {str(k): PH.mean_ci(sub, lambda r, name=name, k=k: r["sets"][name][str(k)]["F"], cfg,
                                             _seed(seed, "ceilF", cell, name, k)) for k in ks} for name in signals}
        c["relative_to_SAE_ALL"] = {name: ratio_ci(sub, lambda r, name=name: r["sets"][name][str(pk)]["F"],
                                                   lambda r: ref_share(r, "SAE-ALL"), cfg, _seed(seed, "rel", cell, name))
                                    for name in signals}
        out[cell] = c
    return out


# --------------------------------------------------------------------------
# 2. The primary contrasts in the cells where the SAE carries the transfer
# --------------------------------------------------------------------------


def sae_carried_cells(dev_rows: list[dict[str, Any]], cfg: dict[str, Any], threshold: float) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in dev_rows:
        if PH._above_floor(r, cfg):
            v = ref_share(r, "SAE-ALL")
            if v is not None:
                by[_cell(r)].append(v)
    med = {c: float(np.median(v)) for c, v in sorted(by.items())}
    return {"dev_median_SAE_ALL": med, "threshold": threshold, "cells": [c for c, m in med.items() if m >= threshold]}


def restricted_primary(rows: list[dict[str, Any]], cfg: dict[str, Any], cells: list[str], pk: int) -> dict[str, Any]:
    sub = [r for r in rows if _cell(r) in cells]
    seed = cfg["seed"]
    return {"cells": cells, "P2.1": PH._brief(S2.contrast(sub, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=seed + 1)),
            "P2.2": PH._brief(S2.contrast(sub, "JL-OUT", "LL-OUT", "M2", k=pk, cfg=cfg, seed=seed + 2))}


# --------------------------------------------------------------------------
# 3. Superadditivity
# --------------------------------------------------------------------------


def additive(r: dict[str, Any], name: str, k: int) -> float | None:
    """Summed single effects of the set's active features over the full gain."""
    eff = r.get("effects") or {}
    gain = r["full_gain"]
    if not gain:
        return None
    return sum(float(eff.get(str(f), eff.get(f, 0.0))) for f in r["sets"][name][str(k)]["active"]) / gain


def superadditivity(rows: list[dict[str, Any]], cfg: dict[str, Any], signals: list[str], ks: list[int], seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    above = [r for r in rows if PH._above_floor(r, cfg)]
    for cell in sorted({_cell(r) for r in above}):
        sub = [r for r in above if _cell(r) == cell]
        per: dict[str, Any] = {}
        for name in signals:
            per[name] = {}
            for k in ks:
                pairs = [(additive(r, name, k), r["sets"][name][str(k)]["F"]) for r in sub]
                pairs = [(a, f) for a, f in pairs if a is not None and f is not None]
                if not pairs:
                    continue
                med_a, med_f = float(np.median([a for a, _ in pairs])), float(np.median([f for _, f in pairs]))
                per[name][str(k)] = {"median_additive": med_a, "median_F": med_f, "ratio_of_medians": med_f / med_a if med_a else None,
                                     "F_minus_additive": PH.mean_ci(sub, lambda r, name=name, k=k: (
                                         None if additive(r, name, k) is None else r["sets"][name][str(k)]["F"] - additive(r, name, k)),
                                         cfg, _seed(seed, "super", cell, name, k))}
        out[cell] = per
    return out


# --------------------------------------------------------------------------
# 4. F(rho) curves, alpha > 1 shares, matched size by interpolation only
# --------------------------------------------------------------------------


def alpha_at_one(r: dict[str, Any], name: str) -> float | None:
    """rho of the primary-k set unscaled (alpha = 1) from the stored M4 curve."""
    full = r["ref_norms"]["FULL"][1]
    for d in (r.get("alpha") or {}).get(name) or []:
        if d["alpha"] == 1.0 and full:
            return d["realized_norm"] / full
    return None


def interp_path(r: dict[str, Any], name: str) -> list[tuple[float, float]] | None:
    """The alpha path up to alpha = 1: shrinking the set toward the source only, never past the donor."""
    curve = (r.get("alpha") or {}).get(name)
    full = r["ref_norms"]["FULL"][1]
    if not curve or not full or not r["full_gain"]:
        return None
    return [(0.0, 0.0)] + [(d["realized_norm"] / full, d["effect"] / r["full_gain"]) for d in sorted(curve, key=lambda d: d["alpha"]) if d["alpha"] <= 1.0]


def curves(rows: list[dict[str, Any]], cfg: dict[str, Any], spec: dict[str, Any], ks: list[int], seed: int) -> dict[str, Any]:
    above = [r for r in rows if PH._above_floor(r, cfg)]
    out: dict[str, Any] = {"cells": {}, "alpha_above_one_share": {}, "matched_interpolation_only": {}}
    for cell in sorted({_cell(r) for r in above}):
        sub = [r for r in above if _cell(r) == cell]
        per: dict[str, Any] = {}
        for name in spec["signals"]:
            kpts = []
            for k in ks:
                x = [PH.rho(r, name, k) for r in sub]
                f = [r["sets"][name][str(k)]["F"] for r in sub]
                ok = [(a, b) for a, b in zip(x, f) if a is not None and b is not None]
                if ok:
                    kpts.append({"k": k, "rho": float(np.mean([a for a, _ in ok])), "F": float(np.mean([b for _, b in ok]))})
            apts: dict[float, list[tuple[float, float]]] = defaultdict(list)
            for r in sub:
                full = r["ref_norms"]["FULL"][1]
                for d in (r.get("alpha") or {}).get(name) or []:
                    if full and r["full_gain"]:
                        apts[d["alpha"]].append((d["realized_norm"] / full, d["effect"] / r["full_gain"]))
            per[name] = {"k_path": kpts, "alpha_path": [{"alpha": a, "rho": float(np.mean([p[0] for p in v])),
                                                          "F": float(np.mean([p[1] for p in v]))} for a, v in sorted(apts.items())]}
        out["cells"][cell] = per
    for name in spec["signals"]:
        out["alpha_above_one_share"][name] = {}
        for target in spec["rho"]:
            vals = [alpha_at_one(r, name) for r in above]
            vals = [v for v in vals if v is not None]
            out["alpha_above_one_share"][name][str(target)] = float(np.mean([v < target for v in vals])) if vals else None
    for a, b in spec["pairs"]:
        per = {}
        for target in spec["rho"]:
            not_reached = 0

            def diff(r, a=a, b=b, target=target):
                nonlocal not_reached
                if not PH._above_floor(r, cfg):
                    return None
                pa, pb = interp_path(r, a), interp_path(r, b)
                fa = PH.at_rho(pa, target) if pa else None
                fb = PH.at_rho(pb, target) if pb else None
                if fa is None or fb is None:
                    not_reached += 1
                    return None
                return fa - fb

            c = PH.mean_ci(rows, diff, cfg, _seed(seed, "interp", a, b, target))
            per[str(target)] = {"difference": c, "trials_not_reached": not_reached, "trials_above_floor": len(above)}
        out["matched_interpolation_only"][f"{a} - {b}"] = per
    return out


# --------------------------------------------------------------------------
# 5. JL-OUT against NORM
# --------------------------------------------------------------------------


def jl_vs_norm(rows: list[dict[str, Any]], cfg: dict[str, Any], ks: list[int], seed: int) -> dict[str, Any]:
    sesoi = cfg["analysis"]["sesoi"]

    def m2(k):
        return lambda r: (None if not PH._above_floor(r, cfg) else r["sets"]["JL-OUT"][str(k)]["F"] - r["sets"]["NORM"][str(k)]["F"])

    def m1abs(r):
        x, y = r["metrics"]["JL-OUT"].get("M1_abs"), r["metrics"]["NORM"].get("M1_abs")
        return None if x is None or y is None else x - y

    out: dict[str, Any] = {}
    for scope in ["pooled", *sorted({_cell(r) for r in rows})]:
        sub = rows if scope == "pooled" else [r for r in rows if _cell(r) == scope]
        out[scope] = {f"M2 k = {k}": verdict_ci(sub, m2(k), cfg, _seed(seed, "jlnorm", scope, k), sesoi=sesoi["M2"]) for k in ks}
        out[scope]["M1-abs"] = verdict_ci(sub, m1abs, cfg, _seed(seed, "jlnorm", scope, "m1abs"), sesoi=sesoi["M1"])
    return out


# --------------------------------------------------------------------------
# 6. Off-target change per unit of recovery
# --------------------------------------------------------------------------


def offtarget_per_recovery(rows: list[dict[str, Any]], cfg: dict[str, Any], signals: list[str], seed: int) -> dict[str, Any]:
    above = [r for r in rows if PH._above_floor(r, cfg)]
    ks = sorted({k for r in above for name in signals for k in r["sets"].get(name, {}) if "offtarget" in r["sets"][name][k]}, key=int)
    out: dict[str, Any] = {}
    for k in ks:
        out[k] = {}
        for name in signals:
            def o(key, name=name, k=k):
                return lambda r: (r["sets"][name][k]["offtarget"][key] if "offtarget" in r["sets"].get(name, {}).get(k, {}) else None)
            out[k][name] = {"O1_per_F": ratio_ci(above, o("O1"), lambda r, name=name, k=k: r["sets"][name][k]["F"], cfg,
                                                 _seed(seed, "o1f", name, k)),
                            "O3": PH.mean_ci(above, o("O3"), cfg, _seed(seed, "o3", name, k))}
    return out


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def run(ctx: RunContext) -> Outcome:
    cfg_a = ctx.config
    report: dict[str, Any] = {"models": {}, "provenance": {}}
    mismatches: list[str] = []
    cache: dict = {}
    for name, entry in cfg_a["models"].items():
        run = PH.load(entry, cache, ctx, keep_effects=True)
        cfg, rows, rep = run["cfg"], run["rows"], run["report"]
        pk = int(rep["primary_k"])
        ks = sorted(int(k) for k in cfg["k_grid"])
        seed = _seed(cfg_a["seed"], name)
        # reproduction guard: the unrestricted primary contrasts with the run's own seeds equal the report
        for key, c in (("P2.1", S2.contrast(rows, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=cfg["seed"] + 1)),
                       ("P2.2", S2.contrast(rows, "JL-OUT", "LL-OUT", "M2", k=pk, cfg=cfg, seed=cfg["seed"] + 2))):
            ref = rep["analysis"]["primary"][key]
            if c is None or ref is None or c["estimate"] != ref["estimate"] or list(c["ci95"]) != list(ref["ci95"]):
                mismatches.append(f"{name} {key}: report {None if ref is None else (ref['estimate'], ref['ci95'])}, "
                                  f"recomputed {None if c is None else (c['estimate'], c['ci95'])}")
        dev_dir = Path(entry["frozen"]).parent
        dev = R.read_run(dev_dir)
        carried = sae_carried_cells(dev["rows"], cfg, cfg_a["sae_carried_threshold"])
        dev_report = dev["report"]
        sig = cfg_a["signals"]
        m: dict[str, Any] = {
            "primary_k": pk,
            "candidate_cells": {"dev_median_R_FULL": dev_report.get("r_full_median"), "eligible": rep.get("eligible_cells")},
            "ceilings": ceilings(rows, cfg, sig, ks, pk, _seed(seed, "ceil")),
            "sae_carried": {**carried, "primary": restricted_primary(rows, cfg, carried["cells"], pk)},
            "superadditivity": superadditivity(rows, cfg, sig, ks, _seed(seed, "super")),
            "curves": curves(rows, cfg, cfg_a["size"], ks, _seed(seed, "curves")),
            "jl_vs_norm": jl_vs_norm(rows, cfg, sorted({k for k in (4, pk) if k in ks}), _seed(seed, "jlnorm")),
        }
        x8 = cfg_a.get("offtarget", {}).get(name)
        if x8:
            xrun = PH.load(x8, cache, ctx)
            m["offtarget_per_recovery"] = offtarget_per_recovery(xrun["rows"], xrun["cfg"], cfg_a["offtarget_signals"], _seed(seed, "off"))
        report["models"][name] = PH._jsonable(m)
        report["provenance"][name] = {"run": entry["run"], "dev_run": dev_dir.as_posix(), **run["provenance"]}
        del run, rows, dev
        cache.clear()
    if mismatches:
        raise GateFailed("reproduction", f"{len(mismatches)} recomputed primary values differ from the reports", mismatches[:10],
                         report={"mismatches": mismatches})
    return Outcome(status="PASS", summary=_summary(report), report=report, send_back=[])


def _ci(c: dict[str, Any] | None, d: int = 3) -> str:
    if not c:
        return "n/a"
    if not c.get("ci95"):
        return f"{c['estimate']:+.{d}f}"
    return f"{c['estimate']:+.{d}f} [{c['ci95'][0]:+.{d}f}, {c['ci95'][1]:+.{d}f}]"


def _summary(report: dict[str, Any]) -> str:
    lines = ["## What ran", "", "The additivity analysis: post hoc re-analyses of the paper's three primary test runs (and the X8 test runs for the "
             "off-target ratio). Pair means; cluster bootstrap within relation; 95% intervals; trials below the M2 floor left out where F "
             "enters. Unsquared. Every result is post hoc or descriptive; the recomputed primary contrasts equal the reports (checked).", ""]
    for name, m in report["models"].items():
        lines += [f"## {name} (primary k {m['primary_k']})", "", "### Ceilings and pool sizes", "",
                  "| Cell | Pool (median) | SAE-ALL share, mean [95%] / median | ERR share, mean [95%] / median |", "|---|---|---|---|"]
        for cell, c in m["ceilings"].items():
            lines.append(f"| {cell} | {c['pool_size']['median'] if c['pool_size'] else 'n/a'} | {_ci(c['SAE-ALL']['mean_ci'])} / "
                         f"{c['SAE-ALL']['median']:.3f} | {_ci(c['ERR']['mean_ci'])} / {c['ERR']['median']:.3f} |")
        sc = m["sae_carried"]
        lines += ["", f"### Cells where the SAE carries the transfer (dev median SAE-ALL >= {sc['threshold']}): {sc['cells']}", "",
                  f"- Dev medians: {', '.join(f'{c} {v:.3f}' for c, v in sc['dev_median_SAE_ALL'].items())}",
                  f"- P2.1 restricted: {_ci(sc['primary']['P2.1'])}, {sc['primary']['P2.1']['reading'] if sc['primary']['P2.1'] else ''}",
                  f"- P2.2 restricted: {_ci(sc['primary']['P2.2'])}, {sc['primary']['P2.2']['reading'] if sc['primary']['P2.2'] else ''}", "",
                  "### JL-OUT minus NORM", "", "| Scope | " + " | ".join(next(iter(m["jl_vs_norm"].values()))) + " |",
                  "|---|" + "---|" * len(next(iter(m["jl_vs_norm"].values())))]
        for scope, per in m["jl_vs_norm"].items():
            lines.append(f"| {scope} | " + " | ".join(f"{_ci(v)} {v['reading'] if v else ''}" for v in per.values()) + " |")
        lines += ["", "### Superadditivity at the primary k (median additive prediction -> median recovered share)", ""]
        for cell, per in m["superadditivity"].items():
            lines.append(f"- {cell}: " + "; ".join(f"{s} {v[str(m['primary_k'])]['median_additive']:.3f} -> {v[str(m['primary_k'])]['median_F']:.3f}"
                                                   for s, v in per.items() if str(m["primary_k"]) in v))
        cu = m["curves"]
        lines += ["", "### Share of trials needing alpha > 1 to reach rho*", ""]
        lines += [f"- {s}: " + ", ".join(f"rho* {t}: {v:.2f}" for t, v in per.items() if v is not None) for s, per in cu["alpha_above_one_share"].items()]
        lines += ["", "### Matched size on the alpha path, interpolation only (alpha <= 1)", "", "| Pair | rho* | Difference [95%] | Trials not reached |", "|---|---|---|---|"]
        for pair, per in cu["matched_interpolation_only"].items():
            for t, v in per.items():
                lines.append(f"| {pair} | {t} | {_ci(v['difference'])} | {v['trials_not_reached']} of {v['trials_above_floor']} |")
        if m.get("offtarget_per_recovery"):
            lines += ["", "### Off-target change per unit of recovery (O1 / F, ratio of pair means)", ""]
            for k, per in m["offtarget_per_recovery"].items():
                lines.append(f"- k = {k}: " + "; ".join(f"{s} {_ci(v['O1_per_F'], 2)}" for s, v in per.items()))
        cc = m["candidate_cells"]
        lines += ["", f"### Candidate cells: dev median R_FULL {json.dumps(cc['dev_median_R_FULL'])}; eligible {cc['eligible']}", ""]
    lines += ["## What this permits", "", "Post hoc or descriptive context for the verdicts of record, none of which changes.",
              "", "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
