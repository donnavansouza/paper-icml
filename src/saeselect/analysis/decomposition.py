"""``python -m saeselect decomposition``: diagnostics of the stored test trials .

    python -m saeselect decomposition --config configs/analysis/decomposition.yaml --out analysis/decomposition

CPU only; no model is loaded. Post hoc: nothing here changes a verdict of record.

1. The selected tail: per run, scope (pooled, cell, task x cell), signal and k, the recovered share F, the active count a, the
   singleton utility U (the additivity analysis's additive prediction), the singleton gap to ORACLE G, the non-additive remainder N = F - U and M3;
   at the primary k, dF = dU + dN for the declared pairs (one set of resamples) with the declared reading rule. Integrity: every
   active selected feature has a stored effect, and at k = 1 a set's F equals its feature's isolated share.
2. Pooling and normalization: the declared contrasts per cell and per task x cell (unadjusted), the M1 contrasts on the M2 trial
   mask, the M2 contrasts over two other denominators (with the FULL-normalized and raw contrasts on the same mask), the SAE-ALL
   and ERR shares, and the norm-budget overshoot with reach counts.
3. Margin-specific tests: each declared family rerun with its run's own procedure and arguments while the bootstrap calls are
   recorded; every test must equal its report and match its recorded resamples; one-sided margin, TOST and reversed-sign
   p-values (approximate bootstrap tests by interval inversion), Holm within each family.
4. ATP influence audit (sensitivity, not numerical correctness): the known joint-selection discrepancy, a screen of stored ATP@in M1, and the
   ATP contrasts without the flagged pairs.
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from saeselect import stats
from saeselect.analysis import rebuild as R
from saeselect.analysis import posthoc as PH
from saeselect.analysis import additivity as ADD
from saeselect.provenance import EngineeringFailed, GateFailed, RunContractError, verify_sealed_protocol
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2

QUANTITIES = ("F", "a", "U", "G", "a_differs", "N", "M3")
SIGNATURE = ("estimate", "ci95", "ci90", "p", "n_pairs")


def _cell(r: dict[str, Any]) -> str:
    return f"{r['layer']}/{r['site']}"


def _scopes(rows: list[dict[str, Any]], pooled: bool = True) -> list[tuple[str, list[dict[str, Any]]]]:
    """The run pooled, each cell, and each task x cell."""
    out = [("pooled", rows)] if pooled else []
    out += [(c, [r for r in rows if _cell(r) == c]) for c in sorted({_cell(r) for r in rows})]
    out += [(f"{t} {c}", [r for r in rows if r["relation"] == t and _cell(r) == c])
            for t, c in sorted({(r["relation"], _cell(r)) for r in rows})]
    return out


# --------------------------------------------------------------------------
# 1. The selected tail
# --------------------------------------------------------------------------


def tail_values(r: dict[str, Any], name: str, k: int) -> dict[str, Any] | None:
    """F, active count a, singleton utility U, remainder N = F - U and M3 of one selector's set at k; None when unset."""
    e = (r["sets"].get(name) or {}).get(str(k))
    u = ADD.additive(r, name, k) if e is not None and e.get("F") is not None else None
    if u is None:
        return None
    m3 = ((r["metrics"].get(name) or {}).get("M3") or {}).get(str(k))
    return {"F": e["F"], "a": float(len(e["active"])), "U": u, "N": e["F"] - u, "M3": m3}


def _value(name: str, k: int, q: str):
    def get(r):
        v = tail_values(r, name, k)
        if v is None:
            return None
        if q in ("G", "a_differs"):  # G is nonnegative only when the active counts agree; a_differs reports how often they do not
            o = tail_values(r, "ORACLE", k)
            return None if o is None else (o["U"] - v["U"] if q == "G" else float(o["a"] != v["a"]))
        return v[q]
    return get


def integrity_violations(rows: list[dict[str, Any]], cfg: dict[str, Any], signals: list[str], ks: list[int], tol: float) -> list[str]:
    """Trials at or above the M2 floor where an active selected feature has no stored effect (at any k; the additivity analysis's helper would count
    it as zero), or where a set at k = 1 differs from its one feature's isolated effect by more than tol log-odds, i.e.
    |F - U| * (m_FULL - m_clean) > tol (on the margin scale, so a small full gain does not inflate FP32 rounding)."""
    bad = []
    for r in rows:
        if not PH._above_floor(r, cfg):
            continue
        eff = r.get("effects") or {}
        for n in signals:
            for k in ks:
                e = (r["sets"].get(n) or {}).get(str(k)) or {}
                missing = [f for f in e.get("active") or [] if str(f) not in eff and f not in eff]
                if missing:
                    bad.append(f"{r['trial']} {_cell(r)} {n} k = {k}: no stored effect for {missing[:5]}")
            v = tail_values(r, n, 1) if 1 in ks else None
            if v is not None and abs(v["N"]) * r["full_gain"] > tol:
                bad.append(f"{r['trial']} {_cell(r)} {n}: at k = 1, (F - U) * gain is {v['N'] * r['full_gain']:.3g} log-odds")
    return bad


def tail(rows, cfg: dict[str, Any], signals: list[str], ks: list[int], pk: int, seed: int) -> dict[str, Any]:
    """Pair means per scope, signal and k; at the primary k with 95% intervals. Trials at or above the M2 floor."""
    above = [r for r in rows if PH._above_floor(r, cfg)]
    return {scope: {n: {"by_k": {str(k): {q: S2._pair_mean(sub, _value(n, k, q)) for q in QUANTITIES} for k in ks},
                        "primary": {q: PH.mean_ci(sub, _value(n, pk, q), cfg, S2._hseed(seed, scope, n, q)) for q in QUANTITIES}}
                    for n in signals}
            for scope, sub in _scopes(above)}


def _resolved(c: dict[str, Any] | None) -> int:
    """+1 or -1 when the 95% interval excludes 0 on that side, 0 otherwise."""
    if not c or not c.get("ci95"):
        return 0
    lo, hi = c["ci95"]
    return 1 if lo > 0 else -1 if hi < 0 else 0


def reading_rule(e: dict[str, Any]) -> str:
    """The declared reading of dF = dU + dN. An unresolved interval does not show a zero contribution."""
    if not e.get("F") or not e["F"].get("ci95"):
        return "not testable"
    f, u, n = _resolved(e["F"]), _resolved(e.get("U")), _resolved(e.get("N"))
    if f == 0:
        return "opposing contributions, F unresolved" if u and n and u != n else "F unresolved"
    if u == f and n == f:
        return "resolved contributions from both"
    if u == f and n == -f:
        return "order reversal: singleton utility outweighs the remainder"
    if n == f and u == -f:
        return "order reversal: the remainder outweighs singleton utility"
    if u == f:
        return "resolved contribution from singleton utility"
    if n == f:
        return "resolved contribution from the remainder"
    return "unresolved"


def decomposition(rows, cfg: dict[str, Any], pairs: list[tuple[str, str]], pk: int, seed: int) -> dict[str, Any]:
    """dF = dU + dN of each pair at the primary k, per scope, on the trials where both sets are defined (at or above the floor);
    dF, dU and dN share one set of resamples, so the identity holds in every resample."""
    above = [r for r in rows if PH._above_floor(r, cfg)]
    out: dict[str, Any] = {}
    for scope, sub in _scopes(above):
        out[scope] = {}
        for a, b in pairs:
            def diff(q, a=a, b=b):
                def get(r):
                    x, y = tail_values(r, a, pk), tail_values(r, b, pk)
                    return None if x is None or y is None or x[q] is None or y[q] is None else x[q] - y[q]
                return get
            e = {q: PH.mean_ci(sub, diff(q), cfg, S2._hseed(seed, scope, a, b)) for q in ("F", "U", "N", "a", "M3")}
            e["reading"] = reading_rule(e)
            out[scope][f"{a} - {b}"] = e
    return out


# --------------------------------------------------------------------------
# 2. Pooling and normalization
# --------------------------------------------------------------------------


def per_scope_contrasts(rows, cfg: dict[str, Any], contrasts: dict[str, tuple[str, str, str]], pk: int, seed: int) -> dict[str, Any]:
    """The declared contrasts per cell and per task x cell, with the runs' procedure; unadjusted."""
    return {scope: {name: PH._brief(S2.contrast(sub, a, b, metric, k=pk if metric == "M2" else None, cfg=cfg,
                                                 seed=S2._hseed(seed, scope, name)))
                    for name, (a, b, metric) in contrasts.items()}
            for scope, sub in _scopes(rows, pooled=False)}


def common_mask(rows, cfg: dict[str, Any], contrasts: dict[str, tuple[str, str, str]], pk: int, seed: int) -> dict[str, Any]:
    """The M1 contrasts on exactly the trials that enter the M2 contrasts (at or above the floor, both F defined at the primary k)."""
    floor = cfg["m2_min_full_gain"]
    out = {}
    for name, (a, b, metric) in contrasts.items():
        if metric != "M1":
            continue

        def value(r, a=a, b=b):
            if r["full_gain"] < floor or S2._F_at(r, a, pk) is None or S2._F_at(r, b, pk) is None:
                return None
            x, y = r["metrics"][a]["M1"], r["metrics"][b]["M1"]
            return None if x is None or y is None else x - y

        out[name] = PH._brief(S2._labelled(rows, value, sesoi=cfg["analysis"]["sesoi"]["M1"], cfg=cfg, seed=S2._hseed(seed, name)))
    return out


def denominators(rows, cfg: dict[str, Any], contrasts: dict[str, tuple[str, str, str]], dens: list[str], pk: int,
                 seed: int) -> dict[str, Any]:
    """The M2 contrasts with the set margins' difference over another denominator (m_den - m_clean), on trials whose FULL gain and
    whose den gain are both at least the M2 floor; on the same mask, the FULL-normalized contrast and the raw margin-gain contrast
    (log-odds, no verdict), and the retained trials. Means of trial-level ratios."""
    floor = cfg["m2_min_full_gain"]
    sesoi = cfg["analysis"]["sesoi"]["M2"]
    out: dict[str, Any] = {}
    for den in dens:
        out[den] = {}
        for name, (a, b, metric) in contrasts.items():
            if metric != "M2":
                continue

            def value(scale, a=a, b=b, den=den):
                def get(r):
                    gain = r["m"].get(den, float("nan")) - r["m"]["clean"]
                    ea, eb = ((r["sets"].get(n) or {}).get(str(pk)) for n in (a, b))
                    if (r["full_gain"] < floor or not gain >= floor or ea is None or eb is None or ea.get("margin") is None
                            or eb.get("margin") is None):
                        return None
                    d = ea["margin"] - eb["margin"]
                    return d / gain if scale == "den" else d / r["full_gain"] if scale == "FULL" else d
                return get

            s = S2._hseed(seed, den, name)
            out[den][name] = {"n_trials": sum(value("raw")(r) is not None for r in rows),
                              "normalized": PH._brief(S2._labelled(rows, value("den"), sesoi=sesoi, cfg=cfg, seed=s)),
                              "FULL_on_this_mask": PH._brief(S2._labelled(rows, value("FULL"), sesoi=sesoi, cfg=cfg, seed=s)),
                              "raw_margin_gain": PH.mean_ci(rows, value("raw"), cfg, s)}
    return out


def overshoot(rows, cfg: dict[str, Any], signals: list[str], above_ratio: float) -> dict[str, Any] | None:
    """The new-task endpoint's norm budget, over trials at or above the M2 floor: how many walks reach B, and for those rho_intended / B of the
    selected prefix and the realized-to-intended norm ratio."""
    above = [r for r in rows if PH._above_floor(r, cfg) and "norm_budget" in r]
    if not above:
        return None
    out: dict[str, Any] = {}
    for scope, sub in _scopes(above):
        out[scope] = {}
        for n in signals:
            per = {}
            for b in sorted(sub[0]["norm_budget"]["sets"][n], key=float):
                es = [r["norm_budget"]["sets"][n][b] for r in sub]
                reached = [e for e in es if e["reached"]]
                v = np.array([e["rho_intended"] / float(b) for e in reached])
                per[b] = {"n_trials": len(es), "n_reached": len(reached)}
                if v.size:
                    per[b].update(median=float(np.median(v)), q90=float(np.percentile(v, 90)), share_above=float(np.mean(v > above_ratio)),
                                  realized_over_intended_median=float(np.median([e["norms"][1] / e["norms"][0] for e in reached if e["norms"][0]])))
            out[scope][n] = per
    return out


# --------------------------------------------------------------------------
# 3. Margin-specific tests
# --------------------------------------------------------------------------


@contextmanager
def recorded_bootstraps():
    """Every stats.cluster_bootstrap call made inside the block, in call order, with its seed, cluster count and resample count.
    The analysis modules call it through ``stats``; the original runs once per call and is restored afterwards."""
    calls: list[dict[str, Any]] = []
    original = stats.cluster_bootstrap

    def recording(*args, **kwargs):
        res = original(*args, **kwargs)
        values = args[0] if args else kwargs["cluster_values"]
        calls.append({"res": res, "n": len(values), "seed": kwargs.get("seed"), "n_resamples": kwargs.get("n_resamples")})
        return res

    stats.cluster_bootstrap = recording
    try:
        yield calls
    finally:
        stats.cluster_bootstrap = original


def _tails(samples: np.ndarray, d: float, s: int) -> tuple[float, float, float]:
    """One-sided p of the margin, TOST p and one-sided p of the reversed sign, as shares of the resamples."""
    return (float(np.mean(s * samples <= d)), float(max(np.mean(samples >= d), np.mean(samples <= -d))), float(np.mean(s * samples >= 0)))


def _same(x: Any, y: Any) -> bool:
    return (list(x) if isinstance(x, (list, tuple)) else x) == (list(y) if isinstance(y, (list, tuple)) else y)


def margin_family(recomputed: dict[str, Any], stored: dict[str, Any], calls: list[dict[str, Any]], *, sesoi, sign, level: float,
                  tost_level: float, alpha_m: float, alpha_e: float) -> tuple[dict[str, Any], list[str]]:
    """Per test of one family: the stored reading and, from the test's recorded resamples, the one-sided p of the margin, the
    TOST p and the one-sided p of the reversed sign, each Holm-adjusted within the family. A test must
    equal its report in estimate, both intervals, p and pair count, and is matched to the recorded calls on the same values;
    several matches must agree in their tail probabilities. ``sesoi`` and ``sign`` map a test's name to its margin and predicted
    sign. Returns the family and the mismatches."""
    out: dict[str, Any] = {}
    bad: list[str] = []
    for name in sorted(set(recomputed) | set(stored)):
        rec, ref = recomputed.get(name), stored.get(name)
        if rec is None or ref is None:
            if (rec is None) != (ref is None):
                bad.append(f"{name}: in the {'report' if ref is not None else 'recomputation'} only")
            continue
        diff = [f for f in SIGNATURE if not _same(rec.get(f), ref.get(f))]
        if diff:
            bad.append(f"{name}: report and recomputation differ in {diff}")
            continue
        d, s = float(sesoi(name)), int(sign(name))
        cands = [c for c in calls if c["n"] == rec["n_pairs"] and c["res"].estimate == rec["estimate"] and c["res"].p_value() == rec["p"]
                 and _same(c["res"].ci(level), rec["ci95"]) and _same(c["res"].ci(tost_level), rec["ci90"])]
        tails = {_tails(c["res"].samples, d, s) for c in cands}
        if len(tails) != 1:
            bad.append(f"{name}: {len(cands)} recorded calls match, with {len(tails)} different tail probabilities")
            continue
        p_m, p_e, p_w = tails.pop()
        out[name] = {"estimate": ref["estimate"], "ci95": ref["ci95"], "reading": PH.reading(ref), "sesoi": d, "predicted_sign": s,
                     "seed": cands[0]["seed"], "n_resamples": cands[0]["n_resamples"], "p_margin": p_m, "p_tost": p_e, "p_reversed": p_w}
    names = list(out)
    if names:
        adj = {key: stats.holm([out[n][key] for n in names]) for key in ("p_margin", "p_tost", "p_reversed")}
        for i, name in enumerate(names):
            out[name].update(p_margin_holm=float(adj["p_margin"][i]), p_tost_holm=float(adj["p_tost"][i]),
                             p_reversed_holm=float(adj["p_reversed"][i]), practically_better_adjusted=bool(adj["p_margin"][i] < alpha_m),
                             equivalent_adjusted=bool(adj["p_tost"][i] < alpha_e), worse_adjusted=bool(adj["p_reversed"][i] < alpha_m))
    return out, bad


def replication_adjusted(fam: dict[str, Any], task: str) -> dict[str, Any]:
    """The ranking-versus-sets test's rule with the adjusted readings: D1 practically better after margin-Holm, and D2 equivalent after TOST-Holm or worse
    after reversed-sign Holm."""
    d1, d2 = fam.get(f"D1 {task}"), fam.get(f"D2 {task}")
    if not d1 or not d2:
        return {"replicates": False, "reason": "not testable"}
    ok1, ok2 = d1["practically_better_adjusted"], d2["equivalent_adjusted"] or d2["worse_adjusted"]
    return {"replicates": ok1 and ok2, "reason": "; ".join(x for x in (
        "" if ok1 else "D1 not practically better after margin-Holm",
        "" if ok2 else "D2 neither equivalent after TOST-Holm nor worse after reversed-sign Holm") if x) or "D1 and D2 as predicted"}


def _alphas(cfg_a: dict[str, Any]) -> dict[str, float]:
    mt = cfg_a["margin_tests"]
    return {"alpha_m": float(mt["alpha_superiority"]), "alpha_e": float(mt["alpha_equivalence"])}


def _levels(cfg: dict[str, Any]) -> dict[str, float]:
    b = cfg["analysis"]["bootstrap"]
    return {"level": b["level"], "tost_level": b["tost_level"]}


def stage2_families(run: dict[str, Any], cfg_a: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The primary family and, in the new-task runs, the ranking-versus-sets test's families and the norm-budget family, with the runs' own seeds."""
    rows, cfg, rep = run["rows"], run["cfg"], run["report"]
    pk, sesoi = int(rep["primary_k"]), cfg["analysis"]["sesoi"]
    divergence = {int(k): v for k, v in rep["divergence"].items()}
    with recorded_bootstraps() as calls:
        primary = S2.analyze(rows, cfg, pk=pk, divergence=divergence, seed=cfg["seed"])["primary"]
    keep = ("P2.1", "P2.2")
    fam, bad = margin_family({k: primary[k] for k in keep}, {k: rep["analysis"]["primary"][k] for k in keep}, calls,
                             sesoi=lambda n: sesoi["M1" if n == "P2.1" else "M2"], sign=lambda n: 1, **_levels(cfg), **_alphas(cfg_a))
    out = {"primary": fam}
    bad = [f"primary {x}" for x in bad]
    if "ranking_vs_sets" in rep:
        with recorded_bootstraps() as calls:
            rvs = S2.ranking_vs_sets_analysis(rows, cfg, pk=pk, seed=cfg["seed"], choices=rep["toolkit_choices"])
            budget = S2.budget_analysis(rows, cfg, pk=pk, seed=cfg["seed"], carried=None)["family"]

        def m2(n):
            return sesoi["M2"]

        specs = {"dissociation": (rvs["dissociation"], rep["ranking_vs_sets"]["dissociation"], lambda n: sesoi["M1" if n.startswith("D1") else "M2"], 1),
                 "budget_dependence": (rvs["budget_dependence"], rep["ranking_vs_sets"]["budget_dependence"], m2, -1),
                 "toolkit k": (rvs["toolkit"]["k"], rep["ranking_vs_sets"]["toolkit"]["k"], m2, 1),
                 "toolkit budget": (rvs["toolkit"]["budget"], rep["ranking_vs_sets"]["toolkit"]["budget"], m2, 1),
                 "budget family": (budget, rep["norm_budget"]["family"], m2, 1)}
        for key, (rec, ref, se, sg) in specs.items():
            fam, b = margin_family(rec, ref, calls, sesoi=se, sign=lambda n, sg=sg: sg, **_levels(cfg), **_alphas(cfg_a))
            out[key] = fam
            bad += [f"{key} {x}" for x in b]
        out["replication_adjusted"] = {t: replication_adjusted(out["dissociation"], t)
                                       for t in sorted({n.split(" ", 1)[1] for n in out["dissociation"]})}
    return out, bad


def extension_families(entry: dict[str, Any], cfg_a: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """The J-variant controls or joint-selection family of one test run, with its own config, its parent's frozen config and seeds."""
    run_dir = Path(entry["run"])
    data = R.read_run(run_dir)
    rep, cfg = data["report"], data["protocol"]["config"]
    if rep.get("phase") != "test":
        raise RunContractError(f"{run_dir} is a {rep.get('phase')} run; the decomposition reads test runs only")
    parent = cfg["parents"][rep["model"]]
    frozen = json.loads(Path(parent["frozen"]).read_text(encoding="utf-8"))
    verify_sealed_protocol(frozen)
    pcfg = frozen.get("config_resolved") or S2._resolve(frozen["config"])
    if entry["kind"] == "jvariants":
        from saeselect.stages import jacobian_controls as J
        contrasts = [tuple(c) for c in cfg["contrasts"]] if cfg.get("contrasts") else None
        with recorded_bootstraps() as calls:
            rec = J.analyze(data["rows"], pcfg, seed=int(cfg["seed"]), holm_alpha=float(cfg["holm_alpha"]), contrasts=contrasts,
                            names=rep["variants"])["family"]
        metric = "M1"
    elif entry["kind"] == "interact":
        from saeselect.stages import joint_selection as I
        pk = int(frozen["primary_k"])
        with recorded_bootstraps() as calls:
            rec = I.analyze(data["rows"], pcfg, cfg, pk=pk, report_k=sorted({int(k) for k in cfg["report_k"]} | {pk}),
                            seed=int(cfg["seed"]))["family"]
        metric = "M2"
    else:
        raise RunContractError(f"unknown extension kind {entry['kind']!r}")
    fam, bad = margin_family(rec, rep["analysis"]["family"], calls, sesoi=lambda n: pcfg["analysis"]["sesoi"][metric], sign=lambda n: 1,
                             **_levels(pcfg), **_alphas(cfg_a))
    prov = {"run": entry["run"], "run_files_sha256": data["hashes"], "frozen_protocol_hash": data["protocol"].get("frozen_protocol_hash"),
            "parent_frozen": parent["frozen"], "parent_frozen_protocol_hash": frozen.get("protocol_hash")}
    return fam, bad, prov


# --------------------------------------------------------------------------
# 4. ATP influence audit
# --------------------------------------------------------------------------


def known_row(rows: list[dict[str, Any]], known: dict[str, Any]) -> dict[str, Any]:
    """The declared discrepancy's row: exactly one match, with the declared stored value (given to four decimals)."""
    hits = [r for r in rows if r["trial"] == known["trial"] and _cell(r) == known["cell"]]
    if len(hits) != 1:
        raise EngineeringFailed("audit", f"the known discrepancy {known['trial']} at {known['cell']} matches {len(hits)} rows", [])
    stored = hits[0]["metrics"]["ATP@in"]["M1"]
    if round(stored, 4) != round(float(known["stored_m1"]), 4):
        raise EngineeringFailed("audit", f"its stored ATP@in M1 is {stored}, not the declared {known['stored_m1']}", [])
    return hits[0]


def influence(run: dict[str, Any], pairs: set[str], seed: int) -> dict[str, Any]:
    """The run's ATP contrasts as reported and without the given concept pairs (the runs' own seeds); and ATP - JL-OUT's
    decomposition at the primary k without them. A contrast that cannot be computed after the removal is None."""
    rows, cfg, rep = run["rows"], run["cfg"], run["report"]
    pk = int(rep["primary_k"])
    kept = [r for r in rows if r["pair"] not in pairs]
    out: dict[str, Any] = {"pairs_left_out": sorted(pairs)}
    for a, b in (("JL-OUT", "ATP"), ("LL-OUT", "ATP"), ("ATP@in", "ATP")):
        for metric in ("M1", "M2"):
            name = f"{a} - {b} ({metric})"
            out[name] = {"reported": PH._brief(rep["analysis"]["secondary"].get(name)),
                         "without": PH._brief(S2.contrast(kept, a, b, metric, k=pk if metric == "M2" else None, cfg=cfg, seed=cfg["seed"] + 11))}
    if "ranking_vs_sets" in rep:
        for t in sorted({r["relation"] for r in rows}):
            for d, metric in (("D1", "M1"), ("D2", "M2")):
                out[f"{d} {t}"] = {"reported": PH._brief(rep["ranking_vs_sets"]["dissociation"].get(f"{d} {t}")),
                                   "without": PH._brief(S2.contrast([r for r in kept if r["relation"] == t], "ATP", "JL-OUT", metric,
                                                                     k=pk if metric == "M2" else None, cfg=cfg, seed=S2._hseed(cfg["seed"], d, t)))}
    out["ATP - JL-OUT decomposition"] = decomposition(kept, cfg, [("ATP", "JL-OUT")], pk, seed).get("pooled", {}).get("ATP - JL-OUT")
    return out


def atp_screen(run: dict[str, Any], cutoff: float, known_pairs: set[str], seed: int) -> dict[str, Any]:
    """Stored ATP@in M1 below the cutoff (an inspection screen, not a classifier of numerical error), nonfinite values listed
    separately, and the influence of the flagged pairs (a sensitivity analysis: the screen selects on the outcome)."""
    vals = [(r, (r["metrics"].get("ATP@in") or {}).get("M1")) for r in run["rows"]]
    flagged = [r for r, v in vals if v is not None and np.isfinite(v) and v < cutoff]
    out: dict[str, Any] = {
        "screen": {"n_trials": len(run["rows"]), "n_flagged": len(flagged), "by_cell": dict(Counter(_cell(r) for r in flagged)),
                   "trials": [{"trial": r["trial"], "cell": _cell(r), "ATP@in M1": r["metrics"]["ATP@in"]["M1"]} for r in flagged],
                   "nonfinite_or_missing": [{"trial": r["trial"], "cell": _cell(r), "ATP@in M1": v} for r, v in vals
                                            if v is None or not np.isfinite(v)]},
        "influence": {}}
    for tag, pairs in (("known", known_pairs), ("screen", {r["pair"] for r in flagged})):
        if pairs:
            out["influence"][tag] = influence(run, pairs, S2._hseed(seed, tag))
    return out


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def run(ctx: RunContext) -> Outcome:
    cfg_a = ctx.config
    seed = int(cfg_a["seed"])
    tl, pl, audit = cfg_a["tail"], cfg_a["pooling"], cfg_a["atp_audit"]
    contrasts = {k: tuple(v) for k, v in pl["contrasts"].items()}
    report: dict[str, Any] = {"runs": {}, "extensions": {}, "atp_audit": {"known": None, "runs": {}}, "provenance": {}}
    bad: list[str] = []
    cache: dict = {}
    for name, entry in cfg_a["stage2"].items():
        run_ = PH.load(entry, cache, ctx, keep_effects=True)  # the loader's cache key ignores keep_effects: load each run once, here
        rows, cfg, rep = run_["rows"], run_["cfg"], run_["report"]
        pk, ks, s = int(rep["primary_k"]), sorted(int(k) for k in cfg["k_grid"]), S2._hseed(seed, name)
        viol = integrity_violations(rows, cfg, tl["signals"], ks, float(tl["k1_tolerance_logodds"]))
        if viol:
            raise EngineeringFailed("integrity", f"{name}: {len(viol)} sets with a missing effect or a k = 1 identity off by more than "
                                    f"{tl['k1_tolerance_logodds']} log-odds", viol[:20])
        margins, b = stage2_families(run_, cfg_a)
        bad += [f"{name} {x}" for x in b]
        known_pairs: set[str] = set()
        if audit["known"]["model"] == name:
            hit = known_row(rows, audit["known"])
            known_pairs = {hit["pair"]}
            report["atp_audit"]["known"] = {**audit["known"], "pair": hit["pair"], "stored_m1_read": hit["metrics"]["ATP@in"]["M1"]}
        report["atp_audit"]["runs"][name] = PH._jsonable(atp_screen(run_, float(audit["screen_below"]), known_pairs, S2._hseed(s, "audit")))
        report["runs"][name] = PH._jsonable({
            "primary_k": pk, "tasks": sorted({r["relation"] for r in rows}), "cells": sorted({_cell(r) for r in rows}),
            "tail": tail(rows, cfg, tl["signals"], ks, pk, S2._hseed(s, "tail")),
            "decomposition": decomposition(rows, cfg, [tuple(x) for x in tl["pairs"]], pk, S2._hseed(s, "dec")),
            "pooling": {"per_scope": per_scope_contrasts(rows, cfg, contrasts, pk, S2._hseed(s, "scope")),
                        "common_mask": common_mask(rows, cfg, contrasts, pk, S2._hseed(s, "mask")),
                        "denominators": denominators(rows, cfg, contrasts, pl["denominators"], pk, S2._hseed(s, "den")),
                        "references": ADD.ceilings(rows, cfg, [], [], pk, S2._hseed(s, "ref")),
                        "overshoot": overshoot(rows, cfg, pl["overshoot_signals"], float(pl["overshoot_share_above"]))},
            "margin_tests": margins})
        report["provenance"][name] = {"run": entry["run"], "frozen": entry["frozen"], **run_["provenance"]}
        del run_, rows
        cache.clear()
    if report["atp_audit"]["known"] is None:
        raise EngineeringFailed("audit", f"the known discrepancy's run {audit['known']['model']} is not among the runs read", [])
    for name, entry in (cfg_a.get("extensions") or {}).items():
        fam, b, prov = extension_families(entry, cfg_a)
        bad += [f"{name} {x}" for x in b]
        report["extensions"][name] = PH._jsonable({"kind": entry["kind"], "margin_tests": fam})
        report["provenance"][name] = prov
    if bad:
        raise GateFailed("reproduction", f"{len(bad)} recomputed family values differ from the reports or match no recorded resamples",
                         bad[:10], report={"mismatches": bad})
    return Outcome(status="PASS", summary=_summary(report), report=report, send_back=[])


# --------------------------------------------------------------------------
# SUMMARY.md
# --------------------------------------------------------------------------


def _ci(c: dict[str, Any] | None, d: int = 3) -> str:
    if not c:
        return "n/a"
    if not c.get("ci95"):
        return f"{c['estimate']:+.{d}f}"
    return f"{c['estimate']:+.{d}f} [{c['ci95'][0]:+.{d}f}, {c['ci95'][1]:+.{d}f}]"


MARGIN_HEADER = ["| Test | Estimate [95%] | Reported reading | Margin p (Holm): practically better | TOST p (Holm): equivalent | "
                 "Reversed p (Holm): worse |", "|---|---|---|---|---|---|"]


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _margin_rows(fam: dict[str, Any]) -> list[str]:
    return [f"| {n} | {_ci(v)} | {v['reading']} | {v['p_margin_holm']:.4f}: {_yes(v['practically_better_adjusted'])} | "
            f"{v['p_tost_holm']:.4f}: {_yes(v['equivalent_adjusted'])} | {v['p_reversed_holm']:.4f}: {_yes(v['worse_adjusted'])} |"
            for n, v in fam.items()]


def _summary(report: dict[str, Any]) -> str:
    lines = ["## What ran", "", "The decomposition: post hoc diagnostics of the stored test trials. Pair means; cluster bootstrap within "
             "relation; 95% intervals; trials below the M2 floor left out wherever a recovered share enters; nothing squared. Checked: "
             "every active selected feature has a stored effect, every set at k = 1 has F equal to its feature's isolated share, and "
             "every recomputed family equals its report. Margin tests are approximate bootstrap tests by interval inversion. Nothing "
             "here changes a verdict of record.", ""]
    for name, m in report["runs"].items():
        pk = m["primary_k"]
        lines += [f"## {name} (primary k {pk}; tasks {', '.join(m['tasks'])})", "",
                  f"### 1. Decomposition at k = {pk}: dF = dU (singleton utility) + dN (non-additive remainder)", "",
                  "| Scope | Pair | dF | dU | dN | d active | Reading |", "|---|---|---|---|---|---|---|"]
        for scope, per in m["decomposition"].items():
            for pair, e in per.items():
                lines.append(f"| {scope} | {pair} | {_ci(e['F'])} | {_ci(e['U'])} | {_ci(e['N'])} | {_ci(e['a'], 2)} | {e['reading']} |")
        po = m["pooling"]
        lines += ["", "### 2. Pooling and normalization", "", "Per scope (unadjusted): " + "; ".join(
            f"{scope}: " + ", ".join(f"{n} {v['reading'] if v else 'n/a'}" for n, v in per.items()) for scope, per in po["per_scope"].items()), "",
                  "M1 contrasts on the M2 mask: " + "; ".join(f"{n} {_ci(v)} {v['reading'] if v else ''}" for n, v in po["common_mask"].items()), "",
                  "Other denominators (same mask: normalized | FULL-normalized | raw margin gain; trials): " + "; ".join(
                      f"{den}: " + ", ".join(f"{n} {_ci(v['normalized'])} {v['normalized']['reading'] if v['normalized'] else ''} | "
                                             f"{_ci(v['FULL_on_this_mask'])} | {_ci(v['raw_margin_gain'], 2)}; {v['n_trials']}" for n, v in per.items())
                      for den, per in po["denominators"].items()), ""]
        if po["overshoot"]:
            lines += ["Norm-budget overshoot, pooled (walks reaching B; rho_intended / B median, 90th percentile, share above the declared "
                      "ratio):", ""]
            for sig, per in po["overshoot"]["pooled"].items():
                lines.append(f"- {sig}: " + "; ".join(f"B = {b}: {v['n_reached']} of {v['n_trials']}; {v['median']:.2f}, {v['q90']:.2f}, "
                                                       f"{v['share_above']:.2f}" for b, v in per.items() if v.get("median") is not None))
            lines.append("")
        lines += ["### 3. Margin-specific tests (Holm within each family)", "", *MARGIN_HEADER]
        for fam_name, fam in m["margin_tests"].items():
            if fam_name != "replication_adjusted":
                lines += [f"| **{fam_name}** | | | | | |", *_margin_rows(fam)]
        if "replication_adjusted" in m["margin_tests"]:
            lines += ["", "The ranking-versus-sets test's replication with the adjusted readings: " + "; ".join(
                f"{t}: {'replicates' if v['replicates'] else 'does not replicate'} ({v['reason']})"
                for t, v in m["margin_tests"]["replication_adjusted"].items())]
        au = report["atp_audit"]["runs"][name]
        lines += ["", f"### 4. ATP audit: {au['screen']['n_flagged']} of {au['screen']['n_trials']} trials with stored ATP@in M1 below the "
                  f"cutoff {json.dumps(au['screen']['by_cell'])}; nonfinite or missing: {len(au['screen']['nonfinite_or_missing'])}", ""]
        for tag, inf in au["influence"].items():
            lines.append(f"- Without the {tag} pairs ({', '.join(inf['pairs_left_out'])}): " + "; ".join(
                f"{n} {_ci(v['reported'])} -> {_ci(v['without']) if v['without'] else 'insufficient data'}"
                for n, v in inf.items() if isinstance(v, dict) and "without" in v))
        lines.append("")
    for name, e in report["extensions"].items():
        lines += [f"## {name} ({e['kind']}): margin-specific tests", "", *MARGIN_HEADER, *_margin_rows(e["margin_tests"]), ""]
    k = report["atp_audit"]["known"]
    lines += [f"Known discrepancy: {k['model']} {k['trial']} at {k['cell']}, stored ATP@in M1 {k['stored_m1_read']:.4f} against "
              f"{k['recomputed_m1']} recomputed ({k['source']}).", "", "## What this permits", "",
              "Descriptive and post hoc context for the verdicts of record, none of which changes. The ATP "
              "audit measures sensitivity to flagged records, not numerical correctness.", "",
              "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
