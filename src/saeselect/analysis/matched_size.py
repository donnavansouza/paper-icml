"""``python -m saeselect matched-size``: the matched-size analysis, post hoc. Attribution patching against the J-lens at matched edit size on the
new tasks. CPU, from the new-task endpoint test runs' stored trials.

    python -m saeselect matched-size --config configs/analysis/matched_size.yaml --out analysis/matched-size

Per run:
1. The post hoc analysis's size analysis (``reanalysis.size_analysis``): edit size rho (realized norm of the set's delta over the full
   patch's) and F at the primary k; F differences at matched rho* along the k path (more features) and the alpha path
   (the primary-k set scaled), on trials where both paths reach rho*.
2. Along the k path, the split of F at matched rho*: the summed single-feature effects U (the decomposition's U at each k, linearly
   interpolated at the same crossing as F) and the remainder N = F - U, with one set of resamples for dF, dU and dN.
Pair means, cluster bootstrap within relation, 95% intervals, unsquared, no verdict. Before any analysis, the pooled F of
every signal at the primary k must equal the run's stored report (reproduction guard).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saeselect.analysis import posthoc as PH
from saeselect.analysis import decomposition as DEC
from saeselect.provenance import GateFailed
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2


def k_path(r: dict[str, Any], name: str, ks: list[int], q: str) -> list[tuple[float, float]] | None:
    """(0, 0), then (rho_k, q_k) over the k grid for q in F, U; None if a point is undefined."""
    pts = [(0.0, 0.0)]
    for k in ks:
        x, v = PH.rho(r, name, k), DEC.tail_values(r, name, k)
        if x is None or v is None:
            return None
        pts.append((x, v[q]))
    return pts


def split_at(r: dict[str, Any], name: str, ks: list[int], target: float) -> dict[str, float] | None:
    """F, U and N = F - U of one signal at edit size ``target`` along its k path; None if the path never reaches it."""
    pf, pu = k_path(r, name, ks, "F"), k_path(r, name, ks, "U")
    if pf is None or pu is None:
        return None
    f, u = PH.at_rho(pf, target), PH.at_rho(pu, target)
    return None if f is None or u is None else {"F": f, "U": u, "N": f - u}


QS = ("F", "U", "N")


def _block(rows: list[dict[str, Any]], cfg: dict[str, Any], diff: dict[int, dict[str, dict[str, float]]], s: int) -> dict[str, Any]:
    """Pair-mean estimates of the matched, the fixed-k (same trials) and the matched-minus-fixed contrasts; one seed for all,
    so every interval comes from the same resamples."""
    def est(kind: str, q: str):
        return PH.mean_ci(rows, lambda r: diff[id(r)][kind][q] if id(r) in diff else None, cfg, s)
    out: dict[str, Any] = {q: est("matched", q) for q in QS}
    out["fixed_same"] = {q: est("fixed", q) for q in QS}
    out["matched_minus_fixed"] = {q: est("delta", q) for q in QS}
    out["trials"] = len(diff)
    return out


def matched_split(rows: list[dict[str, Any]], cfg: dict[str, Any], pairs: list[tuple[str, str]], targets: list[float],
                  seed: int, pk: int) -> dict[str, Any]:
    """a - b at each rho* along the k path, split into dF = dU + dN, on trials at or above the M2 floor where both paths
    reach rho*; on the same trials, the fixed-k contrast at the primary k ``pk`` and the paired difference matched minus
    fixed (the matched-size analysis's extension). Pooled and per task."""
    ks = sorted(int(k) for k in cfg["k_grid"])
    out: dict[str, Any] = {}
    for a, b in pairs:
        per: dict[str, Any] = {}
        for t in targets:
            diff: dict[int, dict[str, dict[str, float]]] = {}  # per trial (by identity): matched, fixed and their difference
            for r in rows:
                if PH._above_floor(r, cfg):
                    sa, sb = split_at(r, a, ks, t), split_at(r, b, ks, t)
                    fa, fb = DEC.tail_values(r, a, pk), DEC.tail_values(r, b, pk)
                    if sa is not None and sb is not None and fa is not None and fb is not None:
                        m = {q: sa[q] - sb[q] for q in QS}
                        f = {q: fa[q] - fb[q] for q in QS}
                        diff[id(r)] = {"matched": m, "fixed": f, "delta": {q: m[q] - f[q] for q in QS}}
            s = S2._hseed(seed, a, b, t)
            per[str(t)] = _block(rows, cfg, diff, s)
            per[str(t)]["trials_above_floor"] = sum(1 for r in rows if PH._above_floor(r, cfg))
            per[str(t)]["by_task"] = {task: _block([r for r in rows if r["relation"] == task], cfg, diff, S2._hseed(s, task))
                                      for task in sorted({r["relation"] for r in rows})}
        out[f"{a} - {b}"] = per
    return out


def reproduction(rows: list[dict[str, Any]], cfg: dict[str, Any], report: dict[str, Any], signals: list[str]) -> list[str]:
    """Pooled F at the primary k (pair means) per signal against the stored report; the mismatches."""
    pk, stored = str(report["primary_k"]), report["summary"]["pooled"]
    bad = []
    for s in signals:
        vals, _, _ = S2._cluster_means(rows, lambda r, s=s: r["sets"][s][pk]["F"] if PH._above_floor(r, cfg) else None)
        got, want = float(np.mean(vals)), stored[s]["F"][pk]
        if abs(got - want) > 1e-9:
            bad.append(f"{s}: F at k = {pk} recomputed {got!r}, report {want!r}")
    return bad


def run(ctx: RunContext) -> Outcome:
    cfg_a = ctx.config
    seed, spec = int(cfg_a["seed"]), cfg_a["size"]
    pairs = [tuple(p) for p in spec["pairs"]]
    report: dict[str, Any] = {"runs": {}, "provenance": {}}
    cache: dict = {}
    for name, entry in cfg_a["runs"].items():
        run_ = PH.load(entry, cache, ctx, keep_effects=True)
        rows, cfg, rep = run_["rows"], run_["cfg"], run_["report"]
        bad = reproduction(rows, cfg, rep, spec["signals"])
        if bad:
            raise GateFailed("reproduction", f"{name}: {len(bad)} pooled F values differ from the stored report", bad[:20],
                             report={"mismatches": bad})
        s = S2._hseed(seed, name)
        report["runs"][name] = PH._jsonable({"primary_k": int(rep["primary_k"]), "size": PH.size_analysis(run_, spec, s),
                                              "matched_split": matched_split(rows, cfg, pairs, spec["rho"], S2._hseed(s, "split"),
                                                                               int(rep["primary_k"]))})
        report["provenance"][name] = {"run": entry["run"], "frozen": entry["frozen"], **run_["provenance"]}
        cache.clear()
    return Outcome(status="PASS", summary=_summary(report), report=report, send_back=[])


def _ci(c: dict[str, Any] | None) -> str:
    if not c:
        return "n/a"
    lo, hi = c["ci95"] if c.get("ci95") else (float("nan"), float("nan"))
    return f"{c['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _summary(report: dict[str, Any]) -> str:
    lines = ["# attribution patching against the J-lens at matched edit size (post hoc)", "",
             "rho = realized norm of the set's delta over the full patch's. Pair means, 95% cluster bootstrap within relation, "
             "unsquared, no verdict. Trials at or above the M2 floor where both paths reach rho*.", ""]
    for name, m in report["runs"].items():
        sz = m["size"]
        lines += [f"## {name} (primary k {m['primary_k']})", "", "| Signal | rho at k | F at k |", "|---|---|---|"]
        lines += [f"| {s} | {_ci(v['rho'])} | {_ci(v['F'])} |" for s, v in sz["at_primary_k"].items()]
        lines += ["", "| Path | Pair | rho* | dF [95%] | Trials not reached / above floor |", "|---|---|---|---|---|"]
        for along, pairs in sz["matched"].items():
            for pair, per in pairs.items():
                for t, v in per.items():
                    lines.append(f"| {along} | {pair} | {t} | {_ci(v['difference'])} | {v['trials_not_reached']} / {v['trials_above_floor']} |")
        lines += ["", "| k path split | rho* | dF | dU | dN | dF, fixed k, same trials | dN, fixed k, same trials | dF matched - fixed | Trials |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for pair, per in m["matched_split"].items():
            for t, v in per.items():
                fs, mf = v["fixed_same"], v["matched_minus_fixed"]
                lines.append(f"| {pair} | {t} | {_ci(v['F'])} | {_ci(v['U'])} | {_ci(v['N'])} | {_ci(fs['F'])} | {_ci(fs['N'])} | "
                             f"{_ci(mf['F'])} | {v['trials']} / {v['trials_above_floor']} |")
        lines.append("")
    return "\n".join(lines)
