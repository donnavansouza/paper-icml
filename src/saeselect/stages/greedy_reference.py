"""``python -m saeselect greedy-reference``: a greedy exact set reference (extension X9).

    python -m saeselect greedy-reference --config configs/extensions/greedy_reference.yaml --model configs/models/<model>.yaml --phase dev --out runs/greedy-reference-dev-<m>-01 --device cuda
    python -m saeselect greedy-reference --config configs/extensions/greedy_reference.yaml --phase test --frozen runs/greedy-reference-dev-<m>-01/frozen_protocol.json \
        --out runs/greedy-reference-test-<m>-01 --device cuda

From the parent Stage 2 run (its dev run in the dev phase, its test run in the test phase), at the declared cell: a seeded
subset of pairs (stratified by relation), one seeded template per pair and both directions. For each trial the candidates
C are the union of the stored top-k_c sets of every support selector and ORACLE, restricted to features active on the
evaluation prompt. The greedy reference adds, one feature at a time up to k_max, the candidate whose addition gives the
largest exact FP32 delta-form margin on the evaluation prompt (the parent's patching mode and strategy; the error term is
kept, rule 6), recording realized norms. F_greedy(k) = (m(S_k) - m_clean) / (m_FULL - m_clean), unsquared, with the parent's
stored m_clean and m_FULL. Parity: the parent's stored JL-OUT set at the largest reported k is re-evaluated with the same
evaluator and must give its stored margin (the parent's parity tolerance). Descriptive: greedy uses the evaluation prompt,
as ORACLE does, so it is a reference, never a selector, and a lower bound on the best k-set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from saeselect import artifacts as A
from saeselect import signals as S
from saeselect import stats
from saeselect.analysis import rebuild as R
from saeselect.stats import sub_seed as _seed
from saeselect.hooks import EditSpec
from saeselect.interventions import candidate_pool
from saeselect.models import load_model
from saeselect.patching import PatchEvaluator, as_ids
from saeselect.provenance import EngineeringFailed, RunContractError, json_hash, seal_protocol, utc_now, verify_sealed_protocol
from saeselect.results import write_results_zip
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2
from saeselect.stages.jacobian_controls import _parent
from saeselect.tasks.records import hash_unit


def choose_rows(rows: list[dict[str, Any]], *, cell: str, n_pairs: int, seed: int) -> list[dict[str, Any]]:
    """The declared subset: pairs allotted to relations in proportion to their counts (largest remainders), drawn in seeded
    order within each relation; one seeded template per pair; both directions of that template."""
    at = [r for r in rows if f"{r['layer']}/{r['site']}" == cell]
    pairs_by_rel: dict[str, list[str]] = defaultdict(list)
    for p, rel in sorted({(r["pair"], r["relation"]) for r in at}):
        pairs_by_rel[rel].append(p)
    total = sum(len(v) for v in pairs_by_rel.values())
    n = min(n_pairs, total)
    quota = {rel: n * len(v) / total for rel, v in pairs_by_rel.items()}
    take = {rel: int(q) for rel, q in quota.items()}
    for rel in sorted(quota, key=lambda x: (-(quota[x] - take[x]), x))[: n - sum(take.values())]:
        take[rel] += 1
    chosen = set()
    for rel, ps in pairs_by_rel.items():
        chosen.update(sorted(ps, key=lambda p: (hash_unit(f"a29:{p}", seed), p))[: take[rel]])
    out = []
    for p in sorted(chosen):
        tpls = sorted({r["template"] for r in at if r["pair"] == p})
        tpl = min(tpls, key=lambda t: (hash_unit(f"a29:{p}:{t}", seed), t))
        out += [r for r in at if r["pair"] == p and r["template"] == tpl]
    return sorted(out, key=lambda r: r["trial"])


def candidates(r: dict[str, Any], names: list[str], k_c: int, pool: set[int]) -> list[int]:
    return sorted({f for n in names for f in r["sets"][n][str(k_c)]["selected"]} & pool)


def greedy(evaluate, cand: list[int], k_max: int) -> list[dict[str, Any]]:
    """Forward selection by the exact margin. ``evaluate(list of feature lists)`` returns PatchResults in order."""
    chosen: list[int] = []
    path = []
    for _ in range(min(k_max, len(cand))):
        rest = [f for f in cand if f not in chosen]
        res = evaluate([chosen + [f] for f in rest])
        best = max(range(len(rest)), key=lambda i: (res[i].margin, -rest[i]))  # ties by the smaller feature id
        chosen.append(rest[best])
        path.append({"k": len(chosen), "feature": rest[best], "margin": res[best].margin,
                     "norms": [res[best].intended_norm, res[best].realized_norm], "n_candidates": len(rest)})
    return path


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    if ctx.phase not in ("dev", "test"):
        raise RunContractError("greedy needs --phase dev or --phase test")
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
    strategy = {(True, True): "batched_suffix", (True, False): "batched", (False, True): "suffix", (False, False): "serial"}[
        (bool(s0["batching_allowed"]), bool(s0["suffix_caching_allowed"]))]
    batch_size = int(s0.get("batch_size", 8))
    cell = entry["cell"]
    if cell not in parent_frozen["eligible_cells"]:
        raise RunContractError(f"{cell} is not an eligible cell of the parent protocol {parent_frozen['eligible_cells']}")
    layer, site = int(cell.split("/")[0]), cell.split("/")[1]
    k_c, k_max, report_k = int(cfg["candidates_k"]), int(cfg["k_max"]), [int(k) for k in cfg["report_k"]]
    names = list(cfg["candidate_selectors"])
    seed = int(cfg["seed"])
    rows_in = choose_rows(data["rows"], cell=cell, n_pairs=int(cfg["n_pairs"]) if ctx.limit is None else ctx.limit, seed=seed)

    trials: dict[str, S2.Trial] = {}
    data_sha = {}
    for family in parent_frozen["relations"]:
        card, records = S2._load(Path(pcfg["data_dir"]), family, ctx.phase)
        data_sha[family] = card["records_sha256"]
        for t in S2.build_trials(records, family):
            trials[t.key] = t
    handle = load_model(model_cfg, device=ctx.device)
    cache = S2.PromptCache(handle, [layer], causal)
    u = s0["sae"][str(layer)]
    sae = A.load_sae_for(model_cfg, layer=layer, width=u["width"], l0=u["l0"], d_model=handle.d_model, suffix=u.get("suffix", ""),
                         device=ctx.device, convention=u.get("convention"), repo_id=u.get("repo_id"), revision=u.get("revision"))
    tol = float(pcfg["parity_probe"]["max_abs_margin_diff"])
    out_rows, parity_max = [], 0.0
    for r in rows_in:
        t = trials[r["trial"]]
        src, don = cache.get(t.source), cache.get(t.donor)
        positions = src["sites"][site]
        h_src, h_don = src["acts"][(layer, site)].to(sae.device), don["acts"][(layer, site)].to(sae.device)
        da_full = sae.encode(h_don) - sae.encode(h_src)
        pool = candidate_pool(da_full)
        col = {f: j for j, f in enumerate(pool.tolist())}
        da, W = da_full.index_select(1, pool), sae.W_dec.index_select(0, pool)
        ev = PatchEvaluator(handle, as_ids(src["ids"], handle.device), t.answers, readout=t.source.meta["readout_position"], mode=causal,
                            strategy=strategy, batch_size=batch_size)

        def evaluate(sets: list[list[int]]):
            specs = []
            for feats in sets:
                c = torch.as_tensor([col[f] for f in feats if f in col], device=da.device)
                specs.append([EditSpec(layer=layer, positions=positions, delta=da.index_select(1, c) @ W.index_select(0, c))])
            return ev.evaluate(specs)

        probe_set = [f for f in r["sets"]["JL-OUT"][str(max(report_k))]["active"] if f in col]
        probe = evaluate([probe_set])[0].margin if probe_set else r["m"]["clean"]
        parity = abs(probe - r["sets"]["JL-OUT"][str(max(report_k))]["margin"])
        parity_max = max(parity_max, parity)
        if parity > tol:
            raise EngineeringFailed("parity", f"{r['trial']}: the stored JL-OUT set re-evaluated gives {probe:.6g}, stored "
                                              f"{r['sets']['JL-OUT'][str(max(report_k))]['margin']:.6g} (tolerance {tol})")
        cand = candidates(r, names, k_c, set(col))
        path = greedy(evaluate, cand, k_max)
        m_clean, m_full = r["m"]["clean"], r["m"]["FULL"]
        for step in path:
            step["F"] = S.recovery(step["margin"], m_clean, m_full) if m_full != m_clean else None
        out_rows.append({"trial": r["trial"], "relation": r["relation"], "pair": r["pair"], "template": r["template"], "layer": layer,
                         "site": site, "full_gain": r["full_gain"], "m": {"clean": m_clean, "FULL": m_full}, "n_candidates": len(cand),
                         "greedy": path, "selectors": {n: {str(k): r["sets"][n][str(k)]["F"] for k in report_k} for n in names},
                         "parity": parity})
    del sae
    S2._free()

    write_results_zip(ctx.run.path, {"trials.jsonl": "".join(json.dumps(x, sort_keys=True, default=S2._jsonable) + "\n" for x in out_rows)})
    analysis = analyze(out_rows, pcfg, names=names, report_k=report_k, seed=seed)
    report = json.loads(json.dumps({"phase": ctx.phase, "model": name, "parent_run": parent_dir.as_posix(), "parent_provenance": prov,
                                    "parent_frozen_protocol_hash": parent_frozen["protocol_hash"], "cell": cell, "strategy": strategy,
                                    "n_trials": len(out_rows), "n_pairs": len({x["pair"] for x in out_rows}),
                                    "parity_max": parity_max, "parity_tolerance": tol, "analysis": analysis}, default=S2._jsonable))
    summary = _summary(ctx, report, names, report_k)
    if ctx.phase == "dev":
        protocol = {"stage": "greedy", "smoke": ctx.smoke, "code_commit": ctx.git.get("commit"),
                    "source_run": ctx.run.path.as_posix(), "created_at": utc_now(), "config": cfg, "model": model_cfg,
                    "parent_protocol_hash": parent_frozen["protocol_hash"], "cell": cell, "data": data_sha}
        frozen_protocol = seal_protocol(protocol)
        ctx.run.write_json("frozen_protocol.json", frozen_protocol)
        report["frozen_protocol_hash"] = frozen_protocol["protocol_hash"]
        return Outcome(status="FROZEN_AWAITING_APPROVAL", summary=summary, report=report, send_back=["frozen_protocol.json", "results.zip"])
    if ctx.frozen.get("parent_protocol_hash") != parent_frozen["protocol_hash"] or ctx.frozen.get("cell") != cell:
        raise RunContractError("the parent protocol or the cell differs from the frozen protocol")
    return Outcome(status="PASS", summary=summary, report=report, send_back=["results.zip"])


def analyze(rows: list[dict[str, Any]], cfg: dict[str, Any], *, names: list[str], report_k: list[int], seed: int) -> dict[str, Any]:
    """Per k: the pair mean of F_greedy(k) and of each selector's F(k), each selector's gap F_greedy - F (pair means, cluster
    bootstrap within relation, 95%), and the share mean F / mean F_greedy. Trials at or above the M2 floor. Descriptive."""
    floor = cfg["m2_min_full_gain"]
    b_ = cfg["analysis"]["bootstrap"]

    def f_greedy(r, k):
        step = next((s for s in r["greedy"] if s["k"] == k), None)
        return None if step is None or r["full_gain"] < floor else step["F"]

    def mean_ci(value, s):
        vals, _, strata = S2._cluster_means(rows, value)
        if len(vals) < 2:
            return {"estimate": float(np.mean(vals)), "ci95": None, "n_pairs": len(vals)} if vals else None
        res = stats.cluster_bootstrap(vals, n_resamples=b_["n_resamples"], seed=s, strata=strata)
        return {"estimate": res.estimate, "ci95": list(res.ci(b_["level"])), "n_pairs": len(vals)}

    out: dict[str, Any] = {}
    for k in report_k:
        g = mean_ci(lambda r, k=k: f_greedy(r, k), _seed(seed, "greedy", k))
        per = {}
        for n in names:
            sel = mean_ci(lambda r, n=n, k=k: r["selectors"][n][str(k)] if r["full_gain"] >= floor else None, _seed(seed, "sel", n, k))
            gap = mean_ci(lambda r, n=n, k=k: (None if f_greedy(r, k) is None or r["selectors"][n][str(k)] is None
                                               else f_greedy(r, k) - r["selectors"][n][str(k)]), _seed(seed, "gap", n, k))
            per[n] = {"F": sel, "gap": gap, "share": (sel["estimate"] / g["estimate"]) if sel and g and g["estimate"] else None}
        out[str(k)] = {"greedy": g, "selectors": per}
    return out


def _summary(ctx: RunContext, report: dict[str, Any], names: list[str], report_k: list[int]) -> str:
    a = report["analysis"]

    def mci(c, signed=False):
        if not c:
            return "n/a"
        f = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
        return f(c["estimate"]) + (f" [{f(c['ci95'][0])}, {f(c['ci95'][1])}]" if c.get("ci95") else "")

    lines = ["## What ran", "",
             f"The greedy reference, {ctx.phase} phase, on `{report['model']}` at {report['cell']}: a greedy exact set reference on "
             f"{report['n_trials']} trials in {report['n_pairs']} pairs of the parent Stage 2 {ctx.phase} run `{report['parent_run']}`. "
             "Candidates: the union of the stored top sets of " + ", ".join(names) + "; each step adds the candidate with the largest exact "
             f"FP32 delta-form margin (patches {report['strategy']}). F = (m(S) - m_clean) / (m_FULL - m_clean), unsquared; trials at or "
             "above the M2 floor. Descriptive: greedy uses the evaluation prompt and is a lower bound on the best set.", "",
             f"Parity: the stored JL-OUT set re-evaluated, worst margin difference {report['parity_max']:.2g} (tolerance "
             f"{report['parity_tolerance']:.2g}).", ""]
    for k in report_k:
        e = a[str(k)]
        lines += [f"## k = {k}: greedy F {mci(e['greedy'])}", "", "| Selector | F | Gap F_greedy - F | Share of greedy |", "|---|---|---|---|"]
        for n in names:
            v = e["selectors"][n]
            share = "n/a" if v["share"] is None else f"{v['share']:.0%}"
            lines.append(f"| {n} | {mci(v['F'])} | {mci(v['gap'], True)} | {share} |")
        lines.append("")
    lines += ["## What this permits", "",
              "How much of a greedy exact k-set's recovery each selector reaches, at this cell: headroom against a reference that "
              "is itself a lower bound on the best set. Descriptive; no verdicts.", "", "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
