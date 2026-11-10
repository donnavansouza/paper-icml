"""``python -m saeselect joint-selection``: interaction-aware selectors.

    python -m saeselect joint-selection --config configs/extensions/joint_selection.yaml --model configs/models/<model>.yaml --phase dev --out runs/joint-selection-dev-<m>-01 --device cuda
    python -m saeselect joint-selection --config configs/extensions/joint_selection.yaml --phase test --frozen runs/joint-selection-dev-<m>-01/frozen_protocol.json \
        --out runs/joint-selection-test-<m>-01 --device cuda

From the parent Stage 2 run (its dev run in the dev phase, its test run in the test phase), at the declared shallow concept
cell, on the greedy reference's subset rule (40 pairs, one seeded template per pair, both directions). Two selectors that use only the
support prompts (the trial's other templates), as every Stage 2 selector does:

- GREEDY-SUP: forward selection, up to k_max steps, of the candidate whose addition gives the largest mean margin gain
  m_s(S) - m_s,clean over the support prompts (exact FP32 delta-form edits with the error term kept, rule 6). Candidates: the
  union of the parent's stored top-k_c sets of the declared support selectors (no ORACLE). Interactions between features are
  measured, not assumed away.
- IG: path-integrated attribution, IG_i = Sum_p delta a_i(p) * (gbar(p) . d_i), with gbar the mean of dm/dh_l(p) at the n_steps
  midpoints h_src + alpha (h_don - h_src), alpha = (j + 1/2) / n_steps, on each prompt; the support score is the mean over the
  support prompts (absent = 0), as ATP's (ATP is the same with alpha = 0 only).

Each selector's sets are evaluated on the evaluation prompt (features inactive there add nothing) and compared with the
parent's stored sets of the same trial. F = (m(S) - m_clean) / (m_FULL - m_clean), unsquared, with the parent's stored
m_clean and m_FULL. Parity: the parent's stored JL-OUT set is re-evaluated on the evaluation and every support prompt, and the
recomputed ATP@in M1 (the alpha = 0 gradient) must equal the stored one. Descriptive: IG completeness, (Sum_i IG_i + IG_err) /
(m_FULL - m_clean), and the additive prediction Sum_{i in S} e_i / (m_FULL - m_clean) of every set.
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
from saeselect.analysis import rebuild as R
from saeselect.hooks import EditSpec
from saeselect.interventions import candidate_pool
from saeselect.models import load_model
from saeselect.patching import PatchEvaluator, as_ids, margin_gradient
from saeselect.provenance import EngineeringFailed, RunContractError, json_hash, seal_protocol, utc_now
from saeselect.results import write_results_zip
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2
from saeselect.stages.greedy_reference import choose_rows
from saeselect.stages.jacobian_controls import _parent


class Prompt:
    """One source prompt at the cell: its pool, activation changes, decoder rows, FULL delta and an evaluator."""

    def __init__(self, handle, sae, cache, t: S2.Trial, layer: int, site: str, *, causal: str, strategy: str, batch_size: int):
        src, don = cache.get(t.source), cache.get(t.donor)
        self.t, self.layer, self.ids, self.positions = t, layer, src["ids"], src["sites"][site]
        h_src, h_don = src["acts"][(layer, site)].to(sae.device), don["acts"][(layer, site)].to(sae.device)
        da_full = sae.encode(h_don) - sae.encode(h_src)
        pool = candidate_pool(da_full)
        self.pool = pool.tolist()
        self.col = {f: j for j, f in enumerate(self.pool)}
        self.da, self.W = da_full.index_select(1, pool), sae.W_dec.index_select(0, pool)
        self.full = (h_don - h_src).float()
        self.err = self.full - self.da.float() @ self.W.float()
        self.readout = t.source.meta["readout_position"]
        self.ev = PatchEvaluator(handle, as_ids(self.ids, handle.device), t.answers, readout=self.readout, mode=causal, strategy=strategy,
                                 batch_size=batch_size)

    def eff(self, feats) -> tuple[int, ...]:
        return tuple(sorted(f for f in feats if f in self.col))

    def evaluate(self, sets: list[tuple[int, ...]]):
        specs = []
        for feats in sets:
            c = torch.as_tensor([self.col[f] for f in feats], dtype=torch.long, device=self.da.device)
            specs.append([EditSpec(layer=self.layer, positions=self.positions, delta=self.da.index_select(1, c) @ self.W.index_select(0, c))])
        return self.ev.evaluate(specs)


def scores_from_gradient(p: Prompt, g: torch.Tensor) -> dict[int, float]:
    """Sum_p delta a_i(p) * (g(p) . d_i) for every pool feature (ATP's formula with the given gradient at the site)."""
    s = (p.da.float() * (g.float().to(p.W.device) @ p.W.float().T)).sum(0)
    return dict(zip(p.pool, s.tolist()))


def path_gradient(handle, p: Prompt, layer: int, *, n_steps: int, mode: str, checkpointed: bool) -> torch.Tensor:
    """The mean of dm/dh_l at the site over the midpoints of the straight path from the source to the donor residual."""
    total = torch.zeros_like(p.full)
    for j in range(n_steps):
        alpha = (j + 0.5) / n_steps
        edit = EditSpec(layer=layer, positions=p.positions, delta=alpha * p.full)
        _, g = margin_gradient(handle, as_ids(p.ids, handle.device), layer, p.t.answers, readout=p.readout, mode=mode, checkpointed=checkpointed,
                               edit=edit)
        total += g[p.positions].to(total.device)
    return total / n_steps


def greedy_support(supports: list[Prompt], m_clean: list[float], cand: list[int], k_max: int) -> list[dict[str, Any]]:
    """Forward selection by the mean margin gain over the support prompts; ties by the smaller feature id. Each distinct
    effective set is evaluated once per prompt."""
    chosen: list[int] = []
    path = []
    for _ in range(min(k_max, len(cand))):
        rest = [f for f in cand if f not in chosen]
        gains = np.zeros(len(rest))
        for p, m0 in zip(supports, m_clean):
            effs = [p.eff(chosen + [f]) for f in rest]
            uniq = sorted(set(effs))
            res = dict(zip(uniq, p.evaluate(uniq)))
            gains += np.array([res[e].margin - m0 for e in effs])
        gains /= max(len(supports), 1)
        best = max(range(len(rest)), key=lambda i: (gains[i], -rest[i]))
        chosen.append(rest[best])
        path.append({"k": len(chosen), "feature": rest[best], "support_gain": float(gains[best]), "n_candidates": len(rest)})
    return path


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    if ctx.phase not in ("dev", "test"):
        raise RunContractError("interact needs --phase dev or --phase test")
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
    causal, grad_mode = s0.get("causal_mode", s0["precision_mode"]), s0.get("gradient_mode", s0["precision_mode"])
    checkpointed = s0.get("backward") == "checkpointed"
    strategy = {(True, True): "batched_suffix", (True, False): "batched", (False, True): "suffix", (False, False): "serial"}[
        (bool(s0["batching_allowed"]), bool(s0["suffix_caching_allowed"]))]
    batch_size = int(s0.get("batch_size", 8))
    cell = entry["cell"]
    if cell not in parent_frozen["eligible_cells"]:
        raise RunContractError(f"{cell} is not an eligible cell of the parent protocol {parent_frozen['eligible_cells']}")
    layer, site = int(cell.split("/")[0]), cell.split("/")[1]
    pk = int(parent_frozen["primary_k"])
    k_c, k_max = int(cfg["candidates_k"]), int(cfg["k_max"])
    report_k = sorted({int(k) for k in cfg["report_k"]} | {pk})
    compare = list(cfg["compare_selectors"])
    seed = int(cfg["seed"])
    rows_in = choose_rows(data["rows"], cell=cell, n_pairs=int(cfg["n_pairs"]) if ctx.limit is None else ctx.limit, seed=int(cfg["subset_seed"]))
    at_cell = {(r["pair"], r["direction"], r["template"]): r for r in data["rows"] if f"{r['layer']}/{r['site']}" == cell}

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
    atp_tol = float(cfg["atp_parity_max_abs_m1"])
    n_steps = int(cfg["ig_steps"])
    ig_cache: dict[str, dict[str, Any]] = {}
    out_rows, parity_max, atp_parity_max = [], 0.0, 0.0

    def prompt(t: S2.Trial) -> Prompt:
        return Prompt(handle, sae, cache, t, layer, site, causal=causal, strategy=strategy, batch_size=batch_size)

    def check_parity(p: Prompt, stored: dict[str, Any], label: str) -> float:
        k = str(max(int(x) for x in cfg["report_k"]))
        e = p.eff(stored["sets"]["JL-OUT"][k]["active"])
        got = p.evaluate([e])[0].margin if e else stored["m"]["clean"]
        diff = abs(got - stored["sets"]["JL-OUT"][k]["margin"])
        if diff > tol:
            raise EngineeringFailed("parity", f"{label}: the stored JL-OUT set re-evaluated gives {got:.6g}, stored "
                                              f"{stored['sets']['JL-OUT'][k]['margin']:.6g} (tolerance {tol})")
        return diff

    def ig_of(p: Prompt) -> dict[str, Any]:
        if p.t.key not in ig_cache:
            gbar = path_gradient(handle, p, layer, n_steps=n_steps, mode=grad_mode, checkpointed=checkpointed)
            ig_cache[p.t.key] = {"IG": scores_from_gradient(p, gbar), "err": float((p.err.to(gbar.device) * gbar).sum())}
        return ig_cache[p.t.key]

    for r in rows_in:
        t = trials[r["trial"]]
        ev = prompt(t)
        support_rows = [at_cell[(r["pair"], r["direction"], tpl)] for tpl in r["support_templates"]]
        sup = [prompt(trials[s["trial"]]) for s in support_rows]
        parity = max([check_parity(ev, r, r["trial"])] + [check_parity(p, s, s["trial"]) for p, s in zip(sup, support_rows)])
        parity_max = max(parity_max, parity)
        effects = {int(f): v for f, v in r["effects"].items()}

        # ATP parity: the alpha = 0 gradient gives the stored in-context ATP M1
        _, g0 = margin_gradient(handle, as_ids(ev.ids, handle.device), layer, t.answers, readout=ev.readout, mode=grad_mode, checkpointed=checkpointed)
        atp_m1 = S.ranking_metrics(scores_from_gradient(ev, g0[ev.positions]), effects, ks=[])["M1"]
        stored_atp = r["metrics"]["ATP@in"]["M1"]
        atp_diff = 0.0 if atp_m1 is None and stored_atp is None else abs((atp_m1 or 0.0) - (stored_atp or 0.0))
        atp_parity_max = max(atp_parity_max, atp_diff)
        if atp_diff > atp_tol:
            raise EngineeringFailed("parity", f"{r['trial']}: recomputed ATP@in M1 {atp_m1} against stored {stored_atp} (tolerance {atp_tol})")

        # IG: in-context on every prompt, support score as ATP's
        ig_eval = ig_of(ev)
        ig_sup = S.support_scores(ig_of(p)["IG"] for p in sup)
        universe = sorted(set().union(*[set(p.pool) for p in sup]))
        ig_sets = {str(k): ev.eff(S.select_top(ig_sup, k, universe)) for k in report_k}
        completeness = ((sum(ig_eval["IG"].values()) + ig_eval["err"]) / r["full_gain"]) if r["full_gain"] else None

        # GREEDY-SUP on the support prompts
        cand = sorted({f for n in cfg["candidate_selectors"] for f in r["sets"][n][str(k_c)]["selected"]})
        m0 = [s["m"]["clean"] for s in support_rows]
        path = greedy_support(sup, m0, cand, k_max)
        sup_full = float(np.mean([s["full_gain"] for s in support_rows])) if support_rows else 0.0
        for step in path:  # the support gain as a share of the support prompts' mean full gain
            step["support_F"] = step["support_gain"] / sup_full if sup_full else None
        order = [step["feature"] for step in path]
        g_sets = {str(k): ev.eff(order[:k]) for k in report_k if k <= k_max}

        wanted = sorted({e for e in [*ig_sets.values(), *g_sets.values()] if e})
        res = dict(zip(wanted, ev.evaluate(wanted)))
        m_clean, m_full = r["m"]["clean"], r["m"]["FULL"]

        def entry(e):
            margin = res[e].margin if e else m_clean
            norms = [res[e].intended_norm, res[e].realized_norm] if e else [0.0, 0.0]
            gain = m_full - m_clean
            return {"active": list(e), "margin": margin, "norms": norms, "F": S.recovery(margin, m_clean, m_full) if gain else None,
                    "F_additive": (sum(effects.get(f, 0.0) for f in e) / gain) if gain else None}

        stored = {n: {str(k): {"F": r["sets"][n][str(k)]["F"], "norms": r["sets"][n][str(k)]["norms"],
                               "F_additive": (sum(effects.get(f, 0.0) for f in r["sets"][n][str(k)]["active"]) / r["full_gain"]) if r["full_gain"] else None}
                      for k in report_k if str(k) in r["sets"][n]} for n in [*compare, "ORACLE"]}
        out_rows.append({
            "trial": r["trial"], "relation": r["relation"], "pair": r["pair"], "template": r["template"], "direction": r["direction"],
            "layer": layer, "site": site, "full_gain": r["full_gain"], "m": {"clean": m_clean, "FULL": m_full},
            "ref_norms": r["ref_norms"], "pool_size": len(ev.pool), "n_candidates": len(cand), "support_templates": r["support_templates"],
            "sets": {"GREEDY-SUP": {k: entry(e) for k, e in g_sets.items()}, "IG": {k: entry(e) for k, e in ig_sets.items()}},
            "stored": stored, "greedy_support_path": path,
            "metrics": {"IG": S.ranking_metrics(ig_sup, effects, ks=[])["M1"], "IG@in": S.ranking_metrics(ig_eval["IG"], effects, ks=[])["M1"],
                        **{n: r["metrics"][n]["M1"] for n in compare}},
            "ig_completeness": completeness, "parity": parity, "atp_parity": atp_diff,
        })
    del sae
    S2._free()

    write_results_zip(ctx.run.path, {"trials.jsonl": "".join(json.dumps(x, sort_keys=True, default=S2._jsonable) + "\n" for x in out_rows)})
    analysis = analyze(out_rows, pcfg, cfg, pk=pk, report_k=report_k, seed=seed)
    report = json.loads(json.dumps({"phase": ctx.phase, "model": name, "parent_run": parent_dir.as_posix(),
                                    "parent_provenance": prov, "parent_frozen_protocol_hash": parent_frozen["protocol_hash"], "cell": cell,
                                    "primary_k": pk, "strategy": strategy, "gradient_mode": grad_mode, "n_trials": len(out_rows),
                                    "n_pairs": len({x["pair"] for x in out_rows}), "parity_max": parity_max, "parity_tolerance": tol,
                                    "atp_parity_max": atp_parity_max, "atp_parity_tolerance": atp_tol, "analysis": analysis}, default=S2._jsonable))
    summary = _summary(ctx, report, cfg)
    if ctx.phase == "dev":
        protocol = {"stage": "interact", "smoke": ctx.smoke, "code_commit": ctx.git.get("commit"),
                    "source_run": ctx.run.path.as_posix(), "created_at": utc_now(), "config": cfg, "model": model_cfg,
                    "parent_protocol_hash": parent_frozen["protocol_hash"], "cell": cell, "data": data_sha}
        frozen_protocol = seal_protocol(protocol)
        ctx.run.write_json("frozen_protocol.json", frozen_protocol)
        report["frozen_protocol_hash"] = frozen_protocol["protocol_hash"]
        return Outcome(status="FROZEN_AWAITING_APPROVAL", summary=summary, report=report, send_back=["frozen_protocol.json", "results.zip"])
    if ctx.frozen.get("parent_protocol_hash") != parent_frozen["protocol_hash"] or ctx.frozen.get("cell") != cell:
        raise RunContractError("the parent protocol or the cell differs from the frozen protocol")
    return Outcome(status="PASS", summary=summary, report=report, send_back=["results.zip"])


def _F(r: dict[str, Any], name: str, k: int) -> float | None:
    if name in r["sets"]:
        e = r["sets"][name].get(str(k))
    else:
        e = r["stored"].get(name, {}).get(str(k))
    return None if e is None else e["F"]


def analyze(rows, pcfg: dict[str, Any], cfg: dict[str, Any], *, pk: int, report_k: list[int], seed: int) -> dict[str, Any]:
    """The declared family (Holm), the secondary contrasts and the descriptive tables; trials at or above the parent's M2 floor."""
    floor = pcfg["m2_min_full_gain"]

    def k_of(k) -> int:
        return pk if k == "primary" else int(k)

    def f_diff(a, b, k):
        def get(r):
            if r["full_gain"] < floor:
                return None
            x, y = _F(r, a, k), _F(r, b, k)
            return None if x is None or y is None else x - y
        return get

    def m1_diff(a, b):
        def get(r):
            x, y = r["metrics"].get(a), r["metrics"].get(b)
            return None if x is None or y is None else x - y
        return get

    sesoi = pcfg["analysis"]["sesoi"]
    family = {}
    for a, b, k in cfg["family"]:
        family[f"{a} - {b} (F at k = {k_of(k)})"] = S2._labelled(rows, f_diff(a, b, k_of(k)), sesoi=sesoi["M2"], cfg=pcfg,
                                                                 seed=S2._hseed(seed, "family", a, b, k), predicted_sign=1)
    S2._holm(family, pcfg["analysis"]["holm_alpha"])
    secondary = {}
    for a, b, k in cfg["secondary"]:
        if k == "M1":
            secondary[f"{a} - {b} (M1)"] = S2._labelled(rows, m1_diff(a, b), sesoi=sesoi["M1"], cfg=pcfg, seed=S2._hseed(seed, "sec", a, b, "M1"))
        else:
            secondary[f"{a} - {b} (F at k = {k_of(k)})"] = S2._labelled(rows, f_diff(a, b, k_of(k)), sesoi=sesoi["M2"], cfg=pcfg,
                                                                         seed=S2._hseed(seed, "sec", a, b, k))
    names = ["GREEDY-SUP", "IG", *cfg["compare_selectors"], "ORACLE"]
    desc: dict[str, Any] = {}
    for k in report_k:
        per = {}
        for n in names:
            def get(r, n=n, k=k, key="F"):
                if r["full_gain"] < floor:
                    return None
                e = r["sets"][n].get(str(k)) if n in r["sets"] else r["stored"][n].get(str(k))
                if e is None:
                    return None
                if key == "rho":
                    full = r["ref_norms"]["FULL"][1]
                    return e["norms"][1] / full if full else None
                return e[key]
            if not any((r["sets"].get(n) or r["stored"].get(n) or {}).get(str(k)) for r in rows):
                continue
            per[n] = {"F": S2._mean_ci(rows, get, pcfg, S2._hseed(seed, "desc", n, k)),
                      "F_additive": S2._pair_mean(rows, lambda r, g=get: g(r, key="F_additive")),
                      "rho": S2._pair_mean(rows, lambda r, g=get: g(r, key="rho"))}
        desc[str(k)] = per
    comp = [r["ig_completeness"] for r in rows if r["ig_completeness"] is not None and r["full_gain"] >= floor]
    m1 = {n: S2._pair_mean(rows, lambda r, n=n: r["metrics"].get(n)) for n in ["IG", "IG@in", *cfg["compare_selectors"]]}
    support_vs_eval = {}  # GREEDY-SUP's recovery on the prompts it was chosen on against the evaluation prompt
    for k in [k for k in report_k if k <= int(cfg["k_max"])]:
        def s_f(r, k=k):
            step = next((x for x in r["greedy_support_path"] if x["k"] == k), None)
            return None if r["full_gain"] < floor or step is None else step.get("support_F")

        def e_f(r, k=k):
            return None if r["full_gain"] < floor else _F(r, "GREEDY-SUP", k)

        def gap(r, s_f=s_f, e_f=e_f):
            a, b = s_f(r), e_f(r)
            return None if a is None or b is None else a - b

        support_vs_eval[str(k)] = {"support_F": S2._pair_mean(rows, s_f), "eval_F": S2._pair_mean(rows, e_f),
                                   "gap": S2._mean_ci(rows, gap, pcfg, S2._hseed(seed, "support_gap", k))}
    return S2._strip({"family": family, "secondary": secondary, "by_k": desc, "M1": m1, "support_vs_eval": support_vs_eval,
                      "ig_completeness": {"median": float(np.median(comp)), "q25": float(np.quantile(comp, 0.25)), "q75": float(np.quantile(comp, 0.75)),
                                          "n": len(comp)} if comp else None})


def _summary(ctx: RunContext, report: dict[str, Any], cfg: dict[str, Any]) -> str:
    a = report["analysis"]

    def mci(c, signed=False):
        if not c:
            return "n/a"
        f = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
        return f(c["estimate"]) + (f" [{f(c['ci95'][0])}, {f(c['ci95'][1])}]" if c.get("ci95") else "")

    lines = ["## What ran", "",
             f"The joint selection, {ctx.phase} phase, on `{report['model']}` at {report['cell']}: {report['n_trials']} trials in "
             f"{report['n_pairs']} pairs of the parent Stage 2 {ctx.phase} run `{report['parent_run']}` (the greedy reference's subset rule). GREEDY-SUP: "
             f"forward selection by the mean margin gain over the support prompts, up to k = {cfg['k_max']}, among the stored top-"
             f"{cfg['candidates_k']} sets of " + ", ".join(cfg["candidate_selectors"]) + f". IG: path-integrated attribution with "
             f"{cfg['ig_steps']} midpoints on the source-to-donor path, support score as ATP's. Sets evaluated on the evaluation prompt "
             f"(patches {report['strategy']}, gradients {report['gradient_mode']}). F = (m(S) - m_clean) / (m_FULL - m_clean), unsquared; "
             "trials at or above the M2 floor.", "",
             f"Parity: stored JL-OUT sets re-evaluated on the evaluation and support prompts, worst margin difference "
             f"{report['parity_max']:.2g} (tolerance {report['parity_tolerance']:.2g}); recomputed ATP@in M1, worst difference "
             f"{report['atp_parity_max']:.2g} (tolerance {report['atp_parity_tolerance']:.2g}).", "",
             "## Family (Holm; pre-specified secondary analyses on the held-out pairs)", ""]
    lines += [f"- {n}: {S2._ci(c)}" for n, c in a["family"].items()]
    lines += ["", "## Secondary (three-way labels, no Holm)", ""] + [f"- {n}: {S2._ci(c)}" for n, c in a["secondary"].items()]
    for k, per in a["by_k"].items():
        lines += ["", f"## k = {k}", "", "| Selector | F [95%] | Additive prediction | Realized rho |", "|---|---|---|---|"]
        for n, v in per.items():
            lines.append(f"| {n} | {mci(v['F'])} | {S2._num(v['F_additive'])} | {S2._num(v['rho'])} |")
    lines += ["", "M1 (pair means): " + ", ".join(f"{n} {S2._num(v)}" for n, v in a["M1"].items())]
    if a.get("support_vs_eval"):
        lines += ["", "GREEDY-SUP on the support prompts it was chosen on (its mean support gain over their mean full gain) against "
                  "the evaluation prompt (descriptive): high on the support prompts but low on the evaluation prompt means the measured "
                  "interactions do not carry over between prompts; low on both means the policy did not find interacting sets.", "",
                  "| k | Support share | Evaluation F | Support minus evaluation [95%] |", "|---|---|---|---|"]
        for k, v in a["support_vs_eval"].items():
            lines.append(f"| {k} | {S2._num(v['support_F'])} | {S2._num(v['eval_F'])} | {mci(v['gap'], True)} |")
    c = a["ig_completeness"]
    if c:
        lines.append(f"IG completeness (Sum IG + IG_err) / (m_FULL - m_clean): median {c['median']:.3f} (IQR {c['q25']:.3f} to {c['q75']:.3f}, n {c['n']}).")
    lines += ["", "## What this permits", "",
              "Whether a selector that measures feature interactions on the support prompts (GREEDY-SUP), or one that integrates the "
              "gradient along the transfer path (IG), picks sets that recover more on held-out prompts than the stored selectors at "
              "the same k. Better: interactions or path nonlinearity explain part of why well-ranking signals pick weaker sets; "
              "equivalent: they do not, at this cell; otherwise inconclusive, never 'no difference'.", "", "## Next step", "",
              "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
