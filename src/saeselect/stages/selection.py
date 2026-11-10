"""Stage 2: selection benchmark on concept transfer.

Question: which signal finds the SAE features whose transfer moves the answer,
and does the J-lens add information over the logit lens, by depth and site?

Units. A trial is one source prompt (relation, concept pair, template,
direction) with its donor prompt; the answer sets are A+ = the donor's answer,
A- = the source's, so m rises toward the donor. A cell is (layer, site), the
site being the readout token or the concept tokens. Clusters are concept
pairs. Selection for a trial uses its support: the same pair and direction
under the other passing templates (5.4); in-context variants use the trial's
own prompt.

Dev phase:
1. Format gate (step 1) on clean readouts: a pair-direction passes when each of
   its two prompts puts >= 0.5 on its own answer set and |m| >= 1.0 with the
   correct sign; a template passes at >= 80%; a relation needs 2 of 3, else the
   reserve replaces it.
2. Cell eligibility (step 2): median R_FULL >= 0.5 over dev trials.
3. Lens divergence D_l (step 3) and the lens-fidelity diagnostic (step 4).
4. Full pipeline on eligible cells (step 5): candidate pools, every signal
   (support and in-context), single-feature effects, set effects for every
   signal and k, 20 random draws per k, references FULL, SAE-ALL, ERR, DIM.
5. Primary k (step 6) (derived on dev), then the alpha-curves at the primary k.
6. Sanity checks (step 7), power simulation (step 8), freeze (step 9).

Test phase: the frozen relations, templates, cells and primary k on test
pairs, once; the primary family (P2.1 to P2.3, Holm) and secondary analyses.

Precision (Stage 0 decisions of the model): causal measurements (clean
activations, margins, every patch) in the causal mode (exact FP32);
gradients (ATP, the local-gradient prediction) in the N7 mode with the
checkpointed backward. Batching only where N9 allowed it, with a parity
probe against serial inside every trial (rule 8). Nothing here is squared.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from saeselect import artifacts as A
from saeselect import lens as L
from saeselect import signals as S
from saeselect import stats
from saeselect.hooks import EditSpec
from saeselect.interventions import candidate_pool, err_delta, full_delta, unit
from saeselect.models import load_model
from saeselect.patching import PatchEvaluator, as_ids, clean_run, margin_gradient
from saeselect.provenance import EngineeringFailed, GateFailed, RunContractError, json_hash, seal_protocol, utc_now
from saeselect.readout import AnswerSets, answer_margin, logits_fp32, next_token_logprobs, off_target, output_direction, vocabulary_direction
from saeselect.results import CheckpointStore, write_results_zip
from saeselect.runner import Outcome, RunContext
from saeselect.tasks.records import TaskRecord, hash_unit, load_records
from saeselect.tasks.tokenization import encode_prompt

ANALYSIS_PLAN = (
    "Primary family (6.3, Holm at 0.05): P2.1 JL-OUT minus LL-OUT on M1 (Spearman of support scores with single-feature "
    "transfer effects over the evaluation pool), pooled over eligible cells; P2.2 the same on M2 (recovered fraction F_k at the "
    "frozen primary k, trials with m_FULL - m_clean >= 1.0); P2.3 slope of the per-layer JL-OUT minus LL-OUT M1 contrast at the "
    "readout site against the dev lens divergence D_l (testable with >= 3 eligible readout layers). Per trial, then averaged within "
    "concept pairs; cluster bootstrap with 10,000 resamples stratified by relation; three-way verdicts with SESOI 0.05 (M1, M2)."
)


# --------------------------------------------------------------------------
# Trials and readouts
# --------------------------------------------------------------------------


@dataclass
class Trial:
    key: str
    relation: str
    pair: str
    template: str
    direction: str
    source: TaskRecord
    donor: TaskRecord
    answers: AnswerSets
    readout: dict[str, float] = field(default_factory=dict)  # m_clean, m_don, masses
    answerable: bool = False


def _answers(record: TaskRecord) -> AnswerSets:
    return AnswerSets(plus=tuple(record.answer_sets["A_plus"]["ids"]), minus=tuple(record.answer_sets["A_minus"]["ids"]))


def build_trials(records, relation: str) -> list[Trial]:
    groups: dict[str, dict[str, TaskRecord]] = defaultdict(dict)
    for r in records:
        groups[r.matched_group_id][r.role] = r
    out = []
    for key, g in sorted(groups.items()):
        if "source" in g and "donor" in g:
            s = g["source"]
            out.append(Trial(key=key, relation=relation, pair=s.cluster_id, template=s.template_id,
                             direction=f"{s.meta['source_concept']}->{s.meta['donor_concept']}", source=s, donor=g["donor"], answers=_answers(s)))
    return out


def set_masses(h_final: torch.Tensor, head, answers: AnswerSets) -> tuple[float, float, float]:
    """(p(A+), p(A-), m) from full-vocabulary FP32 logits at one position."""
    z = logits_fp32(h_final, head)
    logp = torch.log_softmax(z, dim=-1)
    plus = float(logp[list(answers.plus)].exp().sum())
    minus = float(logp[list(answers.minus)].exp().sum())
    return plus, minus, float(answer_margin(h_final, head, answers))


class PromptCache:
    """One clean forward per distinct prompt in the causal mode: the final residual at
    the readout and the residuals at every site position of every usable layer (FP32, CPU)."""

    def __init__(self, handle, layers: list[int], mode: str):
        self.handle, self.layers, self.mode = handle, layers, mode
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, record: TaskRecord) -> dict[str, Any]:
        key = record.prompt
        if key not in self._store:
            tok = self.handle.tokenizer
            ids = encode_prompt(tok, record.prompt, record.prompt_format)
            if len(ids) != record.meta["n_tokens"]:
                raise EngineeringFailed("tokenization", f"{record.id}: {len(ids)} tokens, the data build recorded {record.meta['n_tokens']}")
            acts, h_final = clean_run(self.handle, as_ids(ids, self.handle.device), self.layers, mode=self.mode)
            r = record.meta["readout_position"]
            sites = {"readout": [r], "concept": list(record.meta["concept_positions"])}
            self._store[key] = {
                "ids": ids, "h_final": h_final[r].float().cpu(), "sites": sites,
                "acts": {(layer, site): acts[layer][pos].float().cpu() for layer in self.layers for site, pos in sites.items()},
            }
        return self._store[key]


def format_gate(trials: list[Trial], cfg: dict[str, Any]) -> dict[str, Any]:
    """6.3 dev step 1 (and the test answerability rule): per pair-direction, then per template."""
    g = cfg["format_gate"]
    per_template: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        ro = t.readout
        src_ok = ro["src_own_mass"] >= g["min_own_mass"] and ro["m_clean"] <= -g["min_abs_margin"]
        don_ok = ro["don_own_mass"] >= g["min_own_mass"] and ro["m_don"] >= g["min_abs_margin"]
        t.answerable = bool(src_ok and don_ok)
        per_template[t.template].append(t.answerable)
    templates = {tpl: {"n": len(v), "share": float(np.mean(v)), "pass": float(np.mean(v)) >= g["min_template_share"]} for tpl, v in sorted(per_template.items())}
    passing = [tpl for tpl, v in templates.items() if v["pass"]]
    share_all = float(np.mean([t.answerable for t in trials])) if trials else 0.0
    return {"templates": templates, "passing_templates": passing, "pass": len(passing) >= g["min_templates"], "answerable_share": share_all}


# --------------------------------------------------------------------------
# One pair-direction at one cell: the whole pipeline
# --------------------------------------------------------------------------


def _tokens(tok, name: str) -> list[int]:
    ids = tok.encode(" " + name, add_special_tokens=False) + tok.encode(name, add_special_tokens=False)
    return sorted(set(ids))


@dataclass
class CellContext:
    handle: Any
    sae: Any
    layer: int
    site: str
    J: torch.Tensor
    sd: torch.Tensor  # [width]
    kappa: torch.Tensor  # [width]
    jspace: dict[int, float]
    causal: str
    grad_mode: str
    checkpointed: bool
    strategy: str
    batch_size: int
    cfg: dict[str, Any]
    seed: int


def _evaluate(cx: CellContext, ids, answers, readout: int, positions: list[int], deltas: list[torch.Tensor], scale: float = 1.0,
              readout_fn=None):
    """Margins and norms of edit sets (one delta [|P|, d] each) on one prompt; ``readout_fn`` adds a readout of each set's
    final residual from the same forward pass."""
    if not deltas:
        return []
    ev = PatchEvaluator(cx.handle, as_ids(ids, cx.handle.device), answers, readout=readout, mode=cx.causal,
                        strategy=cx.strategy, batch_size=cx.batch_size, readout_fn=readout_fn)
    return ev.evaluate([[EditSpec(layer=cx.layer, positions=positions, delta=d, scale=scale)] for d in deltas])


def trial_pass(cx: CellContext, t: Trial, cache: PromptCache) -> dict[str, Any]:
    """In-context quantities of one trial: pool, scores, single-feature effects, references."""
    src, don = cache.get(t.source), cache.get(t.donor)
    positions = src["sites"][cx.site]
    dev = cx.sae.device
    h_src = src["acts"][(cx.layer, cx.site)].to(dev)
    h_don = don["acts"][(cx.layer, cx.site)].to(dev)
    da_full = cx.sae.encode(h_don) - cx.sae.encode(h_src)
    pool = candidate_pool(da_full)
    da = da_full.index_select(1, pool)
    W = cx.sae.W_dec.index_select(0, pool)
    pool_ids = pool.tolist()
    head = cx.handle.head
    h_f = src["h_final"].to(dev)
    w_out = output_direction(h_f, head, t.answers)
    w_q = vocabulary_direction(head, _tokens(cx.handle.tokenizer, t.source.meta["donor_concept"]),
                               _tokens(cx.handle.tokenizer, t.source.meta["source_concept"]), dev)
    lens_s = {"LL-OUT": L.lens_sensitivity(w_out, h_f, head, None), "JL-OUT": L.lens_sensitivity(w_out, h_f, head, cx.J),
              "LL-VERB": L.lens_sensitivity(w_q, h_f, head, None), "JL-VERB": L.lens_sensitivity(w_q, h_f, head, cx.J)}
    readout = t.source.meta["readout_position"]
    _, grad = margin_gradient(cx.handle, as_ids(src["ids"], cx.handle.device), cx.layer, t.answers, readout=readout,
                              mode=cx.grad_mode, checkpointed=cx.checkpointed)
    j_norm = None
    if "JL-NORM" in (cx.cfg.get("extra_signals") or []):  # ||J_l d_i||, the decoder row pushed to the last layer
        j_norm = (W.float() @ cx.J.float().to(W.device).T).norm(dim=1)
    scores = S.in_context_scores(da=da, W_dec_pool=W, sd_pool=cx.sd.index_select(0, pool.to(cx.sd.device)).to(dev),
                                 lens_s=lens_s, grad_site=grad[positions].to(dev), j_norm_pool=j_norm)
    m_clean, m_don = t.readout["m_clean"], t.readout["m_don"]

    # single-feature effects (ORACLE on this prompt, PATCH for the others), with a parity probe
    singles = [da[:, [j]] @ W[[j]] for j in range(len(pool_ids))]
    res = _evaluate(cx, src["ids"], t.answers, readout, positions, singles)
    effects = {f: r.margin - m_clean for f, r in zip(pool_ids, res)}
    norms = {f: (r.intended_norm, r.realized_norm) for f, r in zip(pool_ids, res)}
    parity = 0.0
    if cx.strategy != "serial" and singles:
        n = min(cx.cfg["parity_probe"]["n_sets"], len(singles))
        serial = PatchEvaluator(cx.handle, as_ids(src["ids"], cx.handle.device), t.answers, readout=readout, mode=cx.causal, strategy="serial")
        probe = serial.evaluate([[EditSpec(layer=cx.layer, positions=positions, delta=d)] for d in singles[:n]])
        parity = max(abs(a.margin - b.margin) for a, b in zip(probe, res[:n]))

    # references
    full = full_delta(h_src, h_don)
    sae_all = da @ W
    err = err_delta(cx.sae, h_src, h_don)
    sanity = float((sae_all + err - full).abs().max() / full.abs().max().clamp_min(1e-12))
    ref = _evaluate(cx, src["ids"], t.answers, readout, positions, [full, sae_all, err])
    return {
        "trial": t, "positions": positions, "pool": pool_ids, "da": da.cpu(), "W_rows": pool.cpu(),
        "scores": {k: dict(zip(pool_ids, v.tolist())) for k, v in scores.items()},
        "effects": effects, "single_norms": norms, "parity": parity, "sanity_rel": sanity,
        "diff_mean": (h_don - h_src).mean(0).cpu(),
        "m": {"clean": m_clean, "don": m_don, "FULL": ref[0].margin, "SAE-ALL": ref[1].margin, "ERR": ref[2].margin},
        "ref_norms": {n: (r.intended_norm, r.realized_norm) for n, r in zip(("FULL", "SAE-ALL", "ERR"), ref)},
    }


def _jspace(cx: CellContext, bank, features: list[int]) -> None:
    missing = sorted(set(features) - set(cx.jspace))
    for start in range(0, len(missing), 64):
        chunk = missing[start : start + 64]
        idx = torch.as_tensor(chunk, device=cx.sae.device)
        captured, _ = L.jspace_alignment(cx.sae.W_dec.index_select(0, idx), bank, max_atoms=cx.cfg["jspace_max_atoms"])
        cx.jspace.update({f: float(c) for f, c in zip(chunk, captured)})


def evaluate_trial(cx: CellContext, own: dict[str, Any], support: list[dict[str, Any]], *, ks: list[int], alphas: list[float] | None,
                   primary_k: int | None) -> dict[str, Any]:
    """Set effects and metrics of one evaluation trial given its support trials."""
    t: Trial = own["trial"]
    pool = own["pool"]
    universe = sorted(set().union(*[set(s["pool"]) for s in support]))
    variants: dict[str, dict[int, float]] = {}
    for name in S.IN_CONTEXT:
        variants[f"{name}"] = S.support_scores(s["scores"][name] for s in support)
        variants[f"{name}@in"] = own["scores"][name]
    for name in cx.cfg.get("extra_signals") or []:  # JL-NORM, support and in-context like the other in-context signals
        variants[name] = S.support_scores(s["scores"][name] for s in support)
        variants[f"{name}@in"] = own["scores"][name]
    variants["PATCH"] = S.support_scores(s["effects"] for s in support)
    variants["KAPPA"] = {f: float(cx.kappa[f]) for f in universe}
    variants["JSPACE"] = {f: cx.jspace.get(f, 0.0) for f in universe}
    variants["ORACLE"] = own["effects"]
    for hname, (a, b) in (cx.cfg.get("hybrids") or {}).items():  # support selectors combined by rank average, ties by NORM
        variants[hname] = S.rank_average(variants[a], variants[b], universe, tiebreak=variants["NORM"])
    universes = {name: (pool if name.endswith("@in") or name == "ORACLE" else universe) for name in variants}

    m = own["m"]
    col = {f: j for j, f in enumerate(pool)}
    da, W = own["da"].to(cx.sae.device), cx.sae.W_dec.index_select(0, own["W_rows"].to(cx.sae.device))

    def delta_for(features: list[int]) -> torch.Tensor | None:
        cols = [col[f] for f in features if f in col]  # a feature inactive on this prompt contributes nothing (5.4 note 4)
        if not cols:
            return None
        c = torch.as_tensor(cols, device=da.device)
        return da.index_select(1, c) @ W.index_select(0, c)

    rng = np.random.default_rng(int(hashlib.sha256(f"{cx.seed}|{t.key}|{cx.layer}|{cx.site}".encode()).hexdigest()[:8], 16))
    selections: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for name, sc in variants.items():
        for k in ks:
            selections[name][str(k)] = S.select_top(sc, k, universes[name])
    randoms = {str(k): [S.random_draw(universe, k, rng) for _ in range(cx.cfg["n_random"])] for k in ks}
    nb = cx.cfg.get("norm_budget")
    plan = budget_plan(cx, own, variants, universes, universe, col, da, W, nb) if nb else None  

    # evaluate every distinct effective set once
    wanted: dict[tuple, list[int]] = {}
    for per_k in selections.values():
        for sel in per_k.values():
            eff = tuple(sorted(f for f in sel if f in col))
            wanted.setdefault(eff, list(eff))
    for draws in randoms.values():
        for sel in draws:
            eff = tuple(sorted(f for f in sel if f in col))
            wanted.setdefault(eff, list(eff))
    for eff in (plan_sets(plan) if plan else []):
        wanted.setdefault(eff, list(eff))
    keys = [k for k in wanted if k]
    src_pos = own["positions"]
    deltas = [delta_for(list(k)) for k in keys]
    readout = t.source.meta["readout_position"]
    readout_fn, extra_of, off_ref = None, {}, None
    if cx.cfg.get("offtarget"):  # the clean and FULL distributions through the same evaluator, then every set's O1 to O3
        head, ans_ids = cx.handle.head, list(t.answers.ids)
        full_d = own["full_delta"].to(da.device)
        ref = _evaluate(cx, _ids(own), t.answers, readout, src_pos, [torch.zeros_like(full_d), full_d],
                        readout_fn=lambda h: {"logp": next_token_logprobs(h, head)})
        lp_clean, lp_full = ref[0].extra["logp"], ref[1].extra["logp"]
        readout_fn = lambda h: off_target(next_token_logprobs(h, head), lp_clean, lp_full, ans_ids)  # noqa: E731
        off_ref = {"clean": off_target(lp_clean, lp_clean, lp_full, ans_ids), "FULL": off_target(lp_full, lp_clean, lp_full, ans_ids),
                   "full_margin_parity": abs(ref[1].margin - m["FULL"])}
        extra_of[()] = off_ref["clean"]
    results = _evaluate(cx, _ids(own), t.answers, readout, src_pos, deltas, readout_fn=readout_fn)
    margin_of = {(): m["clean"]}
    norm_of = {(): (0.0, 0.0)}
    for k_, r in zip(keys, results):
        margin_of[k_] = r.margin
        norm_of[k_] = (r.intended_norm, r.realized_norm)
        if r.extra is not None:
            extra_of[k_] = r.extra
    # DIM reference: the site difference projected on u, the unit mean difference over the support prompts
    dim_u = unit(torch.stack([s["diff_mean"] for s in support]).mean(0)).to(da.device)
    h_diff = own["full_delta"].to(da.device)
    m["DIM"] = _evaluate(cx, _ids(own), t.answers, readout, src_pos, [(h_diff @ dim_u).unsqueeze(-1) * dim_u])[0].margin

    full_gain = m["FULL"] - m["clean"]
    out_sets: dict[str, dict[str, Any]] = {}
    for name, per_k in selections.items():
        out_sets[name] = {}
        for k, sel in per_k.items():
            eff = tuple(sorted(f for f in sel if f in col))
            out_sets[name][k] = {"selected": sel, "active": list(eff), "margin": margin_of[eff], "norms": norm_of[eff],
                                 "F": S.recovery(margin_of[eff], m["clean"], m["FULL"]) if full_gain != 0 else None}
            if extra_of:
                out_sets[name][k]["offtarget"] = extra_of[eff]
    def eff_of(sel):
        return tuple(sorted(f for f in sel if f in col))

    out_random = {k: [{"active": list(eff_of(sel)), "margin": margin_of[eff_of(sel)], "norms": norm_of[eff_of(sel)],
                       "F": S.recovery(margin_of[eff_of(sel)], m["clean"], m["FULL"]) if full_gain != 0 else None,
                       **({"offtarget": extra_of[eff_of(sel)]} if extra_of else {})}
                      for sel in draws] for k, draws in randoms.items()}
    metrics = {}
    for name, sc in variants.items():
        control = variants["NORM"] if name in ("KAPPA", "JSPACE", "JL-NORM") else (variants["LL-OUT"] if name == "JL-OUT" else None)
        metrics[name] = S.ranking_metrics(sc, own["effects"], ks=ks, control=control)
    result = {
        "trial": t.key, "relation": t.relation, "pair": t.pair, "template": t.template, "direction": t.direction,
        "layer": cx.layer, "site": cx.site, "positions": src_pos, "pool_size": len(pool), "support_templates": [s["trial"].template for s in support],
        "universe_size": len(universe), "m": m, "full_gain": full_gain, "effects": {str(f): v for f, v in own["effects"].items()},
        "in_context_lens": {"LL-OUT": [own["scores"]["LL-OUT"][f] for f in pool], "JL-OUT": [own["scores"]["JL-OUT"][f] for f in pool]},
        "metrics": metrics, "sets": out_sets, "random": out_random, "parity": own["parity"], "sanity_rel": own["sanity_rel"],
        "ref_norms": own["ref_norms"],
    }
    if off_ref is not None:
        result["offtarget_ref"] = off_ref
    if plan is not None:
        result["norm_budget"] = budget_rows(plan, margin_of, norm_of, m)
    if alphas is not None and primary_k is not None:
        result["alpha"] = alpha_curves(cx, own, out_sets, out_random, primary_k, alphas)
    return result


def _ids(own: dict[str, Any]) -> list[int]:
    return own["ids"]


def alpha_curves(cx: CellContext, own: dict[str, Any], out_sets, out_random, primary_k: int, alphas: list[float]) -> dict[str, Any]:
    """M4: effect against realized norm for each alpha, at the primary k."""
    t: Trial = own["trial"]
    col = {f: j for j, f in enumerate(own["pool"])}
    da = own["da"].to(cx.sae.device)
    W = cx.sae.W_dec.index_select(0, own["W_rows"].to(cx.sae.device))
    readout = t.source.meta["readout_position"]
    k = str(primary_k)
    named = {name: per_k[k]["active"] for name, per_k in out_sets.items() if k in per_k}
    named.update({f"RANDOM#{i}": d["active"] for i, d in enumerate(out_random.get(k, []))})
    at_one = {name: per_k[k] for name, per_k in out_sets.items() if k in per_k}
    at_one.update({f"RANDOM#{i}": d for i, d in enumerate(out_random.get(k, []))})
    out: dict[str, list[dict[str, float]]] = {}
    for alpha in alphas:
        if alpha == 1:  # measured in the main pass
            for n, d in at_one.items():
                if d["active"]:
                    out.setdefault(n, []).append({"alpha": 1.0, "effect": d["margin"] - own["m"]["clean"], "realized_norm": d["norms"][1]})
            continue
        items = [(n, f) for n, f in named.items() if f]
        deltas = []
        for _, feats in items:
            c = torch.as_tensor([col[x] for x in feats], device=da.device)
            deltas.append(da.index_select(1, c) @ W.index_select(0, c))
        res = _evaluate(cx, own["ids"], t.answers, readout, own["positions"], deltas, scale=alpha)
        for (n, _), r in zip(items, res):
            out.setdefault(n, []).append({"alpha": alpha, "effect": r.margin - own["m"]["clean"], "realized_norm": r.realized_norm})
    return out


# --------------------------------------------------------------------------
# the norm-budget endpoint (E2)
# --------------------------------------------------------------------------


def _prefix_norms(order: list[int], col: dict[int, int], da: torch.Tensor, W: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """The intended norm (Frobenius over the site positions, float64) of the delta of every prefix of ``order``, features
    active on the prompt: the norm the edit records as intended, before any cast."""
    out, running = [], None
    for start in range(0, len(order), chunk):
        c = torch.as_tensor([col[f] for f in order[start : start + chunk]], device=da.device)
        per = da.index_select(1, c).double().T.unsqueeze(-1) * W.index_select(0, c).double().unsqueeze(1)  # [n, |P|, d]
        cum = per.cumsum(0) if running is None else per.cumsum(0) + running
        running = cum[-1]
        out.append(cum.flatten(1).norm(dim=1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


def budget_plan(cx: CellContext, own: dict[str, Any], variants, universes, universe, col, da, W, nb: dict[str, Any]) -> dict[str, Any]:
    """The new-task endpoint (E2). Each declared selector adds features in its own order (its full ranking of its universe, as select_top
    ranks; a feature inactive on the prompt adds nothing and is skipped) until the intended norm of the edit reaches
    B x ||delta h_FULL|| for each declared budget B; no scaling. RANDOM: ``n_random`` seeded permutations of the universe,
    walked the same way. Curve: each selector's top-k sets on the declared fine k grid. Sets are tuples of active features."""
    full_norm = float(own["full_delta"].double().norm())
    budgets = [str(float(b)) for b in nb["budgets"]]
    rng = np.random.default_rng(_hseed(cx.seed, "budget", own["trial"].key, cx.layer, cx.site))

    def walk(order: list[int]) -> dict[str, dict[str, Any] | None]:
        steps = [(i, f) for i, f in enumerate(order) if f in col]
        norms = _prefix_norms([f for _, f in steps], col, da, W)
        out: dict[str, dict[str, Any] | None] = {}
        for b in budgets:
            hit = (norms >= float(b) * full_norm).nonzero()
            if full_norm == 0 or len(hit) == 0:
                out[b] = None  # the whole ordering never reaches the budget
                continue
            n = int(hit[0]) + 1
            out[b] = {"set": tuple(sorted(f for _, f in steps[:n])), "depth": steps[n - 1][0] + 1,
                      "rho_intended": float(norms[n - 1]) / full_norm}
        return out

    sets, curve = {}, {}
    for name in nb["selectors"]:
        order = S.select_top(variants[name], len(set(universes[name])), universes[name])
        sets[name] = walk(order)
        curve[name] = {str(k): tuple(sorted(f for f in order[:k] if f in col)) for k in nb["curve_k"]}
    random = [walk([int(x) for x in rng.permutation(sorted(set(universe)))]) for _ in range(nb["n_random"])]
    return {"full_norm": full_norm, "budgets": budgets, "sets": sets, "curve": curve, "random": random}


def plan_sets(plan: dict[str, Any]) -> list[tuple]:
    out = [e["set"] for per in plan["sets"].values() for e in per.values() if e]
    out += [e["set"] for walk in plan["random"] for e in walk.values() if e]
    return out + [eff for per in plan["curve"].values() for eff in per.values()]


def budget_rows(plan: dict[str, Any], margin_of, norm_of, m: dict[str, float]) -> dict[str, Any]:
    """The stored norm-budget fields of one trial: per selector and budget the set reached (or not), its depth in the ranking, the
    intended prefix norm that stopped the walk (over ||delta h_FULL||), its margin, intended and realized norms and F; the
    RANDOM walks; the curve points. Unsquared."""
    gain = m["FULL"] - m["clean"]

    def entry(eff):
        return {"active": list(eff), "k_active": len(eff), "margin": margin_of[eff], "norms": norm_of[eff],
                "F": S.recovery(margin_of[eff], m["clean"], m["FULL"]) if gain != 0 else None}

    def at(e):
        return {"reached": False} if e is None else {"reached": True, "depth": e["depth"], "rho_intended": e["rho_intended"], **entry(e["set"])}

    return {"full_norm": plan["full_norm"],
            "sets": {name: {b: at(e) for b, e in per.items()} for name, per in plan["sets"].items()},
            "random": {b: [at(walk[b]) for walk in plan["random"]] for b in plan["budgets"]},
            "curve": {name: {k: entry(eff) for k, eff in per.items()} for name, per in plan["curve"].items()}}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _cluster_means(rows: list[dict[str, Any]], value) -> tuple[list[float], list[str], list[str]]:
    acc: dict[str, list[float]] = defaultdict(list)
    rel: dict[str, str] = {}
    for r in rows:
        v = value(r)
        if v is not None and math.isfinite(v):
            acc[r["pair"]].append(v)
            rel[r["pair"]] = r["relation"]
    pairs = sorted(acc)
    return [float(np.mean(acc[p])) for p in pairs], pairs, [rel[p] for p in pairs]


def contrast(rows, a: str, b: str, metric: str, *, k: int | None, cfg: dict[str, Any], seed: int, predicted_sign: int = 1) -> dict[str, Any] | None:
    """a minus b on M1 (metric "M1") or M2 at k, per trial, averaged within pairs, cluster bootstrap stratified by relation."""
    def value(r):
        if metric == "M1":
            x, y = r["metrics"][a]["M1"], r["metrics"][b]["M1"]
            return None if x is None or y is None else x - y
        if r["full_gain"] < cfg["m2_min_full_gain"]:
            return None
        x, y = r["sets"][a][str(k)]["F"], r["sets"][b][str(k)]["F"]
        return None if x is None or y is None else x - y

    values, pairs, strata = _cluster_means(rows, value)
    if len(values) < 2:
        return None
    b_ = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(values, n_resamples=b_["n_resamples"], seed=seed, strata=strata)
    v = stats.verdict(res, sesoi=cfg["analysis"]["sesoi"]["M1" if metric == "M1" else "M2"], predicted_sign=predicted_sign,
                      level=b_["level"], tost_level=b_["tost_level"])
    return {"estimate": v.estimate, "ci95": list(v.ci95), "ci90": list(v.ci90), "p": v.p_value, "label": v.label, "n_pairs": len(values),
            "better": v.better, "equivalent": v.equivalent, "worse": v.worse, "practically_better": v.practically_better, "_verdict": v}


def slope_p23(rows, divergence: dict[int, float], cfg: dict[str, Any], seed: int) -> dict[str, Any] | None:
    """P2.3: slope of the per-layer JL-OUT minus LL-OUT M1 contrast (readout site) against D_l."""
    layers = sorted(divergence)
    per_pair: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    rel: dict[str, str] = {}
    for r in rows:
        if r["site"] != "readout" or r["layer"] not in divergence:
            continue
        x, y = r["metrics"]["JL-OUT"]["M1"], r["metrics"]["LL-OUT"]["M1"]
        if x is not None and y is not None:
            per_pair[r["pair"]][r["layer"]].append(x - y)
            rel[r["pair"]] = r["relation"]
    pairs = [p for p in sorted(per_pair) if all(per_pair[p][layer] for layer in layers)]
    if len(pairs) < 2 or len(layers) < cfg["analysis"]["p23_min_layers"]:
        return None
    matrix = np.array([[np.mean(per_pair[p][layer]) for layer in layers] for p in pairs])
    d = np.array([divergence[layer] for layer in layers])

    def slope(m: np.ndarray) -> float:
        y = m.mean(0)
        return float(np.polyfit(d, y, 1)[0]) if np.ptp(d) > 0 else float("nan")

    b_ = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(matrix, n_resamples=b_["n_resamples"], seed=seed, statistic=slope, strata=[rel[p] for p in pairs])
    v = stats.verdict(res, sesoi=cfg["analysis"]["sesoi"]["M1"], predicted_sign=1, level=b_["level"], tost_level=b_["tost_level"])
    return {"estimate": v.estimate, "ci95": list(v.ci95), "p": v.p_value, "label": v.label, "n_pairs": len(pairs), "layers": layers,
            "divergence": [divergence[x] for x in layers], "better": v.better, "equivalent": v.equivalent, "worse": v.worse, "_verdict": v}


def lens_divergence(rows) -> dict[int, float]:
    """D_l = 1 - Spearman(JL-OUT, LL-OUT) over every dev candidate at layer l, readout site (in-context scores)."""
    by_layer: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for r in rows:
        if r["site"] == "readout":
            by_layer[r["layer"]][0].extend(r["in_context_lens"]["JL-OUT"])
            by_layer[r["layer"]][1].extend(r["in_context_lens"]["LL-OUT"])
    return {layer: 1.0 - stats.spearman(jl, ll) for layer, (jl, ll) in sorted(by_layer.items()) if len(jl) > 2}


def primary_k(rows, cfg: dict[str, Any]) -> tuple[int, dict[str, float]]:
    """6.3 dev step 6: the smallest k whose median ORACLE recovery over dev trials (FULL gain >= 1.0) reaches the bar."""
    med = {}
    for k in cfg["k_grid"]:
        vals = [r["sets"]["ORACLE"][str(k)]["F"] for r in rows if r["full_gain"] >= cfg["m2_min_full_gain"] and r["sets"]["ORACLE"][str(k)]["F"] is not None]
        med[str(k)] = float(np.median(vals)) if vals else float("nan")
    rule = cfg["primary_k"]
    if rule["default"] not in cfg["k_grid"]:
        raise EngineeringFailed("config", f"primary_k.default {rule['default']} is not in k_grid {cfg['k_grid']}")
    chosen = next((k for k in cfg["k_grid"] if med[str(k)] >= rule["min_median_oracle_recovery"]), rule["default"])
    return int(chosen), med


def summarize(rows, cfg: dict[str, Any], *, pk: int, seed: int) -> dict[str, Any]:
    """Per signal (support and in-context): cluster-averaged M1 and F at every k, per cell and pooled; M5 at the primary k."""
    names = sorted({n for r in rows for n in r["metrics"]})
    cells = sorted({(r["layer"], r["site"]) for r in rows})
    out: dict[str, Any] = {"pooled": {}, "per_cell": {}, "M5": {}}

    def agg(sub, name):
        m1, _, _ = _cluster_means(sub, lambda r: r["metrics"][name]["M1"])
        fk = {}
        for k in cfg["k_grid"]:
            vals, _, _ = _cluster_means(sub, lambda r, k=k: r["sets"][name][str(k)]["F"] if r["full_gain"] >= cfg["m2_min_full_gain"] else None)
            fk[str(k)] = float(np.mean(vals)) if vals else None
        return {"M1": float(np.mean(m1)) if m1 else None, "n_pairs": len(m1), "F": fk}

    for name in names:
        out["pooled"][name] = agg(rows, name)
        for layer, site in cells:
            out["per_cell"].setdefault(f"L{layer}/{site}", {})[name] = agg([r for r in rows if r["layer"] == layer and r["site"] == site], name)
        pct = []
        for r in rows:
            if r["full_gain"] < cfg["m2_min_full_gain"]:
                continue
            f = r["sets"][name][str(pk)]["F"]
            draws = [d["F"] for d in r["random"][str(pk)] if d["F"] is not None]
            if f is not None and draws:
                pct.append(float(np.mean([f > d for d in draws]) + 0.5 * np.mean([f == d for d in draws])))
        out["M5"][name] = float(np.mean(pct)) if pct else None
    rnd = []
    for r in rows:
        if r["full_gain"] >= cfg["m2_min_full_gain"]:
            draws = [d["F"] for d in r["random"][str(pk)] if d["F"] is not None]
            if draws:
                rnd.append(float(np.mean(draws)))
    out["random_mean_F_at_primary_k"] = float(np.mean(rnd)) if rnd else None
    return out


def analyze(rows, cfg: dict[str, Any], *, pk: int, divergence: dict[int, float], seed: int) -> dict[str, Any]:
    """Primary family with Holm, and the declared secondary contrasts."""
    fam = {"P2.1": contrast(rows, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=seed + 1),
           "P2.2": contrast(rows, "JL-OUT", "LL-OUT", "M2", k=pk, cfg=cfg, seed=seed + 2),
           "P2.3": slope_p23(rows, divergence, cfg, seed + 3)}
    testable = {k: v["_verdict"] for k, v in fam.items() if v is not None}
    if testable:
        stats.apply_holm(testable, alpha=cfg["analysis"]["holm_alpha"])
        for k, v in testable.items():
            fam[k]["p_holm"] = v.p_adjusted
            fam[k]["better_after_holm"] = v.better_after_holm
    secondary = {}
    for a, b in (("JL-OUT", "ATP"), ("LL-OUT", "ATP"), ("JL-OUT", "PATCH"), ("LL-OUT", "PATCH"), ("JL-VERB", "LL-VERB"),
                 ("KAPPA", "NORM"), ("JSPACE", "NORM"), ("JL-OUT@in", "LL-OUT@in"), ("ATP@in", "ATP")):
        for metric in ("M1", "M2"):
            c = contrast(rows, a, b, metric, k=pk if metric == "M2" else None, cfg=cfg, seed=seed + 11)
            secondary[f"{a} - {b} ({metric})"] = c
    h3 = {}
    for layer in sorted({r["layer"] for r in rows}):
        per = {site: contrast([r for r in rows if r["layer"] == layer and r["site"] == site], "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=seed + 21)
               for site in ("readout", "concept")}
        h3[str(layer)] = per
    for d in (fam, secondary):
        for v in d.values():
            if v is not None:
                v.pop("_verdict", None)
    for per in h3.values():
        for v in per.values():
            if v is not None:
                v.pop("_verdict", None)
    return {"primary": fam, "secondary": secondary, "H3_per_layer_site": h3}


OFF_TARGET = ("O1", "O2", "O3")


def _hseed(seed: int, *parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).hexdigest()[:8], 16)


def _mean_ci(rows, value, cfg: dict[str, Any], seed: int) -> dict[str, Any] | None:
    """Pair mean of a per-trial value, cluster bootstrap within relation, 95% interval; no verdict (descriptive)."""
    vals, _, strata = _cluster_means(rows, value)
    if len(vals) < 2:
        return {"estimate": float(np.mean(vals)), "ci95": None, "n_pairs": len(vals)} if vals else None
    b_ = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(vals, n_resamples=b_["n_resamples"], seed=seed, strata=strata)
    return {"estimate": res.estimate, "ci95": list(res.ci(b_["level"])), "n_pairs": len(vals)}


def endpoint_analysis(rows, cfg: dict[str, Any], *, pk: int, seed: int) -> dict[str, Any]:
    """Confirmatory: each hybrid minus NORM on M2 at the primary k (Holm across the hybrids), with M1 alongside.
    Descriptive: per selector and k, the pair means of F, O1, O1 / O1(FULL), O2 and O3 (trials at or above the M2 floor),
    the FULL and RANDOM references, and the declared selector contrasts on O1 to O3. Unsquared."""
    hybrids = list(cfg.get("hybrids") or {})
    fam = {h: contrast(rows, h, "NORM", "M2", k=pk, cfg=cfg, seed=_hseed(seed, "hybrid", h, "M2")) for h in hybrids}
    testable = {h: c["_verdict"] for h, c in fam.items() if c is not None}
    if testable:
        stats.apply_holm(testable, alpha=cfg["analysis"]["holm_alpha"])
        for h, v in testable.items():
            fam[h].update(p_holm=v.p_adjusted, better_after_holm=v.better_after_holm)
    m1 = {h: contrast(rows, h, "NORM", "M1", k=None, cfg=cfg, seed=_hseed(seed, "hybrid", h, "M1")) for h in hybrids}
    floor = cfg["m2_min_full_gain"]
    selectors = sorted({n for r in rows for n in r["sets"]})

    def value(name: str, key: str, k: int):
        def get(r):
            if r["full_gain"] < floor:
                return None
            if name == "FULL":
                full = r["offtarget_ref"]["FULL"]
                return 1.0 if key == "F" else (_ratio(full["O1"], r) if key == "O1_ratio" else full[key])
            if name == "RANDOM":
                draws = r["random"][str(k)]
                vals = [d["F"] if key == "F" else (_ratio(d["offtarget"]["O1"], r) if key == "O1_ratio" else d["offtarget"][key]) for d in draws]
                vals = [v for v in vals if v is not None]
                return float(np.mean(vals)) if vals else None
            entry = r["sets"][name][str(k)]
            if key == "F":
                return entry["F"]
            return _ratio(entry["offtarget"]["O1"], r) if key == "O1_ratio" else entry["offtarget"][key]
        return get

    off: dict[str, Any] = {}
    for k in cfg["k_grid"]:
        per: dict[str, Any] = {}
        for name in ["FULL", *selectors, "RANDOM"]:
            per[name] = {key: _mean_ci(rows, value(name, key, k), cfg, _hseed(seed, "off", name, key, k)) for key in ("F", *OFF_TARGET, "O1_ratio")}
        contrasts = {}
        for a, b in cfg.get("offtarget_contrasts") or []:
            contrasts[f"{a} - {b}"] = {key: _mean_ci(rows, lambda r, a=a, b=b, key=key, k=k: (
                None if r["full_gain"] < floor else r["sets"][a][str(k)]["offtarget"][key] - r["sets"][b][str(k)]["offtarget"][key]),
                cfg, _hseed(seed, "offc", a, b, key, k)) for key in OFF_TARGET}
        off[str(k)] = {"selectors": per, "contrasts": contrasts}
    out = {"hybrids": {"M2": fam, "M1": m1}, "offtarget": off,
           "full_margin_parity_max": max((r["offtarget_ref"]["full_margin_parity"] for r in rows), default=0.0)}
    for c in [*fam.values(), *m1.values()]:
        if c is not None:
            c.pop("_verdict", None)
    return out


def _ratio(o1: float, r: dict[str, Any]) -> float | None:
    full = r["offtarget_ref"]["FULL"]["O1"]
    return None if not full else o1 / full


ANALYSIS_PLAN_NEW_TASKS = (
    "The new-task endpoint (E1 with E2): the primary family P2.1 to P2.3 on the new relation, as the analysis plan above. E2 family (Holm at 0.05 "
    "across the declared budgets B): JL-OUT minus LL-OUT on F at the norm budget (each selector adds features in its own order until "
    "the intended edit norm reaches B ||delta h_FULL||, no scaling; trials at or above the M2 floor where both orderings reach B), "
    "predicted above 0, three-way against the M2 SESOI. Secondary (three-way labels with the declared signs, no Holm): the declared "
    "contrasts (NORM - JL-OUT, NORM+JL-OUT - NORM, JL-NORM - NORM, JL-NORM - JL-OUT) on M2 at the primary k and on F at each budget; "
    "P2.1, P2.2 and the E2 family restricted to the cells whose dev median SAE-ALL recovery reaches the declared bar (frozen). "
    "Descriptive: shares reaching each budget, k and realized norm at the budget, F against realized norm on the fine k grid, RANDOM."
)


def sae_carried_cells(rows, cfg: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    """The cells whose median SAE-ALL recovery (m(SAE-ALL) - m_clean) / (m_FULL - m_clean), over trials at or above the
    M2 floor, reaches the declared bar. Derived on dev and frozen."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["full_gain"] >= cfg["m2_min_full_gain"] and r["full_gain"] != 0:
            by[f"{r['layer']}/{r['site']}"].append(S.recovery(r["m"]["SAE-ALL"], r["m"]["clean"], r["m"]["FULL"]))
    medians = {c: float(np.median(v)) for c, v in sorted(by.items())}
    bar = cfg["sae_carried"]["min_median_sae_all_recovery"]
    return [c for c, v in medians.items() if v >= bar], medians


def _labelled(rows, value, *, sesoi: float, cfg: dict[str, Any], seed: int, predicted_sign: int = 1) -> dict[str, Any] | None:
    """A per-trial value averaged within pairs, cluster bootstrap stratified by relation, three-way verdict (as contrast)."""
    values, _, strata = _cluster_means(rows, value)
    if len(values) < 2:
        return None
    b_ = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(values, n_resamples=b_["n_resamples"], seed=seed, strata=strata)
    v = stats.verdict(res, sesoi=sesoi, predicted_sign=predicted_sign, level=b_["level"], tost_level=b_["tost_level"])
    return {"estimate": v.estimate, "ci95": list(v.ci95), "ci90": list(v.ci90), "p": v.p_value, "label": v.label, "n_pairs": len(values),
            "better": v.better, "equivalent": v.equivalent, "worse": v.worse, "practically_better": v.practically_better, "_verdict": v}


def _budget_F(r: dict[str, Any], name: str, budget: str) -> float | None:
    e = r["norm_budget"]["sets"][name][budget]
    return e["F"] if e["reached"] else None


def budget_contrast(rows, a: str, b: str, budget: str, *, cfg: dict[str, Any], seed: int, predicted_sign: int = 1) -> dict[str, Any] | None:
    """A minus b on F at the norm budget, per trial (at or above the M2 floor, both orderings reaching the budget),
    averaged within pairs, cluster bootstrap stratified by relation, three-way against the M2 SESOI."""
    floor = cfg["m2_min_full_gain"]

    def value(r):
        if r["full_gain"] < floor:
            return None
        x, y = _budget_F(r, a, budget), _budget_F(r, b, budget)
        return None if x is None or y is None else x - y

    return _labelled(rows, value, sesoi=cfg["analysis"]["sesoi"]["M2"], cfg=cfg, seed=seed, predicted_sign=predicted_sign)


def _holm(family: dict[str, Any], alpha: float) -> None:
    testable = {k: v["_verdict"] for k, v in family.items() if v is not None}
    if testable:
        stats.apply_holm(testable, alpha=alpha)
        for k, v in testable.items():
            family[k].update(p_holm=v.p_adjusted, better_after_holm=v.better_after_holm)


def _strip(obj: Any) -> Any:
    """Drop the verdict objects from nested analysis dicts."""
    if isinstance(obj, dict):
        obj.pop("_verdict", None)
        for v in obj.values():
            _strip(v)
    return obj


def _pair_mean(rows, value) -> float | None:
    vals, _, _ = _cluster_means(rows, value)
    return float(np.mean(vals)) if vals else None


def budget_descriptive(rows, cfg: dict[str, Any], *, seed: int, tag: str) -> dict[str, Any]:
    """The new-task endpoint, descriptive, over trials at or above the M2 floor: per selector and budget, the share of trials reaching the
    budget, pair-mean F with a 95% interval on trials that reach it, mean k (active features) and mean realized
    rho = realized ||delta(S)|| / realized ||delta h_FULL||; RANDOM averaged over its walks. Curve: pair-mean F and realized
    rho at each k of the fine grid. Unsquared."""
    nb = cfg["norm_budget"]
    floor = cfg["m2_min_full_gain"]
    budgets = [str(float(b)) for b in nb["budgets"]]

    def rho(r, e):
        full = r["ref_norms"]["FULL"][1]
        return e["norms"][1] / full if full else None

    def pick(r, e, key):
        return e["F"] if key == "F" else (e["k_active"] if key == "k" else rho(r, e))

    def at(name, b, key):
        def get(r):
            if r["full_gain"] < floor:
                return None
            if name == "RANDOM":
                walks = r["norm_budget"]["random"][b]
                if key == "reached":
                    return float(np.mean([w["reached"] for w in walks])) if walks else None
                vals = [v for v in (pick(r, w, key) for w in walks if w["reached"]) if v is not None]
                return float(np.mean(vals)) if vals else None
            e = r["norm_budget"]["sets"][name][b]
            if key == "reached":
                return float(e["reached"])
            return pick(r, e, key) if e["reached"] else None
        return get

    def curve_at(name, k, key):
        def get(r):
            if r["full_gain"] < floor:
                return None
            e = r["norm_budget"]["curve"][name][k]
            return e["F"] if key == "F" else rho(r, e)
        return get

    out: dict[str, Any] = {"budget": {}, "curve": {}}
    for name in [*nb["selectors"], "RANDOM"]:
        out["budget"][name] = {b: {"reached": _pair_mean(rows, at(name, b, "reached")),
                                   "F": _mean_ci(rows, at(name, b, "F"), cfg, _hseed(seed, "budgetF", tag, name, b)),
                                   "k": _pair_mean(rows, at(name, b, "k")), "rho": _pair_mean(rows, at(name, b, "rho"))} for b in budgets}
    for name in nb["selectors"]:
        out["curve"][name] = {str(k): {"F": _pair_mean(rows, curve_at(name, str(k), "F")), "rho": _pair_mean(rows, curve_at(name, str(k), "rho"))}
                              for k in nb["curve_k"]}
    return out


def budget_analysis(rows, cfg: dict[str, Any], *, pk: int, seed: int, carried: list[str] | None) -> dict[str, Any]:
    """The new-task endpoint, as ANALYSIS_PLAN_NEW_TASKS: the E2 family with Holm, the declared secondary contrasts, the SAE-carried restriction and
    the descriptive tables."""
    nb = cfg["norm_budget"]
    budgets = [str(float(b)) for b in nb["budgets"]]
    a, b = nb["family"]

    def family(sub, tag):
        fam = {f"{a} - {b} at B = {x}": budget_contrast(sub, a, b, x, cfg=cfg, seed=_hseed(seed, tag, "E2", x)) for x in budgets}
        _holm(fam, cfg["analysis"]["holm_alpha"])
        return fam

    out: dict[str, Any] = {"family": family(rows, "all"), "secondary": {}}
    for x, y, sign in nb["secondary"]:
        entry = {f"M2 at k = {pk}": contrast(rows, x, y, "M2", k=pk, cfg=cfg, seed=_hseed(seed, "sec", x, y, "M2"), predicted_sign=sign)}
        for bb in budgets:
            entry[f"F at B = {bb}"] = budget_contrast(rows, x, y, bb, cfg=cfg, seed=_hseed(seed, "sec", x, y, bb), predicted_sign=sign)
        out["secondary"][f"{x} - {y}"] = entry
    if carried is not None:
        sub = [r for r in rows if f"{r['layer']}/{r['site']}" in carried]
        out["sae_carried"] = {"cells": list(carried), "n_trials": len(sub),
                              "P2.1": contrast(sub, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=_hseed(seed, "carried", "P2.1")),
                              "P2.2": contrast(sub, "JL-OUT", "LL-OUT", "M2", k=pk, cfg=cfg, seed=_hseed(seed, "carried", "P2.2")),
                              "family": family(sub, "carried")}
    cells = sorted({(r["layer"], r["site"]) for r in rows})
    out["descriptive"] = {"pooled": budget_descriptive(rows, cfg, seed=seed, tag="pooled"),
                          "per_cell": {f"L{layer}/{site}": budget_descriptive([r for r in rows if r["layer"] == layer and r["site"] == site], cfg,
                                                                              seed=seed, tag=f"{layer}/{site}") for layer, site in cells}}
    gaps = [abs(e["norms"][1] - e["norms"][0]) / e["norms"][0] for r in rows for per in r["norm_budget"]["sets"].values()
            for e in per.values() if e["reached"] and e["norms"][0]]
    out["max_realized_vs_intended_gap"] = max(gaps, default=0.0)
    return _strip(out)


ANALYSIS_PLAN_RANKING_VS_SETS = (
    "The ranking-versus-sets test: two active relations, each with its own format gate and eligible cells. Confirmatory "
    "ranking-versus-sets family per relation, Holm across the run's D tests: D1 ATP minus JL-OUT on M1, predicted above 0; D2 ATP "
    "minus JL-OUT on M2 at the primary k. The dissociation replicates on a relation in a run when D1 is practically better after "
    "Holm and D2 is equivalent or worse (an inconclusive D2 does not count); across checkpoints it replicates on a relation when it "
    "replicates in every run where the relation passes its gate and has an eligible cell. Secondary: budget dependence, "
    "(F_NORM,B - F_ORACLE,B) - (F_NORM,k - F_ORACLE,k) at B = 0.25 on trials where both reach B, predicted below 0 (Holm across "
    "relations). Toolkit test: per relation and eligible cell, each toolkit's best selector on dev pairs (pair-mean F at or above "
    "the M2 floor; ties by declared order), frozen; on test, F(T1 choice) - F(T0 choice), predicted at or above 0, three-way against "
    "the M2 SESOI, Holm across relations; primary at the primary k, secondary at B = 0.25. Per-relation P2.1 and P2.2, secondary."
)


def _cell(r: dict[str, Any]) -> str:
    return f"{r['layer']}/{r['site']}"


def _F_at(r: dict[str, Any], name: str, k: int | None = None, budget: str | None = None) -> float | None:
    """A set's recovered share: at k from the k grid, or at a norm budget, None when the walk never reached it."""
    if budget is None:
        e = (r["sets"].get(name) or {}).get(str(k))
        return None if e is None else e["F"]
    e = r["norm_budget"]["sets"][name][budget]
    return e["F"] if e["reached"] else None


def toolkit_choices(rows, cfg: dict[str, Any], toolkits: dict[str, list[str]], *, pk: int, budget: str | None = None,
                    by: str = "relation") -> dict[str, dict[str, dict[str, Any]]]:
    """Per relation and cell, each toolkit's selector with the highest pair-mean recovered share on these rows (trials at
    or above the M2 floor), at the primary k or at a norm budget. Ties go to the earlier selector in the toolkit's declared order."""
    floor = cfg["m2_min_full_gain"]
    names = list(dict.fromkeys(n for sels in toolkits.values() for n in sels))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rel, cell in sorted({(r[by], _cell(r)) for r in rows}):
        sub = [r for r in rows if r[by] == rel and _cell(r) == cell]
        means = {n: _pair_mean(sub, lambda r, n=n: _F_at(r, n, pk, budget) if r["full_gain"] >= floor else None) for n in names}
        pick = {t: max(sels, key=lambda n, sels=sels: (means[n] if means[n] is not None else -math.inf, -sels.index(n)))
                for t, sels in toolkits.items()}
        out.setdefault(rel, {})[cell] = {**pick, "means": means}
    return out


def derive_toolkit_choices(rows, cfg: dict[str, Any], *, pk: int) -> dict[str, Any]:
    tk = cfg["ranking_vs_sets"]["toolkit"]
    return {"k": toolkit_choices(rows, cfg, tk["sets"], pk=pk),
            "budget": toolkit_choices(rows, cfg, tk["sets"], pk=pk, budget=str(float(tk["budget"])))}


def toolkit_contrast(rows, cfg: dict[str, Any], choices: dict[str, Any], *, a: str, b: str, pk: int, budget: str | None = None,
                     seed: int, by: str = "relation") -> dict[str, Any] | None:
    """F of toolkit a's frozen choice minus toolkit b's, per trial at its relation and cell, pair means, three-way against the M2
    SESOI, predicted at or above 0."""
    floor = cfg["m2_min_full_gain"]

    def value(r):
        ch = choices.get(r[by], {}).get(_cell(r))
        if r["full_gain"] < floor or not ch:
            return None
        x, y = _F_at(r, ch[a], pk, budget), _F_at(r, ch[b], pk, budget)
        return None if x is None or y is None else x - y

    return _labelled(rows, value, sesoi=cfg["analysis"]["sesoi"]["M2"], cfg=cfg, seed=seed, predicted_sign=1)


def budget_dependence(rows, cfg: dict[str, Any], a: str, b: str, budget: str, *, pk: int, seed: int) -> dict[str, Any] | None:
    """(F_a,B - F_b,B) - (F_a,k - F_b,k) per trial at or above the M2 floor where both walks reach B; predicted below 0."""
    floor = cfg["m2_min_full_gain"]

    def value(r):
        if r["full_gain"] < floor:
            return None
        v = [_F_at(r, n, budget=budget) for n in (a, b)] + [_F_at(r, n, pk) for n in (a, b)]
        return None if None in v else (v[0] - v[1]) - (v[2] - v[3])

    return _labelled(rows, value, sesoi=cfg["analysis"]["sesoi"]["M2"], cfg=cfg, seed=seed, predicted_sign=-1)


def _replicates(d1: dict[str, Any] | None, d2: dict[str, Any] | None) -> dict[str, Any]:
    """The ranking-versus-sets test's rule: D1 practically better after Holm, and D2 equivalent or worse. An inconclusive D2 does not count."""
    if d1 is None or d2 is None:
        return {"replicates": False, "reason": "not testable"}
    ok1 = bool(d1["practically_better"] and d1.get("better_after_holm"))
    ok2 = bool(d2["equivalent"] or d2["worse"])
    reason = "D1 and D2 as predicted" if ok1 and ok2 else "; ".join(
        x for x in ("D1 not practically better after Holm" if not ok1 else "", f"D2 {d2['label']}" if not ok2 else "") if x)
    return {"replicates": ok1 and ok2, "reason": reason}


def ranking_vs_sets_analysis(rows, cfg: dict[str, Any], *, pk: int, seed: int, choices: dict[str, Any] | None) -> dict[str, Any]:
    """The ranking-versus-sets test, as ANALYSIS_PLAN_RANKING_VS_SETS. ``choices`` (the frozen toolkit choices) only on test; on dev the toolkit test is not evaluated,
    because the choices were made on those trials."""
    spec = cfg["ranking_vs_sets"]
    a, b = spec["dissociation"]
    alpha = cfg["analysis"]["holm_alpha"]
    tasks = sorted({r["relation"] for r in rows})
    by_task = {t: [r for r in rows if r["relation"] == t] for t in tasks}
    fam: dict[str, Any] = {}
    for t in tasks:
        fam[f"D1 {t}"] = contrast(by_task[t], a, b, "M1", k=None, cfg=cfg, seed=_hseed(seed, "D1", t), predicted_sign=1)
        fam[f"D2 {t}"] = contrast(by_task[t], a, b, "M2", k=pk, cfg=cfg, seed=_hseed(seed, "D2", t), predicted_sign=1)
    _holm(fam, alpha)
    replication = {t: _replicates(fam[f"D1 {t}"], fam[f"D2 {t}"]) for t in tasks}
    bd = spec["budget_dependence"]
    bud = str(float(bd["budget"]))
    dependence = {t: budget_dependence(by_task[t], cfg, bd["pair"][0], bd["pair"][1], bud, pk=pk, seed=_hseed(seed, "BD", t)) for t in tasks}
    _holm(dependence, alpha)
    out: dict[str, Any] = {"dissociation": fam, "replication": replication, "budget_dependence": dependence,
                           "per_task_primary": {t: {"P2.1": contrast(by_task[t], "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=_hseed(seed, "P21", t)),
                                                    "P2.2": contrast(by_task[t], "JL-OUT", "LL-OUT", "M2", k=pk, cfg=cfg, seed=_hseed(seed, "P22", t))}
                                                for t in tasks}}
    if choices is not None:
        tk = cfg["ranking_vs_sets"]["toolkit"]
        big, small = tk["compare"]
        toolkit = {}
        for endpoint, budget in (("k", None), ("budget", str(float(tk["budget"])))):
            per = {t: toolkit_contrast(by_task[t], cfg, choices[endpoint], a=big, b=small, pk=pk, budget=budget,
                                       seed=_hseed(seed, "TK", endpoint, t)) for t in tasks}
            _holm(per, alpha)
            toolkit[endpoint] = per
        out["toolkit"] = toolkit
    return _strip(out)


def power(rows, cfg: dict[str, Any], *, pk: int, n_test: dict[str, int], seed: int) -> dict[str, Any]:
    """6.3 dev step 8 / 5.6: SE of the primary contrasts at the planned test size (dev pairs resampled within relation)."""
    out = {}
    for name, metric, k in (("P2.1", "M1", None), ("P2.2", "M2", pk)):
        def value(r, metric=metric, k=k):
            if metric == "M1":
                x, y = r["metrics"]["JL-OUT"]["M1"], r["metrics"]["LL-OUT"]["M1"]
            else:
                if r["full_gain"] < cfg["m2_min_full_gain"]:
                    return None
                x, y = r["sets"]["JL-OUT"][str(k)]["F"], r["sets"]["LL-OUT"][str(k)]["F"]
            return None if x is None or y is None else x - y
        values, pairs, strata = _cluster_means(rows, value)
        if len(values) < 2:
            out[name] = None
            continue
        rng = np.random.default_rng(seed)
        by_rel: dict[str, list[float]] = defaultdict(list)
        for v, s in zip(values, strata):
            by_rel[s].append(v)
        sims = []
        for _ in range(cfg["analysis"]["bootstrap"]["n_resamples"]):
            draws = [rng.choice(by_rel[r], size=n_test.get(r, len(by_rel[r])), replace=True) for r in by_rel]
            sims.append(float(np.mean(np.concatenate(draws))))
        se = float(np.std(sims, ddof=1))
        est = float(np.mean(values))
        sesoi = cfg["analysis"]["sesoi"]["M1" if metric == "M1" else "M2"]
        out[name] = {"dev_estimate": est, "se_at_test_size": se, "n_test_pairs": n_test,
                     "power_to_exclude_zero_at_dev_estimate": stats.power_to_exclude_zero(est, se, alpha=0.05) if se > 0 else None,
                     "power_equivalence_if_zero": stats.power_equivalence(se, sesoi=sesoi, alpha=0.05) if se > 0 else None,
                     "min_detectable_effect": stats.min_detectable_effect(se, alpha=0.05, power=0.8)}
    return out


# --------------------------------------------------------------------------
# Lens-fidelity diagnostic (6.3 dev step 4)
# --------------------------------------------------------------------------


def lens_fidelity(handle, lens, trials: list[Trial], cache: PromptCache, layers: list[int], *, causal: str, grad_mode: str,
                  checkpointed: bool, strategy: str, batch_size: int, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Random unit directions at the readout: logit-lens, J-lens and local-gradient predictions of the change
    in m, against a central finite difference in the causal mode."""
    lf = cfg["lens_fidelity"]
    chosen = sorted(trials, key=lambda t: (hash_unit(f"lf:{t.key}", seed), t.key))[: lf["n_prompts"]]
    out = {}
    for layer in layers:
        J = lens.J(layer, handle.device)
        gen = torch.Generator().manual_seed(seed + layer)
        V = torch.randn(lf["n_directions"], handle.d_model, generator=gen)
        V = (V / V.norm(dim=1, keepdim=True)).to(handle.device)
        rows = {"LL": [], "JL": [], "local": [], "fd": []}
        for t in chosen:
            src = cache.get(t.source)
            r = t.source.meta["readout_position"]
            h_f = src["h_final"].to(handle.device)
            w = output_direction(h_f, handle.head, t.answers)
            s_ll = L.lens_sensitivity(w, h_f, handle.head, None)
            s_jl = L.lens_sensitivity(w, h_f, handle.head, J)
            _, grad = margin_gradient(handle, as_ids(src["ids"], handle.device), layer, t.answers, readout=r, mode=grad_mode, checkpointed=checkpointed)
            step = lf["step_fraction"] * float(src["acts"][(layer, "readout")].norm())
            ev = PatchEvaluator(handle, as_ids(src["ids"], handle.device), t.answers, readout=r, mode=causal, strategy=strategy, batch_size=batch_size)
            plus = ev.evaluate([[EditSpec(layer=layer, positions=[r], delta=(step * v).unsqueeze(0))] for v in V])
            minus = ev.evaluate([[EditSpec(layer=layer, positions=[r], delta=(-step * v).unsqueeze(0))] for v in V])
            rows["fd"] += [(p.margin - q.margin) / (2 * step) for p, q in zip(plus, minus)]
            rows["LL"] += (V @ s_ll).tolist()
            rows["JL"] += (V @ s_jl).tolist()
            rows["local"] += (V @ grad[r].to(handle.device)).tolist()
        fd = np.array(rows["fd"])
        out[str(layer)] = {name: {"pearson": stats.pearson(np.array(rows[name]), fd), "spearman": stats.spearman(rows[name], fd)}
                           for name in ("LL", "JL", "local")}
        out[str(layer)]["n"] = len(fd)
        del J
    return out


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def _load(data_dir: Path, family: str, phase: str):
    card = json.loads((data_dir / family / "benchmark_card.json").read_text(encoding="utf-8"))
    return card, load_records(data_dir / family / card["records_file"], phase=phase, expected_sha256=card["records_sha256"])


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """A config with ``base:`` is the base config (itself resolved) with its keys overridden (extensions, 7.2)."""
    if "base" not in cfg:
        return cfg
    from saeselect.config import load_config

    return {**_resolve(load_config(cfg["base"])), **{k: v for k, v in cfg.items() if k != "base"}}


def _sae_plan(ext: dict[str, Any] | None, s0: dict[str, Any], s0_dir: str, variants: dict[str, Any] | None, layer: int) -> list[tuple[str, dict[str, Any], Path]]:
    """(tag, SAE entry, feature-statistics directory) per SAE to run at ``layer``: the
    Stage 0 SAE, or for an extension the declared variants at that layer (X4)."""
    if not ext or variants is None:
        u = s0["sae"][str(layer)]
        return [(u["key"], u, Path(s0_dir) / s0["features_dir"])]
    return [(key, u, Path(ext["sae_run"]) / variants["features_dir"]) for key, u in sorted(variants["sae_variants"].items()) if int(u["layer"]) == layer]


def run(ctx: RunContext) -> Outcome:
    cfg = _resolve(ctx.config)
    if ctx.phase not in ("dev", "test"):
        raise RunContractError("stage2 needs --phase dev or --phase test")
    model_cfg = ctx.model_config if ctx.model_config is not None else (ctx.frozen or {}).get("model")
    if model_cfg is None:
        raise RunContractError("stage2 needs --model configs/models/<model>.yaml")
    if ctx.phase == "test":
        frozen_cfg = ctx.frozen.get("config_resolved", ctx.frozen["config"])
        if json_hash(cfg) != json_hash(frozen_cfg) or json_hash(model_cfg) != json_hash(ctx.frozen["model"]):
            raise RunContractError("--config or --model differs from the frozen protocol; a change needs a new protocol")
    name = model_cfg["name"]
    run_dir = (cfg.get("stage0_runs") or {}).get(name)
    if not run_dir:
        raise EngineeringFailed("stage0_missing", f"no stage0_runs entry for {name}")
    s0 = ctx.load_upstream(Path(run_dir) / "stage0_decisions.json")
    if s0.get("model") != name or s0.get("scope", "full") != "full":
        raise EngineeringFailed("stage0_missing", f"{run_dir} is not a full Stage 0 run of {name}")
    # 7.2 extensions: the parent's frozen protocol with only the named factor changed; each gets its own dev pass,
    # freeze and single test run. X4 (kind "sae"): the SAE (declared variants, measured by a Stage 0 run with scope
    # sae_variants) at the declared cells; relations, templates, primary k and divergence are inherited. X3 (kind
    # "model"): another checkpoint with its own Stage 0 SAEs at the parent's eligible cells; relations, templates and
    # primary k are inherited, while the lens divergence and lens fidelity, properties of the model's own lens, are
    # derived on its dev pass. A cell whose layer has no usable SAE in the model's Stage 0 is dropped on dev, flagged.
    ext = cfg.get("extension")
    model_ext = bool(ext) and ext.get("kind") == "model"
    endpoint_ext = bool(ext) and ext.get("kind") == "endpoint"  # the parent's model and SAEs, with extra readouts and selectors
    if endpoint_ext and not cfg.get("offtarget"):
        raise RunContractError("an endpoint extension declares `offtarget: true`")
    if cfg.get("ranking_vs_sets"):  # The ranking-versus-sets test reads its budget from the new-task endpoint walks
        need = {str(float(cfg["ranking_vs_sets"]["budget_dependence"]["budget"])), str(float(cfg["ranking_vs_sets"]["toolkit"]["budget"]))}
        have = {str(float(b)) for b in (cfg.get("norm_budget") or {}).get("budgets", [])}
        if not need <= have:
            raise EngineeringFailed("config", f"The ranking-versus-sets test's budgets {sorted(need)} are not among norm_budget.budgets {sorted(have)}")
    parent: dict[str, Any] = {}
    variants = None
    ext_cells: list[str] = []
    if ext:
        parent = ctx.load_upstream(ext["parent_frozen"]) if ctx.phase == "dev" else dict(ctx.frozen)
        if not model_ext and not endpoint_ext:
            variants = ctx.load_upstream(Path(ext["sae_run"]) / "stage0_decisions.json")
            if variants.get("model") != name or variants.get("scope") != "sae_variants":
                raise EngineeringFailed("stage0_missing", f"{ext['sae_run']} is not a Stage 0 sae_variants run of {name}")
            if ctx.phase == "test" and json_hash(variants["sae_variants"]) != json_hash(ctx.frozen["extension"]["variants"]):
                raise RunContractError(f"{ext['sae_run']} no longer matches the SAE variants frozen in the protocol")
        ext_cells = list(ext.get("cells") or parent["eligible_cells"])
        unknown = [c for c in ext_cells if c not in parent["eligible_cells"]]
        if unknown:
            raise EngineeringFailed("config", f"extension cells {unknown} are not eligible cells of the parent protocol {parent['eligible_cells']}")
        if model_ext and ctx.phase == "dev":
            missing = [c for c in ext_cells if c.split("/")[0] not in s0["sae"]]
            if missing:
                ctx.flags.append(f"{name}: no usable SAE at the layer of {missing} in {run_dir}; those inherited cells are not evaluated")
                ext_cells = [c for c in ext_cells if c not in missing]
            if not ext_cells:
                raise GateFailed("eligibility", f"none of the parent's eligible cells {parent['eligible_cells']} has a usable SAE in {run_dir}")
        layers = sorted({int(c.split("/")[0]) for c in ext_cells})
    else:
        layers = sorted(int(k) for k in s0["sae"])
    causal, grad_mode = s0.get("causal_mode", s0["precision_mode"]), s0.get("gradient_mode", s0["precision_mode"])
    checkpointed = s0.get("backward") == "checkpointed"
    # 8.8: batching (N9) and suffix caching (N8) only where their parity checks passed; both together when both did,
    # with every trial probing the chosen strategy against serial full forwards (rule 8)
    strategy = {(True, True): "batched_suffix", (True, False): "batched", (False, True): "suffix", (False, False): "serial"}[
        (bool(s0["batching_allowed"]), bool(s0["suffix_caching_allowed"]))]
    batch_size = int(s0.get("batch_size", 8))
    seed = cfg["seed"]
    data_dir = Path(cfg["data_dir"])
    frozen = ctx.frozen or {}
    inherited = parent if ext else frozen  # relations, eligible cells, primary k and divergence fixed upstream

    handle = load_model(model_cfg, device=ctx.device)
    lens = A.load_lens_for(model_cfg, d_model=handle.d_model, n_layers=handle.n_layers)
    cache = PromptCache(handle, layers, causal)
    store = CheckpointStore(ctx.run.path)
    report: dict[str, Any] = {"phase": ctx.phase, "model": name, "stage0_run": run_dir, "layers": layers, "causal_mode": causal,
                              "gradient_mode": grad_mode, "strategy": strategy}

    # ---- step 1: format gate (dev) or answerability (test) ---------------------------------
    rel_cfg = cfg["relations"]
    reserve = rel_cfg.get("reserve")  # optional
    candidates = list(inherited["relations"]) if (ctx.phase == "test" or ext) else [*rel_cfg["active"], *([reserve] if reserve else [])]
    trials_by_rel: dict[str, list[Trial]] = {}
    gates: dict[str, Any] = {}
    cards: dict[str, Any] = {}
    for family in candidates:
        card, records = _load(data_dir, family, ctx.phase)
        cards[family] = card
        trials = build_trials(records, family)
        pairs = sorted({t.pair for t in trials}, key=lambda p: (hash_unit(f"s2:{p}", seed), p))
        if ctx.limit is not None:
            keep = set(pairs[: ctx.limit])
            trials = [t for t in trials if t.pair in keep]
        for t in trials:
            src, don = cache.get(t.source), cache.get(t.donor)
            p_s, m_s, m_src = set_masses(src["h_final"].to(handle.device), handle.head, t.answers)
            p_d, m_d, m_don = set_masses(don["h_final"].to(handle.device), handle.head, t.answers)
            t.readout = {"m_clean": m_src, "m_don": m_don, "src_own_mass": m_s, "don_own_mass": p_d}
        gates[family] = format_gate(trials, cfg)
        trials_by_rel[family] = trials
    if ctx.phase == "dev" and ext:
        relations = {f: list(tpls) for f, tpls in inherited["relations"].items()}  # inherited (7.2); the gate is reported
    elif ctx.phase == "dev":
        active = [f for f in rel_cfg["active"] if gates[f]["pass"]]
        failed = [f for f in rel_cfg["active"] if not gates[f]["pass"]]
        if failed and reserve and gates[reserve]["pass"]:
            active.append(reserve)
            ctx.flags.append(f"relation(s) {failed} failed the format gate; the reserve {reserve} replaces one (6.3, pre-declared swap)")
        elif failed and not reserve:
            ctx.flags.append(f"relation(s) {failed} failed the format gate and are dropped (no reserve)")
        small = [f for f in active if (cards[f].get("minimum") or {}).get("met") is False]
        if small:  # 6.3's minimum of 12 dev and 40 test pairs (R6 was built as a reserve, which the build does not hold to it)
            ctx.flags.append(f"relation(s) {small} have fewer pairs than the 6.3 minimum and are dropped")
            active = [f for f in active if f not in small]
        relations = {f: gates[f]["passing_templates"] for f in active[: len(rel_cfg["active"])]}
    else:
        relations = {}
        for f, tpls in frozen["relations"].items():
            answerable = [t.answerable for t in trials_by_rel[f] if t.template in tpls]
            share = float(np.mean(answerable)) if answerable else 0.0
            gates[f]["test_answerable_share"] = share
            if share >= cfg["format_gate"]["test_min_answerable"]:
                relations[f] = tpls
            else:
                ctx.flags.append(f"{f}: {share:.0%} of test trials answerable (< {cfg['format_gate']['test_min_answerable']:.0%}); the relation is reported as failed (6.3)")
    report["format_gate"] = gates
    report["relations_used"] = relations
    if not relations:
        raise GateFailed("format", "no relation passes the format gate", [json.dumps(gates, default=_jsonable)[:3000]], report=report)
    trials = [t for f, tpls in relations.items() for t in trials_by_rel[f] if t.template in tpls and t.answerable]
    groups: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for t in trials:
        groups[(t.pair, t.direction)].append(t)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}  # a trial needs at least one support trial
    trials = [t for v in groups.values() for t in v]
    report["n_trials"] = len(trials)
    report["n_pairs"] = len({t.pair for t in trials})

    # ---- step 2: cell eligibility (R_FULL) --------------------------------------------------
    sites = cfg["sites"]
    per_rel = bool(cfg["eligibility"].get("per_relation"))  # each relation's own eligible cells
    r_full: dict[str, list[float]] = defaultdict(list)
    r_full_rel: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        src, don = cache.get(t.source), cache.get(t.donor)
        for layer in layers:
            for site in sites:
                d = don["acts"][(layer, site)] - src["acts"][(layer, site)]
                ev = PatchEvaluator(handle, as_ids(src["ids"], handle.device), t.answers, readout=t.source.meta["readout_position"], mode=causal, strategy="serial")
                m_full = ev.evaluate([[EditSpec(layer=layer, positions=src["sites"][site], delta=d)]])[0].margin
                gain = t.readout["m_don"] - t.readout["m_clean"]
                r_full[f"{layer}/{site}"].append((m_full - t.readout["m_clean"]) / gain if gain != 0 else float("nan"))
                r_full_rel[t.relation][f"{layer}/{site}"].append(r_full[f"{layer}/{site}"][-1])
    medians = {c: float(np.nanmedian(v)) for c, v in r_full.items()}
    medians_rel = {f: {c: float(np.nanmedian(v)) for c, v in d.items()} for f, d in sorted(r_full_rel.items())} if per_rel else {}
    bar = cfg["eligibility"]["min_median_r_full"]
    eligible_by_rel: dict[str, list[str]] | None = None
    if ctx.phase == "dev" and ext:
        eligible = list(ext_cells)  # eligible in the parent protocol (checked above)
        if per_rel:
            eligible_by_rel = {f: [c for c in parent["eligible_by_relation"].get(f, []) if c in eligible] for f in relations}
        low = [c for c in eligible if not medians.get(c, float("nan")) >= cfg["eligibility"]["min_median_r_full"]]
        if model_ext and low:
            ctx.flags.append(f"{name}: inherited cells {low} have median R_FULL below {cfg['eligibility']['min_median_r_full']} "
                             "on this model's dev trials (reported, kept)")
    elif ctx.phase == "dev" and per_rel:
        eligible_by_rel = {f: [c for c, v in medians_rel.get(f, {}).items() if v >= bar] for f in relations}
        for f, per in medians_rel.items():
            for c, v in per.items():
                ctx.derive(f"cell_eligible.{name}.{f}.{c}", c in eligible_by_rel.get(f, []), procedure="The ranking-versus-sets test: median R_FULL over the "
                           "relation's dev trials >= 0.5 (6.3 dev step 2, per relation)", inputs={"model": name, "relation": f, "cell": c,
                           "n_trials": len(r_full_rel[f][c])}, numbers={"median_r_full": v})
        eligible = [c for c in medians if any(c in v for v in eligible_by_rel.values())]
    elif ctx.phase == "dev":
        eligible = [c for c, v in medians.items() if v >= cfg["eligibility"]["min_median_r_full"]]
        for c in medians:
            ctx.derive(f"cell_eligible.{name}.{c}", c in eligible, procedure="6.3 dev step 2: median R_FULL over dev trials >= 0.5",
                       inputs={"model": name, "cell": c, "n_trials": len(r_full[c])}, numbers={"median_r_full": medians[c]})
    else:
        eligible = list(frozen["eligible_cells"])
        eligible_by_rel = frozen.get("eligible_by_relation")
    report["r_full_median"] = medians
    report["eligible_cells"] = eligible
    if eligible_by_rel is not None:
        dropped = [f for f, cells in eligible_by_rel.items() if not cells]
        if dropped:
            ctx.flags.append(f"relation(s) {dropped} have no eligible cell and are not evaluated")
        report["r_full_median_by_relation"] = medians_rel
        report["eligible_by_relation"] = eligible_by_rel
    if not eligible:
        raise GateFailed("eligibility", "no layer-site cell reaches median R_FULL >= 0.5 on dev (6.3 step 2)", report=report)

    # ---- step 5: the full pipeline on eligible cells ----------------------------------------
    pk_frozen = inherited.get("primary_k")
    rows: list[dict[str, Any]] = []
    parity_max = 0.0
    sanity_max = 0.0
    done = store.load_all()
    timings: dict[str, float] = {}
    for layer in layers:
        cell_sites = [s for s in sites if f"{layer}/{s}" in eligible]
        if not cell_sites:
            continue
        from safetensors.torch import load_file

        J = lens.J(layer, handle.device)
        bank = L.AtomBank(handle.head, J, handle.device)
        t0 = time.time()
        for tag, u, feat_dir in _sae_plan(ext if variants is not None else None, s0, run_dir, variants, layer):
            sae = A.load_sae_for(model_cfg, layer=layer, width=u["width"], l0=u["l0"], d_model=handle.d_model, suffix=u.get("suffix", ""), device=ctx.device,
                                 convention=u.get("convention"), repo_id=u.get("repo_id"), revision=u.get("revision"))
            feats = load_file(str(feat_dir / f"{u['key']}.safetensors"))
            for site in cell_sites:
                cx = CellContext(handle=handle, sae=sae, layer=layer, site=site, J=J, sd=feats["sd"].to(ctx.device), kappa=feats["kappa_jlens"].cpu(),
                                 jspace={}, causal=causal, grad_mode=grad_mode, checkpointed=checkpointed, strategy=strategy,
                                 batch_size=batch_size, cfg=cfg, seed=seed)
                by_pair: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for key in groups:
                    by_pair[key[0]].append(key)
                for pair in sorted(by_pair):
                    if eligible_by_rel is not None and f"{layer}/{site}" not in eligible_by_rel.get(groups[by_pair[pair][0]][0].relation, []):
                        continue  # not an eligible cell of this pair's relation
                    ckey = f"{tag}|L{layer}|{site}|{pair}" if variants is not None else f"L{layer}|{site}|{pair}"
                    if ckey in done:
                        rows += done[ckey]["rows"]
                        continue
                    pair_rows = []
                    for key in sorted(by_pair[pair]):
                        passes = []
                        for t in sorted(groups[key], key=lambda x: x.template):
                            own = trial_pass(cx, t, cache)
                            own["ids"] = cache.get(t.source)["ids"]
                            own["full_delta"] = cache.get(t.donor)["acts"][(layer, site)] - cache.get(t.source)["acts"][(layer, site)]
                            passes.append(own)
                        _jspace(cx, bank, sorted({f for p in passes for f in p["pool"]}))
                        for i, own in enumerate(passes):
                            support = [p for j, p in enumerate(passes) if j != i]
                            row = evaluate_trial(cx, own, support, ks=cfg["k_grid"], alphas=cfg["alpha_grid"] if pk_frozen else None, primary_k=pk_frozen)
                            if variants is not None:
                                row["sae"] = tag
                            pair_rows.append(row)
                    parity = max((r["parity"] for r in pair_rows), default=0.0)
                    sanity = max((r["sanity_rel"] for r in pair_rows), default=0.0)
                    parity_max, sanity_max = max(parity_max, parity), max(sanity_max, sanity)
                    checks = {"parity": parity <= cfg["parity_probe"]["max_abs_margin_diff"], "sae_all_plus_err_equals_full": sanity <= 1e-3}
                    clean = json.loads(json.dumps(pair_rows, default=_jsonable))
                    store.save(ckey, clean, checks)
                    if not all(checks.values()):
                        raise EngineeringFailed("parity" if not checks["parity"] else "sanity",
                                                f"{ckey}: batched-serial parity {parity:.4g}, (SAE-ALL + ERR - FULL) relative {sanity:.3g}")
                    rows += clean
            del sae, feats
            _free()
        timings[str(layer)] = time.time() - t0
        del J, bank
        _free()
    report["timings_seconds"] = timings
    report["parity_max"] = parity_max
    report["sanity_max_relative"] = sanity_max

    # ---- steps 3, 4, 6, 7, 8 (dev) or the test analysis ----------------------------------------
    full_dev = ctx.phase == "dev" and not ext
    own_lens = ctx.phase == "dev" and (not ext or model_ext)  # the lens belongs to the model: its divergence is derived on its own dev
    divergence = lens_divergence(rows) if own_lens else {int(k): v for k, v in inherited["divergence"].items()}
    if own_lens:
        for layer, dv in divergence.items():
            ctx.derive(f"lens_divergence.{name}.L{layer}", dv, procedure="6.3 dev step 3: 1 - Spearman(JL-OUT, LL-OUT) over dev candidates, readout site",
                       inputs={"model": name, "layer": layer}, numbers={})
    if full_dev:
        pk, medians_oracle = primary_k(rows, cfg)
        ctx.derive(f"primary_k.{name}", pk, procedure="6.3 dev step 6: smallest k with median ORACLE recovery >= 0.5 over dev trials "
                   "(FULL gain >= 1.0), pooled over eligible cells; 32 if none", inputs={"model": name, "cells": eligible},
                   numbers={"median_oracle_F_by_k": medians_oracle})
        report["primary_k_medians"] = medians_oracle
        # alpha-curves at the primary k (dev): a second pass over the eligible cells
        rows = _alpha_pass(ctx, handle, lens, s0, run_dir, model_cfg, groups, cache, rows, pk, eligible, layers, sites, causal, grad_mode,
                           checkpointed, strategy, batch_size, cfg, seed, store)
    else:
        pk = int(pk_frozen)
    if own_lens:
        report["lens_fidelity"] = lens_fidelity(handle, lens, [t for t in trials], cache, layers, causal=causal, grad_mode=grad_mode,
                                                checkpointed=checkpointed, strategy=strategy, batch_size=batch_size, cfg=cfg, seed=seed)
    report["primary_k"] = pk
    report["divergence"] = {str(k): v for k, v in divergence.items()}
    if ext:
        # the immediate parent: on dev the protocol extended (whatever its own lineage), on test the one the frozen protocol names
        parent_hash = parent.get("protocol_hash") if ctx.phase == "dev" else parent.get("parent_protocol_hash")
        report["extension"] = {"id": ext["id"], "cells": eligible, "parent_protocol_hash": parent_hash}
        if model_ext:
            report["extension"].update(kind="model", model=name)
        elif endpoint_ext:
            report["extension"].update(kind="endpoint", hybrids=cfg.get("hybrids") or {})
        else:
            report["extension"].update(sae_run=ext["sae_run"], variants=variants["sae_variants"])
    if variants is not None:  # one analysis per SAE variant, each on the same trials (7.2); no pooling across variants
        report["by_sae"] = {}
        for tag in sorted({r["sae"] for r in rows}):
            sub = [r for r in rows if r["sae"] == tag]
            entry = {"summary": summarize(sub, cfg, pk=pk, seed=seed), "analysis": analyze(sub, cfg, pk=pk, divergence=divergence, seed=seed)}
            entry["sanity"] = sanity_checks(entry["summary"], pk)
            if ctx.phase == "dev":
                entry["power"] = power(sub, cfg, pk=pk, n_test={f: cards[f]["counts"]["test_pairs"] for f in relations}, seed=seed)
            report["by_sae"][tag] = entry
    else:
        report["summary"] = summarize(rows, cfg, pk=pk, seed=seed)
        report["analysis"] = analyze(rows, cfg, pk=pk, divergence=divergence, seed=seed)
        report["sanity"] = sanity_checks(report["summary"], pk)
        if ctx.phase == "dev":
            n_test = {f: cards[f]["counts"]["test_pairs"] for f in relations}
            report["power"] = power(rows, cfg, pk=pk, n_test=n_test, seed=seed)
        if endpoint_ext:
            report["endpoint"] = endpoint_analysis(rows, cfg, pk=pk, seed=seed)
            worst = report["endpoint"]["full_margin_parity_max"]
            if worst > cfg["parity_probe"]["max_abs_margin_diff"]:  # the off-target evaluator must give the trial's own FULL margin
                raise EngineeringFailed("parity", f"FULL margin through the off-target endpoint readout differs from the stored one by {worst:.3g}",
                                        report=json.loads(json.dumps(report, default=_jsonable)))
    carried = None
    if cfg.get("sae_carried") and variants is None:  # the cells where SAE-ALL carries the transfer, derived on dev, frozen
        if ctx.phase == "dev":
            carried, carried_medians = sae_carried_cells(rows, cfg)
            for c, v in carried_medians.items():
                ctx.derive(f"sae_carried.{name}.{c}", c in carried, procedure="The new-task endpoint: median SAE-ALL recovery over dev trials at or above "
                           "the M2 floor reaches sae_carried.min_median_sae_all_recovery", inputs={"model": name, "cell": c},
                           numbers={"median_sae_all_recovery": v})
            report["sae_carried"] = {"cells": carried, "dev_median_sae_all_recovery": carried_medians}
        else:
            carried = list(frozen["sae_carried_cells"])
            report["sae_carried"] = {"cells": carried}
    if cfg.get("norm_budget") and variants is None:  # The new-task endpoint (E2)
        report["norm_budget"] = budget_analysis(rows, cfg, pk=pk, seed=seed, carried=carried)
    choices = None
    if cfg.get("ranking_vs_sets") and variants is None:  # the ranking-versus-sets family, budget dependence and the toolkit test
        if ctx.phase == "dev":
            choices = derive_toolkit_choices(rows, cfg, pk=pk)
            for endpoint, per_rel in choices.items():
                for f, per_cell in per_rel.items():
                    for c, ch in per_cell.items():
                        ctx.derive(f"toolkit.{name}.{endpoint}.{f}.{c}", {t: ch[t] for t in cfg["ranking_vs_sets"]["toolkit"]["sets"]},
                                   procedure="The ranking-versus-sets test: each toolkit's selector with the highest pair-mean recovered share on dev trials at or "
                                   "above the M2 floor (ties by the declared order)", inputs={"model": name, "relation": f, "cell": c,
                                   "endpoint": endpoint}, numbers={"pair_means": ch["means"]})
        else:
            choices = frozen["toolkit_choices"]
        report["toolkit_choices"] = choices
        report["ranking_vs_sets"] = ranking_vs_sets_analysis(rows, cfg, pk=pk, seed=seed, choices=choices if ctx.phase == "test" else None)
    write_results_zip(ctx.run.path, {"trials.jsonl": "".join(json.dumps(r, sort_keys=True, default=_jsonable) + "\n" for r in rows)})
    report = json.loads(json.dumps(report, default=_jsonable))
    summary = _extension_summary(ctx, report) if variants is not None else _summary(ctx, report)
    if endpoint_ext:
        summary += _endpoint_summary(report, cfg, phase=ctx.phase)
    if report.get("norm_budget"):
        summary += _budget_summary(report, cfg, phase=ctx.phase)
    if report.get("ranking_vs_sets"):
        summary += _a38_summary(report, cfg, phase=ctx.phase)
    send_back = ["derived.json", "decisions_entry.md", "results.zip"]
    if ctx.phase == "dev":
        protocol = {
            "stage": "stage2", "smoke": ctx.smoke, "code_commit": ctx.git.get("commit"), "source_run": ctx.run.path.as_posix(), "created_at": utc_now(),
            "config": ctx.config, "config_resolved": cfg, "model": model_cfg, "stage0_run": run_dir, "relations": relations, "eligible_cells": eligible,
            "primary_k": pk, "divergence": {str(k): v for k, v in divergence.items()}, "data": {f: cards[f]["records_sha256"] for f in relations},
            "analysis_plan": ANALYSIS_PLAN,
        }
        if ext:
            protocol["extension"] = report["extension"]
            protocol["parent_protocol_hash"] = parent.get("protocol_hash")
        if carried is not None:
            protocol["sae_carried_cells"] = carried
        if eligible_by_rel is not None:
            protocol["eligible_by_relation"] = eligible_by_rel
        if choices is not None:
            protocol["toolkit_choices"] = choices
        if cfg.get("norm_budget"):
            protocol["analysis_plan_new_tasks"] = ANALYSIS_PLAN_NEW_TASKS
        if cfg.get("ranking_vs_sets"):
            protocol["analysis_plan_ranking_vs_sets"] = ANALYSIS_PLAN_RANKING_VS_SETS
        frozen_protocol = seal_protocol(protocol)
        ctx.run.write_json("frozen_protocol.json", frozen_protocol)
        report["frozen_protocol_hash"] = frozen_protocol["protocol_hash"]
        return Outcome(status="FROZEN_AWAITING_APPROVAL", summary=summary, report=report, send_back=["frozen_protocol.json", *send_back])
    return Outcome(status="PASS", summary=summary, report=report, send_back=send_back)


def _alpha_pass(ctx, handle, lens, s0, run_dir, model_cfg, groups, cache, rows, pk, eligible, layers, sites, causal, grad_mode,
                checkpointed, strategy, batch_size, cfg, seed, store) -> list[dict[str, Any]]:
    """Dev: the alpha-curves at the derived primary k, for every trial of the main pass."""
    from safetensors.torch import load_file

    by_trial_cell = {(r["trial"], r["layer"], r["site"]): r for r in rows}
    for layer in layers:
        cell_sites = [s for s in sites if f"{layer}/{s}" in eligible]
        if not cell_sites:
            continue
        u = s0["sae"][str(layer)]
        sae = A.load_sae_for(model_cfg, layer=layer, width=u["width"], l0=u["l0"], d_model=handle.d_model, suffix=u.get("suffix", ""), device=ctx.device,
                                 convention=u.get("convention"))
        feats = load_file(str(Path(run_dir) / s0["features_dir"] / f"{u['key']}.safetensors"))
        for site in cell_sites:
            cx = CellContext(handle=handle, sae=sae, layer=layer, site=site, J=None, sd=feats["sd"].to(ctx.device), kappa=feats["kappa_jlens"].cpu(),
                             jspace={}, causal=causal, grad_mode=grad_mode, checkpointed=checkpointed, strategy=strategy, batch_size=batch_size,
                             cfg=cfg, seed=seed)
            for key, trials in sorted(groups.items()):
                for t in trials:
                    row = by_trial_cell.get((t.key, layer, site))
                    if row is None or "alpha" in row:
                        continue
                    src, don = cache.get(t.source), cache.get(t.donor)
                    h_src, h_don = src["acts"][(layer, site)].to(sae.device), don["acts"][(layer, site)].to(sae.device)
                    da_full = sae.encode(h_don) - sae.encode(h_src)
                    pool = candidate_pool(da_full)
                    own = {"trial": t, "pool": pool.tolist(), "da": da_full.index_select(1, pool).cpu(), "W_rows": pool.cpu(),
                           "ids": src["ids"], "positions": src["sites"][site], "m": row["m"]}
                    row["alpha"] = alpha_curves(cx, own, row["sets"], row["random"], pk, cfg["alpha_grid"])
        del sae, feats
        _free()
    return rows


def sanity_checks(summary: dict[str, Any], pk: int) -> dict[str, Any]:
    """6.3 dev step 7: mean ORACLE recovery at the primary k above every selector's; RANDOM below every sensitivity-based one."""
    pooled = summary["pooled"]
    oracle = (pooled.get("ORACLE") or {}).get("F", {}).get(str(pk))
    rnd = summary.get("random_mean_F_at_primary_k")
    above = {n: v["F"].get(str(pk)) for n, v in pooled.items() if n != "ORACLE" and v["F"].get(str(pk)) is not None}
    violations = [f"{n} F {v:.3f} >= ORACLE {oracle:.3f}" for n, v in above.items() if oracle is not None and v >= oracle]
    violations += [f"RANDOM mean F {rnd:.3f} >= {n} {above[n]:.3f}" for n in S.SENSITIVITY_BASED if n in above and rnd is not None and rnd >= above[n]]
    return {"oracle_F": oracle, "random_mean_F": rnd, "violations": violations}


# --------------------------------------------------------------------------
# SUMMARY.md
# --------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:+.{digits}f}"
    return str(v)


def _share(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _ci(c: dict[str, Any] | None) -> str:
    if not c:
        return "not testable"
    return f"{c['estimate']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {c['label']}" + (
        f", Holm p {c['p_holm']:.3g}" if c.get("p_holm") is not None else "")


def _summary(ctx: RunContext, report: dict[str, Any]) -> str:
    lines = ["## What ran", "",
             f"Stage 2 {ctx.phase} phase on `{report['model']}`: causal measurements in {report['causal_mode']}, "
             f"gradients in {report['gradient_mode']}, patches {report['strategy']}. M1 = Spearman of a signal's scores with single-feature "
             "transfer effects over the evaluation pool; M2 = F_k = (m(S_k) - m_clean) / (m_FULL - m_clean). Unsquared.", ""]
    ext = report.get("extension") or {}
    if ext.get("kind") == "model":
        lines += [f"Extension {ext['id']}: the parent protocol ({str(ext['parent_protocol_hash'])[:12]}...) with only the "
                  "model changed. Relations, templates, eligible cells and primary k are inherited; the SAEs are this model's Stage 0 "
                  "selections, and the lens divergence and lens fidelity are this model's own (derived on its dev pass).", ""]
    elif ext.get("kind") == "endpoint":
        lines += [f"Extension {ext['id']}: the parent protocol ({str(ext['parent_protocol_hash'])[:12]}...) with the same model and "
                  "SAEs; relations, templates, eligible cells, primary k and divergence are inherited. Added: every evaluated set's "
                  "off-target readout (O1 to O3) from the same forward pass, and the hybrid selectors "
                  + ", ".join(ext.get("hybrids") or {}) + " (rank average, ties by NORM). The off-target endpoint tables follow the standard ones.", ""]
    lines += ["## Format gate", "", "| Relation | Templates passing (share of pair-directions) | Answerable | Used |", "|---|---|---|---|"]
    for f, g in report["format_gate"].items():
        tpl = ", ".join(f"{t} {v['share']:.0%}{'' if v['pass'] else ' (fail)'}" for t, v in g["templates"].items())
        lines.append(f"| {f} | {tpl} | {g.get('test_answerable_share', g['answerable_share']):.0%} | {'yes' if f in report['relations_used'] else 'no'} |")
    lines += ["", f"Trials: {report['n_trials']} in {report['n_pairs']} concept pairs (each with at least one support trial).", "",
              "## Cells", "", "| Cell | Median R_FULL | Eligible |", "|---|---|---|"]
    for c, v in sorted(report["r_full_median"].items()):
        lines.append(f"| {c} | {v:.3f} | {'yes' if c in report['eligible_cells'] else 'no'} |")
    lines += ["", f"Primary k: {report['primary_k']}" + (f" (median ORACLE F by k: {report.get('primary_k_medians')})" if report.get("primary_k_medians") else ""),
              f"Lens divergence D_l: {report['divergence']}", ""]
    a = report["analysis"]["primary"]
    lines += ["## Primary family (Holm)", "",
              f"- P2.1 JL-OUT minus LL-OUT, M1: {_ci(a['P2.1'])}",
              f"- P2.2 JL-OUT minus LL-OUT, M2 at k = {report['primary_k']}: {_ci(a['P2.2'])}",
              f"- P2.3 slope against D_l (readout): {_ci(a['P2.3'])}", ""]
    if ctx.phase == "dev":
        lines += ["These are dev estimates: they size the test (power below) and are not claims.", ""]
    pooled = report["summary"]["pooled"]
    lines += ["## Signals (pooled over eligible cells; pair means)", "",
              "| Signal | M1 | " + " | ".join(f"F k={k}" for k in _resolve(ctx.config)["k_grid"]) + " | M5 at primary k |",
              "|---|---|" + "---|" * len(_resolve(ctx.config)["k_grid"]) + "---|"]
    for n in sorted(pooled):
        v = pooled[n]
        lines.append(f"| {n} | {_fmt(v['M1'])} | " + " | ".join(_fmt(v["F"][str(k)]) for k in _resolve(ctx.config)["k_grid"]) +
                     f" | {_share(report['summary']['M5'].get(n))} |")
    lines += ["", f"RANDOM mean F at the primary k: {_fmt(report['summary']['random_mean_F_at_primary_k'])}",
              f"Sanity (6.3 step 7): violations {report['sanity']['violations'] or 'none'}; batched-serial parity max {report['parity_max']:.2g}; "
              f"(SAE-ALL + ERR - FULL) relative max {report['sanity_max_relative']:.2g}", ""]
    if report.get("lens_fidelity"):
        lines += ["## Lens-fidelity diagnostic (Spearman with the finite-difference change in m)", ""]
        for layer, v in report["lens_fidelity"].items():
            lines.append(f"- L{layer}: logit lens {v['LL']['spearman']:.3f}, J-lens {v['JL']['spearman']:.3f}, local gradient {v['local']['spearman']:.3f} (n {v['n']})")
        lines.append("")
    if report.get("power"):
        lines += ["## Power at the planned test size", ""]
        for k, v in report["power"].items():
            if v:
                lines.append(f"- {k}: dev estimate {v['dev_estimate']:+.3f}, SE {v['se_at_test_size']:.3f}, minimum detectable {v['min_detectable_effect']:.3f}, "
                             f"power to exclude 0 {_share(v['power_to_exclude_zero_at_dev_estimate'])}, TOST power if 0 {_share(v['power_equivalence_if_zero'])}")
        lines.append("")
    lines += ["## What this permits", ""]
    if ctx.phase == "dev":
        lines += ["The frozen protocol holds the relations, templates, eligible cells and primary k. The test run needs explicit approval "
                  "and runs once. Dev estimates are not claims."]
    else:
        lines += ["Read the primary family against the 6.3 decision table: P2.1 or P2.2 better after Holm: J-lens adds information; P2.3 better: "
                  "the advantage tracks lens divergence; P2.1 and P2.2 equivalent: clean negative; otherwise inconclusive (never 'no difference')."]
    lines += ["", "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"


def _extension_summary(ctx: RunContext, report: dict[str, Any]) -> str:
    """SUMMARY.md of an extension run (7.2, X4): one analysis per SAE variant on the same trials."""
    cfg = _resolve(ctx.config)
    ext = report["extension"]
    pk = report["primary_k"]
    lines = ["## What ran", "",
             f"Stage 2 {ctx.phase} phase, extension {ext['id']} on `{report['model']}`: the parent protocol "
             f"({str(ext['parent_protocol_hash'])[:12]}...) with only the SAE changed, at {', '.join(ext['cells'])}. Relations, templates, "
             f"primary k ({pk}) and divergence are inherited. Causal measurements in {report['causal_mode']}, gradients in "
             f"{report['gradient_mode']}, patches {report['strategy']}. M1 = Spearman with single-feature transfer effects; "
             "M2 = F_k = (m(S_k) - m_clean) / (m_FULL - m_clean). Unsquared. Each variant is analysed on its own; nothing is pooled "
             "across variants and there is no multiplicity correction across them (secondary analysis).", "",
             f"Trials: {report['n_trials']} in {report['n_pairs']} concept pairs. Median R_FULL: "
             + ", ".join(f"{c} {v:.3f}" for c, v in sorted(report["r_full_median"].items())), "",
             "## SAE variants (Stage 0, scope sae_variants)", "", "| Variant | Variance explained | Below the N13 floor | Mean L0 |", "|---|---|---|---|"]
    for key, v in sorted(ext["variants"].items()):
        lines.append(f"| {key} | {v['variance_explained']:.3f} | {'yes' if v['low_fidelity'] else 'no'} | {v['mean_l0']:.1f} |")
    lines += ["", "## JL-OUT minus LL-OUT per variant", "", f"| Variant | M1 | M2 at k = {pk} |", "|---|---|---|"]
    for key, e in sorted(report["by_sae"].items()):
        a = e["analysis"]["primary"]
        lines.append(f"| {key} | {_ci(a['P2.1'])} | {_ci(a['P2.2'])} |")
    for key, e in sorted(report["by_sae"].items()):
        pooled = e["summary"]["pooled"]
        lines += ["", f"## Signals, {key}", "", f"| Signal | M1 | F k={pk} | M5 |", "|---|---|---|---|"]
        for n in sorted(pooled):
            lines.append(f"| {n} | {_fmt(pooled[n]['M1'])} | {_fmt(pooled[n]['F'].get(str(pk)))} | {_share(e['summary']['M5'].get(n))} |")
        lines += ["", f"RANDOM mean F at k = {pk}: {_fmt(e['summary']['random_mean_F_at_primary_k'])}; sanity violations: "
                  f"{e['sanity']['violations'] or 'none'}"]
    lines += ["", f"Batched-serial parity max {report['parity_max']:.2g}; (SAE-ALL + ERR - FULL) relative max {report['sanity_max_relative']:.2g}.", "",
              "## What this permits", ""]
    if ctx.phase == "dev":
        lines += ["Dev estimates, not claims. The extension's frozen protocol holds the inherited relations, templates, cell and primary k "
                  "and the SAE variants; its test run needs explicit approval and runs once."]
    else:
        lines += [f"Robustness of the J-lens contrast at {', '.join(ext['cells'])} to the SAE width and seed, read against the parent test's "
                  "result at the same cell. Three-way verdicts per variant; inconclusive is never 'no difference'."]
    lines += ["", "## Next step", "", "Send back the files in the status line."]
    if cfg.get("k_grid") and pk not in cfg["k_grid"]:
        lines.append(f"(primary k {pk} is not in this config's k grid)")
    return "\n".join(lines) + "\n"


def _endpoint_summary(report: dict[str, Any], cfg: dict[str, Any], *, phase: str) -> str:
    """The off-target endpoint section appended to an endpoint extension's SUMMARY.md."""
    e = report["endpoint"]
    pk = report["primary_k"]

    def mci(c: dict[str, Any] | None, signed: bool = False) -> str:
        if not c:
            return "n/a"
        f = (lambda v: f"{v:+.4g}") if signed else (lambda v: f"{v:.4g}")  # KLs can be small: significant digits, not decimals
        return f(c["estimate"]) + (f" [{f(c['ci95'][0])}, {f(c['ci95'][1])}]" if c.get("ci95") else "")

    lines = ["", "## Hybrid selectors (confirmatory)", "",
             f"Each hybrid minus NORM: M2 at k = {pk} (Holm across the hybrids; three-way against the M2 SESOI "
             f"{cfg['analysis']['sesoi']['M2']}) and M1 (alongside).", "", "| Hybrid | M2 minus NORM [95%], verdict | M1 minus NORM [95%], verdict |",
             "|---|---|---|"]
    for h in e["hybrids"]["M2"]:
        lines.append(f"| {h} | {_ci(e['hybrids']['M2'][h])} | {_ci(e['hybrids']['M1'][h])} |")
    lines += ["", "Decision: a hybrid better than NORM means J-space membership adds selection information to NORM; equivalent, "
              "nothing of SESOI size; worse, combining dilutes NORM. Inconclusive is never 'no difference'.", "",
              "## Off-target readout (descriptive)", "",
              "From the same forward pass as each set's margin, at the readout, in nats: O1 = KL(P~_clean || P~_S) over the tokens outside "
              "both answer sets (renormalized); O2 = answer mass lost, AM(clean) - AM(S); O3 = KL(P_FULL || P_S). Pair means with 95% "
              "intervals over trials at or above the M2 floor; no SESOI. O1 / O1(FULL) compares a set's off-answer change "
              "with the full transfer's own.", ""]
    for k, block in e["offtarget"].items():
        lines += [f"### k = {k}", "", "| Selector | F | O1 | O1 / O1(FULL) | O2 | O3 |", "|---|---|---|---|---|---|"]
        for name, v in block["selectors"].items():
            lines.append(f"| {name} | {mci(v['F'])} | {mci(v['O1'])} | {mci(v['O1_ratio'])} | {mci(v['O2'], True)} | {mci(v['O3'])} |")
        if block["contrasts"]:
            lines += ["", "| Contrast | O1 difference | O2 difference | O3 difference |", "|---|---|---|---|"]
            for cname, v in block["contrasts"].items():
                lines.append(f"| {cname} | {mci(v['O1'], True)} | {mci(v['O2'], True)} | {mci(v['O3'], True)} |")
        lines.append("")
    lines += [f"FULL margin through the off-target evaluator against the trial's stored FULL margin, worst absolute difference: "
              f"{e['full_margin_parity_max']:.2g}.", ""]
    if phase == "dev":
        lines += ["Dev estimates, not claims; the frozen protocol holds the parent protocol, the hybrids and the k values."]
    return "\n".join(lines) + "\n"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3g}"


def _budget_summary(report: dict[str, Any], cfg: dict[str, Any], *, phase: str) -> str:
    """The new-task endpoint section appended to SUMMARY.md."""
    nb = cfg["norm_budget"]
    e = report["norm_budget"]
    pk = report["primary_k"]
    budgets = [str(float(b)) for b in nb["budgets"]]
    a, b = nb["family"]
    lines = ["", "## Norm-budget endpoint (E2)", "",
             "Each selector adds features in its own order until the intended edit norm reaches B x ||delta h_FULL||, with no scaling; "
             "F = (m(S) - m_clean) / (m_FULL - m_clean) at that set, unsquared, on trials at or above the M2 floor where both orderings "
             f"reach the budget. Three-way against the M2 SESOI {cfg['analysis']['sesoi']['M2']}.", "",
             f"### Family (Holm across budgets): {a} minus {b}, predicted above 0", ""]
    lines += [f"- {n}: {_ci(c)}" for n, c in e["family"].items()]
    lines += ["", "### Secondary (three-way labels with the declared signs, no Holm)", "",
              f"| Contrast | M2 at k = {pk} | " + " | ".join(f"F at B = {x}" for x in budgets) + " |", "|---|---|" + "---|" * len(budgets)]
    for n, entry in e["secondary"].items():
        lines.append(f"| {n} | " + " | ".join(_ci(v) for v in entry.values()) + " |")
    carried = report.get("sae_carried") or {}
    if carried.get("dev_median_sae_all_recovery"):
        lines += ["", "Dev median SAE-ALL recovery per cell: " + ", ".join(f"{c} {v:.3f}" for c, v in carried["dev_median_sae_all_recovery"].items())]
    if e.get("sae_carried"):
        s_ = e["sae_carried"]
        lines += ["", f"### Restricted to the cells where SAE-ALL carries the transfer: {', '.join(s_['cells']) or 'none'} ({s_['n_trials']} trials)", "",
                  f"- P2.1: {_ci(s_['P2.1'])}", f"- P2.2: {_ci(s_['P2.2'])}"] + [f"- {n}: {_ci(c)}" for n, c in s_["family"].items()]
    d = e["descriptive"]["pooled"]
    for x in budgets:
        lines += ["", f"### B = {x} (pooled; pair means)", "", "| Selector | Share reaching B | F [95%] | k (active) | Realized rho |", "|---|---|---|---|---|"]
        for n, per in d["budget"].items():
            v = per[x]
            f = v["F"]
            fs = "n/a" if not f else f"{f['estimate']:+.3f}" + (f" [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}]" if f.get("ci95") else "")
            lines.append(f"| {n} | {_share(v['reached'])} | {fs} | {_num(v['k'])} | {_num(v['rho'])} |")
    ks = [str(k) for k in nb["curve_k"]]
    lines += ["", "### F against realized rho on the fine k grid (pooled; F, realized rho in parentheses)", "",
              "| Selector | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for n, per in d["curve"].items():
        lines.append(f"| {n} | " + " | ".join(f"{_num(per[k]['F'])} ({_num(per[k]['rho'])})" for k in ks) + " |")
    lines += ["", f"Realized against intended norm of the budget sets, worst relative gap: {e['max_realized_vs_intended_gap']:.2g}.", ""]
    if phase == "dev":
        lines += ["Dev estimates, not claims. The frozen protocol holds the relation, templates, cells, primary k and the SAE-carried cells."]
    return "\n".join(lines) + "\n"


def _a38_summary(report: dict[str, Any], cfg: dict[str, Any], *, phase: str) -> str:
    """The ranking-versus-sets test section appended to SUMMARY.md."""
    e = report["ranking_vs_sets"]
    pk = report["primary_k"]
    a, b = cfg["ranking_vs_sets"]["dissociation"]
    tk = cfg["ranking_vs_sets"]["toolkit"]
    big, small = tk["compare"]
    lines = ["", "## Ranking against sets, budget dependence, toolkit", "",
             f"Confirmatory (Holm across the run's D tests): D1 {a} minus {b} on M1, predicted above 0; D2 the same on M2 at k = {pk}. "
             "The dissociation replicates on a relation when D1 is practically better after Holm and D2 is equivalent or worse.", "",
             "| Relation | D1 (M1) | D2 (M2) | Replicates |", "|---|---|---|---|"]
    for t, rep_ in e["replication"].items():
        lines.append(f"| {t} | {_ci(e['dissociation'][f'D1 {t}'])} | {_ci(e['dissociation'][f'D2 {t}'])} | "
                     f"{'yes' if rep_['replicates'] else 'no'} ({rep_['reason']}) |")
    bd = cfg["ranking_vs_sets"]["budget_dependence"]
    lines += ["", f"Budget dependence (secondary, Holm across relations): ({bd['pair'][0]} - {bd['pair'][1]}) at B = {bd['budget']} minus the "
              f"same at k = {pk}, predicted below 0.", ""] + [f"- {t}: {_ci(c)}" for t, c in e["budget_dependence"].items()]
    lines += ["", "Per relation, JL-OUT minus LL-OUT (secondary):", ""] + [
        f"- {t}: P2.1 {_ci(v['P2.1'])}; P2.2 {_ci(v['P2.2'])}" for t, v in e["per_task_primary"].items()]
    choices = report.get("toolkit_choices") or {}
    sets = " / ".join(f"{n} = {{" + ", ".join(sels) + "}" for n, sels in tk["sets"].items())
    lines += ["", f"Toolkit choices ({sets}; per cell: {small} / {big}), made on dev and frozen:", ""]
    for endpoint, per_rel in choices.items():
        for t, per_cell in per_rel.items():
            lines.append(f"- {endpoint}, {t}: " + "; ".join(f"{c} {ch[small]} / {ch[big]}" for c, ch in per_cell.items()))
    if e.get("toolkit"):
        lines += ["", f"Toolkit test: F({big} choice) minus F({small} choice), predicted at or above 0, Holm across relations:", ""]
        for endpoint, per in e["toolkit"].items():
            lines += [f"- {endpoint}, {t}: {_ci(c)}" for t, c in per.items()]
    elif phase == "dev":
        lines += ["", "The toolkit test is evaluated on test only: its choices were made on these dev trials."]
    return "\n".join(lines) + "\n"
