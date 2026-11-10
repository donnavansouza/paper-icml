"""Selection signals and their ranking metrics.

A trial is one source prompt at one layer and site. Its candidate pool C is
every feature whose activation changes between source and donor at some site
position (delta a_i(p) != 0). Scores predict each feature's effect toward the
goal (for concept transfer: an increase in m, the donor-answer margin).

In-context scores come from the trial's own prompt. Support scores (the
primary variant) average the in-context scores of the support trials (the same
pair and direction under the other passing templates), with a feature that is
not in a support trial's pool counting as 0 there (5.4 note 4). PATCH is the
support mean of measured single-feature effects; ORACLE is the effect measured
on the evaluation prompt (a reference, never a selector). KAPPA and JSPACE are
feature properties.

Formulas (none squared), with Sum_p over the site positions P:
- ASSOC_i  = Sum_p |delta a_i(p)| / sd_i            (sd_i on the neutral corpus, Stage 0)
- NORM_i   = Sum_p |delta a_i(p)| * ||d_i||
- LL-OUT_i = Sum_p delta a_i(p) * s_LL . d_i,   s_LL = N^T W_U^T g        (J = I)
- JL-OUT_i = Sum_p delta a_i(p) * s_JL . d_i,   s_JL = J^T N^T W_U^T g
- LL/JL-VERB: the same with q (the verbalization target) in place of g
- ATP_i    = Sum_p delta a_i(p) * (dm/dh_l(p) . d_i)   (gradient on the unedited prompt)
- JL-NORM_i = Sum_p |delta a_i(p)| * ||J d_i||        (NORM with the decoder row pushed to the last layer by J)
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from saeselect import stats

SIGNED = ("LL-OUT", "JL-OUT", "LL-VERB", "JL-VERB", "ATP", "PATCH", "ORACLE")
UNSIGNED = ("ASSOC", "NORM", "KAPPA", "JSPACE")
IN_CONTEXT = ("ASSOC", "NORM", "LL-OUT", "JL-OUT", "LL-VERB", "JL-VERB", "ATP")
SELECTORS = ("ASSOC", "NORM", "LL-OUT", "JL-OUT", "LL-VERB", "JL-VERB", "ATP", "PATCH", "KAPPA", "JSPACE")
SENSITIVITY_BASED = ("LL-OUT", "JL-OUT", "LL-VERB", "JL-VERB", "ATP", "PATCH")


def in_context_scores(*, da: torch.Tensor, W_dec_pool: torch.Tensor, sd_pool: torch.Tensor, lens_s: dict[str, torch.Tensor],
                      grad_site: torch.Tensor, j_norm_pool: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """Scores of the pool features on this trial's own prompt.

    ``da`` [|P|, |C|] activation changes, ``W_dec_pool`` [|C|, d] decoder rows,
    ``sd_pool`` [|C|], ``lens_s`` maps LL-OUT, JL-OUT, LL-VERB, JL-VERB to their
    residual-space sensitivity s [d], ``grad_site`` [|P|, d] = dm/dh_l at P, ``j_norm_pool`` [|C|] (optional)
    = ||J d_i||, which adds JL-NORM.
    """
    da = da.float()
    W = W_dec_pool.float()
    da_sum = da.sum(0)
    da_abs = da.abs().sum(0)
    sd = sd_pool.float().to(da.device)
    out = {
        "ASSOC": torch.where(sd > 0, da_abs / sd.clamp_min(1e-30), torch.zeros_like(da_abs)),
        "NORM": da_abs * W.norm(dim=1),
        "ATP": (da * (grad_site.float().to(W.device) @ W.T)).sum(0),
    }
    for name, s in lens_s.items():
        out[name] = da_sum * (W @ s.float().to(W.device))
    if j_norm_pool is not None:
        out["JL-NORM"] = da_abs * j_norm_pool.float().to(da.device)
    return {k: v.detach().cpu().double().numpy() for k, v in out.items()}


def support_scores(per_trial: Iterable[dict[int, float]]) -> dict[int, float]:
    """Mean over support trials, a feature absent from a trial's pool counting as 0 there."""
    per_trial = list(per_trial)
    if not per_trial:
        return {}
    total: dict[int, float] = {}
    for scores in per_trial:
        for f, v in scores.items():
            total[f] = total.get(f, 0.0) + float(v)
    return {f: v / len(per_trial) for f, v in total.items()}


def select_top(scores: dict[int, float], k: int, universe: Iterable[int]) -> list[int]:
    """The k features of ``universe`` with the largest scores (absent = 0); ties by feature id.
    Signed signals rank by predicted effect toward the goal, unsigned ones by value (5.4 note 7):
    both are "largest first"."""
    ranked = sorted(set(universe), key=lambda f: (-scores.get(f, 0.0), f))
    return ranked[:k]


def rank_average(a: dict[int, float], b: dict[int, float], universe: Iterable[int], *, tiebreak: dict[int, float]) -> dict[int, float]:
    """Hybrid selector: the mean of a feature's ranks under ``a`` and ``b`` over the universe (rank 0 = largest; absent
    = 0; ties by feature id, as :func:`select_top` orders), as a score where larger is better; equal mean ranks are broken
    by the ``tiebreak`` rank (NORM), which never outweighs a whole rank."""
    feats = sorted(set(universe))

    def ranks(sc: dict[int, float]) -> dict[int, int]:
        return {f: i for i, f in enumerate(sorted(feats, key=lambda f: (-sc.get(f, 0.0), f)))}

    ra, rb, rt = ranks(a), ranks(b), ranks(tiebreak)
    n = len(feats)
    return {f: -(ra[f] + rb[f]) / 2 - rt[f] / (2 * (n + 1)) for f in feats}


def random_draw(universe: Iterable[int], k: int, rng: np.random.Generator) -> list[int]:
    pool = sorted(set(universe))
    if not pool:
        return []
    return sorted(int(x) for x in rng.choice(pool, size=min(k, len(pool)), replace=False))


def ranking_metrics(scores: dict[int, float], effects: dict[int, float], *, ks: Iterable[int], control: dict[int, float] | None = None) -> dict[str, Any]:
    """M1 (Spearman of scores with effects over the pool), M1-abs, M3 (tie-aware
    precision at each k) and, with ``control``, M6 (partial Spearman given the control)."""
    pool = sorted(effects)
    s = np.array([scores.get(f, 0.0) for f in pool], dtype=np.float64)
    e = np.array([effects[f] for f in pool], dtype=np.float64)
    out: dict[str, Any] = {"n": len(pool)}
    if len(pool) < 3 or np.ptp(e) == 0:
        out.update(M1=None, M1_abs=None, M3={}, M6=None)
        return out
    out["M1"] = None if np.ptp(s) == 0 else stats.spearman(s, e)
    out["M1_abs"] = None if np.ptp(np.abs(s)) == 0 else stats.spearman(np.abs(s), np.abs(e))
    out["M3"] = {str(k): stats.precision_at_k(s, e, k) for k in ks if k <= len(pool)}
    if control is not None:
        c = np.array([control.get(f, 0.0) for f in pool], dtype=np.float64)
        out["M6"] = None if np.ptp(s) == 0 or np.ptp(c) == 0 else stats.partial_spearman(s, e, c)
    else:
        out["M6"] = None
    return out


def recovery(m_set: float, m_clean: float, m_full: float) -> float:
    """M2: F_k = (m(S_k) - m_clean) / (m_FULL - m_clean)."""
    return (m_set - m_clean) / (m_full - m_clean)
