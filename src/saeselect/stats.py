"""Statistics.

- Spearman with average ranks for ties (M1), partial Spearman (M6).
- Tie-aware precision at k (M3).
- Cluster bootstrap with percentile intervals, optionally stratified (pooled
  Stage 2 analyses resample clusters within each relation).
- Bootstrap p-values (twice the share of resamples on the far side of zero),
  Holm step-down adjustment, TOST equivalence on the 90% interval, and the
  three-way verdict.
- Power helpers: the standard error of a cluster mean at a planned size, by
  resampling dev clusters; minimum detectable effect; power.
- Area over the common norm support of two effect-norm curves.

Every quantity that a config declares (number of resamples, SESOI, alpha,
power) is a required argument here; nothing measurable has a default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as _sps

# --------------------------------------------------------------------------
# Ranking quality
# --------------------------------------------------------------------------


def _as_float_array(x: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation with average ranks for ties. NaN if either side is constant."""
    x = _as_float_array(x, "x")
    y = _as_float_array(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        return float("nan")
    return pearson(_sps.rankdata(x, method="average"), _sps.rankdata(y, method="average"))


def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float] | np.ndarray) -> float:
    """Spearman of x and y controlling for z (M6).

    Ranks every variable (average ranks), regresses the ranks of x and of y on
    the ranks of z with an intercept, and correlates the residuals.
    """
    x = _as_float_array(x, "x")
    y = _as_float_array(y, "y")
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if not (x.shape[0] == y.shape[0] == z.shape[0]):
        raise ValueError("x, y and z must have the same length")
    rx = _sps.rankdata(x, method="average")
    ry = _sps.rankdata(y, method="average")
    rz = np.column_stack([_sps.rankdata(z[:, j], method="average") for j in range(z.shape[1])])
    design = np.column_stack([np.ones(len(rx)), rz])
    beta_x, *_ = np.linalg.lstsq(design, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, ry, rcond=None)
    return pearson(rx - design @ beta_x, ry - design @ beta_y)


def topk_membership(values: Sequence[float], k: int) -> np.ndarray:
    """Fractional membership of each item in the top k (largest values).

    Items strictly above the k-th largest value get weight 1. Items tied with
    it share the remaining places equally. The weights sum to k, and equal the
    probability of being in the top k under random tie-breaking.
    """
    v = _as_float_array(values, "values")
    n = v.size
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must be between 1 and the pool size {n}")
    kth = np.sort(v)[::-1][k - 1]
    above = v > kth
    tied = v == kth
    weights = above.astype(np.float64)
    weights[tied] = (k - above.sum()) / tied.sum()
    return weights


def precision_at_k(scores: Sequence[float], effects: Sequence[float], k: int) -> float:
    """Tie-aware precision at k (M3): expected share of the signal's top k that
    lies in the top k by effect, with fractional membership at ties on both
    sides. NaN when the pool has fewer than k items."""
    s = _as_float_array(scores, "scores")
    e = _as_float_array(effects, "effects")
    if s.shape != e.shape:
        raise ValueError("scores and effects must have the same length")
    if k > s.size:
        return float("nan")
    return float((topk_membership(s, k) * topk_membership(e, k)).sum() / k)


# --------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    estimate: float
    samples: np.ndarray

    def ci(self, level: float) -> tuple[float, float]:
        """Percentile interval at ``level`` (e.g. 0.95)."""
        lo, hi = np.percentile(self.samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
        return float(lo), float(hi)

    def p_value(self) -> float:
        """Twice the share of resamples on the far side of zero, capped at 1."""
        below = float(np.mean(self.samples <= 0))
        above = float(np.mean(self.samples >= 0))
        return min(1.0, 2.0 * min(below, above))


def cluster_bootstrap(
    cluster_values: Sequence[float] | np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    statistic: Callable[[np.ndarray], float] | None = None,
    strata: Sequence[object] | None = None,
) -> BootstrapResult:
    """Resample clusters with replacement and recompute a statistic.

    ``cluster_values`` has one row per cluster (already averaged over the
    cluster's trials, section 5.6). The default statistic is the mean over
    clusters. With ``strata``, clusters are resampled within each stratum, and
    each stratum keeps its size.
    """
    values = np.asarray(cluster_values, dtype=np.float64)
    if values.shape[0] < 2:
        raise ValueError("need at least two clusters to bootstrap")
    if not np.all(np.isfinite(values)):
        raise ValueError("cluster values contain non-finite entries")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if strata is None:
        index = rng.integers(0, n, size=(n_resamples, n))
    else:
        labels = np.asarray(strata, dtype=object)
        if labels.shape[0] != n:
            raise ValueError("strata must label every cluster")
        parts = []
        for label in sorted(set(labels.tolist()), key=str):
            members = np.flatnonzero(labels == label)
            parts.append(members[rng.integers(0, members.size, size=(n_resamples, members.size))])
        index = np.concatenate(parts, axis=1)
    if statistic is None:
        if values.ndim != 1:
            raise ValueError("the default statistic (mean) needs one value per cluster; pass statistic=")
        estimate = float(values.mean())
        samples = values[index].mean(axis=1)
    else:
        estimate = float(statistic(values))
        samples = np.array([statistic(values[row]) for row in index], dtype=np.float64)
    return BootstrapResult(estimate=estimate, samples=np.asarray(samples, dtype=np.float64))


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    """Three-way verdict on a paired contrast (section 5.6).

    ``label`` is ``better`` (the 95% interval excludes zero in the predicted
    direction; ``practically_better`` if its near bound also clears the SESOI),
    ``equivalent`` (the 90% interval lies inside +/-SESOI: TOST at alpha 0.05),
    ``worse`` (the 95% interval excludes zero against the prediction, i.e. the
    comparator is better), or ``inconclusive``. ``better`` and ``equivalent``
    can both hold (a significant difference smaller than the SESOI); both flags
    are reported and the label is ``better``.
    """

    label: str
    estimate: float
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    p_value: float
    sesoi: float
    predicted_sign: int
    better: bool
    practically_better: bool
    equivalent: bool
    worse: bool
    p_adjusted: float | None = None
    better_after_holm: bool | None = None


def verdict(result: BootstrapResult, *, sesoi: float, predicted_sign: int, level: float, tost_level: float) -> Verdict:
    """Classify a bootstrap contrast. ``level`` is the interval level for
    "better" (0.95); ``tost_level`` the interval level for equivalence (0.90)."""
    if predicted_sign not in (1, -1):
        raise ValueError("predicted_sign must be +1 or -1")
    if sesoi <= 0:
        raise ValueError("sesoi must be positive")
    lo, hi = result.ci(level)
    lo90, hi90 = result.ci(tost_level)
    # Express the interval in the predicted direction.
    near, far = (lo, hi) if predicted_sign == 1 else (-hi, -lo)
    better = near > 0
    practically = near > sesoi
    worse = far < 0
    equivalent = (lo90 > -sesoi) and (hi90 < sesoi)
    if better:
        label = "better"
    elif equivalent:
        label = "equivalent"
    elif worse:
        label = "worse"
    else:
        label = "inconclusive"
    return Verdict(
        label=label,
        estimate=result.estimate,
        ci95=(lo, hi),
        ci90=(lo90, hi90),
        p_value=result.p_value(),
        sesoi=sesoi,
        predicted_sign=predicted_sign,
        better=better,
        practically_better=practically,
        equivalent=equivalent,
        worse=worse,
    )


def apply_holm(verdicts: dict[str, Verdict], *, alpha: float) -> dict[str, Verdict]:
    """Attach Holm-adjusted p-values to a primary family. ``better_after_holm``
    is ``better`` with the adjusted p-value below ``alpha``. Intervals stay
    unadjusted and are reported next to the adjusted p-values (section 5.6)."""
    names = list(verdicts)
    adjusted = holm([verdicts[n].p_value for n in names])
    for name, p_adj in zip(names, adjusted):
        verdicts[name].p_adjusted = float(p_adj)
        verdicts[name].better_after_holm = verdicts[name].better and float(p_adj) < alpha
    return verdicts


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def resampled_se(cluster_values: Sequence[float], *, n_clusters: int, n_sims: int, seed: int) -> float:
    """SD of the mean of ``n_clusters`` clusters drawn with replacement from
    the dev clusters: the standard error expected at the planned test size."""
    values = _as_float_array(cluster_values, "cluster_values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_sims, n_clusters))].mean(axis=1)
    return float(means.std(ddof=1))


def min_detectable_effect(se: float, *, alpha: float, power: float) -> float:
    """Smallest true effect a two-sided test at ``alpha`` detects with ``power``."""
    return float((_sps.norm.ppf(1 - alpha / 2) + _sps.norm.ppf(power)) * se)


def power_to_exclude_zero(effect: float, se: float, *, alpha: float) -> float:
    return float(_sps.norm.cdf(abs(effect) / se - _sps.norm.ppf(1 - alpha / 2)))


def power_equivalence(se: float, *, sesoi: float, alpha: float) -> float:
    """Approximate TOST power when the true difference is zero."""
    return float(max(0.0, 2 * _sps.norm.cdf(sesoi / se - _sps.norm.ppf(1 - alpha)) - 1))


# --------------------------------------------------------------------------
# Norm-effect curves (M4)
# --------------------------------------------------------------------------


def common_support_area(
    norms_a: Sequence[float], effects_a: Sequence[float], norms_b: Sequence[float], effects_b: Sequence[float]
) -> tuple[float, float, float, float]:
    """Area under two effect-versus-norm curves over their common norm support.

    Returns (area_a, area_b, lower, upper). Each curve is linearly
    interpolated; the support is the overlap of the two norm ranges. NaN areas
    when the ranges do not overlap. Never rescales either curve.
    """
    xa, ya = _sorted_curve(norms_a, effects_a)
    xb, yb = _sorted_curve(norms_b, effects_b)
    lower = max(xa[0], xb[0])
    upper = min(xa[-1], xb[-1])
    if not upper > lower:
        return float("nan"), float("nan"), float(lower), float(upper)
    grid = np.unique(np.concatenate([[lower, upper], xa[(xa > lower) & (xa < upper)], xb[(xb > lower) & (xb < upper)]]))
    area_a = float(np.trapezoid(np.interp(grid, xa, ya), grid))
    area_b = float(np.trapezoid(np.interp(grid, xb, yb), grid))
    return area_a, area_b, float(lower), float(upper)


def _sorted_curve(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float_array(x, "norms")
    y = _as_float_array(y, "effects")
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def sub_seed(base: int, *parts: Any) -> int:
    """A 32-bit seed derived from a base seed and labels, so each analysis draws its own reproducible stream."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, (base, *parts))).encode()).digest()[:4], "big")
