"""Stats: Spearman with ties, partial Spearman, tie-aware precision at k, bootstrap
coverage and strata, p-values, verdicts, Holm, power, common-support area."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from saeselect import stats as S


def test_spearman_with_ties_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 60).astype(float)
    y = x + rng.integers(0, 3, 60)
    assert S.spearman(x, y) == pytest.approx(sps.spearmanr(x, y).statistic, rel=1e-12)
    assert np.isnan(S.spearman([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError, match="non-finite"):
        S.spearman([1, np.nan], [1, 2])


def test_partial_spearman_matches_closed_form():
    rng = np.random.default_rng(1)
    z = rng.normal(size=200)
    x = z + rng.normal(size=200)
    y = z + 0.5 * x + rng.normal(size=200)
    r = lambda a, b: sps.spearmanr(a, b).statistic  # noqa: E731
    closed = (r(x, y) - r(x, z) * r(y, z)) / np.sqrt((1 - r(x, z) ** 2) * (1 - r(y, z) ** 2))
    assert S.partial_spearman(x, y, z) == pytest.approx(closed, rel=1e-10)


def test_topk_membership_and_precision_with_ties():
    assert S.topk_membership([3, 2, 2, 1], 2).tolist() == [1.0, 0.5, 0.5, 0.0]
    # signal top-2: {0, 0.5 of 1, 0.5 of 2}; effect top-2: {0, 1} -> overlap 1.5 of 2
    assert S.precision_at_k([3, 2, 2, 1], [3, 2, 1, 0], 2) == pytest.approx(0.75)
    assert S.precision_at_k([1, 2, 3, 4], [1, 2, 3, 4], 2) == pytest.approx(1.0)
    assert S.precision_at_k([4, 3, 2, 1], [1, 2, 3, 4], 2) == pytest.approx(0.0)
    assert np.isnan(S.precision_at_k([1, 2], [1, 2], 3))
    # all tied: every item has membership k / n
    assert S.precision_at_k([1, 1, 1, 1], [4, 3, 2, 1], 2) == pytest.approx(0.5)


def test_bootstrap_interval_coverage():
    rng = np.random.default_rng(2)
    true_mean, hits, experiments = 0.3, 0, 200
    for i in range(experiments):
        data = rng.normal(true_mean, 1.0, size=40)
        lo, hi = S.cluster_bootstrap(data, n_resamples=1000, seed=i).ci(0.95)
        hits += lo <= true_mean <= hi
    assert 0.89 <= hits / experiments <= 0.99


def test_stratified_bootstrap_keeps_stratum_sizes():
    values = np.array([0.0] * 30 + [10.0] * 10)
    strata = ["a"] * 30 + ["b"] * 10
    result = S.cluster_bootstrap(values, n_resamples=500, seed=0, strata=strata)
    # each resample keeps 10 of 40 clusters from stratum b, so the mean is exactly 2.5
    assert np.allclose(result.samples, 2.5)
    unstratified = S.cluster_bootstrap(values, n_resamples=500, seed=0)
    assert unstratified.samples.std() > 0


def test_bootstrap_custom_statistic_and_errors():
    values = np.arange(10, dtype=float)
    result = S.cluster_bootstrap(values, n_resamples=200, seed=0, statistic=np.median)
    assert result.estimate == pytest.approx(4.5)
    with pytest.raises(ValueError, match="one value per cluster"):
        S.cluster_bootstrap(np.ones((5, 2)), n_resamples=10, seed=0)
    with pytest.raises(ValueError, match="two clusters"):
        S.cluster_bootstrap([1.0], n_resamples=10, seed=0)


def _result(samples, estimate=None):
    samples = np.asarray(samples, dtype=float)
    return S.BootstrapResult(estimate=float(samples.mean() if estimate is None else estimate), samples=samples)


def test_p_value_far_side_of_zero():
    assert _result(np.linspace(0.1, 1.0, 1000)).p_value() == 0.0
    assert _result(np.linspace(-1, 1, 1001)).p_value() == pytest.approx(1.0)
    assert _result(np.r_[np.full(40, -0.1), np.full(960, 0.5)]).p_value() == pytest.approx(0.08)


def test_verdicts():
    kw = dict(sesoi=0.05, level=0.95, tost_level=0.90)
    better = S.verdict(_result(np.linspace(0.08, 0.2, 1000)), predicted_sign=1, **kw)
    assert better.label == "better" and better.practically_better and not better.equivalent
    small = S.verdict(_result(np.linspace(0.005, 0.03, 1000)), predicted_sign=1, **kw)
    assert small.label == "better" and small.equivalent and not small.practically_better
    equivalent = S.verdict(_result(np.linspace(-0.03, 0.03, 1000)), predicted_sign=1, **kw)
    assert equivalent.label == "equivalent"
    worse = S.verdict(_result(np.linspace(-0.3, -0.1, 1000)), predicted_sign=1, **kw)
    assert worse.label == "worse"
    flipped = S.verdict(_result(np.linspace(-0.3, -0.1, 1000)), predicted_sign=-1, **kw)
    assert flipped.label == "better"
    wide = S.verdict(_result(np.linspace(-0.3, 0.4, 1000)), predicted_sign=1, **kw)
    assert wide.label == "inconclusive"


def test_holm_known_case():
    assert S.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm([0.5, 0.9]).tolist() == pytest.approx([1.0, 1.0])
    verdicts = {
        "a": S.verdict(_result(np.linspace(0.1, 0.2, 1000)), sesoi=0.05, predicted_sign=1, level=0.95, tost_level=0.9),
        "b": S.verdict(_result(np.r_[np.full(30, -0.1), np.full(970, 0.2)]), sesoi=0.05, predicted_sign=1, level=0.95, tost_level=0.9),
    }
    S.apply_holm(verdicts, alpha=0.05)
    assert verdicts["a"].better_after_holm is True
    # b: raw p = 0.06 -> not better even before adjustment
    assert verdicts["b"].better_after_holm is False


def test_power_helpers():
    assert S.min_detectable_effect(1.0, alpha=0.05, power=0.8) == pytest.approx(1.959964 + 0.841621, rel=1e-5)
    assert S.power_to_exclude_zero(2.801585, 1.0, alpha=0.05) == pytest.approx(0.8, abs=1e-4)
    se = S.resampled_se(np.random.default_rng(0).normal(0, 2.0, 400), n_clusters=100, n_sims=4000, seed=1)
    assert se == pytest.approx(0.2, rel=0.1)
    assert S.power_equivalence(0.001, sesoi=0.05, alpha=0.05) == pytest.approx(1.0)
    assert S.power_equivalence(1.0, sesoi=0.05, alpha=0.05) == 0.0


def test_common_support_area():
    # y = x on [0, 2] and y = 2x on [1, 3]: common support [1, 2]
    area_a, area_b, lo, hi = S.common_support_area([0, 2], [0, 2], [1, 3], [2, 6])
    assert (lo, hi) == (1.0, 2.0)
    assert area_a == pytest.approx(1.5) and area_b == pytest.approx(3.0)
    area_a, area_b, *_ = S.common_support_area([0, 1], [0, 1], [2, 3], [0, 1])
    assert np.isnan(area_a) and np.isnan(area_b)
