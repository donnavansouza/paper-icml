"""The cross-model comparison: per-k contrasts, matched strata and the cross-model difference on fabricated trials."""

from __future__ import annotations

import yaml

from data_fixtures import REPO
from saeselect.analysis import compare_sets as C

KS = [1, 2, 4, 8, 16, 32]


def _cfg() -> dict:
    cfg = yaml.safe_load((REPO / "configs/selection/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg["analysis"]["bootstrap"]["n_resamples"] = 500
    cfg["k_grid"] = KS
    return cfg


def _rows(jl: float, ll: float, *, relations=("R2_element_symbol", "R3_verb_past"), cell=(8, "concept"), n=12, low_gain=False):
    rows = []
    for rel in relations:
        for i in range(n):
            noise = 0.01 * ((i * 7) % 5 - 2)  # deterministic spread so the intervals have width
            f = {"JL-OUT": jl + noise, "LL-OUT": ll, "NORM": 0.6, "ORACLE": 0.7}
            rows.append({"pair": f"{rel}:{i}", "relation": rel, "layer": cell[0], "site": cell[1], "full_gain": 0.5 if low_gain else 5.0,
                         "sets": {s: {str(k): {"F": v} for k in KS} for s, v in f.items()},
                         "random": {str(k): [{"F": 0.01}, {"F": 0.03}, {"F": None}] for k in KS},
                         "metrics": {"JL-OUT": {"M1": 0.3 + noise}, "LL-OUT": {"M1": 0.1}}})
    return rows


def test_per_k_levels_and_matching():
    cfg = _cfg()
    rows = _rows(0.5, 0.3) + _rows(0.9, 0.1, relations=("R1_country_capital",)) + _rows(0.9, 0.1, cell=(27, "readout"))
    sub = C.matched(rows, ["8/concept", "16/concept"], ["R2_element_symbol", "R3_verb_past"])
    assert len(sub) == 24  # the R1 trials and the readout cell are outside the matched strata
    out = C.per_k(sub, cfg, ks=KS, holm_alpha=0.05, seed=1)
    assert set(out) == {str(k) for k in KS}
    assert abs(out["4"]["estimate"] - 0.2) < 0.005 and out["4"]["label"] == "better" and out["4"]["better_after_holm"]
    lv = C.levels(sub, cfg, ks=KS)["32"]
    assert abs(lv["ORACLE"] - 0.7) < 1e-12 and abs(lv["RANDOM"] - 0.02) < 1e-12 and lv["pairs"] == 24
    assert C.levels(_rows(0.5, 0.3, low_gain=True), cfg, ks=KS)["4"]["pairs"] == 0  # below the M2 full-gain floor


def test_cross_model_difference():
    cfg = _cfg()
    c = C.cross(_rows(0.5, 0.3), cfg, _rows(0.3, 0.3), cfg, k=4, seed=3)
    assert abs(c["estimate"] - 0.2) < 0.005 and c["label"] == "better" and c["pairs"] == [24, 24]
    same = C.cross(_rows(0.3, 0.3), cfg, _rows(0.3, 0.3), cfg, k=32, seed=3)
    assert abs(same["estimate"]) < 1e-12 and same["equivalent"]
