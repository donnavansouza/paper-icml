"""Toolkit choices (dev) and the toolkit contrast (test), the budget-dependence value and the replication rule, on small
synthetic trial rows. The end-to-end path (two active relations, per-relation cells, frozen choices, the extension) is in
tests/test_new_tasks.py."""

from __future__ import annotations

import pytest

from saeselect.stages import selection as S2

CFG = {"m2_min_full_gain": 1.0, "analysis": {"bootstrap": {"n_resamples": 300, "level": 0.95, "tost_level": 0.90},
                                             "sesoi": {"M1": 0.05, "M2": 0.05}, "holm_alpha": 0.05}}


def _row(pair, rel, F_k, F_b=None, gain=5.0, cell=(12, "concept")):
    """F_k / F_b: selector -> recovered share at the primary k (32) / at the norm budget (None: not reached)."""
    return {"pair": pair, "relation": rel, "layer": cell[0], "site": cell[1], "full_gain": gain,
            "sets": {n: {"32": {"F": f}} for n, f in F_k.items()},
            "norm_budget": {"sets": {n: {"0.25": ({"reached": True, "F": f} if f is not None else {"reached": False})}
                                     for n, f in (F_b or {}).items()}}}


TK = {"T0": ["NORM", "ATP"], "T1": ["NORM", "ATP", "JL-OUT"]}


def test_toolkit_choices_pick_the_best_mean_with_declared_tie_order():
    rows = [_row("p1", "R5", {"NORM": 0.4, "ATP": 0.2, "JL-OUT": 0.5}), _row("p2", "R5", {"NORM": 0.4, "ATP": 0.4, "JL-OUT": 0.6}),
            _row("p3", "R5", {"NORM": 0.9, "ATP": 0.9, "JL-OUT": 0.9}, gain=0.5),  # below the M2 floor: ignored
            _row("q1", "R6", {"NORM": 0.3, "ATP": 0.3, "JL-OUT": 0.1})]
    ch = S2.toolkit_choices(rows, CFG, TK, pk=32)
    assert ch["R5"]["12/concept"]["T0"] == "NORM" and ch["R5"]["12/concept"]["T1"] == "JL-OUT"
    assert ch["R5"]["12/concept"]["means"]["NORM"] == pytest.approx(0.4)
    assert ch["R6"]["12/concept"]["T0"] == "NORM" and ch["R6"]["12/concept"]["T1"] == "NORM"  # NORM = ATP: the earlier one wins


def test_toolkit_contrast_reads_each_trials_frozen_choice():
    choices = {"R5": {"12/concept": {"T0": "NORM", "T1": "JL-OUT"}}}
    rows = [_row(f"p{i}", "R5", {"NORM": 0.30 + 0.01 * i, "ATP": 0.1, "JL-OUT": 0.40 + 0.01 * i}) for i in range(8)]
    rows.append(_row("x", "R5", {"NORM": 0.1, "ATP": 0.1, "JL-OUT": 0.9}, cell=(24, "concept")))  # no choice at this cell
    c = S2.toolkit_contrast(rows, CFG, choices, a="T1", b="T0", pk=32, seed=1)
    assert c["estimate"] == pytest.approx(0.10) and c["n_pairs"] == 8 and c["better"]


def test_budget_dependence_is_a_difference_of_differences():
    rows = [_row(f"p{i}", "R5", {"NORM": 0.5, "ORACLE": 0.4}, {"NORM": 0.2, "ORACLE": 0.3}) for i in range(6)]
    rows.append(_row("u", "R5", {"NORM": 0.5, "ORACLE": 0.4}, {"NORM": None, "ORACLE": 0.3}))  # NORM never reached B: left out
    c = S2.budget_dependence(rows, CFG, "NORM", "ORACLE", "0.25", pk=32, seed=2)
    assert c["estimate"] == pytest.approx((0.2 - 0.3) - (0.5 - 0.4)) and c["n_pairs"] == 6
    assert c["better"]  # predicted below 0, and it is


def test_replication_rule():
    d1 = {"practically_better": True, "better_after_holm": True, "label": "better"}
    eq = {"equivalent": True, "worse": False, "label": "equivalent"}
    worse = {"equivalent": False, "worse": True, "label": "worse"}
    inc = {"equivalent": False, "worse": False, "label": "inconclusive"}
    assert S2._replicates(d1, eq)["replicates"] and S2._replicates(d1, worse)["replicates"]
    assert not S2._replicates(d1, inc)["replicates"]  # an inconclusive D2 does not count
    assert not S2._replicates({**d1, "better_after_holm": False}, eq)["replicates"]
    assert not S2._replicates(None, eq)["replicates"]
