"""The matched-size analysis (saeselect.analysis.matched_size): the split at matched edit size along the k path, and the reproduction guard."""

from __future__ import annotations

import pytest

from saeselect.analysis import matched_size as MS

CFG = {"k_grid": [1, 2], "m2_min_full_gain": 1.0, "analysis": {"bootstrap": {"n_resamples": 200, "level": 0.95}}}


def _rows() -> list[dict]:
    """Four pairs with one trial each, plus one trial below the M2 floor. Full gain 2, FULL's realized norm 10.
    A: k = 1 rho 0.2, F 0.2, U 0.1; k = 2 rho 0.4, F 0.6, U 0.3. B: k = 1 rho 0.1, F 0.1, U 0.05; k = 2 rho 0.6, F 0.3,
    U 0.25. At rho* = 0.3: A F 0.4, U 0.2; B F 0.18, U 0.13; so dF 0.22 = dU 0.07 + dN 0.15. A never reaches 0.5."""
    def row(pair: str, gain: float) -> dict:
        return {"pair": pair, "relation": "R", "full_gain": gain, "ref_norms": {"FULL": [10.0, 10.0]}, "metrics": {},
                "effects": {"1": 0.2, "2": 0.4, "3": 0.1},
                "sets": {"A": {"1": {"F": 0.2, "norms": [2.0, 2.0], "active": [1]}, "2": {"F": 0.6, "norms": [4.0, 4.0], "active": [1, 2]}},
                         "B": {"1": {"F": 0.1, "norms": [1.0, 1.0], "active": [3]}, "2": {"F": 0.3, "norms": [6.0, 6.0], "active": [3, 2]}}}}
    return [row(f"p{i}", 2.0) for i in range(4)] + [row("low", 0.5)]


def test_matched_split_interpolates_F_and_U_at_the_same_crossing():
    out = MS.matched_split(_rows(), CFG, [("A", "B")], [0.3, 0.5], seed=1, pk=2)
    v = out["A - B"]["0.3"]
    assert v["F"]["estimate"] == pytest.approx(0.22) and v["U"]["estimate"] == pytest.approx(0.07)
    assert v["N"]["estimate"] == pytest.approx(0.15) and v["trials"] == 4 and v["trials_above_floor"] == 4
    assert out["A - B"]["0.5"]["trials"] == 0 and out["A - B"]["0.5"]["F"] is None


def test_extension_compares_matched_and_fixed_k_on_the_same_trials():
    """At k = 2 (fixed): A F 0.6, U 0.3; B F 0.3, U 0.25; so fixed dF 0.3, dU 0.05, dN 0.25. Matched at 0.3: dF 0.22,
    dU 0.07, dN 0.15; matched minus fixed: dF -0.08, dU +0.02, dN -0.10. One task, so by_task equals the pooled values."""
    v = MS.matched_split(_rows(), CFG, [("A", "B")], [0.3], seed=1, pk=2)["A - B"]["0.3"]
    fs, mf = v["fixed_same"], v["matched_minus_fixed"]
    assert fs["F"]["estimate"] == pytest.approx(0.3) and fs["U"]["estimate"] == pytest.approx(0.05) and fs["N"]["estimate"] == pytest.approx(0.25)
    assert mf["F"]["estimate"] == pytest.approx(-0.08) and mf["U"]["estimate"] == pytest.approx(0.02) and mf["N"]["estimate"] == pytest.approx(-0.10)
    assert v["by_task"]["R"]["matched_minus_fixed"]["F"]["estimate"] == pytest.approx(-0.08) and v["by_task"]["R"]["trials"] == 4


def test_reproduction_flags_a_pooled_F_that_differs_from_the_report():
    rep = {"primary_k": 2, "summary": {"pooled": {"A": {"F": {"2": 0.6}}, "B": {"F": {"2": 0.31}}}}}
    bad = MS.reproduction(_rows(), CFG, rep, ["A", "B"])
    assert len(bad) == 1 and bad[0].startswith("B:")
