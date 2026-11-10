"""The decomposition: the selected-tail quantities and their integrity checks, the decomposition's identity and reading rule,
the margin-specific tests from recorded resamples (reproduction and matching), the alternative denominators, the overshoot,
the ATP audit, and the command end to end on the tiny model."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from data_fixtures import REPO
from saeselect.analysis import decomposition as A
from saeselect.provenance import EngineeringFailed
from saeselect.runner import execute
from saeselect.stages import selection as S2
from test_posthoc import KS, _cfg, _row

EFFECTS = {"1": 2.0, "2": 1.0, "3": 0.5, "4": -0.5}  # in margin units; the full gain is 8


def _rows(n: int = 6) -> list[dict]:
    """ORACLE picks features 1, 2; JL-OUT picks 3, 1 with a large joint effect; ATP picks 2, 4 with no remainder."""
    rows = []
    for rel in ("R1", "R2"):
        for i in range(n):
            jit = 0.01 * ((i * 7) % 5 - 2)
            r = _row(rel, i, margins={"JL-OUT": 1.0, "ATP": 1.0, "ORACLE": 1.0}, F={"JL-OUT": 0.6, "ATP": 0.2, "ORACLE": 0.4},
                     rho={"JL-OUT": 0.3, "ATP": 0.3, "ORACLE": 0.3})
            r["effects"] = dict(EFFECTS)
            for name, act, f2 in (("JL-OUT", [3, 1], 0.6 + jit), ("ATP", [2, 4], 0.0625), ("ORACLE", [1, 2], 0.4)):
                for k in KS:
                    sel = act[:k] if k <= 2 else act
                    e = r["sets"][name][str(k)]
                    e["active"] = sel
                    e["F"] = sum(EFFECTS[str(f)] for f in sel) / 8.0 if k == 1 else f2
                    e["margin"] = r["m"]["clean"] + e["F"] * 8.0
                r["metrics"][name]["M3"] = {str(k): 0.5 for k in KS}
            r["m"].update({"SAE-ALL": 2.0, "don": 6.0})  # gains 6 and 10 over clean -4
            rows.append(r)
    return rows


def test_tail_values_utility_remainder_and_gap():
    r = _rows(1)[0]
    v = A.tail_values(r, "JL-OUT", 2)
    assert v["U"] == pytest.approx((0.5 + 2.0) / 8.0) and v["N"] == pytest.approx(v["F"] - v["U"]) and v["a"] == 2.0
    assert A._value("ATP", 2, "G")(r) == pytest.approx((2.0 + 1.0) / 8.0 - (1.0 - 0.5) / 8.0)
    assert A._value("ATP", 2, "a_differs")(r) == 0.0
    assert A.tail_values(r, "NORM", 2) is None  # no such selector in the row


def test_integrity_catches_a_missing_effect_and_a_broken_k1_identity():
    rows, cfg = _rows(2), _cfg()
    assert A.integrity_violations(rows, cfg, ["JL-OUT", "ATP", "ORACLE"], KS, 1e-4) == []
    rows[0]["effects"].pop("4")
    rows[1]["sets"]["ATP"]["1"]["F"] += 0.01
    bad = A.integrity_violations(rows, cfg, ["JL-OUT", "ATP", "ORACLE"], KS, 1e-4)
    assert any("no stored effect for [4]" in b for b in bad) and any("at k = 1, (F - U) * gain" in b for b in bad)
    rows[1]["full_gain"] = 0.5  # below the M2 floor: not checked
    assert not any(rows[1]["trial"] in b for b in A.integrity_violations(rows, cfg, ["ATP"], KS, 1e-4))


def test_decomposition_adds_up_and_reads_a_reversal():
    rows, cfg = _rows(), _cfg()
    e = A.decomposition(rows, cfg, [("JL-OUT", "ORACLE")], 2, seed=3)["pooled"]["JL-OUT - ORACLE"]
    assert e["F"]["estimate"] == pytest.approx(e["U"]["estimate"] + e["N"]["estimate"])
    # JL-OUT's singletons are worth less than ORACLE's (dU < 0), yet its set recovers more (dF > 0): the remainder reverses the order
    assert e["U"]["estimate"] < 0 < e["F"]["estimate"] and e["reading"] == "order reversal: the remainder outweighs singleton utility"


@pytest.mark.parametrize("f, u, n, reading", [
    ((0.1, 0.2), (0.1, 0.2), (-0.01, 0.01), "resolved contribution from singleton utility"),
    ((0.1, 0.2), (-0.01, 0.01), (0.1, 0.2), "resolved contribution from the remainder"),
    ((0.1, 0.2), (0.05, 0.1), (0.05, 0.1), "resolved contributions from both"),
    ((0.1, 0.2), (0.3, 0.4), (-0.2, -0.1), "order reversal: singleton utility outweighs the remainder"),
    ((0.1, 0.2), (-0.01, 0.01), (-0.01, 0.01), "unresolved"),
    ((-0.1, 0.1), (0.1, 0.2), (-0.2, -0.1), "opposing contributions, F unresolved"),
    ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), "F unresolved"),
])
def test_reading_rule(f, u, n, reading):
    e = {q: {"estimate": sum(ci) / 2, "ci95": list(ci)} for q, ci in (("F", f), ("U", u), ("N", n))}
    assert A.reading_rule(e) == reading
    assert A.reading_rule({"F": None}) == "not testable"


def _family(rows, cfg):
    return {"JL-OUT - ATP": S2.contrast(rows, "JL-OUT", "ATP", "M2", k=2, cfg=cfg, seed=5),
            "ORACLE - ATP": S2.contrast(rows, "ORACLE", "ATP", "M2", k=2, cfg=cfg, seed=6)}


def test_margin_family_reads_its_recorded_resamples_and_holm():
    rows, cfg = _rows(), _cfg()
    with A.recorded_bootstraps() as calls:
        rec = _family(rows, cfg)
    stored = json.loads(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "_verdict"} for k, v in rec.items()}))
    levels = {"level": 0.95, "tost_level": 0.90}
    fam, bad = A.margin_family(rec, stored, calls, sesoi=lambda n: 0.05, sign=lambda n: 1, alpha_m=0.025, alpha_e=0.05, **levels)
    assert bad == [] and {c["seed"] for c in calls} == {5, 6} and fam["JL-OUT - ATP"]["seed"] == 5
    samples = next(c["res"].samples for c in calls if c["seed"] == 5)
    assert fam["JL-OUT - ATP"]["p_margin"] == pytest.approx(float(np.mean(samples <= 0.05)))
    p = [fam[n]["p_margin"] for n in sorted(fam)]
    assert [fam[n]["p_margin_holm"] for n in sorted(fam)] == pytest.approx(list(A.stats.holm(p)))
    assert fam["JL-OUT - ATP"]["practically_better_adjusted"] and not fam["JL-OUT - ATP"]["equivalent_adjusted"]
    # a report that differs in its 90% interval is a reproduction failure, not a silent match
    stored["ORACLE - ATP"]["ci90"] = [0.0, 1.0]
    _, bad = A.margin_family(rec, stored, calls, sesoi=lambda n: 0.05, sign=lambda n: 1, alpha_m=0.025, alpha_e=0.05, **levels)
    assert len(bad) == 1 and "ci90" in bad[0]
    # the recorder restores the original function
    assert A.stats.cluster_bootstrap.__name__ == "cluster_bootstrap"


def test_reversed_sign_and_adjusted_replication():
    fam = {"D1 R5": {"practically_better_adjusted": True}, "D2 R5": {"equivalent_adjusted": False, "worse_adjusted": True},
           "D1 R6": {"practically_better_adjusted": False}, "D2 R6": {"equivalent_adjusted": True, "worse_adjusted": False}}
    assert A.replication_adjusted(fam, "R5")["replicates"]
    out = A.replication_adjusted(fam, "R6")
    assert not out["replicates"] and "D1 not practically better" in out["reason"]
    assert A.replication_adjusted(fam, "R7") == {"replicates": False, "reason": "not testable"}
    assert A._tails(np.array([-0.2, 0.0, 0.1]), 0.05, -1) == pytest.approx((2 / 3, 1 / 3, 2 / 3))


def test_denominators_use_both_floors_and_the_same_mask():
    rows, cfg = _rows(), _cfg()
    rows[0]["m"]["SAE-ALL"] = -3.5  # SAE-ALL gain 0.5: below the floor, so the trial leaves the SAE-ALL mask only
    out = A.denominators(rows, cfg, {"P2.2": ("JL-OUT", "ATP", "M2")}, ["SAE-ALL", "don"], 2, seed=1)
    assert out["SAE-ALL"]["P2.2"]["n_trials"] == len(rows) - 1 and out["don"]["P2.2"]["n_trials"] == len(rows)
    raw = np.mean([(r["sets"]["JL-OUT"]["2"]["margin"] - r["sets"]["ATP"]["2"]["margin"]) for r in rows])
    assert out["don"]["P2.2"]["raw_margin_gain"]["estimate"] == pytest.approx(raw)
    assert out["don"]["P2.2"]["normalized"]["estimate"] == pytest.approx(raw / 10.0)
    assert out["don"]["P2.2"]["FULL_on_this_mask"]["estimate"] == pytest.approx(raw / 8.0)


def test_overshoot_counts_reached_walks():
    rows, cfg = _rows(2), _cfg()
    for i, r in enumerate(rows):
        r["norm_budget"] = {"sets": {"JL-OUT": {"0.1": {"reached": i % 2 == 0, "rho_intended": 0.3, "norms": [3.0, 3.0]}}}}
    out = A.overshoot(rows, cfg, ["JL-OUT"], 2.0)["pooled"]["JL-OUT"]["0.1"]
    assert out["n_trials"] == len(rows) and out["n_reached"] == len(rows) // 2
    assert out["median"] == pytest.approx(3.0) and out["share_above"] == 1.0 and out["realized_over_intended_median"] == 1.0
    assert A.overshoot(_rows(1), cfg, ["JL-OUT"], 2.0) is None  # no norm budget in these runs


def test_known_row_and_screen():
    rows = _rows(2)
    for i, r in enumerate(rows):
        r["metrics"]["ATP@in"] = {"M1": 0.04668 if i == 0 else (float("nan") if i == 1 else 0.95)}
    known = {"trial": rows[0]["trial"], "cell": "2/concept", "stored_m1": 0.0467}
    assert A.known_row(rows, known) is rows[0]
    with pytest.raises(EngineeringFailed):
        A.known_row(rows, {**known, "stored_m1": 0.5})
    with pytest.raises(EngineeringFailed):
        A.known_row(rows, {**known, "cell": "9/readout"})
    run = {"rows": rows, "cfg": _cfg(), "report": {"primary_k": 2, "analysis": {"secondary": {}}}}
    for r in rows:
        r["metrics"]["LL-OUT"] = {"M1": 0.1}
        r["sets"]["LL-OUT"], r["sets"]["ATP@in"] = r["sets"]["JL-OUT"], r["sets"]["ATP"]
    out = A.atp_screen(run, 0.5, {rows[0]["pair"]}, seed=1)
    assert out["screen"]["n_flagged"] == 1 and len(out["screen"]["nonfinite_or_missing"]) == 1
    assert set(out["influence"]) == {"known", "screen"} and out["influence"]["known"]["pairs_left_out"] == [rows[0]["pair"]]


@pytest.mark.integration
def test_decomposition_end_to_end(tmp_path, git_repo):
    from test_inventory import _build, _tiny_model_config
    from test_selection import _config, _stage0

    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    config = _config(tmp_path, name, _stage0(tmp_path, model))
    dev, test = tmp_path / "runs" / "s2-dev", tmp_path / "runs" / "s2-test"
    opts = {"config": str(config), "model": str(model), "phase": "dev", "out": str(dev), "device": "cpu"}
    assert execute("select", opts, S2.run, printer=lambda _: None, repo_root=git_repo) == 0
    frozen = dev / "frozen_protocol.json"
    assert execute("select", {**opts, "phase": "test", "frozen": str(frozen), "out": str(test)}, S2.run, printer=lambda _: None,
                   repo_root=git_repo) == 0
    from saeselect.analysis import rebuild as R
    row = next(r for r in R.read_run(test)["rows"] if r["metrics"].get("ATP@in", {}).get("M1") is not None)
    a43 = yaml.safe_load((REPO / "configs/analysis/decomposition.yaml").read_text(encoding="utf-8"))
    a43["stage2"] = {"tiny": {"run": str(test), "frozen": str(frozen)}}
    a43["extensions"] = {}
    a43["atp_audit"]["known"] = {"model": "tiny", "trial": row["trial"], "cell": f"{row['layer']}/{row['site']}",
                                 "stored_m1": round(row["metrics"]["ATP@in"]["M1"], 4), "recomputed_m1": 0.0, "source": "test"}
    path = tmp_path / "a43.yaml"
    path.write_text(yaml.safe_dump(a43), encoding="utf-8")
    out = tmp_path / "analysis" / "a43"
    code = execute("decomposition", {"config": str(path), "out": str(out)}, A.run, printer=lambda _: None, repo_root=git_repo)
    err = (out / "error.txt").read_text(encoding="utf-8") if (out / "error.txt").exists() else ""
    assert code == 0, (out / "SUMMARY.md").read_text(encoding="utf-8") + err
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    m = rep["runs"]["tiny"]
    assert m["tail"]["pooled"] and m["decomposition"]["pooled"] and m["pooling"]["common_mask"]
    assert set(m["margin_tests"]["primary"]) == {"P2.1", "P2.2"} and rep["atp_audit"]["known"]["trial"] == row["trial"]
