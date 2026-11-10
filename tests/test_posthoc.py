"""The post hoc analysis: SESOI readings, absolute endpoints with ties, size matching, leave-one-out, fidelity correlations,
and the command end to end on the tiny model (reproduction checked against the reports, a changed report fails)."""

from __future__ import annotations

import json

import pytest
import yaml

from data_fixtures import REPO
from saeselect.analysis import compare_sets as C
from saeselect.analysis import posthoc as A
from saeselect.runner import execute
from saeselect.stages import selection as S2

KS = [1, 2, 4]


def _cfg() -> dict:
    cfg = yaml.safe_load((REPO / "configs/selection/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg["analysis"]["bootstrap"]["n_resamples"] = 500
    cfg["k_grid"] = KS
    cfg["m2_min_full_gain"] = 1.0
    return cfg


def _row(rel: str, i: int, *, margins: dict, F: dict, rho: dict, clean=-4.0, full=4.0, full_gain=8.0, m1=None) -> dict:
    """One fabricated trial: per selector a margin, F and relative norm at every k (FULL's realized norm is 10)."""
    sets = {s: {str(k): {"margin": margins[s], "F": F[s] * k / max(KS), "norms": [rho[s] * k / max(KS) * 10.0] * 2} for k in KS} for s in margins}
    return {"pair": f"{rel}:{i}", "relation": rel, "trial": f"{rel}:{i}:t", "layer": 2, "site": "concept", "full_gain": full_gain,
            "m": {"clean": clean, "FULL": full}, "ref_norms": {"FULL": [10.0, 10.0]}, "sets": sets,
            "random": {str(k): [{"margin": -3.9, "F": 0.01, "norms": [0.5, 0.5]}, {"margin": 0.0, "F": 0.02, "norms": [0.7, 0.7]}] for k in KS},
            "metrics": {s: {"M1": (m1 or {}).get(s, 0.1)} for s in margins},
            "alpha": {s: [{"alpha": a, "effect": a * F[s] * full_gain, "realized_norm": a * rho[s] * 10.0} for a in (2.0, 0.5, 1.0)] for s in margins}}


def _run(rows: list[dict], cfg: dict, pk: int = 4) -> dict:
    return {"rows": rows, "cfg": cfg, "report": {"primary_k": pk}, "s0": {"tau_tie": {"R1": 0.05, "R2": 0.05}},
            "provenance": {"stage0_decisions": "s0/stage0_decisions.json"}}


def test_reading_follows_the_sesoi():
    cfg = _cfg()
    rows = [_row(rel, i, margins={"JL-OUT": 1.0, "LL-OUT": 1.0}, F={"JL-OUT": 0.54 + 0.01 * ((i * 7) % 5 - 2), "LL-OUT": 0.5},
                 rho={"JL-OUT": 0.2, "LL-OUT": 0.2}) for rel in ("R1", "R2") for i in range(12)]
    read = {s: A.reading(S2.contrast(rows, "JL-OUT", "LL-OUT", "M2", k=4, cfg=A._with_sesoi(cfg, s), seed=3)) for s in (0.025, 0.05, 0.10)}
    assert read == {0.025: "practically better", 0.05: "above 0, within the margin", 0.10: "above 0, within the margin"}
    assert A.reading(None) == "not testable"
    assert A.reading({"better": False, "worse": False, "equivalent": False, "practically_better": False}) == "inconclusive"
    assert A.reading({"better": False, "worse": True, "equivalent": True, "practically_better": False}) == "below 0, within the margin"


def test_switch_leaves_ties_out_and_counts_them():
    assert A.switch(0.2, 0.05) == 1.0 and A.switch(-0.2, 0.05) == 0.0 and A.switch(0.03, 0.05) is None
    cfg = _cfg()
    rows = [_row("R1", i, margins={"JL-OUT": [0.5, 0.01, -0.5, 0.5][i % 4], "LL-OUT": -1.0}, F={"JL-OUT": 0.5, "LL-OUT": 0.1},
                 rho={"JL-OUT": 0.2, "LL-OUT": 0.2}) for i in range(8)]
    rows.append(_row("R1", 99, margins={"JL-OUT": 9.0, "LL-OUT": 9.0}, F={"JL-OUT": 1.0, "LL-OUT": 1.0}, rho={"JL-OUT": 1, "LL-OUT": 1},
                     full_gain=0.5))  # below the M2 floor: left out everywhere
    out = A.absolute_endpoints(_run(rows, cfg), {"k": [4], "signals": ["JL-OUT", "LL-OUT"], "contrasts": [["JL-OUT", "LL-OUT"]]}, seed=1)
    sel = out["by_k"]["4"]["selectors"]
    assert abs(sel["JL-OUT"]["switch"]["estimate"] - 2 / 3) < 1e-12  # 4 switched, 2 not, 2 ties left out
    assert abs(sel["JL-OUT"]["tie"]["estimate"] - 0.25) < 1e-12 and sel["JL-OUT"]["switch"]["n_pairs"] == 6 and sel["JL-OUT"]["tie"]["n_pairs"] == 8
    assert sel["LL-OUT"]["switch"]["estimate"] == 0.0 and sel["FULL"]["switch"]["estimate"] == 1.0 and sel["clean"]["switch"]["estimate"] == 0.0
    assert abs(sel["RANDOM"]["switch"]["estimate"]) < 1e-12 and abs(sel["RANDOM"]["tie"]["estimate"] - 0.5) < 1e-12  # one draw is a tie
    assert abs(sel["JL-OUT"]["delta_m"]["estimate"] - (4.0 + 0.1275)) < 1e-12
    c = out["by_k"]["4"]["contrasts"]["JL-OUT - LL-OUT"]
    assert c["switch"]["n_pairs"] == 6 and abs(c["switch"]["estimate"] - 4 / 6) < 1e-12
    with pytest.raises(Exception, match="no tau_tie"):
        A.absolute_endpoints(_run([{**rows[0], "relation": "R9"}], cfg), {"k": [4], "signals": ["JL-OUT"], "contrasts": []}, seed=1)


def test_reproduction_lists_fields_an_older_report_lacks():
    bad = ["a24: /x/32/practically_better: in the rebuild only", "a24: /x/32/estimate: report 0.1, rebuild 0.2", "a24: /x/4/p: in the report only"]
    failures, extra = A.split_mismatches(bad)
    assert failures == bad[1:] and extra == ["a24: /x/32/practically_better"]


def test_size_matching_along_the_k_path():
    pts = [(0.0, 0.0), (0.1, 0.2), (0.3, 0.4)]
    assert A.at_rho(pts, 0.05) == pytest.approx(0.1) and A.at_rho(pts, 0.2) == pytest.approx(0.3) and A.at_rho(pts, 0.5) is None
    assert A.at_rho([(0.0, 0.0), (0.2, 0.5), (0.1, 0.9)], 0.15) == pytest.approx(0.375)  # first crossing only
    cfg = _cfg()
    # NORM: twice the size of JL-OUT and twice the recovery at every k, so at a matched size the two are equal
    rows = [_row(rel, i, margins={"NORM": 1.0, "JL-OUT": 1.0}, F={"NORM": 0.8, "JL-OUT": 0.4}, rho={"NORM": 0.4, "JL-OUT": 0.2})
            for rel in ("R1", "R2") for i in range(6)]
    out = A.size_analysis(_run(rows, cfg), {"signals": ["NORM", "JL-OUT"], "rho": [0.1, 0.3], "pairs": [["NORM", "JL-OUT"]]}, seed=2)
    assert out["at_primary_k"]["NORM"]["rho"]["estimate"] == pytest.approx(0.4) and out["at_primary_k"]["NORM"]["F"]["estimate"] == pytest.approx(0.8)
    assert out["at_primary_k"]["RANDOM"]["rho"]["estimate"] == pytest.approx(0.06)
    m = out["matched"]["k"]["NORM - JL-OUT"]
    assert m["0.1"]["difference"]["estimate"] == pytest.approx(0.0) and m["0.1"]["trials_not_reached"] == 0
    assert m["0.3"]["difference"] is None and m["0.3"]["trials_not_reached"] == 12  # JL-OUT never reaches 0.3 along k
    a = out["matched"]["alpha"]["NORM - JL-OUT"]  # scaled by alpha up to 2, JL-OUT reaches 0.4: the same F per unit size
    assert a["0.3"]["difference"]["estimate"] == pytest.approx(0.0) and a["0.3"]["trials_not_reached"] == 0
    assert A.alpha_path({"alpha": {}, "ref_norms": {"FULL": [1.0, 1.0]}, "full_gain": 1.0}, "NORM") is None


def test_leave_one_relation_out_and_fidelity_correlations():
    cfg = _cfg()
    rows = [_row(rel, i, margins={"JL-OUT": 1.0, "LL-OUT": 1.0}, F={"JL-OUT": 0.5 + (0.2 if rel == "R1" else 0.0) + 0.01 * (i % 3), "LL-OUT": 0.5},
                 rho={"JL-OUT": 0.2, "LL-OUT": 0.2}, m1={"JL-OUT": 0.3 + 0.01 * (i % 3), "LL-OUT": 0.1}) for rel in ("R1", "R2", "R3") for i in range(6)]
    out = A.leave_one_out(_run(rows, cfg), seed=4)
    assert set(out) == {"R1", "R2", "R3"} and out["R1"]["pairs"] == 12
    assert out["R1"]["P2.2"]["estimate"] < 0.02 < out["R2"]["P2.2"]["estimate"]  # R1 carries the set advantage
    combos = [{"family": f, "variance_explained": ve, "M1": {"estimate": e}, "M2_k4": None} for f, ve, e in
              (("gemma", 0.8, 0.1), ("gemma", 0.9, 0.2), ("gemma", 0.95, 0.3), ("llama", 0.6, 0.0))]
    cor = A.fidelity_correlations(combos, [4])
    assert cor["all"]["M1"] == {"spearman": pytest.approx(1.0), "n": 4} and cor["gemma"]["M1"]["n"] == 3
    assert cor["all"]["M2_k4"] == {"spearman": None, "n": 0}


@pytest.mark.integration
def test_posthoc_end_to_end(tmp_path, git_repo):
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
    assert execute("select", {**opts, "phase": "test", "frozen": str(frozen), "out": str(test)}, S2.run, printer=lambda _: None, repo_root=git_repo) == 0
    entry = {"run": str(test), "frozen": str(frozen), "matched_cells": ["2/concept", "3/concept"], "status": "post hoc"}
    # an the cross-model comparison-like family (default contrasts) and an the SAE swap-like one (declared contrasts, Holm at one k), each analysed by compare-sets
    fams = {"a24": {"models": {"a": entry, "b": entry}, "reference": "a"},
            "a25": {"models": {"a": entry, "b": entry}, "contrasts": [{"name": "b over a", "model": "b", "reference": "a"}], "holm_k": 2}}
    for tag, fam in fams.items():
        fam.update(matched_relations=["R1_country_capital"], cross_model_k=[1, 2], holm_alpha=0.05, seed=7)
        (tmp_path / f"{tag}.yaml").write_text(yaml.safe_dump(fam), encoding="utf-8")
        assert execute("compare-sets", {"config": str(tmp_path / f"{tag}.yaml"), "out": str(tmp_path / "analysis" / tag)}, C.run,
                       printer=lambda _: None, repo_root=git_repo) == 0
    a27 = yaml.safe_load((REPO / "configs/analysis/posthoc.yaml").read_text(encoding="utf-8"))
    run_entry = {"run": str(test), "frozen": str(frozen), "family": "gemma", "release": "own"}
    a27.update(models={"tiny": run_entry}, sae_runs={},
               sesoi={"grid": [0.025, 0.05, 0.10], "cross": [{"config": str(tmp_path / f"{t}.yaml"), "report": str(tmp_path / "analysis" / t / "report.json")}
                                                         for t in fams]},
               absolute={"k": [1, 2], "signals": ["NORM", "LL-OUT", "JL-OUT", "ORACLE"], "contrasts": [["JL-OUT", "LL-OUT"]]},
               size={"signals": ["NORM", "JL-OUT"], "rho": [0.05], "pairs": [["NORM", "JL-OUT"]]},
               swap={"config": str(tmp_path / "a25.yaml"), "k": [2]}, fidelity={"k": [1, 2]})
    cfg_path = tmp_path / "a27.yaml"
    cfg_path.write_text(yaml.safe_dump(a27), encoding="utf-8")
    lines: list[str] = []
    out_dir = tmp_path / "analysis" / "a27"
    assert execute("posthoc", {"config": str(cfg_path), "out": str(out_dir)}, A.run, printer=lines.append, repo_root=git_repo) == 0, \
        (out_dir / "SUMMARY.md").read_text(encoding="utf-8")
    rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    test_report = json.loads((test / "report.json").read_text(encoding="utf-8"))
    p21 = rep["sesoi"]["runs"]["tiny"]["P2.1"]
    assert p21["0.05"]["estimate"] == test_report["analysis"]["primary"]["P2.1"]["estimate"]
    assert p21["0.025"]["ci95"] == p21["0.1"]["ci95"]  # the interval does not depend on the SESOI
    assert set(rep["sesoi"]["cross"]) == {"a24", "a25"} and "b over a, k = 2" in rep["sesoi"]["cross"]["a25"]
    assert set(rep["absolute"]["tiny"]["by_k"]) == {"1", "2"}  # the declared k (the primary k is one of them)
    assert rep["fidelity"]["combinations"] and rep["fidelity"]["combinations"][0]["sae"] == "stage0"
    assert set(rep["swap_cells"]["cells"]) == {"2/concept", "3/concept"}
    summary = (out_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "## 6. Leave one relation out" in summary and "post hoc" in summary

    # a changed number in an analysis report: the declared-SESOI recomputation no longer matches, and the command fails loudly
    bad = json.loads((tmp_path / "analysis" / "a25" / "report.json").read_text(encoding="utf-8"))
    bad["cross_model"]["b over a"]["2"]["estimate"] += 0.5
    (tmp_path / "analysis" / "a25" / "report.json").write_text(json.dumps(bad), encoding="utf-8")
    lines = []
    assert execute("posthoc", {"config": str(cfg_path), "out": str(tmp_path / "analysis" / "a27-bad")}, A.run, printer=lines.append,
                   repo_root=git_repo) == 2
    assert "GATE_FAILED:reproduction" in lines[-1]
