"""The additivity analysis: ceilings, superadditivity, interpolation-only matching and alpha > 1 shares on fabricated trials,
and the command end to end on the tiny model (reproduction checked against the report)."""

from __future__ import annotations

import json

import pytest
import yaml

from data_fixtures import REPO
from saeselect.analysis import additivity as A
from saeselect.runner import execute
from test_posthoc import KS, _cfg, _row


def _rows():
    rows = []
    for rel in ("R1", "R2"):
        for i in range(6):
            r = _row(rel, i, margins={"JL-OUT": 1.0, "NORM": 1.0}, F={"JL-OUT": 0.4, "NORM": 0.6}, rho={"JL-OUT": 0.3, "NORM": 0.6})
            r["m"].update({"SAE-ALL": 0.0, "ERR": -3.0})  # clean -4, FULL 4: SAE-ALL recovers 0.5, ERR 0.125
            r["pool_size"] = 10 + i
            r["effects"] = {"1": 0.8, "2": 0.8, "3": 9.0}
            for s in ("JL-OUT", "NORM"):
                for k in KS:
                    r["sets"][s][str(k)]["active"] = [1, 2]
            rows.append(r)
    return rows


def test_ceilings_and_relative_recovery():
    cfg = _cfg()
    out = A.ceilings(_rows(), cfg, ["JL-OUT", "NORM"], KS, 4, seed=1)["2/concept"]
    assert out["SAE-ALL"]["median"] == pytest.approx(0.5) and out["ERR"]["median"] == pytest.approx(0.125)
    assert out["relative_to_SAE_ALL"]["NORM"]["estimate"] == pytest.approx(0.6 / 0.5)
    assert out["pool_size"]["median"] == pytest.approx(12.5)


def test_superadditivity_sums_the_active_features_only():
    out = A.superadditivity(_rows(), _cfg(), ["NORM"], [4], seed=1)["2/concept"]["NORM"]["4"]
    assert out["median_additive"] == pytest.approx(1.6 / 8.0)  # features 1 and 2 (0.8 each) over the full gain 8; feature 3 is not in the set
    assert out["median_F"] == pytest.approx(0.6) and out["ratio_of_medians"] == pytest.approx(0.6 / 0.2)


def test_alpha_shares_and_interpolation_only_matching():
    cfg = _cfg()
    spec = {"signals": ["JL-OUT", "NORM"], "rho": [0.25, 0.5], "pairs": [["NORM", "JL-OUT"]]}
    out = A.curves(_rows(), cfg, spec, KS, seed=1)
    # unscaled rho: JL-OUT 0.3, NORM 0.6; reaching 0.5 needs alpha > 1 for JL-OUT on every trial, never for NORM
    assert out["alpha_above_one_share"]["JL-OUT"]["0.5"] == 1.0 and out["alpha_above_one_share"]["NORM"]["0.5"] == 0.0
    m = out["matched_interpolation_only"]["NORM - JL-OUT"]
    assert m["0.5"]["difference"] is None and m["0.5"]["trials_not_reached"] == 12  # JL-OUT's path stops at alpha = 1 (rho 0.3)
    # at 0.25 both interpolate below alpha = 1: NORM 0.6 * 0.25/0.6 = 0.25, JL-OUT 0.4 * 0.25/0.3 = 0.333
    assert m["0.25"]["difference"]["estimate"] == pytest.approx(0.25 - 0.4 * 0.25 / 0.3)


def test_sae_carried_cells_use_the_development_median():
    cfg = _cfg()
    dev = _rows()
    out = A.sae_carried_cells(dev, cfg, 0.5)
    assert out["cells"] == ["2/concept"]
    assert A.sae_carried_cells(dev, cfg, 0.6)["cells"] == []


@pytest.mark.integration
def test_additivity_end_to_end(tmp_path, git_repo):
    from saeselect.stages import selection as S2
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
    a34 = yaml.safe_load((REPO / "configs/analysis/additivity.yaml").read_text(encoding="utf-8"))
    a34.update(models={"tiny": {"run": str(test), "frozen": str(frozen)}}, offtarget={},
               size={"signals": ["NORM", "JL-OUT", "LL-OUT"], "rho": [0.1], "pairs": [["JL-OUT", "LL-OUT"]]})
    path = tmp_path / "a34.yaml"
    path.write_text(yaml.safe_dump(a34), encoding="utf-8")
    out = tmp_path / "analysis" / "a34"
    lines: list[str] = []
    code = execute("additivity", {"config": str(path), "out": str(out)}, A.run, printer=lines.append, repo_root=git_repo)
    err = (out / "error.txt").read_text(encoding="utf-8") if (out / "error.txt").exists() else ""
    assert code == 0, (out / "SUMMARY.md").read_text(encoding="utf-8") + err
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))["models"]["tiny"]
    assert rep["ceilings"] and rep["superadditivity"] and rep["jl_vs_norm"]["pooled"]
    assert set(rep["sae_carried"]) >= {"cells", "dev_median_SAE_ALL", "primary"}
    assert rep["candidate_cells"]["eligible"] is not None
