"""The off-target endpoint: the off-target quantities, the rank-average hybrids, and an endpoint extension of a frozen Stage 2 protocol on the
tiny model (dev pass, freeze, single test, rebuild)."""

from __future__ import annotations

import json
import math
import zipfile

import pytest
import torch
import yaml

from saeselect import signals as S
from saeselect.analysis import rebuild as R
from saeselect.readout import off_target
from saeselect.runner import execute
from saeselect.stages import selection as S2


def test_rank_average_orders_by_mean_rank_and_breaks_ties_by_norm():
    norm = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0}
    kappa = {1: 0.0, 2: 9.0, 3: 8.0, 4: 1.0}
    h = S.rank_average(norm, kappa, [1, 2, 3, 4], tiebreak=norm)
    # mean ranks: 1 -> (0 + 3) / 2, 2 -> (1 + 0) / 2, 3 -> (2 + 1) / 2, 4 -> (3 + 2) / 2; features 1 and 3 tie, NORM prefers 1
    assert S.select_top(h, 4, [1, 2, 3, 4]) == [2, 1, 3, 4]
    assert S.select_top(S.rank_average(norm, norm, [1, 2, 3, 4], tiebreak=norm), 2, [1, 2, 3, 4]) == [1, 2]


def test_off_target_quantities():
    lp_clean = torch.log(torch.tensor([0.4, 0.1, 0.25, 0.25], dtype=torch.float64))  # tokens 0 and 1 are the answers
    lp_full = torch.log(torch.tensor([0.1, 0.5, 0.2, 0.2], dtype=torch.float64))
    same = off_target(lp_clean, lp_clean, lp_full, [0, 1])
    assert same["O1"] == pytest.approx(0.0, abs=1e-12) and same["O2"] == pytest.approx(0.0, abs=1e-12) and same["O3"] > 0
    full = off_target(lp_full, lp_clean, lp_full, [0, 1])
    assert full["O3"] == pytest.approx(0.0, abs=1e-12)
    assert full["O1"] == pytest.approx(0.0, abs=1e-12)  # outside the answers both are (0.5, 0.5) after renormalizing
    assert full["O2"] == pytest.approx(0.5 - 0.6)  # answer mass rises from 0.5 to 0.6
    lp = torch.log(torch.tensor([0.4, 0.1, 0.4, 0.1], dtype=torch.float64))
    o = off_target(lp, lp_clean, lp_full, [0, 1])
    assert o["O1"] == pytest.approx(0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)) and o["O2"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.integration
def test_endpoint_extension_dev_test_and_rebuild(tmp_path, git_repo):
    from test_inventory import _build, _tiny_model_config
    from test_selection import _config, _stage0

    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    parent_cfg = _config(tmp_path, name, _stage0(tmp_path, model))
    parent = tmp_path / "runs" / "s2-dev"
    lines: list[str] = []
    assert execute("select", {"config": str(parent_cfg), "model": str(model), "phase": "dev", "out": str(parent), "device": "cpu"},
                   S2.run, printer=lines.append, repo_root=git_repo) == 0
    parent_frozen = json.loads((parent / "frozen_protocol.json").read_text(encoding="utf-8"))

    x8 = tmp_path / "stage2_x8.yaml"
    x8.write_text(yaml.safe_dump({"base": str(parent_cfg), "extension": {"id": "X8", "kind": "endpoint",
                                                                         "parent_frozen": str(parent / "frozen_protocol.json")},
                                  "k_grid": [1, 2], "alpha_grid": [1], "offtarget": True,
                                  "hybrids": {"NORM+KAPPA": ["NORM", "KAPPA"], "NORM+JSPACE": ["NORM", "JSPACE"]},
                                  "offtarget_contrasts": [["JL-OUT", "LL-OUT"], ["NORM+KAPPA", "NORM"]]}), encoding="utf-8")
    dev = tmp_path / "runs" / "x8-dev"
    opts = {"config": str(x8), "model": str(model), "phase": "dev", "out": str(dev), "device": "cpu"}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (dev / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=FROZEN_AWAITING_APPROVAL"), summary + ((dev / "error.txt").read_text() if (dev / "error.txt").exists() else "")
    frozen = json.loads((dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert frozen["extension"]["kind"] == "endpoint" and frozen["parent_protocol_hash"] == parent_frozen["protocol_hash"]
    assert frozen["relations"] == parent_frozen["relations"] and frozen["eligible_cells"] == parent_frozen["eligible_cells"]
    assert frozen["primary_k"] == parent_frozen["primary_k"]
    rows = [json.loads(x) for x in zipfile.ZipFile(dev / "results.zip").read("trials.jsonl").decode().splitlines()]
    r = rows[0]
    assert {"NORM+KAPPA", "NORM+JSPACE"} <= set(r["sets"]) and set(r["sets"]["NORM"]["1"]["offtarget"]) == {"O1", "O2", "O3"}
    assert r["offtarget_ref"]["clean"]["O1"] == pytest.approx(0.0, abs=1e-9) and r["offtarget_ref"]["FULL"]["O3"] == pytest.approx(0.0, abs=1e-9)
    assert all("offtarget" in d for d in r["random"]["1"])
    report = json.loads((dev / "report.json").read_text(encoding="utf-8"))
    assert set(report["endpoint"]["hybrids"]["M2"]) == {"NORM+KAPPA", "NORM+JSPACE"}
    assert report["endpoint"]["full_margin_parity_max"] < 1e-6
    assert "## Hybrid selectors (confirmatory)" in summary and "## Off-target readout (descriptive)" in summary

    test = tmp_path / "runs" / "x8-test"
    opts = {**opts, "phase": "test", "frozen": str(dev / "frozen_protocol.json"), "out": str(test)}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), (test / "SUMMARY.md").read_text(encoding="utf-8")
    report = json.loads((test / "report.json").read_text(encoding="utf-8"))
    assert "p_holm" in report["endpoint"]["hybrids"]["M2"]["NORM+KAPPA"] and "JL-OUT - LL-OUT" in report["endpoint"]["offtarget"]["2"]["contrasts"]

    # M8: the endpoint extension's standard tables are rebuilt exactly
    out = tmp_path / "analysis" / "x8-test"
    code = execute("analyze", {"run": str(test), "frozen": str(dev / "frozen_protocol.json"), "out": str(out)}, R.run,
                   printer=lines.append, repo_root=git_repo)
    rebuilt = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and rebuilt["reproduction"]["mismatches"] == [] and rebuilt["reproduction"]["summary_identical"], (out / "SUMMARY.md").read_text()
