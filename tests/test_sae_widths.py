"""SAE widths: Stage 0 on declared SAE variants, then Stage 2 rerun with only the SAE changed,
dev, freeze and test, end to end on the tiny Gemma 3 with the fixture build."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from saeselect import cli
from saeselect.analysis import rebuild as R
from saeselect.runner import execute
from saeselect.stages import selection as S2
from test_inventory import _build, _stage0_config, _tiny_model_config
from test_selection import _config, _stage0


def _model_with_variants(tmp: Path) -> Path:
    path = _tiny_model_config(tmp)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["sae"]["robustness"] = [{"width": cfg["sae"]["width"] + 32}, {"width": cfg["sae"]["width"] + 64}]
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.integration
def test_sae_widths_dev_freezes_and_test_runs_once(tmp_path, git_repo):
    _build(tmp_path)
    model = _model_with_variants(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    parent_cfg = _config(tmp_path, name, _stage0(tmp_path, model))
    lines: list[str] = []
    parent = tmp_path / "runs" / "s2-dev"
    assert execute("select", {"config": str(parent_cfg), "model": str(model), "phase": "dev", "out": str(parent), "device": "cpu"},
                   S2.run, printer=lines.append, repo_root=git_repo) == 0, (parent / "SUMMARY.md").read_text(encoding="utf-8")

    # the declared variants at layer 2: fidelity, feature statistics and kappa
    s0 = yaml.safe_load(Path(_stage0_config(tmp_path, tmp_path / "built", floor=1.5)).read_text(encoding="utf-8"))
    s0_path = tmp_path / "s0_x4.yaml"
    s0_path.write_text(yaml.safe_dump({"base": str(_stage0_config(tmp_path, tmp_path / "built", floor=1.5)), "scope": "sae_variants",
                                       "sae_variant_layers": [2], "sae_variant_list": "robustness"}), encoding="utf-8")
    variants = tmp_path / "runs" / "s0-x4"
    assert cli.main(["inventory", "--config", str(s0_path), "--model", str(model), "--out", str(variants)]) == 0
    decisions = json.loads((variants / "stage0_decisions.json").read_text(encoding="utf-8"))
    assert decisions["scope"] == "sae_variants" and len(decisions["sae_variants"]) == 2
    for key, v in decisions["sae_variants"].items():
        assert v["layer"] == 2 and (variants / "features" / f"{key}.safetensors").is_file()
    assert s0["checks"]  # the base config's declared values are used

    ext = tmp_path / "stage2_x4.yaml"
    ext.write_text(yaml.safe_dump({"base": str(parent_cfg), "extension": {
        "id": "X4", "parent_frozen": str(parent / "frozen_protocol.json"), "cells": ["2/concept"], "sae_run": str(variants)}}), encoding="utf-8")
    dev = tmp_path / "runs" / "x4-dev"
    opts = {"config": str(ext), "model": str(model), "phase": "dev", "out": str(dev), "device": "cpu"}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (dev / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=FROZEN_AWAITING_APPROVAL"), summary + ((dev / "error.txt").read_text() if (dev / "error.txt").exists() else "")
    parent_frozen = json.loads((parent / "frozen_protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert frozen["eligible_cells"] == ["2/concept"] and frozen["primary_k"] == parent_frozen["primary_k"]
    assert frozen["relations"] == parent_frozen["relations"] and frozen["parent_protocol_hash"] == parent_frozen["protocol_hash"]
    assert set(frozen["extension"]["variants"]) == set(decisions["sae_variants"])
    report = json.loads((dev / "report.json").read_text(encoding="utf-8"))
    assert set(report["by_sae"]) == set(decisions["sae_variants"]) and "summary" not in report
    rows = [json.loads(x) for x in zipfile.ZipFile(dev / "results.zip").read("trials.jsonl").decode().splitlines()]
    assert {r["sae"] for r in rows} == set(decisions["sae_variants"]) and {(r["layer"], r["site"]) for r in rows} == {(2, "concept")}
    assert all("alpha" in r for r in rows)  # the inherited primary k drives the alpha-curves in the main pass
    assert "extension X4" in summary and "JL-OUT minus LL-OUT per variant" in summary

    test = tmp_path / "runs" / "x4-test"
    opts = {**opts, "phase": "test", "frozen": str(dev / "frozen_protocol.json"), "out": str(test)}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (test / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), summary + ((test / "error.txt").read_text() if (test / "error.txt").exists() else "")
    assert set(json.loads((test / "report.json").read_text(encoding="utf-8"))["by_sae"]) == set(decisions["sae_variants"])
    code = execute("select", {**opts, "out": str(tmp_path / "runs" / "x4-test-2")}, S2.run, printer=lines.append, repo_root=git_repo)
    assert code == 1 and "test marker" in lines[-2]

    # M8: the extension's test run is rebuilt per variant from its results.zip
    out = tmp_path / "analysis" / "x4-test"
    code = execute("analyze", {"run": str(test), "frozen": str(dev / "frozen_protocol.json"), "out": str(out)}, R.run,
                   printer=lines.append, repo_root=git_repo)
    rebuilt = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert set(rebuilt["rebuilt"]["by_sae"]) == set(rebuilt["random_baseline"]) == set(decisions["sae_variants"])
    assert rebuilt["reproduction"]["mismatches"] == [] and rebuilt["reproduction"]["summary_identical"]
