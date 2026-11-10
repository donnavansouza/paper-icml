"""Model extension: Stage 2 with only the model changed. The parent protocol is frozen on one tiny Gemma 3; a second
tiny model (another seed and name, standing in for 12B-PT) gets its own Stage 0 on the relation families, then the
extension's dev pass, freeze, single test run and rebuild."""

from __future__ import annotations

import json
import shutil
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


def _second_model(tmp: Path) -> Path:
    cfg = yaml.safe_load(_tiny_model_config(tmp).read_text(encoding="utf-8"))
    cfg.update(name="tiny_gemma3_test_pt", seed=1)
    path = tmp / "tiny_pt_model.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _stage0_relations(tmp: Path, model: Path, *, layers: tuple[str, ...]) -> Path:
    """Stage 0 scoped like configs/inventory/gemma_base.yaml (relation families only), relabelled as a full run at ``layers``."""
    scoped = tmp / "stage0_x3_test.yaml"
    scoped.write_text(yaml.safe_dump({"base": str(_stage0_config(tmp, tmp / "built", floor=1.5)),
                                      "families": ["R1_country_capital", "R2_element_symbol"]}), encoding="utf-8")
    out = tmp / "runs" / "s0-pt"
    assert cli.main(["inventory", "--config", str(scoped), "--model", str(model), "--out", str(out), "--limit", "3"]) == 0
    decisions = json.loads((out / "stage0_decisions.json").read_text(encoding="utf-8"))
    assert decisions.get("scope", "full") == "full" and set(decisions["tau_tie"]) == {"R1_country_capital", "R2_element_symbol"}
    full = tmp / "runs" / "s0-pt-full"
    full.mkdir(parents=True)
    shutil.copytree(out / "features", full / "features")
    decisions.update(smoke=False, batching_allowed=True, suffix_caching_allowed=True, sae={k: v for k, v in decisions["sae"].items() if k in layers})
    (full / "stage0_decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    return full


@pytest.mark.integration
def test_model_extension_inherits_the_parent_protocol(tmp_path, git_repo):
    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    parent_cfg = _config(tmp_path, name, _stage0(tmp_path, model))
    lines: list[str] = []
    parent = tmp_path / "runs" / "s2-dev"
    assert execute("select", {"config": str(parent_cfg), "model": str(model), "phase": "dev", "out": str(parent), "device": "cpu"},
                   S2.run, printer=lines.append, repo_root=git_repo) == 0, (parent / "SUMMARY.md").read_text(encoding="utf-8")
    parent_frozen = json.loads((parent / "frozen_protocol.json").read_text(encoding="utf-8"))

    pt = _second_model(tmp_path)
    pt_name = "tiny_gemma3_test_pt"
    s0 = _stage0_relations(tmp_path, pt, layers=("2",))  # no usable SAE at layer 3: its inherited cells are dropped, flagged
    x3 = tmp_path / "stage2_x3.yaml"
    x3.write_text(yaml.safe_dump({"base": str(parent_cfg), "stage0_runs": {pt_name: str(s0)},
                                  "extension": {"id": "X3", "kind": "model", "parent_frozen": str(parent / "frozen_protocol.json")}}), encoding="utf-8")

    dev = tmp_path / "runs" / "x3-dev"
    opts = {"config": str(x3), "model": str(pt), "phase": "dev", "out": str(dev), "device": "cpu"}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (dev / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=FROZEN_AWAITING_APPROVAL"), summary + ((dev / "error.txt").read_text() if (dev / "error.txt").exists() else "")
    frozen = json.loads((dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert frozen["model"]["name"] == pt_name and frozen["parent_protocol_hash"] == parent_frozen["protocol_hash"]
    assert frozen["relations"] == parent_frozen["relations"] and frozen["primary_k"] == parent_frozen["primary_k"]
    assert frozen["eligible_cells"] == [c for c in parent_frozen["eligible_cells"] if c.startswith("2/")]
    assert frozen["extension"]["kind"] == "model" and "variants" not in frozen["extension"]
    report = json.loads((dev / "report.json").read_text(encoding="utf-8"))
    assert report["lens_fidelity"] and set(report["divergence"]) == {"2"} and "summary" in report and "by_sae" not in report
    assert any("no usable SAE" in f for f in report["flags"])
    derived = json.loads((dev / "derived.json").read_text(encoding="utf-8"))
    assert f"lens_divergence.{pt_name}.L2" in derived and f"primary_k.{pt_name}" not in derived  # k is inherited
    rows = [json.loads(x) for x in zipfile.ZipFile(dev / "results.zip").read("trials.jsonl").decode().splitlines()]
    assert rows and all("sae" not in r and r["layer"] == 2 and "alpha" in r for r in rows)
    assert "Extension X3" in summary and "only the model changed" in summary

    test = tmp_path / "runs" / "x3-test"
    opts = {**opts, "phase": "test", "frozen": str(dev / "frozen_protocol.json"), "out": str(test)}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (test / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), summary + ((test / "error.txt").read_text() if (test / "error.txt").exists() else "")
    report = json.loads((test / "report.json").read_text(encoding="utf-8"))
    assert report["eligible_cells"] == frozen["eligible_cells"] and report["divergence"] == frozen["divergence"]
    code = execute("select", {**opts, "out": str(tmp_path / "runs" / "x3-test-2")}, S2.run, printer=lines.append, repo_root=git_repo)
    assert code == 1 and "test marker" in lines[-2]

    # the extension's test run is rebuilt exactly, like any Stage 2 run
    out = tmp_path / "analysis" / "x3-test"
    code = execute("analyze", {"run": str(test), "frozen": str(dev / "frozen_protocol.json"), "out": str(out)}, R.run,
                   printer=lines.append, repo_root=git_repo)
    rebuilt = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and rebuilt["reproduction"]["mismatches"] == [] and rebuilt["reproduction"]["summary_identical"], (out / "SUMMARY.md").read_text()

    # The SAE swap: the PT stand-in with another release's SAE at layer 2, as an SAE extension of the X3 protocol
    pt_cfg = yaml.safe_load(pt.read_text(encoding="utf-8"))
    pt_cfg["sae"]["swap"] = [{"width": pt_cfg["sae"]["width"], "l0": None, "repo_id": "example/other-release", "revision": "0" * 40, "tag": "other_sae"}]
    pt.write_text(yaml.safe_dump(pt_cfg), encoding="utf-8")
    s0_swap = tmp_path / "stage0_x7_test.yaml"
    s0_swap.write_text(yaml.safe_dump({"base": str(_stage0_config(tmp_path, tmp_path / "built", floor=1.5)), "scope": "sae_variants",
                                       "sae_variant_layers": [2], "sae_variant_list": "swap"}), encoding="utf-8")
    swap_run = tmp_path / "runs" / "s0-x7"
    assert cli.main(["inventory", "--config", str(s0_swap), "--model", str(pt), "--out", str(swap_run)]) == 0
    decisions = json.loads((swap_run / "stage0_decisions.json").read_text(encoding="utf-8"))
    key = f"L2_{pt_cfg['sae']['width']}_None_other_sae"
    assert set(decisions["sae_variants"]) == {key} and decisions["sae_variants"][key]["repo_id"] == "example/other-release"
    x7 = tmp_path / "stage2_x7.yaml"
    x7.write_text(yaml.safe_dump({"base": str(parent_cfg), "stage0_runs": {pt_name: str(s0)},
                                  "extension": {"id": "X7b", "parent_frozen": str(dev / "frozen_protocol.json"), "cells": ["2/concept"],
                                                "sae_run": str(swap_run)}}), encoding="utf-8")
    x7_dev, x7_test = tmp_path / "runs" / "x7-dev", tmp_path / "runs" / "x7-test"
    opts = {"config": str(x7), "model": str(pt), "phase": "dev", "out": str(x7_dev), "device": "cpu"}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    assert code == 0 and lines[-1].startswith("STATUS=FROZEN_AWAITING_APPROVAL"), (x7_dev / "SUMMARY.md").read_text(encoding="utf-8")
    x7_frozen = json.loads((x7_dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert x7_frozen["parent_protocol_hash"] == frozen["protocol_hash"] and set(x7_frozen["extension"]["variants"]) == {key}
    # the extension block names the immediate parent (the model-extension stand-in), not the parent's own parent
    assert x7_frozen["extension"]["parent_protocol_hash"] == frozen["protocol_hash"] != parent_frozen["protocol_hash"]
    code = execute("select", {**opts, "phase": "test", "frozen": str(x7_dev / "frozen_protocol.json"), "out": str(x7_test)}, S2.run,
                   printer=lines.append, repo_root=git_repo)
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), (x7_test / "SUMMARY.md").read_text(encoding="utf-8")
    rows = [json.loads(x) for x in zipfile.ZipFile(x7_test / "results.zip").read("trials.jsonl").decode().splitlines()]
    assert rows and {r["sae"] for r in rows} == {key}
    x7_report = json.loads((x7_test / "report.json").read_text(encoding="utf-8"))
    assert x7_report["extension"]["parent_protocol_hash"] == frozen["protocol_hash"]

    # The SAE swap's analysis: declared contrasts with Holm at one k
    a25 = tmp_path / "a25.yaml"
    a25.write_text(yaml.safe_dump({
        "models": {"own_sae": {"run": str(test), "frozen": str(dev / "frozen_protocol.json"), "matched_cells": ["2/concept"], "status": "post hoc"},
                   "swapped_sae": {"run": str(x7_test), "frozen": str(x7_dev / "frozen_protocol.json"), "matched_cells": ["2/concept"],
                                   "status": "declared before data"}},
        "matched_relations": ["R1_country_capital"], "contrasts": [{"name": "SAE effect", "model": "swapped_sae", "reference": "own_sae"}],
        "cross_model_k": [2, 1], "holm_k": 2, "holm_alpha": 0.05, "seed": 1}), encoding="utf-8")
    out = tmp_path / "analysis" / "a25"
    from saeselect.analysis import compare_sets as C

    assert execute("compare-sets", {"config": str(a25), "out": str(out)}, C.run, printer=lines.append, repo_root=git_repo) == 0, lines[-3:]
    cmp = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "p_holm" in cmp["cross_model"]["SAE effect"]["2"] and "p_holm" not in cmp["cross_model"]["SAE effect"]["1"]
