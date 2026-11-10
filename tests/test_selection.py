"""Stage 2 (selection benchmark) dev and test phases end to end on the tiny Gemma 3 with the fixture build."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from data_fixtures import REPO
from saeselect import cli
from saeselect.runner import execute
from saeselect.stages import selection as S2
from test_inventory import _build, _stage0_config, _tiny_model_config


def _stage0(tmp: Path, model: Path) -> Path:
    """A tiny Stage 0 run, relabelled as a full (non-SMOKE) run limited to two layers, so dev and test can use it."""
    out = tmp / "runs" / "s0"
    assert cli.main(["inventory", "--config", str(_stage0_config(tmp, tmp / "built", floor=1.5)), "--model", str(model),
                     "--out", str(out), "--limit", "3"]) == 0
    decisions = json.loads((out / "stage0_decisions.json").read_text(encoding="utf-8"))
    fake = tmp / "runs" / "s0-full"
    fake.mkdir(parents=True)
    shutil.copytree(out / "features", fake / "features")
    decisions.update(smoke=False, batching_allowed=True, suffix_caching_allowed=True, sae={k: v for k, v in decisions["sae"].items() if k in ("2", "3")})
    (fake / "stage0_decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    return fake


def _config(tmp: Path, name: str, s0: Path) -> Path:
    cfg = yaml.safe_load((REPO / "configs/selection/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg.update(data_dir=str(tmp / "built"), stage0_runs={name: str(s0)}, k_grid=[1, 2], n_random=2)
    cfg["primary_k"]["default"] = 2  # the largest k of the test grid, as 32 is of the declared one
    cfg["relations"] = {"active": ["R1_country_capital"], "reserve": "R2_element_symbol"}  # one relation keeps the test short
    # the random tiny model answers nothing, so the gates are opened to exercise every step
    cfg["format_gate"].update(min_own_mass=0.0, min_abs_margin=-1.0e9, min_template_share=0.0, test_min_answerable=0.0)
    cfg["eligibility"]["min_median_r_full"] = -1.0e9
    cfg["m2_min_full_gain"] = -1.0e9
    cfg["lens_fidelity"].update(n_directions=3, n_prompts=2)
    cfg["analysis"]["bootstrap"]["n_resamples"] = 200
    cfg["analysis"]["p23_min_layers"] = 2
    path = tmp / "stage2_test.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.integration
def test_selection_dev_freezes_and_test_runs_once(tmp_path, git_repo, capsys):
    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    config = _config(tmp_path, name, _stage0(tmp_path, model))
    lines: list[str] = []
    dev = tmp_path / "runs" / "s2-dev"
    opts = {"config": str(config), "model": str(model), "phase": "dev", "out": str(dev), "device": "cpu"}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (dev / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=FROZEN_AWAITING_APPROVAL"), summary + ((dev / "error.txt").read_text() if (dev / "error.txt").exists() else "")
    frozen = json.loads((dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert set(frozen["relations"]) == {"R1_country_capital"}
    assert frozen["primary_k"] in (1, 2) and set(frozen["eligible_cells"]) == {"2/readout", "2/concept", "3/readout", "3/concept"}
    report = json.loads((dev / "report.json").read_text(encoding="utf-8"))
    assert set(report["analysis"]["primary"]) == {"P2.1", "P2.2", "P2.3"} and report["lens_fidelity"]
    assert report["sanity_max_relative"] <= 1e-3 and report["parity_max"] <= 0.01 and report["strategy"] == "batched_suffix"
    derived = json.loads((dev / "derived.json").read_text(encoding="utf-8"))
    assert f"primary_k.{name}" in derived and f"lens_divergence.{name}.L2" in derived
    rows = [json.loads(x) for x in zipfile.ZipFile(dev / "results.zip").read("trials.jsonl").decode().splitlines()]
    row = rows[0]
    assert {"JL-OUT", "LL-OUT", "JL-OUT@in", "PATCH", "KAPPA", "JSPACE", "ORACLE", "ATP"} <= set(row["metrics"])
    assert "alpha" in row and row["support_templates"] and row["template"] not in row["support_templates"]
    assert all(t.split("|")[0] != "test" for t in [row["trial"]])

    test = tmp_path / "runs" / "s2-test"
    opts = {**opts, "phase": "test", "frozen": str(dev / "frozen_protocol.json"), "out": str(test)}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    summary = (test / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), summary + ((test / "error.txt").read_text() if (test / "error.txt").exists() else "")
    report = json.loads((test / "report.json").read_text(encoding="utf-8"))
    assert report["primary_k"] == frozen["primary_k"] and report["eligible_cells"] == frozen["eligible_cells"]
    rows = [json.loads(x) for x in zipfile.ZipFile(test / "results.zip").read("trials.jsonl").decode().splitlines()]
    assert rows and all("alpha" in r for r in rows)
    code = execute("select", {**opts, "out": str(tmp_path / "runs" / "s2-test-2")}, S2.run, printer=lines.append, repo_root=git_repo)
    assert code == 1 and "test marker" in lines[-2]
