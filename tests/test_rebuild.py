"""``analyze``: a Stage 2 run is rebuilt exactly from its results.zip, and a changed number fails loudly."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from saeselect.analysis import compare_sets as C
from saeselect.analysis import rebuild as R
from saeselect.runner import execute
from saeselect.stages import selection as S2
from test_inventory import _build, _tiny_model_config
from test_selection import _config, _stage0


def test_compare_counts_leaves_and_reports_paths():
    n, bad = R.compare({"a": [1.0, float("nan")], "b": {"c": "x", "d": None}}, {"a": [1, float("nan")], "b": {"c": "x", "d": None}})
    assert n == 4 and bad == []
    n, bad = R.compare({"a": [1.0, 2.0], "b": True}, {"a": [1.0, 2.0000001], "c": True})
    assert bad == ["/a[1]: report 2.0, rebuild 2.0000001", "/b: in the report only", "/c: in the rebuild only"]


def _analyze(run: Path, out: Path, repo: Path, frozen: Path | None = None) -> tuple[int, list[str]]:
    lines: list[str] = []
    opts = {"run": str(run), "out": str(out), **({"frozen": str(frozen)} if frozen else {})}
    return execute("analyze", opts, R.run, printer=lines.append, repo_root=repo), lines


@pytest.mark.integration
def test_analyze_rebuilds_dev_and_test_and_fails_on_a_changed_number(tmp_path, git_repo):
    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    config = _config(tmp_path, name, _stage0(tmp_path, model))
    dev, test = tmp_path / "runs" / "s2-dev", tmp_path / "runs" / "s2-test"
    opts = {"config": str(config), "model": str(model), "phase": "dev", "out": str(dev), "device": "cpu"}
    assert execute("select", opts, S2.run, printer=lambda _: None, repo_root=git_repo) == 0
    frozen = dev / "frozen_protocol.json"
    assert execute("select", {**opts, "phase": "test", "frozen": str(frozen), "out": str(test)}, S2.run,
                   printer=lambda _: None, repo_root=git_repo) == 0

    code, lines = _analyze(test, tmp_path / "analysis" / "s2-test", git_repo, frozen)
    out = json.loads((tmp_path / "analysis" / "s2-test" / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and lines[-1].startswith("STATUS=PASS"), (tmp_path / "analysis" / "s2-test" / "SUMMARY.md").read_text()
    assert out["reproduction"]["mismatches"] == [] and out["reproduction"]["summary_identical"]
    assert out["reproduction"]["leaves_compared"] > 100 and out["provenance"]["phase"] == "test"
    assert {"JL-OUT", "LL-OUT", "ORACLE", "NORM"} <= set(out["random_baseline"])
    assert out["rebuilt"]["analysis"]["primary"]["P2.1"] == json.loads((test / "report.json").read_text())["analysis"]["primary"]["P2.1"]

    # The cross-model comparison compare-sets wiring: the same test run as model and reference gives a zero cross-model difference
    a24 = tmp_path / "a24.yaml"
    a24.write_text(yaml.safe_dump({
        "models": {m: {"run": str(test), "frozen": str(frozen), "matched_cells": ["2/concept", "3/concept"], "status": "post hoc"} for m in ("a", "b")},
        "reference": "a", "matched_relations": ["R1_country_capital"], "cross_model_k": [1, 2], "holm_alpha": 0.05, "seed": 1}), encoding="utf-8")
    lines = []
    assert execute("compare-sets", {"config": str(a24), "out": str(tmp_path / "analysis" / "a24")}, C.run, printer=lines.append, repo_root=git_repo) == 0, lines
    cmp = json.loads((tmp_path / "analysis" / "a24" / "report.json").read_text(encoding="utf-8"))
    assert set(cmp["models"]["a"]["matched"]["M2"]) == {"1", "2"} and abs(cmp["cross_model"]["b - a"]["1"]["estimate"]) < 1e-12

    # a dev run is rebuilt from its own frozen protocol and flagged as estimates
    code, lines = _analyze(dev, tmp_path / "analysis" / "s2-dev", git_repo)
    assert code == 0 and "PASS_WITH_FLAGS" in lines[-1]
    assert "not claims" in (tmp_path / "analysis" / "s2-dev" / "SUMMARY.md").read_text(encoding="utf-8")

    # one changed number in results.zip: the rebuild no longer matches report.json
    bad = tmp_path / "runs" / "s2-test-changed"
    shutil.copytree(test, bad)
    rows = [json.loads(x) for x in zipfile.ZipFile(test / "results.zip").read("trials.jsonl").decode().splitlines()]
    rows[0]["metrics"]["JL-OUT"]["M1"] = (rows[0]["metrics"]["JL-OUT"]["M1"] or 0.0) + 0.5
    with zipfile.ZipFile(bad / "results.zip", "w") as z:
        z.writestr("trials.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    code, lines = _analyze(bad, tmp_path / "analysis" / "s2-changed", git_repo, frozen)
    assert code == 2 and "GATE_FAILED:reproduction" in lines[-1]
    assert "differ from report.json" in (tmp_path / "analysis" / "s2-changed" / "SUMMARY.md").read_text(encoding="utf-8")

    # an edited frozen protocol is refused (its seal no longer holds)
    edited = tmp_path / "edited_frozen.json"
    data = json.loads(frozen.read_text(encoding="utf-8"))
    data["primary_k"] = 99
    edited.write_text(json.dumps(data), encoding="utf-8")
    code, lines = _analyze(test, tmp_path / "analysis" / "s2-edited", git_repo, edited)
    assert code != 0 and lines[-1].startswith("STATUS=ERROR")
