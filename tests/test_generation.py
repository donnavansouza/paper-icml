"""The generation check: the answer-matching rule, and on the tiny model a parent Stage 2 dev and test pass, then the
generation check's dev pass (first-step margins equal the stored ones) that freezes and its test that runs once."""

from __future__ import annotations

import json
import zipfile

import pytest
import yaml

from data_fixtures import REPO
from saeselect.runner import execute
from saeselect.stages import generation as G
from saeselect.stages import selection as S2


def test_answer_matching_needs_a_word_boundary():
    assert G.starts_with_answer(" Tokyo\n", "Tokyo") and G.starts_with_answer("tokyo is", "tokyo")
    assert G.starts_with_answer(" Paris, France", "paris")  # the capitalized form counts
    assert not G.starts_with_answer(" Ca\n", "C") and G.starts_with_answer(" C.", "C")  # element symbols: "C" is not "Ca"
    assert G.outcome(" Rome", "Rome", "Oslo") == "donor" and G.outcome(" Oslo", "Rome", "Oslo") == "source"
    assert G.outcome(" the", "Rome", "Oslo") == "other"


@pytest.mark.integration
def test_generation_check_on_a_tiny_parent(tmp_path, git_repo):
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

    a39 = yaml.safe_load((REPO / "configs/extensions/generation.yaml").read_text(encoding="utf-8"))
    a39["parents"] = {name: {"frozen": str(frozen), "dev_run": str(dev), "test_run": str(test)}}
    a39.update(max_new_tokens=3, selectors=["JL-OUT", "NORM"])
    cfg = tmp_path / "a39.yaml"
    cfg.write_text(yaml.safe_dump(a39), encoding="utf-8")
    lines: list[str] = []
    gdev = tmp_path / "runs" / "a39-dev"
    code = execute("generate", {"config": str(cfg), "model": str(model), "phase": "dev", "out": str(gdev), "device": "cpu"},
                   G.run, printer=lines.append, repo_root=git_repo)
    err = (gdev / "error.txt").read_text(encoding="utf-8") if (gdev / "error.txt").exists() else ""
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], (gdev / "SUMMARY.md").read_text(encoding="utf-8") + err
    rep = json.loads((gdev / "report.json").read_text(encoding="utf-8"))
    assert rep["parity_max"] < 1e-3 and rep["n_trials"] > 0
    assert set(rep["analysis"]["pooled"]) == {"clean", "FULL", "JL-OUT", "NORM"}
    with zipfile.ZipFile(gdev / "results.zip") as z:
        row = json.loads(z.read("trials.jsonl").decode().splitlines()[0])
    assert all(c["outcome"] in ("donor", "source", "other") and 1 <= c["n_new"] <= 3 for c in row["conditions"].values())
    lines = []
    gtest = tmp_path / "runs" / "a39-test"
    assert execute("generate", {"config": str(cfg), "phase": "test", "frozen": str(gdev / "frozen_protocol.json"), "out": str(gtest),
                                "device": "cpu"}, G.run, printer=lines.append, repo_root=git_repo) == 0, \
        (gtest / "SUMMARY.md").read_text(encoding="utf-8")

