"""The greedy reference: the subset draw, the candidates, the greedy step, and the stage end to end on the tiny model (parity against
the parent's stored margins, dev freeze, single test)."""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import pytest
import yaml

from data_fixtures import REPO
from saeselect.runner import execute
from saeselect.stages import greedy_reference as G
from saeselect.stages import selection as S2


def _rows():
    rows = []
    for rel, n in (("R1", 6), ("R2", 3)):
        for i in range(n):
            for tpl in ("T1", "T2", "T3"):
                for d in ("a->b", "b->a"):
                    rows.append({"trial": f"{rel}:{i}:{tpl}:{d}", "relation": rel, "pair": f"{rel}:{i}", "template": tpl, "layer": 2, "site": "concept"})
    return rows


def test_subset_is_stratified_seeded_and_keeps_both_directions():
    rows = _rows()
    out = G.choose_rows(rows, cell="2/concept", n_pairs=3, seed=1)
    pairs = {r["pair"] for r in out}
    assert len(pairs) == 3 and sum(p.startswith("R1") for p in pairs) == 2  # 6:3 -> 2:1
    for p in pairs:
        sub = [r for r in out if r["pair"] == p]
        assert len({r["template"] for r in sub}) == 1 and len(sub) == 2  # one template, both directions
    assert G.choose_rows(rows, cell="2/concept", n_pairs=3, seed=1) == out and G.choose_rows(rows, cell="2/concept", n_pairs=99, seed=1)
    assert G.choose_rows(rows, cell="3/concept", n_pairs=3, seed=1) == []


def test_candidates_and_greedy_step():
    r = {"sets": {"A": {"2": {"selected": [1, 2]}}, "B": {"2": {"selected": [2, 3, 9]}}}}
    assert G.candidates(r, ["A", "B"], 2, {1, 2, 3}) == [1, 2, 3]  # 9 is not active on this prompt

    value = {(1,): 1.0, (2,): 3.0, (3,): 2.0, (2, 1): 3.5, (2, 3): 6.0, (1, 2, 3): 7.0, (2, 3, 1): 7.0}

    def evaluate(sets):
        return [SimpleNamespace(margin=value[tuple(s)], intended_norm=1.0, realized_norm=1.0) for s in sets]

    path = G.greedy(evaluate, [1, 2, 3], 3)
    assert [s["feature"] for s in path] == [2, 3, 1] and [s["margin"] for s in path] == [3.0, 6.0, 7.0]  # 2 then 3, not 2 then 1
    assert [s["n_candidates"] for s in path] == [3, 2, 1]


@pytest.mark.integration
def test_greedy_dev_freezes_and_test_runs_once(tmp_path, git_repo):
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

    a29 = yaml.safe_load((REPO / "configs/extensions/greedy_reference.yaml").read_text(encoding="utf-8"))
    a29.update(n_pairs=3, k_max=2, report_k=[1, 2], candidates_k=2,
               parents={name: {"frozen": str(frozen), "dev_run": str(dev), "test_run": str(test), "cell": "2/concept"}})
    cfg = tmp_path / "a29.yaml"
    cfg.write_text(yaml.safe_dump(a29), encoding="utf-8")
    lines: list[str] = []
    gdev = tmp_path / "runs" / "a29-dev"
    code = execute("greedy-reference", {"config": str(cfg), "model": str(model), "phase": "dev", "out": str(gdev), "device": "cpu"}, G.run,
                   printer=lines.append, repo_root=git_repo)
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], (gdev / "SUMMARY.md").read_text(encoding="utf-8")
    rep = json.loads((gdev / "report.json").read_text(encoding="utf-8"))
    parent_rows = [json.loads(x) for x in zipfile.ZipFile(dev / "results.zip").read("trials.jsonl").decode().splitlines() if x.strip()]
    available = len({r["pair"] for r in parent_rows if r["layer"] == 2 and r["site"] == "concept"})
    assert rep["parity_max"] < 1e-6 and rep["n_pairs"] == min(3, available) and rep["n_trials"] == 2 * rep["n_pairs"]
    assert set(rep["analysis"]["2"]["selectors"]) == set(a29["candidate_selectors"])

    gtest = tmp_path / "runs" / "a29-test"
    lines = []
    assert execute("greedy-reference", {"config": str(cfg), "phase": "test", "frozen": str(gdev / "frozen_protocol.json"), "out": str(gtest), "device": "cpu"},
                   G.run, printer=lines.append, repo_root=git_repo) == 0, (gtest / "SUMMARY.md").read_text(encoding="utf-8")
    trep = json.loads((gtest / "report.json").read_text(encoding="utf-8"))
    assert trep["parent_run"].endswith("s2-test") and trep["analysis"]["1"]["greedy"] is not None
    assert execute("greedy-reference", {"config": str(cfg), "phase": "test", "frozen": str(gdev / "frozen_protocol.json"),
                              "out": str(tmp_path / "runs" / "a29-test-2"), "device": "cpu"}, G.run, printer=lambda _: None, repo_root=git_repo) != 0
