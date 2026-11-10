"""The J-variant check: the J variants keep what they declare to keep, and on the tiny model the recomputed twins equal the parent's
stored M1, the dev pass freezes, and the test runs once on the parent's test run."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
import yaml

from data_fixtures import REPO
from saeselect.runner import execute
from saeselect.stages import jacobian_controls as JV
from saeselect.stages import selection as S2


def test_variants_keep_the_spectrum():
    J = torch.randn(12, 12, generator=torch.Generator().manual_seed(1))
    q = JV.haar(12, torch.Generator().manual_seed(2))
    assert torch.allclose(q.T @ q, torch.eye(12, dtype=torch.float64), atol=1e-12)
    sv = torch.linalg.svdvals(J.double())
    rand = JV.spectrum_matched(J, seed=3)
    assert torch.allclose(torch.linalg.svdvals(rand.double()), sv, atol=1e-4) and not torch.allclose(rand, J, atol=1e-2)
    perm = JV.permuted(J, seed=4)
    assert torch.allclose(torch.linalg.svdvals(perm.double()), sv, atol=1e-5)
    assert sorted(map(tuple, perm.tolist())) == sorted(map(tuple, J.tolist()))  # the same rows, reordered
    assert torch.equal(JV.spectrum_matched(J, seed=3), rand) and torch.equal(JV.permuted(J, seed=4), perm)  # seeded


def test_trial_row_parity_and_draw_mean():
    effects = {"1": 0.3, "2": -0.1, "3": 0.05, "4": 0.2}
    good = {1: 3.0, 2: -1.0, 3: 0.5, 4: 2.0}
    support = [{"LL-OUT": [good], "JL-OUT": [good], "JL-RAND": [good, {1: -3.0, 2: 1.0, 3: -0.5, 4: -2.0}], "JL-PERM": [good], "JL-OTHER": [good]}]
    r = {"trial": "t", "relation": "R1", "pair": "p", "layer": 2, "site": "concept", "effects": effects,
         "metrics": {"LL-OUT": {"M1": 1.0}, "JL-OUT": {"M1": 0.9}}}
    row = JV.trial_row(r, support)
    assert row["metrics"]["JL-OUT"]["M1"] == pytest.approx(1.0) and row["parity"]["LL-OUT"] == pytest.approx(0.0)
    assert row["parity"]["JL-OUT"] == pytest.approx(0.1)
    assert row["draws"]["JL-RAND"] == [pytest.approx(1.0), pytest.approx(-1.0)] and row["metrics"]["JL-RAND"]["M1"] == pytest.approx(0.0)


@pytest.mark.integration
def test_jacobian_variants_dev_freezes_and_test_runs_once(tmp_path, git_repo):
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

    a30 = yaml.safe_load((REPO / "configs/extensions/jacobian_variants.yaml").read_text(encoding="utf-8"))
    a30["parents"] = {name: {"frozen": str(frozen), "dev_run": str(dev), "test_run": str(test), "other_layer": {2: 3, 3: 2}}}
    a30["variants"] = {"JL-RAND": {"draws": 2}, "JL-PERM": {"draws": 2}}
    cfg = tmp_path / "a30.yaml"
    cfg.write_text(yaml.safe_dump(a30), encoding="utf-8")

    # a --model whose weights are not the parent's is refused (the parent's frozen model config is the one used)
    other = yaml.safe_load(model.read_text(encoding="utf-8"))
    other["revision"] = "0" * 40
    (tmp_path / "other_model.yaml").write_text(yaml.safe_dump(other), encoding="utf-8")
    lines: list[str] = []
    assert execute("jacobian-controls", {"config": str(cfg), "model": str(tmp_path / "other_model.yaml"), "phase": "dev",
                                 "out": str(tmp_path / "runs" / "a30-dev-other"), "device": "cpu"}, JV.run, printer=lines.append, repo_root=git_repo) != 0
    assert "is not the parent's" in (tmp_path / "runs" / "a30-dev-other" / "SUMMARY.md").read_text(encoding="utf-8")

    lines = []
    jdev = tmp_path / "runs" / "a30-dev"
    code = execute("jacobian-controls", {"config": str(cfg), "model": str(model), "phase": "dev", "out": str(jdev), "device": "cpu"}, JV.run,
                   printer=lines.append, repo_root=git_repo)
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], (jdev / "SUMMARY.md").read_text(encoding="utf-8")
    rep = json.loads((jdev / "report.json").read_text(encoding="utf-8"))
    assert max(rep["parity_max_abs_m1"].values()) < 1e-9  # same arithmetic on the CPU: the stored twins are reproduced exactly
    parent_rows = [x for x in zipfile.ZipFile(dev / "results.zip").read("trials.jsonl").decode().splitlines() if x.strip()]
    assert rep["n_trials"] == len(parent_rows) > 0  # one row per parent trial row
    assert set(rep["analysis"]["family"]) == {"JL-RAND", "JL-PERM", "JL-OTHER"}
    assert set(rep["analysis"]["mean_M1"]) == {"LL-OUT", "JL-OUT", "JL-RAND", "JL-PERM", "JL-OTHER"}

    jfrozen = jdev / "frozen_protocol.json"
    jtest = tmp_path / "runs" / "a30-test"
    lines = []
    assert execute("jacobian-controls", {"config": str(cfg), "phase": "test", "frozen": str(jfrozen), "out": str(jtest), "device": "cpu"}, JV.run,
                   printer=lines.append, repo_root=git_repo) == 0, (jtest / "SUMMARY.md").read_text(encoding="utf-8")
    trep = json.loads((jtest / "report.json").read_text(encoding="utf-8"))
    assert trep["parent_run"].endswith("s2-test") and trep["analysis"]["family"]["JL-OTHER"] is not None
    assert "p_holm" in trep["analysis"]["family"]["JL-RAND"]
    # the test runs once
    lines = []
    assert execute("jacobian-controls", {"config": str(cfg), "phase": "test", "frozen": str(jfrozen), "out": str(tmp_path / "runs" / "a30-test-2"),
                                 "device": "cpu"}, JV.run, printer=lines.append, repo_root=git_repo) != 0
