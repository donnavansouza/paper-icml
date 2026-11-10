"""The gradient at an edited residual, GREEDY-SUP's support objective, and on the tiny model a dev pass over a parent
Stage 2 run that freezes (parity of the stored sets and of ATP, IG completeness), then one test run."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch
import yaml

from data_fixtures import REPO
from saeselect.hooks import EditSpec
from saeselect.patching import margin_gradient
from saeselect.readout import AnswerSets
from saeselect.runner import execute
from saeselect.stages import joint_selection as I
from saeselect.stages import selection as S2


def test_gradient_at_an_edited_residual(tiny_model, input_ids):
    answers = AnswerSets(plus=(5,), minus=(7,))
    kw = dict(readout=input_ids.shape[1] - 1, mode="fp32_full")
    m0, g0 = margin_gradient(tiny_model, input_ids, 2, answers, **kw)
    zero = EditSpec(layer=2, positions=[3, 4], delta=torch.zeros(2, tiny_model.d_model))
    m1, g1 = margin_gradient(tiny_model, input_ids, 2, answers, **kw, edit=zero)
    assert m1 == m0 and torch.equal(g1, g0)  # a zero edit is the unedited gradient
    push = EditSpec(layer=2, positions=[3, 4], delta=torch.randn(2, tiny_model.d_model, generator=torch.Generator().manual_seed(2)))
    m2, g2 = margin_gradient(tiny_model, input_ids, 2, answers, **kw, edit=push)
    assert m2 != m0 and not torch.allclose(g2, g0) and torch.isfinite(g2).all()
    with pytest.raises(ValueError):
        margin_gradient(tiny_model, input_ids, 3, answers, **kw, edit=push)


class _P:
    """A stand-in prompt whose margin is the sum of fixed per-feature gains plus one interaction."""

    def __init__(self, active, gain, pair_bonus):
        self.active, self.gain, self.pair_bonus = set(active), gain, pair_bonus

    def eff(self, feats):
        return tuple(sorted(f for f in feats if f in self.active))

    def evaluate(self, sets):
        out = []
        for s in sets:
            m = sum(self.gain[f] for f in s) + (self.pair_bonus if {1, 2} <= set(s) else 0.0)
            out.append(type("R", (), {"margin": m})())
        return out


def test_greedy_support_uses_measured_interactions():
    gain = {1: 0.2, 2: 0.2, 3: 0.5, 4: 0.1}
    sup = [_P([1, 2, 3, 4], gain, 2.0), _P([1, 2, 3], gain, 2.0)]  # feature 4 is inactive on the second support prompt
    path = I.greedy_support(sup, [0.0, 0.0], [1, 2, 3, 4], 3)
    assert [s["feature"] for s in path] == [3, 1, 2]  # 1 (tie with 2, smaller id), then 2 for the measured interaction
    assert path[2]["support_gain"] == pytest.approx(0.5 + 0.2 + 0.2 + 2.0)
    assert np.isclose(path[0]["support_gain"], 0.5)


@pytest.mark.integration
def test_joint_selection_dev_freezes_and_test_runs_once(tmp_path, git_repo):
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
    cell = json.loads(frozen.read_text(encoding="utf-8"))["eligible_cells"][0]

    a37 = yaml.safe_load((REPO / "configs/extensions/joint_selection.yaml").read_text(encoding="utf-8"))
    a37["parents"] = {name: {"frozen": str(frozen), "dev_run": str(dev), "test_run": str(test), "cell": cell}}
    a37.update(k_max=2, report_k=[1, 2], candidates_k=2, ig_steps=16)
    a37["family"] = [["GREEDY-SUP", "JL-OUT", 2], ["IG", "ATP", "primary"]]
    a37["secondary"] = [["GREEDY-SUP", "NORM", 2], ["IG", "ATP", "M1"]]
    cfg = tmp_path / "a37.yaml"
    cfg.write_text(yaml.safe_dump(a37), encoding="utf-8")
    lines: list[str] = []
    jdev = tmp_path / "runs" / "a37-dev"
    code = execute("joint-selection", {"config": str(cfg), "model": str(model), "phase": "dev", "out": str(jdev), "device": "cpu"}, I.run,
                   printer=lines.append, repo_root=git_repo)
    err = (jdev / "error.txt").read_text(encoding="utf-8") if (jdev / "error.txt").exists() else ""
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], (jdev / "SUMMARY.md").read_text(encoding="utf-8") + err
    rep = json.loads((jdev / "report.json").read_text(encoding="utf-8"))
    assert rep["parity_max"] < 1e-6 and rep["atp_parity_max"] < 1e-6 and rep["n_trials"] > 0
    assert set(rep["analysis"]["family"]) == {"GREEDY-SUP - JL-OUT (F at k = 2)", "IG - ATP (F at k = 2)"}
    comp = rep["analysis"]["ig_completeness"]
    assert comp and abs(comp["median"] - 1.0) < 0.1  # the midpoint rule recovers the transfer's own margin change
    with zipfile.ZipFile(jdev / "results.zip") as z:
        row = json.loads(z.read("trials.jsonl").decode().splitlines()[0])
    assert set(row["sets"]) == {"GREEDY-SUP", "IG"} and len(row["greedy_support_path"]) <= 2 and "IG@in" in row["metrics"]

    jtest = tmp_path / "runs" / "a37-test"
    assert execute("joint-selection", {"config": str(cfg), "phase": "test", "frozen": str(jdev / "frozen_protocol.json"), "out": str(jtest), "device": "cpu"},
                   I.run, printer=lambda _: None, repo_root=git_repo) == 0, (jtest / "SUMMARY.md").read_text(encoding="utf-8")
    fam = json.loads((jtest / "report.json").read_text(encoding="utf-8"))["analysis"]["family"]
    assert all(v is None or "p_holm" in v for v in fam.values())
