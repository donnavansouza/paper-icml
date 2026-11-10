"""The identity-keeping J variants keep what they declare, the J-variant check's configuration is unchanged, and on the tiny model the
dev pass (with NORM parity, the partial correlation, J diagnostics and the timing benchmark) freezes and the test runs once."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from data_fixtures import REPO
from saeselect.runner import execute
from saeselect.stages import jacobian_controls as JV
from saeselect.stages import selection as S2


class _Lens:
    def __init__(self, J):
        self._J = J

    def J(self, layer, device):
        return self._J


def test_residual_variants_keep_the_identity_and_the_residual_spectrum():
    J = torch.eye(10) + 0.3 * torch.randn(10, 10, generator=torch.Generator().manual_seed(1))
    cfg = yaml.safe_load((REPO / "configs/extensions/jacobian_controls.yaml").read_text(encoding="utf-8"))
    names = JV.variant_names(cfg)
    assert names == ["JL-RES", "JL-RESPERM", "JL-T0.25", "JL-T0.5", "JL-T0.75"]
    mats = JV.variant_matrices(_Lens(J), 2, None, cfg["variants"], 7, names)
    eye = torch.eye(10)
    sv_r = torch.linalg.svdvals((J - eye).double())
    for M in mats["JL-RES"]:
        assert torch.allclose(torch.linalg.svdvals((M - eye).double()), sv_r, atol=1e-4) and not torch.allclose(M, J, atol=1e-2)
    for M in mats["JL-RESPERM"]:  # the residual's rows, reordered
        got, want = (M - eye), (J - eye)
        assert torch.allclose(got[got[:, 0].argsort()], want[want[:, 0].argsort()], atol=1e-6)
    assert torch.allclose(mats["JL-T0.5"][0], 0.5 * eye + 0.5 * J) and len(mats["JL-RES"]) == 3
    # the J-variant check's configuration declares no design: its three variants, unchanged
    a30 = yaml.safe_load((REPO / "configs/extensions/jacobian_variants.yaml").read_text(encoding="utf-8"))
    assert JV.variant_names(a30) == ["JL-RAND", "JL-PERM", "JL-OTHER"]
    d = JV.j_diagnostics(eye)
    assert d["rel_frobenius_R"] == pytest.approx(0.0) and d["cos_J_I"] == pytest.approx(1.0)


def test_trial_row_partial_given_norm():
    effects = {"1": 0.3, "2": -0.1, "3": 0.05, "4": 0.2}
    s = {1: 3.0, 2: -1.0, 3: 0.5, 4: 2.0}
    norm = {1: 1.0, 2: 0.5, 3: 0.2, 4: 0.9}
    support = [{"LL-OUT": [s], "JL-OUT": [s], "NORM": [norm]}]
    r = {"trial": "t", "relation": "R1", "pair": "p", "layer": 2, "site": "concept", "effects": effects,
         "metrics": {"LL-OUT": {"M1": 1.0}, "JL-OUT": {"M1": 1.0}, "NORM": {"M1": 0.8}}}
    row = JV.trial_row(r, support)
    assert set(row["parity"]) == {"LL-OUT", "JL-OUT", "NORM"} and row["parity"]["NORM"] == pytest.approx(0.0)
    assert "M6_given_NORM" in row["metrics"]["JL-OUT"]


def test_parity_tolerance_admits_one_near_tie_swap_only():
    # swapping two adjacent score ranks moves Spearman by 12 |e_a - e_b| / (n (n^2 - 1)), at most 12 / (n (n + 1))
    from scipy.stats import spearmanr
    n = 34
    effects = [float(i) for i in range(n)]
    scores = [float(i) for i in range(n)]
    scores[0], scores[n - 1], scores[16], scores[17] = 16.0, 17.0, 0.0, 33.0  # the extreme-effect items at adjacent score ranks
    swapped = scores[:]
    swapped[0], swapped[n - 1] = scores[n - 1], scores[0]
    one_swap = abs(spearmanr(scores, effects)[0] - spearmanr(swapped, effects)[0])
    tol = JV.parity_tolerance(1e-3, n, True)
    assert one_swap == pytest.approx(12 / (n * (n + 1))) and tol == pytest.approx(one_swap)
    assert JV.parity_tolerance(1e-3, 200, True) == 1e-3  # large pools keep the declared tolerance
    assert JV.parity_tolerance(1e-3, n, False) == 1e-3  # The J-variant check, and any config without the key: unchanged


@pytest.mark.integration
def test_jacobian_controls_dev_freezes_and_test_runs_once(tmp_path, git_repo):
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

    a35 = yaml.safe_load((REPO / "configs/extensions/jacobian_controls.yaml").read_text(encoding="utf-8"))
    a35["parents"] = {name: {"frozen": str(frozen), "dev_run": str(dev), "test_run": str(test)}}
    a35["variants"] = {"JL-RES": {"draws": 2}, "JL-RESPERM": {"draws": 2}, "JL-INTERP": {"t": [0.5]}}
    a35["timing"] = {"n_prompts": 2}
    cfg = tmp_path / "a35.yaml"
    cfg.write_text(yaml.safe_dump(a35), encoding="utf-8")
    lines: list[str] = []
    jdev = tmp_path / "runs" / "a35-dev"
    code = execute("jacobian-controls", {"config": str(cfg), "model": str(model), "phase": "dev", "out": str(jdev), "device": "cpu"}, JV.run,
                   printer=lines.append, repo_root=git_repo)
    err = (jdev / "error.txt").read_text(encoding="utf-8") if (jdev / "error.txt").exists() else ""
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], (jdev / "SUMMARY.md").read_text(encoding="utf-8") + err
    rep = json.loads((jdev / "report.json").read_text(encoding="utf-8"))
    assert max(rep["parity_max_abs_m1"].values()) < 1e-9 and "NORM" in rep["parity_max_abs_m1"]
    assert set(rep["analysis"]["family"]) == {"JL-OUT - JL-RES", "JL-OUT - JL-RESPERM"}
    assert {"JL-T0.5", "NORM"} <= set(rep["analysis"]["mean_M1"]) and "JL-OUT partial given NORM" in rep["analysis"]
    assert rep["J_diagnostics"] and rep["timing"]["n_prompts"] == 2 and rep["design"] == "controls"
    assert rep["parity_swap_bound"] == {"trials_over_base_tolerance": 0, "n_trials": rep["n_trials"]}  # The swap bound, reported
    assert json.loads((jdev / "frozen_protocol.json").read_text(encoding="utf-8"))["design"] == "controls"

    jtest = tmp_path / "runs" / "a35-test"
    lines = []
    assert execute("jacobian-controls", {"config": str(cfg), "phase": "test", "frozen": str(jdev / "frozen_protocol.json"), "out": str(jtest),
                                 "device": "cpu"}, JV.run, printer=lines.append, repo_root=git_repo) == 0, (jtest / "SUMMARY.md").read_text(encoding="utf-8")
    assert "p_holm" in json.loads((jtest / "report.json").read_text(encoding="utf-8"))["analysis"]["family"]["JL-OUT - JL-RES"]
