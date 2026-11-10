"""Llama support. The Llama Scope loader and its Stage 0 convention decision on a
synthetic SAE whose true convention is known, a build with another tokenizer through a base config with
overrides, and Stage 0 and Stage 2 dev end to end on a tiny Llama."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from data_fixtures import REPO, write_config
from saeselect import cli, tiny
from saeselect.patching import as_ids, clean_run
from saeselect.provenance import EngineeringFailed
from saeselect.runner import execute
from saeselect.sae import load_llama_scope
from saeselect.stages import inventory as S0
from saeselect.stages import selection as S2
from test_inventory import _build, _stage0_config, _tiny_model_config

LAYER = 2


def _tiny_llama():
    return tiny.build_model({"architecture": "llama", "multimodal": False, "num_hidden_layers": 4})


def _residuals(handle, n_windows=6, length=12):
    gen = torch.Generator().manual_seed(3)
    windows = [[2] + torch.randint(5, 200, (length,), generator=gen).tolist() for _ in range(n_windows)]
    return windows, torch.cat([clean_run(handle, as_ids(w, "cpu"), [LAYER], mode="bf16")[0][LAYER][1:].float() for w in windows])


def _write_llama_scope(root: Path, folder: str, d: int, h: torch.Tensor, *, avg_norm: float) -> dict:
    """A Llama Scope folder whose true convention is (input_scaled=True, p=0): in raw space, features +e_j and -e_j
    with one scalar threshold, stored as the training-time (normalized-input) parameters."""
    from safetensors.torch import save_file

    s = math.sqrt(d) / avg_norm
    eye = torch.eye(d)
    w_enc_raw = torch.cat([eye, -eye], dim=1)  # [d, 2d]
    w_dec_raw = torch.cat([eye, -eye], dim=0)  # [2d, d]
    theta = float(h.abs().median())
    pre = h @ w_enc_raw
    true_l0 = float(((pre > theta) & (pre > 0)).sum()) / h.shape[0]
    hp = {"act_fn": "jumprelu", "jump_relu_threshold": theta, "hook_point_in": f"blocks.{LAYER}.hook_resid_post",
          "hook_point_out": f"blocks.{LAYER}.hook_resid_post", "d_model": d, "d_sae": 2 * d, "top_k": round(true_l0),
          "dataset_average_activation_norm": {"in": avg_norm, "out": avg_norm}, "apply_decoder_bias_to_pre_encoder": False,
          "use_glu_encoder": False, "norm_activation": "dataset-wise"}
    (root / folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / folder / "hyperparams.json").write_text(json.dumps(hp), encoding="utf-8")
    save_file({"encoder.weight": (w_enc_raw / s).T.contiguous(), "encoder.bias": torch.zeros(2 * d),
               "decoder.weight": (w_dec_raw * s).T.contiguous(), "decoder.bias": torch.zeros(d)},
              str(root / folder / "checkpoints" / "final.safetensors"))
    return {"theta": theta, "true_l0": true_l0, "s": s}


def _llama_cfg(local_dir: Path) -> dict:
    cfg = yaml.safe_load((REPO / "configs/models/llama3_1_8b.yaml").read_text(encoding="utf-8"))
    cfg["sae"]["local_dir"] = str(local_dir)
    return cfg


def test_llama_scope_convention_is_decided_from_data(tmp_path):
    handle = _tiny_llama()
    windows, h = _residuals(handle)
    cfg = _llama_cfg(tmp_path)
    folder = cfg["sae"]["folder_pattern"].format(layer=LAYER)
    truth = _write_llama_scope(tmp_path, folder, handle.d_model, h, avg_norm=3.0)
    # the true convention reproduces the raw-space SAE; the others do not
    sae = load_llama_scope(repo_id="x", revision="x", folder=folder, layer=LAYER, d_model=handle.d_model,
                           convention={"input_scaled": True, "p": 0}, local_dir=str(tmp_path))
    pre = h @ torch.cat([torch.eye(handle.d_model), -torch.eye(handle.d_model)], dim=1)
    assert torch.allclose(sae.encode(h), torch.relu(pre) * (pre > truth["theta"]), atol=1e-5)
    decided = S0._llama_scope_convention(handle, cfg, [LAYER], windows, torch.device("cpu"))
    assert decided["convention"] == {"input_scaled": True, "p": 0}, decided
    table = decided["tables"][str(LAYER)]
    assert decided["target_l0"][str(LAYER)] == round(truth["true_l0"]) and len(table) == 6
    # with zero biases, unscaled input with threshold / s is the same SAE up to a factor s on the activations (the
    # forms differ only through the biases): L0 and VE tie exactly, and the rule's last tie-break (smaller |p|) decides
    same, chosen = table["input_scaled=False,p=-1"], table["input_scaled=True,p=0"]
    assert same["mean_l0"] == chosen["mean_l0"] and abs(same["variance_explained"] - chosen["variance_explained"]) < 1e-9
    with pytest.raises(EngineeringFailed):
        load_llama_scope(repo_id="x", revision="x", folder=folder, layer=LAYER + 1, d_model=handle.d_model,
                         convention={"input_scaled": True, "p": 0}, local_dir=str(tmp_path))  # wrong hook point


@pytest.mark.integration
def test_build_with_another_tokenizer_through_overrides(tmp_path):
    data, model = write_config(tmp_path)
    cfg = {"base": str(data), "overrides": {"build": {"out_dir": str(tmp_path / "built_other"), "tokenizer_model": str(model)}},
           "tasks": ["R1", "neutral"]}
    path = tmp_path / "data_other.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli.main(["build-data", "--config", str(path), "--out", str(tmp_path / "runs" / "b")]) == 0
    built = tmp_path / "built_other"
    assert sorted(p.name for p in built.iterdir()) == ["R1_country_capital", "neutral"] and not (tmp_path / "built").exists()
    card = json.loads((built / "R1_country_capital" / "benchmark_card.json").read_text(encoding="utf-8"))
    assert card["based_on"]["base_config"] == str(data) and card["based_on"]["overrides"]["build"]["out_dir"] == str(built)


def _tiny_llama_config(tmp: Path) -> Path:
    path = _tiny_model_config(tmp)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg.update(name="tiny_llama_test", architecture="llama", multimodal=False)
    out = tmp / "tiny_llama.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


@pytest.mark.integration
def test_inventory_and_selection_dev_on_a_tiny_llama(tmp_path, git_repo):
    _build(tmp_path)
    model = _tiny_llama_config(tmp_path)
    s0_cfg = yaml.safe_load(_stage0_config(tmp_path, tmp_path / "built", floor=1.5).read_text(encoding="utf-8"))
    s0_cfg["families"] = ["R1_country_capital", "R2_element_symbol", "R3_verb_past", "R4_country_language"]
    s0_path = tmp_path / "s0_llama.yaml"
    s0_path.write_text(yaml.safe_dump(s0_cfg), encoding="utf-8")
    s0 = tmp_path / "runs" / "s0-llama"
    code = cli.main(["inventory", "--config", str(s0_path), "--model", str(model), "--out", str(s0), "--limit", "3"])
    summary = (s0 / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0, summary + ((s0 / "error.txt").read_text(encoding="utf-8") if (s0 / "error.txt").exists() else "")
    artifacts = json.loads((s0 / "artifacts.json").read_text(encoding="utf-8"))
    assert artifacts["model"]["model_class"] == "LlamaForCausalLM" and artifacts["model"]["final_norm_gain_form"] == "weight"
    decisions = json.loads((s0 / "stage0_decisions.json").read_text(encoding="utf-8"))

    # Stage 2 dev on the tiny Llama, through a full (non-SMOKE) copy of that Stage 0 run, as in test_selection
    fake = tmp_path / "runs" / "s0-llama-full"
    fake.mkdir()
    import shutil

    shutil.copytree(s0 / "features", fake / "features")
    decisions.update(smoke=False, sae={k: v for k, v in decisions["sae"].items() if k in ("2", "3")})
    (fake / "stage0_decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    from test_selection import _config

    name = yaml.safe_load(model.read_text())["name"]
    config = _config(tmp_path, name, fake)
    out = tmp_path / "runs" / "s2-llama"
    lines: list[str] = []
    code = execute("select", {"config": str(config), "model": str(model), "phase": "dev", "out": str(out), "device": "cpu"},
                   S2.run, printer=lines.append, repo_root=git_repo)
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and lines[-1].startswith("STATUS=FROZEN_AWAITING_APPROVAL"), summary + ((out / "error.txt").read_text() if (out / "error.txt").exists() else "")
    assert json.loads((out / "frozen_protocol.json").read_text(encoding="utf-8"))["model"]["architecture"] == "llama"


def test_fp32_grad_mode_matches_the_exact_fp32_finite_difference_and_restores_the_weights():
    """On a BF16 model, the fp32_grad gradient's directional derivative equals the exact-FP32 (fp32_full) central
    finite difference, as N10 checks; the BF16 weights come back bit-identical; the BF16-matmul gradient is further off."""
    from saeselect.hooks import EditSpec
    from saeselect.models import wrap
    from saeselect.patching import PatchEvaluator, margin_gradient
    from saeselect.readout import AnswerSets

    handle = wrap(tiny.build_hf_model(architecture="llama", multimodal=False, num_hidden_layers=4, dtype="bfloat16"), tokenizer=None)
    before = {n: p.detach().clone() for n, p in handle.hf_model.named_parameters()}
    ids = torch.tensor([[2, 17, 40, 81, 9, 33, 70]])
    answers = AnswerSets(plus=(11, 12), minus=(13,))
    positions = list(range(1, ids.shape[1]))
    v = torch.randn(len(positions), handle.d_model, generator=torch.Generator().manual_seed(4))
    eps = 1e-4  # the random tiny model is strongly curved; the central difference converges by here
    ev = PatchEvaluator(handle, ids, answers, readout=6, mode="fp32_full", strategy="serial")
    up, down = ev.evaluate([[EditSpec(layer=1, positions=positions, delta=eps * v)], [EditSpec(layer=1, positions=positions, delta=-eps * v)]])
    fd = (up.margin - down.margin) / (2 * eps)
    _, g = margin_gradient(handle, ids, 1, answers, readout=6, mode="fp32_grad", checkpointed=True)
    _, g_mixed = margin_gradient(handle, ids, 1, answers, readout=6, mode="fp32_residual")
    exact = float((g[positions] * v).sum())
    mixed = float((g_mixed[positions] * v).sum())
    assert abs(exact - fd) <= 1e-3 * max(abs(fd), 1e-3), (exact, fd)
    assert abs(mixed - fd) > abs(exact - fd)
    after = dict(handle.hf_model.named_parameters())
    assert all(after[n].dtype == torch.bfloat16 and torch.equal(after[n], before[n]) for n in before)


def test_one_convention_per_release_on_the_measured_table():
    """User 2026-10-03: runs/s0-llama-02's mean L0 per layer; judged per layer, layer 27 would take unscaled input with
    threshold / s (L0 57.3), but pooled over the release the one form with the same L0 at every layer wins."""
    measured = {  # (mean L0, VE) per layer for scaled x s, scaled x 1, unscaled / s; the other forms are far from K
        "8": {"True,1": (30.8, 0.539), "True,0": (356.2, 0.545), "False,-1": (159.5, -0.060)},
        "16": {"True,1": (32.3, 0.630), "True,0": (198.5, 0.680), "False,-1": (172.3, 0.562)},
        "21": {"True,1": (31.1, 0.653), "True,0": (111.9, 0.710), "False,-1": (93.1, 0.698)},
        "27": {"True,1": (30.4, 0.584), "True,0": (60.7, 0.631), "False,-1": (57.3, 0.625)},
    }
    far = {"True,-1": (500.0, 0.5), "False,0": (25.0, 0.5), "False,1": (1.0, 0.0)}
    tables = {layer: {f"input_scaled={k.split(',')[0]},p={k.split(',')[1]}": {"mean_l0": l0, "variance_explained": ve}
                      for k, (l0, ve) in {**rows, **far}.items()} for layer, rows in measured.items()}
    targets = {layer: 50 for layer in tables}
    chosen = S0.choose_convention(tables, targets, [-1, 0, 1])
    assert chosen["convention"] == {"input_scaled": True, "p": 1}
    per_layer_27 = S0.choose_convention({"27": tables["27"]}, {"27": 50}, [-1, 0, 1])
    assert per_layer_27["convention"] == {"input_scaled": False, "p": -1}
