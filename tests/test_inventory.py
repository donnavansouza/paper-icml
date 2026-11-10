"""Stage 0 end to end on the tiny Gemma 3 with the fixture data build (--limit),
and the FP32-residual mode and patch strategies it decides between."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from data_fixtures import REPO, WORDS, write_config
from saeselect import cli, tiny
from saeselect.hooks import EditSpec
from saeselect.models import precision_mode, wrap
from saeselect.patching import PatchEvaluator
from saeselect.readout import AnswerSets


def _tiny_model_config(tmp: Path) -> Path:
    cfg = yaml.safe_load((REPO / "configs/models/tiny_gemma3_test.yaml").read_text(encoding="utf-8"))
    tok = tiny.build_tokenizer(WORDS)
    cfg["vocab_size"] = max(cfg["vocab_size"], len(tok) + 8)
    cfg["tokenizer_words"] = sorted(set(WORDS))
    cfg["sae"]["width"] = 2 * cfg["hidden_size"] + 32
    path = tmp / "tiny_stage0_model.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _stage0_config(tmp: Path, data_dir: Path, *, floor: float | None = None, margin: float = -1.0) -> Path:
    cfg = yaml.safe_load((REPO / "configs/inventory/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg["data_dir"] = str(data_dir)
    cfg["checks"]["N3"]["min_margin_over_neighbors"] = margin  # the tiny basis SAE reconstructs every layer exactly (margin 0)
    if floor is not None:
        cfg["checks"]["N13"]["min_variance_explained"] = floor
    cfg["checks"]["N12"]["tokens"] = 48
    path = tmp / f"stage0_test_{floor}_{margin}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _build(tmp_path):
    data, build_model = write_config(tmp_path)
    assert cli.main(["build-data", "--config", str(data), "--model", str(build_model), "--out", str(tmp_path / "runs" / "build")]) == 0


@pytest.mark.integration
def test_inventory_end_to_end(tmp_path, capsys):
    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    # A floor above 1 sends every layer down the keep-and-flag path: the tiny
    # config has no fallback, so the primary SAE is kept and flagged low-fidelity.
    config = _stage0_config(tmp_path, tmp_path / "built", floor=1.5)
    capsys.readouterr()
    out = tmp_path / "runs" / "s0"
    code = cli.main(["inventory", "--config", str(config), "--model", str(model), "--out", str(out), "--limit", "3"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0, summary + (out / "error.txt").read_text(encoding="utf-8") if (out / "error.txt").exists() else summary
    assert line.startswith("STATUS=PASS_WITH_FLAGS") and "stage0_decisions.json" in line
    assert "SMOKE" in summary and "low-fidelity" in summary
    decisions = json.loads((out / "stage0_decisions.json").read_text(encoding="utf-8"))
    assert decisions["precision_mode"] in ("bf16", "fp32_residual") and decisions["smoke"] is True
    assert set(decisions["tau_tie"]) == set(yaml.safe_load(config.read_text())["families"])
    assert set(decisions["sae"]) == {"1", "2", "3", "4"}
    assert decisions["low_fidelity_layers"] == [1, 2, 3, 4] and decisions["excluded_layers"] == {}
    n7 = json.loads((out / "n7_precision_audit.json").read_text(encoding="utf-8"))["summary"]
    assert "spearman_fp32_residual_vs_exact_fp32" in n7["descriptive"]
    derived = json.loads((out / "derived.json").read_text(encoding="utf-8"))
    assert "precision_mode" in derived and any(k.startswith("tau_tie.") for k in derived)
    from safetensors.torch import load_file

    feats = load_file(str(out / "features" / f"{decisions['sae']['2']['key']}.safetensors"))
    assert {"sd", "firing_rate", "kappa_jlens", "kappa_logit_lens"} <= set(feats)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    for name in ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13"):
        assert report["checks"][name]["pass"], (name, report["checks"][name])
    assert report["checks"]["N10"]["reference_mode"] == "fp32_full"
    assert all(abs(f["finite_difference"] - f["finite_difference_half_step"]) <= 0.05 * abs(f["finite_difference"]) + 1e-6
               for f in report["checks"]["N10"]["features"])
    assert (out / "chat_snapshots.md").read_text(encoding="utf-8").count("```") >= 2


@pytest.mark.integration
def test_inventory_excludes_layers_that_fail_n3_and_stops_when_none_remain(tmp_path, capsys):
    _build(tmp_path)
    config = _stage0_config(tmp_path, tmp_path / "built", margin=0.5)  # no layer can beat its neighbors by 0.5
    out = tmp_path / "runs" / "s0x"
    capsys.readouterr()
    code = cli.main(["inventory", "--config", str(config), "--model", str(_tiny_model_config(tmp_path)), "--out", str(out), "--limit", "2"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 3 and line.startswith("STATUS=ENGINEERING_FAILED:N3")
    assert "no candidate layer has a usable SAE" in (out / "error.txt").read_text(encoding="utf-8")


@pytest.mark.integration
def test_inventory_keeps_the_better_sae_when_the_fallback_also_misses_the_floor(tmp_path, capsys, monkeypatch):
    """Primary and D2 fallback both below the N13 floor: the one explaining more is kept
    (here the primary, an exact basis SAE, against a random fallback)."""
    from saeselect import artifacts

    _build(tmp_path)
    model = _tiny_model_config(tmp_path)
    cfg = yaml.safe_load(model.read_text())
    fallback_width = cfg["sae"]["width"] + 64
    cfg["sae"]["fallback"] = {"width": fallback_width, "l0": "medium"}
    model.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    real = artifacts.load_sae_for

    def load(model_cfg, *, layer, width, l0, d_model, suffix="", device="cpu", convention=None):
        if int(width) == fallback_width:
            return tiny.random_sae(d_model, int(width), seed=layer, layer=layer).to(device)
        return real(model_cfg, layer=layer, width=width, l0=l0, d_model=d_model, suffix=suffix, device=device)

    monkeypatch.setattr(artifacts, "load_sae_for", load)
    config = _stage0_config(tmp_path, tmp_path / "built", floor=1.5)  # both SAEs below the floor at every layer
    out = tmp_path / "runs" / "s0b"
    capsys.readouterr()
    code = cli.main(["inventory", "--config", str(config), "--model", str(model), "--out", str(out), "--limit", "2"])
    assert code == 0, (out / "SUMMARY.md").read_text(encoding="utf-8")
    decisions = json.loads((out / "stage0_decisions.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    per_config = report["checks"]["N13"]["value"]["per_config"]
    for layer, entry in decisions["sae"].items():
        primary, fallback = f"L{layer}_{cfg['sae']['width']}_None", f"L{layer}_{fallback_width}_medium"
        assert per_config[primary]["variance_explained"] > per_config[fallback]["variance_explained"]
        assert entry["key"] == primary and entry["low_fidelity"] and not entry["fallback_used"]
    assert decisions["causal_mode"] == "fp32_full" and decisions["backward"] == "checkpointed"
    assert report["checks"]["N4_fp32_full"]["pass"] and report["checks"]["N8"]["mode"] == "fp32_full"


def test_checkpointed_gradient_equals_plain_gradient():
    from saeselect.patching import margin_gradient

    for dtype, mode in (("float32", "bf16"), ("bfloat16", "fp32_residual")):
        m = wrap(tiny.build_hf_model(dtype=dtype), tokenizer=None)
        ids = torch.randint(4, 400, (1, 30), generator=torch.Generator().manual_seed(0))
        ids[0, 0] = 2
        answers = AnswerSets(plus=(5, 9), minus=(40,))
        m1, g1 = margin_gradient(m, ids, 1, answers, readout=29, mode=mode)
        m2, g2 = margin_gradient(m, ids, 1, answers, readout=29, mode=mode, checkpointed=True)
        assert m1 == m2 and torch.allclose(g1, g2, atol=1e-6)
        assert all("forward" not in layer.__dict__ for layer in m.layers)  # wrappers removed


def test_fp32_full_mode_is_exact_and_restores_weights():
    import copy

    from saeselect.hooks import ResidualRecorder

    hf16 = tiny.build_hf_model(dtype="bfloat16")
    m16 = wrap(hf16, tokenizer=None)
    m32 = wrap(copy.deepcopy(hf16).float(), tokenizer=None)
    before = {n: p.data.clone() for n, p in m16.hf_model.named_parameters()}
    ids = torch.randint(4, 400, (1, 16), generator=torch.Generator().manual_seed(5))
    ids[0, 0] = 2
    last = m16.n_layers - 1

    def final(m, mode):
        with torch.no_grad(), precision_mode(m, mode), ResidualRecorder(m.layers, at=[last]) as rec:
            m.forward(ids)
        return rec.activations[last]

    assert torch.equal(final(m16, "fp32_full"), final(m32, "bf16"))
    assert all(torch.equal(before[n], p.data) and p.dtype == torch.bfloat16 for n, p in m16.hf_model.named_parameters())
    with pytest.raises(ValueError, match="no-grad"):
        with torch.enable_grad(), precision_mode(m16, "fp32_full"):
            pass


def test_inventory_only(tmp_path, capsys):
    config = _stage0_config(tmp_path, tmp_path / "no_data_needed")
    out = tmp_path / "runs" / "inv"
    code = cli.main(["inventory", "--config", str(config), "--model", str(_tiny_model_config(tmp_path)), "--out", str(out), "--inventory-only"])
    assert code == 0, (out / "SUMMARY.md").read_text(encoding="utf-8")
    inv = json.loads((out / "artifacts.json").read_text(encoding="utf-8"))
    assert inv["tokenizer"]["vocab_sha256"] and inv["chat_template"]["present"] is True
    assert inv["chat_template"]["separate_system_turn"] is False  # merged into the first user turn, as Gemma's template does
    assert "artifacts.json" in capsys.readouterr().out.strip().splitlines()[-1]


@pytest.mark.integration
def test_raw_completions(tmp_path, capsys):
    data, build_model = write_config(tmp_path)
    assert cli.main(["build-data", "--config", str(data), "--model", str(build_model), "--out", str(tmp_path / "runs" / "build")]) == 0
    cfg = yaml.safe_load((REPO / "configs/completions/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg["data_dir"] = str(tmp_path / "built")
    cfg["max_new_tokens"] = 4
    config = tmp_path / "raw.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "runs" / "raw"
    capsys.readouterr()
    code = cli.main(["raw-completions", "--config", str(config), "--model", str(_tiny_model_config(tmp_path)), "--out", str(out), "--limit", "2"])
    assert code == 0, (out / "SUMMARY.md").read_text(encoding="utf-8")
    text = (out / "raw_completions.md").read_text(encoding="utf-8")
    for family in ("R1_country_capital",):
        assert f"## {family}" in text
    rows = [json.loads(line) for line in (out / "raw_completions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(0 <= r["answer_mass"] <= 1 and r["first"] in ("A+", "A-", "neither") for r in rows)
    assert not any("|test|" in r["id"] for r in rows)
    assert "raw_completions.md" in capsys.readouterr().out.strip().splitlines()[-1]


def test_patch_strategies_agree_in_both_precision_modes():
    m = wrap(tiny.build_hf_model(dtype="bfloat16"), tokenizer=None)
    ids = torch.randint(4, 400, (1, 18), generator=torch.Generator().manual_seed(0))
    ids[0, 0] = 2
    answers = AnswerSets(plus=(5, 9), minus=(40,))
    gen = torch.Generator().manual_seed(1)
    sets = [[EditSpec(layer=3, positions=[17], delta=0.5 * torch.randn(1, 64, generator=gen))] for _ in range(6)]
    for mode in ("bf16", "fp32_residual"):
        serial = PatchEvaluator(m, ids, answers, mode=mode).evaluate(sets)
        batched = PatchEvaluator(m, ids, answers, mode=mode, strategy="batched", batch_size=4).evaluate(sets)
        suffix = PatchEvaluator(m, ids, answers, mode=mode, strategy="suffix").evaluate(sets)
        assert max(abs(a.margin - b.margin) for a, b in zip(serial, batched)) < 1e-6
        assert max(abs(a.margin - b.margin) for a, b in zip(serial, suffix)) < 5e-3
        both = PatchEvaluator(m, ids, answers, mode=mode, strategy="batched_suffix", batch_size=4).evaluate(sets)
        assert max(abs(a.margin - b.margin) for a, b in zip(suffix, both)) < 1e-5
        assert [round(b.realized_norm, 4) for b in both] == [round(s.realized_norm, 4) for s in suffix]
    # In exact FP32 the cached-prefix strategies agree with the full forward (N8 passed there on 12B)
    serial = PatchEvaluator(m, ids, answers, mode="fp32_full").evaluate(sets)
    both = PatchEvaluator(m, ids, answers, mode="fp32_full", strategy="batched_suffix", batch_size=4).evaluate(sets)
    assert max(abs(a.margin - b.margin) for a, b in zip(serial, both)) < 1e-4
    concept = [[EditSpec(layer=2, positions=[9, 10], delta=0.5 * torch.randn(2, 64, generator=gen))] for _ in range(5)]
    serial = PatchEvaluator(m, ids, answers, mode="fp32_full").evaluate(concept)
    both = PatchEvaluator(m, ids, answers, mode="fp32_full", strategy="batched_suffix", batch_size=4).evaluate(concept)
    assert max(abs(a.margin - b.margin) for a, b in zip(serial, both)) < 1e-4
    # An edit inside the cached prefix can never be applied in a suffix run.
    import copy

    from saeselect.hooks import ResidualEdits

    cache = copy.deepcopy(PatchEvaluator(m, ids, answers, mode="bf16", strategy="suffix")._prefix_cache(10))
    with pytest.raises(ValueError, match="cached prefix"):
        with ResidualEdits(m.layers, [EditSpec(layer=1, positions=[4], delta=torch.zeros(1, 64))], position_offset=10):
            m.text_module(input_ids=ids[:, 10:], past_key_values=cache, use_cache=True)


def test_fp32_residual_mode_keeps_small_deltas():
    m = wrap(tiny.build_hf_model(dtype="bfloat16"), tokenizer=None)
    ids = torch.randint(4, 400, (1, 12), generator=torch.Generator().manual_seed(2))
    ids[0, 0] = 2
    delta = 1e-3 * torch.randn(1, 64, generator=torch.Generator().manual_seed(3))
    ratios = {}
    for mode in ("bf16", "fp32_residual"):
        res = PatchEvaluator(m, ids, AnswerSets(plus=(5,), minus=(6,)), mode=mode).evaluate([[EditSpec(layer=2, positions=[5], delta=delta)]])[0]
        ratios[mode] = res.realized_norm / res.intended_norm
    assert abs(ratios["fp32_residual"] - 1) < 1e-3
    assert abs(ratios["bf16"] - 1) > 0.05
    with precision_mode(m, "fp32_residual"):
        pass
    assert all(len(mod._forward_hooks) == 0 for mod in m.hf_model.modules())
