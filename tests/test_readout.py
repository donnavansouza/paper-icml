"""Readout: log-sum-exp margins, g against autograd, FP32 head against the model, ties."""

from __future__ import annotations

import math

import pytest
import torch

from saeselect import readout as ro
from saeselect.hooks import ResidualRecorder
from saeselect.lens import final_norm_vjp
from saeselect.readout import AnswerSets, HeadSpec


def test_answer_sets_validation():
    with pytest.raises(ValueError, match="overlap"):
        AnswerSets(plus=(1, 2), minus=(2, 3))
    with pytest.raises(ValueError, match="non-empty"):
        AnswerSets(plus=(), minus=(3,))
    with pytest.raises(ValueError, match="repeat"):
        AnswerSets(plus=(1, 1), minus=(3,))
    assert AnswerSets(plus=(1,), minus=(2, 3)).swapped() == AnswerSets(plus=(2, 3), minus=(1,))


def test_logsumexp_margin_by_hand():
    z_plus = torch.tensor([1.0, 2.0])
    z_minus = torch.tensor([0.5])
    expected = math.log(math.exp(1) + math.exp(2)) - 0.5
    assert float(ro.margin_from_logits(z_plus, z_minus)) == pytest.approx(expected, rel=1e-6)


def test_metric_gradient_matches_autograd():
    z_plus = torch.tensor([1.0, -0.3, 2.0], requires_grad=True)
    z_minus = torch.tensor([0.2, 0.9], requires_grad=True)
    m = ro.margin_from_logits(z_plus, z_minus)
    gp, gm = torch.autograd.grad(m, (z_plus, z_minus))
    ap, am = ro.metric_grad_logits(z_plus.detach(), z_minus.detach())
    assert torch.allclose(gp, ap) and torch.allclose(gm, am)
    assert float(ap.sum()) == pytest.approx(1.0) and float(am.sum()) == pytest.approx(-1.0)


@pytest.mark.parametrize("softcap", [None, 5.0])
def test_metric_grad_raw_includes_softcap(softcap):
    gen = torch.Generator().manual_seed(0)
    spec = HeadSpec(gamma=1 + 0.2 * torch.randn(16, generator=gen), eps=1e-6, W_U=torch.randn(40, 16, generator=gen), softcap=softcap, gain_form="1+weight")
    answers = AnswerSets(plus=(3, 7), minus=(11,))
    h = 4 * torch.randn(16, generator=gen)
    z_raw = ro.raw_logits_fp32(h, spec, answers.ids).detach().requires_grad_(True)
    z = z_raw if softcap is None else softcap * torch.tanh(z_raw / softcap)
    m = ro.margin_from_logits(z[:2], z[2:])
    (g_auto,) = torch.autograd.grad(m, z_raw)
    assert torch.allclose(ro.metric_grad_raw(h, spec, answers), g_auto, atol=1e-6)


def test_output_direction_chain_rule(tiny_model):
    """dm/dh_f = N^T (W_U^T g_raw): the pieces the lens scores are built from."""
    spec = tiny_model.head
    answers = AnswerSets(plus=(5, 9), minus=(40, 41, 42))
    h = (3 * torch.randn(tiny_model.d_model, generator=torch.Generator().manual_seed(1))).requires_grad_(True)
    (grad,) = torch.autograd.grad(ro.answer_margin(h, spec, answers), h)
    w = ro.output_direction(h.detach(), spec, answers)
    assert torch.allclose(final_norm_vjp(w, h.detach(), spec), grad, atol=1e-5)


def test_fp32_head_matches_model_logits(tiny_model, input_ids):
    last = tiny_model.n_layers - 1
    with ResidualRecorder(tiny_model.layers, at=[last]) as rec:
        tiny_model.forward(input_ids)
    ours = ro.logits_fp32(rec.activations[last][0], tiny_model.head)
    theirs = tiny_model.hf_model(input_ids=input_ids).logits[0].float()
    assert torch.allclose(ours, theirs, atol=1e-5)
    subset = ro.logits_fp32(rec.activations[last][0], tiny_model.head, [3, 17, 200])
    assert torch.allclose(subset, ours[:, [3, 17, 200]], atol=1e-6)


def test_full_readout_mass_and_margin(tiny_model, input_ids, monkeypatch):
    monkeypatch.setattr(ro, "VOCAB_CHUNK", 100)  # exercise chunking over the 512-token vocabulary
    last = tiny_model.n_layers - 1
    with ResidualRecorder(tiny_model.layers, at=[last]) as rec:
        tiny_model.forward(input_ids)
    h = rec.activations[last][0, -1]
    answers = AnswerSets(plus=(5, 9), minus=(40, 41))
    out = ro.full_readout(h, tiny_model.head, answers)
    probs = torch.softmax(tiny_model.hf_model(input_ids=input_ids).logits[0, -1].float(), -1)
    assert float(out["answer_mass"]) == pytest.approx(float(probs[[5, 9, 40, 41]].sum()), rel=1e-5)
    assert float(out["margin"]) == pytest.approx(float(ro.answer_margin(h, tiny_model.head, answers)), abs=1e-5)
    assert float(out["p_fav"]) == pytest.approx(1 / (1 + math.exp(-float(out["margin"]))), rel=1e-6)


def test_bf16_head_differs_from_fp32_head(tiny_model_bf16, input_ids):
    last = tiny_model_bf16.n_layers - 1
    with ResidualRecorder(tiny_model_bf16.layers, at=[last]) as rec:
        tiny_model_bf16.forward(input_ids)
    answers = AnswerSets(plus=(5, 9), minus=(40, 41))
    h = rec.activations[last][0, 1:]
    m32 = ro.answer_margin(h, tiny_model_bf16.head, answers)
    m16 = ro.margin_bf16_head(h, tiny_model_bf16.head, answers)
    diff = (m32 - m16).abs()
    assert float(diff.max()) > 0
    assert float(diff.max()) < 0.5


def test_ties_are_never_flips():
    assert ro.is_tie(0.01, tau_tie=0.05)
    assert not ro.is_tie(-0.2, tau_tie=0.05)
    assert ro.decision_flipped(0.01, -3.0, tau_tie=0.05) is None
    assert ro.decision_flipped(2.0, -0.04, tau_tie=0.05) is None
    assert ro.decision_flipped(2.0, -3.0, tau_tie=0.05) is True
    assert ro.decision_flipped(2.0, 3.0, tau_tie=0.05) is False
