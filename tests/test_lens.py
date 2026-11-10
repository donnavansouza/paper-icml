"""Lens: linearized score against a finite difference of the readout, LL = JL when J = I,
kappa against a reference kurtosis, JSPACE on planted and random directions, and
parity with the jlens reference implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from saeselect import lens as L
from saeselect.hooks import ResidualRecorder
from saeselect.readout import AnswerSets, HeadSpec, answer_margin, final_norm_fp32, output_direction, vocabulary_direction


def _final_residual(model, input_ids):
    last = model.n_layers - 1
    with ResidualRecorder(model.layers, at=[last]) as rec:
        model.forward(input_ids)
    return rec.activations[last][0, -1]


def test_final_norm_jvp_matches_autograd(tiny_model):
    spec = tiny_model.head
    gen = torch.Generator().manual_seed(0)
    h = 5 * torch.randn(tiny_model.d_model, generator=gen)
    v = torch.randn(tiny_model.d_model, generator=gen)
    _, jvp = torch.autograd.functional.jvp(lambda x: final_norm_fp32(x, spec), (h,), (v,))
    assert torch.allclose(L.final_norm_jvp(v, h, spec), jvp, atol=1e-6)
    w = torch.randn(tiny_model.d_model, generator=gen)
    # N^T is the adjoint of N.
    assert float(w @ L.final_norm_jvp(v, h, spec)) == pytest.approx(float(L.final_norm_vjp(w, h, spec) @ v), rel=1e-5)


@pytest.mark.parametrize("use_lens", [False, True])
def test_linearized_score_matches_finite_difference(tiny_model, tiny_lens, input_ids, use_lens):
    """g^T W_U N(J d) equals d/dt m(h_f + t J d) at t = 0.

    The finite difference runs in float64 on a float64 copy of the readout
    (final norm, answer rows of W_U, log-sum-exp margin), checked first against
    the FP32 head at h_f. An FP32 finite difference is noise-limited at 1e-4 to
    1e-3 relative and failed across platforms (GPU machine, 2026-09-28)."""
    spec = tiny_model.head
    answers = AnswerSets(plus=(5, 9), minus=(40, 41, 42))
    rows = spec.W_U[torch.tensor(answers.ids)].double()

    def m64(h: torch.Tensor) -> float:
        h = h.double()
        z = (h * torch.rsqrt(h.pow(2).mean() + spec.eps) * spec.gamma.double()) @ rows.T
        return float(torch.logsumexp(z[:2], -1) - torch.logsumexp(z[2:], -1))

    h_f = _final_residual(tiny_model, input_ids)
    assert m64(h_f) == pytest.approx(float(answer_margin(h_f, spec, answers)), abs=1e-5)
    J = tiny_lens.J(3) if use_lens else None
    s = L.lens_sensitivity(output_direction(h_f, spec, answers), h_f, spec, J)
    d = torch.randn(4, tiny_model.d_model, generator=torch.Generator().manual_seed(1))
    scores = L.direction_scores(d, s)
    for i in range(4):
        direction = (d[i] if J is None else L.transport(d[i], J)).double()
        t = 1e-4 * float(h_f.norm()) / float(direction.norm())
        fd = (m64(h_f.double() + t * direction) - m64(h_f.double() - t * direction)) / (2 * t)
        assert float(scores[i]) == pytest.approx(fd, rel=1e-4, abs=1e-7)


def test_logit_lens_twin_equals_jlens_with_identity(tiny_model, input_ids):
    spec = tiny_model.head
    h_f = _final_residual(tiny_model, input_ids)
    eye = torch.eye(tiny_model.d_model)
    for w in (output_direction(h_f, spec, AnswerSets(plus=(5,), minus=(6,))), vocabulary_direction(spec, [1, 2], [3], "cpu")):
        assert torch.allclose(L.lens_sensitivity(w, h_f, spec, eye), L.lens_sensitivity(w, h_f, spec, None))
    d = torch.randn(10, tiny_model.d_model, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(L.kappa(d, spec, eye), L.kappa(d, spec, None), atol=1e-5)
    frac_eye, _ = L.jspace_alignment(d[:3], L.AtomBank(spec, eye, "cpu"), max_atoms=5)
    frac_none, _ = L.jspace_alignment(d[:3], L.AtomBank(spec, None, "cpu"), max_atoms=5)
    assert np.allclose(frac_eye, frac_none, atol=1e-5)


def test_kappa_matches_reference_kurtosis(monkeypatch):
    monkeypatch.setattr(L, "VOCAB_CHUNK", 97)
    monkeypatch.setattr(L, "FEATURE_CHUNK", 3)
    gen = torch.Generator().manual_seed(3)
    d, V = 12, 1000
    W_U = torch.randn(V, d, generator=gen)
    W_U[:5] *= 20  # heavy tail on a few tokens
    x = torch.randn(7, d, generator=gen)
    W_U_before = W_U.clone()
    ours = L.excess_kurtosis_over_vocab(x, W_U)
    assert torch.equal(W_U, W_U_before)  # an FP32 W_U must not be modified in place
    reference = stats.kurtosis((W_U @ x.T).double().numpy(), axis=0, fisher=True, bias=True)
    assert np.allclose(ours.numpy(), reference, rtol=1e-4, atol=1e-4)
    spec = HeadSpec(gamma=1 + 0.1 * torch.rand(d, generator=gen), eps=1e-6, W_U=W_U, softcap=None, gain_form="1+weight")
    J = torch.eye(d) + 0.1 * torch.randn(d, d, generator=gen)
    whole = L.excess_kurtosis_over_vocab(L.lens_directions(x, spec, J), W_U)
    assert torch.allclose(L.kappa(x, spec, J), whole, rtol=1e-4)  # chunk-wise lens directions equal the whole-matrix ones
    # kappa is scale-free
    assert torch.allclose(L.excess_kurtosis_over_vocab(3.7 * x, W_U), ours, rtol=1e-4)


def _random_spec(d: int, V: int, seed: int) -> HeadSpec:
    gen = torch.Generator().manual_seed(seed)
    return HeadSpec(gamma=1 + 0.3 * torch.rand(d, generator=gen), eps=1e-6, W_U=torch.randn(V, d, generator=gen), softcap=None, gain_form="1+weight")


def test_jspace_recovers_planted_combination_and_rejects_random():
    d, V = 256, 1500
    spec = _random_spec(d, V, seed=4)
    gen = torch.Generator().manual_seed(5)
    J = torch.eye(d) + 0.05 * torch.randn(d, d, generator=gen)
    bank = L.AtomBank(spec, J, "cpu")
    planted_ids = [17, 400, 901, 1333]
    coefs = torch.tensor([2.0, 1.0, 0.7, 1.5])
    planted = (coefs.unsqueeze(1) * bank.atoms(planted_ids)).sum(0)
    random_dir = torch.randn(d, generator=gen)
    frac, chosen = L.jspace_alignment(torch.stack([planted, random_dir]), bank, max_atoms=25)
    assert frac[0] > 0.999
    assert set(planted_ids) <= set(chosen[0])
    assert frac[1] < 0.6
    assert len(chosen[1]) <= 25 and len(set(chosen[1])) == len(chosen[1])


def test_lens_file_roundtrip(tmp_path):
    d = 6
    J = {0: torch.randn(d, d).half(), 3: torch.randn(d, d).half()}
    path = tmp_path / "lens.pt"
    torch.save({"J": J, "n_prompts": 10, "source_layers": [0, 3], "d_model": d}, path)
    lens = L.load_lens_file(str(path))
    assert lens.layers == [0, 3] and lens.stored_dtype == "torch.float16"
    assert lens.J(3).dtype == torch.float32
    assert torch.equal(lens.J(3), J[3].float())


class _StubTokenizer:
    """Just enough for jlens.HFLensModel.encode: fixed token ids for any text."""

    bos_token_id = 2

    def __init__(self, ids):
        self._ids = ids

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=self._ids.clone())


def test_parity_with_jlens_apply(tiny_model, tiny_lens, input_ids):
    """Our layer capture + transport + reference unembed reproduce lens.apply (N2 on the tiny model)."""
    jlens = pytest.importorskip("jlens")
    lens_model = jlens.from_hf(tiny_model.hf_model, _StubTokenizer(input_ids))
    assert lens_model.n_layers == tiny_model.n_layers
    reference = jlens.JacobianLens(jacobians=dict(tiny_lens.jacobians), n_prompts=0, d_model=tiny_model.d_model)
    layers = [1, 3, 4]
    positions = [3, 8, -1]
    lens_logits, model_logits, _ = reference.apply(lens_model, "ignored", layers=layers, positions=positions)
    with ResidualRecorder(tiny_model.layers, at=layers) as rec:
        tiny_model.forward(input_ids)
    for layer in layers:
        h = rec.activations[layer][0, positions]
        ours = L.jlens_unembed(L.transport(h, tiny_lens.J(layer)), tiny_model.final_norm, tiny_model.lm_head, tiny_model.head.softcap).float()
        assert torch.allclose(ours, lens_logits[layer], atol=1e-5)
        assert torch.equal(ours.topk(10).indices, lens_logits[layer].topk(10).indices)
