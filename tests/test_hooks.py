"""Hooks: sham is bit-identical, hooks are removed after an exception, BOS is never a site,
norms are recorded, edits are causal, and the grad root gives d metric / d h_l."""

from __future__ import annotations

import pytest
import torch

from saeselect.hooks import EditSpec, ResidualEdits, ResidualRecorder, hook_count, validate_positions
from saeselect.readout import AnswerSets, answer_margin


def _final(model, input_ids, edits=()):
    last = model.n_layers - 1
    with ResidualRecorder(model.layers, at=[last]) as rec, ResidualEdits(model.layers, list(edits)):
        model.forward(input_ids)
    return rec.activations[last]


@pytest.mark.parametrize("fixture", ["tiny_model", "tiny_model_bf16"])
def test_sham_edit_is_bit_identical(request, fixture, input_ids):
    model = request.getfixturevalue(fixture)
    clean = _final(model, input_ids)
    zero = torch.zeros(3, model.d_model)
    edits = [EditSpec(layer=2, positions=[3, 5, 9], delta=zero)]
    shammed = _final(model, input_ids, edits)
    assert torch.equal(clean, shammed)
    assert hook_count(model.hf_model) == 0


def test_hooks_removed_after_exception(tiny_model, input_ids):
    def boom(h):
        raise RuntimeError("inside hook")

    with pytest.raises(RuntimeError, match="inside hook"):
        with ResidualRecorder(tiny_model.layers, at=[1, 4]), ResidualEdits(
            tiny_model.layers, [EditSpec(layer=2, positions=[4], delta_fn=boom)]
        ):
            tiny_model.forward(input_ids)
    assert hook_count(tiny_model.hf_model) == 0


def test_bos_is_never_a_site():
    with pytest.raises(ValueError, match="BOS"):
        validate_positions([0, 3], seq_len=10)
    with pytest.raises(ValueError, match="out of range"):
        validate_positions([3, 10], seq_len=10)
    assert validate_positions([5, 3, 3], seq_len=10) == [3, 5]


def test_bos_edit_refused_at_forward(tiny_model, input_ids):
    edit = EditSpec(layer=1, positions=[0], delta=torch.ones(1, tiny_model.d_model))
    with pytest.raises(ValueError, match="BOS"):
        with ResidualEdits(tiny_model.layers, [edit]):
            tiny_model.forward(input_ids)
    assert hook_count(tiny_model.hf_model) == 0


def test_edit_is_causal_and_changes_later_positions(tiny_model, input_ids):
    clean = _final(tiny_model, input_ids)
    delta = torch.randn(1, tiny_model.d_model, generator=torch.Generator().manual_seed(0))
    edited = _final(tiny_model, input_ids, [EditSpec(layer=2, positions=[6], delta=delta)])
    assert torch.equal(clean[0, :6], edited[0, :6])
    assert not torch.allclose(clean[0, 6:], edited[0, 6:])


def test_norms_recorded_fp32_and_bf16(tiny_model, tiny_model_bf16, input_ids):
    gen = torch.Generator().manual_seed(1)
    delta = 1e-3 * torch.randn(2, tiny_model.d_model, generator=gen)
    for model, exact in ((tiny_model, True), (tiny_model_bf16, False)):
        edits = ResidualEdits(model.layers, [EditSpec(layer=3, positions=[2, 7], delta=delta)])
        with edits:
            model.forward(input_ids)
        record = edits.records[0]
        assert record.calls == 1
        intended = record.intended_norm[0]
        realized = record.realized_norm[0]
        assert intended == pytest.approx(float(delta.norm()), rel=1e-6)
        if exact:
            assert realized == pytest.approx(intended, rel=1e-3)
        else:
            # BF16 rounding changes a small delta; the realized norm records it.
            assert realized != pytest.approx(intended, rel=1e-3)


def test_edit_scale_and_delta_fn(tiny_model, input_ids):
    delta = torch.randn(1, tiny_model.d_model, generator=torch.Generator().manual_seed(2))
    a = _final(tiny_model, input_ids, [EditSpec(layer=2, positions=[5], delta=delta, scale=2.0)])
    b = _final(tiny_model, input_ids, [EditSpec(layer=2, positions=[5], delta_fn=lambda h: 2.0 * delta.expand_as(h))])
    assert torch.allclose(a, b)


def test_grad_root_matches_finite_difference(input_ids):
    """d m / d h_l(p) from the grad root equals a finite difference (the AtP gradient).

    The model and the margin run in float64 here so the finite difference is
    not limited by rounding; this checks the hook plumbing, not the head.
    Gemma 3's RMSNorm and attention softmax still compute in FP32 inside a
    float64 model, so the step must stay well above FP32 noise (1e-4 is not;
    1e-3 to 1e-1 agree with the gradient to about 1e-4)."""
    from saeselect import tiny
    from saeselect.models import wrap

    model = wrap(tiny.build_hf_model(dtype="float64"), tokenizer=None)
    spec = model.head
    ids = torch.tensor([10, 11, 20])

    def m64(h):
        x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + spec.eps) * spec.gamma.double()
        z = x @ spec.W_U[ids].double().T
        return torch.logsumexp(z[:2], -1) - z[2]

    layer, pos, last = 2, 7, model.n_layers - 1
    with ResidualRecorder(model.layers, at=[last], grad_at=layer) as rec:
        model.forward(input_ids)
        (grad,) = torch.autograd.grad(m64(rec.activations[last][0, -1]), rec.activations[layer])
    g = grad[0, pos]
    gen = torch.Generator().manual_seed(3)
    for v in (g / g.norm(), torch.randn(model.d_model, generator=gen, dtype=torch.float64)):
        v = v / v.norm()
        eps = 1e-2

        def margin_with(t):
            h = _final(model, input_ids, [EditSpec(layer=layer, positions=[pos], delta=(t * v)[None])])
            return float(m64(h[0, -1]))

        fd = (margin_with(eps) - margin_with(-eps)) / (2 * eps)
        assert float(g @ v) == pytest.approx(fd, rel=5e-4, abs=1e-4 * float(g.norm()))
