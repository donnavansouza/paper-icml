"""SAE: shapes and orientation, JumpReLU thresholding, delta-form arithmetic against
hand-computed values, and FULL = SAE-ALL + ERR."""

from __future__ import annotations

import pytest
import torch

from saeselect import interventions as iv
from saeselect.provenance import EngineeringFailed
from saeselect.sae import JumpReLUSAE, folder_name, from_tensors


def hand_sae() -> JumpReLUSAE:
    """d = 2, width = 3 with round numbers."""
    return JumpReLUSAE(
        W_enc=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        b_enc=torch.tensor([0.0, -0.5, 0.0]),
        W_dec=torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        b_dec=torch.tensor([0.1, -0.1]),
        threshold=torch.tensor([0.5, 0.2, 2.5]),
    )


def test_jumprelu_thresholding_by_hand():
    sae = hand_sae()
    h = torch.tensor([[1.0, 1.0], [0.4, 2.0]])
    # pre = h W_enc + b_enc: row0 [1, 0.5, 2]; row1 [0.4, 1.5, 2.4]
    # thresholds [0.5, 0.2, 2.5] -> row0 [1, 0.5, 0]; row1 [0, 1.5, 0]
    expected = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.5, 0.0]])
    assert torch.allclose(sae.encode(h), expected)
    # decode: a W_dec + b_dec
    assert torch.allclose(sae.decode(expected), torch.tensor([[1.1, 0.9], [0.1, 2.9]]))
    assert torch.allclose(sae.error(h), h - torch.tensor([[1.1, 0.9], [0.1, 2.9]]))


def test_pre_activation_at_threshold_is_inactive():
    sae = hand_sae()
    h = torch.tensor([[0.5, 0.0]])  # pre[0] == threshold[0] exactly
    assert sae.encode(h)[0, 0] == 0


def test_shapes_and_orientation_asserted():
    d, n = 4, 6
    good = {
        "w_enc": torch.zeros(d, n), "w_dec": torch.zeros(n, d),
        "b_enc": torch.zeros(n), "b_dec": torch.zeros(d), "threshold": torch.zeros(n),
    }
    assert from_tensors(good, d_model=d, width=n, meta={}).width == n
    with pytest.raises(EngineeringFailed, match="orientation"):
        from_tensors({**good, "w_enc": torch.zeros(n, d)}, d_model=d, width=n, meta={})
    with pytest.raises(EngineeringFailed, match="orientation"):
        from_tensors({**good, "w_dec": torch.zeros(d, n)}, d_model=d, width=n, meta={})
    with pytest.raises(EngineeringFailed, match="missing"):
        from_tensors({k: v for k, v in good.items() if k != "threshold"}, d_model=d, width=n, meta={})
    with pytest.raises(EngineeringFailed, match="shape"):
        JumpReLUSAE(W_enc=torch.zeros(d, n), b_enc=torch.zeros(n + 1), W_dec=torch.zeros(n, d), b_dec=torch.zeros(d), threshold=torch.zeros(n))


def test_folder_name_pattern():
    assert folder_name("resid_post", 31, "65k", "medium") == "resid_post/layer_31_width_65k_l0_medium"
    assert folder_name("resid_post", 31, "262k", "medium", "seed_1") == "resid_post/layer_31_width_262k_l0_medium_seed_1"


def test_transfer_and_ablation_by_hand():
    sae = hand_sae()
    h_src = torch.tensor([[1.0, 1.0]])  # a = [1, 0.5, 0]
    h_don = torch.tensor([[0.4, 2.0]])  # a = [0, 1.5, 0]
    da = iv.transfer_activation_delta(sae, h_src, h_don)
    assert torch.allclose(da, torch.tensor([[-1.0, 1.0, 0.0]]))
    assert iv.candidate_pool(da).tolist() == [0, 1]
    # transfer of feature 1 only: +1.0 * d_1 = [0, 2]
    assert torch.allclose(iv.transfer_delta(sae, h_src, h_don, [1]), torch.tensor([[0.0, 2.0]]))
    # transfer of both: -1 * [1, 0] + 1 * [0, 2]
    assert torch.allclose(iv.transfer_delta(sae, h_src, h_don, [0, 1]), torch.tensor([[-1.0, 2.0]]))
    # mean ablation of features 0 and 1 to abar = [0.25, 0.25, 0]: (0.25-1) d_0 + (0.25-0.5) d_1
    abar = torch.tensor([0.25, 0.25, 0.0])
    assert torch.allclose(iv.mean_ablation_delta(sae, h_src, [0, 1], abar), torch.tensor([[-0.75, -0.5]]))
    # zero ablation of feature 0: -1 * d_0
    assert torch.allclose(iv.zero_ablation_delta(sae, h_src, [0]), torch.tensor([[-1.0, 0.0]]))
    assert torch.allclose(iv.transfer_delta(sae, h_src, h_don, []), torch.zeros(1, 2))


def test_full_equals_sae_all_plus_err(tiny_sae):
    gen = torch.Generator().manual_seed(7)
    h_src = torch.randn(5, tiny_sae.d_model, generator=gen)
    h_don = torch.randn(5, tiny_sae.d_model, generator=gen)
    full = iv.full_delta(h_src, h_don)
    total = iv.sae_all_delta(tiny_sae, h_src, h_don) + iv.err_delta(tiny_sae, h_src, h_don)
    assert torch.allclose(total, full, atol=1e-5 * float(full.abs().max()))
    pool = iv.candidate_pool(iv.transfer_activation_delta(tiny_sae, h_src, h_don))
    assert pool.numel() > 0, "random SAE should have active features"
    assert torch.allclose(iv.transfer_delta(tiny_sae, h_src, h_don, pool), iv.sae_all_delta(tiny_sae, h_src, h_don))


def test_reconstruction_replacement_is_minus_error(tiny_sae):
    h = torch.randn(3, tiny_sae.d_model, generator=torch.Generator().manual_seed(8))
    assert torch.allclose(h + iv.reconstruction_replacement_delta(tiny_sae, h), tiny_sae.reconstruct(h), atol=1e-6)


def test_direction_operators():
    gen = torch.Generator().manual_seed(9)
    u = iv.unit(torch.randn(8, generator=gen))
    h = torch.randn(4, 8, generator=gen)
    h_don = torch.randn(4, 8, generator=gen)
    c = 0.7
    # ACE with alpha = 1 sets the projection to c; alpha = 0.5 halves the distance.
    assert torch.allclose((h + iv.ace_delta(h, u, c, 1.0)) @ u, torch.full((4,), c), atol=1e-6)
    assert torch.allclose((h + iv.ace_delta(h, u, c, 0.5)) @ u - c, 0.5 * (h @ u - c), atol=1e-6)
    # DIM-ABL with alpha = 1 removes the projection.
    assert torch.allclose((h + iv.dim_ablation_delta(h, u, 1.0)) @ u, torch.zeros(4), atol=1e-6)
    # DIM moves the projection onto the donor's and leaves the orthogonal part.
    moved = h + iv.dim_delta(h, h_don, u)
    assert torch.allclose(moved @ u, h_don @ u, atol=1e-6)
    ortho = lambda x: x - (x @ u).unsqueeze(-1) * u  # noqa: E731
    assert torch.allclose(ortho(moved), ortho(h), atol=1e-6)
    with pytest.raises(ValueError, match="unit-norm"):
        iv.ace_delta(h, 2 * u, c, 1.0)


def test_quadrature_norm():
    delta = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    assert iv.quadrature_norm(delta) == pytest.approx(5.0)
