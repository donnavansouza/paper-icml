"""Delta-form interventions.

Every function returns an FP32 delta of shape [|P|, d] to add at the site
positions through :class:`saeselect.hooks.EditSpec`. None of them replaces the
residual with a reconstruction, so the SAE error term is always preserved;
reconstruction replacement exists only as the diagnostic
:func:`reconstruction_replacement_delta` and is never used as an intervention.

Inputs ``h``, ``h_src``, ``h_don`` are the clean residuals at the site
positions [|P|, d]. Transfer needs source and donor aligned position by
position over P.

Identities checked by the tests: FULL = SAE-ALL + ERR exactly in FP32 (b_dec
cancels); transfer of every active feature equals SAE-ALL.
"""

from __future__ import annotations

import torch

from saeselect.sae import JumpReLUSAE


def _f32(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32)


def _features(features: torch.Tensor | list[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(features, device=device, dtype=torch.long)


def quadrature_norm(delta: torch.Tensor) -> float:
    """Norm of a delta summed in quadrature over positions: sqrt(sum_p ||delta(p)||^2)."""
    return float(delta.float().pow(2).sum().sqrt())


# -- activation changes ------------------------------------------------------


def transfer_activation_delta(sae: JumpReLUSAE, h_src: torch.Tensor, h_don: torch.Tensor) -> torch.Tensor:
    """delta a_i(p) = a_i(h_don(p)) - a_i(h_src(p)), shape [|P|, width]."""
    if h_src.shape != h_don.shape:
        raise ValueError("transfer needs source and donor aligned over the same positions")
    return sae.encode(h_don) - sae.encode(h_src)


def mean_ablation_activation_delta(sae: JumpReLUSAE, h: torch.Tensor, abar: torch.Tensor) -> torch.Tensor:
    """delta a_i(p) = abar_i - a_i(h(p)), shape [|P|, width]."""
    return _f32(abar, sae.device).unsqueeze(0) - sae.encode(h)


def candidate_pool(activation_delta: torch.Tensor) -> torch.Tensor:
    """Features whose delta is non-zero at some site position (section 5.4)."""
    return torch.nonzero((activation_delta != 0).any(dim=0), as_tuple=False).flatten()


def delta_from_activation_change(sae: JumpReLUSAE, activation_delta: torch.Tensor, features: torch.Tensor | list[int]) -> torch.Tensor:
    """sum_{i in S} delta a_i(p) d_i, shape [|P|, d]."""
    idx = _features(features, sae.device)
    if idx.numel() == 0:
        return torch.zeros(activation_delta.shape[0], sae.d_model, device=sae.device)
    return activation_delta.index_select(1, idx) @ sae.W_dec.index_select(0, idx)


# -- feature interventions ---------------------------------------------------


def transfer_delta(sae: JumpReLUSAE, h_src: torch.Tensor, h_don: torch.Tensor, features: torch.Tensor | list[int]) -> torch.Tensor:
    """T(S): h' = h_src + sum_{i in S} [a_i(h_don) - a_i(h_src)] d_i."""
    return delta_from_activation_change(sae, transfer_activation_delta(sae, h_src, h_don), features)


def sae_all_delta(sae: JumpReLUSAE, h_src: torch.Tensor, h_don: torch.Tensor) -> torch.Tensor:
    """SAE-ALL: transfer of every feature active in either prompt = x_hat(don) - x_hat(src)."""
    da = transfer_activation_delta(sae, h_src, h_don)
    return delta_from_activation_change(sae, da, candidate_pool(da))


def mean_ablation_delta(sae: JumpReLUSAE, h: torch.Tensor, features: torch.Tensor | list[int], abar: torch.Tensor) -> torch.Tensor:
    """MA(S): h' = h + sum_{i in S} [abar_i - a_i(h)] d_i."""
    return delta_from_activation_change(sae, mean_ablation_activation_delta(sae, h, abar), features)


def zero_ablation_delta(sae: JumpReLUSAE, h: torch.Tensor, features: torch.Tensor | list[int]) -> torch.Tensor:
    """ZA(S): h' = h - sum_{i in S} a_i(h) d_i."""
    return delta_from_activation_change(sae, -sae.encode(h), features)


# -- references --------------------------------------------------------------


def full_delta(h_src: torch.Tensor, h_don: torch.Tensor) -> torch.Tensor:
    """FULL: h' = h_don (as a delta, h_don - h_src)."""
    return h_don.float() - h_src.float()


def err_delta(sae: JumpReLUSAE, h_src: torch.Tensor, h_don: torch.Tensor) -> torch.Tensor:
    """ERR: h' = h_src + eps(h_don) - eps(h_src)."""
    return sae.error(h_don) - sae.error(h_src)


def _unit(u: torch.Tensor) -> torch.Tensor:
    u = u.float()
    norm = u.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("direction has zero or non-finite norm")
    if abs(float(norm) - 1.0) > 1e-4:
        raise ValueError(f"direction must be unit-norm, got norm {float(norm):.6f}")
    return u


def unit(v: torch.Tensor) -> torch.Tensor:
    v = v.float()
    return v / v.norm()


def dim_delta(h_src: torch.Tensor, h_don: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """DIM: h' = h_src + [u . (h_don - h_src)] u, u the unit mean difference over support prompts."""
    u = _unit(u).to(h_src.device)
    diff = h_don.float() - h_src.float()
    return (diff @ u).unsqueeze(-1) * u


def ace_delta(h: torch.Tensor, u: torch.Tensor, c: float, alpha: float) -> torch.Tensor:
    """ACE(alpha): h' = h - alpha (u . h - c) u."""
    u = _unit(u).to(h.device)
    return -alpha * ((h.float() @ u) - c).unsqueeze(-1) * u


def dim_ablation_delta(h: torch.Tensor, u: torch.Tensor, alpha: float) -> torch.Tensor:
    """DIM-ABL(alpha): h' = h - alpha (u . h) u (projection removal)."""
    u = _unit(u).to(h.device)
    return -alpha * (h.float() @ u).unsqueeze(-1) * u


def reconstruction_replacement_delta(sae: JumpReLUSAE, h: torch.Tensor) -> torch.Tensor:
    """DIAGNOSTIC ONLY: replacing h by its reconstruction, as a delta (-eps(h)).
    Never an intervention."""
    return -sae.error(h)
