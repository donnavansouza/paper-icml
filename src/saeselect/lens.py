"""J-lens loading, parity readout, linearized direction scores, kappa and JSPACE.

Conventions follow the jlens reference implementation (commit 581d398):
J_l is stored per layer as [d, d] (fp16 on disk, used in FP32) and applied as
``v @ J.T``. ``jlens``'s ``unembed`` applies the final norm, the LM head and
soft-capping to whatever vector it gets, which divides each direction by its
own RMS. So direction scores are linearized here:

    N(v) = gamma * (v - h_f (h_f . v) / (h_f . h_f + d eps)) / rms(h_f)

is the exact Jacobian of the RMS final norm at the prompt's own final residual
h_f (rms(h) = sqrt(mean(h^2) + eps)). Section 5.4 writes the projection
denominator as h_f . h_f; the ``d eps`` term makes it exact and is negligible
at Gemma's residual norms (relative size d eps / ||h_f||^2).

The J-lens score of a feature direction d_i for a target w = W_U^T g is
    g^T W_U N(J d_i) = s . d_i  with  s = J^T N^T w,
and the logit-lens twin is the same with J = I. Only J differs between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.optimize import nnls

from saeselect.provenance import EngineeringFailed
from saeselect.readout import HeadSpec

# Memory knobs for vocabulary-sized products, not measured values.
FEATURE_CHUNK = 1024
VOCAB_CHUNK = 32768


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class JLens:
    """Per-layer J matrices from a jlens ``save`` file."""

    jacobians: dict[int, torch.Tensor]
    n_prompts: int
    d_model: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        return sorted(self.jacobians)

    def J(self, layer: int, device: torch.device | str = "cpu") -> torch.Tensor:
        if layer not in self.jacobians:
            raise KeyError(f"layer {layer} not in lens layers {self.layers}")
        return self.jacobians[layer].to(device=device, dtype=torch.float32)

    @property
    def stored_dtype(self) -> str:
        return str(next(iter(self.jacobians.values())).dtype)


def load_lens_file(path: str, *, meta: dict[str, Any] | None = None) -> JLens:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in checkpoint:
        raise EngineeringFailed("lens_file", f"{path} has no 'J' entry (keys {sorted(checkpoint)})")
    jacobians = {int(k): v for k, v in checkpoint["J"].items()}
    d_model = int(checkpoint["d_model"])
    for layer, J in jacobians.items():
        if tuple(J.shape) != (d_model, d_model):
            raise EngineeringFailed("lens_file", f"layer {layer} J has shape {tuple(J.shape)}, expected {(d_model, d_model)}")
    return JLens(jacobians=jacobians, n_prompts=int(checkpoint["n_prompts"]), d_model=d_model, meta=dict(meta or {}, path=str(path)))


def load_released_lens(*, repo_id: str, revision: str, filename: str) -> JLens:
    from huggingface_hub import hf_hub_download

    if len(revision) != 40:
        raise EngineeringFailed("lens_revision", f"revision must be a pinned 40-character SHA, got {revision!r}")
    path = hf_hub_download(repo_id, filename, revision=revision)
    return load_lens_file(path, meta={"repo_id": repo_id, "revision": revision, "filename": filename})


def identity_lens(d_model: int, layers: list[int]) -> JLens:
    eye = torch.eye(d_model)
    return JLens(jacobians={layer: eye for layer in layers}, n_prompts=0, d_model=d_model, meta={"identity": True})


# --------------------------------------------------------------------------
# Readout (parity with jlens)
# --------------------------------------------------------------------------


def transport(residual: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """J_l h as jlens computes it: ``h @ J.T`` in FP32."""
    return residual.float() @ J.to(residual.device, torch.float32).T


def jlens_unembed(residual: torch.Tensor, final_norm: torch.nn.Module, lm_head: torch.nn.Module, softcap: float | None) -> torch.Tensor:
    """The reference ``HFLensModel.unembed``: cast to the head dtype, the model's
    own final norm, the LM head, then soft-capping. Used only for parity (N2)."""
    W = lm_head.weight
    logits = lm_head(final_norm(residual.to(W.dtype).to(W.device)))
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


# --------------------------------------------------------------------------
# Linearized final norm and direction scores
# --------------------------------------------------------------------------


def _norm_terms(h_f: torch.Tensor, spec: HeadSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = h_f.float()
    if h.dim() != 1:
        raise ValueError("linearization point h_f must be one residual vector [d_model]")
    d = h.shape[0]
    rms = torch.sqrt(h.pow(2).mean() + spec.eps)
    denom = h.dot(h) + d * spec.eps
    return h, rms, denom


def final_norm_jvp(v: torch.Tensor, h_f: torch.Tensor, spec: HeadSpec) -> torch.Tensor:
    """N(v): the final norm's Jacobian at h_f applied to v ([..., d])."""
    h, rms, denom = _norm_terms(h_f, spec)
    v = v.float().to(h.device)
    projected = v - ((v @ h) / denom).unsqueeze(-1) * h
    return spec.gamma.to(h.device) * projected / rms


def final_norm_vjp(w: torch.Tensor, h_f: torch.Tensor, spec: HeadSpec) -> torch.Tensor:
    """N^T w ([..., d])."""
    h, rms, denom = _norm_terms(h_f, spec)
    u = spec.gamma.to(h.device) * w.float().to(h.device)
    return (u - ((u @ h) / denom).unsqueeze(-1) * h) / rms


def lens_sensitivity(w: torch.Tensor, h_f: torch.Tensor, spec: HeadSpec, J: torch.Tensor | None) -> torch.Tensor:
    """s = J^T N^T w, so that w^T N(J x) = s . x for every direction x.

    ``w`` is a residual-space readout direction: W_U^T g for OUT scores
    (:func:`saeselect.readout.output_direction`), W_U^T q for VERB scores
    (:func:`saeselect.readout.vocabulary_direction`). ``J=None`` is the logit lens.
    """
    s = final_norm_vjp(w, h_f, spec)
    if J is not None:
        s = s @ J.to(s.device, torch.float32)
    return s


def direction_scores(directions: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """s . d_i for each row d_i of ``directions`` ([k, d])."""
    return directions.float().to(s.device) @ s


# --------------------------------------------------------------------------
# KAPPA: excess kurtosis of each direction's full-vocabulary lens logits
# --------------------------------------------------------------------------


def lens_directions(directions: torch.Tensor, spec: HeadSpec, J: torch.Tensor | None) -> torch.Tensor:
    """gamma * (J d_i) for each row, FP32. The vocabulary projection of row i is W_U times this."""
    x = directions.float()
    if J is not None:
        x = x @ J.to(x.device, torch.float32).T
    return x * spec.gamma.to(x.device)


def _vocab_mean(W_U: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mean row of W_U in FP32, summed chunk by chunk in FP64 (no FP64 copy of a chunk)."""
    total = torch.zeros(W_U.shape[1], dtype=torch.float64, device=device)
    for start in range(0, W_U.shape[0], VOCAB_CHUNK):
        total += W_U[start : start + VOCAB_CHUNK].to(device=device, dtype=torch.float32).sum(0, dtype=torch.float64)
    return (total / W_U.shape[0]).float()


def excess_kurtosis_over_vocab(x: torch.Tensor, W_U: torch.Tensor, *, transform=None) -> torch.Tensor:
    """Excess kurtosis (population moments, Fisher: minus 3) of W_U x_i over the
    vocabulary, for each row x_i of ``x`` [n, d]. FP32 products and powers,
    FP64 accumulation, chunked over features and vocabulary so memory stays
    bounded for any width (a 262k SAE on 12B ran out of GPU memory when whole
    [n, d] copies were made; 2026-09-29). ``transform`` maps a chunk of rows to
    the directions whose projections are measured (applied per chunk)."""
    device = x.device
    V = W_U.shape[0]
    w_bar = _vocab_mean(W_U, device)
    out = torch.empty(x.shape[0], dtype=torch.float64, device=device)
    for f0 in range(0, x.shape[0], FEATURE_CHUNK):
        xc = x[f0 : f0 + FEATURE_CHUNK].float()
        if transform is not None:
            xc = transform(xc)
        m2 = torch.zeros(xc.shape[0], dtype=torch.float64, device=device)
        m4 = torch.zeros_like(m2)
        for v0 in range(0, V, VOCAB_CHUNK):
            # Out of place: if W_U is already FP32, .to() returns a view of it, and an
            # in-place subtraction would corrupt W_U itself.
            rows = W_U[v0 : v0 + VOCAB_CHUNK].to(device=device, dtype=torch.float32) - w_bar
            y2 = (xc @ rows.T).square_()
            m2 += y2.sum(1, dtype=torch.float64)
            m4 += y2.square_().sum(1, dtype=torch.float64)
            del rows, y2
        m2 /= V
        m4 /= V
        out[f0 : f0 + FEATURE_CHUNK] = m4 / (m2 * m2) - 3.0
    return out.float()


def kappa(directions: torch.Tensor, spec: HeadSpec, J: torch.Tensor | None) -> torch.Tensor:
    """KAPPA for each decoder row: excess kurtosis of W_U (gamma * J d_i). J=None is the logit-lens twin.
    The lens directions are formed chunk by chunk, never for the whole dictionary at once."""
    return excess_kurtosis_over_vocab(directions, spec.W_U, transform=lambda chunk: lens_directions(chunk, spec, J))


# --------------------------------------------------------------------------
# JSPACE: share of ||d_i||^2 captured by <= max_atoms J-lens vectors
# --------------------------------------------------------------------------


class AtomBank:
    """Unit-normalized J-lens vectors: row v of W_U diag(gamma) J, i.e.
    (gamma * W_U[v]) @ J, divided by its norm. Never materialized in full:
    inner products are W_U[v] . (gamma * (J r)) / ||atom_v||."""

    def __init__(self, spec: HeadSpec, J: torch.Tensor | None, device: torch.device | str):
        self.spec = spec
        self.device = torch.device(device)
        d = spec.d_model
        self.J = (J if J is not None else torch.eye(d)).to(self.device, torch.float32)
        self.gamma = spec.gamma.to(self.device, torch.float32)
        norms = []
        for start in range(0, spec.vocab_size, VOCAB_CHUNK):
            rows = spec.W_U[start : start + VOCAB_CHUNK].to(self.device, torch.float32) * self.gamma
            norms.append((rows @ self.J).norm(dim=1))
        self.norms = torch.cat(norms)
        self.valid = self.norms > 0

    def atoms(self, index: torch.Tensor | list[int]) -> torch.Tensor:
        idx = torch.as_tensor(index, device=self.spec.W_U.device, dtype=torch.long)
        rows = self.spec.W_U.index_select(0, idx).to(self.device, torch.float32) * self.gamma
        return (rows @ self.J) / self.norms[idx.to(self.device)].unsqueeze(1)

    def best_positive(self, residuals: torch.Tensor, exclude: list[set[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Largest inner product of each residual row with any atom not in its
        excluded set. Returns (values, indices)."""
        y = (residuals.float().to(self.device) @ self.J.T) * self.gamma  # [B, d]
        best_val = torch.full((y.shape[0],), -float("inf"), device=self.device)
        best_idx = torch.full((y.shape[0],), -1, dtype=torch.long, device=self.device)
        for start in range(0, self.spec.vocab_size, VOCAB_CHUNK):
            rows = self.spec.W_U[start : start + VOCAB_CHUNK].to(self.device, torch.float32)
            scores = (y @ rows.T) / self.norms[start : start + rows.shape[0]]
            scores[:, ~self.valid[start : start + rows.shape[0]]] = -float("inf")
            for b, excluded in enumerate(exclude):
                local = [i - start for i in excluded if start <= i < start + rows.shape[0]]
                if local:
                    scores[b, local] = -float("inf")
            val, idx = scores.max(dim=1)
            better = val > best_val
            best_val = torch.where(better, val, best_val)
            best_idx = torch.where(better, idx + start, best_idx)
        return best_val, best_idx


def jspace_alignment(directions: torch.Tensor, bank: AtomBank, *, max_atoms: int) -> tuple[np.ndarray, list[list[int]]]:
    """Greedy non-negative sparse approximation of each direction by at most
    ``max_atoms`` unit J-lens vectors: pick the atom with the largest positive
    inner product with the residual, refit non-negative coefficients over the
    chosen atoms (NNLS), repeat. Returns the captured share
    1 - ||residual||^2 / ||d_i||^2 per direction, and the chosen atoms."""
    targets = directions.float().to(bank.device)
    residuals = targets.clone()
    chosen: list[list[int]] = [[] for _ in range(targets.shape[0])]
    active = list(range(targets.shape[0]))
    for _ in range(max_atoms):
        if not active:
            break
        values, indices = bank.best_positive(residuals[active], [set(chosen[i]) for i in active])
        still_active = []
        for row, value, index in zip(active, values.tolist(), indices.tolist()):
            if value <= 0 or index < 0:
                continue
            chosen[row].append(index)
            A = bank.atoms(chosen[row]).double().cpu().numpy()  # [k, d]
            target = targets[row].double().cpu().numpy()
            coef, _ = nnls(A.T, target)
            residuals[row] = torch.as_tensor(target - A.T @ coef, device=bank.device, dtype=torch.float32)
            still_active.append(row)
        active = still_active
    total = targets.pow(2).sum(1)
    captured = 1.0 - residuals.pow(2).sum(1) / total
    return captured.cpu().numpy(), chosen
