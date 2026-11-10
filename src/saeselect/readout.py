"""FP32 output head, answer-set margins, metric gradient, answer mass and ties.

- z are FP32 logits: W_U . FinalNorm(h_f), soft-capped if the config sets it;
- m = logsumexp(z[A+]) - logsumexp(z[A-]) over declared answer sets, never a
  first-token exact match;
- g = dm/dz is the softmax within A+ on A+ minus the softmax within A- on A-;
  ``metric_grad_raw`` includes the soft-cap derivative so it is dm/dz_raw,
  the vector the linearized lens scores need;
- ties are |m| < tau_tie, reported separately and never counted as flips.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Rows of W_U converted to FP32 at a time when full-vocabulary logits are
# needed. A memory knob, not a measured value.
VOCAB_CHUNK = 32768


@dataclass(frozen=True)
class AnswerSets:
    """Disjoint token-id sets for the favorable (A+) and unfavorable (A-) answers."""

    plus: tuple[int, ...]
    minus: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.plus or not self.minus:
            raise ValueError("answer sets must both be non-empty")
        if len(set(self.plus)) != len(self.plus) or len(set(self.minus)) != len(self.minus):
            raise ValueError("answer sets must not repeat token ids")
        if set(self.plus) & set(self.minus):
            raise ValueError(f"answer sets overlap: {sorted(set(self.plus) & set(self.minus))}")

    @property
    def ids(self) -> list[int]:
        return [*self.plus, *self.minus]

    def swapped(self) -> "AnswerSets":
        return AnswerSets(plus=self.minus, minus=self.plus)


@dataclass
class HeadSpec:
    """Everything the FP32 head needs, read from the loaded model.

    ``gamma`` is the effective final-norm gain in FP32 (for Gemma RMSNorm,
    1 + weight; ``gain_form`` records which form was confirmed numerically).
    ``W_U`` is the LM head weight in the model dtype, shape [vocab, d_model].
    """

    gamma: torch.Tensor
    eps: float
    W_U: torch.Tensor
    softcap: float | None
    gain_form: str

    @property
    def d_model(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.W_U.shape[0])


def final_norm_fp32(h: torch.Tensor, spec: HeadSpec) -> torch.Tensor:
    """RMS final norm computed entirely in FP32 (no cast back to the model dtype)."""
    x = h.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + spec.eps)
    return x * spec.gamma.to(device=x.device, dtype=torch.float32)


def _softcap(z: torch.Tensor, softcap: float | None) -> torch.Tensor:
    return z if softcap is None else softcap * torch.tanh(z / softcap)


def raw_logits_fp32(h_final: torch.Tensor, spec: HeadSpec, token_ids: list[int] | torch.Tensor | None = None) -> torch.Tensor:
    """FP32 logits before soft-capping, for all tokens or only ``token_ids``."""
    normed = final_norm_fp32(h_final, spec)
    W = spec.W_U
    if token_ids is not None:
        idx = torch.as_tensor(token_ids, device=W.device, dtype=torch.long)
        return normed @ W.index_select(0, idx).to(device=normed.device, dtype=torch.float32).T
    chunks = []
    for start in range(0, W.shape[0], VOCAB_CHUNK):
        rows = W[start : start + VOCAB_CHUNK].to(device=normed.device, dtype=torch.float32)
        chunks.append(normed @ rows.T)
    return torch.cat(chunks, dim=-1)


def logits_fp32(h_final: torch.Tensor, spec: HeadSpec, token_ids: list[int] | torch.Tensor | None = None) -> torch.Tensor:
    """FP32 logits z, soft-capped if configured."""
    return _softcap(raw_logits_fp32(h_final, spec, token_ids), spec.softcap)


def margin_from_logits(z_plus: torch.Tensor, z_minus: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(z_plus, dim=-1) - torch.logsumexp(z_minus, dim=-1)


def answer_margin(h_final: torch.Tensor, spec: HeadSpec, answers: AnswerSets) -> torch.Tensor:
    """m for residual(s) ``h_final`` [..., d]. Differentiable; uses only the answer rows of W_U."""
    z = logits_fp32(h_final, spec, answers.ids)
    n_plus = len(answers.plus)
    return margin_from_logits(z[..., :n_plus], z[..., n_plus:])


def full_readout(h_final: torch.Tensor, spec: HeadSpec, answers: AnswerSets) -> dict[str, torch.Tensor]:
    """Margin, answer mass and the favorable probability from full-vocabulary FP32 logits."""
    z = logits_fp32(h_final, spec)
    ids = torch.as_tensor(answers.ids, device=z.device)
    n_plus = len(answers.plus)
    z_ans = z.index_select(-1, ids)
    log_probs = torch.log_softmax(z, dim=-1)
    mass = log_probs.index_select(-1, ids).exp().sum(-1)
    margin = margin_from_logits(z_ans[..., :n_plus], z_ans[..., n_plus:])
    return {"margin": margin, "answer_mass": mass, "p_fav": torch.sigmoid(margin), "top_token": z.argmax(-1)}


def next_token_logprobs(h_final: torch.Tensor, spec: HeadSpec) -> torch.Tensor:
    """Full-vocabulary log-softmax of the FP32 head at one position, in float64 for the sums that follow."""
    return torch.log_softmax(logits_fp32(h_final, spec).double(), dim=-1)


def off_target(lp: torch.Tensor, lp_clean: torch.Tensor, lp_full: torch.Tensor, answer_ids: list[int]) -> dict[str, float]:
    """The off-target quantities of an edited next-token distribution (log-probs ``lp``), in nats, unsquared:
    O1 = KL(P~_clean || P~_S), with P~ renormalized over the tokens outside both answer sets (the off-answer change);
    O2 = AM(clean) - AM(S), AM the probability of both answer sets (answer mass lost);
    O3 = KL(P_FULL || P_S) over the full vocabulary (distance to the full transfer)."""
    ids = torch.as_tensor(list(answer_ids), device=lp.device)
    keep = torch.ones(lp.shape[-1], dtype=torch.bool, device=lp.device)
    keep[ids] = False

    def off(x: torch.Tensor) -> torch.Tensor:
        y = x[keep]
        return y - torch.logsumexp(y, 0)

    oc, os_ = off(lp_clean.to(lp.device)), off(lp)
    o1 = (oc.exp() * (oc - os_)).sum()
    o2 = lp_clean.to(lp.device)[ids].logsumexp(0).exp() - lp[ids].logsumexp(0).exp()
    lf = lp_full.to(lp.device)
    o3 = (lf.exp() * (lf - lp)).sum()
    return {"O1": float(o1), "O2": float(o2), "O3": float(o3)}


def metric_grad_logits(z_plus: torch.Tensor, z_minus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """g = dm/dz on the answer tokens: (softmax within A+, -softmax within A-)."""
    return torch.softmax(z_plus, dim=-1), -torch.softmax(z_minus, dim=-1)


def metric_grad_raw(h_final: torch.Tensor, spec: HeadSpec, answers: AnswerSets) -> torch.Tensor:
    """dm/dz_raw on the answer tokens (A+ then A-), including the soft-cap derivative."""
    z_raw = raw_logits_fp32(h_final, spec, answers.ids)
    z = _softcap(z_raw, spec.softcap)
    n_plus = len(answers.plus)
    g_plus, g_minus = metric_grad_logits(z[..., :n_plus], z[..., n_plus:])
    g = torch.cat([g_plus, g_minus], dim=-1)
    if spec.softcap is not None:
        g = g * (1 - torch.tanh(z_raw / spec.softcap) ** 2)
    return g


def output_direction(h_final: torch.Tensor, spec: HeadSpec, answers: AnswerSets) -> torch.Tensor:
    """w = W_U^T g_raw (FP32, [d_model]): the residual-space direction that moves m
    through the unembedding, before the final-norm Jacobian is applied."""
    if h_final.dim() != 1:
        raise ValueError("output_direction takes one residual vector [d_model]")
    g = metric_grad_raw(h_final, spec, answers)
    idx = torch.as_tensor(answers.ids, device=spec.W_U.device, dtype=torch.long)
    rows = spec.W_U.index_select(0, idx).to(device=g.device, dtype=torch.float32)
    return g @ rows


def vocabulary_direction(spec: HeadSpec, plus_ids: list[int], minus_ids: list[int], device: torch.device | str) -> torch.Tensor:
    """w_q = W_U^T q for the verbalization target q (+1/|T+| on T+, -1/|T-| on T-)."""
    W = spec.W_U
    plus = W.index_select(0, torch.as_tensor(plus_ids, device=W.device)).float().mean(0)
    minus = W.index_select(0, torch.as_tensor(minus_ids, device=W.device)).float().mean(0)
    return (plus - minus).to(device)


def margin_bf16_head(h_final: torch.Tensor, spec: HeadSpec, answers: AnswerSets) -> torch.Tensor:
    """m as a model-dtype head computes it (norm output cast to the model dtype,
    matmul in the model dtype over the full vocabulary). Used only to measure
    how far the BF16 head is from the FP32 head (check N6, tau_tie)."""
    dtype = spec.W_U.dtype
    normed = final_norm_fp32(h_final, spec).to(dtype)
    z = (normed.to(spec.W_U.device) @ spec.W_U.T).float()
    z = _softcap(z, spec.softcap)
    ids = torch.as_tensor(answers.ids, device=z.device)
    z_ans = z.index_select(-1, ids)
    n_plus = len(answers.plus)
    return margin_from_logits(z_ans[..., :n_plus], z_ans[..., n_plus:])


def is_tie(margin: float, tau_tie: float) -> bool:
    """A decision with |m| below the derived tolerance is a tie, never a flip."""
    return abs(float(margin)) < tau_tie


def decision_flipped(margin_before: float, margin_after: float, tau_tie: float) -> bool | None:
    """True/False for a real flip or not; None when either side is a tie."""
    if is_tie(margin_before, tau_tie) or is_tie(margin_after, tau_tie):
        return None
    return (margin_before > 0) != (margin_after > 0)
