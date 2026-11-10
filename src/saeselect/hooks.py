"""Read and write hooks at decoder-layer outputs (resid_post).

Layer index l means the output of decoder block l, which is where Gemma Scope
2 ``layer_l`` SAEs and ``jlens`` layer l read.

Write hooks add a delta at a set of positions P (never BOS, position 0). The
delta is FP32; the edited residual is ``(h.float() + delta).to(h.dtype)``, a
single rounding. Every edit records its intended norm (the FP32 delta) and its
realized norm (the difference actually present after the cast), each summed in
quadrature over P (section 5.3). A zero delta leaves the
output bit-identical (check N4).

All hooks are context managers that remove themselves on exit, including when
the forward pass raises (check N5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn

DeltaFn = Callable[[torch.Tensor], torch.Tensor]


def validate_positions(positions: Iterable[int], seq_len: int) -> list[int]:
    """Sorted unique positions; refuses BOS (0), negatives and out-of-range."""
    pos = sorted(set(int(p) for p in positions))
    if not pos:
        raise ValueError("an intervention site needs at least one position")
    if pos[0] <= 0:
        raise ValueError("position 0 is BOS and is never an intervention site; negative positions are not allowed")
    if pos[-1] >= seq_len:
        raise ValueError(f"position {pos[-1]} out of range for sequence length {seq_len}")
    return pos


def _split(output: Any) -> tuple[torch.Tensor, tuple | None]:
    if torch.is_tensor(output):
        return output, None
    return output[0], tuple(output[1:])


def _join(tensor: torch.Tensor, rest: tuple | None, original: Any) -> Any:
    if rest is None:
        return tensor
    return type(original)((tensor, *rest)) if isinstance(original, tuple) else (tensor, *rest)


def hook_count(model: nn.Module) -> int:
    """Number of forward and forward-pre hooks registered anywhere in ``model``."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


class ResidualRecorder:
    """Capture layer outputs on the next forward pass.

    ``grad_at`` marks one layer's output as requiring grad before later layers
    see it, so ``torch.autograd.grad(metric, recorder.activations[grad_at])``
    gives d metric / d h_l at every position (used by ATP). Captures at or
    after the grad root keep their graph; earlier captures are detached.
    """

    def __init__(self, layers: Sequence[nn.Module], at: Iterable[int], *, grad_at: int | None = None):
        self._layers = layers
        self._at = sorted(set(at) | ({grad_at} if grad_at is not None else set()))
        self._grad_at = grad_at
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, index: int):
        def hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            tensor, rest = _split(output)
            if index == self._grad_at:
                tensor = tensor.detach().requires_grad_(True)
                self.activations[index] = tensor
                return _join(tensor, rest, output)
            keep_graph = self._grad_at is not None and index > self._grad_at
            self.activations[index] = tensor if keep_graph else tensor.detach()
            return None

        return hook

    def __enter__(self) -> "ResidualRecorder":
        try:
            for index in self._at:
                self._handles.append(self._layers[index].register_forward_hook(self._hook(index)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


@dataclass
class EditRecord:
    layer: int
    positions: list[int]
    intended_norm: list[float] = field(default_factory=list)  # one per batch row
    realized_norm: list[float] = field(default_factory=list)
    calls: int = 0


@dataclass
class EditSpec:
    """One delta-form edit: add ``delta`` (or ``delta_fn(h_sel)``) at ``positions`` of ``layer``.

    ``delta`` has shape [|P|, d] (applied to every batch row) or [B, |P|, d].
    ``delta_fn`` receives the current FP32 residual at P, shape [B, |P|, d],
    and returns the delta; use it when the edit depends on the residual it is
    applied to (ACE, DIM-ABL, and any edit stacked after an upstream edit).
    """

    layer: int
    positions: Sequence[int]
    delta: torch.Tensor | None = None
    delta_fn: DeltaFn | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if (self.delta is None) == (self.delta_fn is None):
            raise ValueError("give exactly one of delta and delta_fn")


class ResidualEdits:
    """Apply one or more :class:`EditSpec` during forward passes inside the block.

    ``position_offset`` is the absolute position of the first token of the
    sequence being run (non-zero when only a suffix is recomputed from a cached
    prefix). Edit positions are absolute; BOS is always absolute position 0.
    """

    def __init__(self, layers: Sequence[nn.Module], edits: Sequence[EditSpec], *, position_offset: int = 0):
        self._layers = layers
        self._edits = list(edits)
        self._offset = position_offset
        self.records: list[EditRecord] = [EditRecord(layer=e.layer, positions=list(e.positions)) for e in self._edits]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, edit: EditSpec, record: EditRecord):
        def hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            h, rest = _split(output)
            positions = validate_positions(edit.positions, self._offset + h.shape[1])
            if positions[0] < self._offset:
                raise ValueError(f"edit position {positions[0]} lies in the cached prefix (suffix starts at {self._offset})")
            index = torch.as_tensor([p - self._offset for p in positions], device=h.device)
            # FP32 for BF16/FP16/FP32 residuals; never downcast a float64 test model.
            work = torch.promote_types(h.dtype, torch.float32)
            selected = h.index_select(1, index).to(work)  # [B, |P|, d]
            if edit.delta_fn is not None:
                delta = edit.delta_fn(selected).to(work)
            else:
                delta = edit.delta.to(device=h.device, dtype=work)
                if delta.dim() == 2:
                    delta = delta.unsqueeze(0).expand_as(selected)
            if delta.shape != selected.shape:
                raise ValueError(f"delta shape {tuple(delta.shape)} does not match site {tuple(selected.shape)}")
            if edit.scale != 1.0:
                delta = delta * edit.scale
            # Where the delta is exactly zero keep the original value, so a sham
            # edit is bit-identical even for -0.0 entries (-0.0 + 0.0 == +0.0).
            edited = torch.where(delta == 0, selected, selected + delta).to(h.dtype)
            new_h = h.clone()
            new_h.index_copy_(1, index, edited)
            realized = edited.to(work) - selected
            record.intended_norm = delta.pow(2).sum(dim=(1, 2)).sqrt().tolist()
            record.realized_norm = realized.pow(2).sum(dim=(1, 2)).sqrt().tolist()
            record.calls += 1
            return _join(new_h, rest, output)

        return hook

    def __enter__(self) -> "ResidualEdits":
        try:
            for edit, record in zip(self._edits, self.records):
                self._handles.append(self._layers[edit.layer].register_forward_hook(self._hook(edit, record)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
