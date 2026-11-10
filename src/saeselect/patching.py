"""Clean runs, gradients, and patch evaluation.

``PatchEvaluator`` evaluates many edit sets on one prompt and returns, per set,
the FP32-head margin at the readout position with the intended and realized
perturbation norms. Strategies:

- ``serial`` (default): one full forward per edit set;
- ``batched``: edit sets that share layer and positions run B at a time as
  batch rows of the same prompt (allowed only after check N9 passes);
- ``suffix``: the unchanged prefix (every position before the first edited
  one) is run once with a KV cache, and each edit set recomputes only the
  suffix from a copy of that cache (allowed only after check N8 passes);
- ``batched_suffix``: both at once: the cached prefix repeated over B batch
  rows, the suffix recomputed for B edit sets per forward (allowed only when N8
  and N9 both pass; production runs probe it against serial full forwards).

Every strategy runs inside the precision mode chosen by check N7.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from saeselect.hooks import EditSpec, ResidualEdits, ResidualRecorder
from saeselect.models import ModelHandle, precision_mode
from saeselect.readout import AnswerSets, answer_margin, full_readout

STRATEGIES = ("serial", "batched", "suffix", "batched_suffix")


def as_ids(token_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(list(token_ids), dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def clean_run(handle: ModelHandle, ids: torch.Tensor, layers: Sequence[int], *, mode: str) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Residuals at ``layers`` ([T, d], as computed) and the final-layer residual [T, d]."""
    last = handle.n_layers - 1
    with precision_mode(handle, mode), ResidualRecorder(handle.layers, at=sorted(set(layers) | {last})) as rec:
        handle.forward(ids)
    acts = {layer: rec.activations[layer][0] for layer in layers}
    return acts, rec.activations[last][0]


class activation_checkpointing:
    """Recompute the decoder layers after ``after`` during the backward pass instead
    of storing their activations (torch.utils.checkpoint, non-reentrant, which
    keeps the autocast state). The gradient is mathematically the same; only
    memory changes. Forward hooks still run once, on the original forward."""

    def __init__(self, handle: ModelHandle, *, after: int):
        self.layers = [layer for i, layer in enumerate(handle.layers) if i > after]
        self._originals: list[tuple[torch.nn.Module, Any, bool]] = []

    def __enter__(self) -> "activation_checkpointing":
        from torch.utils.checkpoint import checkpoint

        for layer in self.layers:
            had_own = "forward" in layer.__dict__
            original = layer.forward

            def wrapped(*args, _original=original, **kwargs):
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)

            self._originals.append((layer, original, had_own))
            layer.forward = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        for layer, original, had_own in self._originals:
            if had_own:
                layer.forward = original
            else:
                del layer.forward  # back to the class method
        self._originals = []


def margin_gradient(handle: ModelHandle, ids: torch.Tensor, layer: int, answers: AnswerSets, *, readout: int, mode: str,
                    checkpointed: bool = False, edit: EditSpec | None = None) -> tuple[float, torch.Tensor]:
    """(m, dm/dh_layer at every position [T, d] in FP32) on the unedited prompt: the ATP gradient.
    ``checkpointed`` recomputes later layers in the backward pass to save memory (same gradient). ``edit`` (at
    ``layer``) is applied before the gradient root, so the gradient is taken at the edited residual (path-integrated
    attribution)."""
    import contextlib

    if edit is not None and edit.layer != layer:
        raise ValueError(f"the edit is at layer {edit.layer}, the gradient at layer {layer}")
    last = handle.n_layers - 1
    ckpt = activation_checkpointing(handle, after=layer) if checkpointed else contextlib.nullcontext()
    edits = ResidualEdits(handle.layers, [edit]) if edit is not None else contextlib.nullcontext()  # registered before the root
    with torch.enable_grad(), precision_mode(handle, mode), ckpt, edits, ResidualRecorder(handle.layers, at=[last], grad_at=layer) as rec:
        handle.forward(ids)
        m = answer_margin(rec.activations[last][0, readout], handle.head, answers)
        (grad,) = torch.autograd.grad(m, rec.activations[layer])
    return float(m.detach()), grad[0].float()


@dataclass
class PatchResult:
    margin: float
    intended_norm: float
    realized_norm: float
    extra: dict[str, Any] | None = None  # the optional readout's output, from the same forward pass


class PatchEvaluator:
    """Margins of many edit sets on one prompt, with norms."""

    def __init__(self, handle: ModelHandle, ids: torch.Tensor, answers: AnswerSets, *, readout: int | None = None,
                 mode: str, strategy: str = "serial", batch_size: int = 1, readout_fn: Callable[[torch.Tensor], dict[str, Any]] | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.handle = handle
        self.ids = ids
        self.answers = answers
        self.readout = ids.shape[1] - 1 if readout is None else readout
        self.mode = mode
        self.strategy = strategy
        self.batch_size = batch_size
        self.readout_fn = readout_fn  # called on each set's final residual at the readout
        self._caches: dict[int, Any] = {}

    # -- public -----------------------------------------------------------------

    @torch.no_grad()
    def clean_margin(self) -> float:
        _, h_final = clean_run(self.handle, self.ids, [], mode=self.mode)
        return float(answer_margin(h_final[self.readout], self.handle.head, self.answers))

    @torch.no_grad()
    def clean_readout(self) -> dict[str, float]:
        _, h_final = clean_run(self.handle, self.ids, [], mode=self.mode)
        return {k: float(v) for k, v in full_readout(h_final[self.readout], self.handle.head, self.answers).items()}

    @torch.no_grad()
    def evaluate(self, edit_sets: Sequence[Sequence[EditSpec]]) -> list[PatchResult]:
        if self.strategy == "batched":
            return self._batched(edit_sets)
        if self.strategy == "batched_suffix":
            return self._batched(edit_sets, suffix=True)
        if self.strategy == "suffix":
            return [self._suffix(list(edits)) for edits in edit_sets]
        return [self._serial(list(edits)) for edits in edit_sets]

    # -- strategies -----------------------------------------------------------------

    def _result(self, h_final_row: torch.Tensor, edits: ResidualEdits, row: int = 0) -> PatchResult:
        intended = sum(r.intended_norm[row] ** 2 for r in edits.records if r.calls) ** 0.5
        realized = sum(r.realized_norm[row] ** 2 for r in edits.records if r.calls) ** 0.5
        extra = self.readout_fn(h_final_row) if self.readout_fn is not None else None
        return PatchResult(float(answer_margin(h_final_row, self.handle.head, self.answers)), float(intended), float(realized), extra)

    def _serial(self, edits: list[EditSpec]) -> PatchResult:
        last = self.handle.n_layers - 1
        with precision_mode(self.handle, self.mode), ResidualRecorder(self.handle.layers, at=[last]) as rec, \
                ResidualEdits(self.handle.layers, edits) as applied:
            self.handle.forward(self.ids)
        return self._result(rec.activations[last][0, self.readout], applied)

    def _batched(self, edit_sets: Sequence[Sequence[EditSpec]], *, suffix: bool = False) -> list[PatchResult]:
        results: list[PatchResult] = []
        last = self.handle.n_layers - 1
        for start in range(0, len(edit_sets), self.batch_size):
            chunk = [list(s) for s in edit_sets[start : start + self.batch_size]]
            shape = [(e.layer, tuple(e.positions), e.scale) for e in chunk[0]]
            if any([(e.layer, tuple(e.positions), e.scale) for e in s] != shape or any(e.delta is None for e in s) for s in chunk):
                raise ValueError("batched evaluation needs edit sets with the same layers, positions and scales and static deltas")
            merged = [
                EditSpec(layer=layer, positions=list(pos), scale=scale,
                         delta=torch.stack([s[j].delta.to(torch.float32) for s in chunk]))
                for j, (layer, pos, scale) in enumerate(shape)
            ]
            if suffix:  # the cached prefix, repeated over the batch rows; only the suffix is recomputed
                first = min(min(e.positions) for e in chunk[0])
                if first <= 0:
                    raise ValueError("an edit at position 0 (BOS) is never allowed")
                cache = copy.deepcopy(self._prefix_cache(first))
                cache.batch_repeat_interleave(len(chunk))
                with precision_mode(self.handle, self.mode), ResidualRecorder(self.handle.layers, at=[last]) as rec, \
                        ResidualEdits(self.handle.layers, merged, position_offset=first) as applied:
                    self.handle.text_module(input_ids=self.ids[:, first:].expand(len(chunk), -1), past_key_values=cache, use_cache=True)
                for row in range(len(chunk)):
                    results.append(self._result(rec.activations[last][row, self.readout - first], applied, row))
                continue
            ids = self.ids.expand(len(chunk), -1)
            with precision_mode(self.handle, self.mode), ResidualRecorder(self.handle.layers, at=[last]) as rec, \
                    ResidualEdits(self.handle.layers, merged) as applied:
                self.handle.forward(ids)
            for row in range(len(chunk)):
                results.append(self._result(rec.activations[last][row, self.readout], applied, row))
        return results

    def _prefix_cache(self, start: int) -> Any:
        if start not in self._caches:
            with precision_mode(self.handle, self.mode):
                out = self.handle.text_module(input_ids=self.ids[:, :start], use_cache=True)
            self._caches[start] = out.past_key_values
        return self._caches[start]

    def _suffix(self, edits: list[EditSpec]) -> PatchResult:
        start = min(min(e.positions) for e in edits)
        if start <= 0:
            raise ValueError("an edit at position 0 (BOS) is never allowed")
        cache = copy.deepcopy(self._prefix_cache(start))
        last = self.handle.n_layers - 1
        with precision_mode(self.handle, self.mode), ResidualRecorder(self.handle.layers, at=[last]) as rec, \
                ResidualEdits(self.handle.layers, edits, position_offset=start) as applied:
            self.handle.text_module(input_ids=self.ids[:, start:], past_key_values=cache, use_cache=True)
        return self._result(rec.activations[last][0, self.readout - start], applied)
