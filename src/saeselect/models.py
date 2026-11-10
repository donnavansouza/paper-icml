"""Model loading with pinned revisions, module discovery and the head spec.

Gemma 3 12B and 4B load as multimodal conditional-generation models; the text
decoder sits at ``model.language_model``. Discovery follows the same layout
table as ``jlens.hf`` so both agree on which modules are "the layers", and the
path found is recorded.

The final-norm gain form is not assumed: :func:`final_norm_spec` evaluates the
module on random FP32 input and confirms whether the gain is ``1 + weight``
(Gemma) or ``weight`` (Llama), failing loudly if neither matches.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from saeselect.provenance import EngineeringFailed
from saeselect.readout import HeadSpec


@dataclass(frozen=True)
class DecoderLayout:
    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


# Same order as jlens.hf._LAYOUTS for the families this project uses.
LAYOUTS: tuple[DecoderLayout, ...] = (
    DecoderLayout("model"),
    DecoderLayout("model.language_model"),
    DecoderLayout("language_model"),
)


def _resolve(obj: Any, dotted: str) -> Any:
    return functools.reduce(getattr, dotted.split("."), obj)


def find_layout(hf_model: nn.Module) -> DecoderLayout:
    for layout in LAYOUTS:
        try:
            candidate = _resolve(hf_model, layout.path)
        except AttributeError:
            continue
        if all(hasattr(candidate, a) for a in (layout.layers, layout.norm, layout.embed)) and hasattr(
            hf_model, layout.lm_head
        ):
            return layout
    raise EngineeringFailed("module_discovery", f"no known decoder layout inside {type(hf_model).__name__}")


def final_norm_spec(norm: nn.Module, d_model: int, *, seed: int = 0) -> tuple[torch.Tensor, float, str]:
    """Return (gamma FP32, eps, gain_form) confirmed against the module itself."""
    weight = getattr(norm, "weight", None)
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
    if weight is None or eps is None:
        raise EngineeringFailed("final_norm", f"{type(norm).__name__} has no weight/eps; not an RMS norm")
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(4, d_model, generator=gen) * 3.0).to(weight.device)
    with torch.no_grad():
        reference = norm(x.float()).float()
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        candidates = {"1+weight": 1.0 + weight.float(), "weight": weight.float()}
        errors = {form: (normalized * g - reference).abs().max().item() for form, g in candidates.items()}
    best = min(errors, key=errors.get)
    scale = reference.abs().max().item()
    if errors[best] > 1e-5 * max(scale, 1.0):
        raise EngineeringFailed("final_norm", f"final norm matches neither gain form: errors {errors}")
    return candidates[best].detach().clone(), float(eps), best


@dataclass
class ModelHandle:
    """A loaded model with the modules and constants every stage needs."""

    hf_model: nn.Module
    tokenizer: Any
    layout: DecoderLayout
    text_module: nn.Module
    layers: nn.ModuleList
    final_norm: nn.Module
    embed: nn.Module
    lm_head: nn.Module
    n_layers: int
    d_model: int
    head: HeadSpec
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Residual stack only (the final norm and HF head are not used for margins)."""
        return self.text_module(input_ids=input_ids.to(self.device), use_cache=False)


def wrap(hf_model: nn.Module, tokenizer: Any, *, info: dict[str, Any] | None = None) -> ModelHandle:
    """Wrap a loaded HF model. Puts it in eval mode and freezes its parameters."""
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad_(False)
    layout = find_layout(hf_model)
    text_module = _resolve(hf_model, layout.path)
    layers = getattr(text_module, layout.layers)
    final_norm = getattr(text_module, layout.norm)
    embed = getattr(text_module, layout.embed)
    lm_head = getattr(hf_model, layout.lm_head)
    text_config = hf_model.config.get_text_config()
    n_layers = int(text_config.num_hidden_layers)
    d_model = int(text_config.hidden_size)
    if len(layers) != n_layers:
        raise EngineeringFailed("module_discovery", f"config says {n_layers} layers, found {len(layers)}")
    gamma, eps, gain_form = final_norm_spec(final_norm, d_model)
    softcap = getattr(text_config, "final_logit_softcapping", None)
    head = HeadSpec(gamma=gamma, eps=eps, W_U=lm_head.weight, softcap=softcap, gain_form=gain_form)
    tied = lm_head.weight.data_ptr() == embed.weight.data_ptr()
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    details = {
        "model_class": type(hf_model).__name__,
        "decoder_path": f"{layout.path}.{layout.layers}",
        "final_norm_class": type(final_norm).__name__,
        "final_norm_gain_form": gain_form,
        "final_norm_eps": eps,
        "final_logit_softcapping": softcap,
        "num_hidden_layers": n_layers,
        "hidden_size": d_model,
        "vocab_size": int(lm_head.weight.shape[0]),
        "tied_embeddings": tied,
        "embed_scale": _embed_scale(embed),
        "sliding_window": getattr(text_config, "sliding_window", None),
        "layer_types": layer_types,
        "global_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
        "dtype": str(lm_head.weight.dtype),
    }
    details.update(info or {})
    return ModelHandle(
        hf_model=hf_model,
        tokenizer=tokenizer,
        layout=layout,
        text_module=text_module,
        layers=layers,
        final_norm=final_norm,
        embed=embed,
        lm_head=lm_head,
        n_layers=n_layers,
        d_model=d_model,
        head=head,
        info=details,
    )


def _embed_scale(embed: nn.Module) -> float | None:
    scale = getattr(embed, "embed_scale", None)
    if scale is None:
        return None
    return float(scale) if not torch.is_tensor(scale) else float(scale.item())


_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def load_model(model_config: dict[str, Any], *, device: str, dtype: str | None = None) -> ModelHandle:
    """Load a model from a configs/models/*.yaml mapping.

    ``source: hf`` loads ``repo_id`` at the pinned ``revision`` (a full commit
    SHA; refused otherwise) straight onto ``device``. ``dtype`` overrides the
    config's dtype (the 4B full-FP32 precision reference, check N7).
    ``source: tiny-random`` builds the seeded tiny Gemma 3 used by the CPU tests.
    """
    source = model_config.get("source")
    if source == "tiny-random":
        from saeselect import tiny

        cfg = dict(model_config)
        if dtype:
            cfg["dtype"] = dtype
        handle = tiny.build_model(cfg, device=device)
        if cfg.get("tokenizer_words") is not None:
            handle.tokenizer = tiny.build_tokenizer(cfg["tokenizer_words"])
        return handle
    if source != "hf":
        raise ValueError(f"unknown model source {source!r}")
    # For model_type gemma3, AutoModelForCausalLM resolves to the multimodal
    # Gemma3ForConditionalGeneration (checked on Transformers 5.16.1); the class
    # actually loaded is recorded in info["model_class"].
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_id = model_config["repo_id"]
    revision = model_config.get("revision")
    if not revision or len(str(revision)) != 40:
        raise EngineeringFailed("model_revision", f"{repo_id}: revision must be a pinned 40-character commit SHA, got {revision!r}")
    torch_dtype = _DTYPES[dtype or model_config["dtype"]]
    # device_map places weights directly on the device (no full copy in CPU RAM).
    hf_model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=torch_dtype, device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    resolved = getattr(hf_model.config, "_commit_hash", None)
    if resolved is not None and resolved != revision:
        raise EngineeringFailed("model_revision", f"{repo_id}: loaded commit {resolved} differs from pinned {revision}")
    return wrap(hf_model, tokenizer, info={"repo_id": repo_id, "revision": revision, "loaded_commit": resolved})


# --------------------------------------------------------------------------
# Precision modes
# --------------------------------------------------------------------------

PRECISION_MODES = ("bf16", "fp32_residual", "fp32_full", "fp32_grad")


class precision_mode:
    """Context manager for the precision of forward passes.

    ``bf16``: the model as loaded (no-op).

    ``fp32_residual``: the residual stream is held in FP32 (the embedding
    output is cast to FP32, and every residual addition then happens in FP32
    because FP32 + BF16 promotes), while sublayer matmuls run in BF16 under
    autocast, so small deltas added to the residual are not rounded away (F7).

    ``fp32_full``: every computation in FP32, for any model size: each decoder
    layer's weights are upcast just before its forward and the original tensors
    restored right after (a BF16 -> FP32 -> BF16 round trip is exact), so only
    one layer is ever held in FP32. No-grad forwards only: a backward pass would
    need the FP32 weights after they are restored. Used as the precise
    reference for numerical checks (N10 finite differences, N7 descriptive).

    ``fp32_grad``: every computation in FP32 with
    gradients allowed: the decoder layers' and final norm's weights are
    converted to FP32 in place for the whole context and converted back on
    exit (a BF16 -> FP32 -> BF16 round trip is exact). The model then holds
    FP32 weights only (32 GB for 8B), so a model config declares it where N10
    needs it.
    """

    def __init__(self, handle: ModelHandle, mode: str):
        if mode not in PRECISION_MODES:
            raise ValueError(f"unknown precision mode {mode!r}")
        self.handle = handle
        self.mode = mode
        self._hooks: list[Any] = []
        self._autocast = None

    def __enter__(self) -> "precision_mode":
        if self.mode in ("fp32_residual", "fp32_full"):
            self._hooks.append(self.handle.embed.register_forward_hook(lambda module, inputs, output: output.float()))
        if self.mode == "fp32_residual":
            self._autocast = torch.autocast(device_type=self.handle.device.type, dtype=torch.bfloat16)
            self._autocast.__enter__()
        elif self.mode == "fp32_full" and self.handle.dtype != torch.float32:
            if torch.is_grad_enabled():
                raise ValueError("fp32_full mode supports no-grad forwards only (weights are restored after each layer)")
            for layer in self.handle.layers:
                self._hooks.append(layer.register_forward_pre_hook(_upcast_weights))
                self._hooks.append(layer.register_forward_hook(_restore_weights))
        elif self.mode == "fp32_grad":
            self._converted = []
            for module in [*self.handle.layers, self.handle.final_norm]:
                for param in module.parameters():
                    if param.dtype != torch.float32:
                        self._converted.append((param, param.dtype))
                        param.data = param.data.float()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._autocast is not None:
            self._autocast.__exit__(*exc)
            self._autocast = None
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for param, dtype in getattr(self, "_converted", []):
            param.data = param.data.to(dtype)
        self._converted = []
        for layer in self.handle.layers:  # restore even if a forward raised inside a layer
            _restore_weights(layer, None, None)


def _upcast_weights(module: torch.nn.Module, inputs: Any) -> None:
    originals = {}
    for name, param in module.named_parameters():
        if param.dtype != torch.float32:
            originals[name] = param.data
            param.data = param.data.float()
    module._jsel_originals = originals


def _restore_weights(module: torch.nn.Module, inputs: Any, output: Any) -> None:
    originals = getattr(module, "_jsel_originals", None)
    if originals:
        for name, param in module.named_parameters():
            if name in originals:
                param.data = originals[name]
    module._jsel_originals = None
