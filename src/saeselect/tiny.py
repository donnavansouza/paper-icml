"""Tiny random Gemma 3 stack for CPU tests.

A seeded, randomly initialized model with the real Hugging Face Gemma 3
architecture (the multimodal ``Gemma3ForConditionalGeneration`` wrapper by
default, sliding-window and global layers, (1 + weight) RMS norms, tied
embeddings), plus a random JumpReLU SAE and a random J. Nothing here is ever
used for a claim; it exercises code paths.
"""

from __future__ import annotations

from typing import Any

import torch

from saeselect.lens import JLens
from saeselect.models import ModelHandle, wrap
from saeselect.sae import JumpReLUSAE

DEFAULTS: dict[str, Any] = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 6,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "sliding_window": 8,
    "max_position_embeddings": 512,
    "final_logit_softcapping": None,
    "multimodal": True,
    "seed": 0,
    "dtype": "float32",
    "architecture": "gemma3",  # or "llama" (X2: a tiny Llama, weight-gain RMS norms, untied embeddings)
}


def _build_llama(cfg: dict[str, Any]) -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
                         num_hidden_layers=cfg["num_hidden_layers"], num_attention_heads=cfg["num_attention_heads"],
                         num_key_value_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
                         max_position_embeddings=cfg["max_position_embeddings"], tie_word_embeddings=False,
                         bos_token_id=2, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(cfg["seed"])
    return LlamaForCausalLM(config)


def build_hf_model(**overrides: Any) -> torch.nn.Module:
    from transformers import Gemma3Config, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

    cfg = {**DEFAULTS, **overrides}
    if cfg["architecture"] == "llama":
        model = _build_llama(cfg)
        gen = torch.Generator().manual_seed(cfg["seed"] + 1)
        with torch.no_grad():  # HF initializes RMSNorm weights to one; randomize so the weight gain form is exercised
            for name, param in model.named_parameters():
                if "norm" in name and param.dim() == 1:
                    param.copy_(1.0 + 0.3 * torch.randn(param.shape, generator=gen))
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float64": torch.float64}[cfg["dtype"]]
        return model.to(dtype).eval()
    text = Gemma3TextConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        sliding_window=cfg["sliding_window"],
        max_position_embeddings=cfg["max_position_embeddings"],
        query_pre_attn_scalar=cfg["head_dim"],
        final_logit_softcapping=cfg["final_logit_softcapping"],
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(cfg["seed"])
    if cfg["multimodal"]:
        vision = SiglipVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14
        )
        vocab = cfg["vocab_size"]
        config = Gemma3Config(
            text_config=text.to_dict(),
            vision_config=vision.to_dict(),
            mm_tokens_per_image=4,
            image_token_index=vocab - 1,
            boi_token_index=vocab - 3,
            eoi_token_index=vocab - 2,
        )
        model = Gemma3ForConditionalGeneration(config)
    else:
        model = Gemma3ForCausalLM(text)
    # HF initializes RMSNorm weights to zero, which would hide a wrong gain
    # form; randomize them so (1 + weight) is exercised.
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and param.dim() == 1:
                param.copy_(0.3 * torch.randn(param.shape, generator=gen))
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float64": torch.float64}[cfg["dtype"]]
    return model.to(dtype).eval()


def build_model(model_config: dict[str, Any], *, device: str = "cpu") -> ModelHandle:
    params = {k: v for k, v in model_config.items() if k in DEFAULTS}
    hf_model = build_hf_model(**params).to(device)
    return wrap(hf_model, model_config.get("tokenizer"), info={"repo_id": "tiny-random", "revision": None, "tiny_params": params})


def random_sae(d_model: int, width: int, *, seed: int, layer: int = 0) -> JumpReLUSAE:
    """A JumpReLU SAE whose features fire on a few percent of random residuals."""
    gen = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(width, d_model, generator=gen)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True) * (0.5 + torch.rand(width, 1, generator=gen))
    W_enc = W_dec.T.clone() + 0.05 * torch.randn(d_model, width, generator=gen)
    b_enc = -0.1 * torch.rand(width, generator=gen)
    threshold = 0.05 + 0.1 * torch.rand(width, generator=gen)
    b_dec = 0.01 * torch.randn(d_model, generator=gen)
    return JumpReLUSAE(
        W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, threshold=threshold,
        meta={"source": "tiny-random", "seed": seed, "layer": layer},
    )


def basis_sae(d_model: int, width: int, *, seed: int, layer: int = 0) -> JumpReLUSAE:
    """A JumpReLU SAE that reconstructs any residual exactly: rows 2j and 2j+1
    are +e_j and -e_j with zero bias and a tiny threshold, so each coordinate is
    carried by one of the pair; the remaining rows are random and never fire.
    Lets CPU tests exercise the paths that need a high-fidelity SAE (N13)."""
    if width < 2 * d_model:
        raise ValueError("basis_sae needs width >= 2 * d_model")
    gen = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(width, d_model, generator=gen)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    eye = torch.eye(d_model)
    W_dec[0 : 2 * d_model : 2] = eye
    W_dec[1 : 2 * d_model : 2] = -eye
    threshold = torch.full((width,), 1e6)
    threshold[: 2 * d_model] = 1e-8
    return JumpReLUSAE(W_enc=W_dec.T.clone(), b_enc=torch.zeros(width), W_dec=W_dec, b_dec=torch.zeros(d_model),
                       threshold=threshold, meta={"source": "tiny-basis", "seed": seed, "layer": layer})


def random_lens(d_model: int, layers: list[int], *, seed: int) -> JLens:
    gen = torch.Generator().manual_seed(seed)
    jac = {layer: torch.eye(d_model) + 0.1 * torch.randn(d_model, d_model, generator=gen) for layer in layers}
    return JLens(jacobians=jac, n_prompts=0, d_model=d_model, meta={"source": "tiny-random", "seed": seed})


# A simplified Gemma-style chat template: no system turn (a system message is
# merged into the first user turn), BOS written into the rendered text.
TINY_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% set ns = namespace(sys='') %}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{% set ns.sys = m['content'] + '\n\n' %}"
    "{% elif m['role'] == 'user' %}<start_of_turn>user\n{{ ns.sys }}{{ m['content'] }}<end_of_turn>\n{% set ns.sys = '' %}"
    "{% else %}<start_of_turn>model\n{{ m['content'] }}<end_of_turn>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def build_tokenizer(words: list[str]):
    """A Gemma-like test tokenizer: SentencePiece-style spaces (a word and its
    leading space form one piece "▁word"), whole-word pieces for ``words``,
    and single-character fallback for everything else, so unknown words become
    several tokens. Offsets are supported; the chat template adds one BOS."""
    import string

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    vocab: list[tuple[str, float]] = [(s, 0.0) for s in specials]
    seen = {s for s in specials}

    def add(piece: str, score: float) -> None:
        if piece not in seen:
            seen.add(piece)
            vocab.append((piece, score))

    for word in words:
        add("▁" + word, -1.0)
        add(word, -1.5)
    for ch in string.printable:
        if ch in " \t\r\x0b\x0c":
            continue
        add(ch, -10.0)
        add("▁" + ch, -10.0)
    add("▁", -12.0)
    for ch in "’–—é":
        add(ch, -10.0)
    model = models.Unigram(vocab, unk_id=specials.index("<unk>"), byte_fallback=False)
    tok = Tokenizer(model)
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=True)
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never", split=True)
    # Raw prompts get one BOS from the post-processor, as Gemma's tokenizer does.
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", specials.index("<bos>"))])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>"],
    )
    fast.chat_template = TINY_CHAT_TEMPLATE
    return fast
