"""Tokenizer loading and the prompt-encoding rules every stage shares.

- Raw prompts are encoded with the tokenizer's special tokens (one BOS).
- Chat prompts are rendered with the model's own template, which already
  starts with the BOS string, then encoded WITHOUT special tokens; encoding a
  rendered template with special tokens adds a second BOS.
  :func:`encode_prompt` asserts exactly one BOS at position 0 either way (N11).
- Character spans (concept words, names) map to token positions through the
  fast tokenizer's offset mapping.
- An answer string's scored token is its first token (margins are read at one
  position); :func:`answer_token_ids` returns those ids, deduplicated.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from saeselect.provenance import EngineeringFailed


def load_tokenizer(model_config: dict[str, Any]):
    """The tokenizer of a configs/models/*.yaml model at its pinned revision."""
    from transformers import AutoTokenizer

    source = model_config.get("source")
    if source == "tiny-random":
        from saeselect import tiny

        return tiny.build_tokenizer(model_config.get("tokenizer_words", []))
    if model_config.get("tokenizer_path"):  # local copy for offline dry runs; recorded in the card
        return AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
    revision = model_config["revision"]
    if len(str(revision)) != 40:
        raise EngineeringFailed("tokenizer_revision", f"revision must be a pinned 40-character SHA, got {revision!r}")
    return AutoTokenizer.from_pretrained(model_config["repo_id"], revision=revision)


def tokenizer_fingerprint(tokenizer) -> dict[str, Any]:
    """Hashes that identify a tokenizer's behaviour: vocabulary, specials, chat template."""
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    vocab_hash = hashlib.sha256(json.dumps(vocab, ensure_ascii=False).encode("utf-8")).hexdigest()
    template = getattr(tokenizer, "chat_template", None) or ""
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "vocab_sha256": vocab_hash,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "name_or_path": getattr(tokenizer, "name_or_path", None),
    }


def render_chat(tokenizer, messages: list[dict[str, str]], prefill: str = "") -> str:
    """The model's chat template with a generation prompt, then the assistant prefill."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + prefill


def render_prompt(tokenizer, messages: list[dict[str, str]], prefill: str = "", *, style: str = "chat",
                  system_user_separator: str = "\n\n", prefill_separator: str = "\n\n") -> str:
    """``chat``: :func:`render_chat`. ``raw`` (for checkpoints without a chat
    template): the same texts in the same order as Gemma's chat rendering (system
    text, then the user text, then the prefill) joined by the declared separators,
    with no control tokens and no BOS (``encode_prompt(..., "raw")`` adds it)."""
    if style == "chat":
        return render_chat(tokenizer, messages, prefill)
    if style != "raw":
        raise ValueError(f"unknown prompt style {style!r}")
    system = [m["content"] for m in messages if m["role"] == "system"]
    user = [m["content"] for m in messages if m["role"] == "user"]
    if len(user) != 1 or len(system) > 1 or len(system) + len(user) != len(messages):
        raise ValueError("raw rendering takes at most one system and exactly one user message")
    text = (system[0] + system_user_separator if system else "") + user[0]
    return text + (prefill_separator + prefill if prefill else "")


def encode_prompt(tokenizer, text: str, prompt_format: str, *, offsets: bool = False):
    """Token ids (and optionally offsets) with exactly one BOS, at position 0."""
    if prompt_format == "raw":
        add_special = True
    elif prompt_format == "chat":
        add_special = False
    else:
        raise ValueError(f"unknown prompt format {prompt_format!r}")
    encoded = tokenizer(text, add_special_tokens=add_special, return_offsets_mapping=offsets)
    ids = list(encoded["input_ids"])
    bos = tokenizer.bos_token_id
    n_bos = sum(1 for i in ids if i == bos)
    if n_bos != 1 or ids[0] != bos:
        raise EngineeringFailed(
            "N11_bos", f"{prompt_format} prompt encodes with {n_bos} BOS tokens (first id {ids[0]}); expected exactly one at position 0"
        )
    if offsets:
        return ids, [tuple(o) for o in encoded["offset_mapping"]]
    return ids


def token_positions(offsets: Sequence[tuple[int, int]], span: Sequence[int]) -> list[int]:
    """Token positions whose character range overlaps ``span`` = [start, end)."""
    start, end = span
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def piece_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def first_token_id(tokenizer, text: str) -> int:
    ids = piece_ids(tokenizer, text)
    if not ids:
        raise ValueError(f"{text!r} encodes to no tokens")
    return ids[0]


def token_count(tokenizer, text: str) -> int:
    return len(piece_ids(tokenizer, text))


def answer_token_ids(tokenizer, candidates: Sequence[str]) -> tuple[list[int], list[str]]:
    """First-token ids of the candidate surface forms, deduplicated in order,
    with the first candidate string that produced each id."""
    ids: list[int] = []
    strings: list[str] = []
    for candidate in candidates:
        tid = first_token_id(tokenizer, candidate)
        if tid not in ids:
            ids.append(tid)
            strings.append(candidate)
    return ids, strings
