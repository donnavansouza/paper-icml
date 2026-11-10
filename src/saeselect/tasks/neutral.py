"""Neutral corpus: 200 windows of 128 tokens (BOS + 127 text tokens, the lens
fit's max_seq_len) from the wikitext-103 validation split, the corpus family of
the lens fit (which used the train split). Each article's text is tokenized,
cut into non-overlapping 127-token windows, and 200 windows are chosen by
seeded hash. Used for the SAE fidelity check and feature statistics (sd_i).
"""

from __future__ import annotations

import json
from typing import Any

from saeselect.tasks.records import hash_unit


def wikitext_articles(lines: list[str]) -> list[tuple[str, str]]:
    """(title, text) per article: a top-level heading is ' = Title = '."""
    articles: list[tuple[str, list[str]]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= ="):
            articles.append((stripped.strip("= ").strip(), []))
        elif articles and stripped:
            articles[-1][1].append(line)
    return [(title, "".join(body)) for title, body in articles]


def build_neutral(lines: list[str], *, tokenizer, n_windows: int, window_tokens: int, seed: int) -> dict[str, Any]:
    text_tokens = window_tokens - 1  # BOS is prepended
    candidates = []
    for title, text in wikitext_articles(lines):
        ids = tokenizer.encode(text, add_special_tokens=False)
        for k in range(len(ids) // text_tokens):
            candidates.append((title, k, ids[k * text_tokens : (k + 1) * text_tokens]))
    if len(candidates) < n_windows:
        raise ValueError(f"only {len(candidates)} windows available, {n_windows} required")
    chosen = sorted(candidates, key=lambda c: (hash_unit(f"neutral:{c[0]}:{c[1]}", seed), c[0], c[1]))[:n_windows]
    bos = tokenizer.bos_token_id
    windows = [
        {"id": f"neutral:{title}:{k}", "article": title, "window": k, "token_ids": [bos, *ids], "text": tokenizer.decode(ids)}
        for title, k, ids in sorted(chosen, key=lambda c: (c[0], c[1]))
    ]
    card = {
        "task": "neutral",
        "counts": {"articles": len({c[0] for c in candidates}), "candidate_windows": len(candidates), "windows": len(windows),
                   "articles_used": len({w["article"] for w in windows})},
        "window_tokens": window_tokens,
    }
    return {"items": windows, "card": card}


def write_items(path, items: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
