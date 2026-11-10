"""``python -m saeselect raw-completions``.

Prints greedy completions for dev records only: ``n_per_group`` per task,
template and prompt format (label order x polarity for decisions), so answer
sets and formats are read by a person before any scoring rule is frozen. Each
completion is shown with the FP32-head margin, the answer mass and whether the
first generated token is in A+, A- or neither (a first-token look is shown
here only to help the reading; scoring is always log-sum-exp over answer sets).

Writes ``raw_completions.md`` (for reading) and ``raw_completions.jsonl`` (all fields).

A group may list ``prefills`` (B2′): each chosen prompt is then shown
once per candidate prefill appended to the assistant turn, with the margin read
at the prefill's last token, so a prefill is chosen before it is built in.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from saeselect.models import load_model
from saeselect.patching import as_ids, clean_run
from saeselect.readout import AnswerSets, full_readout
from saeselect.runner import Outcome, RunContext
from saeselect.tasks.records import hash_unit, load_records
from saeselect.tasks.tokenization import encode_prompt


def _groups(cfg: dict[str, Any], seed: int) -> list[tuple[str, str, list]]:
    data_dir = Path(cfg["data_dir"])
    specs = []
    rel = cfg["groups"].get("relations")  # optional: a checkpoint read only on decision benchmarks has none
    for family in (rel or {}).get("families", []):
        specs.append((family, rel))
    for family, spec in cfg["groups"].items():  # every other key names a task family directory (raw variants)
        if family != "relations":
            specs.append((family, spec))
    out = []
    for family, spec in specs:
        card = json.loads((data_dir / family / "benchmark_card.json").read_text(encoding="utf-8"))
        records = load_records(data_dir / family / card["records_file"], phase="dev", expected_sha256=card["records_sha256"])
        grouped: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.role not in spec["roles"]:
                continue
            if any(r.meta.get(k) not in allowed for k, allowed in (spec.get("meta") or {}).items()):
                continue  # e.g. B2′: only the 10% level
            key = " / ".join(str(getattr(r, field)) for field in spec["by"])
            grouped.setdefault(key, {}).setdefault(r.prompt, r)  # one record per distinct prompt
        for key in sorted(grouped):
            chosen = sorted(grouped[key].values(), key=lambda r: (hash_unit(f"raw:{family}:{r.id}", seed), r.id))[: cfg["n_per_group"]]
            if not spec.get("prefills"):
                out.append((family, key, chosen))
                continue
            for prefill in spec["prefills"]:  # B2′: candidate prefills probed on the same prompts before any build
                probed = [replace(r, id=f"{r.id}|prefill={prefill}", prompt=r.prompt + prefill,
                                  meta={**r.meta, "readout_position": -1, "probe_prefill": prefill}) for r in chosen]
                out.append((family, f"{key} / prefill {prefill!r}", probed))
    return out


def complete(handle, record, *, max_new_tokens: int) -> dict[str, Any]:
    """FP32-head readout and a greedy completion of one record's prompt (BF16 model)."""
    tok = handle.tokenizer
    ids = as_ids(encode_prompt(tok, record.prompt, record.prompt_format), handle.device)
    answers = AnswerSets(plus=tuple(record.answer_sets["A_plus"]["ids"]), minus=tuple(record.answer_sets["A_minus"]["ids"]))
    _, h_final = clean_run(handle, ids, [], mode="bf16")
    read = {k: float(v) for k, v in full_readout(h_final[record.meta["readout_position"]], handle.head, answers).items()}
    with torch.no_grad():
        generated = handle.hf_model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                             do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = generated[0, ids.shape[1]:].tolist()
    first = new[0] if new else None
    category = "A+" if first in answers.plus else "A-" if first in answers.minus else "neither"
    return {"id": record.id, "margin": read["margin"], "answer_mass": read["answer_mass"], "p_fav": read["p_fav"], "first": category,
            "first_token": tok.convert_ids_to_tokens([first])[0] if first is not None else None,
            "completion": tok.decode(new, skip_special_tokens=False),
            "A_plus": record.answer_sets["A_plus"]["strings"], "A_minus": record.answer_sets["A_minus"]["strings"]}


def completion_markdown(row: dict[str, Any], record) -> list[str]:
    return [f"- `{row['id']}`  m = {row['margin']:+.3f}, mass = {row['answer_mass']:.3f}, first = {row['first']} ({row['first_token']!r})",
            f"  - prompt ends: `{record.prompt[-160:]!r}`",
            f"  - completion: `{row['completion']!r}`",
            f"  - A+ {row['A_plus']} / A- {row['A_minus']}"]


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    model_cfg = ctx.model_config
    if model_cfg is None:
        raise ValueError("raw-completions needs --model configs/models/<model>.yaml")
    groups = _groups(cfg, cfg["seed"])
    if ctx.limit is not None:
        groups = [(f, k, recs[: ctx.limit]) for f, k, recs in groups]
    handle = load_model(model_cfg, device=ctx.device)
    rows: list[dict[str, Any]] = []
    md = [f"# Raw completions: `{model_cfg.get('name')}`", "",
          "Dev records only. Greedy decoding, BF16 model; margin m and answer mass from the FP32 head at the answer position. "
          "`first` says whether the first generated token is in A+ (the favorable or donor answer), A- or neither.", ""]
    summary_rows = []
    for family, key, records in groups:
        md += [f"## {family}: {key}", ""]
        masses, cats = [], []
        for r in records:
            row = {"family": family, "group": key, **complete(handle, r, max_new_tokens=cfg["max_new_tokens"])}
            masses.append(row["answer_mass"])
            cats.append(row["first"])
            rows.append(row)
            md += completion_markdown(row, r)
        share = {c: cats.count(c) / len(cats) for c in ("A+", "A-", "neither")} if cats else {}
        med = statistics.median(masses) if masses else float("nan")
        md += ["", f"Group summary: n = {len(records)}, median answer mass {med:.3f}, first token A+ {share.get('A+', 0):.0%}, "
               f"A- {share.get('A-', 0):.0%}, neither {share.get('neither', 0):.0%}", ""]
        summary_rows.append((family, key, len(records), med, share.get("neither", 0)))
    ctx.run.write_text("raw_completions.md", "\n".join(md) + "\n")
    ctx.run.write_text("raw_completions.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    lines = ["## What ran", "", f"{len(rows)} greedy completions of dev prompts for `{model_cfg.get('name')}`, "
             f"{cfg['n_per_group']} per task, template and prompt format.", "",
             "| Task | Group | n | Median answer mass | First token in neither set |", "|---|---|---|---|---|"]
    lines += [f"| {f} | {k} | {n} | {m:.3f} | {nei:.0%} |" for f, k, n, m, nei in summary_rows]
    lines += ["", "## What this permits", "", "Nothing is gated here. Rule 4: read the completions in raw_completions.md and confirm that "
              "the declared answer sets cover what the model says before any answer set or format is frozen (Stage 1 and 2 format gates).",
              "", "## Next step", "", "Send back raw_completions.md and SUMMARY.md."]
    return Outcome(status="PASS", summary="\n".join(lines) + "\n", report={"groups": [dict(zip(("family", "group", "n", "median_mass", "neither"), s)) for s in summary_rows]},
                   send_back=["raw_completions.md"])
