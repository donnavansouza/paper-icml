"""``python -m saeselect build-data``: fetch the pinned sources and build every task.

Writes per task under ``build.out_dir``:
- ``records.jsonl.gz`` (git-ignored; text of third-party sources is never committed),
- ``splits.json`` (cluster -> dev/test; committed),
- ``benchmark_card.json`` (sources and hashes, licenses, counts per split,
  filters with counts removed, exclusions with reasons, the tokenizer
  fingerprint and the sha256 of the uncompressed records; committed).

Rebuilding is deterministic. If committed splits or a committed card's records
hash would change, the build stops instead of overwriting: splits are fixed
before any model run.

A config may name a ``base`` config and override some of its keys (another
tokenizer and output directory, or a subset of tasks).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from saeselect.config import load_config
from saeselect.provenance import EngineeringFailed, GateFailed, sha256_file
from saeselect.runner import Outcome, RunContext
from saeselect.tasks import neutral, relations
from saeselect.tasks.records import write_records
from saeselect.tasks.sources import SourceStore
from saeselect.tasks.tokenization import load_tokenizer, tokenizer_fingerprint

RELATION_SOURCES = {"R1": "countries", "R2": "elements", "R3": "irregular_verbs", "R4": "countries"}


def _relation_concepts(rel: str, rel_cfg: dict, path: Path):
    if rel_cfg.get("items"):  # a declared item file in the repository (R5, R6)
        return relations.parse_declared(path, kind=rel_cfg["items"]["kind"], answers=rel_cfg["answers"])
    if rel == "R1":
        return relations.parse_countries(path, key="capital")
    if rel == "R4":
        return relations.parse_countries(path, key="language")
    if rel == "R2":
        return relations.parse_elements(path, lower=rel_cfg.get("concept_case") == "lower")
    if rel == "R3":
        return relations.parse_irregular_verbs(path, modal_verbs=rel_cfg["modal_verbs"])
    raise ValueError(rel)


def _check_unchanged(task_dir: Path, splits: dict[str, str], records_sha: str | None) -> None:
    split_path = task_dir / "splits.json"
    if split_path.is_file():
        committed = json.loads(split_path.read_text(encoding="utf-8"))
        if committed != splits:
            changed = sorted(k for k in set(committed) | set(splits) if committed.get(k) != splits.get(k))
            raise EngineeringFailed(
                "splits_changed",
                f"{task_dir.as_posix()}: rebuilt split assignments differ from the committed splits.json for {len(changed)} clusters "
                f"(e.g. {changed[:5]}); splits are fixed at data-build time. Remove the committed files deliberately to rebuild.",
            )
    card_path = task_dir / "benchmark_card.json"
    if records_sha and card_path.is_file():
        committed_sha = json.loads(card_path.read_text(encoding="utf-8")).get("records_sha256")
        if committed_sha and committed_sha != records_sha:
            raise EngineeringFailed(
                "records_changed",
                f"{task_dir.as_posix()}: rebuilt records hash {records_sha} differs from the committed card's {committed_sha} "
                "(tokenizer, source or code changed). Records are fixed at data-build time.",
            )


def _override(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        out[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    return out


def _resolved(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """A config with ``base`` and ``overrides``: the base config, itself resolved, with those keys overridden."""
    if "overrides" not in cfg:
        return cfg, None
    base, _ = _resolved(load_config(cfg["base"]))
    merged = _override({**base, "_source": cfg["_source"], "tasks": cfg.get("tasks", base.get("tasks"))}, cfg["overrides"])
    return merged, {"base_config": cfg["base"], "base_config_sha256": base["_source"]["sha256"], "overrides": cfg["overrides"]}


def run(ctx: RunContext) -> Outcome:
    cfg, based_on = _resolved(ctx.config)
    build = cfg["build"]
    model_cfg = ctx.model_config or load_config(build["tokenizer_model"])
    tokenizer = load_tokenizer(model_cfg)
    fingerprint = tokenizer_fingerprint(tokenizer)
    seed = build["split_seed"]
    store = SourceStore(cfg["sources"], build["raw_dir"])
    out_dir = Path(build["out_dir"])
    wanted = set(cfg.get("tasks") or ["R1", "R2", "R3", "R4", "neutral"])
    common = {"data_config_sha256": cfg["_source"]["sha256"], "split_seed": seed, "tokenizer": fingerprint}
    if based_on:
        common["based_on"] = based_on
    built: dict[str, dict[str, Any]] = {}

    rel_cfg = cfg["relations"]
    for rel in sorted((k for k in rel_cfg if re.fullmatch(r"R\d+", k)), key=lambda k: int(k[1:])):
        if rel not in wanted:
            continue
        items = rel_cfg[rel].get("items")
        if items:  # the item file is part of the repository and hashed into the card
            path = Path(items["file"])
            concepts, excluded = _relation_concepts(rel, rel_cfg[rel], path)
            sources = [{"file": path.as_posix(), "sha256": sha256_file(path), "license": items["license"]}]
        else:
            src = RELATION_SOURCES[rel]
            concepts, excluded = _relation_concepts(rel, rel_cfg[rel], store.url_file(src).path)
            sources = store.card_entries(src)
        result = relations.build_relation(rel, rel_cfg, rel_cfg[rel], concepts=concepts, source_excluded=excluded, tokenizer=tokenizer, seed=seed)
        result["card"]["sources"] = sources
        result["card"]["role_in_design"] = "active" if rel in rel_cfg["active"] else "reserve"
        built[f"{rel}_{rel_cfg[rel]['name']}"] = result

    items: dict[str, dict[str, Any]] = {}
    if "neutral" in wanted:
        import pyarrow.parquet as pq

        spec = cfg["sources"]["wikitext"]
        path = store.hub_file("wikitext", next(iter(spec["files"]))).path
        lines = pq.read_table(path).column("text").to_pylist()
        n_cfg = cfg["neutral"]
        result = neutral.build_neutral(lines, tokenizer=tokenizer, n_windows=n_cfg["n_windows"], window_tokens=n_cfg["window_tokens"], seed=seed)
        result["card"]["sources"] = store.card_entries("wikitext")
        items["neutral"] = {**result, "file": "windows.jsonl"}

    # -- write, refusing silent changes to committed splits and cards ----------
    summary_rows = []
    cards_dir = ctx.run.file("cards")
    cards_dir.mkdir(exist_ok=True)
    reference = build.get("reference_splits_dir")  # another tokenizer's build keeps the reference build's splits
    for task, result in built.items():
        if reference:
            ref = Path(reference) / task / "splits.json"
            if not ref.is_file():
                raise EngineeringFailed("reference_splits", f"{task}: no {ref.as_posix()} to compare with (build.reference_splits_dir)")
            if json.loads(ref.read_text(encoding="utf-8")) != result["splits"]:
                raise EngineeringFailed("reference_splits", f"{task}: splits differ from {ref.as_posix()}, the reference build")
            result["card"]["reference_splits"] = {"file": ref.as_posix(), "sha256": sha256_file(ref), "identical": True}
        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = task_dir / "records.new.jsonl.gz"
        count, sha = write_records(tmp_path, result["records"])
        try:
            _check_unchanged(task_dir, result["splits"], sha)
        except EngineeringFailed:
            tmp_path.unlink(missing_ok=True)
            raise
        tmp_path.replace(task_dir / "records.jsonl.gz")
        card = {**result["card"], **common, "records_file": "records.jsonl.gz", "records_sha256": sha, "n_records": count}
        (task_dir / "splits.json").write_text(json.dumps(result["splits"], indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (task_dir / "benchmark_card.json").write_text(json.dumps(card, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        shutil.copy(task_dir / "benchmark_card.json", cards_dir / f"{task}.json")
        splits = result["splits"]
        summary_rows.append((task, count, sum(v == "dev" for v in splits.values()), sum(v == "test" for v in splits.values()), card))
    for task, result in items.items():
        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        neutral.write_items(task_dir / result["file"], result["items"])
        sha = sha256_file(task_dir / result["file"])
        ids = [item["id"] for item in result["items"]]
        _check_unchanged(task_dir, {i: "all" for i in ids}, sha)
        card = {**result["card"], **common, "items_file": result["file"], "records_sha256": sha, "ids": ids}
        (task_dir / "splits.json").write_text(json.dumps({i: "all" for i in ids}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (task_dir / "benchmark_card.json").write_text(json.dumps(card, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        shutil.copy(task_dir / "benchmark_card.json", cards_dir / f"{task}.json")
        summary_rows.append((task, len(ids), None, None, card))

    # -- summary ------------------------------------------------------------
    flags = []
    below = [t for t, r in built.items() if r.get("below_minimum") and t.split("_")[0] in rel_cfg["active"]]
    reserve_below = [t for t, r in built.items() if r.get("below_minimum") and t.split("_")[0] == rel_cfg["reserve"]]
    if reserve_below:
        flags.append(f"reserve relation below the declared minimum: {reserve_below}")

    lines = ["## What ran", "", "Data build: pinned sources fetched and verified by sha256, every task built with the tokenizer "
             f"`{fingerprint['name_or_path']}` (vocabulary sha256 `{fingerprint['vocab_sha256'][:16]}…`).", "",
             "## Tasks", "", "| Task | Records or items | Dev clusters | Test clusters | Records sha256 |", "|---|---|---|---|---|"]
    for task, count, n_dev, n_test, card in summary_rows:
        lines.append(f"| {task} | {count} | {'' if n_dev is None else n_dev} | {'' if n_test is None else n_test} | `{card['records_sha256'][:16]}…` |")
    lines += ["", f"## Relation minimums ({rel_cfg['dev_pairs']} dev pairs, at least {rel_cfg['min_test_pairs']} test pairs)", ""]
    for task, result in built.items():
        if "minimum" in result["card"]:
            c = result["card"]["counts"]
            lines.append(f"- {task} ({result['card']['role_in_design']}): {c['dev_pairs']} dev + {c['test_pairs']} test pairs, "
                         f"{'met' if result['card']['minimum']['met'] else 'NOT MET'}")
    report = {"tasks": {t: {"n": n, "dev": d, "test": te, "records_sha256": c["records_sha256"]} for t, n, d, te, c in summary_rows},
              "tokenizer": fingerprint}
    if below:
        raise GateFailed("relation_minimum", f"active relations below the declared minimum of {rel_cfg['dev_pairs']} dev and "
                         f"{rel_cfg['min_test_pairs']} test pairs: {below}", ["\n".join(lines)])
    return Outcome(status="PASS", summary="\n".join(lines) + "\n", report=report, send_back=[], flags=flags)
