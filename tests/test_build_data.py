"""`python -m saeselect build-data` end to end on the fixture sources: outputs, a
deterministic rebuild, refusal to change committed splits or records, and the
relation-minimum gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_fixtures import write_config
from saeselect import cli
from saeselect.tasks.records import load_records

TASKS = ["R1_country_capital", "R2_element_symbol", "R3_verb_past", "R4_country_language"]


def build(data: Path, model: Path, out: Path, capsys) -> tuple[int, str]:
    code = cli.main(["build-data", "--config", str(data), "--model", str(model), "--out", str(out)])
    return code, capsys.readouterr().out.strip().splitlines()[-1]


@pytest.mark.integration
def test_build_data_end_to_end(tmp_path, capsys):
    data, model = write_config(tmp_path)
    code, line = build(data, model, tmp_path / "runs" / "b1", capsys)
    assert code == 0, (tmp_path / "runs" / "b1" / "SUMMARY.md").read_text()
    assert line.startswith("STATUS=PASS")
    built = tmp_path / "built"
    for task in TASKS:
        card = json.loads((built / task / "benchmark_card.json").read_text(encoding="utf-8"))
        splits = json.loads((built / task / "splits.json").read_text(encoding="utf-8"))
        assert card["tokenizer"]["vocab_sha256"] and card["sources"]
        dev = load_records(built / task / "records.jsonl.gz", phase="dev", expected_sha256=card["records_sha256"])
        assert {r.cluster_id for r in dev} == {c for c, s in splits.items() if s == "dev"}
    for task in ("neutral",):
        assert json.loads((built / task / "benchmark_card.json").read_text(encoding="utf-8"))["ids"]

    # Rebuilding with the same inputs is identical and passes the unchanged-output check.
    code, line = build(data, model, tmp_path / "runs" / "b2", capsys)
    assert code == 0 and line.startswith("STATUS=PASS")

    # A committed split that the rebuild would change stops the build.
    splits_path = built / "R1_country_capital" / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    first = sorted(splits)[0]
    splits[first] = "test" if splits[first] == "dev" else "dev"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    code, line = build(data, model, tmp_path / "runs" / "b3", capsys)
    assert code == 3 and line.startswith("STATUS=ENGINEERING_FAILED:splits_changed")


@pytest.mark.integration
def test_relation_minimum_gate(tmp_path, capsys):
    data, model = write_config(tmp_path, min_test_pairs=40)
    code, line = build(data, model, tmp_path / "runs" / "g", capsys)
    assert code == 2 and line.startswith("STATUS=GATE_FAILED:relation_minimum")
    assert "relation_minimum" in (tmp_path / "runs" / "g" / "SUMMARY.md").read_text(encoding="utf-8")
