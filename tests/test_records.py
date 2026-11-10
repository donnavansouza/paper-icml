"""Tasks: schema validation, split determinism, clusters never cross splits, and no
test record is reachable in the dev phase."""

from __future__ import annotations

import pytest

from saeselect.tasks.records import (
    RecordError,
    SplitViolation,
    TaskRecord,
    halve,
    load_records,
    split_by_count,
    split_by_fraction,
    write_records,
)


def make_record(i: int, cluster: str, split: str, **overrides) -> TaskRecord:
    fields = dict(
        id=f"r{i}",
        cluster_id=cluster,
        split=split,
        family="R1",
        template_id="t1",
        prompt="The capital of France is",
        prompt_format="raw",
        role="source",
        answer_sets={"A_plus": {"ids": [10], "strings": [" Paris"]}, "A_minus": {"ids": [11], "strings": [" Rome"]}},
    )
    fields.update(overrides)
    return TaskRecord(**fields)


def test_split_determinism_and_exact_counts():
    clusters = [f"c{i}" for i in range(50)]
    a = split_by_fraction(clusters, seed=7, dev_fraction=0.5)
    assert a == split_by_fraction(list(reversed(clusters)), seed=7, dev_fraction=0.5)
    assert sum(v == "dev" for v in a.values()) == 25
    assert a != split_by_fraction(clusters, seed=8, dev_fraction=0.5)
    b = split_by_count(clusters, seed=7, n_dev=12)
    assert sum(v == "dev" for v in b.values()) == 12
    assert b == split_by_count(clusters, seed=7, n_dev=12)
    with pytest.raises(ValueError):
        split_by_count(clusters, seed=7, n_dev=51)
    halves = halve([c for c, s in a.items() if s == "dev"], seed=3, labels=("dev_A", "dev_B"))
    assert sorted(halves.values()).count("dev_A") == 12


def test_validation_rejects_bad_records():
    with pytest.raises(RecordError, match="share token ids"):
        make_record(0, "c", "dev", answer_sets={"A_plus": {"ids": [1], "strings": ["a"]}, "A_minus": {"ids": [1], "strings": ["b"]}}).validate()
    with pytest.raises(RecordError, match="role"):
        make_record(0, "c", "dev", role="unknown").validate()
    with pytest.raises(RecordError, match="split"):
        make_record(0, "c", "train").validate()
    with pytest.raises(RecordError, match="span"):
        make_record(0, "c", "dev", attribute_spans=[[5, 999]]).validate()
    with pytest.raises(RecordError, match="unknown fields"):
        TaskRecord.from_dict({**make_record(0, "c", "dev").__dict__, "extra": 1})


def test_clusters_never_cross_splits(tmp_path):
    with pytest.raises(RecordError, match="both dev and test"):
        write_records(tmp_path / "r.jsonl", [make_record(0, "c1", "dev"), make_record(1, "c1", "test")])


def test_no_test_record_reachable_in_dev(tmp_path):
    path = tmp_path / "records.jsonl"
    write_records(path, [make_record(0, "c1", "dev"), make_record(1, "c2", "test"), make_record(2, "c3", "dev")])
    dev = load_records(path, phase="dev")
    assert {r.id for r in dev} == {"r0", "r2"}
    assert dev.clusters() == ["c1", "c3"]
    with pytest.raises(SplitViolation):
        dev.get("r1")
    test = load_records(path, phase="test")
    assert {r.id for r in test} == {"r1"}
    with pytest.raises(SplitViolation):
        test.get("r0")
    both = load_records(path, phase="test", include_dev_in_test=True)
    assert len(both) == 3
    with pytest.raises(ValueError):
        load_records(path, phase="train")
