"""TaskRecord schema, seeded splits, and the split-enforcing loader.

Every adapter produces TaskRecords; nothing
downstream depends on a benchmark's name. Splits are assigned at build time
by a seeded hash of ``cluster_id`` and stored in each record. A dev-phase run
never materializes a test record: the loader skips test lines on read and
raises :class:`SplitViolation` if a test record is asked for.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SPLITS = ("dev", "test")
ROLES = ("source", "donor")
PROMPT_FORMATS = ("raw", "chat")


class SplitViolation(Exception):
    """A run asked for a record outside the splits its phase may load."""


class RecordError(ValueError):
    """A record violates the schema."""


@dataclass
class TaskRecord:
    id: str
    cluster_id: str
    split: str
    family: str
    template_id: str
    prompt: str
    prompt_format: str
    role: str
    answer_sets: dict[str, dict[str, list]]
    attribute: dict[str, Any] | None = None
    attribute_spans: list[list[int]] = field(default_factory=list)
    matched_group_id: str | None = None
    label_order: str | None = None
    polarity: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TaskRecord":
        if self.split not in SPLITS:
            raise RecordError(f"{self.id}: split {self.split!r} not in {SPLITS}")
        if self.role not in ROLES:
            raise RecordError(f"{self.id}: role {self.role!r} not in {ROLES}")
        if self.prompt_format not in PROMPT_FORMATS:
            raise RecordError(f"{self.id}: prompt_format {self.prompt_format!r} not in {PROMPT_FORMATS}")
        if not self.prompt:
            raise RecordError(f"{self.id}: empty prompt")
        for key in ("A_plus", "A_minus"):
            entry = self.answer_sets.get(key)
            if not entry or not entry.get("ids") or not entry.get("strings"):
                raise RecordError(f"{self.id}: answer_sets.{key} needs non-empty ids and strings")
            if len(entry["ids"]) != len(entry["strings"]):
                raise RecordError(f"{self.id}: answer_sets.{key} ids and strings differ in length")
        overlap = set(self.answer_sets["A_plus"]["ids"]) & set(self.answer_sets["A_minus"]["ids"])
        if overlap:
            raise RecordError(f"{self.id}: A_plus and A_minus share token ids {sorted(overlap)}")
        for span in self.attribute_spans:
            if len(span) != 2 or not 0 <= span[0] < span[1] <= len(self.prompt):
                raise RecordError(f"{self.id}: attribute span {span} outside the prompt")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise RecordError(f"{data.get('id')}: unknown fields {sorted(unknown)}")
        return cls(**data).validate()


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def hash_unit(key: str, seed: int) -> float:
    """Deterministic value in [0, 1) from a seeded SHA-256 of ``key``."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_by_fraction(cluster_ids: Iterable[str], *, seed: int, dev_fraction: float) -> dict[str, str]:
    """Order clusters by seeded hash; the first round(n * dev_fraction) are dev.

    Ordering (rather than thresholding each hash) makes the dev count exact.
    """
    ids = sorted(set(cluster_ids))
    ordered = sorted(ids, key=lambda c: (hash_unit(c, seed), c))
    n_dev = round(len(ordered) * dev_fraction)
    return {c: ("dev" if i < n_dev else "test") for i, c in enumerate(ordered)}


def split_by_count(cluster_ids: Iterable[str], *, seed: int, n_dev: int) -> dict[str, str]:
    """The first ``n_dev`` clusters in seeded-hash order are dev; the rest test."""
    ids = sorted(set(cluster_ids))
    if n_dev > len(ids):
        raise ValueError(f"asked for {n_dev} dev clusters out of {len(ids)}")
    ordered = sorted(ids, key=lambda c: (hash_unit(c, seed), c))
    return {c: ("dev" if i < n_dev else "test") for i, c in enumerate(ordered)}


def halve(cluster_ids: Iterable[str], *, seed: int, labels: tuple[str, str]) -> dict[str, str]:
    """Split clusters into two seeded halves."""
    ids = sorted(set(cluster_ids))
    ordered = sorted(ids, key=lambda c: (hash_unit(c, seed), c))
    half = len(ordered) // 2
    return {c: (labels[0] if i < half else labels[1]) for i, c in enumerate(ordered)}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def allowed_splits(phase: str) -> frozenset[str]:
    """Which splits a phase may load. Dev runs load dev only; test runs load
    test only, unless a stage explicitly opts into dev records as well."""
    if phase == "dev":
        return frozenset({"dev"})
    if phase == "test":
        return frozenset({"test"})
    raise ValueError(f"unknown phase {phase!r}")


class RecordSet:
    """Records of one task file visible to one phase."""

    def __init__(self, records: list[TaskRecord], hidden_ids: set[str], *, phase: str, path: str):
        self._by_id = {r.id: r for r in records}
        self._hidden = hidden_ids
        self.phase = phase
        self.path = path

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[TaskRecord]:
        return iter(self._by_id.values())

    def get(self, record_id: str) -> TaskRecord:
        if record_id in self._hidden:
            raise SplitViolation(f"record {record_id} is not loadable in the {self.phase} phase")
        return self._by_id[record_id]

    def clusters(self) -> list[str]:
        return sorted({r.cluster_id for r in self})

    def by_cluster(self) -> dict[str, list[TaskRecord]]:
        out: dict[str, list[TaskRecord]] = {}
        for record in self:
            out.setdefault(record.cluster_id, []).append(record)
        return out


def _open_text(path: Path, mode: str):
    """Read-side text stream for .jsonl or .jsonl.gz."""
    if mode != "r":
        raise ValueError("write through write_records")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_records(path: str | Path, *, phase: str, include_dev_in_test: bool = False, expected_sha256: str | None = None) -> RecordSet:
    """Read ``records.jsonl[.gz]`` keeping only the splits the phase may load.

    Lines of other splits are skipped after reading their ``split`` and ``id``;
    they are never turned into TaskRecords. ``include_dev_in_test`` lets a
    test-phase stage read dev records it needs by design (it is recorded by
    the caller in its protocol). With ``expected_sha256`` (from the benchmark
    card), the uncompressed content must match it. Identical prompt strings are
    shared between records to keep memory near the distinct-prompt size.
    """
    path = Path(path)
    allowed = set(allowed_splits(phase))
    if phase == "test" and include_dev_in_test:
        allowed.add("dev")
    records: list[TaskRecord] = []
    hidden: set[str] = set()
    prompts: dict[str, str] = {}
    digest = hashlib.sha256()
    with _open_text(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            data = json.loads(line)
            split = data.get("split")
            if split not in SPLITS:
                raise RecordError(f"{path}:{line_no}: split {split!r} not in {SPLITS}")
            if split not in allowed:
                hidden.add(data.get("id"))
                continue
            data["prompt"] = prompts.setdefault(data["prompt"], data["prompt"])
            records.append(TaskRecord.from_dict(data))
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise RecordError(f"{path}: content sha256 {digest.hexdigest()} differs from the benchmark card's {expected_sha256}")
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise RecordError(f"{path}: duplicate record ids")
    return RecordSet(records, hidden, phase=phase, path=str(path))


def write_records(path: str | Path, records: Iterable[TaskRecord]) -> tuple[int, str]:
    """Validate and write records as JSON lines (gzip if the name ends in .gz),
    checking that no cluster crosses splits. Returns (count, sha256 of the
    uncompressed content)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_split: dict[str, str] = {}
    count = 0
    digest = hashlib.sha256()
    with open(path, "wb") as raw:
        # mtime=0 and no stored file name keep the compressed bytes deterministic
        stream = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) if path.suffix == ".gz" else raw
        with io.TextIOWrapper(stream, encoding="utf-8", newline="\n") as handle:
            for record in records:
                record.validate()
                previous = cluster_split.setdefault(record.cluster_id, record.split)
                if previous != record.split:
                    raise RecordError(f"cluster {record.cluster_id} appears in both {previous} and {record.split}")
                line = record.to_json() + "\n"
                digest.update(line.encode("utf-8"))
                handle.write(line)
                count += 1
    return count, digest.hexdigest()
