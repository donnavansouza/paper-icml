"""TrialResult schema, per-cluster checkpoints, and results.zip.

Progress is checkpointed per cluster.
A checkpoint records the checks run for that cluster; resume refuses to
continue if any saved checkpoint has a failed check (failure F13), so a failed
check can never be skipped by resuming.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from saeselect.provenance import RunContractError


@dataclass
class TrialResult:
    """One trial. Every field the analysis needs to rebuild a table."""

    trial_id: str
    cluster_id: str
    stage: str
    layer: int | None
    site: str | None
    positions: list[int]
    margins: dict[str, float] = field(default_factory=dict)  # clean, don, FULL, SAE-ALL, ERR, DIM, ...
    pool: list[int] = field(default_factory=list)
    activation_delta: dict[str, float] = field(default_factory=dict)  # feature id -> summed delta a over P
    signals: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)  # variant -> signal -> feature -> score
    single_feature: dict[str, dict[str, float]] = field(default_factory=dict)  # feature -> {effect, intended_norm, realized_norm}
    sets: list[dict[str, Any]] = field(default_factory=list)  # {signal, k, alpha, features, margin, intended_norm, realized_norm}
    random_draws: list[dict[str, Any]] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class CheckpointStore:
    """``<run>/checkpoints/<cluster_id>.json``: result rows plus check outcomes."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cluster_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cluster_id)
        return self.dir / f"{safe}.json"

    def save(self, cluster_id: str, rows: list[dict[str, Any]], checks: dict[str, bool]) -> None:
        payload = {"cluster_id": cluster_id, "checks": checks, "passed": all(checks.values()), "rows": rows}
        tmp = self._path(cluster_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path(cluster_id))

    def load_all(self) -> dict[str, dict[str, Any]]:
        """All saved checkpoints; refuses if any has a failed check."""
        loaded: dict[str, dict[str, Any]] = {}
        for path in sorted(self.dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("passed", False):
                failed = [name for name, ok in payload.get("checks", {}).items() if not ok]
                raise RunContractError(
                    f"resume refused: checkpoint {path.name} has failed checks {failed}; "
                    "rerun into a new directory and keep this one"
                )
            loaded[payload["cluster_id"]] = payload
        return loaded

    def done(self) -> set[str]:
        return set(self.load_all())


def write_results_zip(run_dir: Path, members: dict[str, str]) -> Path:
    """Write ``results.zip`` with the given {name: text} members (JSON lines)."""
    target = Path(run_dir) / "results.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in sorted(members.items()):
            archive.writestr(name, text)
    return target
