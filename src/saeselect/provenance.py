"""Run directories, hashing, git state, environment capture and the status line.

Implements the run contract and the provenance rule: every report records the commit id, hashes of the source
files actually loaded, data hashes, artifact revisions and the environment.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Status codes and exit codes.
EXIT_CODES: dict[str, int] = {
    "PASS": 0,
    "PASS_WITH_FLAGS": 0,
    "FROZEN_AWAITING_APPROVAL": 0,
    "INCONCLUSIVE": 0,
    "GATE_FAILED": 2,
    "ENGINEERING_FAILED": 3,
    "ERROR": 1,
}

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths whose uncommitted changes do not count as "code not committed": notes
# under docs/ must not block the next run. Everything else (src, configs, tests,
# data/built, pyproject.toml) must be committed.
_DIRTY_IGNORED_PREFIXES = ("docs/",)


class RunContractError(Exception):
    """The command line or output directory violates the run contract."""


class GateFailed(Exception):
    """A scientific gate failed. Exit code 2.

    ``send_back`` names result files the run wrote before failing (they are
    sent back with the standard files); ``report`` is merged into report.json.
    """

    def __init__(self, gate: str, reason: str, examples: list[str] | None = None, *,
                 send_back: list[str] | None = None, report: dict[str, Any] | None = None):
        super().__init__(f"{gate}: {reason}")
        self.gate = gate
        self.reason = reason
        self.examples = examples or []
        self.send_back = send_back or []
        self.report = report or {}


class EngineeringFailed(Exception):
    """A numerical or integrity check failed. Exit code 3. ``send_back`` and
    ``report`` as for :class:`GateFailed`."""

    def __init__(self, check: str, reason: str, examples: list[str] | None = None, *,
                 send_back: list[str] | None = None, report: dict[str, Any] | None = None):
        super().__init__(f"{check}: {reason}")
        self.check = check
        self.reason = reason
        self.examples = examples or []
        self.send_back = send_back or []
        self.report = report or {}


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 characters kept. The object is round-tripped through JSON first,
    so a YAML map with integer keys hashes as the string-keyed file it is written to (8, 16 would sort as 16, 8 once read back)."""
    native = json.loads(json.dumps(obj, ensure_ascii=False))
    return json.dumps(native, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def protocol_hash(protocol: dict[str, Any]) -> str:
    """Hash of a protocol dict, excluding its own ``protocol_hash`` field."""
    body = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    return json_hash(body)


def seal_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``protocol`` with ``protocol_hash`` set."""
    sealed = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    sealed["protocol_hash"] = protocol_hash(sealed)
    return sealed


def verify_sealed_protocol(protocol: dict[str, Any]) -> str:
    """Check that a sealed protocol was not edited after sealing; return its hash."""
    stored = protocol.get("protocol_hash")
    if not stored:
        raise RunContractError("frozen protocol has no protocol_hash field")
    actual = protocol_hash(protocol)
    if actual != stored:
        raise RunContractError(
            f"frozen protocol hash mismatch: stored {stored}, recomputed {actual}; "
            "the file was edited after freezing"
        )
    return stored


def loaded_source_hashes(prefixes: tuple[str, ...] = ("saeselect", "jlens")) -> dict[str, str]:
    """SHA-256 of every source file of the named packages that is loaded now.

    Called at the end of a run so that it covers every module the run imported
    (failure F14: hash the files actually loaded, not a list written by hand).
    """
    hashes: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        path = getattr(module, "__file__", None)
        if path and os.path.isfile(path):
            try:
                rel = str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                rel = str(Path(path).resolve()).replace("\\", "/")
            hashes[rel] = sha256_file(path)
    return hashes


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout


def git_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit id and whether the tracked code differs from it.

    ``dirty`` counts untracked and modified files outside the ignored prefixes; ignored-by-.gitignore files such as runs/ never count.
    """
    try:
        commit = _git(["rev-parse", "HEAD"], repo_root).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": True, "dirty_files": [], "error": "not a git repository or no commits"}
    porcelain = _git(["status", "--porcelain", "--untracked-files=all"], repo_root)
    changed = []
    for line in porcelain.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.append(path.replace("\\", "/"))
    dirty_files = [p for p in changed if not p.startswith(_DIRTY_IGNORED_PREFIXES)]
    return {
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "ignored_changes": [p for p in changed if p.startswith(_DIRTY_IGNORED_PREFIXES)],
    }


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _vcs_commit(dist_name: str) -> str | None:
    """Commit a package was installed from, if pip recorded one (PEP 610)."""
    try:
        text = importlib.metadata.distribution(dist_name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    info = json.loads(text)
    return info.get("vcs_info", {}).get("commit_id") or info.get("url")


def capture_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch", "transformers", "numpy", "scipy", "safetensors",
                "huggingface_hub", "tokenizers", "pyyaml", "jlens", "saeselect",
            )
        },
        "jlens_install_source": _vcs_commit("jlens"),
        "env_vars": {
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "HF_HUB_OFFLINE")
        },
    }
    try:
        import torch

        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
            env["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        env["torch_cuda"] = None
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if smi.returncode == 0:
            env["nvidia_smi"] = smi.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        env["nvidia_smi"] = None
    return env


# --------------------------------------------------------------------------
# Status line
# --------------------------------------------------------------------------


def status_code(status: str) -> int:
    base = status.split(":", 1)[0]
    if base not in EXIT_CODES:
        raise ValueError(f"unknown status {status!r}")
    return EXIT_CODES[base]


def status_line(status: str, run_dir: str | os.PathLike | None, send_back: list[str]) -> str:
    """The final printed line of every command."""
    status_code(status)  # validates
    report = f"{_posix(run_dir)}/SUMMARY.md" if run_dir is not None else "none"
    files = ",".join(send_back) if send_back else "none"
    return f"STATUS={status} REPORT={report} SEND_BACK={files}"


def _posix(path: str | os.PathLike | None) -> str:
    return str(path).replace("\\", "/")


# --------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------


@dataclass
class RunDir:
    """One run's output directory.

    A fresh run needs a directory that does not exist or is empty. ``--resume``
    needs an existing directory whose ``protocol.json`` has the same protocol
    hash as the current invocation.
    """

    path: Path
    resume: bool = False
    written: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | os.PathLike, *, resume: bool, run_protocol_hash: str) -> "RunDir":
        path = Path(path)
        if resume:
            proto = path / "protocol.json"
            if not proto.is_file():
                raise RunContractError(f"--resume given but {_posix(proto)} does not exist")
            stored = json.loads(proto.read_text(encoding="utf-8")).get("run_protocol_hash")
            if stored != run_protocol_hash:
                raise RunContractError(
                    f"--resume refused: protocol hash of {_posix(path)} is {stored}, "
                    f"this invocation's is {run_protocol_hash}"
                )
            return cls(path=path, resume=True)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RunContractError(
                f"--out {_posix(path)} exists and is not empty; every run needs a new "
                "directory (failed runs are kept untouched)"
            )
        path.mkdir(parents=True, exist_ok=True)
        return cls(path=path, resume=False)

    def file(self, name: str) -> Path:
        return self.path / name

    def write_json(self, name: str, obj: Any) -> Path:
        target = self.file(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        self._note(name)
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.file(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self._note(name)
        return target

    def _note(self, name: str) -> None:
        if name not in self.written:
            self.written.append(name)

    def rel(self, name: str) -> str:
        return _posix(self.file(name))


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return _posix(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
