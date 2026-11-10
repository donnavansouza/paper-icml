"""Run contract: one wrapper around every command.

The wrapper
- refuses a non-empty ``--out`` unless ``--resume`` matches the protocol hash;
- enforces the dev/test separation (``--phase test`` needs a sealed frozen
  protocol, the matching code commit and an unused test marker);
- requires committed code for GPU and test runs;
- writes ``protocol.json`` and ``environment.json`` before the work starts, and
  ``report.json`` and ``SUMMARY.md`` after it, whatever happened;
- maps gate and check failures to their exit codes and always prints the final
  status line.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from saeselect import config as cfg
from saeselect.provenance import (
    REPO_ROOT,
    EngineeringFailed,
    GateFailed,
    RunContractError,
    RunDir,
    capture_environment,
    git_state,
    json_hash,
    loaded_source_hashes,
    seal_protocol,
    status_code,
    status_line,
    utc_now,
    verify_sealed_protocol,
)

STANDARD_FILES = ("SUMMARY.md", "report.json", "protocol.json", "environment.json")
# Partial results a stage writes while it runs; sent back when a run fails so nothing measured is lost.
PARTIAL_FILES = ("checks.json", "sae_progress.json")


@dataclass
class Outcome:
    """What a command returns when it finishes without an exception."""

    status: str
    summary: str
    report: dict[str, Any] = field(default_factory=dict)
    send_back: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    command: str
    run: RunDir
    config: dict[str, Any]
    model_config: dict[str, Any] | None
    phase: str | None
    limit: int | None
    device: str
    frozen: dict[str, Any] | None
    protocol: dict[str, Any]
    git: dict[str, Any]
    options: dict[str, Any]
    repo_root: Path = REPO_ROOT
    derived: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def smoke(self) -> bool:
        return self.limit is not None

    def derive(self, name: str, value: Any, *, procedure: str, inputs: dict[str, Any], numbers: dict[str, Any]) -> Any:
        """Record a (derived on dev) value with the procedure and numbers behind it.

        Written to ``derived.json`` and to ``decisions_entry.md`` in the run
        directory, so every derived value can be traced to its run.
        """
        self.derived[name] = {
            "value": value,
            "procedure": procedure,
            "inputs": inputs,
            "numbers": numbers,
            "run": str(self.run.path).replace("\\", "/"),
            "code_commit": self.git.get("commit"),
            "smoke": self.smoke,
        }
        self.run.write_json("derived.json", self.derived)
        self.run.write_text("decisions_entry.md", _decisions_markdown(self))
        return value

    def load_upstream(self, path: str | Path) -> dict[str, Any]:
        """Load a JSON output of an earlier run, refusing SMOKE outputs in real runs."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("smoke") and not self.smoke:
            raise RunContractError(
                f"{path} comes from a SMOKE run (--limit) and cannot be used by a later stage"
            )
        return data


def _decisions_markdown(ctx: RunContext) -> str:
    lines = [f"## Derived values from `{ctx.run.path.as_posix()}` (commit {ctx.git.get('commit')})", ""]
    if ctx.smoke:
        lines += ["**SMOKE run: not usable by any later stage.**", ""]
    for name, entry in ctx.derived.items():
        lines += [
            f"### {name} = {json.dumps(entry['value'])}",
            f"- Procedure: {entry['procedure']}",
            f"- Inputs: `{json.dumps(entry['inputs'], sort_keys=True)}`",
            f"- Numbers: `{json.dumps(entry['numbers'], sort_keys=True)}`",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Test-phase guard
# --------------------------------------------------------------------------


def test_marker_path(runs_root: Path, frozen_hash: str) -> Path:
    return runs_root / ".test_used" / frozen_hash


def check_test_phase(
    *,
    frozen_path: str | None,
    git: dict[str, Any],
    runs_root: Path,
    rerun_reason: str | None,
    resume_out: Path | None = None,
) -> dict[str, Any]:
    """Validate a test-phase invocation; return the frozen protocol.

    Raises RunContractError when: no frozen protocol; its hash does not verify;
    the code commit differs from the one recorded at freezing (unless a
    reason is given); or the protocol's test marker exists (unless a
    reason is given, or this is ``--resume`` of the run that set it).
    """
    if not frozen_path:
        raise RunContractError("--phase test requires --frozen <frozen_protocol.json>")
    path = Path(frozen_path)
    if not path.is_file():
        raise RunContractError(f"frozen protocol {frozen_path} not found")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen_hash = verify_sealed_protocol(frozen)
    if frozen.get("smoke"):
        raise RunContractError("the frozen protocol comes from a SMOKE run and cannot start a test run")
    frozen_commit = frozen.get("code_commit")
    if frozen_commit != git.get("commit") and not rerun_reason:
        raise RunContractError(
            f"code commit {git.get('commit')} differs from the frozen protocol's {frozen_commit}; "
            "a different commit requires a stated reason (--rerun-reason)"
        )
    marker = test_marker_path(runs_root, frozen_hash)
    if marker.exists() and not rerun_reason:
        if resume_out is not None:
            started = json.loads(marker.read_text(encoding="utf-8")).get("runs", [])
            if any(r.get("out") == resume_out.as_posix() for r in started):
                return frozen
        raise RunContractError(
            f"test marker {marker.as_posix()} exists: this frozen protocol already had its test run; "
            "a second test run requires --rerun-reason \"<reason>\""
        )
    return frozen


def write_test_marker(runs_root: Path, frozen_hash: str, *, out_dir: Path, rerun_reason: str | None) -> Path:
    marker = test_marker_path(runs_root, frozen_hash)
    marker.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if marker.exists():
        history = json.loads(marker.read_text(encoding="utf-8")).get("runs", [])
    history.append({"started_at": utc_now(), "out": out_dir.as_posix(), "rerun_reason": rerun_reason})
    marker.write_text(json.dumps({"frozen_protocol_hash": frozen_hash, "runs": history}, indent=2), encoding="utf-8")
    return marker


# --------------------------------------------------------------------------
# The wrapper
# --------------------------------------------------------------------------


def execute(
    command: str,
    options: dict[str, Any],
    work: Callable[[RunContext], Outcome],
    *,
    printer: Callable[[str], None] = print,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Run ``work`` under the run contract and return the process exit code."""
    run: RunDir | None = None
    ctx: RunContext | None = None
    status = "ERROR"
    send_back: list[str] = []
    try:
        ctx = _prepare(command, options, repo_root)
        run = ctx.run
        outcome = work(ctx)
        status = outcome.status
        flags = [*ctx.flags, *outcome.flags]
        if flags and status == "PASS":
            status = "PASS_WITH_FLAGS"
        _finish(ctx, status, outcome.summary, outcome.report, flags)
        send_back = _send_back(run, [*STANDARD_FILES, *outcome.send_back])
    except RunContractError as exc:
        printer(f"RUN CONTRACT: {exc}")
        if run is None:
            # Never write into a directory the contract refused to touch.
            printer(status_line("ERROR", None, []))
            return status_code("ERROR")
        status = "ERROR"
        _write_failure(ctx, run, status, f"Run contract violation: {exc}", [], None)
        send_back = _send_back(run, [*STANDARD_FILES, "error.txt", *PARTIAL_FILES])
    except GateFailed as exc:
        status = f"GATE_FAILED:{exc.gate}"
        _write_failure(ctx, run, status, exc.reason, exc.examples, None, report=exc.report,
                       flags=list(ctx.flags) if ctx is not None else [])
        send_back = _send_back(run, [*STANDARD_FILES, "error.txt", "raw_completions.md", *exc.send_back, *PARTIAL_FILES])
    except EngineeringFailed as exc:
        status = f"ENGINEERING_FAILED:{exc.check}"
        _write_failure(ctx, run, status, exc.reason, exc.examples, traceback.format_exc(), report=exc.report,
                       flags=list(ctx.flags) if ctx is not None else [])
        send_back = _send_back(run, [*STANDARD_FILES, "error.txt", *exc.send_back, *PARTIAL_FILES])
    except Exception as exc:  # noqa: BLE001 -- every exception must end in a status line
        status = "ERROR"
        tb = traceback.format_exc()
        if run is None:
            printer(tb)
            printer(status_line("ERROR", None, []))
            return status_code("ERROR")
        _write_failure(ctx, run, status, f"{type(exc).__name__}: {exc}", [], tb)
        send_back = _send_back(run, [*STANDARD_FILES, "error.txt", *PARTIAL_FILES])
    printer(status_line(status, run.path if run else None, send_back))
    return status_code(status)


def _prepare(command: str, options: dict[str, Any], repo_root: Path) -> RunContext:
    out = options.get("out")
    if not out:
        raise RunContractError("--out <new run directory> is required")
    phase = options.get("phase")
    device = options.get("device") or "cpu"
    limit = options.get("limit")
    if limit is not None and limit <= 0:
        raise RunContractError("--limit must be a positive number of clusters")

    config = cfg.load_config(options["config"]) if options.get("config") else {}
    model_config = cfg.load_config(options["model"]) if options.get("model") else None

    git = git_state(repo_root)
    needs_clean = device.startswith("cuda") or phase == "test"
    if needs_clean and (git.get("commit") is None or git.get("dirty")):
        raise RunContractError(
            "code must be committed before a GPU or test run; "
            f"commit={git.get('commit')}, uncommitted files={git.get('dirty_files')}"
        )

    runs_root = Path(out).resolve().parent
    frozen = None
    frozen_hash = None
    if phase == "test":
        frozen = check_test_phase(
            frozen_path=options.get("frozen"),
            git=git,
            runs_root=runs_root,
            rerun_reason=options.get("rerun_reason"),
            resume_out=Path(out).resolve() if options.get("resume") else None,
        )
        frozen_hash = frozen["protocol_hash"]
    elif options.get("frozen") and command != "analyze":  # analyze reads a frozen protocol to rebuild a run
        raise RunContractError("--frozen is only valid with --phase test")

    run_protocol = {
        "command": command,
        "phase": phase,
        "limit": limit,
        "smoke": limit is not None,
        "device": device,
        "config_path": options.get("config"),
        "config": config,
        "model_config_path": options.get("model"),
        "model_config": model_config,
        "frozen_protocol_hash": frozen_hash,
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "extra_options": {k: options[k] for k in sorted(options) if k in ("inventory_only", "run")},
    }
    run_protocol_hash = json_hash(run_protocol)
    run = RunDir.open(out, resume=bool(options.get("resume")), run_protocol_hash=run_protocol_hash)

    protocol = dict(run_protocol, run_protocol_hash=run_protocol_hash, rerun_reason=options.get("rerun_reason"))
    if not run.resume:
        protocol["created_at"] = utc_now()
        run.write_json("protocol.json", protocol)
    run.write_json("environment.json", capture_environment())

    ctx = RunContext(
        command=command,
        run=run,
        config=config,
        model_config=model_config,
        phase=phase,
        limit=limit,
        device=device,
        frozen=frozen,
        protocol=protocol,
        git=git,
        options=options,
        repo_root=repo_root,
    )
    if git.get("dirty"):
        ctx.flags.append(f"UNCOMMITTED_CODE: {', '.join(git.get('dirty_files', [])[:10])}")
    if phase == "test":
        write_test_marker(runs_root, frozen_hash, out_dir=run.path.resolve(), rerun_reason=options.get("rerun_reason"))
    return ctx


def _report_base(ctx: RunContext | None, status: str) -> dict[str, Any]:
    base: dict[str, Any] = {"status": status, "finished_at": utc_now(), "source_hashes": loaded_source_hashes()}
    if ctx is not None:
        base.update(
            command=ctx.command,
            phase=ctx.phase,
            smoke=ctx.smoke,
            code_commit=ctx.git.get("commit"),
            code_dirty=ctx.git.get("dirty"),
            run_protocol_hash=ctx.protocol.get("run_protocol_hash"),
            derived=ctx.derived,
        )
    return base


def _header(ctx: RunContext | None, status: str) -> str:
    lines = [f"# {ctx.command if ctx else 'run'}{' ' + ctx.phase if ctx and ctx.phase else ''}: {status}", ""]
    if ctx is not None:
        if ctx.smoke:
            lines += [f"**SMOKE run (--limit {ctx.limit}). Not usable by any later stage.**", ""]
        lines += [
            f"- Run directory: `{ctx.run.path.as_posix()}`",
            f"- Code commit: `{ctx.git.get('commit')}`{' (UNCOMMITTED CHANGES)' if ctx.git.get('dirty') else ''}",
            f"- Device: `{ctx.device}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def _finish(ctx: RunContext, status: str, summary: str, report: dict[str, Any], flags: list[str]) -> None:
    body = _header(ctx, status) + summary.rstrip() + "\n"
    if flags:
        body += "\n## Flags\n\n" + "\n".join(f"- {f}" for f in flags) + "\n"
    ctx.run.write_text("SUMMARY.md", body)
    ctx.run.write_json("report.json", {**_report_base(ctx, status), "flags": flags, **report})


def _write_failure(
    ctx: RunContext | None, run: RunDir, status: str, reason: str, examples: list[str], tb: str | None,
    *, report: dict[str, Any] | None = None, flags: list[str] | None = None,
) -> None:
    text = f"{status}\n\n{reason}\n"
    if tb:
        text += "\n" + tb
    run.write_text("error.txt", text)
    body = _header(ctx, status) + f"## Why the run stopped\n\n{reason}\n"
    if examples:
        body += "\n## Raw examples\n\n" + "\n\n".join(f"```\n{e}\n```" for e in examples) + "\n"
    if flags:
        body += "\n## Flags\n\n" + "\n".join(f"- {f}" for f in flags) + "\n"
    body += "\n## Next step\n\nRead error.txt and the raw examples before changing anything.\n"
    run.write_text("SUMMARY.md", body)
    run.write_json("report.json", {**_report_base(ctx, status), "reason": reason, "examples": examples,
                                   **({"flags": flags} if flags else {}), **(report or {})})


def _send_back(run: RunDir, names: list[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        if name not in seen and run.file(name).exists():
            seen.append(name)
    return [run.rel(n) for n in seen]
