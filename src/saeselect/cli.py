"""Command-line entry points.

    python -m saeselect <command> --config configs/<group>/<file>.yaml --out runs/<new-dir>
        [--model configs/models/<model>.yaml] [--phase dev|test] [--frozen <dev-run>/frozen_protocol.json]
        [--resume] [--device cuda] [--limit N]

Every command runs under :func:`saeselect.runner.execute`, which prints a final
``STATUS=... REPORT=... SEND_BACK=...`` line and returns the exit code.
Stage modules are imported only when their command runs.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

from saeselect.runner import Outcome, RunContext, execute

# command -> (module, function)
COMMANDS: dict[str, tuple[str, str]] = {
    "build-data": ("saeselect.tasks.build", "run"),
    "inventory": ("saeselect.stages.inventory", "run"),
    "raw-completions": ("saeselect.stages.raw_completions", "run"),
    "select": ("saeselect.stages.selection", "run"),
    "jacobian-controls": ("saeselect.stages.jacobian_controls", "run"),
    "greedy-reference": ("saeselect.stages.greedy_reference", "run"),
    "joint-selection": ("saeselect.stages.joint_selection", "run"),
    "generate": ("saeselect.stages.generation", "run"),
    "analyze": ("saeselect.analysis.rebuild", "run"),
    "compare-sets": ("saeselect.analysis.compare_sets", "run"),
    "posthoc": ("saeselect.analysis.posthoc", "run"),
    "additivity": ("saeselect.analysis.additivity", "run"),
    "decomposition": ("saeselect.analysis.decomposition", "run"),
    "matched-size": ("saeselect.analysis.matched_size", "run"),
    "figures": ("saeselect.analysis.figures", "run"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m saeselect",
        description="Controlled benchmark of signals for choosing which SAE features to edit.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="what to run")
    parser.add_argument("--config", help="configs/<file>.yaml with declared values")
    parser.add_argument("--out", help="new run directory (must not exist or be empty)")
    parser.add_argument("--model", help="configs/models/<model>.yaml")
    parser.add_argument("--phase", choices=["dev", "test"], help="dev or test phase of a stage")
    parser.add_argument("--frozen", help="frozen_protocol.json of the dev run (test phase, and analyze)")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run in --out")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cuda")
    parser.add_argument("--limit", type=int, help="run N clusters only; output is labeled SMOKE")
    parser.add_argument("--inventory-only", action="store_true", help="inventory: list artifacts without running checks")
    parser.add_argument("--rerun-reason", help="why a test is run again or on a new commit (recorded in the protocol)")
    parser.add_argument("--run", help="analyze: the test run directory to rebuild from")
    return parser


def _resolve(command: str):
    module_name, func_name = COMMANDS[command]
    return getattr(importlib.import_module(module_name), func_name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options: dict[str, Any] = {
        "config": args.config,
        "out": args.out,
        "model": args.model,
        "phase": args.phase,
        "frozen": args.frozen,
        "resume": args.resume,
        "device": args.device,
        "limit": args.limit,
        "inventory_only": args.inventory_only,
        "rerun_reason": args.rerun_reason,
        "run": args.run,
    }
    return execute(args.command, options, lambda ctx: _resolve(args.command)(ctx))


if __name__ == "__main__":
    sys.exit(main())
