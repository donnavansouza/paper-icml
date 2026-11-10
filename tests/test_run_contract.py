"""Run contract: new output directories, status lines and exit codes, SMOKE labels,
resume rules, and the test-phase guard (frozen protocol, commit, marker)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import commit_all
from saeselect import cli
from saeselect.provenance import EngineeringFailed, GateFailed, RunContractError, git_state, seal_protocol, status_line
from saeselect.results import CheckpointStore
from saeselect.runner import Outcome, execute


def run(repo: Path, out: Path, work, **options):
    lines: list[str] = []
    opts = {"out": str(out), "device": "cpu", **options}
    code = execute("inventory", opts, work, printer=lines.append, repo_root=repo)
    return code, lines


def ok(ctx):
    return Outcome(status="PASS", summary="## Gates\n\nall good\n")


def test_pass_writes_standard_files_and_status_line(git_repo, tmp_path):
    out = tmp_path / "runs" / "a"
    code, lines = run(git_repo, out, ok)
    assert code == 0
    assert lines[-1].startswith("STATUS=PASS REPORT=") and lines[-1].endswith("environment.json")
    for name in ("SUMMARY.md", "report.json", "protocol.json", "environment.json"):
        assert (out / name).is_file()
    report = json.loads((out / "report.json").read_text())
    assert report["code_commit"] == git_state(git_repo)["commit"]
    assert any(path.endswith("saeselect/runner.py") for path in report["source_hashes"])


def test_refuses_non_empty_out_and_leaves_it_untouched(git_repo, tmp_path):
    out = tmp_path / "runs" / "a"
    out.mkdir(parents=True)
    (out / "old.txt").write_text("keep me")
    code, lines = run(git_repo, out, ok)
    assert code == 1
    assert lines[-1] == "STATUS=ERROR REPORT=none SEND_BACK=none"
    assert sorted(p.name for p in out.iterdir()) == ["old.txt"]


@pytest.mark.parametrize(
    "exc, status, code",
    [
        (GateFailed("format_gate", "answer mass 0.4 < 0.9", ["raw completion 1"]), "GATE_FAILED:format_gate", 2),
        (EngineeringFailed("N4", "sham changed logits"), "ENGINEERING_FAILED:N4", 3),
        (RuntimeError("boom"), "ERROR", 1),
    ],
)
def test_failures_fail_loudly(git_repo, tmp_path, exc, status, code):
    def work(ctx):
        raise exc

    out = tmp_path / "runs" / "f"
    got, lines = run(git_repo, out, work)
    assert got == code
    assert lines[-1].startswith(f"STATUS={status} ")
    assert (out / "error.txt").is_file()
    summary = (out / "SUMMARY.md").read_text()
    assert status in summary
    if isinstance(exc, GateFailed):
        assert "raw completion 1" in summary
    if isinstance(exc, RuntimeError):
        assert "Traceback" in (out / "error.txt").read_text()


def test_flags_turn_pass_into_pass_with_flags(git_repo, tmp_path):
    code, lines = run(git_repo, tmp_path / "runs" / "a", lambda ctx: Outcome(status="PASS", summary="x", flags=["N8 failed: suffix caching off"]))
    assert code == 0 and lines[-1].startswith("STATUS=PASS_WITH_FLAGS ")


def test_limit_labels_smoke(git_repo, tmp_path):
    out = tmp_path / "runs" / "s"
    code, _ = run(git_repo, out, ok, limit=2)
    assert code == 0
    assert "SMOKE" in (out / "SUMMARY.md").read_text()
    assert json.loads((out / "protocol.json").read_text())["smoke"] is True


def test_load_upstream_refuses_smoke_in_real_run(git_repo, tmp_path):
    smoke_file = tmp_path / "stage0_decisions.json"
    smoke_file.write_text(json.dumps({"smoke": True}))

    def work(ctx):
        ctx.load_upstream(smoke_file)
        return Outcome(status="PASS", summary="x")

    code, lines = run(git_repo, tmp_path / "runs" / "u", work)
    assert code == 1 and lines[-1].startswith("STATUS=ERROR")


def test_derive_writes_derived_json(git_repo, tmp_path):
    out = tmp_path / "runs" / "d"

    def work(ctx):
        ctx.derive("tau_tie", 0.031, procedure="99th percentile of |m_fp32 - m_bf16|", inputs={"n": 200}, numbers={"p99": 0.031})
        return Outcome(status="PASS", summary="x")

    assert run(git_repo, out, work)[0] == 0
    derived = json.loads((out / "derived.json").read_text())
    assert derived["tau_tie"]["value"] == 0.031
    assert "tau_tie" in (out / "decisions_entry.md").read_text()


def test_resume_requires_matching_protocol(git_repo, tmp_path):
    out = tmp_path / "runs" / "r"
    assert run(git_repo, out, ok, limit=3)[0] == 0
    assert run(git_repo, out, ok, limit=3, resume=True)[0] == 0
    code, lines = run(git_repo, out, ok, limit=4, resume=True)
    assert code == 1 and "STATUS=ERROR" in lines[-1]
    assert run(git_repo, tmp_path / "runs" / "never", ok, resume=True)[0] == 1


def test_resume_refuses_failed_checkpoint(git_repo, tmp_path):
    out = tmp_path / "runs" / "c"

    def first(ctx):
        store = CheckpointStore(ctx.run.path)
        store.save("cluster-1", [{"x": 1}], {"parity_probe": True})
        store.save("cluster-2", [{"x": 2}], {"parity_probe": False})
        raise EngineeringFailed("parity_probe", "batched and serial disagree")

    def resumed(ctx):
        CheckpointStore(ctx.run.path).load_all()
        return Outcome(status="PASS", summary="x")

    assert run(git_repo, out, first)[0] == 3
    code, lines = run(git_repo, out, resumed, resume=True)
    assert code == 1 and lines[-1].startswith("STATUS=ERROR")
    assert "failed checks" in (out / "error.txt").read_text()


def test_gpu_run_requires_committed_code(git_repo, tmp_path):
    (git_repo / "src" / "code.py").write_text("x = 2\n")
    code, lines = run(git_repo, tmp_path / "runs" / "g", ok, device="cuda")
    assert code == 1 and "REPORT=none" in lines[-1]
    # documentation edits do not count as uncommitted code
    commit_all(git_repo)
    (git_repo / "docs" / "STATE.md").write_text("updated\n")
    assert git_state(git_repo)["dirty"] is False


def _frozen(repo: Path, tmp_path: Path, **extra) -> Path:
    protocol = seal_protocol({"stage": "inventory", "code_commit": git_state(repo)["commit"], "smoke": False, **extra})
    path = tmp_path / "frozen_protocol.json"
    path.write_text(json.dumps(protocol))
    return path


def test_test_phase_guard(git_repo, tmp_path):
    runs = tmp_path / "runs"
    # no frozen protocol
    assert run(git_repo, runs / "t0", ok, phase="test")[0] == 1
    frozen = _frozen(git_repo, tmp_path)
    # edited after freezing
    edited = json.loads(frozen.read_text())
    edited["stage"] = "tampered"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(edited))
    assert run(git_repo, runs / "t1", ok, phase="test", frozen=str(tampered))[0] == 1
    # first test run passes and sets the marker
    assert run(git_repo, runs / "t2", ok, phase="test", frozen=str(frozen))[0] == 0
    assert any((runs / ".test_used").iterdir())
    # resuming that same test run is allowed
    assert run(git_repo, runs / "t2", ok, phase="test", frozen=str(frozen), resume=True)[0] == 0
    # a second test run is refused without a reason ...
    code, lines = run(git_repo, runs / "t3", ok, phase="test", frozen=str(frozen))
    assert code == 1 and "REPORT=none" in lines[-1]
    # ... and allowed with one, which is recorded
    assert run(git_repo, runs / "t4", ok, phase="test", frozen=str(frozen), rerun_reason="rerun after disk failure")[0] == 0


def test_test_phase_refuses_other_commit_and_smoke(git_repo, tmp_path):
    frozen = _frozen(git_repo, tmp_path)
    (git_repo / "src" / "code.py").write_text("x = 3\n")
    commit_all(git_repo)
    assert run(git_repo, tmp_path / "runs" / "c1", ok, phase="test", frozen=str(frozen))[0] == 1
    smoke = _frozen(git_repo, tmp_path, smoke=True)
    assert run(git_repo, tmp_path / "runs" / "c2", ok, phase="test", frozen=str(smoke))[0] == 1
    assert run(git_repo, tmp_path / "runs" / "c3", ok, phase="dev", frozen=str(smoke))[0] == 1


def test_status_line_format():
    assert status_line("PASS", "runs/x", ["runs/x/SUMMARY.md", "runs/x/report.json"]) == (
        "STATUS=PASS REPORT=runs/x/SUMMARY.md SEND_BACK=runs/x/SUMMARY.md,runs/x/report.json"
    )
    with pytest.raises(ValueError):
        status_line("MAYBE", "runs/x", [])


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0


def test_contract_error_is_exception_type():
    assert issubclass(RunContractError, Exception)


def test_a_sealed_protocol_with_integer_keys_verifies_after_a_round_trip():
    # a YAML map with integer keys (configs/extensions/jacobian_variants.yaml other_layer) must hash as the string-keyed file written to disk
    import json

    from saeselect.provenance import verify_sealed_protocol

    sealed = seal_protocol({"config": {"other_layer": {8: 16, 16: 8, 27: 8}}})
    assert verify_sealed_protocol(json.loads(json.dumps(sealed))) == sealed["protocol_hash"]
