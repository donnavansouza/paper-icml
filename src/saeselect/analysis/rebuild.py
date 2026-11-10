"""``python -m saeselect analyze``: rebuild a run's analysis from its ``results.zip``.

    python -m saeselect analyze --run runs/selection-gemma-it-test --frozen runs/selection-gemma-it-dev/frozen_protocol.json \
        --out analysis/selection-gemma-it

CPU only: no model, no records. Stage 2 runs.

1. Provenance: the frozen protocol's seal; the run used that protocol (hash) on its commit; the hashes of the run's files;
   whether the source files the run loaded are the ones in this checkout.
2. Every Stage 2 table (signals, primary family with Holm, secondary contrasts, per-cell contrasts, sanity; one analysis
   per SAE variant for an extension) is recomputed from ``trials.jsonl`` with the stage's own functions and compared with
   the run's ``report.json`` (every value equal) and ``SUMMARY.md`` (the text regenerated from the recomputed tables is
   identical). A run whose numbers are not reproduced fails loudly (GATE_FAILED:reproduction).
3. The check against RANDOM: per signal, F at the primary k minus the mean F of the same
   trial's random draws (trials with m_FULL - m_clean at the M2 floor or above), averaged within pairs, cluster bootstrap
   within relation, three-way verdict against the M2 SESOI. Unsquared.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from saeselect import stats
from saeselect.provenance import REPO_ROOT, GateFailed, RunContractError, sha256_file, verify_sealed_protocol
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2

RUN_FILES = ("protocol.json", "report.json", "SUMMARY.md", "results.zip")
COMPARED = ("summary", "analysis", "sanity")
# the modules whose code computes the numbers; a difference from the run's copy is flagged
ANALYSIS_SOURCES = ("src/saeselect/stages/selection.py", "src/saeselect/stats.py", "src/saeselect/signals.py")


def read_run(run_dir: Path) -> dict[str, Any]:
    missing = [n for n in RUN_FILES if not (run_dir / n).exists()]
    if missing:
        raise RunContractError(f"{run_dir}: missing {missing}")
    with zipfile.ZipFile(run_dir / "results.zip") as archive:
        rows = [json.loads(line) for line in archive.read("trials.jsonl").decode("utf-8").splitlines() if line.strip()]
    return {"protocol": json.loads((run_dir / "protocol.json").read_text(encoding="utf-8")),
            "report": json.loads((run_dir / "report.json").read_text(encoding="utf-8")),
            "summary": (run_dir / "SUMMARY.md").read_text(encoding="utf-8").replace("\r\n", "\n"),
            "rows": rows, "hashes": {n: sha256_file(run_dir / n) for n in RUN_FILES}}


def compare(expected: Any, actual: Any, path: str = "") -> tuple[int, list[str]]:
    """Number of leaves compared and the mismatches; numbers must be equal (NaN equals NaN), nothing is rounded."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        n, bad = 0, []
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in expected or key not in actual:
                bad.append(f"{path}/{key}: in {'the report' if key in expected else 'the rebuild'} only")
                continue
            m, b = compare(expected[key], actual[key], f"{path}/{key}")
            n, bad = n + m, bad + b
        return n, bad
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return 1, [f"{path}: {len(expected)} items in the report, {len(actual)} in the rebuild"]
        n, bad = 0, []
        for i, (a, b) in enumerate(zip(expected, actual)):
            m, x = compare(a, b, f"{path}[{i}]")
            n, bad = n + m, bad + x
        return n, bad
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (expected, actual))
    if numbers:
        same = expected == actual or (math.isnan(expected) and math.isnan(actual))
    else:
        same = expected == actual
    return 1, [] if same else [f"{path}: report {expected!r}, rebuild {actual!r}"]


def random_baseline(rows: list[dict[str, Any]], cfg: dict[str, Any], *, pk: int, seed: int) -> dict[str, Any]:
    """Per signal: F at k = pk minus the mean F of the trial's random draws, pair means, cluster bootstrap within relation."""
    b_ = cfg["analysis"]["bootstrap"]
    out: dict[str, Any] = {}
    for name in sorted({n for r in rows for n in r["sets"]}):
        def value(r, name=name):
            if r["full_gain"] < cfg["m2_min_full_gain"]:
                return None
            f = r["sets"][name][str(pk)]["F"]
            draws = [d["F"] for d in r["random"][str(pk)] if d["F"] is not None]
            return None if f is None or not draws else f - float(np.mean(draws))

        values, _, strata = S2._cluster_means(rows, value)
        if len(values) < 2:
            out[name] = None
            continue
        res = stats.cluster_bootstrap(values, n_resamples=b_["n_resamples"], seed=seed, strata=strata)
        v = stats.verdict(res, sesoi=cfg["analysis"]["sesoi"]["M2"], predicted_sign=1, level=b_["level"], tost_level=b_["tost_level"])
        out[name] = {"estimate": v.estimate, "ci95": list(v.ci95), "ci90": list(v.ci90), "p": v.p_value, "label": v.label,
                     "n_pairs": len(values), "better": v.better, "equivalent": v.equivalent, "worse": v.worse}
    return json.loads(json.dumps(out, default=S2._jsonable))


def appendix_tables(rows: list[dict[str, Any]], cfg: dict[str, Any], *, pk: int, seed: int) -> dict[str, Any]:
    """Descriptive tables for the write-up (the paper's Appendix B), not part of any test or of the reproduction check:
    the primary contrasts (JL-OUT - LL-OUT on M1, and on M2 at the primary k) per relation, with the P2 procedure; and per
    signal the pair means of M1-abs (Spearman of |score| with |effect|), M3 at the primary k (tie-aware precision) and M6
    (partial Spearman controlling for NORM for KAPPA and JSPACE, for LL-OUT for JL-OUT). Unsquared."""
    out: dict[str, Any] = {"per_relation": {}, "secondary_metrics": {}}
    for rel in sorted({r["relation"] for r in rows}):
        sub = [r for r in rows if r["relation"] == rel]
        entry: dict[str, Any] = {"pairs": len({r["pair"] for r in sub}), "trials": len({r["trial"] for r in sub})}
        for metric, k, off in (("M1", None, 41), ("M2", pk, 42)):
            c = S2.contrast(sub, "JL-OUT", "LL-OUT", metric, k=k, cfg=cfg, seed=seed + off)
            if c is not None:
                c.pop("_verdict", None)
            entry[metric] = c
        out["per_relation"][rel] = entry
    for name in sorted({n for r in rows for n in r["metrics"]}):
        values = {}
        for key, get in (("M1_abs", lambda r, n=name: r["metrics"][n].get("M1_abs")),
                         ("M3", lambda r, n=name: (r["metrics"][n].get("M3") or {}).get(str(pk))),
                         ("M6", lambda r, n=name: r["metrics"][n].get("M6"))):
            v, _, _ = S2._cluster_means(rows, get)
            values[key] = float(np.mean(v)) if v else None
        out["secondary_metrics"][name] = values
    return json.loads(json.dumps(out, default=S2._jsonable))


def rebuild_stage2(report: dict[str, Any], rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """The run's tables recomputed from its trial rows: {summary, analysis, sanity}, per SAE variant for an extension."""
    pk = int(report["primary_k"])
    divergence = {int(k): v for k, v in report["divergence"].items()}
    seed = cfg["seed"]

    def one(sub: list[dict[str, Any]]) -> dict[str, Any]:
        summary = S2.summarize(sub, cfg, pk=pk, seed=seed)
        out = {"summary": summary, "analysis": S2.analyze(sub, cfg, pk=pk, divergence=divergence, seed=seed),
               "sanity": S2.sanity_checks(summary, pk)}
        return json.loads(json.dumps(out, default=S2._jsonable))

    if "by_sae" in report:
        return {"by_sae": {tag: one([r for r in rows if r["sae"] == tag]) for tag in sorted({r["sae"] for r in rows})}}
    return one(rows)


def _reported(report: dict[str, Any]) -> dict[str, Any]:
    if "by_sae" in report:
        return {"by_sae": {tag: {k: e[k] for k in COMPARED} for tag, e in report["by_sae"].items()}}
    return {k: report[k] for k in COMPARED}


def _merged(report: dict[str, Any], rebuilt: dict[str, Any]) -> dict[str, Any]:
    if "by_sae" in report:
        return {**report, "by_sae": {tag: {**report["by_sae"][tag], **rebuilt["by_sae"].get(tag, {})} for tag in report["by_sae"]}}
    return {**report, **rebuilt}


def _frozen(ctx: RunContext, run_dir: Path) -> tuple[dict[str, Any], Path]:
    path = Path(ctx.options["frozen"]) if ctx.options.get("frozen") else run_dir / "frozen_protocol.json"
    if not path.exists():
        raise RunContractError(f"no frozen protocol: pass --frozen <dev run>/frozen_protocol.json for a test run ({path} not found)")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    verify_sealed_protocol(frozen)
    return frozen, path


def provenance(run: dict[str, Any], frozen: dict[str, Any], frozen_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Checks that fail the command, and differences that are flagged."""
    protocol, report = run["protocol"], run["report"]
    phase = protocol.get("phase")
    errors = []
    if protocol.get("command") != "select":
        errors.append(f"the run is a '{protocol.get('command')}' run; analyze rebuilds Stage 2 runs")
    if phase == "test" and protocol.get("frozen_protocol_hash") != frozen["protocol_hash"]:
        errors.append(f"the run used frozen protocol {protocol.get('frozen_protocol_hash')}, not {frozen['protocol_hash']} ({frozen_path})")
    if protocol.get("code_commit") != frozen.get("code_commit"):
        errors.append(f"the run's commit {protocol.get('code_commit')} is not the frozen commit {frozen.get('code_commit')}")
    if protocol.get("code_dirty") or protocol.get("smoke"):
        errors.append("the run had uncommitted code or was a SMOKE run")
    if str(report.get("status", "")).split(":")[0] not in ("PASS", "PASS_WITH_FLAGS", "FROZEN_AWAITING_APPROVAL"):
        errors.append(f"the run's status is {report.get('status')}")
    if errors:
        raise RunContractError("; ".join(errors))
    loaded = report.get("source_hashes", {})
    differs = sorted(rel for rel, h in loaded.items() if (REPO_ROOT / rel).exists() and sha256_file(REPO_ROOT / rel) != h)
    out = {"phase": phase, "run_files_sha256": run["hashes"], "frozen_protocol": str(frozen_path).replace("\\", "/"),
           "frozen_protocol_hash": frozen["protocol_hash"], "code_commit": protocol.get("code_commit"),
           "data_sha256": frozen.get("data"), "sources_loaded": len(loaded), "sources_differing_from_this_checkout": differs}
    flags = []
    if phase != "test":
        flags.append(f"a {phase} run: its numbers are estimates, not claims")
    hit = [rel for rel in differs if rel in ANALYSIS_SOURCES]
    if hit:
        flags.append(f"analysis code differs from the run's copy: {hit} (reproduction is still checked)")
    return out, flags


def run(ctx: RunContext) -> Outcome:
    if not ctx.options.get("run"):
        raise RunContractError("analyze needs --run <run directory>")
    run_dir = Path(ctx.options["run"])
    data = read_run(run_dir)
    frozen, frozen_path = _frozen(ctx, run_dir)
    prov, flags = provenance(data, frozen, frozen_path)
    ctx.flags.extend(flags)
    cfg = frozen.get("config_resolved") or S2._resolve(frozen["config"])
    report = data["report"]
    rebuilt = rebuild_stage2(report, data["rows"], cfg)
    n_leaves, mismatches = compare(_reported(report), rebuilt)
    fake = SimpleNamespace(phase=data["protocol"]["phase"], config=cfg)
    text = (S2._extension_summary if "by_sae" in report else S2._summary)(fake, _merged(report, rebuilt))
    summary_identical = text.rstrip() in data["summary"]
    pk = int(report["primary_k"])
    if "by_sae" in report:
        baseline = {tag: random_baseline([r for r in data["rows"] if r["sae"] == tag], cfg, pk=pk, seed=cfg["seed"]) for tag in rebuilt["by_sae"]}
    else:
        baseline = random_baseline(data["rows"], cfg, pk=pk, seed=cfg["seed"])
    if "by_sae" in report:
        appendix = {tag: appendix_tables([r for r in data["rows"] if r["sae"] == tag], cfg, pk=pk, seed=cfg["seed"]) for tag in rebuilt["by_sae"]}
    else:
        appendix = appendix_tables(data["rows"], cfg, pk=pk, seed=cfg["seed"])
    out = {"run": str(run_dir).replace("\\", "/"), "provenance": prov,
           "reproduction": {"leaves_compared": n_leaves, "mismatches": mismatches, "summary_identical": summary_identical},
           "rebuilt": rebuilt, "random_baseline": baseline, "appendix": appendix, "primary_k": pk, "n_trials": report.get("n_trials"),
           "n_pairs": report.get("n_pairs")}
    if mismatches or not summary_identical:
        reason = (f"{len(mismatches)} of {n_leaves} values differ from report.json" if mismatches else "every report.json value is reproduced")
        reason += "" if summary_identical else "; the SUMMARY.md text regenerated from the rebuilt tables differs from the run's"
        raise GateFailed("reproduction", reason, mismatches[:20] or [text[:3000]], report=out)
    return Outcome(status="PASS", summary=_summary(out, cfg), report=out)


def _ci(c: dict[str, Any] | None) -> str:
    if not c:
        return "not testable"
    return f"{c['estimate']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {c['label']}" + (
        f", Holm p {c['p_holm']:.3g}" if c.get("p_holm") is not None else "")


def _tables(lines: list[str], entry: dict[str, Any], baseline: dict[str, Any], pk: int) -> None:
    a = entry["analysis"]
    lines += ["| Contrast | Estimate [95%], verdict |", "|---|---|"]
    lines += [f"| {k} | {_ci(v)} |" for k, v in a["primary"].items()]
    lines += [f"| JL-OUT - LL-OUT (M1), L{layer}/{site} | {_ci(v)} |" for layer, per in a["H3_per_layer_site"].items() for site, v in per.items() if v]
    lines += [f"| {k} | {_ci(v)} |" for k, v in a["secondary"].items() if v]
    lines += ["", f"Against RANDOM: F at k = {pk} minus the mean F of the trial's random draws (pair means, unsquared):", "",
              "| Signal | Estimate [95%], verdict | Pairs |", "|---|---|---|"]
    lines += [f"| {n} | {_ci(v)} | {v['n_pairs']} |" for n, v in baseline.items() if v]
    lines += ["", f"RANDOM mean F at k = {pk}: {entry['summary']['random_mean_F_at_primary_k']:.3f}; sanity violations: "
              f"{entry['sanity']['violations'] or 'none'}", ""]


def _appendix(lines: list[str], appx: dict[str, Any], pk: int) -> None:
    lines += ["Appendix tables (descriptive; the paper's Appendix B). JL-OUT - LL-OUT per relation:", "",
              f"| Relation | Pairs | M1 | M2 at k = {pk} |", "|---|---|---|---|"]
    lines += [f"| {rel} | {e['pairs']} | {_ci(e['M1'])} | {_ci(e['M2'])} |" for rel, e in appx["per_relation"].items()]
    lines += ["", f"Secondary metrics per signal (pair means): M1-abs, M3 at k = {pk}, M6:", "", "| Signal | M1-abs | M3 | M6 |", "|---|---|---|---|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    lines += [f"| {n} | {fmt(v['M1_abs'])} | {fmt(v['M3'])} | {fmt(v['M6'])} |" for n, v in appx["secondary_metrics"].items()]
    lines.append("")


def _summary(out: dict[str, Any], cfg: dict[str, Any]) -> str:
    p, r = out["provenance"], out["reproduction"]
    lines = ["## What ran", "",
             f"Rebuild of `{out['run']}` ({p['phase']} phase) from its results.zip, with the frozen protocol `{p['frozen_protocol']}` "
             f"({p['frozen_protocol_hash'][:12]}..., seal verified; the run used it on commit {p['code_commit']}). CPU only.", "",
             "## Reproduction", "",
             f"- report.json: {r['leaves_compared']} values compared, {len(r['mismatches'])} differ.",
             f"- SUMMARY.md: the text regenerated from the rebuilt tables is {'identical' if r['summary_identical'] else 'different'}.",
             f"- Files: " + ", ".join(f"{n} sha256 {h[:16]}..." for n, h in p["run_files_sha256"].items()),
             f"- Source files the run loaded: {p['sources_loaded']}; differing from this checkout: {p['sources_differing_from_this_checkout'] or 'none'}.",
             f"- Data (frozen protocol): " + ", ".join(f"{f} {h[:16]}..." for f, h in (p["data_sha256"] or {}).items()), ""]
    pk = out["primary_k"]
    if "by_sae" in out["rebuilt"]:
        for tag, entry in out["rebuilt"]["by_sae"].items():
            lines += [f"## SAE variant {tag}", ""]
            _tables(lines, entry, out["random_baseline"][tag], pk)
            _appendix(lines, out["appendix"][tag], pk)
    else:
        lines += ["## Tables", ""]
        _tables(lines, out["rebuilt"], out["random_baseline"], pk)
        _appendix(lines, out["appendix"], pk)
    lines += ["## What this permits", "",
              "Every number above is regenerated from the run's results.zip by this command. Verdicts are three-way (5.6); inconclusive is never 'no difference'.", "",
              "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
