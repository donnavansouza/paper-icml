"""``python -m saeselect compare-sets``: set recovery compared across models .

    python -m saeselect compare-sets --config configs/analysis/compare_sets.yaml --out analysis/compare-sets

CPU only. Reads Stage 2 test runs' ``results.zip``. Each run is checked against its sealed frozen protocol, as in
``analyze``, and analysed with its own frozen config.

Per model:
1. JL-OUT minus LL-OUT on M2 at every k of the frozen grid, with the P2.2 procedure (``selection.contrast``):
   pair means over trials at or above the M2 full-gain floor; cluster bootstrap within relation; three-way verdict
   against the frozen M2 SESOI; Holm across k within the model.
2. The same on the matched strata: the declared depth-matched concept cells and the shared relations.
3. Per stratum and k: the mean F of JL-OUT, LL-OUT, NORM, ORACLE (ceiling) and of the trial's random draws (floor),
   with the number of pairs. The M1 contrast (no k) is reported as description.

Across models, in the matched strata, at the declared k:
- the statistic is (model JL - LL) - (reference JL - LL), on pair means;
- each model is bootstrapped within relation, independently (the pairs are different items);
- the result is the difference of the two bootstrap distributions, given a three-way verdict against the M2 SESOI.

Each model carries a status. "post hoc" means its test result was seen before the cross-model comparison was proposed. "declared before
data" means its test ran after the cross-model comparison (X3). Descriptive only: the X4 variants at their cell, per k.
Unsquared throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from saeselect import stats
from saeselect.analysis import rebuild as R
from saeselect.stats import sub_seed as _seed
from saeselect.provenance import RunContractError, verify_sealed_protocol
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2

LEVEL_SIGNALS = ("JL-OUT", "LL-OUT", "NORM", "ORACLE")


def _flag(flag: str) -> str:
    """``analyze`` flags say reproduction is checked there; compare-sets does not re-check it."""
    return flag.replace("(reproduction is still checked)", "(compare-sets does not re-check reproduction; see the run's `saeselect analyze` rebuild)")


def load(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    """Trial rows, the frozen (resolved) config, provenance and flags of one Stage 2 run."""
    run_dir, frozen_path = Path(entry["run"]), Path(entry["frozen"])
    data = R.read_run(run_dir)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    verify_sealed_protocol(frozen)
    prov, flags = R.provenance(data, frozen, frozen_path)
    prov["primary_k"] = frozen.get("primary_k")
    return data["rows"], frozen.get("config_resolved") or S2._resolve(frozen["config"]), prov, [_flag(f) for f in flags]


def matched(rows: list[dict[str, Any]], cells: list[str], relations: list[str]) -> list[dict[str, Any]]:
    return [r for r in rows if f"{r['layer']}/{r['site']}" in cells and r["relation"] in relations]


def per_k(rows: list[dict[str, Any]], cfg: dict[str, Any], *, ks: list[int], holm_alpha: float, seed: int) -> dict[str, Any]:
    """JL-OUT - LL-OUT on M2 at each k (P2.2 procedure), Holm across k."""
    out: dict[str, Any] = {}
    verdicts = {}
    for k in ks:
        c = S2.contrast(rows, "JL-OUT", "LL-OUT", "M2", k=k, cfg=cfg, seed=_seed(seed, "M2", k))
        out[str(k)] = c
        if c is not None:
            verdicts[str(k)] = c["_verdict"]
    if verdicts:
        stats.apply_holm(verdicts, alpha=holm_alpha)
    for k, c in out.items():
        if c is not None:
            v = c.pop("_verdict")
            c.update(p_holm=v.p_adjusted, better_after_holm=v.better_after_holm)
    return out


def levels(rows: list[dict[str, Any]], cfg: dict[str, Any], *, ks: list[int]) -> dict[str, Any]:
    """Mean F (pair means) per signal and k, the random draws' mean F, and the pairs counted."""
    floor = cfg["m2_min_full_gain"]
    out = {}
    for k in ks:
        row = {}
        for name in LEVEL_SIGNALS:
            vals, _, _ = S2._cluster_means(rows, lambda r, n=name: r["sets"][n][str(k)]["F"] if r["full_gain"] >= floor else None)
            row[name] = float(np.mean(vals)) if vals else None
        rnd, pairs, _ = S2._cluster_means(rows, lambda r: (float(np.mean([d["F"] for d in r["random"][str(k)] if d["F"] is not None]))
                                                           if r["full_gain"] >= floor and any(d["F"] is not None for d in r["random"][str(k)]) else None))
        row["RANDOM"] = float(np.mean(rnd)) if rnd else None
        row["pairs"] = len(pairs)
        out[str(k)] = row
    return out


def _pair_diffs(rows: list[dict[str, Any]], cfg: dict[str, Any], k: int) -> tuple[list[float], list[str]]:
    floor = cfg["m2_min_full_gain"]

    def value(r):
        if r["full_gain"] < floor:
            return None
        x, y = r["sets"]["JL-OUT"][str(k)]["F"], r["sets"]["LL-OUT"][str(k)]["F"]
        return None if x is None or y is None else x - y

    values, _, strata = S2._cluster_means(rows, value)
    return values, strata


def cross(model_rows: list[dict[str, Any]], model_cfg: dict[str, Any], ref_rows: list[dict[str, Any]], ref_cfg: dict[str, Any], *,
          k: int, seed: int, sesoi: float | None = None) -> dict[str, Any] | None:
    """(model JL - LL) - (reference JL - LL) at k, independent bootstraps within relation, three-way against the M2 SESOI
    (the reference's frozen one unless ``sesoi`` is given: the SESOI sensitivity analysis)."""
    vm, sm = _pair_diffs(model_rows, model_cfg, k)
    vr, sr = _pair_diffs(ref_rows, ref_cfg, k)
    if len(vm) < 2 or len(vr) < 2:
        return None
    b = ref_cfg["analysis"]["bootstrap"]
    bm = stats.cluster_bootstrap(vm, n_resamples=b["n_resamples"], seed=_seed(seed, "model", k), strata=sm)
    br = stats.cluster_bootstrap(vr, n_resamples=b["n_resamples"], seed=_seed(seed, "ref", k), strata=sr)
    diff = stats.BootstrapResult(estimate=bm.estimate - br.estimate, samples=bm.samples - br.samples)
    v = stats.verdict(diff, sesoi=ref_cfg["analysis"]["sesoi"]["M2"] if sesoi is None else sesoi, predicted_sign=1, level=b["level"], tost_level=b["tost_level"])
    return {"estimate": v.estimate, "ci95": list(v.ci95), "ci90": list(v.ci90), "p": v.p_value, "label": v.label, "better": v.better,
            "practically_better": v.practically_better, "equivalent": v.equivalent, "worse": v.worse, "model_mean": bm.estimate,
            "reference_mean": br.estimate, "pairs": [len(vm), len(vr)], "_verdict": v}


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    if not cfg.get("models") or (not cfg.get("contrasts") and cfg.get("reference") not in cfg["models"]):
        raise RunContractError("compare-sets needs `models` and either `contrasts` or a `reference` among them (configs/analysis/compare_sets.yaml)")
    for c in cfg.get("contrasts") or []:
        if c["model"] not in cfg["models"] or c["reference"] not in cfg["models"]:
            raise RunContractError(f"contrast {c['name']}: {c['model']} and {c['reference']} must both be declared models")
    loaded = {}
    report: dict[str, Any] = {"models": {}, "cross_model": {}, "descriptive": {}, "matched_relations": cfg["matched_relations"]}
    for name, entry in cfg["models"].items():
        rows, run_cfg, prov, flags = load(entry)
        ctx.flags.extend(f"{name}: {f}" for f in flags)
        ks = list(run_cfg["k_grid"])
        sub = matched(rows, entry["matched_cells"], cfg["matched_relations"])
        if not sub:
            raise RunContractError(f"{name}: no trial in the matched strata {entry['matched_cells']} x {cfg['matched_relations']}")
        seed = _seed(cfg["seed"], name)
        report["models"][name] = {
            "status": entry["status"], "provenance": prov, "k_grid": ks, "primary_k": prov["primary_k"],
            "matched_cells": entry["matched_cells"],
            "all_strata": {"M2": per_k(rows, run_cfg, ks=ks, holm_alpha=cfg["holm_alpha"], seed=seed), "levels": levels(rows, run_cfg, ks=ks)},
            "matched": {"M2": per_k(sub, run_cfg, ks=ks, holm_alpha=cfg["holm_alpha"], seed=_seed(seed, "matched")), "levels": levels(sub, run_cfg, ks=ks),
                        "M1": _m1(sub, run_cfg, seed)},
        }
        loaded[name] = (sub, run_cfg)
    contrasts = cfg.get("contrasts") or [{"name": f"{n} - {cfg['reference']}", "model": n, "reference": cfg["reference"]}
                                          for n in loaded if n != cfg["reference"]]
    for c in contrasts:
        (sub, run_cfg), (ref_rows, ref_cfg) = loaded[c["model"]], loaded[c["reference"]]
        seed_name = c["model"] if not cfg.get("contrasts") else c["name"]
        report["cross_model"][c["name"]] = {str(k): cross(sub, run_cfg, ref_rows, ref_cfg, k=k, seed=_seed(cfg["seed"], "cross", seed_name))
                                            for k in cfg["cross_model_k"]}
    if cfg.get("holm_k") is not None:  # the declared contrasts form one family at this k
        family = {name: by_k[str(cfg["holm_k"])]["_verdict"] for name, by_k in report["cross_model"].items() if by_k.get(str(cfg["holm_k"]))}
        stats.apply_holm(family, alpha=cfg["holm_alpha"])
        for name, v in family.items():
            report["cross_model"][name][str(cfg["holm_k"])].update(p_holm=v.p_adjusted, better_after_holm=v.better_after_holm)
    for by_k in report["cross_model"].values():
        for c in by_k.values():
            if c:
                c.pop("_verdict", None)
    for name, entry in (cfg.get("descriptive") or {}).items():
        rows, run_cfg, prov, flags = load(entry)
        ctx.flags.extend(f"{name}: {f}" for f in flags)
        ks = list(run_cfg["k_grid"])
        report["descriptive"][name] = {tag: {"M2": per_k([r for r in rows if r.get("sae") == tag], run_cfg, ks=ks, holm_alpha=cfg["holm_alpha"],
                                                         seed=_seed(cfg["seed"], name, tag)),
                                             "levels": levels([r for r in rows if r.get("sae") == tag], run_cfg, ks=ks)}
                                       for tag in sorted({r.get("sae") for r in rows if r.get("sae")})} or {
            "all": {"M2": per_k(rows, run_cfg, ks=ks, holm_alpha=cfg["holm_alpha"], seed=_seed(cfg["seed"], name)), "levels": levels(rows, run_cfg, ks=ks)}}
    report = json.loads(json.dumps(report, default=S2._jsonable))
    return Outcome(status="PASS", summary=_summary(report, cfg), report=report)


def _m1(rows: list[dict[str, Any]], cfg: dict[str, Any], seed: int) -> dict[str, Any] | None:
    c = S2.contrast(rows, "JL-OUT", "LL-OUT", "M1", k=None, cfg=cfg, seed=_seed(seed, "M1"))
    if c is not None:
        c.pop("_verdict", None)
    return c


def _ci(c: dict[str, Any] | None) -> str:
    if not c:
        return "not estimable"
    holm = f", Holm p {c['p_holm']:.3g}" if c.get("p_holm") is not None else ""
    return f"{c['estimate']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {c['label']}{holm}"


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _summary(report: dict[str, Any], cfg: dict[str, Any]) -> str:
    lines = ["## What ran", "",
             "The cross-model comparison: JL-OUT minus LL-OUT on M2 (F_k, unsquared) at every k, per model, on all strata and on the matched strata "
             f"(depth-matched concept cells; relations {', '.join(cfg['matched_relations'])}); Holm across k within each model; the "
             "cross-model difference with independent bootstraps. Results for a model whose test was seen before the cross-model comparison are post hoc.", ""]
    for name, m in report["models"].items():
        lines += [f"## {name} ({m['status']}; matched cells {', '.join(m['matched_cells'])})", "",
                  "| k | All strata | Matched strata | Matched: ORACLE / RANDOM / JL / LL / NORM F (pairs) |", "|---|---|---|---|"]
        for k in m["k_grid"]:
            lv = m["matched"]["levels"][str(k)]
            lines.append(f"| {k} | {_ci(m['all_strata']['M2'][str(k)])} | {_ci(m['matched']['M2'][str(k)])} | "
                         f"{_f(lv['ORACLE'])} / {_f(lv['RANDOM'])} / {_f(lv['JL-OUT'])} / {_f(lv['LL-OUT'])} / {_f(lv['NORM'])} ({lv['pairs']}) |")
        lines += ["", f"M1 (ranking) in the matched strata: {_ci(m['matched']['M1'])}", ""]
    if report["cross_model"]:
        lines += ["## Cross-model difference in the matched strata", "", "| Contrast | k | Difference [95%], verdict | Model / reference JL - LL | Pairs |",
                  "|---|---|---|---|---|"]
        for name, by_k in report["cross_model"].items():
            for k, c in by_k.items():
                lines.append(f"| {name} | {k} | {_ci(c)} | " + (f"{c['model_mean']:+.3f} / {c['reference_mean']:+.3f} | {c['pairs'][0]} / {c['pairs'][1]} |" if c else "n/a | n/a |"))
        if cfg.get("holm_k") is not None:
            lines += ["", f"Holm across the declared contrasts at k = {cfg['holm_k']} (alpha {cfg['holm_alpha']}); other k are secondary."]
        lines.append("")
    for name, variants in report["descriptive"].items():
        lines += [f"## Descriptive: {name}", "", "| Variant | k | JL - LL on M2 | ORACLE / RANDOM F |", "|---|---|---|---|"]
        for tag, d in variants.items():
            for k, c in d["M2"].items():
                lv = d["levels"][k]
                lines.append(f"| {tag} | {k} | {_ci(c)} | {_f(lv['ORACLE'])} / {_f(lv['RANDOM'])} |")
        lines.append("")
    lines += ["## What this permits", "",
              "Post hoc results are exploratory: they describe the "
              "disparity and its floor and ceiling, they do not test it. Verdicts are three-way; inconclusive is never 'no difference'.", "",
              "## Next step", "", "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
