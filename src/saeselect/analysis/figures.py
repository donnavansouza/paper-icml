"""``python -m saeselect figures``: the paper's figures, drawn from the runs' raw files (descriptive).

    python -m saeselect figures --config configs/analysis/figures.yaml --out analysis/figures

CPU only. Every Stage 2 run is read from its ``results.zip`` and checked against its sealed frozen protocol, as in
compare-sets, and analysed with its own frozen config. Every number drawn is written to ``figures_data.json`` next to
the PDFs, so each figure traces to its runs. Nothing here is a test: intervals
are 95% cluster bootstraps within relation with the runs' own settings, and bands are the frozen M2 SESOI.

The figures (configs/analysis/figures.yaml):
- ``fk``: per checkpoint, F_k of JL-OUT, LL-OUT, NORM, ORACLE and RANDOM against k (top row), and JL-OUT - LL-OUT
  with its 95% interval against the +/-SESOI band (bottom row). All eligible cells; the compare-sets procedure and
  seeds, so the contrasts equal Appendix B.1.
- ``dissociation``: per checkpoint, each signal's mean M1 (Spearman with single-feature effects) against its mean F
  at the primary k, both pair means with 95% intervals; RANDOM's mean F as a line.
- ``superadditivity``: per trial at one cell and k, the additive prediction (the selected set's summed single-feature
  effects over the full gain) against the recovered share F, for the declared selectors. y = x is additivity.
- ``swap``: the SAE swap's 2 x 2, JL-OUT - LL-OUT on M2 at k in the matched strata with 95% intervals, from the
  compare-sets report.
- ``replication``: the ranking-versus-sets test's family per task and checkpoint, D1 (ATP - JL-OUT on M1) and D2 (the same on F at the primary
  k) with 95% intervals against the +/-SESOI band, from the new-task test reports.
- ``decomposition``: dF = dU + dN for one pair at the primary k, pooled per run, from the decomposition report.
- ``jlens``: JL-OUT - LL-OUT on M1 (P2.1) and on F at the primary k (P2.2) with 95% intervals, the factual relations
  pooled from the Stage 2 test reports and the new tasks per task from the new-task test reports (the ranking-versus-sets test's secondary).
Unsquared throughout.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from saeselect import stats
from saeselect.analysis import compare_sets as C
from saeselect.stats import sub_seed as _seed
from saeselect.provenance import RunContractError
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2

FIGURES = ("fk", "dissociation", "superadditivity", "swap", "replication", "decomposition", "jlens")
FULL_WIDTH, COLUMN_WIDTH = 6.75, 3.25  # inches, ICML text and column widths

# Okabe-Ito colours (colour-blind safe); one marker per signal, so the legend carries the identity
STYLE = {
    "JL-OUT": ("#0072B2", "o", "JL-OUT (J-lens)"),
    "LL-OUT": ("#D55E00", "s", "LL-OUT (logit lens)"),
    "NORM": ("#555555", "^", "NORM"),
    "ASSOC": ("#56B4E9", "v", "ASSOC"),
    "ATP": ("#009E73", "D", "ATP"),
    "ATP@in": ("#7FCEB3", "d", "ATP@in"),
    "PATCH": ("#CC79A7", "P", "PATCH"),
    "KAPPA": ("#E69F00", "X", "KAPPA"),
    "JSPACE": ("#A07800", "*", "JSPACE"),
    "ORACLE": ("#000000", "h", "ORACLE"),
    "RANDOM": ("#999999", "", "RANDOM"),
}


# ------------------------------------------------------------------------------------------------ data

def mean_ci(rows: list[dict[str, Any]], value, cfg: dict[str, Any], *, seed: int) -> dict[str, Any] | None:
    """Pair mean of ``value`` with its 95% cluster-bootstrap interval within relation (the run's settings)."""
    vals, pairs, strata = S2._cluster_means(rows, value)
    if len(vals) < 2:
        return None
    b = cfg["analysis"]["bootstrap"]
    res = stats.cluster_bootstrap(vals, n_resamples=b["n_resamples"], seed=seed, strata=strata)
    lo, hi = res.ci(b["level"])
    return {"mean": float(res.estimate), "ci95": [lo, hi], "pairs": len(pairs)}


def fk_data(rows: list[dict[str, Any]], cfg: dict[str, Any], *, ks: list[int], seed: int, holm_alpha: float) -> dict[str, Any]:
    """Levels and the JL-OUT - LL-OUT contrast per k: compare-sets' all-strata procedure (Appendix B.1)."""
    return {"levels": C.levels(rows, cfg, ks=ks), "contrast": C.per_k(rows, cfg, ks=ks, holm_alpha=holm_alpha, seed=seed),
            "sesoi": cfg["analysis"]["sesoi"]["M2"]}


def dissociation_data(rows: list[dict[str, Any]], cfg: dict[str, Any], *, signals: list[str], pk: int, seed: int) -> dict[str, Any]:
    """Per signal: mean M1 and mean F at the primary k (pair means, 95% intervals); RANDOM's mean F."""
    floor = cfg["m2_min_full_gain"]
    out: dict[str, Any] = {"primary_k": pk, "signals": {}}
    for name in signals:
        m1 = mean_ci(rows, lambda r, n=name: (r["metrics"].get(n) or {}).get("M1"), cfg, seed=_seed(seed, name, "M1"))
        f = mean_ci(rows, lambda r, n=name: r["sets"][n][str(pk)]["F"] if r["full_gain"] >= floor and n in r["sets"] else None,
                    cfg, seed=_seed(seed, name, "F"))
        out["signals"][name] = {"M1": m1, "F": f}
    out["RANDOM_F"] = C.levels(rows, cfg, ks=[pk])[str(pk)]["RANDOM"]
    return out


def superadditivity_data(rows: list[dict[str, Any]], cfg: dict[str, Any], *, cell: str, k: int, selectors: list[str]) -> dict[str, Any]:
    """Per trial in ``cell`` above the M2 floor: x = (sum of the selected features' single effects) / full gain, the
    additive prediction of F; y = F. A selected feature outside the evaluation pool contributes 0 (5.4, note 4)."""
    floor = cfg["m2_min_full_gain"]
    out: dict[str, Any] = {"cell": cell, "k": k, "selectors": {}}
    sub = [r for r in rows if f"{r['layer']}/{r['site']}" == cell and r["full_gain"] >= floor]
    for name in selectors:
        xs, ys, summed, joint = [], [], [], []
        for r in sub:
            s = r["sets"][name][str(k)]
            if s["F"] is None:
                continue
            total = float(sum(r["effects"].get(str(i), 0.0) for i in s["selected"]))
            xs.append(total / r["full_gain"])
            ys.append(float(s["F"]))
            summed.append(total)
            joint.append(float(s["F"]) * r["full_gain"])
        out["selectors"][name] = {
            "x": xs, "y": ys, "trials": len(xs),
            "median_x": float(np.median(xs)) if xs else None, "median_y": float(np.median(ys)) if ys else None,
            "mean_summed_effect": float(np.mean(summed)) if summed else None, "mean_set_effect": float(np.mean(joint)) if joint else None,
            "share_above_diagonal": float(np.mean([y > x for x, y in zip(xs, ys)])) if xs else None,
        }
    out["mean_full_gain"] = float(np.mean([r["full_gain"] for r in sub])) if sub else None
    return out


def swap_data(report: dict[str, Any], cells: dict[str, dict[str, str]], *, k: int) -> dict[str, Any]:
    """The 2 x 2 from a compare-sets report: weights x SAE release -> JL - LL on M2 at k in the matched strata."""
    out: dict[str, Any] = {"k": k, "cells": {}}
    for weights, by_sae in cells.items():
        for sae, key in by_sae.items():
            if key not in report["models"]:
                raise RunContractError(f"swap: {key} is not a model of the compare-sets report")
            c = report["models"][key]["matched"]["M2"].get(str(k))
            if not c:
                raise RunContractError(f"swap: {key} has no matched M2 contrast at k = {k}")
            out["cells"][f"{weights}|{sae}"] = {"model": key, "estimate": c["estimate"], "ci95": c["ci95"], "label": c["label"],
                                                "pairs": report["models"][key]["matched"]["levels"][str(k)]["pairs"]}
    out["contrasts"] = {name: {"estimate": by_k[str(k)]["estimate"], "ci95": by_k[str(k)]["ci95"], "label": by_k[str(k)]["label"],
                               "p_holm": by_k[str(k)].get("p_holm")}
                        for name, by_k in report.get("cross_model", {}).items() if by_k.get(str(k))}
    return out


TASKS = {"factual": "factual", "R5_number_agreement": "agreement", "R6_film_sentiment": "sentiment"}


def replication_data(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The ranking-versus-sets family per task and checkpoint, as each new-task test report states it: D1 (ATP - JL-OUT on
    M1) and D2 (the same on F at the primary k), estimates, 95% intervals and verdicts, with the run's replication."""
    rows = []
    for label, rep in reports.items():
        rvs = rep.get("ranking_vs_sets") or {}
        if not rvs.get("dissociation") or not rvs.get("replication"):
            raise RunContractError(f"replication: {label} has no ranking-versus-sets family")
        for task, verdict in sorted(rvs["replication"].items()):
            row = {"model": label, "task": task, "k": int(rep["primary_k"]), "replicates": bool(verdict["replicates"])}
            for name in ("D1", "D2"):
                c = rvs["dissociation"][f"{name} {task}"]
                row[name] = {"estimate": c["estimate"], "ci95": c["ci95"], "label": c["label"]}
            rows.append(row)
    return rows


def jlens_data(factual: dict[str, dict[str, Any]], new_tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The J-lens against its twin as the reports state it, per checkpoint: P2.1 and P2.2 pooled over the factual relations
    (the Stage 2 primary family), then per new task (the ranking-versus-sets test's secondary per-task P2.1 and P2.2)."""
    def row(label: str, task: str, k: int, fam: dict[str, Any]) -> dict[str, Any]:
        if "P2.1" not in fam or "P2.2" not in fam:
            raise RunContractError(f"jlens: {label} {task} has no P2.1 and P2.2")
        return {"model": label, "task": task, "k": k,
                **{n: {"estimate": fam[n]["estimate"], "ci95": fam[n]["ci95"], "label": fam[n]["label"]} for n in ("P2.1", "P2.2")}}

    rows = []
    for label in dict.fromkeys([*factual, *new_tasks]):
        if label in factual:
            rep = factual[label]
            rows.append(row(label, "factual", int(rep["primary_k"]), (rep.get("analysis") or {}).get("primary") or {}))
        if label in new_tasks:
            rep = new_tasks[label]
            per = (rep.get("ranking_vs_sets") or {}).get("per_task_primary") or {}
            if not per:
                raise RunContractError(f"jlens: {label} has no per-task P2.1 and P2.2")
            rows += [row(label, task, int(rep["primary_k"]), fam) for task, fam in sorted(per.items())]
    return rows


def decomposition_data(report: dict[str, Any], runs: dict[str, str], pair: str) -> list[dict[str, Any]]:
    """The decomposition of one pair at the primary k, pooled per run: dF, dU (singleton utility), dN (remainder)."""
    out = []
    for key, label in runs.items():
        m = (report.get("runs") or {}).get(key)
        if m is None or pair not in m["decomposition"].get("pooled", {}):
            raise RunContractError(f"decomposition: {key} or its pair {pair!r} is not in the decomposition report")
        e = m["decomposition"]["pooled"][pair]
        out.append({"run": key, "label": label, "k": int(m["primary_k"]), "reading": e["reading"],
                    **{q: {"estimate": e[q]["estimate"], "ci95": e[q]["ci95"]} for q in ("F", "U", "N")}})
    return out


# ------------------------------------------------------------------------------------------------ drawing

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = [f for f in ("Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif") if f in available]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": serif or ["DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
    })
    return plt


def _sesoi_band(ax, sesoi: float) -> None:
    ax.axhspan(-sesoi, sesoi, color="#DDDDDD", alpha=0.6, lw=0, zorder=0)
    ax.axhline(0, color="#777777", lw=0.6, zorder=1)


def draw_fk(plt, data: dict[str, Any], labels: dict[str, str]):
    models = list(data)
    fig, axes = plt.subplots(2, len(models), figsize=(FULL_WIDTH, 3.1), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.12, "wspace": 0.12})
    axes = np.atleast_2d(axes)
    for j, name in enumerate(models):
        d = data[name]
        ks = sorted(int(k) for k in d["levels"])
        top, bottom = axes[0, j], axes[1, j]
        for sig in ("ORACLE", "NORM", "LL-OUT", "JL-OUT", "RANDOM"):
            colour, marker, label = STYLE[sig]
            ys = [d["levels"][str(k)][sig] for k in ks]
            style = {"ORACLE": "--", "RANDOM": ":"}.get(sig, "-")
            top.plot(ks, ys, style, color=colour, marker=marker or None, ms=3, label=label)
        top.set_title(labels[name])
        est = np.array([d["contrast"][str(k)]["estimate"] for k in ks])
        lo = np.array([d["contrast"][str(k)]["ci95"][0] for k in ks])
        hi = np.array([d["contrast"][str(k)]["ci95"][1] for k in ks])
        _sesoi_band(bottom, d["sesoi"])
        bottom.fill_between(ks, lo, hi, color=STYLE["JL-OUT"][0], alpha=0.25, lw=0)
        bottom.plot(ks, est, color=STYLE["JL-OUT"][0], lw=1.1)
        better = lo > 0
        bottom.scatter(np.array(ks)[better], est[better], s=12, color=STYLE["JL-OUT"][0], zorder=3)
        bottom.scatter(np.array(ks)[~better], est[~better], s=12, facecolor="white", edgecolor=STYLE["JL-OUT"][0], zorder=3)
        if d.get("primary_k"):
            for ax in (top, bottom):
                ax.axvline(d["primary_k"], color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=0)
        bottom.set_xscale("log", base=2)
        bottom.set_xticks(ks, [str(k) for k in ks])
        bottom.set_xlabel("set size $k$")
    axes[0, 0].set_ylabel("recovered share $F_k$")
    axes[1, 0].set_ylabel("JL $-$ LL on $F_k$")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    return fig


def draw_dissociation(plt, data: dict[str, Any], labels: dict[str, str]):
    models = list(data)
    fig, axes = plt.subplots(1, len(models), figsize=(FULL_WIDTH, 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        d = data[name]
        for sig, v in d["signals"].items():
            if not v["M1"] or not v["F"]:
                continue
            colour, marker, label = STYLE.get(sig, ("#333333", "o", sig))
            x, y = v["M1"]["mean"], v["F"]["mean"]
            ax.errorbar(x, y, xerr=[[x - v["M1"]["ci95"][0]], [v["M1"]["ci95"][1] - x]], yerr=[[y - v["F"]["ci95"][0]], [v["F"]["ci95"][1] - y]],
                        fmt=marker, color=colour, ms=4.5, elinewidth=0.7, capsize=0, label=label)
        if d.get("RANDOM_F") is not None:
            ax.axhline(d["RANDOM_F"], color=STYLE["RANDOM"][0], ls=":", lw=0.9, label="RANDOM (mean $F$)")
        ax.set_title(f"{labels[name]} ($k$ = {d['primary_k']})")
        ax.set_xlim(-0.1, 1.05)
    axes[0].set_ylabel("sets: recovered share $F_k$")
    fig.supxlabel("ranking: M1, Spearman with single-feature effects", fontsize=8, y=-0.04)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.1), handletextpad=0.3, columnspacing=1.0)
    return fig


def draw_superadditivity(plt, data: dict[str, Any], labels: dict[str, str]):
    """Per selector, two boxes over trials: the additive prediction (light) and the recovered share F (solid). Boxes are
    quartiles, whiskers the 5th and 95th percentiles; F far above the prediction is superadditivity."""
    from matplotlib.patches import Patch

    models = list(data)
    fig, axes = plt.subplots(1, len(models), figsize=(COLUMN_WIDTH, 1.85), sharey=True, gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        d = data[name]
        sels = list(d["selectors"])
        for i, sig in enumerate(sels):
            v = d["selectors"][sig]
            colour = STYLE.get(sig, ("#333333",))[0]
            for dx, values, alpha in ((-0.19, v["x"], 0.3), (0.19, v["y"], 0.9)):
                if not values:
                    continue
                b = ax.boxplot([values], positions=[i + dx], widths=0.32, whis=(5, 95), showfliers=False, patch_artist=True,
                               medianprops={"color": "white" if alpha > 0.5 else colour, "lw": 1.0},
                               whiskerprops={"color": colour, "lw": 0.7}, capprops={"color": colour, "lw": 0.7},
                               boxprops={"edgecolor": colour, "lw": 0.7})
                b["boxes"][0].set_facecolor(colour)
                b["boxes"][0].set_alpha(alpha)
        ax.axhline(0, color="#777777", lw=0.6)
        ax.set_xticks(range(len(sels)), [s.replace("-OUT", "") for s in sels], fontsize=6.5)
        ax.set_title(labels[name])
    axes[0].set_ylabel(f"share of the full patch, $k$ = {data[models[0]]['k']}")
    grey = "#555555"
    fig.legend([Patch(facecolor=grey, alpha=0.3, edgecolor=grey), Patch(facecolor=grey, alpha=0.9, edgecolor=grey)],
               ["additive prediction (summed single effects)", "recovered share $F$ (set patched together)"],
               loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.2))
    return fig


def draw_swap(plt, data: dict[str, Any], sesoi: float):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.62, 1.8))
    _sesoi_band(ax, sesoi)
    weights = ("it", "pt")
    for dodge, sae, colour, marker, label in ((-0.06, "it", "#0072B2", "o", "IT SAEs"), (0.06, "pt", "#E69F00", "s", "base SAEs")):
        xs, ys, err = [], [], [[], []]
        for i, w in enumerate(weights):
            c = data["cells"][f"{w}|{sae}"]
            xs.append(i + dodge)
            ys.append(c["estimate"])
            err[0].append(c["estimate"] - c["ci95"][0])
            err[1].append(c["ci95"][1] - c["estimate"])
        ax.errorbar(xs, ys, yerr=err, fmt=marker + "-", color=colour, ms=4, lw=1.0, elinewidth=0.8, capsize=2, label=label)
    ax.set_xticks([0, 1], ["instruction-tuned", "base"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_xlabel("weights")
    ax.set_ylabel(f"JL $-$ LL on $F_{{{data['k']}}}$")
    ax.legend(loc="upper left", handlelength=1.5)
    return fig


def _interval(ax, x: float, ci: list[float], y: float, **kw) -> None:
    ax.errorbar(x, y, xerr=[[x - ci[0]], [ci[1] - x]], elinewidth=0.8, capsize=0, **kw)


def draw_replication(plt, rows: list[dict[str, Any]], sesoi: float, keys: tuple[str, str] = ("D1", "D2"),
                     xlabel: str = "ATP $-$ J-lens"):
    """Two panels sharing the task-checkpoint rows: a contrast on ranking (M1) and on the recovered share; by default
    The ranking-versus-sets test's ATP - J-lens, with ``keys`` ("P2.1", "P2.2") the J-lens against its twin."""
    marks = {"factual": "D", "R5_number_agreement": "o", "R6_film_sentiment": "s"}
    ys = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 0.27 * len(rows) + 0.75), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, key, title in ((axes[0], keys[0], "ranking (M1)"), (axes[1], keys[1], "sets ($F$ at the primary $k$)")):
        ax.axvspan(-sesoi, sesoi, color="#DDDDDD", alpha=0.6, lw=0, zorder=0)
        ax.axvline(0, color="#777777", lw=0.6, zorder=1)
        for y, r in zip(ys, rows):
            _interval(ax, r[key]["estimate"], r[key]["ci95"], y, fmt=marks.get(r["task"], "o"), color="#222222", ms=3.5, zorder=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(ys, [f"{r['model']}, {TASKS.get(r['task'], r['task'])}, $k$ = {r['k']}" for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    return fig


def draw_decomposition(plt, rows: list[dict[str, Any]], pair: str):
    """Per run, dF (black), dU (singleton utility) and dN (non-additive remainder) with 95% intervals."""
    parts = (("F", "#000000", "o", r"$\Delta F$, recovered share"),
             ("U", STYLE["ATP"][0], "D", r"$\Delta U$, summed single-feature effects"),
             ("N", STYLE["JL-OUT"][0], "s", r"$\Delta N$, non-additive remainder"))
    ys = np.arange(len(rows))[::-1]
    height = 0.3 * len(rows) + 0.8
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, height))
    ax.axvline(0, color="#777777", lw=0.6, zorder=1)
    for dy, (q, colour, marker, label) in zip((0.22, 0.0, -0.22), parts):
        for i, (y, r) in enumerate(zip(ys, rows)):
            _interval(ax, r[q]["estimate"], r[q]["ci95"], y + dy, fmt=marker, color=colour, ms=3.2, label=label if i == 0 else None, zorder=3)
    ax.set_yticks(ys, [f"{r['label']}, $k$ = {r['k']}" for r in rows])
    ax.set_xlabel(pair.replace(" - ", " $-$ ").replace("JL-OUT", "J-lens") + " at the primary $k$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    return fig


# ------------------------------------------------------------------------------------------------ command

def _save(ctx: RunContext, fig, name: str) -> list[str]:
    written = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        rel = f"figures/{name}.{ext}"
        target = ctx.run.file(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, **kw)
        ctx.run._note(rel)
        written.append(rel)
    return written


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    figs = cfg.get("figures") or {}
    unknown = sorted(set(figs) - set(FIGURES))
    if unknown:
        raise RunContractError(f"figures: unknown figures {unknown}; the paper has {list(FIGURES)}")
    needed = sorted({m for spec in figs.values() for m in spec.get("models", [])})
    missing = [m for m in needed if m not in (cfg.get("models") or {})]
    if missing:
        raise RunContractError(f"figures: models {missing} are used by a figure but not declared under `models`")
    loaded: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = {}
    for name in needed:
        rows, run_cfg, prov, flags = C.load(cfg["models"][name])
        ctx.flags.extend(f"{name}: {f}" for f in flags)
        loaded[name] = (rows, run_cfg, prov)
    labels = {name: entry.get("label", name) for name, entry in cfg["models"].items()}
    plt = _pyplot()
    data: dict[str, Any] = {"provenance": {name: prov for name, (_, _, prov) in loaded.items()}}
    send: list[str] = []
    notes: list[str] = []

    if "fk" in figs:
        d = {}
        for name in figs["fk"]["models"]:
            rows, run_cfg, prov = loaded[name]
            d[name] = fk_data(rows, run_cfg, ks=list(run_cfg["k_grid"]), seed=_seed(cfg["seed"], name), holm_alpha=cfg["holm_alpha"])
            d[name]["primary_k"] = prov["primary_k"]
        data["fk"] = d
        send += _save(ctx, draw_fk(plt, d, labels), "fig_fk")
        notes.append("fk: " + "; ".join(f"{labels[n]} JL - LL at k = {v['primary_k']}: {v['contrast'][str(v['primary_k'])]['estimate']:+.3f}" for n, v in d.items()))

    if "dissociation" in figs:
        spec = figs["dissociation"]
        d = {}
        for name in spec["models"]:
            rows, run_cfg, prov = loaded[name]
            d[name] = dissociation_data(rows, run_cfg, signals=spec["signals"], pk=int(prov["primary_k"]), seed=_seed(cfg["seed"], name, "dissociation"))
        data["dissociation"] = d
        send += _save(ctx, draw_dissociation(plt, d, labels), "fig_dissociation")
        notes.append("dissociation: " + "; ".join(
            f"{labels[n]} " + ", ".join(f"{s} M1 {v['signals'][s]['M1']['mean']:.2f} F {v['signals'][s]['F']['mean']:.2f}"
                                        for s in ("JL-OUT", "ATP", "NORM") if v["signals"].get(s, {}).get("M1") and v["signals"][s].get("F"))
            for n, v in d.items()))

    if "superadditivity" in figs:
        spec = figs["superadditivity"]
        d = {}
        for name in spec["models"]:
            rows, run_cfg, _ = loaded[name]
            d[name] = superadditivity_data(rows, run_cfg, cell=spec["cell"], k=int(spec["k"]), selectors=spec["selectors"])
        data["superadditivity"] = d
        send += _save(ctx, draw_superadditivity(plt, d, labels), "fig_superadditivity")
        notes.append("superadditivity: " + "; ".join(
            f"{labels[n]} " + ", ".join(f"{s} median additive {v['median_x']:.3f} vs F {v['median_y']:.3f} ({v['trials']} trials)"
                                        for s, v in dd["selectors"].items() if v["trials"]) for n, dd in d.items()))

    if "swap" in figs:
        spec = figs["swap"]
        path = Path(spec["report"])
        if not path.exists():
            raise RunContractError(f"swap: compare-sets report {path} not found")
        report = json.loads(path.read_text(encoding="utf-8"))
        d = swap_data(report, spec["cells"], k=int(spec["k"]))
        d["source"] = {"report": path.as_posix(), "sha256": _sha256(path), "code_commit": report.get("code_commit")}
        sesoi = next(iter(loaded.values()))[1]["analysis"]["sesoi"]["M2"] if loaded else float(spec["sesoi"])
        d["sesoi"] = sesoi
        data["swap"] = d
        send += _save(ctx, draw_swap(plt, d, sesoi), "fig_swap")
        notes.append("swap: " + ", ".join(f"{k} {v['estimate']:+.3f}" for k, v in d["cells"].items()))

    def load_reports(figure: str, paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        reports, sources = {}, {}
        for label, rel in paths.items():
            path = Path(rel)
            if not path.exists():
                raise RunContractError(f"{figure}: report {path} not found")
            reports[label] = json.loads(path.read_text(encoding="utf-8"))
            sources[label] = {"report": path.as_posix(), "sha256": _sha256(path), "code_commit": reports[label].get("code_commit")}
        return reports, sources

    if "jlens" in figs:
        spec = figs["jlens"]
        factual, src_f = load_reports("jlens", spec["factual"])
        new_tasks, src_n = load_reports("jlens", spec["new_tasks"])
        rows = jlens_data(factual, new_tasks)
        data["jlens"] = {"rows": rows, "sources": {"factual": src_f, "new_tasks": src_n}, "sesoi": float(spec["sesoi"])}
        send += _save(ctx, draw_replication(plt, rows, float(spec["sesoi"]), keys=("P2.1", "P2.2"), xlabel="J-lens $-$ logit lens"),
                      "fig_jlens")
        notes.append("jlens: " + "; ".join(f"{r['model']} {r['task']} P2.1 {r['P2.1']['estimate']:+.3f} P2.2 {r['P2.2']['estimate']:+.3f}"
                                         for r in rows))

    if "replication" in figs:
        spec = figs["replication"]
        reports, sources = load_reports("replication", spec["reports"])
        rows = replication_data(reports)
        data["replication"] = {"rows": rows, "sources": sources, "sesoi": float(spec["sesoi"])}
        send += _save(ctx, draw_replication(plt, rows, float(spec["sesoi"])), "fig_replication")
        notes.append("replication: " + "; ".join(f"{r['model']} {r['task']} D1 {r['D1']['estimate']:+.3f} D2 {r['D2']['estimate']:+.3f}" for r in rows))

    if "decomposition" in figs:
        spec = figs["decomposition"]
        path = Path(spec["report"])
        if not path.exists():
            raise RunContractError(f"decomposition: report {path} not found")
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = decomposition_data(report, spec["runs"], spec["pair"])
        data["decomposition"] = {"rows": rows, "pair": spec["pair"],
                                 "source": {"report": path.as_posix(), "sha256": _sha256(path), "code_commit": report.get("code_commit")}}
        send += _save(ctx, draw_decomposition(plt, rows, spec["pair"]), "fig_decomposition")
        notes.append("decomposition: " + "; ".join(f"{r['label']} dF {r['F']['estimate']:+.3f} dU {r['U']['estimate']:+.3f} "
                                                    f"dN {r['N']['estimate']:+.3f}" for r in rows))

    ctx.run.write_json("figures_data.json", data)
    plt.close("all")
    lines = ["## What ran", "",
             "Figures (descriptive), drawn from the Stage 2 test runs' results.zip, each checked against its sealed frozen "
             "protocol. Every number drawn is in figures_data.json. Intervals: 95% cluster bootstrap within relation; bands: the frozen "
             "M2 SESOI. Unsquared.", "", "## Figures", ""]
    lines += [f"- `{s}`" for s in send if s.endswith(".pdf")]
    lines += ["", "## Key numbers", ""] + [f"- {n}" for n in notes]
    lines += ["", "## Next step", "", "Send back the files in the status line."]
    return Outcome(status="PASS", summary="\n".join(lines) + "\n", report={"figures": [s for s in send if s.endswith(".pdf")]},
                   send_back=["figures_data.json", *send])
