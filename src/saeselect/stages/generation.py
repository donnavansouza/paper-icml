"""``python -m saeselect generate``: a generation check of the margin endpoint.

    python -m saeselect generate --config configs/extensions/generation.yaml --model configs/models/<model>.yaml --phase dev --out runs/generation-dev-<m>-01 --device cuda
    python -m saeselect generate --config configs/extensions/generation.yaml --phase test --frozen runs/generation-dev-<m>-01/frozen_protocol.json \
        --out runs/generation-test-<m>-01 --device cuda

From the parent Stage 2 run of the phase, at every eligible cell, on the greedy reference's subset rule (40 pairs, one seeded template per pair,
both directions): for the unedited prompt, the full patch and each declared selector's stored set at the primary k, the edit is
applied at the prompt positions of the cell (delta form, error term kept) and the model decodes greedily from the FP32 head,
each step a full forward over the prompt and the tokens so far with the edit at the same prompt positions, so every generated
token conditions on the edited prompt. Success: the generated text, after leading whitespace, starts with the donor's answer (or
its capitalized form) followed by a non-alphanumeric character or the end; the source's answer is counted the same way. The first
step's margin must equal the parent's stored margin of the same condition (the parent's parity tolerance). Descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from saeselect import artifacts as A
from saeselect.analysis import rebuild as R
from saeselect.hooks import EditSpec, ResidualEdits, ResidualRecorder
from saeselect.interventions import candidate_pool
from saeselect.models import load_model, precision_mode
from saeselect.patching import as_ids
from saeselect.provenance import EngineeringFailed, RunContractError, json_hash, seal_protocol, utc_now
from saeselect.readout import answer_margin, logits_fp32
from saeselect.results import write_results_zip
from saeselect.runner import Outcome, RunContext
from saeselect.stages import selection as S2
from saeselect.stages.greedy_reference import choose_rows
from saeselect.stages.jacobian_controls import _parent


def generate_edited(handle, ids: torch.Tensor, edits: list[EditSpec], answers, readout: int, *, mode: str, n_new: int,
                    stop: str = "\n") -> tuple[float, list[int], str]:
    """(the margin at ``readout`` on the edited prompt, the new token ids, their text): greedy from the FP32 head, a full forward
    per step with ``edits`` at the same prompt positions, stopping after ``n_new`` tokens or once ``stop`` appears."""
    last = handle.n_layers - 1
    seq, new, margin, text = ids, [], None, ""
    for _ in range(n_new):
        with torch.no_grad(), precision_mode(handle, mode), ResidualRecorder(handle.layers, at=[last]) as rec, \
                ResidualEdits(handle.layers, edits):
            handle.forward(seq)
        h = rec.activations[last][0]
        if margin is None:
            margin = float(answer_margin(h[readout], handle.head, answers))
        token = int(logits_fp32(h[-1], handle.head).argmax())
        new.append(token)
        text = handle.tokenizer.decode(new, skip_special_tokens=True)
        if token == handle.tokenizer.eos_token_id or stop in text:
            break
        seq = torch.cat([seq, torch.tensor([[token]], device=seq.device)], dim=1)
    return margin, new, text


def starts_with_answer(text: str, answer: str) -> bool:
    t = text.lstrip()
    for v in dict.fromkeys((answer, answer[:1].upper() + answer[1:])):
        if t.startswith(v) and (len(t) == len(v) or not t[len(v)].isalnum()):
            return True
    return False


def outcome(text: str, donor: str, source: str) -> str:
    return "donor" if starts_with_answer(text, donor) else "source" if starts_with_answer(text, source) else "other"


def run(ctx: RunContext) -> Outcome:
    cfg = ctx.config
    if ctx.phase not in ("dev", "test"):
        raise RunContractError("generate needs --phase dev or --phase test")
    if ctx.phase == "test" and json_hash(cfg) != json_hash(ctx.frozen["config"]):
        raise RunContractError("--config differs from the frozen protocol; a change needs a new protocol")
    name, entry, parent_frozen, parent_dir = _parent(ctx)
    data = R.read_run(parent_dir)
    prov, flags = R.provenance(data, parent_frozen, Path(entry["frozen"]))
    if data["protocol"].get("phase") != ctx.phase:
        raise RunContractError(f"{parent_dir} is a {data['protocol'].get('phase')} run; the {ctx.phase} phase reads the parent's {ctx.phase} run")
    ctx.flags.extend(f"parent {parent_dir.as_posix()}: {f}" for f in flags if "a dev run" not in f)
    pcfg = parent_frozen.get("config_resolved") or S2._resolve(parent_frozen["config"])
    model_cfg = parent_frozen["model"]
    s0 = ctx.load_upstream(Path(parent_frozen["stage0_run"]) / "stage0_decisions.json")
    causal = s0.get("causal_mode", s0["precision_mode"])
    pk = int(parent_frozen["primary_k"])
    tol = float(pcfg["parity_probe"]["max_abs_margin_diff"])
    selectors = list(cfg["selectors"])
    n_pairs = int(cfg["n_pairs"]) if ctx.limit is None else ctx.limit
    cells = list(parent_frozen["eligible_cells"])
    rows_by_cell = {c: choose_rows(data["rows"], cell=c, n_pairs=n_pairs, seed=int(cfg["subset_seed"])) for c in cells}

    trials: dict[str, S2.Trial] = {}
    data_sha = {}
    for family in parent_frozen["relations"]:
        card, records = S2._load(Path(pcfg["data_dir"]), family, ctx.phase)
        data_sha[family] = card["records_sha256"]
        for t in S2.build_trials(records, family):
            trials[t.key] = t
    handle = load_model(model_cfg, device=ctx.device)
    layers = sorted({int(c.split("/")[0]) for c in cells})
    cache = S2.PromptCache(handle, layers, causal)
    out_rows, parity_max = [], 0.0
    for layer in layers:
        u = s0["sae"][str(layer)]
        sae = A.load_sae_for(model_cfg, layer=layer, width=u["width"], l0=u["l0"], d_model=handle.d_model, suffix=u.get("suffix", ""),
                             device=ctx.device, convention=u.get("convention"), repo_id=u.get("repo_id"), revision=u.get("revision"))
        for cell in [c for c in cells if int(c.split("/")[0]) == layer]:
            site = cell.split("/")[1]
            for r in rows_by_cell[cell]:
                t = trials[r["trial"]]
                src, don = cache.get(t.source), cache.get(t.donor)
                positions = src["sites"][site]
                h_src, h_don = src["acts"][(layer, site)].to(sae.device), don["acts"][(layer, site)].to(sae.device)
                da_full = sae.encode(h_don) - sae.encode(h_src)
                pool = candidate_pool(da_full)
                col = {f: j for j, f in enumerate(pool.tolist())}
                da, W = da_full.index_select(1, pool), sae.W_dec.index_select(0, pool)
                conditions: dict[str, tuple[list[EditSpec], float]] = {
                    "clean": ([], r["m"]["clean"]),
                    "FULL": ([EditSpec(layer=layer, positions=positions, delta=(h_don - h_src).float())], r["m"]["FULL"])}
                for sel in selectors:
                    entry_k = r["sets"][sel][str(pk)]
                    c = torch.as_tensor([col[f] for f in entry_k["active"] if f in col], dtype=torch.long, device=da.device)
                    edits = [EditSpec(layer=layer, positions=positions, delta=da.index_select(1, c) @ W.index_select(0, c))] if len(c) else []
                    conditions[sel] = (edits, entry_k["margin"])
                ids = as_ids(src["ids"], handle.device)
                donor, source = t.donor.attribute["answer"], t.source.attribute["answer"]
                result: dict[str, Any] = {}
                for cond, (edits, stored) in conditions.items():
                    margin, new, text = generate_edited(handle, ids, edits, t.answers, t.source.meta["readout_position"], mode=causal,
                                                        n_new=int(cfg["max_new_tokens"]))
                    diff = abs(margin - stored)
                    parity_max = max(parity_max, diff)
                    if diff > tol:
                        raise EngineeringFailed("parity", f"{r['trial']} at {cell}, {cond}: first-step margin {margin:.6g}, stored {stored:.6g} "
                                                          f"(tolerance {tol})")
                    result[cond] = {"margin": margin, "text": text, "outcome": outcome(text, donor, source), "n_new": len(new)}
                out_rows.append({"trial": r["trial"], "relation": r["relation"], "pair": r["pair"], "template": r["template"],
                                 "direction": r["direction"], "layer": layer, "site": site, "full_gain": r["full_gain"],
                                 "donor_answer": donor, "source_answer": source, "conditions": result})
        del sae
        S2._free()

    write_results_zip(ctx.run.path, {"trials.jsonl": "".join(json.dumps(x, sort_keys=True, default=S2._jsonable) + "\n" for x in out_rows)})
    conds = ["clean", "FULL", *selectors]
    analysis = analyze(out_rows, pcfg, conds, seed=int(cfg["seed"]))
    report = json.loads(json.dumps({"phase": ctx.phase, "model": name, "parent_run": parent_dir.as_posix(),
                                    "parent_provenance": prov, "parent_frozen_protocol_hash": parent_frozen["protocol_hash"], "cells": cells,
                                    "primary_k": pk, "n_trials": len(out_rows), "n_pairs": len({x["pair"] for x in out_rows}),
                                    "parity_max": parity_max, "parity_tolerance": tol, "analysis": analysis}, default=S2._jsonable))
    summary = _summary(ctx, report, conds)
    if ctx.phase == "dev":
        protocol = {"stage": "generate", "smoke": ctx.smoke, "code_commit": ctx.git.get("commit"),
                    "source_run": ctx.run.path.as_posix(), "created_at": utc_now(), "config": cfg, "model": model_cfg,
                    "parent_protocol_hash": parent_frozen["protocol_hash"], "cells": cells, "data": data_sha}
        frozen_protocol = seal_protocol(protocol)
        ctx.run.write_json("frozen_protocol.json", frozen_protocol)
        report["frozen_protocol_hash"] = frozen_protocol["protocol_hash"]
        return Outcome(status="FROZEN_AWAITING_APPROVAL", summary=summary, report=report, send_back=["frozen_protocol.json", "results.zip"])
    if ctx.frozen.get("parent_protocol_hash") != parent_frozen["protocol_hash"] or ctx.frozen.get("cells") != cells:
        raise RunContractError("the parent protocol or the cells differ from the frozen protocol")
    return Outcome(status="PASS", summary=summary, report=report, send_back=["results.zip"])


def analyze(rows: list[dict[str, Any]], pcfg: dict[str, Any], conds: list[str], *, seed: int) -> dict[str, Any]:
    """Per condition, pooled and per cell, over trials at or above the parent's M2 floor: the pair-mean share of generations that
    start with the donor's answer (95% interval), with the source's answer, and with neither; and the agreement of the margin
    endpoint with generation, the share of trials where (m > 0) equals (the generation starts with the donor's answer)."""
    floor = pcfg["m2_min_full_gain"]

    def share(cond, what):
        return lambda r: None if r["full_gain"] < floor else float(r["conditions"][cond]["outcome"] == what)

    def agree(cond):
        return lambda r: None if r["full_gain"] < floor else float((r["conditions"][cond]["margin"] > 0) == (r["conditions"][cond]["outcome"] == "donor"))

    def block(sub, tag):
        return {c: {"donor": S2._mean_ci(sub, share(c, "donor"), pcfg, S2._hseed(seed, tag, c)),
                    "source": S2._pair_mean(sub, share(c, "source")), "other": S2._pair_mean(sub, share(c, "other")),
                    "agreement": S2._pair_mean(sub, agree(c))} for c in conds}

    cells = sorted({(r["layer"], r["site"]) for r in rows})
    return {"pooled": block(rows, "pooled"),
            "per_cell": {f"{layer}/{site}": block([r for r in rows if r["layer"] == layer and r["site"] == site], f"{layer}/{site}")
                         for layer, site in cells}}


def _summary(ctx: RunContext, report: dict[str, Any], conds: list[str]) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.0%}"

    def mci(c):
        if not c:
            return "n/a"
        return pct(c["estimate"]) + (f" [{pct(c['ci95'][0])}, {pct(c['ci95'][1])}]" if c.get("ci95") else "")

    lines = ["## What ran", "",
             f"The generation check, {ctx.phase} phase, on `{report['model']}` at {', '.join(report['cells'])}: greedy generations (up to the "
             f"declared token budget) for {report['n_trials']} trials in {report['n_pairs']} pairs of the parent Stage 2 {ctx.phase} run "
             f"`{report['parent_run']}`, unedited, with the full patch, and with each selector's stored set at k = {report['primary_k']}, the "
             "edit kept at the prompt positions. Success: the generation starts with the donor's answer. Descriptive.", "",
             f"Parity: the first step's margin against the parent's stored margin, worst difference {report['parity_max']:.2g} "
             f"(tolerance {report['parity_tolerance']:.2g}).", "", "## Pooled (pair means, trials at or above the M2 floor)", "",
             "| Condition | Starts with the donor's answer [95%] | Source's answer | Neither | Margin sign agrees with generation |",
             "|---|---|---|---|---|"]
    for c in conds:
        v = report["analysis"]["pooled"][c]
        lines.append(f"| {c} | {mci(v['donor'])} | {pct(v['source'])} | {pct(v['other'])} | {pct(v['agreement'])} |")
    for cell, block in report["analysis"]["per_cell"].items():
        lines += ["", f"### {cell}", "", "| Condition | Donor | Source | Neither | Agreement |", "|---|---|---|---|---|"]
        for c in conds:
            v = block[c]
            lines.append(f"| {c} | {mci(v['donor'])} | {pct(v['source'])} | {pct(v['other'])} | {pct(v['agreement'])} |")
    lines += ["", "## What this permits", "",
              "Whether recovering the answer margin at the first answer token goes with the model generating the donor's answer, per "
              "selector, on a small fixed sample. A descriptive check of the endpoint, not a generation benchmark.", "", "## Next step", "",
              "Send back the files in the status line."]
    return "\n".join(lines) + "\n"
