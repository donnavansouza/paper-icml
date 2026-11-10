"""Paper figures (saeselect.analysis.figures): the numbers drawn, on fabricated trials, and that every figure renders."""

from __future__ import annotations

import pytest
import yaml

from data_fixtures import REPO
from saeselect.analysis import figures as F

KS = [1, 2, 4, 8, 16, 32]
SIGNALS = ["NORM", "LL-OUT", "JL-OUT", "ORACLE"]


def _cfg() -> dict:
    cfg = yaml.safe_load((REPO / "configs/selection/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg["analysis"]["bootstrap"]["n_resamples"] = 300
    cfg["k_grid"] = KS
    return cfg


def _rows(n: int = 10) -> list[dict]:
    rows = []
    for rel in ("R2_element_symbol", "R3_verb_past"):
        for i in range(n):
            noise = 0.01 * ((i * 7) % 5 - 2)
            f = {"JL-OUT": 0.5 + noise, "LL-OUT": 0.3, "NORM": 0.6, "ORACLE": 0.4}
            m1 = {"JL-OUT": 0.3 + noise, "LL-OUT": 0.1, "NORM": 0.05, "ORACLE": 1.0}
            rows.append({
                "pair": f"{rel}:{i}", "relation": rel, "layer": 12, "site": "concept", "full_gain": 5.0,
                "effects": {"1": 0.5, "2": 0.25, "7": -0.1},
                "sets": {s: {str(k): {"F": v, "selected": [1, 2, 3]} for k in KS} for s, v in f.items()},
                "random": {str(k): [{"F": 0.01}, {"F": 0.03}] for k in KS},
                "metrics": {s: {"M1": v} for s, v in m1.items()},
            })
    return rows


def test_superadditivity_is_the_selected_sets_summed_single_effects_over_the_full_gain():
    d = F.superadditivity_data(_rows(), _cfg(), cell="12/concept", k=32, selectors=["NORM", "ORACLE"])
    v = d["selectors"]["NORM"]
    # features 1 and 2 are in the evaluation pool (0.5 + 0.25), feature 3 is not and contributes 0; full gain 5.0
    assert v["trials"] == 20 and all(abs(x - 0.15) < 1e-12 for x in v["x"]) and all(abs(y - 0.6) < 1e-12 for y in v["y"])
    assert abs(v["mean_summed_effect"] - 0.75) < 1e-12 and abs(v["mean_set_effect"] - 3.0) < 1e-12 and v["share_above_diagonal"] == 1.0
    assert F.superadditivity_data(_rows(), _cfg(), cell="24/concept", k=32, selectors=["NORM"])["selectors"]["NORM"]["trials"] == 0


def test_dissociation_means_and_random_line():
    d = F.dissociation_data(_rows(), _cfg(), signals=SIGNALS, pk=32, seed=1)
    assert d["primary_k"] == 32 and abs(d["RANDOM_F"] - 0.02) < 1e-12
    assert abs(d["signals"]["NORM"]["F"]["mean"] - 0.6) < 1e-12 and abs(d["signals"]["LL-OUT"]["M1"]["mean"] - 0.1) < 1e-12
    lo, hi = d["signals"]["JL-OUT"]["M1"]["ci95"]
    assert lo <= d["signals"]["JL-OUT"]["M1"]["mean"] <= hi and d["signals"]["JL-OUT"]["M1"]["pairs"] == 20


def test_fk_matches_compare_sets():
    from saeselect.analysis import compare_sets as C

    cfg, rows = _cfg(), _rows()
    d = F.fk_data(rows, cfg, ks=KS, seed=7, holm_alpha=0.05)
    assert d["contrast"] == C.per_k(rows, cfg, ks=KS, holm_alpha=0.05, seed=7)  # the same procedure and seed as Appendix B.1
    assert d["levels"]["4"]["NORM"] == pytest.approx(0.6) and d["sesoi"] == cfg["analysis"]["sesoi"]["M2"]


def _swap_report() -> dict:
    models = {}
    for key, est in (("ii", 0.001), ("ip", 0.022), ("pi", 0.069), ("pp", 0.082)):
        models[key] = {"matched": {"M2": {"32": {"estimate": est, "ci95": [est - 0.02, est + 0.02], "label": "x"}},
                                   "levels": {"32": {"pairs": 10}}}}
    return {"models": models, "cross_model": {"weights": {"32": {"estimate": 0.068, "ci95": [0.046, 0.091], "label": "better", "p_holm": 0.0}}}}


def test_swap_cells_and_contrasts():
    d = F.swap_data(_swap_report(), {"it": {"it": "ii", "pt": "ip"}, "pt": {"it": "pi", "pt": "pp"}}, k=32)
    assert d["cells"]["pt|it"]["estimate"] == 0.069 and d["cells"]["it|pt"]["model"] == "ip" and d["contrasts"]["weights"]["p_holm"] == 0.0
    with pytest.raises(Exception):
        F.swap_data(_swap_report(), {"it": {"it": "missing"}}, k=32)


def _a36_report(pk: int = 8) -> dict:
    fam = {f"{d} {t}": {"estimate": e, "ci95": [e - 0.01, e + 0.01], "label": "x"}
           for t in ("R5_number_agreement", "R6_film_sentiment") for d, e in (("D1", 0.06), ("D2", -0.03))}
    return {"primary_k": pk, "ranking_vs_sets": {"dissociation": fam, "replication": {t: {"replicates": True, "reason": "ok"}
                                                                           for t in ("R5_number_agreement", "R6_film_sentiment")}}}


def _s2_report(pk: int = 32) -> dict:
    fam = {n: {"estimate": e, "ci95": [e - 0.01, e + 0.01], "label": "better"} for n, e in (("P2.1", 0.11), ("P2.2", 0.004))}
    return {"primary_k": pk, "analysis": {"primary": fam}}


def _a36_per_task(pk: int = 8) -> dict:
    rep = _a36_report(pk)
    rep["ranking_vs_sets"]["per_task_primary"] = {t: {n: {"estimate": e, "ci95": [e - 0.01, e + 0.01], "label": "better"}
                                          for n, e in (("P2.1", 0.07), ("P2.2", 0.03))}
                                      for t in ("R6_film_sentiment", "R5_number_agreement")}
    return rep


def test_jlens_rows_are_factual_then_tasks_per_checkpoint():
    rows = F.jlens_data({"A": _s2_report(), "B": _s2_report(4)}, {"A": _a36_per_task(), "C": _a36_per_task(1)})
    assert [(r["model"], r["task"], r["k"]) for r in rows] == [
        ("A", "factual", 32), ("A", "R5_number_agreement", 8), ("A", "R6_film_sentiment", 8), ("B", "factual", 4),
        ("C", "R5_number_agreement", 1), ("C", "R6_film_sentiment", 1)]
    assert rows[0]["P2.2"]["estimate"] == 0.004 and rows[1]["P2.1"]["ci95"] == pytest.approx([0.06, 0.08])
    with pytest.raises(Exception):
        F.jlens_data({"A": {"primary_k": 32, "analysis": {}}}, {})
    with pytest.raises(Exception):
        F.jlens_data({}, {"A": _a36_report()})


def _a43_report() -> dict:
    e = {q: {"estimate": v, "ci95": [v - 0.01, v + 0.01]} for q, v in (("F", -0.09), ("U", 0.03), ("N", -0.12))}
    return {"runs": {"pt": {"primary_k": 32, "decomposition": {"pooled": {"ATP - JL-OUT": {**e, "reading": "order reversal"}}}}}}


def test_replication_and_decomposition_read_the_reports_as_stated():
    rows = F.replication_data({"Model A": _a36_report()})
    assert [r["task"] for r in rows] == ["R5_number_agreement", "R6_film_sentiment"] and rows[0]["k"] == 8
    assert rows[0]["D1"]["estimate"] == 0.06 and rows[0]["D2"]["ci95"] == pytest.approx([-0.04, -0.02]) and rows[1]["replicates"]
    d = F.decomposition_data(_a43_report(), {"pt": "Gemma base"}, "ATP - JL-OUT")
    assert d[0]["F"]["estimate"] == -0.09 and d[0]["N"]["estimate"] == -0.12 and d[0]["reading"] == "order reversal"
    with pytest.raises(Exception):
        F.replication_data({"Model A": {"primary_k": 8}})
    with pytest.raises(Exception):
        F.decomposition_data(_a43_report(), {"missing": "x"}, "ATP - JL-OUT")


def test_every_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    plt = F._pyplot()
    cfg, rows = _cfg(), _rows()
    labels = {"a": "Model A", "b": "Model B"}
    fk = {m: dict(F.fk_data(rows, cfg, ks=KS, seed=1, holm_alpha=0.05), primary_k=32) for m in labels}
    dis = {m: F.dissociation_data(rows, cfg, signals=SIGNALS, pk=32, seed=1) for m in labels}
    sup = {m: F.superadditivity_data(rows, cfg, cell="12/concept", k=32, selectors=["ORACLE", "NORM", "JL-OUT"]) for m in labels}
    swap = F.swap_data(_swap_report(), {"it": {"it": "ii", "pt": "ip"}, "pt": {"it": "pi", "pt": "pp"}}, k=32)
    rep = F.replication_data({"Model A": _a36_report(), "Model B": _a36_report(1)})
    dec = F.decomposition_data(_a43_report(), {"pt": "Gemma base"}, "ATP - JL-OUT")
    jl = F.jlens_data({"Model A": _s2_report()}, {"Model A": _a36_per_task()})
    for name, fig in (("fk", F.draw_fk(plt, fk, labels)), ("dis", F.draw_dissociation(plt, dis, labels)),
                      ("sup", F.draw_superadditivity(plt, sup, labels)), ("swap", F.draw_swap(plt, swap, 0.05)),
                      ("rep", F.draw_replication(plt, rep, 0.05)), ("dec", F.draw_decomposition(plt, dec, "ATP - JL-OUT")),
                      ("jl", F.draw_replication(plt, jl, 0.05, keys=("P2.1", "P2.2"), xlabel="J-lens $-$ logit lens"))):
        out = tmp_path / f"{name}.pdf"
        fig.savefig(out)
        assert out.stat().st_size > 1000
    plt.close("all")


def test_command_refuses_an_unknown_figure_or_model(tmp_path):
    from saeselect.runner import execute

    for body in ({"figures": {"nope": {}}}, {"models": {}, "figures": {"fk": {"models": ["x"]}}}):
        cfg = tmp_path / f"figs_{len(list(tmp_path.iterdir()))}.yaml"
        cfg.write_text(yaml.safe_dump(body), encoding="utf-8")
        lines: list[str] = []
        code = execute("figures", {"config": str(cfg), "out": str(tmp_path / "runs" / cfg.stem)}, F.run, printer=lines.append, repo_root=REPO)
        assert code != 0 and lines[-1].startswith("STATUS=ERROR")
