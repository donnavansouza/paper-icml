"""The R5 and R6 builds (same-lemma pairs, one exemplar per answer, answer forms), JL-NORM, the norm-budget walk, the
nested config bases, and on the tiny model a Stage 2 dev pass with the new-task endpoint endpoint that freezes, then one test run."""

from __future__ import annotations

import copy
import json
import zipfile
from types import SimpleNamespace

import pytest
import torch
import yaml

from data_fixtures import ADJECTIVES, NOUNS, REPO, WORDS
from saeselect import cli, tiny
from saeselect import signals as S
from saeselect.config import load_config
from saeselect.runner import execute
from saeselect.stages import selection as S2
from saeselect.tasks import relations as R


def _relations_cfg(tmp_path):
    """configs/data/relations.yaml's relation settings with the new-task endpoint's R5 and R6, the item files replaced by small fixtures."""
    cfg = yaml.safe_load((REPO / "configs/data/relations.yaml").read_text(encoding="utf-8"))["relations"]
    a36 = yaml.safe_load((REPO / "configs/data/new_tasks.yaml").read_text(encoding="utf-8"))["overrides"]["relations"]
    cfg.update(copy.deepcopy(a36))
    nouns, adjectives = tmp_path / "nouns.yaml", tmp_path / "adjectives.yaml"
    nouns.write_text(yaml.safe_dump({"nouns": NOUNS}), encoding="utf-8")
    adjectives.write_text(yaml.safe_dump(ADJECTIVES), encoding="utf-8")
    cfg["R5"]["items"]["file"], cfg["R6"]["items"]["file"] = str(nouns), str(adjectives)
    cfg["dev_pairs"], cfg["min_test_pairs"] = 2, 2
    return cfg


def test_r5_pairs_the_two_forms_of_one_noun(tmp_path):
    cfg = _relations_cfg(tmp_path)
    tok = tiny.build_tokenizer(WORDS)
    r5 = cfg["R5"]
    concepts, excluded = R.parse_declared(r5["items"]["file"], kind=r5["items"]["kind"], answers=r5["answers"])
    out = R.build_relation("R5", cfg, r5, concepts=concepts, source_excluded=excluded, tokenizer=tok, seed=7)
    card = out["card"]
    ex = card["exemplars"]
    assert [e["answer"] for e in ex] == ["is", "are"]  # one singular, one plural exemplar, in that order
    lemma = {pl: sg for sg, pl in NOUNS} | {sg: sg for sg, _ in NOUNS}
    ex_lemmas = {lemma[e["concept"]] for e in ex}
    assert len(ex_lemmas) == 2  # two different nouns
    pairs = [p.split("|") for split in card["pairs"].values() for p in split]
    assert pairs and all(lemma[a] == lemma[b] == a and b != a for a, b in pairs)  # (singular, plural) of one noun
    assert not ex_lemmas & {a for a, _ in pairs}  # an exemplar's noun leaves the pool, both forms
    assert any(e["item"] == "knife|knives" and "token count" in e["reason"] for e in card["excluded"])
    assert card["pairing"] == "same_lemma" and card["answer_forms"]["is"] == ["is", "was"]
    first = {v: tok.encode(v, add_special_tokens=False)[0] for v in (" is", " was", " are", " were")}
    for r in out["records"]:
        plus, minus = set(r.answer_sets["A_plus"]["ids"]), set(r.answer_sets["A_minus"]["ids"])
        assert not plus & minus
        singular_side = minus if r.meta["source_concept"] == lemma[r.meta["source_concept"]] else plus
        assert {first[" is"], first[" was"]} <= singular_side and not {first[" are"], first[" were"]} & singular_side
        assert r.meta["concept_positions"][-1] < r.meta["readout_position"] - 1  # the subject is not the readout token
    assert not out["below_minimum"]


def test_r6_reserves_one_exemplar_per_verdict(tmp_path):
    cfg = _relations_cfg(tmp_path)
    tok = tiny.build_tokenizer(WORDS)
    r6 = cfg["R6"]
    concepts, excluded = R.parse_declared(r6["items"]["file"], kind=r6["items"]["kind"], answers=r6["answers"])
    out = R.build_relation("R6", cfg, r6, concepts=concepts, source_excluded=excluded, tokenizer=tok, seed=7)
    assert [e["answer"] for e in out["card"]["exemplars"]] == ["positive", "negative"]
    verdict = {w: cls for cls, ws in ADJECTIVES.items() for w in ws}
    pairs = [p.split("|") for split in out["card"]["pairs"].values() for p in split]
    assert pairs and all(verdict[a] != verdict[b] for a, b in pairs)


def test_declared_lists_have_no_duplicates():
    nouns = yaml.safe_load((REPO / "configs/data/agreement_nouns.yaml").read_text(encoding="utf-8"))["nouns"]
    words = [w for pair in nouns for w in pair]
    assert len(words) == len(set(words)) and all(sg != pl for sg, pl in nouns)
    adj = yaml.safe_load((REPO / "configs/data/sentiment_adjectives.yaml").read_text(encoding="utf-8"))
    assert not set(adj["positive"]) & set(adj["negative"])
    assert len(adj["positive"]) == len(set(adj["positive"])) and len(adj["negative"]) == len(set(adj["negative"]))


def test_jl_norm_is_norm_through_j():
    g = torch.Generator().manual_seed(0)
    da, W = torch.randn(2, 5, generator=g), torch.randn(5, 4, generator=g)
    base = dict(da=da, W_dec_pool=W, sd_pool=torch.ones(5), lens_s={}, grad_site=torch.zeros(2, 4))
    J = 2 * torch.eye(4)
    out = S.in_context_scores(**base, j_norm_pool=(W @ J.T).norm(dim=1))
    assert torch.allclose(torch.as_tensor(out["JL-NORM"]), 2 * torch.as_tensor(out["NORM"]))
    assert "JL-NORM" not in S.in_context_scores(**base)  # absent unless asked: earlier runs are unchanged


def test_budget_walk_stops_at_the_first_prefix_reaching_the_budget():
    g = torch.Generator().manual_seed(1)
    pool = [3, 5, 8, 11]
    da = torch.randn(2, 4, generator=g)  # [|P|, |C|]
    W = torch.randn(4, 6, generator=g)
    col = {f: j for j, f in enumerate(pool)}
    full = (da.T.unsqueeze(-1) * W.unsqueeze(1)).sum(0)  # SAE-ALL as the full delta here
    full_norm = float(full.double().norm())
    order = [11, 3, 8, 5]
    norms = S2._prefix_norms(order, col, da, W, chunk=3)  # the chunked running sum equals the direct one
    direct = [float(sum(da[:, [col[f]]] @ W[[col[f]]] for f in order[: n + 1]).double().norm()) for n in range(4)]
    assert torch.allclose(norms, torch.tensor(direct, dtype=torch.float64))
    universe = [3, 5, 8, 11, 99]  # 99 is inactive on the prompt: skipped, adds nothing
    scores = {11: 4.0, 99: 3.5, 3: 3.0, 8: 2.0, 5: 1.0}
    nb = {"budgets": [0.1, 0.5, 10.0], "selectors": ["X"], "n_random": 3, "curve_k": [1, 2]}
    cx = SimpleNamespace(seed=1, layer=2, site="concept")
    own = {"full_delta": full, "trial": SimpleNamespace(key="t")}
    plan = S2.budget_plan(cx, own, {"X": scores}, {"X": universe}, universe, col, da, W, nb)
    for b, target in (("0.1", 0.1), ("0.5", 0.5)):
        n = next(i for i, v in enumerate(direct) if v >= target * full_norm) + 1
        e = plan["sets"]["X"][b]
        assert e["set"] == tuple(sorted(order[:n])) and e["rho_intended"] >= target
        assert e["depth"] == n + (1 if n > 1 else 0)  # the inactive 99 sits second in the ranking
    assert plan["sets"]["X"]["10.0"] is None  # never reached
    assert plan["curve"]["X"] == {"1": (11,), "2": (11,)}  # top-2 is {11, 99}; 99 is inactive
    assert len(plan["random"]) == 3 and set(S2.plan_sets(plan)) >= {plan["sets"]["X"]["0.5"]["set"], (11,)}


def test_nested_bases_resolve():
    llama = S2._resolve(load_config(REPO / "configs/selection/new_tasks_llama.yaml"))
    assert llama["data_dir"] == "data/built_llama" and llama["norm_budget"]["budgets"] == [0.1, 0.25, 0.5]
    assert llama["k_grid"] == [1, 2, 4, 8, 16, 32] and llama["relations"]["active"] == ["R5_number_agreement", "R6_film_sentiment"]
    assert llama["relations"]["reserve"] is None and llama["eligibility"]["per_relation"] and "base" not in llama
    pt = S2._resolve(load_config(REPO / "configs/selection/new_tasks_gemma_base.yaml"))
    assert pt["extension"]["kind"] == "model" and set(pt["stage0_runs"]) == {"gemma3_12b_pt"}
    x3 = load_config(REPO / "configs/selection/gemma_base.yaml")  # a one-level base resolves as before
    old = {**load_config(x3["base"]), **{k: v for k, v in x3.items() if k != "base"}}
    assert S2._resolve(x3) == old


@pytest.mark.integration
def test_new_tasks_dev_freezes_and_test_runs_once(tmp_path, git_repo):
    from test_inventory import _build, _tiny_model_config
    from test_selection import _stage0

    _build(tmp_path)  # the base fixture build (every family Stage 0 reads)
    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
    data["relations"] = _relations_cfg(tmp_path) | {"active": ["R5"], "reserve": "R6"}
    data["tasks"] = ["R5", "R6"]
    data_path = tmp_path / "data_a36.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert cli.main(["build-data", "--config", str(data_path), "--model", str(tmp_path / "tiny_model.yaml"), "--out", str(tmp_path / "runs" / "build-a36")]) == 0
    assert (tmp_path / "built" / "R5_number_agreement" / "benchmark_card.json").is_file()

    model = _tiny_model_config(tmp_path)
    name = yaml.safe_load(model.read_text())["name"]
    s0 = _stage0(tmp_path, model)
    cfg = S2._resolve(load_config(REPO / "configs/selection/new_tasks_gemma_it.yaml"))
    cfg.pop("_source")
    cfg.update(data_dir=str(tmp_path / "built"), stage0_runs={name: str(s0)}, k_grid=[1, 2], n_random=2)
    cfg["primary_k"] = {**cfg["primary_k"], "default": 2}
    cfg["format_gate"] = {**cfg["format_gate"], "min_own_mass": 0.0, "min_abs_margin": -1.0e9, "min_template_share": 0.0, "test_min_answerable": 0.0}
    cfg["eligibility"] = {"min_median_r_full": -1.0e9, "per_relation": True}
    cfg["m2_min_full_gain"] = -1.0e9
    cfg["lens_fidelity"] = {**cfg["lens_fidelity"], "n_directions": 3, "n_prompts": 2}
    cfg["analysis"] = {**cfg["analysis"], "bootstrap": {**cfg["analysis"]["bootstrap"], "n_resamples": 200}, "p23_min_layers": 2}
    cfg["norm_budget"] = {**cfg["norm_budget"], "budgets": [0.1, 0.25], "n_random": 2, "curve_k": [1, 2, 3]}
    cfg["sae_carried"] = {"min_median_sae_all_recovery": -1.0e9}
    path = tmp_path / "stage2_a36_test.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dev, test = tmp_path / "runs" / "a36-dev", tmp_path / "runs" / "a36-test"
    lines: list[str] = []
    opts = {"config": str(path), "model": str(model), "phase": "dev", "out": str(dev), "device": "cpu"}
    code = execute("select", opts, S2.run, printer=lines.append, repo_root=git_repo)
    err = (dev / "error.txt").read_text(encoding="utf-8") if (dev / "error.txt").exists() else ""
    summary = (dev / "SUMMARY.md").read_text(encoding="utf-8")
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], summary + err
    frozen = json.loads((dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    both = {"R5_number_agreement", "R6_film_sentiment"}
    assert set(frozen["relations"]) == both  # two active relations
    assert frozen["sae_carried_cells"] and "analysis_plan_new_tasks" in frozen and "analysis_plan_ranking_vs_sets" in frozen
    assert set(frozen["eligible_by_relation"]) == both and set(frozen["toolkit_choices"]) == {"k", "budget"}
    choice = frozen["toolkit_choices"]["k"]["R5_number_agreement"]
    assert all(set(ch["means"]) == {"NORM", "ATP", "JL-OUT"} and ch["T0"] in ("NORM", "ATP") for ch in choice.values())
    rep = json.loads((dev / "report.json").read_text(encoding="utf-8"))
    nb = rep["norm_budget"]
    assert set(nb["family"]) == {"JL-OUT - LL-OUT at B = 0.1", "JL-OUT - LL-OUT at B = 0.25"}
    assert set(nb["secondary"]) == {"NORM - JL-OUT", "NORM+JL-OUT - NORM", "JL-NORM - NORM", "JL-NORM - JL-OUT"}
    assert nb["max_realized_vs_intended_gap"] < 1e-4 and "Norm-budget endpoint (E2)" in summary
    a38 = rep["ranking_vs_sets"]
    assert set(a38["dissociation"]) == {f"{d} {t}" for d in ("D1", "D2") for t in both} and set(a38["replication"]) == both
    assert "toolkit" not in a38 and "Ranking against sets, budget dependence, toolkit" in summary  # the toolkit test runs on test only
    with zipfile.ZipFile(dev / "results.zip") as z:
        row = json.loads(z.read("trials.jsonl").decode().splitlines()[0])
    assert {"JL-NORM", "JL-NORM@in", "NORM+JL-OUT"} <= set(row["metrics"]) and "M6" in row["metrics"]["JL-NORM"]
    assert set(row["norm_budget"]["sets"]) == {"NORM", "LL-OUT", "JL-OUT", "ATP", "PATCH", "JL-NORM", "NORM+JL-OUT", "ORACLE"}
    assert len(row["norm_budget"]["random"]["0.25"]) == 2 and set(row["norm_budget"]["curve"]["JL-OUT"]) == {"1", "2", "3"}

    lines = []
    assert execute("select", {"config": str(path), "phase": "test", "frozen": str(dev / "frozen_protocol.json"), "out": str(test), "device": "cpu"},
                   S2.run, printer=lines.append, repo_root=git_repo) == 0, (test / "SUMMARY.md").read_text(encoding="utf-8")
    trep = json.loads((test / "report.json").read_text(encoding="utf-8"))
    assert trep["sae_carried"]["cells"] == frozen["sae_carried_cells"]
    fam = trep["norm_budget"]["family"]
    assert all(v is None or "p_holm" in v for v in fam.values())
    tk = trep["ranking_vs_sets"]["toolkit"]
    assert set(tk) == {"k", "budget"} and set(tk["k"]) == both and trep["toolkit_choices"] == frozen["toolkit_choices"]

    # The ranking-versus-sets test in a model extension (as 12B base extends 12B-IT): the parent's per-relation cells are inherited. With one cell taken
    # from R6 in a re-sealed copy of the parent, R6's trials must not run there.
    from saeselect.provenance import seal_protocol

    parent = {k: v for k, v in frozen.items() if k != "protocol_hash"}
    cell = parent["eligible_by_relation"]["R6_film_sentiment"][0]
    parent["eligible_by_relation"]["R6_film_sentiment"] = parent["eligible_by_relation"]["R6_film_sentiment"][1:]
    parent_path = tmp_path / "parent_frozen.json"
    parent_path.write_text(json.dumps(seal_protocol(parent)), encoding="utf-8")
    ext_cfg = {**cfg, "extension": {"id": "A36-PT", "kind": "model", "parent_frozen": str(parent_path)}}
    ext_path = tmp_path / "stage2_a36_ext.yaml"
    ext_path.write_text(yaml.safe_dump(ext_cfg), encoding="utf-8")
    ext_dev = tmp_path / "runs" / "a36-ext-dev"
    lines = []
    code = execute("select", {"config": str(ext_path), "model": str(model), "phase": "dev", "out": str(ext_dev), "device": "cpu"}, S2.run,
                   printer=lines.append, repo_root=git_repo)
    err = (ext_dev / "error.txt").read_text(encoding="utf-8") if (ext_dev / "error.txt").exists() else ""
    assert code == 0 and "FROZEN_AWAITING_APPROVAL" in lines[-1], (ext_dev / "SUMMARY.md").read_text(encoding="utf-8") + err
    ext_frozen = json.loads((ext_dev / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert cell not in ext_frozen["eligible_by_relation"]["R6_film_sentiment"] and "toolkit_choices" in ext_frozen
    with zipfile.ZipFile(ext_dev / "results.zip") as z:
        rows = [json.loads(x) for x in z.read("trials.jsonl").decode().splitlines()]
    at = {(r["relation"], f"{r['layer']}/{r['site']}") for r in rows}
    assert ("R6_film_sentiment", cell) not in at and ("R5_number_agreement", cell) in at
