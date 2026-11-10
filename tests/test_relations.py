"""Relations: source rules, the copy filter, equal-token-count pairing, alignment,
reserved exemplars, disjoint answer sets, and determinism."""

from __future__ import annotations

import yaml

from data_fixtures import REPO, WORDS, write_sources
from saeselect import tiny
from saeselect.tasks import relations as R


def _cfg():
    cfg = yaml.safe_load((REPO / "configs/data/relations.yaml").read_text(encoding="utf-8"))["relations"]
    cfg["dev_pairs"], cfg["min_test_pairs"] = 2, 2
    return cfg


def test_source_rules(tmp_path):
    src = write_sources(tmp_path)
    concepts, excluded = R.parse_countries(src["countries"]["path"], key="capital")
    names = {c.name for c in concepts}
    assert "Papa" not in names and any(e["item"] == "Papa" for e in excluded)
    langs, lexcl = R.parse_countries(src["countries"]["path"], key="language")
    assert "Oscar" not in {c.name for c in langs} and any(e["item"] == "Oscar" for e in lexcl)
    verbs, vexcl = R.parse_irregular_verbs(src["irregular_verbs"]["path"], modal_verbs=["can"])
    got = {c.name: c.answer for c in verbs}
    assert got["go"] == "went" and got["forbear"] == "forbore"
    reasons = {e["item"]: e["reason"] for e in vexcl}
    assert "listed more than once" in reasons["bid"]
    assert "not a single form" in reasons["dream"]
    assert reasons["can"] == "modal verb"
    assert "archaic" in reasons["*misbear"] and "not a single form" in reasons["bear"]
    elements, _ = R.parse_elements(src["elements"]["path"], lower=True)
    assert ("carbon", "C") in {(c.name, c.answer) for c in elements}


def test_build_relation_pairs_align_and_filter(tmp_path):
    src = write_sources(tmp_path)
    tok = tiny.build_tokenizer(WORDS)
    cfg = _cfg()
    concepts, excluded = R.parse_countries(src["countries"]["path"], key="capital")
    out = R.build_relation("R1", cfg, cfg["R1"], concepts=concepts, source_excluded=excluded, tokenizer=tok, seed=7)
    card = out["card"]
    assert any(e["item"] == "Novem" and "first token" in e["reason"] for e in card["excluded"])
    exemplars = {e["concept"] for e in card["exemplars"]}
    assert len(exemplars) == 2
    paired = {n for split in card["pairs"].values() for pid in split for n in pid.split("|")}
    assert not exemplars & paired and "Novem" not in paired
    assert card["pairs_by_token_count"].get(7) == 1  # the two unknown 7-letter names pair with each other
    assert card["counts"]["dev_pairs"] == 2 and not out["below_minimum"]
    groups: dict[str, list] = {}
    for r in out["records"]:
        assert not set(r.answer_sets["A_plus"]["ids"]) & set(r.answer_sets["A_minus"]["ids"])
        assert r.meta["concept_positions"][-1] < r.meta["readout_position"]
        groups.setdefault(r.matched_group_id, []).append(r)
    for group in groups.values():
        src_rec = next(r for r in group if r.role == "source")
        don_rec = next(r for r in group if r.role == "donor")
        assert src_rec.meta["n_tokens"] == don_rec.meta["n_tokens"]
        assert src_rec.meta["concept_positions"] == don_rec.meta["concept_positions"]
        assert src_rec.answer_sets == don_rec.answer_sets
        assert src_rec.attribute["answer"] in src_rec.answer_sets["A_minus"]["strings"][0]
    assert len(out["records"]) == 12 * card["counts"]["pairs"]
    again = R.build_relation("R1", cfg, cfg["R1"], concepts=R.parse_countries(src["countries"]["path"], key="capital")[0],
                             source_excluded=excluded, tokenizer=tok, seed=7)
    assert [r.to_json() for r in again["records"]] == [r.to_json() for r in out["records"]]


def test_language_pairs_never_share_an_answer(tmp_path):
    src = write_sources(tmp_path)
    tok = tiny.build_tokenizer(WORDS)
    cfg = _cfg()
    concepts, excluded = R.parse_countries(src["countries"]["path"], key="language")
    out = R.build_relation("R4", cfg, cfg["R4"], concepts=concepts, source_excluded=excluded, tokenizer=tok, seed=7)
    for r in out["records"]:
        assert r.answer_sets["A_plus"]["strings"][0].strip() != r.answer_sets["A_minus"]["strings"][0].strip()


def test_below_minimum_is_reported(tmp_path):
    src = write_sources(tmp_path)
    cfg = _cfg()
    cfg["min_test_pairs"] = 50
    concepts, excluded = R.parse_elements(src["elements"]["path"], lower=True)
    out = R.build_relation("R2", cfg, cfg["R2"], concepts=concepts, source_excluded=excluded, tokenizer=tiny.build_tokenizer(WORDS), seed=7)
    assert out["below_minimum"] and out["card"]["minimum"]["met"] is False
