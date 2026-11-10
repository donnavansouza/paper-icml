"""Stage 2 relations: concepts, filters, exemplars, pairs, templates, records.

R1 to R4:
- concepts pair only with concepts of the same token count, so source and
  donor prompts align position by position; the concept site is every concept
  token;
- a concept whose answer's first token equals the concept's first token is
  excluded;
- answer first tokens are distinct within a pair, including every declared
  surface variant; pairs that share an answer are impossible by construction;
- two exemplar concepts per relation are reserved and appear in all three
  templates.

Records: for each pair (c, d), template t and direction (source, donor) there is
one source record and one donor record. Both carry the same answer sets:
A_plus = the donor's answer, A_minus = the source's answer, so one margin
m = LSE(z[donor answer]) - LSE(z[source answer]) is read on both prompts.

R5 and R6 are relations built from declared item files in the repository: R5
(subject-verb number agreement: the singular and plural of one noun form a
pair, answers {is, was} against {are, were}) and R6 (film-review sentiment,
the reserve). A relation may declare ``answer_forms`` (several base answers
per class), ``exemplar_answers`` (one reserved exemplar per answer class, in
that order, its lemma leaving the pool) and ``pairing: same_lemma``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import yaml

from saeselect.tasks.records import TaskRecord, hash_unit, split_by_count
from saeselect.tasks.tokenization import answer_token_ids, encode_prompt, first_token_id, piece_ids, token_positions


@dataclass
class Concept:
    name: str
    answer: str
    source_id: str
    token_ids: list[int] = field(default_factory=list)
    answer_ids: list[int] = field(default_factory=list)
    answer_strings: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


# --------------------------------------------------------------------------
# Source parsing
# --------------------------------------------------------------------------


def parse_countries(path, *, key: str) -> tuple[list[Concept], list[dict]]:
    data = json.loads(open(path, encoding="utf-8").read())
    concepts, excluded = [], []
    for entry in data:
        name = entry["name"]["common"]
        if not (entry.get("independent") and entry.get("unMember")):
            excluded.append({"item": name, "reason": "not an independent UN member"})
            continue
        if key == "capital":
            capitals = entry.get("capital", [])
            if len(capitals) != 1:
                excluded.append({"item": name, "reason": f"{len(capitals)} capitals listed"})
                continue
            concepts.append(Concept(name=name, answer=capitals[0], source_id=entry.get("cca3", name)))
        elif key == "language":
            languages = list(entry.get("languages", {}).values())
            if len(languages) != 1:
                excluded.append({"item": name, "reason": f"{len(languages)} languages listed"})
                continue
            concepts.append(Concept(name=name, answer=languages[0], source_id=entry.get("cca3", name)))
        else:
            raise ValueError(key)
    return concepts, excluded


def parse_elements(path, *, lower: bool) -> tuple[list[Concept], list[dict]]:
    table = json.loads(open(path, encoding="utf-8").read())["Table"]
    columns = table["Columns"]["Column"]
    i_num, i_sym, i_name = columns.index("AtomicNumber"), columns.index("Symbol"), columns.index("Name")
    concepts = []
    for row in table["Row"]:
        cell = row["Cell"]
        name = cell[i_name].lower() if lower else cell[i_name]
        concepts.append(Concept(name=name, answer=cell[i_sym], source_id=f"Z{cell[i_num]}"))
    return concepts, []


_VERB_ROW = re.compile(r"^[|:]''\[\[([^\]|]+)(?:\|([^\]]*))?\]\][^ ]* (.*?)''", re.MULTILINE)


def parse_irregular_verbs(path, *, modal_verbs: list[str]) -> tuple[list[Concept], list[dict]]:
    """Headword ('|' rows) and compound (':' rows) entries of the Wiktionary appendix."""
    text = open(path, encoding="utf-8").read()
    rows = _VERB_ROW.findall(text)
    words = Counter((display or target).lstrip("*") for target, display, _ in rows)
    concepts, excluded = [], []
    for target, display, rest in rows:
        word = display or target
        if word.startswith("*"):
            excluded.append({"item": word, "reason": "archaic or rare form (*)"})
            continue
        if words[word] > 1:
            excluded.append({"item": word, "reason": "listed more than once (different senses)"})
            continue
        if word in modal_verbs:
            excluded.append({"item": word, "reason": "modal verb"})
            continue
        forms = re.sub(r"\([^)]*\)", "", rest).split()
        if not forms:
            excluded.append({"item": word, "reason": "no forms parsed"})
            continue
        past = forms[0]
        if any(ch in past for ch in "/*<[") or not past.isalpha():
            excluded.append({"item": word, "reason": f"simple past {past!r} is not a single form"})
            continue
        if " " in word or not word.isalpha():
            excluded.append({"item": word, "reason": "not a single word"})
            continue
        concepts.append(Concept(name=word, answer=past, source_id=word))
    return concepts, excluded


def parse_declared(path, *, kind: str, answers: dict[str, str]) -> tuple[list[Concept], list[dict]]:
    """Concepts from a declared item file. ``number_agreement``: ``nouns`` lists [singular, plural]; both forms carry
    the noun's singular as their lemma (``source_id``). ``word_classes``: one list of words per answer class. ``answers``
    maps each class (singular, plural; or the class names) to its answer."""
    data = yaml.safe_load(open(path, encoding="utf-8").read())
    items: list[tuple[str, str, str]] = []  # (word, class, lemma)
    if kind == "number_agreement":
        for singular, plural in data["nouns"]:
            items += [(singular, "singular", singular), (plural, "plural", singular)]
    elif kind == "word_classes":
        items = [(w, cls, w) for cls in answers for w in data[cls]]
    else:
        raise ValueError(kind)
    counts = Counter(w for w, _, _ in items)
    concepts, excluded = [], []
    for word, cls, lemma in items:
        if counts[word] > 1:
            excluded.append({"item": word, "reason": "listed more than once"})
        elif not word.isalpha():
            excluded.append({"item": word, "reason": "not a single alphabetic word"})
        else:
            concepts.append(Concept(name=word, answer=answers[cls], source_id=lemma))
    return concepts, excluded


# --------------------------------------------------------------------------
# Variants, filters, pairing
# --------------------------------------------------------------------------


def expand_variants(patterns: list[str], answer: str) -> list[str]:
    upper = answer[:1].upper() + answer[1:]
    return [p.replace("{a}", answer).replace("{A}", upper) for p in patterns]


def tokenize_concepts(tokenizer, concepts: list[Concept], patterns: list[str], forms: dict[str, list[str]] | None = None) -> None:
    """``forms`` maps an answer to the base forms its set holds (e.g. is -> is, was); by default the answer alone."""
    for c in concepts:
        c.token_ids = piece_ids(tokenizer, " " + c.name)
        variants = [v for form in (forms or {}).get(c.answer, [c.answer]) for v in expand_variants(patterns, form)]
        c.answer_ids, c.answer_strings = answer_token_ids(tokenizer, variants)


def compatible(a: Concept, b: Concept) -> bool:
    return a.answer != b.answer and not (set(a.answer_ids) & set(b.answer_ids))


def pair_concepts(concepts: list[Concept], *, key: str, seed: int) -> tuple[list[tuple[Concept, Concept]], list[dict]]:
    """Seeded, disjoint pairing within equal-token-count groups; greedy on compatibility."""
    pairs, leftover = [], []
    groups: dict[int, list[Concept]] = {}
    for c in concepts:
        groups.setdefault(c.n_tokens, []).append(c)
    for n in sorted(groups):
        order = sorted(groups[n], key=lambda c: (hash_unit(f"{key}:pair:{c.name}", seed), c.name))
        unpaired = list(order)
        while unpaired:
            first = unpaired.pop(0)
            partner = next((c for c in unpaired if compatible(first, c)), None)
            if partner is None:
                leftover.append({"item": first.name, "reason": f"no compatible partner with {n} tokens"})
                continue
            unpaired.remove(partner)
            pairs.append((first, partner))
    return pairs, leftover


def pair_same_lemma(concepts: list[Concept], *, answer_order: list[str]) -> tuple[list[tuple[Concept, Concept]], list[dict]]:
    """R5: the two forms of one lemma are a pair, in ``answer_order`` (singular first), when they have the same
    token count and disjoint answer first tokens."""
    by_lemma: dict[str, list[Concept]] = {}
    for c in concepts:
        by_lemma.setdefault(c.source_id, []).append(c)
    pairs, leftover = [], []
    for lemma in sorted(by_lemma):
        forms = sorted(by_lemma[lemma], key=lambda c: answer_order.index(c.answer))
        if [c.answer for c in forms] != list(answer_order):
            leftover.append({"item": lemma, "reason": f"forms {[c.name for c in forms]} do not give one per answer"})
            continue
        a, b = forms
        if a.n_tokens != b.n_tokens:
            leftover.append({"item": f"{a.name}|{b.name}", "reason": f"forms differ in token count ({a.n_tokens} and {b.n_tokens})"})
            continue
        if not compatible(a, b):
            leftover.append({"item": f"{a.name}|{b.name}", "reason": "answer sets share a first token"})
            continue
        pairs.append((a, b))
    return pairs, leftover


def fill(template: str, exemplars: list[Concept], query: str) -> tuple[str, tuple[int, int]]:
    """The prompt and the character span of the query concept."""
    values = {"c1": exemplars[0].name, "a1": exemplars[0].answer, "c2": exemplars[1].name, "a2": exemplars[1].answer}
    before, after = template.split("{q}")
    before = before.format(**values)
    after = after.format(**values)
    return before + query + after, (len(before), len(before) + len(query))


def aligned(tokenizer, template: str, exemplars: list[Concept], a: Concept, b: Concept) -> str | None:
    """None if the two prompts align position by position with the concept tokens
    as tokenized in isolation; otherwise the reason they do not."""
    encoded = []
    for c in (a, b):
        prompt, span = fill(template, exemplars, c.name)
        ids, offsets = encode_prompt(tokenizer, prompt, "raw", offsets=True)
        positions = token_positions(offsets, span)
        if [ids[p] for p in positions] != c.token_ids:
            return f"{c.name!r} tokenizes differently in context"
        encoded.append((ids, positions))
    (ids_a, pos_a), (ids_b, pos_b) = encoded
    if len(ids_a) != len(ids_b) or pos_a != pos_b:
        return "prompts differ in length or concept position"
    if any(x != y for i, (x, y) in enumerate(zip(ids_a, ids_b)) if i not in pos_a):
        return "prompts differ outside the concept tokens"
    return None


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_relation(rel: str, cfg: dict[str, Any], rel_cfg: dict[str, Any], *, concepts: list[Concept], source_excluded: list[dict], tokenizer, seed: int) -> dict[str, Any]:
    """Filter, reserve exemplars, pair, split and emit records for one relation."""
    patterns = cfg["answer_variants"]
    counts = {"source": len(concepts) + len(source_excluded), "after_source_rules": len(concepts)}
    excluded = list(source_excluded)
    tokenize_concepts(tokenizer, concepts, patterns, rel_cfg.get("answer_forms"))

    kept = []
    for c in concepts:
        if cfg["exclude_copy_answers"] and first_token_id(tokenizer, " " + c.answer) == c.token_ids[0]:
            excluded.append({"item": c.name, "reason": "answer's first token equals the concept's first token"})
            continue
        kept.append(c)
    counts["after_copy_filter"] = len(kept)

    n_ex = cfg["exemplars"]
    ranked = sorted(kept, key=lambda c: (hash_unit(f"{rel}:exemplar:{c.name}", seed), c.name))
    order = rel_cfg.get("exemplar_answers")
    if order:  # one exemplar per answer class, in the declared order; an exemplar's lemma leaves the pool
        if len(order) != n_ex:
            raise ValueError(f"{rel}: exemplar_answers {order} does not give {n_ex} exemplars")
        exemplars, used = [], set()
        for answer in order:
            pick = next((c for c in ranked if c.answer == answer and c.source_id not in used), None)
            if pick is None:
                raise ValueError(f"{rel}: no concept left with answer {answer!r} to reserve as an exemplar")
            exemplars.append(pick)
            used.add(pick.source_id)
        pool = [c for c in kept if c.source_id not in used]
    else:
        exemplars = ranked[:n_ex]
        pool = [c for c in kept if c not in exemplars]

    if rel_cfg.get("pairing") == "same_lemma":
        pairs, leftover = pair_same_lemma(pool, answer_order=order)
    else:
        pairs, leftover = pair_concepts(pool, key=rel, seed=seed)
    excluded += leftover
    good_pairs = []
    for a, b in pairs:
        reason = next((r for t in rel_cfg["templates"].values() if (r := aligned(tokenizer, t, exemplars, a, b))), None)
        if reason:
            excluded.append({"item": f"{a.name}|{b.name}", "reason": f"pair dropped: {reason}"})
            continue
        good_pairs.append((a, b))
    counts["pairs"] = len(good_pairs)

    pair_ids = {f"{rel}:{a.name}|{b.name}": (a, b) for a, b in good_pairs}
    n_dev = min(cfg["dev_pairs"], len(pair_ids))
    splits = split_by_count(pair_ids, seed=seed, n_dev=n_dev) if pair_ids else {}
    n_test = sum(1 for s in splits.values() if s == "test")
    counts.update(dev_pairs=n_dev, test_pairs=n_test)

    records: list[TaskRecord] = []
    for pair_id, (a, b) in sorted(pair_ids.items()):
        for t_id, template in rel_cfg["templates"].items():
            for src, don in ((a, b), (b, a)):
                group = f"{pair_id}|{t_id}|{src.name}->{don.name}"
                answers = {
                    "A_plus": {"ids": don.answer_ids, "strings": don.answer_strings},
                    "A_minus": {"ids": src.answer_ids, "strings": src.answer_strings},
                }
                for role, concept in (("source", src), ("donor", don)):
                    prompt, span = fill(template, exemplars, concept.name)
                    ids, offsets = encode_prompt(tokenizer, prompt, "raw", offsets=True)
                    records.append(
                        TaskRecord(
                            id=f"{group}|{role}",
                            cluster_id=pair_id,
                            split=splits[pair_id],
                            family=rel,
                            template_id=t_id,
                            prompt=prompt,
                            prompt_format="raw",
                            role=role,
                            answer_sets=answers,
                            attribute={"concept": concept.name, "answer": concept.answer},
                            attribute_spans=[list(span)],
                            matched_group_id=group,
                            meta={
                                "relation": rel_cfg["name"],
                                "concept_positions": token_positions(offsets, span),
                                "concept_n_tokens": concept.n_tokens,
                                "readout_position": len(ids) - 1,
                                "n_tokens": len(ids),
                                "source_concept": src.name,
                                "donor_concept": don.name,
                            },
                        ).validate()
                    )
    below_minimum = n_test < cfg["min_test_pairs"] or n_dev < cfg["dev_pairs"]
    card = {
        "task": rel,
        "relation": rel_cfg["name"],
        "rules": rel_cfg["keep"],
        "templates": rel_cfg["templates"],
        "answer_variants": patterns,
        "exemplars": [{"concept": c.name, "answer": c.answer} for c in exemplars],
        "counts": counts,
        "pairs_by_token_count": dict(sorted(Counter(a.n_tokens for a, _ in good_pairs).items())),
        "pairs": {
            split: sorted(pid.split(":", 1)[1] for pid, s in splits.items() if s == split) for split in ("dev", "test")
        },
        "excluded": excluded,
        "minimum": {"dev_pairs": cfg["dev_pairs"], "test_pairs": cfg["min_test_pairs"], "met": not below_minimum},
    }
    for key in ("answer_forms", "exemplar_answers", "pairing"):
        if rel_cfg.get(key):
            card[key] = rel_cfg[key]
    return {"records": records, "card": card, "splits": splits, "below_minimum": below_minimum}
