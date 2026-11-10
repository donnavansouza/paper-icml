"""Small local stand-ins for every data source, and a data config that points at them.

The config is the real configs/data/relations.yaml with sources replaced by local files
(their hashes computed here) and minimums scaled down, so the build code and
the declared templates, prompts and filters are exercised end to end offline.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from saeselect.provenance import sha256_file

REPO = Path(__file__).resolve().parents[1]

COUNTRIES = [
    # name, capital, languages, independent and UN member
    ("Alpha", "Ardon", ["Alphan"], True),
    ("Bravo", "Bexley", ["Bravan"], True),
    ("Cobalt", "Corin", ["Cobaltic"], True),
    ("Delta", "Dunmore", ["Deltan"], True),
    ("Echo", "Elbury", ["Alphan"], True),  # shares a language with Alpha
    ("Foxtrot", "Fenwick", ["Foxic"], True),
    ("Golfo", "Garrow", ["Golfic"], True),
    ("Hotel", "Harlow", ["Hotelan"], True),
    ("India", "Ilford", ["Indic"], True),
    ("Juliet", "Jarrow", ["Julian"], True),
    ("Kilo", "Kendal", ["Kilan"], True),
    ("Lima", "Ludlow", ["Liman"], True),
    ("Mike", "Morley", ["Mikan"], True),
    ("Novem", "Novem", ["Novan"], True),  # copy case: capital's first token is the country's
    ("Oscar", "Otley", ["Oscan", "Oscaric"], True),  # two languages: out of R4
    ("Papa", "Pinner", ["Papan"], False),  # not a UN member
    ("Qwertyx", "Quorn", ["Qwan"], True),  # unknown word: several tokens
    ("Zyxwvut", "Zennor", ["Zyan"], True),  # unknown word, same length
]

ELEMENTS = [(1, "H", "Hydrogen"), (2, "He", "Helium"), (3, "Li", "Lithium"), (4, "Be", "Beryllium"), (5, "B", "Boron"),
            (6, "C", "Carbon"), (7, "N", "Nitrogen"), (8, "O", "Oxygen"), (9, "F", "Fluorine"), (10, "Ne", "Neon"),
            (11, "Na", "Sodium"), (12, "Mg", "Magnesium")]

VERB_ROWS = [
    ("begin", "began begun"), ("break", "broke broken"), ("bring", "brought brought"), ("choose", "chose chosen"),
    ("drink", "drank drunk"), ("drive", "drove driven"), ("eat", "ate eaten"), ("fall", "fell fallen"),
    ("fly", "flew flown"), ("give", "gave given"), ("go", "went gone"), ("grow", "grew grown"),
    ("cut", "cut cut"),  # copy case
    ("bid", "(in cards) bid bid"), ("bid", "(to request) bade/bid bidden"),  # listed twice
    ("dream", "dreamt/dreamed dreamt/dreamed"),  # two past forms
    ("can", "could -"),  # modal
]

WORDS = [c[0] for c in COUNTRIES[:-2]] + [c[1] for c in COUNTRIES] + sorted({l for c in COUNTRIES for l in c[2]}) \
    + [e[2].lower() for e in ELEMENTS] + [e[1] for e in ELEMENTS] \
    + [w for base, forms in VERB_ROWS for w in [base, *forms.replace("(", " ").replace(")", " ").split()] if w.isalpha()] \
    + ["forbear", "forbore", "yes", "Yes", "YES", "no", "No", "NO", "Reject", "reject", "Invite", "invite", "she", "She", "he", "He", "Ms", "Miss", "Mr", "Mrs",
       "Female", "female", "Male", "male", "Woman", "woman", "Man", "man", "Black", "black", "White", "white",
       "Country", "Capital", "Element", "Symbol", "Verb", "Past", "tense", "The", "capital", "of", "is", "the", "Q", "A",
       "What", "city", "chemical", "symbol", "for", "simple", "past", "verb", "Language", "official", "language",
       "spoken", "in", "officially",
       # R5 nouns (knives is left out on purpose: it splits into several tokens, so knife|knives is excluded), R6 adjectives, answers
       "key", "keys", "box", "boxes", "cup", "cups", "lamp", "lamps", "desk", "desks", "bell", "bells", "coin", "coins", "card",
       "cards", "knife", "good", "great", "fine", "nice", "superb", "bad", "poor", "dull", "weak", "awful", "Is", "are", "Are",
       "was", "Was", "were", "Were", "positive", "Positive", "negative", "Negative", "over", "there", "new"]

NOUNS = [["key", "keys"], ["box", "boxes"], ["cup", "cups"], ["lamp", "lamps"], ["desk", "desks"], ["bell", "bells"],
             ["coin", "coins"], ["card", "cards"], ["knife", "knives"]]
ADJECTIVES = {"positive": ["good", "great", "fine", "nice", "superb"], "negative": ["bad", "poor", "dull", "weak", "awful"]}

def write_sources(root: Path) -> dict:
    """Write every fixture source under ``root`` and return the ``sources`` config."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    countries = [{"name": {"common": n}, "capital": [c], "languages": {f"l{i}": l for i, l in enumerate(langs)},
                  "independent": ok, "unMember": ok, "cca3": n[:3].upper()} for n, c, langs, ok in COUNTRIES]
    (root / "countries.json").write_text(json.dumps(countries), encoding="utf-8")
    table = {"Table": {"Columns": {"Column": ["AtomicNumber", "Symbol", "Name", "AtomicMass"]},
                       "Row": [{"Cell": [str(z), s, n, "1.0"]} for z, s, n in ELEMENTS]}}
    (root / "elements.json").write_text(json.dumps(table), encoding="utf-8")
    wiki = ['{| class="wikitable sortable"', "!verb forms !!verb class and notes"]
    for base, forms in VERB_ROWS:
        wiki += ["|-", f"|''[[{base}]] {forms}''", "| class note"]
    wiki += ["|-", "|''[[bear]] bore/*bare born/borne''", ":''[[forbear]] forbore forborne''", ":''[[misbear|*misbear]] misbore misborne''", "| note", "|}"]
    (root / "verbs.wiki").write_text("\n".join(wiki) + "\n", encoding="utf-8")

    (root / "wikitext/wikitext-103-raw-v1").mkdir(parents=True)
    lines = []
    for a in range(4):
        lines += ["", f" = Article {a} = \n", ""] + [f" Sentence {a} {k} about the article with several words . \n" for k in range(20)]
    pq.write_table(pa.Table.from_pylist([{"text": l} for l in lines]), root / "wikitext/wikitext-103-raw-v1/validation-00000-of-00001.parquet")

    def files(base: Path, rels: list[str]) -> dict[str, str]:
        return {rel: sha256_file(base / rel) for rel in rels}

    return {
        "countries": {"url": "https://example.invalid/countries.json", "path": str(root / "countries.json"),
                      "sha256": sha256_file(root / "countries.json"), "license": "test"},
        "elements": {"url": "https://example.invalid/elements.json", "path": str(root / "elements.json"),
                     "sha256": sha256_file(root / "elements.json"), "license": "test"},
        "irregular_verbs": {"url": "https://example.invalid/verbs?raw", "path": str(root / "verbs.wiki"),
                            "sha256": sha256_file(root / "verbs.wiki"), "license": "test"},
        "wikitext": {"repo_id": "x/wt", "revision": "0" * 40, "local_dir": str(root / "wikitext"), "license": "test",
                     "files": files(root / "wikitext", ["wikitext-103-raw-v1/validation-00000-of-00001.parquet"])},
    }


def write_config(tmp: Path, *, min_test_pairs: int = 2) -> tuple[Path, Path]:
    """(data config path, tiny model config path) for a build into ``tmp``."""
    cfg = copy.deepcopy(yaml.safe_load((REPO / "configs/data/relations.yaml").read_text(encoding="utf-8")))
    cfg["sources"] = write_sources(tmp / "sources")
    cfg["build"]["out_dir"] = str(tmp / "built")
    cfg["build"]["raw_dir"] = str(tmp / "raw")
    cfg["relations"]["dev_pairs"] = 2
    cfg["relations"]["min_test_pairs"] = min_test_pairs
    cfg["neutral"] = {"n_windows": 5, "window_tokens": 16, "source": "wikitext"}
    data_path = tmp / "data.yaml"
    data_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model = {"name": "tiny", "source": "tiny-random", "tokenizer_words": sorted(set(WORDS))}
    model_path = tmp / "tiny_model.yaml"
    model_path.write_text(yaml.safe_dump(model), encoding="utf-8")
    return data_path, model_path
