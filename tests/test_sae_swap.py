"""The SAE swap: SAE variants from another release, and compare-sets with declared contrasts and Holm."""

from __future__ import annotations

import yaml

from data_fixtures import REPO
from saeselect import artifacts as A
from saeselect.analysis import compare_sets as C
from saeselect.stages.inventory import variant_key
from test_compare_sets import KS, _rows


def test_variant_key_names_the_release_tag():
    assert variant_key({"layer": 12, "width": "65k", "l0": "medium", "suffix": ""}) == "L12_65k_medium"
    assert variant_key({"layer": 24, "width": "262k", "l0": "medium", "suffix": "seed_1"}) == "L24_262k_medium_seed_1"
    assert variant_key({"layer": 12, "width": "65k", "l0": "medium", "suffix": "", "tag": "pt_sae"}) == "L12_65k_medium_pt_sae"


def test_load_sae_for_reads_the_named_release(monkeypatch):
    seen = {}
    monkeypatch.setattr(A, "load_gemma_scope", lambda **kw: seen.update(kw) or "sae")
    model = yaml.safe_load((REPO / "configs/models/gemma3_12b_it.yaml").read_text(encoding="utf-8"))
    swap = model["sae"]["swap"][0]
    A.load_sae_for(model, layer=12, width="65k", l0="medium", d_model=8, repo_id=swap["repo_id"], revision=swap["revision"])
    assert (seen["repo_id"], seen["revision"]) == ("google/gemma-scope-2-12b-pt", swap["revision"])
    A.load_sae_for(model, layer=12, width="65k", l0="medium", d_model=8)
    assert (seen["repo_id"], seen["revision"]) == (model["sae"]["repo_id"], model["sae"]["revision"])


def test_plan_downloads_groups_sae_files_by_release(monkeypatch):
    calls = []
    monkeypatch.setattr(A, "file_sizes", lambda repo, names, rev, repo_type="model": calls.append((repo, rev, tuple(names))) or {n: 1 for n in names})
    monkeypatch.setattr(A, "_cached", lambda *a: True)
    model = yaml.safe_load((REPO / "configs/models/gemma3_12b_pt.yaml").read_text(encoding="utf-8"))
    swap = model["sae"]["swap"][0]
    cfgs = [{"layer": 12, "width": "65k", "l0": "medium", "suffix": ""},
            {"layer": 12, "width": "65k", "l0": "medium", "suffix": "", "repo_id": swap["repo_id"], "revision": swap["revision"]}]
    needed = A.plan_downloads(model, sae_cfgs=cfgs, include_model=False, include_lens=False)
    assert {c[0] for c in calls} == {"google/gemma-scope-2-12b-pt", "google/gemma-scope-2-12b-it"} and len(needed) == 4


def test_declared_contrasts_with_holm(monkeypatch, tmp_path):
    cfg = yaml.safe_load((REPO / "configs/selection/gemma_it.yaml").read_text(encoding="utf-8"))
    cfg["analysis"]["bootstrap"]["n_resamples"] = 500
    cfg["k_grid"] = KS
    runs = {"ii": _rows(0.30, 0.30, cell=(12, "concept")), "pp": _rows(0.50, 0.30, cell=(12, "concept")),
            "ip": _rows(0.32, 0.30, cell=(12, "concept")), "pi": _rows(0.48, 0.30, cell=(12, "concept"))}
    monkeypatch.setattr(C, "load", lambda entry: (runs[entry["run"]], cfg, {"primary_k": 32}, []))
    a25 = yaml.safe_load((REPO / "configs/analysis/sae_swap.yaml").read_text(encoding="utf-8"))
    names = {"it_weights_it_sae": "ii", "pt_weights_pt_sae": "pp", "it_weights_pt_sae": "ip", "pt_weights_it_sae": "pi"}
    for name, entry in a25["models"].items():
        entry["run"] = names[name]
    from types import SimpleNamespace

    ctx = SimpleNamespace(config=a25, flags=[])
    out = C.run(ctx).report["cross_model"]
    # weights carry the difference here: about +0.18 either way; the SAE release about +0.02
    assert abs(out["weights effect, IT SAEs"]["32"]["estimate"] - 0.18) < 0.005 and out["weights effect, IT SAEs"]["32"]["better_after_holm"]
    assert abs(out["SAE effect, IT weights"]["32"]["estimate"] - 0.02) < 0.005 and out["SAE effect, IT weights"]["32"]["equivalent"]
    assert all("p_holm" in out[n]["32"] for n in out) and all("p_holm" not in out[n]["4"] for n in out)
    assert "_verdict" not in out["weights effect, PT SAEs"]["4"]
