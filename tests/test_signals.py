"""Selection signals (5.4) and ranking metrics (5.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saeselect import lens as L
from saeselect import signals as S
from saeselect.readout import AnswerSets, output_direction


def test_in_context_formulas_match_hand_computation():
    gen = torch.Generator().manual_seed(0)
    da = torch.randn(3, 5, generator=gen)  # 3 positions, 5 pool features
    W = torch.randn(5, 8, generator=gen)
    sd = torch.tensor([1.0, 2.0, 0.5, 0.0, 4.0])
    s = {"LL-OUT": torch.randn(8, generator=gen), "JL-OUT": torch.randn(8, generator=gen),
         "LL-VERB": torch.randn(8, generator=gen), "JL-VERB": torch.randn(8, generator=gen)}
    grad = torch.randn(3, 8, generator=gen)
    out = S.in_context_scores(da=da, W_dec_pool=W, sd_pool=sd, lens_s=s, grad_site=grad)
    np.testing.assert_allclose(out["LL-OUT"], (da.sum(0) * (W @ s["LL-OUT"])).numpy(), rtol=1e-5)
    np.testing.assert_allclose(out["ATP"], sum(da[p] * (W @ grad[p]) for p in range(3)).numpy(), rtol=1e-5)
    np.testing.assert_allclose(out["NORM"], (da.abs().sum(0) * W.norm(dim=1)).numpy(), rtol=1e-5)
    assert out["ASSOC"][3] == 0.0  # a feature with sd 0 gets no association score
    assert out["ASSOC"][0] == pytest.approx(float(da[:, 0].abs().sum()), rel=1e-6)


def test_logit_lens_and_jlens_coincide_when_J_is_identity(tiny_model, input_ids):
    from saeselect.patching import clean_run

    _, h_final = clean_run(tiny_model, input_ids, [], mode="bf16")
    h_f = h_final[-1]
    answers = AnswerSets(plus=(5, 6), minus=(7,))
    w = output_direction(h_f, tiny_model.head, answers)
    s_ll = L.lens_sensitivity(w, h_f, tiny_model.head, None)
    s_jl = L.lens_sensitivity(w, h_f, tiny_model.head, torch.eye(tiny_model.d_model))
    torch.testing.assert_close(s_ll, s_jl)


def test_support_selection_and_random():
    support = S.support_scores([{1: 2.0, 2: 1.0}, {2: 3.0, 3: -1.0}])
    assert support == {1: 1.0, 2: 2.0, 3: -0.5}  # a feature missing from a support trial counts 0 there
    assert S.select_top(support, 2, [1, 2, 3, 4]) == [2, 1]
    assert S.select_top({}, 2, [5, 3, 4]) == [3, 4]  # all ties: by feature id
    rng = np.random.default_rng(0)
    draw = S.random_draw([1, 2, 3, 4, 5], 3, rng)
    assert len(draw) == 3 and set(draw) <= {1, 2, 3, 4, 5}


def test_ranking_metrics():
    effects = {1: 0.1, 2: 0.5, 3: -0.2, 4: 0.9}
    perfect = {1: 1.0, 2: 2.0, 3: 0.0, 4: 3.0}
    out = S.ranking_metrics(perfect, effects, ks=[1, 2], control={1: 3.0, 2: 1.0, 3: 2.0, 4: 0.5})
    assert out["M1"] == pytest.approx(1.0) and out["M3"]["2"] == pytest.approx(1.0) and out["M6"] is not None
    constant = S.ranking_metrics({}, effects, ks=[1])
    assert constant["M1"] is None  # a signal that scores every pool feature the same has no ranking
    assert S.recovery(2.0, 1.0, 3.0) == pytest.approx(0.5)
