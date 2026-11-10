"""Module discovery and the final-norm gain form, on real HF Gemma 3 classes."""

from __future__ import annotations

import pytest
import torch

from saeselect import tiny
from saeselect.models import final_norm_spec, find_layout, wrap
from saeselect.provenance import EngineeringFailed


def test_multimodal_wrapper_layout(tiny_model):
    assert tiny_model.info["model_class"] == "Gemma3ForConditionalGeneration"
    assert tiny_model.info["decoder_path"] == "model.language_model.layers"
    assert tiny_model.info["tied_embeddings"] is True
    assert tiny_model.info["final_norm_gain_form"] == "1+weight"
    assert tiny_model.info["final_logit_softcapping"] is None
    assert tiny_model.info["global_layers"] == [5]
    assert tiny_model.n_layers == 6 and tiny_model.d_model == 64
    assert all(not p.requires_grad for p in tiny_model.hf_model.parameters())


def test_text_only_layout():
    handle = wrap(tiny.build_hf_model(multimodal=False), tokenizer=None)
    assert handle.layout.path == "model"
    assert handle.info["model_class"] == "Gemma3ForCausalLM"


def test_unknown_layout_fails_loudly():
    with pytest.raises(EngineeringFailed, match="layout"):
        find_layout(torch.nn.Linear(2, 2))


def test_gain_form_detection():
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    gemma = Gemma3RMSNorm(8)
    llama = LlamaRMSNorm(8)
    with torch.no_grad():
        gemma.weight.copy_(torch.linspace(-0.5, 0.5, 8))
        llama.weight.copy_(torch.linspace(0.5, 1.5, 8))
    gamma, eps, form = final_norm_spec(gemma, 8)
    assert form == "1+weight" and torch.allclose(gamma, 1 + gemma.weight)
    gamma, eps, form = final_norm_spec(llama, 8)
    assert form == "weight" and torch.allclose(gamma, llama.weight)


def test_softcap_is_read_from_config():
    handle = wrap(tiny.build_hf_model(final_logit_softcapping=30.0), tokenizer=None)
    assert handle.head.softcap == 30.0
