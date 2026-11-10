"""Shared fixtures: the tiny random Gemma 3 stack and a throwaway git repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from saeselect import tiny
from saeselect.models import wrap


@pytest.fixture(scope="session")
def tiny_model():
    """Tiny multimodal Gemma 3 in FP32 (6 layers, d=64, vocab 512)."""
    return wrap(tiny.build_hf_model(), tokenizer=None)


@pytest.fixture(scope="session")
def tiny_model_bf16():
    return wrap(tiny.build_hf_model(dtype="bfloat16"), tokenizer=None)


@pytest.fixture(scope="session")
def tiny_sae(tiny_model):
    return tiny.random_sae(tiny_model.d_model, 256, seed=3, layer=2)


@pytest.fixture(scope="session")
def tiny_lens(tiny_model):
    return tiny.random_lens(tiny_model.d_model, list(range(tiny_model.n_layers)), seed=5)


@pytest.fixture
def input_ids():
    gen = torch.Generator().manual_seed(11)
    ids = torch.randint(4, 400, (1, 14), generator=gen)
    ids[0, 0] = 2  # BOS
    return ids


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A committed git repository with one source file and an ignored runs/."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "code.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("runs/\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "STATE.md").write_text("state\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def commit_all(repo: Path, message: str = "change") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
