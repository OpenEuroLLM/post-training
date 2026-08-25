"""Tests for which repos prefetch_assets downloads."""

import pytest

from post_training.config import PostTrainingConfig
from post_training.utils import prefetch


@pytest.fixture
def downloads(monkeypatch):
    """Record snapshot_download calls instead of contacting the Hub."""
    calls = []

    def fake_snapshot_download(repo_id, repo_type=None, revision=None, allow_patterns=None):
        calls.append((repo_id, revision, allow_patterns is not None))
        return f"/cache/{repo_id}@{revision or 'main'}"

    monkeypatch.setattr(prefetch, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(prefetch, "_prefetch_dataset", lambda entry: None)
    return calls


def _config(**model_overrides) -> PostTrainingConfig:
    config = PostTrainingConfig()
    config.model.name_or_path = "org/model"
    for key, value in model_overrides.items():
        setattr(config.model, key, value)
    return config


def test_tokenizer_reuses_the_model_snapshot(downloads):
    """The model snapshot already holds the tokenizer, so nothing extra is fetched."""
    paths = prefetch.prefetch_assets(_config(revision="r1"))

    assert downloads == [("org/model", "r1", False)]
    assert paths.tokenizer == paths.model


def test_tokenizer_revision_is_fetched_separately(downloads):
    """A tokenizer pinned to another revision of the model repo needs its own snapshot."""
    paths = prefetch.prefetch_assets(_config(revision="r1", tokenizer_revision="r2"))

    assert downloads == [("org/model", "r1", False), ("org/model", "r2", True)]
    assert paths.model == "/cache/org/model@r1"
    assert paths.tokenizer == "/cache/org/model@r2"


def test_model_revision_is_ignored_when_tokenizer_name_is_specified(downloads):
    """The model revision is ignored when a tokenizer revision is specified."""
    paths = prefetch.prefetch_assets(_config(revision="r1", tokenizer_name_or_path="org/tokenizer"))

    assert downloads == [("org/model", "r1", False), ("org/tokenizer", None, True)]
    assert paths.model == "/cache/org/model@r1"
    assert paths.tokenizer == "/cache/org/tokenizer@main"


def test_tokenizer_repo_is_fetched_without_weights(downloads):
    """A tokenizer from another repo is fetched with allow_patterns, so no weights."""
    paths = prefetch.prefetch_assets(_config(tokenizer_name_or_path="org/tokenizer"))

    assert downloads == [("org/model", None, False), ("org/tokenizer", None, True)]
    assert paths.tokenizer == "/cache/org/tokenizer@main"


def test_local_tokenizer_path_is_not_downloaded(downloads, tmp_path):
    """An existing local tokenizer directory is used as-is."""
    paths = prefetch.prefetch_assets(_config(tokenizer_name_or_path=str(tmp_path)))

    assert downloads == [("org/model", None, False)]
    assert paths.tokenizer == str(tmp_path)
