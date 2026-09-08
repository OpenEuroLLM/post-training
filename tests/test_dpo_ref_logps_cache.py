"""Tests for the ZeRO-stage-independent reference log-prob cache key.

TRL keys its reference log-prob cache on ``hash_module(model)``, which hashes
the bytes of ``state_dict()``.  Under DeepSpeed ZeRO-3 the parameters are
already sharded when ``DPOTrainer.__init__`` runs, so one checkpoint hashes
differently under stage 3 than under stage 2.  The two-stage workflow —
precompute the log-probs under ZeRO-3, then train under ZeRO-2 — therefore
writes a cache the second run never finds, and pays for the precomputation
twice.  That second pass is exactly what ZeRO-3 was chosen to avoid.

``StableCacheKeyDPOTrainer`` swaps the weight hash for a key built from the
model identity, which no DeepSpeed stage can change.  Two things need pinning:
the key itself, and the swap.  The swap must cover the precompute call and
must be undone afterwards, including when the call raises — a leaked patch
would silently disable the weight hash for the rest of the process.
"""

from __future__ import annotations

import pytest
from trl import DPOTrainer
from trl.trainer import dpo_trainer as trl_dpo_trainer

from post_training.config import PostTrainingConfig
from post_training.methods.dpo import StableCacheKeyDPOTrainer, build_ref_logps_cache_key


@pytest.fixture
def config():
    cfg = PostTrainingConfig()
    cfg.model.name_or_path = "allenai/Olmo-3-1025-7B"
    return cfg


# ---------------------------------------------------------------------------
# the key itself
# ---------------------------------------------------------------------------


def test_key_defaults_to_the_model_identity(config):
    assert build_ref_logps_cache_key(config) == "allenai/Olmo-3-1025-7B@main"


def test_key_carries_the_pinned_revision(config):
    """A pinned revision names different weights, so it must change the key."""
    config.model.revision = "abc1234"

    assert build_ref_logps_cache_key(config) == "allenai/Olmo-3-1025-7B@abc1234"


def test_explicit_key_overrides_the_derived_one(config):
    """The escape hatch for weights that change under a fixed path."""
    config.dpo.ref_logps_cache_key = "olmo3-sft-epoch2"

    assert build_ref_logps_cache_key(config) == "olmo3-sft-epoch2"


def test_key_ignores_the_deepspeed_stage(config):
    """The point of the whole change: stage 3 and stage 2 agree on one key."""
    config.deepspeed = {"zero_optimization": {"stage": 3}}
    stage_3_key = build_ref_logps_cache_key(config)

    config.deepspeed = {"zero_optimization": {"stage": 2}}

    assert build_ref_logps_cache_key(config) == stage_3_key


# ---------------------------------------------------------------------------
# the hash_module swap
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Stands in for the tokenized dataset. Only ``_fingerprint`` is read."""

    _fingerprint = "cafebabe"


def _uninitialized_trainer(key: str) -> StableCacheKeyDPOTrainer:
    """Build the subclass without TRL's constructor, which would load a model."""
    trainer = StableCacheKeyDPOTrainer.__new__(StableCacheKeyDPOTrainer)
    trainer._ref_logps_cache_key = key
    return trainer


def test_hash_module_returns_the_key_during_precompute(monkeypatch):
    observed = {}

    def spy(self, dataset, name, batch_size):
        observed["hash"] = trl_dpo_trainer.hash_module(object())
        return dataset

    monkeypatch.setattr(DPOTrainer, "_precompute_ref_logps", spy)

    _uninitialized_trainer("olmo@main")._precompute_ref_logps(_FakeDataset(), "train", 1)

    assert observed["hash"] == "olmo@main"


def test_hash_module_is_restored_after_precompute(monkeypatch):
    original = trl_dpo_trainer.hash_module
    monkeypatch.setattr(DPOTrainer, "_precompute_ref_logps", lambda self, d, n, b: d)

    _uninitialized_trainer("olmo@main")._precompute_ref_logps(_FakeDataset(), "train", 1)

    assert trl_dpo_trainer.hash_module is original


def test_hash_module_is_restored_after_a_failure(monkeypatch):
    """An OOM during precompute must not leave the weight hash disabled."""
    original = trl_dpo_trainer.hash_module

    def raise_oom(self, dataset, name, batch_size):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(DPOTrainer, "_precompute_ref_logps", raise_oom)

    with pytest.raises(RuntimeError):
        _uninitialized_trainer("olmo@main")._precompute_ref_logps(_FakeDataset(), "train", 1)

    assert trl_dpo_trainer.hash_module is original
