"""Tests for the ZeRO-stage-independent reference log-prob cache key.

TRL keys its reference log-prob cache on ``hash_module(model)``, which hashes
the bytes of ``state_dict()``.  Under DeepSpeed ZeRO-3 the parameters are
already sharded when ``DPOTrainer.__init__`` runs, so one checkpoint hashes
differently under stage 3 than under stage 2.  The two-stage workflow —
precompute the log-probs under ZeRO-3, then train under ZeRO-2 — therefore
writes a cache the second run never finds, and pays for the precomputation
twice.  That second pass is exactly what ZeRO-3 was chosen to avoid.

``StableCacheKeyDPOTrainer`` swaps the weight hash for a key built from the
model identity and the run name, which no DeepSpeed stage can change.  Two
things need pinning: the key itself, and the swap.  The swap must cover the
precompute call and must be undone afterwards, including when the call
raises — a leaked patch would silently disable the weight hash for the rest
of the process.

The dataset fingerprint is the other half of the key.  From TRL 1.9 it hashes
the whole ``DPOConfig`` unless the subclass narrows ``args`` first.
"""

from __future__ import annotations

import pytest
from datasets import Dataset
from omegaconf import OmegaConf
from trl import DPOConfig, DPOTrainer
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


def test_key_defaults_to_the_model_and_run_name(config):
    key = build_ref_logps_cache_key(config, "dpo-olmo-run")

    assert key == "allenai-Olmo-3-1025-7B-main-dpo-olmo-run"


def test_key_carries_the_pinned_revision(config):
    """A pinned revision names different weights, so it must change the key."""
    config.model.revision = "abc1234"

    key = build_ref_logps_cache_key(config, "dpo-olmo-run")

    assert key == "allenai-Olmo-3-1025-7B-abc1234-dpo-olmo-run"


def test_key_changes_with_the_run_name(config):
    assert build_ref_logps_cache_key(config, "run-a") != build_ref_logps_cache_key(config, "run-b")


def test_explicit_key_is_the_entire_key(config):
    """The key a --tokenize-only run logged, pasted into the training run's config."""
    config.dpo.ref_logps_cache_key = "olmo3-sft-epoch2"
    config.model.revision = "abc1234"

    assert build_ref_logps_cache_key(config, "another-run") == "olmo3-sft-epoch2"


def test_key_from_a_snapshot_path_pastes_into_yaml(config):
    """After prefetch the model is a local snapshot path full of slashes."""
    config.model.name_or_path = "/cache/hub/models--allenai--Olmo/snapshots/0123abc"

    key = build_ref_logps_cache_key(config, "dpo-olmo-run")

    assert key == "cache-hub-models--allenai--Olmo-snapshots-0123abc-main-dpo-olmo-run"
    assert OmegaConf.create(f"ref_logps_cache_key: {key}").ref_logps_cache_key == key


def test_key_ignores_the_deepspeed_stage(config):
    """The point of the whole change: stage 3 and stage 2 agree on one key."""
    config.deepspeed = {"zero_optimization": {"stage": 3}}
    stage_3_key = build_ref_logps_cache_key(config, "dpo-olmo-run")

    config.deepspeed = {"zero_optimization": {"stage": 2}}

    assert build_ref_logps_cache_key(config, "dpo-olmo-run") == stage_3_key


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


# ---------------------------------------------------------------------------
# the dataset half: _prepare_dataset
# ---------------------------------------------------------------------------


def _trl_1_9_prepare_dataset(self, dataset, processing_class, args, dataset_name):
    """The step of TRL 1.9's ``_prepare_dataset`` that hashes ``args``: a filter lambda over it."""
    return dataset.filter(lambda example: len(example["prompt_ids"]) < args.max_length)


def _dpo_config(output_dir, gradient_accumulation_steps, max_length=2):
    return DPOConfig(
        output_dir=str(output_dir),
        report_to="none",
        use_cpu=True,
        bf16=False,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


@pytest.fixture
def tokenized():
    return Dataset.from_dict({"prompt_ids": [[1, 2], [3]]})


def test_raw_dpo_config_leaks_the_allocation_into_the_fingerprint(tmp_path, tokenized):
    """The control: TRL 1.9's filter on the full DPOConfig of a 1-GPU and a 4-GPU job."""
    tokenize_only = _dpo_config(tmp_path / "tokenize", gradient_accumulation_steps=4)
    training = _dpo_config(tmp_path / "train", gradient_accumulation_steps=1)

    tokenize_only_fp = _trl_1_9_prepare_dataset(
        None, tokenized, None, tokenize_only, "train"
    )._fingerprint
    training_fp = _trl_1_9_prepare_dataset(None, tokenized, None, training, "train")._fingerprint

    assert tokenize_only_fp != training_fp


def test_fingerprint_ignores_the_allocation(monkeypatch, tmp_path, tokenized):
    """A 1-GPU precompute run and a 4-GPU training run must share one cache."""
    monkeypatch.setattr(DPOTrainer, "_prepare_dataset", _trl_1_9_prepare_dataset)
    trainer = _uninitialized_trainer("olmo@main")
    tokenize_only = _dpo_config(tmp_path / "tokenize", gradient_accumulation_steps=4)
    training = _dpo_config(tmp_path / "train", gradient_accumulation_steps=1)

    tokenize_only_fp = trainer._prepare_dataset(
        tokenized, None, tokenize_only, "train"
    )._fingerprint
    training_fp = trainer._prepare_dataset(tokenized, None, training, "train")._fingerprint

    assert tokenize_only_fp == training_fp


def test_fingerprint_still_tracks_max_length(monkeypatch, tmp_path, tokenized):
    """max_length decides which rows survive, so it must stay in the fingerprint."""
    monkeypatch.setattr(DPOTrainer, "_prepare_dataset", _trl_1_9_prepare_dataset)
    trainer = _uninitialized_trainer("olmo@main")

    short = trainer._prepare_dataset(
        tokenized, None, _dpo_config(tmp_path, 1, max_length=2), "train"
    )
    long = trainer._prepare_dataset(
        tokenized, None, _dpo_config(tmp_path, 1, max_length=4), "train"
    )

    assert (len(short), len(long)) == (1, 2)
    assert short._fingerprint != long._fingerprint
