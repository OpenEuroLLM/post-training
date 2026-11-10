"""Tests for the stage-agnostic DPO reference log-prob cache key.

TRL names the precomputed reference-logprob cache (``*.npz`` next to the dataset
cache) ``Hasher.hash((dataset._fingerprint, hash_module(model)))``, and
``hash_module`` hashes the in-memory ``state_dict()``.  Under DeepSpeed ZeRO-3
the parameters are already partitioned when ``DPOTrainer.__init__`` computes the
hash, so the same checkpoint produces a different cache file per ZeRO stage and
per world size, and the cache is silently recomputed (OpenEuroLLM/post-training
#62).  ``post_training.methods.dpo`` replaces the hash with one over the
checkpoint files on disk, which does not depend on the partitioning.
"""

import json
import types

import pytest
import torch
import yaml
from trl.trainer.utils import hash_module as trl_hash_module

from post_training.config import PostTrainingConfig
from post_training.methods.dpo import (
    hash_checkpoint_files,
    install_stage_agnostic_hash_module,
)


def _write_checkpoint(root, weight_bytes: bytes = b"weights-v1", config: dict | None = None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or {"model_type": "llama"}))
    (root / "model.safetensors").write_bytes(weight_bytes)
    return root


def test_hash_is_stable_across_checkpoint_copies(tmp_path):
    a = _write_checkpoint(tmp_path / "a")
    b = _write_checkpoint(tmp_path / "b")
    assert hash_checkpoint_files(str(a), None, "bfloat16") == hash_checkpoint_files(
        str(b), None, "bfloat16"
    )


def test_hash_changes_when_weights_change(tmp_path):
    a = _write_checkpoint(tmp_path / "a", weight_bytes=b"weights-v1")
    b = _write_checkpoint(tmp_path / "b", weight_bytes=b"weights-v2")
    assert hash_checkpoint_files(str(a), None, "bfloat16") != hash_checkpoint_files(
        str(b), None, "bfloat16"
    )


def test_hash_changes_with_dtype(tmp_path):
    a = _write_checkpoint(tmp_path / "a")
    assert hash_checkpoint_files(str(a), None, "bfloat16") != hash_checkpoint_files(
        str(a), None, "float32"
    )


def test_hash_changes_when_config_changes(tmp_path):
    a = _write_checkpoint(tmp_path / "a", config={"model_type": "llama"})
    b = _write_checkpoint(tmp_path / "b", config={"model_type": "qwen2"})
    assert hash_checkpoint_files(str(a), None, "bfloat16") != hash_checkpoint_files(
        str(b), None, "bfloat16"
    )


def test_hash_ignores_non_weight_files(tmp_path):
    a = _write_checkpoint(tmp_path / "a")
    before = hash_checkpoint_files(str(a), None, "bfloat16")
    (a / "tokenizer.json").write_text("{}")
    (a / "README.md").write_text("not part of the checkpoint identity")
    assert hash_checkpoint_files(str(a), None, "bfloat16") == before


def test_hash_requires_config_or_weight_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        hash_checkpoint_files(str(empty), None, "bfloat16")


def test_installed_hash_ignores_partitioned_state_dict(tmp_path):
    """Two partition views of one checkpoint must share one cache key.

    The first assertion pins the upstream behavior that caused the bug: TRL's
    ``hash_module`` hashes whatever ``state_dict()`` returns, so two different
    partitionings of the same weights hash differently.  The installed
    replacement must return the same file-based hash for both.
    """
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    view_a = torch.nn.Linear(4, 4)
    view_b = torch.nn.Linear(4, 4)
    with torch.no_grad():
        view_b.weight.copy_(view_a.weight + 1.0)  # stand-in for a different shard
    assert trl_hash_module(view_a) != trl_hash_module(view_b)

    fake_module = types.SimpleNamespace(hash_module=trl_hash_module)
    expected = hash_checkpoint_files(str(ckpt), None, "bfloat16")
    installed = install_stage_agnostic_hash_module(fake_module, str(ckpt), None, "bfloat16")

    assert installed == expected
    assert fake_module.hash_module(view_a) == expected
    assert fake_module.hash_module(view_b) == expected


def test_install_patches_trl_trainer_global_lookup(tmp_path, monkeypatch):
    """DPOTrainer resolves ``hash_module`` as a module global at call time."""
    import trl.trainer.dpo_trainer as dpo_trainer_module
    from trl.trainer.dpo_trainer import DPOTrainer

    monkeypatch.setattr(dpo_trainer_module, "hash_module", dpo_trainer_module.hash_module)
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    expected = hash_checkpoint_files(str(ckpt), None, "bfloat16")

    installed = install_stage_agnostic_hash_module(dpo_trainer_module, str(ckpt), None, "bfloat16")

    assert installed == expected
    assert (
        DPOTrainer._precompute_ref_logps.__globals__["hash_module"]
        is dpo_trainer_module.hash_module
    )
    assert DPOTrainer._precompute_ref_logps.__globals__["hash_module"](object()) == expected


def _minimal_dpo_config(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "dpo",
                "backend": "trl",
                "container": None,
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                    "lr_scheduler_kwargs": None,
                    "gradient_checkpointing_kwargs": None,
                },
                "deepspeed": None,
                "data": {
                    "datasets": [{"name": "dummy", "path": "dummy/path", "weight": 1.0}],
                },
            }
        )
    )
    return str(config_path)


def test_precompute_ref_log_probs_defaults_to_false(tmp_path):
    config = PostTrainingConfig.load(_minimal_dpo_config(tmp_path))
    assert config.dpo.precompute_ref_log_probs is False


def test_precompute_ref_log_probs_cli_override(tmp_path):
    config = PostTrainingConfig.load(
        _minimal_dpo_config(tmp_path), ["dpo.precompute_ref_log_probs=true"]
    )
    assert config.dpo.precompute_ref_log_probs is True
