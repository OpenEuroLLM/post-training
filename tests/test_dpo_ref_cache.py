"""Tests for the stage-agnostic DPO reference log-prob cache key.

TRL names the precomputed reference-logprob cache (``*.npz`` next to the dataset
cache) ``Hasher.hash((dataset._fingerprint, hash_module(model)))``, and
``hash_module`` hashes the in-memory ``state_dict()``.  Under DeepSpeed ZeRO-3
the parameters are already partitioned when ``DPOTrainer.__init__`` computes the
hash, so the same checkpoint produced a different cache file per ZeRO stage and
per world size, and the cache was silently recomputed (OpenEuroLLM/post-training
#62).  ``post_training.methods.dpo`` replaces the hash with a cheap key that
identifies the checkpoint on disk instead: the resolved commit SHA for Hub
models and cache snapshot dirs, and file name/size/safetensors-header (plus
``config.json``) for plain local dirs.
"""

import json
import os
import types

import pytest
import torch
import yaml
from safetensors.torch import save_file
from trl.trainer.utils import hash_module as trl_hash_module

from post_training.config import PostTrainingConfig
from post_training.methods.dpo import (
    _ref_source_and_revision,
    checkpoint_cache_key,
    install_stage_agnostic_hash_module,
)


def _write_local(root, weight_bytes: bytes = b"weights-v1", config: dict | None = None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or {"model_type": "llama"}))
    (root / "model.bin").write_bytes(weight_bytes)
    return root


def _write_snapshot(root, sha: str, weight_bytes: bytes = b"weights-v1"):
    snap = root / "models--org--model" / "snapshots" / sha
    return _write_local(snap, weight_bytes)


def _pin_mtime(root, ns: int = 1_700_000_000_000_000_000) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.utime(p, ns=(ns, ns))


def test_local_key_is_stable_across_copies(tmp_path):
    """Identical content with identical mtimes shares one key (a plain copy sets new
    mtimes and therefore misses the cache — accepted tradeoff, see mtime test)."""
    a = _write_local(tmp_path / "a")
    b = _write_local(tmp_path / "b")
    _pin_mtime(a)
    _pin_mtime(b)
    assert checkpoint_cache_key(str(a), None, "bfloat16") == checkpoint_cache_key(
        str(b), None, "bfloat16"
    )


def test_local_key_changes_with_mtime(tmp_path):
    """A checkpoint retrained in place at the same path must not reuse stale log-probs.

    Same file name, same size, same safetensors header — only the mtime moves, exactly
    what happens when `/checkpoints/sft-final` is overwritten by a new training run.
    """
    ckpt = _write_local(tmp_path / "ckpt", weight_bytes=b"weights-v1")
    key_before = checkpoint_cache_key(str(ckpt), None, "bfloat16")

    weight_file = ckpt / "model.bin"
    weight_file.write_bytes(b"WEIGHTS-v1")  # same length, different content
    st = weight_file.stat()
    os.utime(weight_file, ns=(st.st_mtime_ns + 10**9, st.st_mtime_ns + 10**9))

    assert checkpoint_cache_key(str(ckpt), None, "bfloat16") != key_before


def test_local_key_changes_with_file_size(tmp_path):
    a = _write_local(tmp_path / "a", weight_bytes=b"weights-v1")
    b = _write_local(tmp_path / "b", weight_bytes=b"weights-v1-plus")
    assert checkpoint_cache_key(str(a), None, "bfloat16") != checkpoint_cache_key(
        str(b), None, "bfloat16"
    )


def test_local_key_ignores_weight_values_but_not_the_safetensors_header(tmp_path):
    """Plain local dirs are keyed by name/size/header, not by reading weight contents.

    Same shape, different values -> same key (header and size are unchanged).
    Different shape -> the safetensors header changes -> different key.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    save_file({"w": torch.zeros(4, 4)}, str(a / "model.safetensors"))
    save_file({"w": torch.ones(4, 4)}, str(b / "model.safetensors"))
    save_file({"w": torch.zeros(4, 5)}, str(c / "model.safetensors"))
    for d in (a, b, c):
        _pin_mtime(d)

    key_a = checkpoint_cache_key(str(a), None, "bfloat16")
    assert checkpoint_cache_key(str(b), None, "bfloat16") == key_a
    assert checkpoint_cache_key(str(c), None, "bfloat16") != key_a


def test_local_key_changes_with_config(tmp_path):
    a = _write_local(tmp_path / "a", config={"model_type": "llama"})
    b = _write_local(tmp_path / "b", config={"model_type": "qwen2"})
    assert checkpoint_cache_key(str(a), None, "bfloat16") != checkpoint_cache_key(
        str(b), None, "bfloat16"
    )


def test_key_changes_with_dtype_and_extra(tmp_path):
    a = _write_local(tmp_path / "a")
    base = checkpoint_cache_key(str(a), None, "bfloat16")
    assert checkpoint_cache_key(str(a), None, "float32") != base
    assert checkpoint_cache_key(
        str(a), None, "bfloat16", extra={"max_length": 1024, "truncation_mode": "keep_start"}
    ) != checkpoint_cache_key(
        str(a), None, "bfloat16", extra={"max_length": 2048, "truncation_mode": "keep_start"}
    )


def test_local_key_requires_config_or_weight_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        checkpoint_cache_key(str(empty), None, "bfloat16")


def test_snapshot_dir_key_uses_commit_sha(tmp_path):
    """A cache snapshot dir is keyed by its commit SHA; weight contents are never read."""
    sha = "a" * 40
    snap = _write_snapshot(tmp_path, sha, weight_bytes=b"weights-v1")
    key = checkpoint_cache_key(str(snap), None, "bfloat16")

    _write_snapshot(tmp_path, sha, weight_bytes=b"weights-v2-changed-values")
    assert checkpoint_cache_key(str(snap), None, "bfloat16") == key

    other = _write_snapshot(tmp_path, "b" * 40)
    assert checkpoint_cache_key(str(other), None, "bfloat16") != key


def test_hub_key_uses_resolved_commit_sha_and_light_allow_patterns(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = []
    sha = "c" * 40

    def fake_snapshot_download(
        repo_id, repo_type=None, revision=None, allow_patterns=None, local_files_only=False
    ):
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": allow_patterns,
                "local_files_only": local_files_only,
            }
        )
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path / "models--org--model" / "snapshots" / sha)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    key = checkpoint_cache_key("org/model", "main", "bfloat16")

    assert [c["local_files_only"] for c in calls] == [True, False]
    assert all(c["allow_patterns"] == ["config.json"] for c in calls)
    assert calls[0]["revision"] == "main"
    assert key == checkpoint_cache_key("org/model", "main", "bfloat16")
    assert key != checkpoint_cache_key("org/model", "main", "float32")

    # The key is the resolved commit SHA: another repo id resolving to the same
    # commit shares the cache, a different commit does not.
    assert key == checkpoint_cache_key("org/other", "main", "bfloat16")

    def fake_other_sha(
        repo_id, repo_type=None, revision=None, allow_patterns=None, local_files_only=False
    ):
        return str(tmp_path / "snapshots" / ("d" * 40))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_other_sha)
    assert checkpoint_cache_key("org/model", "main", "bfloat16") != key

    # A local snapshot dir with the same SHA gets the same key as the Hub id.
    snap = _write_snapshot(tmp_path, sha)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: str(tmp_path / "models--org--model" / "snapshots" / sha),
    )
    assert checkpoint_cache_key("org/model", None, "bfloat16") == checkpoint_cache_key(
        str(snap), None, "bfloat16"
    )


def test_installed_hash_ignores_partitioned_state_dict(tmp_path):
    """Two partition views of one checkpoint must share one cache key.

    The first assertion pins the upstream behavior that caused the bug: TRL's
    ``hash_module`` hashes whatever ``state_dict()`` returns, so two different
    partitionings of the same weights hash differently.  The installed
    replacement must return the same disk-based key for both.
    """
    ckpt = _write_local(tmp_path / "ckpt")
    view_a = torch.nn.Linear(4, 4)
    view_b = torch.nn.Linear(4, 4)
    with torch.no_grad():
        view_b.weight.copy_(view_a.weight + 1.0)  # stand-in for a different shard
    assert trl_hash_module(view_a) != trl_hash_module(view_b)

    fake_module = types.SimpleNamespace(hash_module=trl_hash_module)
    expected = checkpoint_cache_key(str(ckpt), None, "bfloat16", extra={"max_length": 2048})
    installed = install_stage_agnostic_hash_module(
        fake_module, str(ckpt), None, "bfloat16", extra={"max_length": 2048}
    )

    assert installed == expected
    assert fake_module.hash_module(view_a) == expected
    assert fake_module.hash_module(view_b) == expected


def test_install_patches_trl_trainer_global_lookup(tmp_path, monkeypatch):
    """DPOTrainer resolves ``hash_module`` as a module global at call time."""
    import trl.trainer.dpo_trainer as dpo_trainer_module
    from trl.trainer.dpo_trainer import DPOTrainer

    monkeypatch.setattr(dpo_trainer_module, "hash_module", dpo_trainer_module.hash_module)
    ckpt = _write_local(tmp_path / "ckpt")
    expected = checkpoint_cache_key(str(ckpt), None, "bfloat16")

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


def test_precompute_ref_log_probs_and_ref_revision_cli_overrides(tmp_path):
    config = PostTrainingConfig.load(
        _minimal_dpo_config(tmp_path),
        ["dpo.precompute_ref_log_probs=true", "dpo.ref_model_revision=abc123"],
    )
    assert config.dpo.precompute_ref_log_probs is True
    assert config.dpo.ref_model_revision == "abc123"


def test_ref_source_and_revision_uses_model_pair_by_default(tmp_path):
    config = PostTrainingConfig.load(_minimal_dpo_config(tmp_path))
    config.model.name_or_path = "org/policy"
    config.model.revision = "policy-sha"
    assert _ref_source_and_revision(config) == ("org/policy", "policy-sha")


def test_ref_source_and_revision_prefers_explicit_ref_with_own_revision(tmp_path):
    config = PostTrainingConfig.load(_minimal_dpo_config(tmp_path))
    config.model.name_or_path = "org/policy"
    config.model.revision = "policy-sha"
    config.dpo.ref_model_name_or_path = "org/ref"
    config.dpo.ref_model_revision = "ref-sha"
    assert _ref_source_and_revision(config) == ("org/ref", "ref-sha")
    config.dpo.ref_model_revision = None
    assert _ref_source_and_revision(config) == ("org/ref", None)
