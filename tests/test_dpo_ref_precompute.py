"""Opt-in DPO reference-model log-probability precomputation tests.

Covers the ``dpo.precompute_ref_log_probs`` / ``dpo.precompute_ref_batch_size``
options (adopted from PR #69 into PR #70): both default to off/None so existing
DPO runs are unchanged, both survive a YAML round-trip, and both are forwarded
to TRL's ``DPOConfig`` by ``build_dpo_trainer``.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml

import post_training.methods.dpo as dpo_module
from post_training.config import DPOMethodConfig, PostTrainingConfig


def _minimal_dpo_config(tmp_path, dpo: dict | None = None) -> str:
    config = {
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
        "dpo": dpo or {},
        "data": {
            "datasets": [{"name": "dummy", "path": "dummy/path", "weight": 1.0}],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


def test_precompute_defaults_do_not_change_existing_dpo_behavior(tmp_path):
    config = DPOMethodConfig()
    assert config.precompute_ref_log_probs is False
    assert config.precompute_ref_batch_size is None

    loaded = PostTrainingConfig.load(_minimal_dpo_config(tmp_path))
    assert loaded.dpo.precompute_ref_log_probs is False
    assert loaded.dpo.precompute_ref_batch_size is None


def test_precompute_options_survive_yaml_round_trip(tmp_path):
    source = _minimal_dpo_config(
        tmp_path,
        dpo={"precompute_ref_log_probs": True, "precompute_ref_batch_size": 1},
    )

    config = PostTrainingConfig.load(source)
    assert config.dpo.precompute_ref_log_probs is True
    assert config.dpo.precompute_ref_batch_size == 1

    frozen = tmp_path / "frozen.yaml"
    config.save(frozen)
    restored = PostTrainingConfig.load(frozen)
    assert restored.dpo.precompute_ref_log_probs is True
    assert restored.dpo.precompute_ref_batch_size == 1


def test_precompute_options_cli_overrides(tmp_path):
    config = PostTrainingConfig.load(
        _minimal_dpo_config(tmp_path),
        ["dpo.precompute_ref_log_probs=true", "dpo.precompute_ref_batch_size=1"],
    )
    assert config.dpo.precompute_ref_log_probs is True
    assert config.dpo.precompute_ref_batch_size == 1


@pytest.mark.parametrize(
    ("precompute", "batch_size"),
    [(False, None), (True, 1)],
)
def test_precompute_options_reach_trl_dpo_config(tmp_path, monkeypatch, precompute, batch_size):
    config = PostTrainingConfig.load(_minimal_dpo_config(tmp_path))
    config.dpo.precompute_ref_log_probs = precompute
    config.dpo.precompute_ref_batch_size = batch_size

    # Exercise the real builder but do not load GPU model weights.
    monkeypatch.setattr(
        dpo_module,
        "PartialState",
        lambda: SimpleNamespace(main_process_first=nullcontext),
    )
    monkeypatch.setattr(dpo_module, "build_tokenizer", lambda _: object())
    monkeypatch.setattr(dpo_module, "load_and_mix_datasets", lambda *_a, **_kw: object())
    monkeypatch.setattr(dpo_module, "build_common_training_kwargs", lambda *_a: {})
    monkeypatch.setattr(dpo_module, "build_model_init_kwargs", lambda *_a: {})
    monkeypatch.setattr(
        dpo_module,
        "DPOConfig",
        lambda **kwargs: SimpleNamespace(truncation_mode="keep_end", **kwargs),
    )
    monkeypatch.setattr(dpo_module, "DPOTrainer", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dpo_module, "build_callbacks", lambda *_a: [])
    monkeypatch.setattr(
        dpo_module, "install_stage_agnostic_hash_module", lambda *_a, **_kw: "fake-cache-key"
    )
    for name in (
        "sanitize_generation_config",
        "align_generation_eos",
        "prioritize_metric_callbacks",
        "bind_inference_checkpoint_callback",
    ):
        monkeypatch.setattr(dpo_module, name, lambda *_a: None)

    trainer = dpo_module.build_dpo_trainer(config, tmp_path)
    assert trainer.args.precompute_ref_log_probs is precompute
    assert trainer.args.precompute_ref_batch_size == batch_size
