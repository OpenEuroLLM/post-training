"""Opt-in DPO reference-model log-probability precomputation tests."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml

import post_training.methods.dpo as dpo_module
from post_training.config import DPOMethodConfig, PostTrainingConfig


def test_precompute_defaults_do_not_change_existing_dpo_behavior():
    config = DPOMethodConfig()
    assert config.precompute_ref_log_probs is False
    assert config.precompute_ref_batch_size is None


def test_precompute_options_survive_yaml_round_trip(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "method": "dpo",
                "backend": "trl",
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                },
                "deepspeed": None,
                "dpo": {
                    "precompute_ref_log_probs": True,
                    "precompute_ref_batch_size": 1,
                },
                "data": {
                    "datasets": [
                        {"name": "dummy", "path": "dummy/path", "weight": 1.0}
                    ]
                },
            }
        )
    )

    config = PostTrainingConfig.load(source)
    assert config.dpo.precompute_ref_log_probs is True
    assert config.dpo.precompute_ref_batch_size == 1

    frozen = tmp_path / "frozen.yaml"
    config.save(frozen)
    restored = PostTrainingConfig.load(frozen)
    assert restored.dpo.precompute_ref_log_probs is True
    assert restored.dpo.precompute_ref_batch_size == 1


@pytest.mark.parametrize(
    ("precompute", "batch_size"),
    [(False, None), (True, 1)],
)
def test_precompute_options_reach_trl_dpo_config(
    tmp_path, monkeypatch, precompute, batch_size
):
    config = PostTrainingConfig()
    config.method = "dpo"
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
    monkeypatch.setattr(dpo_module, "DPOConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        dpo_module, "DPOTrainer", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(dpo_module, "build_callbacks", lambda *_a: [])
    for name in (
        "sanitize_generation_config",
        "align_generation_eos",
        "prioritize_metric_callbacks",
        "bind_inference_checkpoint_callback",
    ):
        monkeypatch.setattr(dpo_module, name, lambda *_a: None)

    trainer = dpo_module.build_dpo_trainer(config, tmp_path)
    assert trainer.args["precompute_ref_log_probs"] is precompute
    assert trainer.args["precompute_ref_batch_size"] == batch_size
