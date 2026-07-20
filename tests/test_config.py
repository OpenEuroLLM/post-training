"""Tests for nullable config fields."""

import pytest
import yaml

from post_training.config import PostTrainingConfig
from post_training.methods.common import build_common_training_kwargs


def test_nullable_container_and_training_kwargs_load(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
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
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    kwargs = build_common_training_kwargs(config, tmp_path)

    assert config.container is None
    assert kwargs["lr_scheduler_kwargs"] is None
    assert kwargs["gradient_checkpointing_kwargs"] is None
    assert kwargs["deepspeed"] is None


def test_deepspeed_empty_dict_normalized_to_none(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                },
                "deepspeed": {},
                "data": {
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    kwargs = build_common_training_kwargs(config, tmp_path)

    assert kwargs["deepspeed"] is None


def _base_config_dict(**overrides):
    config = {
        "method": "sft",
        "backend": "trl",
        "training": {
            "max_steps": 1,
            "effective_batch_size": 1,
            "per_device_train_batch_size": 1,
        },
        "deepspeed": None,
        "data": {
            "datasets": [
                {
                    "name": "dummy",
                    "path": "dummy/path",
                    "weight": 1.0,
                }
            ]
        },
    }
    config.update(overrides)
    return config


def test_max_failures_capped_without_explicit_run_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_config_dict(slurm={"max_failures": 3})))

    config = PostTrainingConfig.load(config_path)

    assert config.run_name is None
    assert config.slurm.max_failures == 1


def test_max_failures_kept_with_explicit_run_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_base_config_dict(run_name="my-explicit-run", slurm={"max_failures": 3}))
    )

    config = PostTrainingConfig.load(config_path)

    assert config.run_name == "my-explicit-run"
    assert config.slurm.max_failures == 3


def test_deepspeed_old_style_config_path_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                },
                "deepspeed": {"config_path": "configs/deepspeed/zero2.yaml"},
                "data": {
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    with pytest.raises(ValueError, match="deepspeed.config_path is no longer supported"):
        PostTrainingConfig.load(config_path)
