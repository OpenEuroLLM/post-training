"""Tests for nullable config fields."""

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
                "deepspeed": {"config_path": None},
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
