"""Tests for nullable config fields."""

import pytest
import yaml

from post_training.config import ModelConfig, PostTrainingConfig
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


@pytest.mark.parametrize(
    ("tokenizer_name_or_path", "tokenizer_revision", "expected"),
    [
        # No tokenizer override: the tokenizer follows the model.
        (None, None, ("org/model", "r1")),
        # A tokenizer revision alone pins the tokenizer inside the model repo.
        (None, "r2", ("org/model", "r2")),
        # A separate repo ignores model.revision, which pins a different repo.
        ("org/tokenizer", None, ("org/tokenizer", None)),
        ("org/tokenizer", "r2", ("org/tokenizer", "r2")),
    ],
)
def test_resolve_tokenizer(tokenizer_name_or_path, tokenizer_revision, expected):
    model = ModelConfig(
        name_or_path="org/model",
        revision="r1",
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_revision=tokenizer_revision,
    )

    assert model.resolve_tokenizer() == expected


def test_resolve_tokenizer_without_any_revision():
    assert ModelConfig(name_or_path="org/model").resolve_tokenizer() == ("org/model", None)
