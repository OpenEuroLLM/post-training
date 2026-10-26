"""Tests for nullable config fields."""

from pathlib import Path

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


@pytest.mark.parametrize(
    ("budget_field", "budget_value"),
    [
        ("num_training_samples", 33),
        ("num_training_tokens", 2_097_152),
    ],
)
def test_derived_step_budget_round_trips_without_mutation(
    tmp_path, monkeypatch, budget_field, budget_value
):
    """A frozen sample/token budget remains valid when train.py reloads it."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    budget_field: budget_value,
                    "effective_batch_size": 32,
                    "per_device_train_batch_size": 1,
                },
                "sft": {"max_seq_length": 32_768, "packing": True},
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
    assert config.training.max_steps is None
    assert config.resolve_max_steps() == 2
    assert build_common_training_kwargs(config, tmp_path)["max_steps"] == 2

    frozen_path = tmp_path / "frozen.yaml"
    config.save(frozen_path)
    reloaded = PostTrainingConfig.load(frozen_path)

    assert reloaded.training.max_steps is None
    assert reloaded.resolve_max_steps() == 2


def test_explicit_and_derived_step_budgets_conflict(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 2,
                    "num_training_tokens": 2_097_152,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="Training length is over-specified"):
        PostTrainingConfig.load(config_path)


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


def test_reasoning_lumi_profile_resolves_exact_production_budget():
    config_path = Path(__file__).parents[1] / "configs" / "trl" / "reasoning-sft-lumi.yaml"

    config = PostTrainingConfig.load(config_path)

    assert config.resolve_max_steps() == 500
    assert config.model.revision == "85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31"
    assert config.data.chat_template == "qwen3"
    assert config.sft.truncated_span_action == "drop"
    assert config.accelerate.config_file == "configs/accelerate/fsdp-lumi.yaml"
    assert config.deepspeed is None
