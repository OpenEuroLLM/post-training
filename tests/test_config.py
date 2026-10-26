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


# ---------------------------------------------------------------------------
# backend: llamafactory takes all training settings from the llamafactory block
# ---------------------------------------------------------------------------

_TRL_SECTION_NAMES = (
    "model",
    "training",
    "checkpointing",
    "data",
    "sft",
    "dpo",
    "deepspeed",
    "accelerate",
    "logging",
)


def _write_llamafactory_yaml(tmp_path, **extra):
    config = {
        "backend": "llamafactory",
        "container": {
            "image": "/shared/containers/llamafactory.sif",
            "bind_mounts": ["/scratch:/scratch"],
            "env_file": "/shared/env/cluster.env",
        },
        "llamafactory": {"model_name_or_path": "org/model", "stage": "sft", "do_train": True},
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_llamafactory_config_rejects_trl_sections(tmp_path):
    path = _write_llamafactory_yaml(
        tmp_path, training={"learning_rate": 1e-5, "seed": 1}, sft={"max_seq_length": 4096}
    )

    with pytest.raises(ValueError, match="training.learning_rate") as excinfo:
        PostTrainingConfig.load(path)
    assert "training.seed" in str(excinfo.value)
    assert "sft.max_seq_length" in str(excinfo.value)


def test_llamafactory_config_rejects_trl_cli_override(tmp_path):
    path = _write_llamafactory_yaml(tmp_path)

    with pytest.raises(ValueError, match="sft.packing"):
        PostTrainingConfig.load(path, ["sft.packing=false"])


def test_llamafactory_config_rejects_null_trl_section(tmp_path):
    """A key written with a null value counts as set."""
    path = _write_llamafactory_yaml(tmp_path, deepspeed=None)

    with pytest.raises(ValueError, match="deepspeed"):
        PostTrainingConfig.load(path)


def test_llamafactory_backend_from_cli_override_is_checked(tmp_path):
    path = _write_llamafactory_yaml(tmp_path, backend="trl", training={"max_steps": 1})

    with pytest.raises(ValueError, match="training.max_steps"):
        PostTrainingConfig.load(path, ["backend=llamafactory"])


def test_llamafactory_config_allows_top_level_method(tmp_path):
    path = _write_llamafactory_yaml(tmp_path, method="sft")

    assert PostTrainingConfig.load(path).method == "sft"


def test_trl_config_with_all_sections_loads(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "backend": "trl",
                "container": None,
                "model": {"name_or_path": "org/model"},
                "training": {"max_steps": 1},
                "checkpointing": {"save_steps": 10},
                "data": {"datasets": [{"name": "dummy", "path": "dummy/path"}]},
                "sft": {"max_seq_length": 128},
                "dpo": {"beta": 0.1},
                "deepspeed": None,
                "accelerate": {"dynamo_backend": "no"},
                "logging": {"logging_steps": 5},
            }
        )
    )

    PostTrainingConfig.load(path)


def test_llamafactory_save_omits_trl_sections(tmp_path):
    """The frozen config holds only the llamafactory block and the submission keys, and loads again."""
    config = PostTrainingConfig.load(_write_llamafactory_yaml(tmp_path))
    frozen_path = tmp_path / "frozen.yaml"

    config.save(frozen_path)

    frozen = yaml.safe_load(frozen_path.read_text())
    assert not set(frozen) & set(_TRL_SECTION_NAMES)
    assert {"backend", "llamafactory", "container", "slurm", "paths"} <= set(frozen)
    PostTrainingConfig.load(frozen_path)


def test_trl_save_keeps_all_sections(tmp_path):
    config = PostTrainingConfig()
    config.container = None
    config.training.max_steps = 1
    frozen_path = tmp_path / "frozen.yaml"

    config.save(frozen_path)

    frozen = yaml.safe_load(frozen_path.read_text())
    assert set(_TRL_SECTION_NAMES) <= set(frozen)
