"""Tests for the `fsdp:` config block and its validation."""

import pytest
import yaml

from post_training.backend import LlamaFactoryBackend, TRLBackend
from post_training.config import (
    ContainerConfig,
    FSDPConfig,
    GradientCheckpointingKwargs,
    PostTrainingConfig,
)


def _fsdp_config() -> PostTrainingConfig:
    """A TRL config that passes validation with FSDP enabled."""
    cfg = PostTrainingConfig()
    cfg.training.max_steps = 1
    cfg.container = ContainerConfig(
        image="/shared/containers/trl.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file="/shared/env/cluster.env",
    )
    cfg.deepspeed = None
    cfg.fsdp = FSDPConfig()
    cfg.training.gradient_checkpointing = True
    cfg.training.gradient_checkpointing_kwargs = GradientCheckpointingKwargs(early_stop=False)
    return cfg


def test_valid_fsdp_config_passes():
    TRLBackend().validate(_fsdp_config())


def test_fsdp_block_loads_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "dpo",
                "backend": "trl",
                "container": {
                    "image": "/shared/containers/trl.sif",
                    "bind_mounts": ["/scratch:/scratch"],
                    "env_file": "/shared/env/cluster.env",
                },
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_kwargs": {"early_stop": False},
                },
                "deepspeed": None,
                "fsdp": {"transformer_layer_cls_to_wrap": ["Olmo3DecoderLayer"]},
                "data": {"datasets": [{"name": "dummy", "path": "dummy/path", "weight": 1.0}]},
            }
        )
    )

    config = PostTrainingConfig.load(config_path)

    assert config.fsdp.transformer_layer_cls_to_wrap == ["Olmo3DecoderLayer"]
    assert config.fsdp.auto_wrap_policy == "TRANSFORMER_BASED_WRAP"
    assert config.fsdp.state_dict_type == "SHARDED_STATE_DICT"


def test_fsdp_defaults_to_disabled():
    assert PostTrainingConfig().fsdp is None


def test_fsdp_and_deepspeed_rejected_together():
    cfg = _fsdp_config()
    cfg.deepspeed = {"zero_optimization": {"stage": 2}}

    with pytest.raises(ValueError, match="either `deepspeed:` or `fsdp:`"):
        TRLBackend().validate(cfg)


def test_fsdp_without_container_rejected():
    cfg = _fsdp_config()
    cfg.container = None

    with pytest.raises(ValueError, match="fsdp requires container.image"):
        TRLBackend().validate(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sharding_strategy", "full_shard"),
        ("auto_wrap_policy", "LAYER_WRAP"),
        ("backward_prefetch", "PRE"),
        ("state_dict_type", "LOCAL_STATE_DICT"),
    ],
)
def test_fsdp_unknown_choice_rejected(field, value):
    cfg = _fsdp_config()
    setattr(cfg.fsdp, field, value)

    with pytest.raises(ValueError, match=f"fsdp.{field} must be one of"):
        TRLBackend().validate(cfg)


def test_fsdp_efficient_loading_requires_sync_module_states():
    cfg = _fsdp_config()
    cfg.fsdp.sync_module_states = False

    with pytest.raises(ValueError, match="requires fsdp.sync_module_states"):
        TRLBackend().validate(cfg)


def test_fsdp_activation_checkpointing_conflicts_with_gradient_checkpointing():
    cfg = _fsdp_config()
    cfg.fsdp.activation_checkpointing = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        TRLBackend().validate(cfg)


def test_fsdp_activation_checkpointing_alone_passes():
    cfg = _fsdp_config()
    cfg.fsdp.activation_checkpointing = True
    cfg.training.gradient_checkpointing = False

    TRLBackend().validate(cfg)


@pytest.mark.parametrize("kwargs", [None, GradientCheckpointingKwargs(early_stop=True)])
def test_fsdp_gradient_checkpointing_requires_early_stop_off(kwargs):
    cfg = _fsdp_config()
    cfg.training.gradient_checkpointing_kwargs = kwargs

    with pytest.raises(ValueError, match="early_stop: false"):
        TRLBackend().validate(cfg)


def test_fsdp_rejected_for_llamafactory():
    cfg = _fsdp_config()
    cfg.backend = "llamafactory"
    cfg.llamafactory = {"stage": "sft"}

    with pytest.raises(ValueError, match="only supported with backend='trl'"):
        LlamaFactoryBackend().validate(cfg)
