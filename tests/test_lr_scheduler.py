"""Tests for LR scheduler validation and kwargs resolution.

The build tests go through TRL's ``SFTConfig`` and ``Trainer.create_scheduler``,
the path that both TRL 1.7.0 and 1.17.1 use to create the scheduler.
"""

from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import Trainer
from trl import SFTConfig

from post_training.config import PostTrainingConfig
from post_training.methods.common import build_common_training_kwargs
from post_training.utils.lr_scheduler import LR_SCHEDULER_KWARGS, describe_lr_scheduler

PEAK_LR = 1.0e-3


def _load(tmp_path, training: dict, overrides: list[str] | None = None) -> PostTrainingConfig:
    training = {
        "effective_batch_size": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": PEAK_LR,
        **training,
    }
    if not any(
        k in training
        for k in ("max_steps", "num_train_epochs", "num_training_samples", "num_training_tokens")
    ):
        training["max_steps"] = 100
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": training,
                "deepspeed": None,
                "data": {"datasets": [{"name": "dummy", "path": "dummy/path", "weight": 1.0}]},
            }
        )
    )
    return PostTrainingConfig.load(config_path, overrides)


def _build_scheduler(tmp_path, config: PostTrainingConfig, num_training_steps: int):
    """Create the scheduler the way the Trainer does, from the real TrainingArguments."""
    kwargs = build_common_training_kwargs(config, tmp_path)
    args = SFTConfig(
        output_dir=str(tmp_path / "out"),
        max_steps=kwargs["max_steps"],
        learning_rate=kwargs["learning_rate"],
        warmup_steps=kwargs["warmup_steps"],
        lr_scheduler_type=kwargs["lr_scheduler_type"],
        lr_scheduler_kwargs=kwargs["lr_scheduler_kwargs"],
        bf16=False,
        use_cpu=True,
        report_to="none",
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=args.learning_rate)
    fake_trainer = SimpleNamespace(args=args, lr_scheduler=None, optimizer=optimizer)
    return Trainer.create_scheduler(fake_trainer, num_training_steps, optimizer)


def _lrs(scheduler, steps: int) -> list[float]:
    """Return the LR at steps 0..steps (inclusive)."""
    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(steps):
        scheduler.optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


@pytest.mark.parametrize(
    ("scheduler_type", "scheduler_kwargs"),
    [
        ("linear", None),
        ("linear", {}),
        ("cosine", {"num_cycles": 0.5}),
        ("cosine_with_restarts", {"num_cycles": 2}),
        ("polynomial", {"lr_end": 1.0e-5, "power": 2.0}),
        ("constant_with_warmup", None),
        ("inverse_sqrt", None),
        ("cosine_with_min_lr", {"min_lr_rate": 0.1}),
        ("cosine_with_min_lr", {"min_lr": 1.0e-5}),
        ("cosine_warmup_with_min_lr", {"min_lr_rate": 0.1, "warmup_lr_rate": 0.01}),
        ("warmup_stable_decay", {"num_decay_steps": 0.2}),
        (
            "warmup_stable_decay",
            {
                "num_decay_steps": 20,
                "warmup_type": "cosine",
                "decay_type": "1-sqrt",
                "min_lr_ratio": 0.1,
            },
        ),
    ],
)
def test_supported_scheduler_builds(tmp_path, monkeypatch, scheduler_type, scheduler_kwargs):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(
        tmp_path,
        {
            "warmup_steps": 0.1,
            "lr_scheduler_type": scheduler_type,
            "lr_scheduler_kwargs": scheduler_kwargs,
        },
    )
    lrs = _lrs(_build_scheduler(tmp_path, config, 100), 100)
    assert max(lrs) == pytest.approx(PEAK_LR)


def test_constant_without_warmup_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(tmp_path, {"warmup_steps": 0, "lr_scheduler_type": "constant"})
    assert set(_lrs(_build_scheduler(tmp_path, config, 100), 100)) == {PEAK_LR}


def test_every_supported_scheduler_is_tested():
    tested = {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "cosine_with_min_lr",
        "cosine_warmup_with_min_lr",
        "warmup_stable_decay",
    }
    assert set(LR_SCHEDULER_KWARGS) == tested


def test_wsd_lr_curve(tmp_path, monkeypatch):
    """10 warmup steps, 70 stable steps, 20 linear decay steps to 10% of the peak."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(
        tmp_path,
        {
            "max_steps": 100,
            "warmup_steps": 0.1,
            "lr_scheduler_type": "warmup_stable_decay",
            "lr_scheduler_kwargs": {
                "num_decay_steps": 0.2,
                "decay_type": "linear",
                "min_lr_ratio": 0.1,
            },
        },
    )
    lrs = _lrs(_build_scheduler(tmp_path, config, 100), 100)

    assert lrs[5] < PEAK_LR
    assert all(lr == pytest.approx(PEAK_LR) for lr in lrs[10:81])
    assert lrs[90] == pytest.approx(PEAK_LR * 0.55)
    assert lrs[100] == pytest.approx(PEAK_LR * 0.1)


@pytest.mark.parametrize(
    ("training", "expected_decay_steps"),
    [
        ({"max_steps": 100}, 20),
        ({"num_training_samples": 1000, "effective_batch_size": 8}, 25),  # 125 steps
    ],
)
def test_wsd_ratio_resolves_to_steps(tmp_path, monkeypatch, training, expected_decay_steps):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(
        tmp_path,
        {
            **training,
            "lr_scheduler_type": "warmup_stable_decay",
            "lr_scheduler_kwargs": {"num_decay_steps": 0.2},
        },
    )
    kwargs = build_common_training_kwargs(config, tmp_path)["lr_scheduler_kwargs"]
    assert kwargs == {"num_decay_steps": expected_decay_steps}
    assert isinstance(kwargs["num_decay_steps"], int)
    assert f"({expected_decay_steps} steps)" in describe_lr_scheduler(config)


def test_wsd_from_cli_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(
        tmp_path,
        {},
        [
            "training.lr_scheduler_type=warmup_stable_decay",
            "training.lr_scheduler_kwargs.num_decay_steps=0.1",
            "training.lr_scheduler_kwargs.decay_type=1-sqrt",
        ],
    )
    kwargs = build_common_training_kwargs(config, tmp_path)["lr_scheduler_kwargs"]
    assert kwargs == {"num_decay_steps": 10, "decay_type": "1-sqrt"}


def test_wsd_epochs_with_absolute_decay(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(
        tmp_path,
        {
            "num_train_epochs": 1,
            "lr_scheduler_type": "warmup_stable_decay",
            "lr_scheduler_kwargs": {"num_decay_steps": 50},
        },
    )
    kwargs = build_common_training_kwargs(config, tmp_path)["lr_scheduler_kwargs"]
    assert kwargs == {"num_decay_steps": 50}


def test_cosine_with_restarts_num_cycles_is_int(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = _load(
        tmp_path,
        {"lr_scheduler_type": "cosine_with_restarts", "lr_scheduler_kwargs": {"num_cycles": 3}},
    )
    kwargs = build_common_training_kwargs(config, tmp_path)["lr_scheduler_kwargs"]
    assert kwargs == {"num_cycles": 3}
    assert isinstance(kwargs["num_cycles"], int)


@pytest.mark.parametrize(
    ("training", "error"),
    [
        ({"lr_scheduler_type": "reduce_lr_on_plateau"}, "Unsupported training.lr_scheduler_type"),
        ({"lr_scheduler_type": "greedy"}, "Unsupported training.lr_scheduler_type"),
        # The old pitfall: a min_lr_rate default left in place for `linear`.
        (
            {"lr_scheduler_type": "linear", "lr_scheduler_kwargs": {"min_lr_rate": 0.1}},
            "min_lr_rate, which lr_scheduler_type='linear' does not accept",
        ),
        (
            {"lr_scheduler_type": "cosine", "lr_scheduler_kwargs": {"num_decay_steps": 0.1}},
            "does not accept",
        ),
        ({"lr_scheduler_type": "cosine_with_min_lr"}, "exactly one of"),
        (
            {
                "lr_scheduler_type": "cosine_with_min_lr",
                "lr_scheduler_kwargs": {"min_lr": 1.0e-5, "min_lr_rate": 0.1},
            },
            "exactly one of",
        ),
        (
            {"lr_scheduler_type": "cosine_with_min_lr", "lr_scheduler_kwargs": {"min_lr_rate": 2}},
            "between 0 and 1",
        ),
        (
            {"lr_scheduler_type": "cosine_with_min_lr", "lr_scheduler_kwargs": {"min_lr": 1.0}},
            "between 0 and training.learning_rate",
        ),
        ({"lr_scheduler_type": "constant", "warmup_steps": 0.1}, "constant_with_warmup"),
        (
            {
                "lr_scheduler_type": "cosine_with_restarts",
                "lr_scheduler_kwargs": {"num_cycles": 1.5},
            },
            "must be an integer",
        ),
        (
            {"lr_scheduler_type": "polynomial", "lr_scheduler_kwargs": {"lr_end": 1.0}},
            "smaller than training.learning_rate",
        ),
        ({"lr_scheduler_type": "warmup_stable_decay"}, "requires training.lr_scheduler_kwargs"),
        (
            {
                "lr_scheduler_type": "warmup_stable_decay",
                "lr_scheduler_kwargs": {"num_decay_steps": 0.2, "decay_type": "exp"},
            },
            "decay_type must be one of",
        ),
        (
            {
                "lr_scheduler_type": "warmup_stable_decay",
                "lr_scheduler_kwargs": {"num_decay_steps": 10.5},
            },
            "must be an integer",
        ),
        (
            {
                "max_steps": 100,
                "warmup_steps": 0.5,
                "lr_scheduler_type": "warmup_stable_decay",
                "lr_scheduler_kwargs": {"num_decay_steps": 0.6},
            },
            "is longer than the training run",
        ),
        (
            {
                "num_train_epochs": 1,
                "lr_scheduler_type": "warmup_stable_decay",
                "lr_scheduler_kwargs": {"num_decay_steps": 0.2},
            },
            "set num_decay_steps to an absolute number",
        ),
    ],
)
def test_invalid_scheduler_config_is_rejected(tmp_path, training, error):
    with pytest.raises(ValueError, match=error):
        _load(tmp_path, training)
