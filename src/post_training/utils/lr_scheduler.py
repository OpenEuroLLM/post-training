"""LR schedulers that the TRL backend supports, with their keyword arguments.

TRL does not build the LR scheduler itself (TRL 1.7.0 and 1.17.1 alike).
``Trainer.create_scheduler`` passes ``lr_scheduler_type`` and
``lr_scheduler_kwargs`` to ``transformers.get_scheduler``.  This is also true
under DeepSpeed (when its config has no ``scheduler`` section) and under FSDP.
The table below therefore follows the ``transformers`` scheduler signatures,
which are the same from 4.56.2 (the TRL minimum) through 5.x.

Some schedulers are left out on purpose:

* ``reduce_lr_on_plateau`` and ``greedy`` step on evaluation metrics, and this
  framework runs no evaluation loop.  ``greedy`` also exists only in recent
  ``transformers`` releases.
* ``inverse_sqrt`` is supported, but without ``timescale``: ``transformers``
  5.2 silently drops that argument.

This module must stay importable without ``transformers``, because the
submission phase validates configs on the login node.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig, TrainingConfig

_COSINE_MIN_LR_KWARGS = frozenset({"num_cycles", "min_lr", "min_lr_rate"})

# lr_scheduler_type -> the lr_scheduler_kwargs fields it accepts.
LR_SCHEDULER_KWARGS: dict[str, frozenset[str]] = {
    "linear": frozenset(),
    "cosine": frozenset({"num_cycles"}),
    "cosine_with_restarts": frozenset({"num_cycles"}),
    "polynomial": frozenset({"lr_end", "power"}),
    "constant": frozenset(),
    "constant_with_warmup": frozenset(),
    "inverse_sqrt": frozenset(),
    "cosine_with_min_lr": _COSINE_MIN_LR_KWARGS,
    "cosine_warmup_with_min_lr": _COSINE_MIN_LR_KWARGS | {"warmup_lr_rate"},
    "warmup_stable_decay": frozenset(
        {"num_decay_steps", "warmup_type", "decay_type", "min_lr_ratio", "num_cycles"}
    ),
}

# Warmup and decay shapes that transformers' get_wsd_schedule accepts.
WSD_SHAPES = ("linear", "cosine", "1-sqrt")


def _set_kwargs(training: TrainingConfig) -> dict[str, Any]:
    """Return the lr_scheduler_kwargs fields that are set (not ``None``)."""
    if training.lr_scheduler_kwargs is None:
        return {}
    return {
        k: v for k, v in dataclasses.asdict(training.lr_scheduler_kwargs).items() if v is not None
    }


def _resolve_steps(value: float, total_steps: int) -> int:
    """Resolve a step count given as a ratio (< 1) or as an absolute value (>= 1).

    Uses the same rule as ``TrainingArguments.get_warmup_steps``.
    """
    return int(value) if value >= 1 else math.ceil(total_steps * value)


def validate_lr_scheduler(config: PostTrainingConfig) -> None:
    """Reject a scheduler config that would fail, or silently misbehave, in training.

    Call this after the training length is validated: the WSD checks use
    ``config.resolve_max_steps()``.
    """
    t = config.training
    name = t.lr_scheduler_type
    if name not in LR_SCHEDULER_KWARGS:
        raise ValueError(
            f"Unsupported training.lr_scheduler_type {name!r}. "
            f"Supported: {', '.join(LR_SCHEDULER_KWARGS)}."
        )

    kwargs = _set_kwargs(t)
    accepted = LR_SCHEDULER_KWARGS[name]
    unaccepted = sorted(set(kwargs) - accepted)
    if unaccepted:
        raise ValueError(
            f"training.lr_scheduler_kwargs sets {', '.join(unaccepted)}, which "
            f"lr_scheduler_type={name!r} does not accept. Accepted: "
            f"{', '.join(sorted(accepted)) or 'none (set the unused fields to null)'}."
        )

    if t.warmup_steps < 0:
        raise ValueError("training.warmup_steps must be non-negative.")
    if name == "constant" and t.warmup_steps > 0:
        raise ValueError(
            "lr_scheduler_type='constant' ignores training.warmup_steps. "
            "Use 'constant_with_warmup', or set training.warmup_steps: 0."
        )

    if "num_cycles" in kwargs:
        num_cycles = kwargs["num_cycles"]
        if num_cycles <= 0:
            raise ValueError("training.lr_scheduler_kwargs.num_cycles must be positive.")
        if name == "cosine_with_restarts" and not float(num_cycles).is_integer():
            raise ValueError(
                "training.lr_scheduler_kwargs.num_cycles must be an integer for "
                "lr_scheduler_type='cosine_with_restarts'."
            )

    for key in ("min_lr_rate", "min_lr_ratio", "warmup_lr_rate"):
        if key in kwargs and not 0 <= kwargs[key] <= 1:
            raise ValueError(f"training.lr_scheduler_kwargs.{key} must be between 0 and 1.")

    if "min_lr" in kwargs and not 0 <= kwargs["min_lr"] <= t.learning_rate:
        raise ValueError(
            "training.lr_scheduler_kwargs.min_lr must be between 0 and training.learning_rate."
        )

    if name in ("cosine_with_min_lr", "cosine_warmup_with_min_lr") and (
        ("min_lr" in kwargs) == ("min_lr_rate" in kwargs)
    ):
        raise ValueError(
            f"lr_scheduler_type={name!r} requires exactly one of "
            "training.lr_scheduler_kwargs.min_lr or training.lr_scheduler_kwargs.min_lr_rate."
        )

    if "lr_end" in kwargs and not 0 <= kwargs["lr_end"] < t.learning_rate:
        raise ValueError(
            "training.lr_scheduler_kwargs.lr_end must be non-negative and smaller than "
            "training.learning_rate."
        )
    if "power" in kwargs and kwargs["power"] <= 0:
        raise ValueError("training.lr_scheduler_kwargs.power must be positive.")

    if name == "warmup_stable_decay":
        _validate_wsd(config, kwargs)


def _validate_wsd(config: PostTrainingConfig, kwargs: dict[str, Any]) -> None:
    t = config.training
    for key in ("warmup_type", "decay_type"):
        if key in kwargs and kwargs[key] not in WSD_SHAPES:
            raise ValueError(
                f"training.lr_scheduler_kwargs.{key} must be one of {', '.join(WSD_SHAPES)}; "
                f"got {kwargs[key]!r}."
            )

    decay = kwargs.get("num_decay_steps")
    if decay is None:
        raise ValueError(
            "lr_scheduler_type='warmup_stable_decay' requires "
            "training.lr_scheduler_kwargs.num_decay_steps (a ratio in (0, 1) of the "
            "training steps, or an absolute number of steps >= 1)."
        )
    if decay <= 0:
        raise ValueError("training.lr_scheduler_kwargs.num_decay_steps must be positive.")
    if decay >= 1 and not float(decay).is_integer():
        raise ValueError(
            "training.lr_scheduler_kwargs.num_decay_steps >= 1 is an absolute number of "
            f"steps and must be an integer; got {decay}."
        )

    total = config.resolve_max_steps()
    if total is None:
        # Epoch-based training: the step count is known only inside the Trainer.
        if decay < 1:
            raise ValueError(
                "A ratio for training.lr_scheduler_kwargs.num_decay_steps needs a step "
                "budget that is known before training. With training.num_train_epochs, "
                "set num_decay_steps to an absolute number of steps."
            )
        return

    warmup_steps = _resolve_steps(t.warmup_steps, total)
    decay_steps = _resolve_steps(decay, total)
    if warmup_steps + decay_steps > total:
        raise ValueError(
            f"warmup_stable_decay: warmup ({warmup_steps} steps) + decay ({decay_steps} "
            f"steps) is longer than the training run ({total} steps)."
        )


def resolve_lr_scheduler_kwargs(
    training: TrainingConfig, max_steps: int | None
) -> dict[str, Any] | None:
    """Return the ``lr_scheduler_kwargs`` to pass to ``TrainingArguments``.

    Unset fields are dropped.  A ratio ``num_decay_steps`` is converted to an
    absolute step count, because ``transformers.get_wsd_schedule`` accepts only
    an integer.  ``max_steps`` is ``None`` for epoch-based training, where
    validation allows only an absolute ``num_decay_steps``.
    """
    if training.lr_scheduler_kwargs is None:
        return None
    kwargs = _set_kwargs(training)

    if training.lr_scheduler_type == "cosine_with_restarts" and "num_cycles" in kwargs:
        kwargs["num_cycles"] = int(kwargs["num_cycles"])

    if "num_decay_steps" in kwargs:
        decay = kwargs["num_decay_steps"]
        if decay < 1 and max_steps is None:
            raise ValueError("A ratio num_decay_steps requires step-based training.")
        kwargs["num_decay_steps"] = _resolve_steps(decay, max_steps or 0)

    return kwargs


def describe_lr_scheduler(config: PostTrainingConfig) -> str:
    """Return a one-line summary of the scheduler, for the submission summary."""
    t = config.training
    kwargs = _set_kwargs(t)
    if not kwargs:
        return t.lr_scheduler_type

    parts = [f"{k}={v}" for k, v in kwargs.items()]
    decay = kwargs.get("num_decay_steps")
    if decay is not None and 0 < decay < 1:
        total = config.resolve_max_steps()
        if total is not None:
            index = list(kwargs).index("num_decay_steps")
            parts[index] += f" ({_resolve_steps(decay, total)} steps)"
    return f"{t.lr_scheduler_type}  ({', '.join(parts)})"
