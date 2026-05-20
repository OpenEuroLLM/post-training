"""Tests for LRSchedulerKwargs validation and guardrail rendering.

Covers the per-scheduler allow-list in TRLBackend._validate_lr_scheduler_kwargs
and the scheduler-aware summary used by the submission guardrails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from post_training.config import PostTrainingConfig
from post_training.utils.guardrails import _lr_scheduler_summary

_FIXTURE_YAML = Path(__file__).resolve().parent / "fixtures" / "sft_lr_scheduler.yaml"


def _load(*overrides: str) -> PostTrainingConfig:
    return PostTrainingConfig.load(_FIXTURE_YAML, cli_overrides=list(overrides))


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------


def test_wsd_kwarg_rejected_under_cosine():
    with pytest.raises(ValueError, match="num_stable_steps.*cosine_with_min_lr"):
        _load("training.lr_scheduler_kwargs.num_stable_steps=500")


def test_cosine_kwarg_rejected_under_wsd():
    with pytest.raises(ValueError, match="min_lr_rate.*warmup_stable_decay"):
        _load(
            "training.lr_scheduler_type=warmup_stable_decay",
            "training.lr_scheduler_kwargs.num_decay_steps=200",
            # min_lr_rate is left at the yaml default (0.1) -> belongs to cosine, not WSD.
        )


def test_no_kwargs_scheduler_rejects_any_kwarg():
    with pytest.raises(ValueError, match="linear.*accepts no extra kwargs"):
        _load("training.lr_scheduler_type=linear")  # min_lr_rate still set from yaml


# ---------------------------------------------------------------------------
# WSD-specific checks
# ---------------------------------------------------------------------------


def test_wsd_requires_num_decay_steps():
    with pytest.raises(ValueError, match="num_decay_steps is required"):
        _load(
            "training.lr_scheduler_type=warmup_stable_decay",
            "training.lr_scheduler_kwargs.min_lr_rate=null",
        )


@pytest.mark.parametrize("field", ["warmup_type", "decay_type"])
def test_wsd_rejects_invalid_warmup_or_decay_type(field):
    with pytest.raises(ValueError, match=f"{field}='quadratic'"):
        _load(
            "training.lr_scheduler_type=warmup_stable_decay",
            "training.lr_scheduler_kwargs.min_lr_rate=null",
            "training.lr_scheduler_kwargs.num_decay_steps=200",
            f"training.lr_scheduler_kwargs.{field}=quadratic",
        )


@pytest.mark.parametrize("value", ["linear", "cosine", "1-sqrt"])
def test_wsd_accepts_valid_warmup_and_decay_types(value):
    c = _load(
        "training.lr_scheduler_type=warmup_stable_decay",
        "training.lr_scheduler_kwargs.min_lr_rate=null",
        "training.lr_scheduler_kwargs.num_decay_steps=200",
        f"training.lr_scheduler_kwargs.warmup_type={value}",
        f"training.lr_scheduler_kwargs.decay_type={value}",
    )
    assert c.training.lr_scheduler_kwargs.warmup_type == value
    assert c.training.lr_scheduler_kwargs.decay_type == value


# ---------------------------------------------------------------------------
# Guardrail rendering
# ---------------------------------------------------------------------------


def test_lr_scheduler_summary_cosine():
    c = _load()
    summary = _lr_scheduler_summary(c)
    assert summary.startswith("cosine_with_min_lr")
    assert "min_lr_rate=" in summary
    # Unset WSD fields must not leak into the string.
    assert "num_stable_steps" not in summary
    assert "num_decay_steps" not in summary


def test_lr_scheduler_summary_wsd_lists_set_fields_only():
    c = _load(
        "training.lr_scheduler_type=warmup_stable_decay",
        "training.lr_scheduler_kwargs.min_lr_rate=null",
        "training.lr_scheduler_kwargs.num_stable_steps=500",
        "training.lr_scheduler_kwargs.num_decay_steps=200",
    )
    summary = _lr_scheduler_summary(c)
    assert summary.startswith("warmup_stable_decay")
    assert "num_stable_steps=500" in summary
    assert "num_decay_steps=200" in summary
    # Fields left at None must not appear.
    assert "min_lr_rate" not in summary
    assert "min_lr_ratio" not in summary
    assert "num_cycles" not in summary
    assert "warmup_type" not in summary
    assert "decay_type" not in summary


def test_lr_scheduler_summary_no_kwargs_set():
    c = _load(
        "training.lr_scheduler_type=linear",
        "training.lr_scheduler_kwargs.min_lr_rate=null",
    )
    # When no extra kwargs are set, the scheduler name appears alone.
    assert _lr_scheduler_summary(c) == "linear"
