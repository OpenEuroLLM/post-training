"""Training backend dispatch."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class Backend(ABC):
    @abstractmethod
    def validate(self, config: PostTrainingConfig) -> None:
        """Backend-specific config validation."""

    @abstractmethod
    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        """Return an auto-generated run name."""

    @abstractmethod
    def run_dir_subdirs(self) -> list[str]:
        """Extra subdirectories to create inside the run directory."""

    @abstractmethod
    def render_slurm_script(
        self,
        config: PostTrainingConfig,
        run_dir: Path,
        frozen_config_path: str,
        *,
        tokenize_only: bool = False,
    ) -> Path:
        """Render the SLURM batch script and return its path."""

    @abstractmethod
    def post_freeze(self, config: PostTrainingConfig, run_dir: Path) -> None:
        """Copy any extra artifacts into the run directory after config freeze."""


# ---------------------------------------------------------------------------
# TRL
# ---------------------------------------------------------------------------

_SUPPORTED_METHODS = ("sft", "dpo")


def _shorten_model_name(name_or_path: str) -> str:
    short = name_or_path.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9\-]", "", short.lower())


def _shorten_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "_", name.lower())


def _dataset_mix_hash(datasets: list) -> str:
    canonical = sorted(
        [{"name": d.name, "path": d.path, "weight": d.weight} for d in datasets],
        key=lambda x: x["name"],
    )
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()[:8]


class TRLBackend(Backend):
    # LRSchedulerKwargs fields → set of lr_scheduler_type values that accept them.
    # A scheduler not listed here accepts none of the kwargs in LRSchedulerKwargs.
    _LR_SCHEDULER_ALLOWED_KWARGS: dict[str, set[str]] = {
        "cosine_with_min_lr": {"min_lr_rate"},
        "warmup_stable_decay": {
            "num_stable_steps",
            "num_decay_steps",
            "min_lr_ratio",
            "num_cycles",
            "warmup_type",
            "decay_type",
        },
    }

    _WSD_WARMUP_DECAY_TYPES = {"linear", "cosine", "1-sqrt"}

    @classmethod
    def _validate_lr_scheduler_kwargs(cls, config: PostTrainingConfig) -> None:
        t = config.training
        allowed = cls._LR_SCHEDULER_ALLOWED_KWARGS.get(t.lr_scheduler_type, set())
        set_fields = {
            k: v for k, v in dataclasses.asdict(t.lr_scheduler_kwargs).items() if v is not None
        }
        bad = sorted(k for k in set_fields if k not in allowed)
        if bad:
            if allowed:
                raise ValueError(
                    f"training.lr_scheduler_kwargs.{bad[0]} is not a valid kwarg for "
                    f"lr_scheduler_type='{t.lr_scheduler_type}'. "
                    f"Allowed kwargs for this scheduler: {sorted(allowed)}. "
                    f"Offending kwargs: {bad}."
                )
            raise ValueError(
                f"lr_scheduler_type='{t.lr_scheduler_type}' accepts no extra kwargs, "
                f"but training.lr_scheduler_kwargs has: {bad}. Unset them or pick a "
                f"scheduler that supports them."
            )

        if t.lr_scheduler_type == "warmup_stable_decay":
            if t.lr_scheduler_kwargs.num_decay_steps is None:
                raise ValueError(
                    "training.lr_scheduler_kwargs.num_decay_steps is required when "
                    "lr_scheduler_type='warmup_stable_decay' (no default in HF)."
                )
            for field_name in ("warmup_type", "decay_type"):
                value = getattr(t.lr_scheduler_kwargs, field_name)
                if value is not None and value not in cls._WSD_WARMUP_DECAY_TYPES:
                    raise ValueError(
                        f"training.lr_scheduler_kwargs.{field_name}={value!r} is invalid. "
                        f"Must be one of {sorted(cls._WSD_WARMUP_DECAY_TYPES)}."
                    )

    def validate(self, config: PostTrainingConfig) -> None:
        if config.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{config.method}'. "
                f"Supported methods: {', '.join(_SUPPORTED_METHODS)}"
            )

        # Container validation (only when container.image is set)
        if config.container.image:
            if not config.container.bind_mounts:
                raise ValueError(
                    "container.bind_mounts must be non-empty when container.image is set."
                )
            if not config.container.env_file:
                raise ValueError("container.env_file must be set when container.image is set.")

        t = config.training

        if t.effective_batch_size <= 0:
            raise ValueError("training.effective_batch_size must be positive.")
        if t.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive.")

        self._validate_lr_scheduler_kwargs(config)

        # Training length: exactly one of max_steps / num_train_epochs /
        # num_training_samples / num_training_tokens must be specified.
        length_fields = {
            "training.max_steps": t.max_steps,
            "training.num_train_epochs": t.num_train_epochs,
            "training.num_training_samples": t.num_training_samples,
            "training.num_training_tokens": t.num_training_tokens,
        }
        specified = [name for name, value in length_fields.items() if value is not None]
        if len(specified) == 0:
            raise ValueError(
                "Exactly one of training.max_steps, training.num_train_epochs, "
                "training.num_training_samples, or training.num_training_tokens "
                "must be specified."
            )
        if len(specified) > 1:
            raise ValueError(
                "Training length is over-specified. Choose exactly one of "
                "training.max_steps, training.num_train_epochs, "
                "training.num_training_samples, or training.num_training_tokens. "
                f"You set: {', '.join(specified)}."
            )

        if t.num_train_epochs is not None and t.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be a positive number.")

        if t.num_training_tokens is not None:
            if config.method != "sft" or not config.sft.packing:
                raise ValueError(
                    "training.num_training_tokens is only valid for method='sft' "
                    "when sft.packing=true."
                )
            if t.num_training_tokens <= 0:
                raise ValueError("training.num_training_tokens must be a positive integer.")
            if config.sft.max_seq_length <= 0:
                raise ValueError("sft.max_seq_length must be positive.")

            tokens_per_step = t.effective_batch_size * config.sft.max_seq_length
            t.max_steps = math.ceil(t.num_training_tokens / tokens_per_step)

        if t.num_training_samples is not None:
            if t.num_training_samples <= 0:
                raise ValueError("training.num_training_samples must be a positive integer.")
            t.max_steps = math.ceil(t.num_training_samples / t.effective_batch_size)

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        model_short = _shorten_model_name(config.model.name_or_path)
        datasets = config.data.datasets
        if len(datasets) == 1:
            ds_part = _shorten_dataset_name(datasets[0].name)
        else:
            ds_part = f"mix_{_dataset_mix_hash(datasets)}"
        return f"{config.method}-{model_short}-{ds_part}-{timestamp}"

    def run_dir_subdirs(self) -> list[str]:
        return ["checkpoints", "inference_checkpoints"]

    def render_slurm_script(self, config, run_dir, frozen_config_path, *, tokenize_only=False):
        if config.container.image:
            from post_training.slurm.launcher import render_trl_container_slurm_script

            return render_trl_container_slurm_script(
                config, run_dir, frozen_config_path, tokenize_only=tokenize_only
            )

        from post_training.slurm.launcher import render_trl_slurm_script

        return render_trl_slurm_script(
            config, run_dir, frozen_config_path, tokenize_only=tokenize_only
        )

    def post_freeze(self, config, run_dir):
        pass


# ---------------------------------------------------------------------------
# LlamaFactory
# ---------------------------------------------------------------------------


class LlamaFactoryBackend(Backend):
    def validate(self, config: PostTrainingConfig) -> None:
        if not config.llamafactory:
            raise ValueError("llamafactory must be set when backend='llamafactory'.")
        if not config.container.image:
            raise ValueError("container.image must be set when backend='llamafactory'.")

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        return f"llamafactory-{config.method}-{timestamp}"

    def run_dir_subdirs(self) -> list[str]:
        return []

    def render_slurm_script(self, config, run_dir, frozen_config_path, *, tokenize_only=False):
        # LlamaFactory has no --tokenize-only equivalent; submit.py rejects it before this point.
        del tokenize_only
        from post_training.slurm.launcher import render_llamafactory_slurm_script

        return render_llamafactory_slurm_script(config, run_dir)

    def post_freeze(self, config, run_dir):
        lf_config_path = run_dir / "llamafactory_config.yaml"
        with open(lf_config_path, "w") as f:
            yaml.dump(config.llamafactory, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Backend] = {
    "trl": TRLBackend(),
    "llamafactory": LlamaFactoryBackend(),
}


def get_backend(name: str) -> Backend:
    """Return the backend implementation for *name*."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Supported: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]
