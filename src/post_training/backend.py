"""Training backend dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

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
    def validate(self, config: PostTrainingConfig) -> None:
        if config.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{config.method}'. "
                f"Supported methods: {', '.join(_SUPPORTED_METHODS)}"
            )

        t = config.training

        if t.effective_batch_size <= 0:
            raise ValueError("training.effective_batch_size must be positive.")
        if t.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive.")

        length_fields = {
            "training.max_steps": t.max_steps,
            "training.num_training_samples": t.num_training_samples,
            "training.num_training_tokens": t.num_training_tokens,
        }
        specified = [name for name, value in length_fields.items() if value is not None]
        if len(specified) == 0:
            raise ValueError(
                "Exactly one of training.max_steps, training.num_training_samples, "
                "or training.num_training_tokens must be specified."
            )
        if len(specified) > 1:
            raise ValueError(
                "Training length is over-specified. Choose exactly one of "
                f"training.max_steps, training.num_training_samples, or "
                f"training.num_training_tokens. You set: {', '.join(specified)}."
            )

        if t.num_training_tokens is not None:
            if config.method != "sft" or not config.sft.packing:
                raise ValueError(
                    "training.num_training_tokens is only valid for method='sft' "
                    "when sft.packing=true."
                )
            if t.num_training_tokens <= 0:
                raise ValueError(
                    "training.num_training_tokens must be a positive integer."
                )
            if config.sft.max_seq_length <= 0:
                raise ValueError("sft.max_seq_length must be positive.")

            tokens_per_step = t.effective_batch_size * config.sft.max_seq_length
            t.max_steps = math.ceil(t.num_training_tokens / tokens_per_step)

        if t.num_training_samples is not None:
            if t.num_training_samples <= 0:
                raise ValueError(
                    "training.num_training_samples must be a positive integer."
                )
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

    def render_slurm_script(self, config, run_dir, frozen_config_path):
        from post_training.slurm.launcher import render_trl_slurm_script

        return render_trl_slurm_script(config, run_dir, frozen_config_path)

    def post_freeze(self, config, run_dir):
        pass


# ---------------------------------------------------------------------------
# LlamaFactory
# ---------------------------------------------------------------------------


class LlamaFactoryBackend(Backend):
    def validate(self, config: PostTrainingConfig) -> None:
        if not config.llamafactory_config:
            raise ValueError(
                "llamafactory_config must be set when backend='llamafactory'."
            )
        if not config.container.image:
            raise ValueError(
                "container.image must be set when backend='llamafactory'."
            )

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        config_stem = Path(config.llamafactory_config).stem
        return f"llamafactory-{config_stem}-{timestamp}"

    def run_dir_subdirs(self) -> list[str]:
        return []

    def render_slurm_script(self, config, run_dir, frozen_config_path):
        from post_training.slurm.launcher import render_llamafactory_slurm_script

        return render_llamafactory_slurm_script(config, run_dir)

    def post_freeze(self, config, run_dir):
        shutil.copy2(config.llamafactory_config, run_dir / "llamafactory_config.yaml")


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
        raise ValueError(
            f"Unknown backend '{name}'. Supported: {', '.join(_BACKENDS)}"
        )
    return _BACKENDS[name]
