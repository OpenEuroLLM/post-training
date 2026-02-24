"""Supervised fine-tuning (SFT) method."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from trl import SFTConfig, SFTTrainer

from post_training.data.loader import load_and_mix_datasets
from post_training.methods.common import (
    build_callbacks,
    build_common_training_kwargs,
    build_model_init_kwargs,
    build_tokenizer,
)

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)


def _sft_row_filter(example: dict) -> bool:
    """Keep only rows with a non-empty ``messages`` list."""
    return len(example.get("messages", [])) > 0


def build_sft_trainer(config: PostTrainingConfig, run_dir: Path) -> SFTTrainer:
    """Build a TRL :class:`SFTTrainer` from *config*.

    Parameters
    ----------
    config:
        Fully resolved post-training configuration.
    run_dir:
        Run output directory (already created).

    Returns
    -------
    SFTTrainer
        Ready to call ``.train()``.
    """
    mc = config.sft  # method-specific config

    tokenizer = build_tokenizer(config)
    dataset = load_and_mix_datasets(config.data, row_filter=_sft_row_filter)

    sft_config = SFTConfig(
        **build_common_training_kwargs(config, run_dir),
        max_length=mc.max_seq_length,
        packing=mc.packing,
        dataset_num_proc=mc.dataset_num_proc,
        model_init_kwargs=build_model_init_kwargs(config),
    )

    return SFTTrainer(
        model=config.model.name_or_path,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        callbacks=build_callbacks(config, run_dir),
    )
