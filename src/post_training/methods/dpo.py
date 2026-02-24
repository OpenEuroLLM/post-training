"""Direct preference optimisation (DPO) method."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from trl import DPOConfig, DPOTrainer

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


def _dpo_row_filter(example: dict) -> bool:
    """Keep only rows with non-empty ``chosen`` and ``rejected`` fields."""
    return len(example.get("chosen", [])) > 0 and len(example.get("rejected", [])) > 0


def build_dpo_trainer(config: PostTrainingConfig, run_dir: Path) -> DPOTrainer:
    """Build a TRL :class:`DPOTrainer` from *config*.

    Parameters
    ----------
    config:
        Fully resolved post-training configuration.
    run_dir:
        Run output directory (already created).

    Returns
    -------
    DPOTrainer
        Ready to call ``.train()``.
    """
    mc = config.dpo  # method-specific config

    tokenizer = build_tokenizer(config)
    dataset = load_and_mix_datasets(config.data, row_filter=_dpo_row_filter)

    dpo_config = DPOConfig(
        **build_common_training_kwargs(config, run_dir),
        beta=mc.beta,
        loss_type=mc.loss_type,
        max_length=mc.max_seq_length,
        dataset_num_proc=mc.dataset_num_proc,
        model_init_kwargs=build_model_init_kwargs(config),
    )

    return DPOTrainer(
        model=config.model.name_or_path,
        ref_model=mc.ref_model_name_or_path,  # None → TRL creates implicit copy
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        callbacks=build_callbacks(config, run_dir),
    )
