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

    trainer = SFTTrainer(
        model=config.model.name_or_path,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        callbacks=build_callbacks(config, run_dir),
    )

    _sanitize_generation_config(trainer)
    return trainer


def _sanitize_generation_config(trainer: SFTTrainer) -> None:
    """Fix inconsistent ``generation_config`` so checkpoint saves don't fail.

    Some upstream models (notably Olmo-3 Think variants) ship a
    ``generation_config.json`` that sets sampling-only parameters
    (``temperature``, ``top_p``) while leaving ``do_sample=False``.  This is
    benign at training time — we never call ``model.generate`` — but
    ``transformers >= 5.x`` runs strict validation inside
    ``GenerationConfig.save_pretrained`` and refuses to write the file::

        ValueError: GenerationConfig is invalid:
          - `temperature` is set to 0.6 -- this flag is only used in
            sample-based generation modes. You should set `do_sample=True`
            or unset `temperature`.

    Every checkpoint save (HF Trainer's ``_save`` and our
    ``InferenceCheckpointCallback``) ultimately calls ``model.save_pretrained``
    which writes the generation config, so an unfixed model crashes the
    very first save.  We patch ``do_sample`` to ``True`` once, on the
    in-memory model object, immediately after the trainer (and therefore
    the model) has been constructed.  The fix is local to this run — the
    upstream model files on the Hub are unchanged.

    AllenAI's open-instruct solves the same issue in ``model_utils.py``
    by setting ``temperature=None, top_p=None`` (stripping the params).
    We instead set ``do_sample=True`` to preserve the upstream model's
    recommended inference settings in the saved checkpoint.
    """
    model = getattr(trainer, "model", None)
    if model is None:
        return
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    has_sampling_param = (
        getattr(gc, "temperature", None) is not None
        or getattr(gc, "top_p", None) is not None
        or getattr(gc, "top_k", None) not in (None, 0)
    )
    if has_sampling_param and not getattr(gc, "do_sample", False):
        logger.info(
            "Sanitizing generation_config: setting do_sample=True so that "
            "checkpoint saves can write generation_config.json without "
            "tripping transformers' strict validation."
        )
        gc.do_sample = True
