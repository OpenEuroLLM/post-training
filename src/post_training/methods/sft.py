"""Supervised fine-tuning (SFT) method."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from accelerate import PartialState
from trl import SFTConfig, SFTTrainer

from post_training.chat_templates.registry import has_generation_markers
from post_training.data.loader import load_and_mix_datasets
from post_training.methods.common import (
    build_callbacks,
    build_common_training_kwargs,
    build_model_init_kwargs,
    build_tokenizer,
    sanitize_generation_config,
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

    # Fail fast if the chat template can't drive `assistant_only_loss=True`.
    # Missing markers silently degrade SFT to full-sequence loss — a 21h run
    # produces a measurably worse model and nothing in the logs shouts.
    if not has_generation_markers(tokenizer.chat_template):
        raise ValueError(
            f"Chat template '{config.data.chat_template}' is missing "
            "{% generation %}…{% endgeneration %} markers\n"
            "around the assistant content emission. Without them, "
            "`assistant_only_loss=True`\n"
            "is a silent no-op — SFT would compute CE loss on every "
            "token in the sequence\n"
            "(system + user + assistant).\n"
            "\n"
            "To fix:\n"
            "  • Switch to a registered marker-bearing template:\n"
            '      data.chat_template: "olmo3-instruct-sft"   '
            "# AllenAI OLMo-3-Instruct-SFT recipe\n"
            '      data.chat_template: "olmo3-think-sft"      '
            "# AllenAI OLMo-3-Think-SFT recipe\n"
            "  • Or add `{% generation %}…{% endgeneration %}` markers "
            "around the assistant\n"
            "    content emission in your own jinja template.\n"
            "\n"
            "Reference: open-instruct's sft_tulu_tokenize_and_truncate_v1\n"
            "(open-instruct/open_instruct/dataset_transformation.py L1111-L1176)."
        )

    with PartialState().main_process_first():
        dataset = load_and_mix_datasets(config.data, row_filter=_sft_row_filter)

    sft_config = SFTConfig(
        **build_common_training_kwargs(config, run_dir),
        max_length=mc.max_seq_length,
        packing=mc.packing,
        dataset_num_proc=mc.dataset_num_proc,
        model_init_kwargs=build_model_init_kwargs(config),
        # Mask loss on everything except the assistant content.  Requires the
        # chat template to wrap assistant turns in {% generation %}…{% endgeneration %}.
        # Without this, SFTTrainer trains on user + system tokens too.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=config.model.name_or_path,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        callbacks=build_callbacks(config, run_dir),
    )

    sanitize_generation_config(trainer)
    return trainer
