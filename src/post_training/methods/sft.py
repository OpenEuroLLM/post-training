"""Supervised fine-tuning (SFT) method."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from accelerate import PartialState
from datasets import Dataset, Features, List, Value
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from post_training.chat_templates.registry import has_generation_markers
from post_training.data.loader import load_and_mix_datasets
from post_training.methods.common import (
    build_callbacks,
    build_common_training_kwargs,
    build_model_init_kwargs,
    build_tokenizer,
    prioritize_metric_callbacks,
    sanitize_generation_config,
)

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)

# Warn instead of info once this share of rows is dropped.
_DROP_WARN_PERCENT = 1.0

MESSAGES_FEATURES = Features(
    {
        "messages": List(
            {
                "content": Value("string"),
                "role": Value("string"),
            }
        )
    }
)


def _filter_sft_rows(
    ds: Dataset,
    num_proc: int,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> Dataset:
    """Drop rows that contribute nothing to the SFT loss.

    A row is dropped when its ``messages`` list is empty, or when rendering it
    through the chat template gives an all-zero ``assistant_masks``. The second
    case happens when a conversation ends on a non-assistant turn, or when the
    template strips the only assistant content the row holds. Such rows cost
    compute and produce a zero gradient.

    Parameters
    ----------
    ds:
        Dataset with a ``messages`` column.
    num_proc:
        Number of worker processes for ``.filter()`` and ``.map()``.
    tokenizer:
        Tokenizer whose ``chat_template`` carries ``{% generation %}`` markers.
        Bind it at the call site with :func:`functools.partial`.
    max_length:
        Optional truncation length that must match the trainer's own truncation,
        so a row whose assistant tokens fall outside the window is dropped too.
        Leave it ``None`` only if the trainer keeps every token of every row.

    Returns
    -------
    datasets.Dataset
        The rows that stay.

    Raises
    ------
    ValueError
        If no row survives either stage.
    """
    ds = ds.filter(
        lambda row: len(row["messages"]) > 0, num_proc=num_proc, desc="filtering empty messages"
    )

    if len(ds) == 0:
        raise ValueError(
            "No rows remain after dropping empty 'messages' lists. "
            "Check the dataset and its transform."
        )

    template_kwargs = {"truncation": True, "max_length": max_length} if max_length else {}

    # remove_columns keeps the map cache tiny: it holds the flag alone, not a copy of the data.
    in_loss = ds.map(
        lambda row: {
            "in_loss": any(
                tokenizer.apply_chat_template(
                    row["messages"],
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    **template_kwargs,
                )["assistant_masks"]
            )
        },
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="computing assistant loss masks",
    )["in_loss"]

    keep: list[int] = []
    dropped: list[int] = []
    for index, ok in enumerate(in_loss):
        if ok:
            keep.append(index)
        else:
            dropped.append(index)

    dropped_percent = 100 * len(dropped) / len(ds)
    summary = (
        f"{len(ds)} rows, {len(dropped)} with an all-zero assistant mask ({dropped_percent:.3f}%)."
    )

    if not keep:
        cause = (
            "The chat template is probably missing the {% generation %}…{% endgeneration %} "
            "markers around the assistant content, which makes every mask all-zero."
        )
        if max_length is not None:
            cause += (
                f" A max_length of {max_length} may also be short enough to truncate "
                "away every assistant turn."
            )
        raise ValueError(f"{summary} Every row would contribute zero loss. {cause}")

    if dropped_percent > _DROP_WARN_PERCENT:
        logger.warning(summary)
    else:
        logger.info(summary)

    if dropped:
        patterns = {
            " -> ".join(message["role"] for message in row["messages"][-3:])
            for row in ds.select(dropped)
        }
        logger.info(
            "Role sequences of the %d dropped rows (%d unique):\n%s",
            len(dropped),
            len(patterns),
            "\n".join(sorted(patterns)),
        )

    return ds.select(keep)


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
        dataset = load_and_mix_datasets(
            config.data,
            dataset_filter_fn=partial(
                _filter_sft_rows,
                tokenizer=tokenizer,
                # Both TRL paths cut each row at max_length, so the filter must
                # measure the same window. Without packing, TRL truncates the
                # row. With packing, the default "bfd" strategy keeps only the
                # first max_length fragment and discards the overflow
                # (trl/data_utils.py _pack_bfd). Revisit if packing_strategy
                # ever becomes configurable: "bfd-requeue" and "wrapped" both
                # keep the overflow, so truncation-based filtering would then
                # drop rows the trainer still trains on.
                max_length=mc.max_seq_length,
            ),
            columns_to_keep=["messages"],
            features=MESSAGES_FEATURES,
        )

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
    prioritize_metric_callbacks(trainer)
    return trainer
