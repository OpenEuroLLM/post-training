"""Supervised fine-tuning (SFT) method."""

from __future__ import annotations

import logging
from collections import Counter
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

# What `max_length` does to one row's supervised span.
_KEEP = 0  # the whole supervised span survives
_CUT = 1  # kept, but max_length cuts the span part-way through
_NO_ASSISTANT = 2  # dropped: no assistant tokens anywhere in the row
_BEYOND_CAP = 3  # dropped: assistant tokens exist, all of them past max_length

#: valid values for SFTMethodConfig.truncated_span_action
_TRUNCATED_SPAN_ACTIONS = ("warn", "drop")


def _classify_row(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None,
) -> int:
    """Say what ``max_length`` does to one row's supervised span.

    One untruncated render answers all three questions, so this costs the same
    as the plain ``any(assistant_masks)`` check and distinguishes three cases a
    single boolean conflates:

    * no assistant tokens at all — the row's SHAPE is wrong, or the template
      excludes every turn it has
    * assistant tokens exist but lie entirely past ``max_length`` — the CAP is
      wrong, and the row would be fine at a larger one
    * the span starts inside the window and continues past it — the row stays in
      training and teaches an answer that stops mid-sentence

    The third is the dangerous one: under final-turn-only supervision the
    assistant content sits at the END of the row, so a row only slightly over
    ``max_length`` keeps a non-zero mask, passes any ``any(...)`` check, and
    trains on a truncated trace with no end-of-turn token. Nothing downstream
    reports it.

    The render must be UNTRUNCATED: a truncated mask stops at ``max_length``, so
    the third case would be invisible. The window is then applied by slicing,
    which is right-truncation by construction — what both TRL paths do (the
    non-packing path truncates the row; ``bfd`` keeps the first fragment).
    """
    rendered = tokenizer.apply_chat_template(
        messages,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = rendered["assistant_masks"]

    if not any(mask):
        return _NO_ASSISTANT
    if max_length is None:
        return _KEEP
    if not any(mask[:max_length]):
        return _BEYOND_CAP
    if any(mask[max_length:]):
        return _CUT
    return _KEEP


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
    truncated_span_action: str = "warn",
) -> Dataset:
    """Drop rows that contribute nothing to the SFT loss, and report what does.

    A row is dropped when its ``messages`` list is empty, or when rendering it
    through the chat template leaves no assistant token inside the ``max_length``
    window. The second case happens when a conversation ends on a non-assistant
    turn, when the template strips the only assistant content the row holds, or
    when the assistant turn sits entirely past the cap. Such rows cost compute
    and produce a zero gradient.

    Dropping is only half the job. A row whose supervised span STARTS inside the
    window and continues past it is kept — it does produce gradient — but it
    teaches an answer that stops mid-sentence with no end-of-turn token. That is
    invisible to a keep/drop count, so it is reported separately and loudly; see
    :func:`_classify_row`.

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
    truncated_span_action:
        ``"warn"`` (default) keeps a row whose supervised span straddles
        ``max_length`` and reports it; ``"drop"`` removes it, for when the
        sequence length is fixed by memory and shortening the data is the only
        lever left. Drop still warns: it is not the quiet option, because
        ``data.datasets[].weight`` multiplies the rows that survive this filter,
        so an uneven drop changes the mixture the config asked for.

    Returns
    -------
    datasets.Dataset
        The rows that stay.

    Raises
    ------
    ValueError
        If no row survives either stage, or ``truncated_span_action`` is not a
        recognised value.
    """
    if truncated_span_action not in _TRUNCATED_SPAN_ACTIONS:
        raise ValueError(
            f"sft.truncated_span_action must be one of "
            f"{', '.join(_TRUNCATED_SPAN_ACTIONS)}; got {truncated_span_action!r}."
        )

    ds = ds.filter(
        lambda row: len(row["messages"]) > 0, num_proc=num_proc, desc="filtering empty messages"
    )

    if len(ds) == 0:
        raise ValueError(
            "No rows remain after dropping empty 'messages' lists. "
            "Check the dataset and its transform."
        )

    # remove_columns keeps the map cache tiny: it holds the verdict alone, not a copy of the data.
    verdicts = ds.map(
        lambda row: {"verdict": _classify_row(row["messages"], tokenizer, max_length)},
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="computing assistant loss masks",
    )["verdict"]

    counts = Counter(verdicts)
    cut_is_kept = truncated_span_action == "warn"
    kept_verdicts = (_KEEP, _CUT) if cut_is_kept else (_KEEP,)
    keep: list[int] = []
    dropped: list[int] = []
    for index, verdict in enumerate(verdicts):
        (keep if verdict in kept_verdicts else dropped).append(index)

    dropped_percent = 100 * len(dropped) / len(ds)
    # "within the length window" only when there IS one — the total-drop error
    # below must not name max_length when it was never set.
    scope = " within the length window" if max_length is not None else ""
    summary = (
        f"{len(ds)} rows, {len(dropped)} with an all-zero assistant mask{scope} "
        f"({dropped_percent:.3f}%)."
    )

    if not keep:
        if counts[_BEYOND_CAP] and not counts[_NO_ASSISTANT]:
            # Every row HAS an assistant turn, so the template is fine and the
            # cap is the whole story. Sending the reader to the template here
            # would cost them a long detour.
            cause = (
                f"Every row has assistant tokens, but in every row all of them fall "
                f"beyond max_length={max_length}. This is a max_seq_length problem, "
                f"not a template problem: raise sft.max_seq_length, or shorten the "
                f"rows upstream."
            )
        else:
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

    # Warned about in BOTH modes, at any rate above zero. Under "warn" the rows are
    # kept, so no drop count can reveal them. Under "drop" they are gone, which is
    # not quiet either: data.datasets[].weight multiplies surviving rows, so an
    # uneven drop silently changes this dataset's share of the mixture.
    if counts[_CUT]:
        logger.warning(
            "%d of %d rows (%.3f%%) are cut by max_length=%d PART-WAY THROUGH their "
            "supervised span — each teaches an answer that stops mid-sentence with no "
            "end-of-turn token, an unterminated <think> block for reasoning data. %s "
            "Raising sft.max_seq_length above the rows' length is the fix that keeps "
            "the data.",
            counts[_CUT],
            len(ds),
            100 * counts[_CUT] / len(ds),
            max_length,
            (
                "They STAY in training (sft.truncated_span_action='warn')."
                if cut_is_kept
                else "They were DROPPED (sft.truncated_span_action='drop'), which "
                "shifts this dataset's share of the mixture: data.datasets[].weight "
                "applies to the rows that survive filtering."
            ),
        )

    if max_length is not None and (counts[_NO_ASSISTANT] or counts[_BEYOND_CAP]):
        # Which of the two reasons a row was dropped decides what to change:
        # the data and template for the first, sft.max_seq_length for the second.
        logger.info(
            "Dropped rows by cause: %d with no assistant tokens at all, "
            "%d with every assistant token beyond max_length=%d%s.",
            counts[_NO_ASSISTANT],
            counts[_BEYOND_CAP],
            max_length,
            "" if cut_is_kept else f", {counts[_CUT]} with a span cut part-way",
        )

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
                truncated_span_action=mc.truncated_span_action,
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
