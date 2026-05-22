"""Token-counting helpers for conversational datasets."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

    from post_training.config import PostTrainingConfig


@dataclass(frozen=True)
class TokenStats:
    """Summary statistics for per-row token counts."""

    total_tokens: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    std_tokens: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a YAML-friendly mapping with stable key order."""
        return {
            "avg_tokens": self.avg_tokens,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "std_tokens": self.std_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class TokenStatsResult:
    """Token-counting result plus row counts for progress/reporting."""

    loaded_rows: int
    tokenized_rows: int
    stats: TokenStats | None


def summarize_token_lengths(lengths: Sequence[int]) -> TokenStats | None:
    """Compute summary statistics for a sequence of token lengths."""
    if not lengths:
        return None

    total_tokens = sum(lengths)
    return TokenStats(
        total_tokens=total_tokens,
        avg_tokens=total_tokens / len(lengths),
        min_tokens=min(lengths),
        max_tokens=max(lengths),
        std_tokens=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
    )


def token_stats_as_dict(stats: TokenStats | None) -> dict[str, float | int]:
    """Return stats as a YAML-friendly mapping, or ``{}`` for no valid rows."""
    return {} if stats is None else stats.as_dict()


def load_token_stats_dataset(config: PostTrainingConfig) -> Dataset:
    """Load data for token counting using the same SFT path as training."""
    from post_training.data.loader import load_and_mix_datasets
    from post_training.methods.sft import MESSAGES_FEATURES, _sft_row_filter

    if config.method != "sft":
        raise ValueError("Token stats currently support method='sft' configs only.")

    return load_and_mix_datasets(
        config.data,
        row_filter=_sft_row_filter,
        columns_to_keep=["messages"],
        features=MESSAGES_FEATURES,
    )


def count_dataset_tokens(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_proc: int | None = None,
) -> TokenStatsResult:
    """Count chat-template tokens for each row in an already-loaded dataset."""
    loaded_rows = len(dataset)
    if loaded_rows == 0:
        return TokenStatsResult(loaded_rows=0, tokenized_rows=0, stats=None)

    def token_length(example: dict[str, Any]) -> dict[str, int]:
        tokenized = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        token_ids = tokenized["input_ids"] if isinstance(tokenized, Mapping) else tokenized
        return {"num_tokens": len(token_ids)}

    map_num_proc = num_proc if num_proc is not None and num_proc > 1 else None
    tokenized = dataset.map(
        token_length,
        remove_columns=dataset.column_names,
        num_proc=map_num_proc,
        desc="Counting tokens",
    )
    lengths = tokenized["num_tokens"]
    return TokenStatsResult(
        loaded_rows=loaded_rows,
        tokenized_rows=len(tokenized),
        stats=summarize_token_lengths(lengths),
    )


def count_token_stats_for_config(
    config: PostTrainingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TokenStatsResult:
    """Load an SFT config's data and compute chat-template token stats."""
    from post_training.data.loader import _resolve_num_proc
    from post_training.methods.common import build_tokenizer

    if tokenizer is None:
        tokenizer = build_tokenizer(config)

    dataset = load_token_stats_dataset(config)
    return count_dataset_tokens(
        dataset,
        tokenizer,
        num_proc=_resolve_num_proc(config.data.num_proc),
    )
