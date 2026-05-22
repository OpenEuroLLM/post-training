"""Tests for token-counting utilities."""

import pytest
from datasets import Dataset

from post_training.utils.token_stats import (
    count_dataset_tokens,
    summarize_token_lengths,
    token_stats_as_dict,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is False
        n_tokens = sum(len(message["content"].split()) for message in messages)
        return list(range(n_tokens))


class FakeDictTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        token_ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return {
            "input_ids": token_ids,
            "assistant_masks": [1] * len(token_ids),
        }


def test_summarize_token_lengths():
    stats = summarize_token_lengths([2, 4, 4])

    assert stats is not None
    assert stats.total_tokens == 10
    assert stats.avg_tokens == pytest.approx(10 / 3)
    assert stats.min_tokens == 2
    assert stats.max_tokens == 4
    assert stats.std_tokens == pytest.approx(1.1547005383792515)
    assert list(stats.as_dict()) == [
        "avg_tokens",
        "max_tokens",
        "min_tokens",
        "std_tokens",
        "total_tokens",
    ]


def test_summarize_token_lengths_single_and_empty():
    single = summarize_token_lengths([7])

    assert single is not None
    assert single.total_tokens == 7
    assert single.avg_tokens == 7
    assert single.min_tokens == 7
    assert single.max_tokens == 7
    assert single.std_tokens == 0.0
    assert summarize_token_lengths([]) is None
    assert token_stats_as_dict(None) == {}


def test_count_dataset_tokens_with_fake_tokenizer():
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "one two"},
                    {"role": "assistant", "content": "three"},
                ],
                [{"role": "user", "content": "solo"}],
            ],
            "unused": ["drop", "drop"],
        }
    )

    result = count_dataset_tokens(dataset, FakeTokenizer(), num_proc=1)

    assert result.loaded_rows == 2
    assert result.tokenized_rows == 2
    assert result.stats is not None
    assert result.stats.total_tokens == 4
    assert result.stats.avg_tokens == 2.0
    assert result.stats.min_tokens == 1
    assert result.stats.max_tokens == 3
    assert result.stats.std_tokens == pytest.approx(1.4142135623730951)


def test_count_dataset_tokens_handles_dict_tokenizer_output():
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "one two"},
                    {"role": "assistant", "content": "three four five"},
                ]
            ],
        }
    )

    result = count_dataset_tokens(dataset, FakeDictTokenizer(), num_proc=1)

    assert result.stats is not None
    assert result.stats.total_tokens == 5
    assert result.stats.avg_tokens == 5
    assert result.stats.min_tokens == 5
    assert result.stats.max_tokens == 5


def test_count_dataset_tokens_empty_dataset():
    result = count_dataset_tokens(Dataset.from_dict({"messages": []}), FakeTokenizer(), num_proc=1)

    assert result.loaded_rows == 0
    assert result.tokenized_rows == 0
    assert result.stats is None
    assert token_stats_as_dict(result.stats) == {}
