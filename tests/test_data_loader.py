"""Tests for dataset loading, schema enforcement, and weight semantics."""

from collections import Counter

import pytest
from datasets import Dataset, Features, List, Value

from post_training.config import DataConfig, DatasetEntry
from post_training.data import loader


def _dataset(source: str, n: int) -> Dataset:
    return Dataset.from_dict(
        {
            "source": [source] * n,
            "idx": list(range(n)),
            "messages": [
                [
                    {"role": "user", "content": f"{source}-{i}"},
                    {"role": "assistant", "content": f"reply-{i}"},
                ]
                for i in range(n)
            ],
        }
    )


def _patch_load_dataset(monkeypatch, datasets_by_path: dict[str, Dataset]) -> None:
    def fake_load_dataset(path: str, **kwargs) -> Dataset:
        del kwargs
        return datasets_by_path[path]

    monkeypatch.setattr(loader, "load_dataset", fake_load_dataset)


def _config(*entries: DatasetEntry, seed: int = 123) -> DataConfig:
    return DataConfig(num_proc=1, seed=seed, datasets=list(entries))


def test_resolve_num_proc_auto_uses_affinity_cap(monkeypatch):
    monkeypatch.setattr(loader, "_MAX_NUM_PROC", 8)
    monkeypatch.setattr(loader.os, "cpu_count", lambda: 4)

    assert loader._resolve_num_proc(None) == 8


def test_resolve_num_proc_clamps_explicit_value_to_affinity_cap(monkeypatch):
    monkeypatch.setattr(loader, "_MAX_NUM_PROC", 8)
    monkeypatch.setattr(loader.os, "cpu_count", lambda: 64)

    assert loader._resolve_num_proc(16) == 8


def test_weight_resampling_uses_each_dataset_size(monkeypatch):
    _patch_load_dataset(
        monkeypatch,
        {
            "dataset-a": _dataset("a", 4),
            "dataset-b": _dataset("b", 2),
        },
    )

    mixed = loader.load_and_mix_datasets(
        _config(
            DatasetEntry(name="a", path="dataset-a", weight=0.5),
            DatasetEntry(name="b", path="dataset-b", weight=2.0),
        )
    )

    assert len(mixed) == 6
    assert Counter(mixed["source"]) == {"a": 2, "b": 4}


def test_weight_resampling_is_reproducible(monkeypatch):
    _patch_load_dataset(
        monkeypatch,
        {
            "dataset-a": _dataset("a", 5),
            "dataset-b": _dataset("b", 5),
        },
    )
    config = _config(
        DatasetEntry(name="a", path="dataset-a", weight=0.6),
        DatasetEntry(name="b", path="dataset-b", weight=0.6),
        seed=7,
    )

    first = loader.load_and_mix_datasets(config)
    second = loader.load_and_mix_datasets(config)
    different_seed = loader.load_and_mix_datasets(
        _config(
            DatasetEntry(name="a", path="dataset-a", weight=0.6),
            DatasetEntry(name="b", path="dataset-b", weight=0.6),
            seed=8,
        )
    )

    assert list(zip(first["source"], first["idx"], strict=True)) == list(
        zip(second["source"], second["idx"], strict=True)
    )
    assert list(zip(first["source"], first["idx"], strict=True)) != list(
        zip(different_seed["source"], different_seed["idx"], strict=True)
    )


def test_zero_negative_and_all_zero_weights(monkeypatch):
    _patch_load_dataset(
        monkeypatch,
        {
            "dataset-a": _dataset("a", 2),
            "dataset-b": _dataset("b", 2),
        },
    )

    mixed = loader.load_and_mix_datasets(
        _config(
            DatasetEntry(name="a", path="dataset-a", weight=0.0),
            DatasetEntry(name="b", path="dataset-b", weight=1.0),
        )
    )
    assert Counter(mixed["source"]) == {"b": 2}

    with pytest.raises(ValueError, match="non-negative"):
        loader.load_and_mix_datasets(_config(DatasetEntry(name="a", path="dataset-a", weight=-0.1)))

    with pytest.raises(ValueError, match="No rows left"):
        loader.load_and_mix_datasets(_config(DatasetEntry(name="a", path="dataset-a", weight=0.0)))


def test_native_dataset_preserves_extra_message_fields(monkeypatch):
    _patch_load_dataset(
        monkeypatch,
        {
            "dataset-a": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {
                                "role": "user",
                                "content": "Use this function.",
                                "functions": '[{"name":"lookup"}]',
                                "function_calls": None,
                            },
                            {
                                "role": "assistant",
                                "content": None,
                                "functions": None,
                                "function_calls": 'lookup(query="x")',
                            },
                        ]
                    ],
                    "unused": ["drop me"],
                }
            )
        },
    )
    features = Features(
        {
            "messages": List(
                {
                    "content": Value("string"),
                    "role": Value("string"),
                }
            )
        }
    )

    mixed = loader.load_and_mix_datasets(
        _config(DatasetEntry(name="a", path="dataset-a")),
        columns_to_keep=["messages"],
        features=features,
    )

    assert mixed.column_names == ["messages"]
    assert "functions" in mixed.features["messages"].feature
    assert "function_calls" in mixed.features["messages"].feature
    assert mixed[0]["messages"][0]["functions"] == '[{"name":"lookup"}]'
    assert mixed[0]["messages"][1]["function_calls"] == 'lookup(query="x")'


def test_native_plain_chat_dataset_skips_schema_cast(monkeypatch):
    _patch_load_dataset(monkeypatch, {"dataset-a": _dataset("a", 2)})
    features = Features(
        {
            "messages": List(
                {
                    "content": Value("string"),
                    "role": Value("string"),
                }
            )
        }
    )

    mixed = loader.load_and_mix_datasets(
        _config(DatasetEntry(name="a", path="dataset-a")),
        columns_to_keep=["messages"],
        features=features,
    )

    assert mixed.column_names == ["messages"]
assert {(msgs[0]["role"], msgs[0]["content"]) for msgs in mixed["messages"]} == {("user", "a-0"), ("user", "a-1")}


def test_features_with_transform_remove_original_columns(monkeypatch):
    _patch_load_dataset(
        monkeypatch,
        {
            "dataset-a": Dataset.from_dict(
                {
                    "prompt": ["hello"],
                    "unused": ["drop me"],
                }
            )
        },
    )
    monkeypatch.setattr(
        loader,
        "get_transform",
        lambda name: lambda row: {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": "hi"},
            ]
        },
    )
    features = Features(
        {
            "messages": List(
                {
                    "content": Value("string"),
                    "role": Value("string"),
                }
            )
        }
    )

    mixed = loader.load_and_mix_datasets(
        _config(DatasetEntry(name="a", path="dataset-a", transform="custom")),
        features=features,
    )

    assert mixed.column_names == ["messages"]
    assert mixed.features == features
