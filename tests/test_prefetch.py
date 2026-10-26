"""Tests for which repos prefetch_assets downloads."""

import json
import logging
from pathlib import Path

import pytest

from post_training.config import PostTrainingConfig
from post_training.utils import prefetch


@pytest.fixture
def downloads(monkeypatch):
    """Record snapshot_download calls instead of contacting the Hub."""
    calls = []

    def fake_snapshot_download(
        repo_id, repo_type=None, revision=None, ignore_patterns=None, cache_dir=None
    ):
        calls.append((repo_id, revision, ignore_patterns))
        return f"/cache/{repo_id}@{revision or 'main'}"

    monkeypatch.setattr(prefetch, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(prefetch, "_prefetch_dataset", lambda entry: None)
    return calls


def _config(**model_overrides) -> PostTrainingConfig:
    config = PostTrainingConfig()
    config.model.name_or_path = "org/model"
    for key, value in model_overrides.items():
        setattr(config.model, key, value)
    return config


def test_tokenizer_reuses_the_model_snapshot(downloads):
    """The model snapshot already holds the tokenizer, so nothing extra is fetched."""
    paths = prefetch.prefetch_assets(_config(revision="r1"))

    assert downloads == [("org/model", "r1", None)]
    assert paths.tokenizer == paths.model


def test_tokenizer_revision_is_fetched_separately(downloads):
    """A tokenizer pinned to another revision of the model repo needs its own snapshot."""
    paths = prefetch.prefetch_assets(_config(revision="r1", tokenizer_revision="r2"))

    assert downloads == [
        ("org/model", "r1", None),
        ("org/model", "r2", prefetch._WEIGHT_PATTERNS),
    ]
    assert paths.model == "/cache/org/model@r1"
    assert paths.tokenizer == "/cache/org/model@r2"


def test_model_revision_is_ignored_when_tokenizer_name_is_specified(downloads):
    """The model revision is ignored when a tokenizer revision is specified."""
    paths = prefetch.prefetch_assets(_config(revision="r1", tokenizer_name_or_path="org/tokenizer"))

    assert downloads == [
        ("org/model", "r1", None),
        ("org/tokenizer", None, prefetch._WEIGHT_PATTERNS),
    ]
    assert paths.model == "/cache/org/model@r1"
    assert paths.tokenizer == "/cache/org/tokenizer@main"


def test_tokenizer_repo_is_fetched_without_weights(downloads):
    """A tokenizer from another repo is fetched without the weight files."""
    paths = prefetch.prefetch_assets(_config(tokenizer_name_or_path="org/tokenizer"))

    assert downloads == [
        ("org/model", None, None),
        ("org/tokenizer", None, prefetch._WEIGHT_PATTERNS),
    ]
    assert "*.safetensors" in prefetch._WEIGHT_PATTERNS
    assert paths.tokenizer == "/cache/org/tokenizer@main"


def test_local_tokenizer_path_is_not_downloaded(downloads, tmp_path):
    """An existing local tokenizer directory is used as-is."""
    paths = prefetch.prefetch_assets(_config(tokenizer_name_or_path=str(tmp_path)))

    assert downloads == [("org/model", None, None)]
    assert paths.tokenizer == str(tmp_path)


# ---------------------------------------------------------------------------
# LlamaFactory backend
# ---------------------------------------------------------------------------


class _FakeHub:
    """Records the Hub calls that the LlamaFactory prefetch makes."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.models = []
        self.datasets = []
        self.registries = []
        self.remote = {}  # repo id -> dataset_info.json content served by hf_hub_download

    def snapshot_download(
        self, repo_id, repo_type=None, revision=None, ignore_patterns=None, cache_dir=None
    ):
        self.models.append((repo_id, revision, cache_dir))
        return f"/cache/{repo_id}@{revision or 'main'}"

    def load_dataset(self, path, **kwargs):
        self.datasets.append({"path": path, **kwargs})

    def hf_hub_download(self, repo_id, filename, repo_type=None, **kwargs):
        self.registries.append((repo_id, filename, repo_type))
        dst = self.tmp_path / "remote" / repo_id / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(self.remote[repo_id]))
        return str(dst)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    fake = _FakeHub(tmp_path)
    monkeypatch.setattr(prefetch, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(prefetch, "load_dataset", fake.load_dataset)
    monkeypatch.setattr(prefetch, "hf_hub_download", fake.hf_hub_download, raising=False)
    monkeypatch.chdir(tmp_path)
    return fake


def _lf_config(offline: bool = False, **block) -> PostTrainingConfig:
    config = PostTrainingConfig()
    config.backend = "llamafactory"
    config.offline = offline
    config.llamafactory = {"model_name_or_path": "org/lf-model", "stage": "sft", **block}
    return config


def _write_registry(entries: dict, dataset_dir: str = "data") -> None:
    registry = Path(dataset_dir) / "dataset_info.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(entries))


def _hub_load(path, name=None, data_dir=None, split="train", cache_dir=None) -> dict:
    return {
        "path": path,
        "name": name,
        "data_dir": data_dir,
        "split": split,
        "cache_dir": cache_dir,
    }


def test_llamafactory_downloads_its_model_not_the_trl_default(hub, tmp_path):
    cache_dir = str(tmp_path / "hf-cache")

    paths = prefetch.prefetch_assets(_lf_config(model_revision="r1", cache_dir=cache_dir))

    assert hub.models == [("org/lf-model", "r1", cache_dir)]
    assert paths.model == "/cache/org/lf-model@r1"
    assert paths.tokenizer == paths.model
    assert paths.ref_model is None


def test_llamafactory_ref_model_uses_the_policy_revision(hub):
    """LlamaFactory loads ref_model with the policy's model_revision, so the prefetch does too."""
    paths = prefetch.prefetch_assets(_lf_config(model_revision="r1", ref_model="org/ref"))

    assert hub.models == [("org/lf-model", "r1", None), ("org/ref", "r1", None)]
    assert paths.ref_model == "/cache/org/ref@r1"


def test_llamafactory_local_model_is_not_downloaded(hub, tmp_path):
    (tmp_path / "models" / "local").mkdir(parents=True)

    paths = prefetch.prefetch_assets(_lf_config(model_name_or_path="models/local"))

    assert hub.models == []
    assert paths.model == "models/local"


def test_llamafactory_loads_only_named_registry_entries(hub):
    """Each named hub entry is loaded as LlamaFactory loads it; local files need no download."""
    _write_registry(
        {
            "sft": {
                "hf_hub_url": "org/sft",
                "subset": "chat",
                "folder": "en",
                "split": "validation",
            },
            "unused": {"hf_hub_url": "org/unused"},
            "local": {"file_name": "local.jsonl"},
            "both": {"hf_hub_url": "org/both", "ms_hub_url": "ms/both"},
            "eval": {"hf_hub_url": "org/eval"},
        }
    )

    prefetch.prefetch_assets(_lf_config(dataset="sft, local,both", eval_dataset="eval"))

    assert hub.datasets == [
        _hub_load("org/sft", name="chat", data_dir="en", split="validation"),
        _hub_load("org/both"),
        _hub_load("org/eval"),
    ]


def test_llamafactory_unknown_dataset_name_is_rejected(hub):
    _write_registry({"sft": {"hf_hub_url": "org/sft"}})

    with pytest.raises(ValueError, match="missing_set"):
        prefetch.prefetch_assets(_lf_config(dataset="sft,missing_set"))
    assert hub.models == []
    assert hub.datasets == []


def test_llamafactory_online_names_load_with_split_train(hub):
    prefetch.prefetch_assets(_lf_config(dataset_dir="ONLINE", dataset="org/a,org/b"))

    assert hub.datasets == [_hub_load("org/a"), _hub_load("org/b")]


def test_llamafactory_remote_registry_is_downloaded_first(hub):
    hub.remote["org/registry"] = {"sft": {"hf_hub_url": "org/sft", "subset": "chat"}}

    prefetch.prefetch_assets(_lf_config(dataset_dir="REMOTE:org/registry", dataset="sft"))

    assert hub.registries == [("org/registry", "dataset_info.json", "dataset")]
    assert hub.datasets == [_hub_load("org/sft", name="chat")]


_UNFETCHABLE_ENTRIES = [
    {"ms_hub_url": "ms/unfetchable"},
    {"om_hub_url": "om/unfetchable"},
    {"cloud_file_name": "s3://bucket/unfetchable.json"},
    {"script_url": "unfetchable_script"},
]


@pytest.mark.parametrize("entry", _UNFETCHABLE_ENTRIES)
def test_llamafactory_offline_rejects_sources_it_cannot_fetch(hub, entry):
    """An offline job could not load these, so submission stops before any download."""
    _write_registry({"sft": {"hf_hub_url": "org/sft"}, "unfetchable_set": entry})

    with pytest.raises(ValueError, match="unfetchable_set"):
        prefetch.prefetch_assets(_lf_config(offline=True, dataset="sft,unfetchable_set"))
    assert hub.models == []
    assert hub.datasets == []


def test_llamafactory_offline_rejects_streaming(hub):
    _write_registry({"sft": {"hf_hub_url": "org/sft"}})

    with pytest.raises(ValueError, match="streaming"):
        prefetch.prefetch_assets(_lf_config(offline=True, dataset="sft", streaming=True))
    assert hub.models == []
    assert hub.datasets == []


@pytest.mark.parametrize("entry", _UNFETCHABLE_ENTRIES)
def test_llamafactory_online_skips_sources_it_cannot_fetch(hub, caplog, entry):
    """Without offline mode the job can load these itself, so the prefetch only logs them."""
    _write_registry({"sft": {"hf_hub_url": "org/sft"}, "unfetchable_set": entry})

    with caplog.at_level(logging.INFO, logger=prefetch.logger.name):
        prefetch.prefetch_assets(_lf_config(dataset="sft,unfetchable_set"))

    assert hub.models == [("org/lf-model", None, None)]
    assert hub.datasets == [_hub_load("org/sft")]
    assert "unfetchable_set" in caplog.text


def test_llamafactory_tokenized_data_skips_datasets(hub, tmp_path):
    """LlamaFactory loads a non-empty tokenized_path and ignores the dataset arguments."""
    (tmp_path / "tokenized").mkdir()
    (tmp_path / "tokenized" / "dataset_dict.json").write_text("{}")

    prefetch.prefetch_assets(_lf_config(dataset="not_in_any_registry", tokenized_path="tokenized"))

    assert hub.models == [("org/lf-model", None, None)]
    assert hub.datasets == []
