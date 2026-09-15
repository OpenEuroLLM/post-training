"""Asset pre-fetching for offline / air-gapped cluster runs.

When ``offline=True`` in the config, compute nodes have no internet access.
This module downloads models and datasets to the local HuggingFace cache on
the login node (which does have internet) before the SLURM job is submitted.

Models are fetched via :func:`huggingface_hub.snapshot_download` (fills
``~/.cache/huggingface/hub/``).  Datasets are fetched via
:func:`datasets.load_dataset` so that the processed Arrow cache is populated
(``~/.cache/huggingface/datasets/``), which is what ``HF_DATASETS_OFFLINE=1``
requires at runtime.

For ``backend: llamafactory`` the assets come from the ``llamafactory`` block
instead, and are fetched with the same arguments that LlamaFactory uses.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

if TYPE_CHECKING:
    from post_training.config import DatasetEntry, PostTrainingConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PrefetchedPaths:
    """Resolved local snapshot directories for the models in *config*.

    Substituting these back into ``config.model.name_or_path``,
    ``config.model.tokenizer_name_or_path`` (and, for DPO,
    ``config.dpo.ref_model_name_or_path``) before training starts makes
    ``from_pretrained`` treat the model as a plain local directory, skipping
    huggingface_hub's cache-resolution/filelock path entirely — the source of
    spurious "does not appear to have a file named ..." errors when many
    ranks call ``from_pretrained`` on the same shared (e.g. Lustre) cache at
    once.

    ``tokenizer`` equals ``model`` when the tokenizer comes from the model
    repo, because the model snapshot already contains the tokenizer files.
    """

    model: str
    tokenizer: str
    ref_model: str | None = None


def _is_local(path: str) -> bool:
    """Return True if *path* points to an existing local file or directory."""
    return Path(path).exists()


def _prefetch_model(
    name_or_path: str, revision: str | None = None, cache_dir: str | None = None
) -> str:
    """Ensure *name_or_path* is cached locally and return its local directory."""
    if _is_local(name_or_path):
        logger.info("Model '%s' is a local path, skipping download.", name_or_path)
        return name_or_path
    logger.info(
        "Downloading model '%s' (revision=%s) to HF cache...", name_or_path, revision or "main"
    )
    local_path = snapshot_download(
        repo_id=name_or_path, repo_type="model", revision=revision, cache_dir=cache_dir
    )
    logger.info("Model '%s' cached at '%s'.", name_or_path, local_path)
    return local_path


# Weight formats, skipped when a repo is fetched for its tokenizer alone.
# Denying the large blobs is safer than allowing a list of tokenizer files:
# every tokenizer format arrives, including the sentencepiece and legacy names
# (``spiece.model``, ``sentencepiece.bpe.model``, ``bpe.codes``, ``*.spm``) and
# the ``additional_chat_templates/`` directory.  The worst case is a few extra
# small files, not a missing file on an offline compute node.
_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "*.ot",  # rust_model.ot
    "*.tflite",
    "*.mlmodel",
    "*.npz",
]


def _prefetch_tokenizer(name_or_path: str, revision: str | None = None) -> str:
    """Cache *name_or_path* without its weights; return the local directory."""
    if _is_local(name_or_path):
        logger.info("Tokenizer '%s' is a local path, skipping download.", name_or_path)
        return name_or_path
    logger.info(
        "Downloading tokenizer '%s' (revision=%s) to HF cache...", name_or_path, revision or "main"
    )
    local_path = snapshot_download(
        repo_id=name_or_path,
        repo_type="model",
        revision=revision,
        ignore_patterns=_WEIGHT_PATTERNS,
    )
    logger.info("Tokenizer '%s' cached at '%s'.", name_or_path, local_path)
    return local_path


def _prefetch_dataset(entry: DatasetEntry) -> None:
    if _is_local(entry.path):
        logger.info("Dataset '%s' is a local path, skipping download.", entry.name)
        return
    logger.info("Downloading dataset '%s' ('%s') to HF cache...", entry.name, entry.path)
    load_kwargs: dict = {}
    if entry.data_dir is not None:
        load_kwargs["data_dir"] = entry.data_dir
    if entry.subset is not None:
        load_kwargs["name"] = entry.subset
    load_dataset(entry.path, split=entry.split, **load_kwargs)
    logger.info("Dataset '%s' cached.", entry.name)


def prefetch_assets(config: PostTrainingConfig) -> PrefetchedPaths:
    """Download all models and datasets in *config* to the local HF cache.

    Safe to call multiple times — HuggingFace caching is idempotent. Returns
    the resolved local model directories so the caller can substitute them
    into the config before it's frozen (see :class:`PrefetchedPaths`).
    """
    logger.info("Pre-fetching assets for offline run...")

    if config.backend == "llamafactory":
        return _prefetch_llamafactory(config)

    model_path = _prefetch_model(config.model.name_or_path, revision=config.model.revision)

    tokenizer_name_or_path, tokenizer_revision = config.model.resolve_tokenizer()
    if (tokenizer_name_or_path, tokenizer_revision) == (
        config.model.name_or_path,
        config.model.revision,
    ):
        # The tokenizer files are part of the model snapshot fetched above.
        tokenizer_path = model_path
    else:
        tokenizer_path = _prefetch_tokenizer(tokenizer_name_or_path, revision=tokenizer_revision)

    ref_model_path = None
    if config.method == "dpo" and config.dpo.ref_model_name_or_path is not None:
        ref_model_path = _prefetch_model(config.dpo.ref_model_name_or_path)

    for entry in config.data.datasets:
        _prefetch_dataset(entry)

    logger.info("All assets pre-fetched successfully.")
    return PrefetchedPaths(model=model_path, tokenizer=tokenizer_path, ref_model=ref_model_path)


# ---------------------------------------------------------------------------
# LlamaFactory
# ---------------------------------------------------------------------------

_LLAMAFACTORY_REGISTRY = "dataset_info.json"


@dataclasses.dataclass
class _HubDataset:
    """One load_dataset call, with the arguments LlamaFactory passes for a Hub dataset."""

    name: str  # the name used in `dataset` / `eval_dataset`
    path: str
    subset: str | None = None
    folder: str | None = None
    split: str = "train"


def _llamafactory_dataset_names(lf: dict) -> list[str]:
    """The names in `dataset` and `eval_dataset`, split like LlamaFactory's split_arg()."""
    names: list[str] = []
    for key in ("dataset", "eval_dataset"):
        value = lf.get(key)
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list):
            names.extend(str(name).strip() for name in value if str(name).strip())
    return names


def _llamafactory_has_tokenized_data(path: str) -> bool:
    """Same test as LlamaFactory's has_tokenized_data(): a directory that is not empty."""
    tokenized = Path(path)
    return tokenized.is_dir() and any(tokenized.iterdir())


def _llamafactory_source(entry: dict) -> str:
    """The load_from that LlamaFactory's get_dataset_list() picks for a registry entry.

    LlamaFactory also lets USE_MODELSCOPE_HUB / USE_OPENMIND_HUB prefer those hubs;
    the jobs here do not set them.
    """
    has_hf, has_ms, has_om = ("hf_hub_url" in entry, "ms_hub_url" in entry, "om_hub_url" in entry)
    if has_hf or has_ms or has_om:
        if has_ms and not has_hf:
            return "ms_hub"
        if has_om and not has_hf:
            return "om_hub"
        return "hf_hub"
    if "script_url" in entry:
        return "script"
    if "cloud_file_name" in entry:
        return "cloud_file"
    return "file"


def _read_llamafactory_registry(dataset_dir: str) -> dict:
    if dataset_dir.startswith("REMOTE:"):
        registry = hf_hub_download(
            repo_id=dataset_dir[len("REMOTE:") :],
            filename=_LLAMAFACTORY_REGISTRY,
            repo_type="dataset",
        )
    else:
        registry = Path(dataset_dir) / _LLAMAFACTORY_REGISTRY
    return json.loads(Path(registry).read_text())


def _plan_llamafactory_datasets(lf: dict) -> tuple[list[_HubDataset], list[str]]:
    """Return the Hub loads to pre-fetch, and the datasets the prefetch cannot fetch."""
    names = _llamafactory_dataset_names(lf)
    tokenized_path = lf.get("tokenized_path")
    if isinstance(tokenized_path, str) and _llamafactory_has_tokenized_data(tokenized_path):
        logger.info(
            "tokenized_path '%s' holds data; LlamaFactory ignores the datasets.", tokenized_path
        )
        return [], []
    if not names:
        return [], []

    dataset_dir = lf.get("dataset_dir") or "data"
    loads: list[_HubDataset] = []
    unfetchable: list[str] = []
    if dataset_dir == "ONLINE":
        loads = [_HubDataset(name=name, path=name) for name in names]
    else:
        registry = _read_llamafactory_registry(dataset_dir)
        for name in names:
            if name not in registry:
                raise ValueError(
                    f"Dataset '{name}' is not defined in {dataset_dir}/{_LLAMAFACTORY_REGISTRY}."
                )
            entry = registry[name]
            source = _llamafactory_source(entry)
            if source == "hf_hub":
                loads.append(
                    _HubDataset(
                        name=name,
                        path=entry["hf_hub_url"],
                        subset=entry.get("subset"),
                        folder=entry.get("folder"),
                        split=entry.get("split", "train"),
                    )
                )
            elif source != "file":
                unfetchable.append(f"{name} ({source})")

    # A streaming load reads the Hub directly and does not use the prepared cache.
    if lf.get("streaming") and loads:
        unfetchable.extend(f"{load.name} (streaming)" for load in loads)
        loads = []
    return loads, unfetchable


def _prefetch_llamafactory(config: PostTrainingConfig) -> PrefetchedPaths:
    """Download what llamafactory-cli loads: the model, ref_model and the Hub datasets.

    The dataset check runs first, so an offline run that could not load a dataset
    is refused before any large download.
    """
    lf = config.llamafactory
    loads, unfetchable = _plan_llamafactory_datasets(lf)
    if unfetchable:
        if config.offline:
            raise ValueError(
                "offline=true, but the prefetch cannot download these datasets, so the job "
                f"could not load them: {', '.join(unfetchable)}."
            )
        logger.warning(
            "Not pre-fetching these datasets; the job loads them itself: %s",
            ", ".join(unfetchable),
        )

    # LlamaFactory loads ref_model with a copy of the policy's model arguments.
    cache_dir = lf.get("cache_dir")
    revision = lf.get("model_revision")
    model_path = _prefetch_model(lf["model_name_or_path"], revision=revision, cache_dir=cache_dir)
    ref_model_path = None
    if lf.get("ref_model"):
        ref_model_path = _prefetch_model(lf["ref_model"], revision=revision, cache_dir=cache_dir)

    for load in loads:
        logger.info("Downloading dataset '%s' ('%s') to HF cache...", load.name, load.path)
        load_dataset(
            load.path, name=load.subset, data_dir=load.folder, split=load.split, cache_dir=cache_dir
        )

    logger.info("All assets pre-fetched successfully.")
    return PrefetchedPaths(model=model_path, tokenizer=model_path, ref_model=ref_model_path)
