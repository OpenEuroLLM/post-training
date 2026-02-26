"""Asset pre-fetching for offline / air-gapped cluster runs.

When ``offline=True`` in the config, compute nodes have no internet access.
This module downloads models and datasets to the local HuggingFace cache on
the login node (which does have internet) before the SLURM job is submitted.

Models are fetched via :func:`huggingface_hub.snapshot_download` (fills
``~/.cache/huggingface/hub/``).  Datasets are fetched via
:func:`datasets.load_dataset` so that the processed Arrow cache is populated
(``~/.cache/huggingface/datasets/``), which is what ``HF_DATASETS_OFFLINE=1``
requires at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from post_training.config import DatasetEntry, PostTrainingConfig

logger = logging.getLogger(__name__)


def _is_local(path: str) -> bool:
    """Return True if *path* points to an existing local file or directory."""
    return Path(path).exists()


def _prefetch_model(name_or_path: str) -> None:
    if _is_local(name_or_path):
        logger.info("Model '%s' is a local path, skipping download.", name_or_path)
        return
    logger.info("Downloading model '%s' to HF cache...", name_or_path)
    snapshot_download(repo_id=name_or_path, repo_type="model")
    logger.info("Model '%s' cached.", name_or_path)


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


def prefetch_assets(config: PostTrainingConfig) -> None:
    """Download all models and datasets in *config* to the local HF cache.

    Safe to call multiple times — HuggingFace caching is idempotent.
    """
    logger.info("Pre-fetching assets for offline run...")

    _prefetch_model(config.model.name_or_path)

    if config.method == "dpo" and config.dpo.ref_model_name_or_path is not None:
        _prefetch_model(config.dpo.ref_model_name_or_path)

    for entry in config.data.datasets:
        _prefetch_dataset(entry)

    logger.info("All assets pre-fetched successfully.")
