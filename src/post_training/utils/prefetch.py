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

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset
from huggingface_hub import snapshot_download

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


def _prefetch_model(name_or_path: str, revision: str | None = None) -> str:
    """Ensure *name_or_path* is cached locally and return its local directory."""
    if _is_local(name_or_path):
        logger.info("Model '%s' is a local path, skipping download.", name_or_path)
        return name_or_path
    logger.info(
        "Downloading model '%s' (revision=%s) to HF cache...", name_or_path, revision or "main"
    )
    local_path = snapshot_download(repo_id=name_or_path, repo_type="model", revision=revision)
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
