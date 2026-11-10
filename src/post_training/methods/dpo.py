"""Direct preference optimisation (DPO) method."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from accelerate import PartialState
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from post_training.data.loader import load_and_mix_datasets
from post_training.methods.common import (
    align_generation_eos,
    bind_inference_checkpoint_callback,
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


def _filter_dpo_rows(ds: Dataset, num_proc: int) -> Dataset:
    """Drop rows with an empty ``chosen`` or ``rejected`` field."""
    return ds.filter(
        lambda row: len(row["chosen"]) > 0 and len(row["rejected"]) > 0,
        num_proc=num_proc,
        desc="filtering empty preference pairs",
    )


_WEIGHT_FILE_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _resolve_checkpoint_dir(name_or_path: str, revision: str | None) -> Path:
    """Return the local directory holding the checkpoint files of *name_or_path*."""
    if Path(name_or_path).is_dir():
        return Path(name_or_path)
    from huggingface_hub import LocalEntryNotFoundError, snapshot_download

    try:
        return Path(
            snapshot_download(
                name_or_path, repo_type="model", revision=revision, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return Path(snapshot_download(name_or_path, repo_type="model", revision=revision))


def hash_checkpoint_files(name_or_path: str, revision: str | None, dtype: str) -> str:
    """Hash a model checkpoint's files (weights + ``config.json``) and *dtype*.

    TRL keys the precomputed reference-logprob cache (``*.npz`` next to the dataset cache)
    by ``hash_module(model)``, which hashes the in-memory ``state_dict()``. Under DeepSpeed
    ZeRO-3 the parameters are already partitioned when ``DPOTrainer.__init__`` computes it,
    so the same checkpoint hashes differently per ZeRO stage and per world size, and the
    cache is silently missed. Hashing the checkpoint files instead makes the key depend only
    on the weights, the model config, and the load dtype.
    """
    root = _resolve_checkpoint_dir(name_or_path, revision)
    files: list[Path] = []
    config_json = root / "config.json"
    if config_json.is_file():
        files.append(config_json)
    for pattern in _WEIGHT_FILE_PATTERNS:
        files.extend(sorted(root.glob(pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No config.json or weight files found for model '{name_or_path}' in '{root}'."
        )
    h = hashlib.sha256()
    h.update(b"post-training/ref-logprob-cache/v1")
    h.update(dtype.encode())
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as f:
            h.update(hashlib.file_digest(f, "sha256").digest())
    return h.hexdigest()


def install_stage_agnostic_hash_module(
    trainer_module: Any, name_or_path: str, revision: str | None, dtype: str
) -> str:
    """Point ``trainer_module.hash_module`` at a checkpoint-file hash and return that hash.

    ``trainer_module`` is ``trl.trainer.dpo_trainer`` in production; ``DPOTrainer`` looks the
    name up as a module global when it builds the ``.npz`` cache file name, so replacing it
    there redirects the cache key without touching TRL's read/write logic.
    """
    model_hash = hash_checkpoint_files(name_or_path, revision, dtype)

    def hash_module(module: Any) -> str:
        return model_hash

    trainer_module.hash_module = hash_module
    return model_hash


def build_dpo_trainer(config: PostTrainingConfig, run_dir: Path) -> DPOTrainer:
    """Build a TRL :class:`DPOTrainer` from *config*.

    Parameters
    ----------
    config:
        Fully resolved post-training configuration.
    run_dir:
        Run output directory (already created).

    Returns
    -------
    DPOTrainer
        Ready to call ``.train()``.
    """
    mc = config.dpo  # method-specific config

    tokenizer = build_tokenizer(config)
    with PartialState().main_process_first():
        dataset = load_and_mix_datasets(config.data, dataset_filter_fn=_filter_dpo_rows)

    dpo_config = DPOConfig(
        **build_common_training_kwargs(config, run_dir),
        beta=mc.beta,
        loss_type=mc.loss_type,
        max_length=mc.max_seq_length,
        dataset_num_proc=mc.dataset_num_proc,
        precompute_ref_log_probs=mc.precompute_ref_log_probs,
        model_init_kwargs=build_model_init_kwargs(config),
    )

    if mc.precompute_ref_log_probs:
        import trl.trainer.dpo_trainer as dpo_trainer_module

        ref_source = mc.ref_model_name_or_path or config.model.name_or_path
        model_hash = install_stage_agnostic_hash_module(
            dpo_trainer_module, ref_source, config.model.revision, config.model.dtype
        )
        logger.info(
            "precompute_ref_log_probs=true: reference log-prob cache key '%s' is derived from "
            "the checkpoint files of '%s' and is shared across DeepSpeed ZeRO stages.",
            model_hash,
            ref_source,
        )

    trainer = DPOTrainer(
        model=config.model.name_or_path,
        ref_model=mc.ref_model_name_or_path,  # None → TRL creates implicit copy
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        callbacks=build_callbacks(config, run_dir),
    )
    sanitize_generation_config(trainer)
    align_generation_eos(trainer)
    prioritize_metric_callbacks(trainer)
    bind_inference_checkpoint_callback(trainer)
    return trainer
