"""Direct preference optimisation (DPO) method."""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from collections.abc import Mapping
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
_CACHE_KEY_VERSION = b"post-training/ref-logprob-cache/v2"


def _snapshot_commit(repo_id: str, revision: str | None) -> str:
    """Resolve a Hub repo id to the commit SHA of its snapshot.

    Only ``config.json`` is fetched when the snapshot is not cached yet, so resolving the
    SHA never downloads weight files, GGUFs, or other large blobs.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    kwargs = {"repo_type": "model", "revision": revision, "allow_patterns": ["config.json"]}
    try:
        path = snapshot_download(repo_id, local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        path = snapshot_download(repo_id, local_files_only=False, **kwargs)
    return Path(path).name  # .../snapshots/<commit_sha>


def _cache_snapshot_sha(path: Path) -> str | None:
    """Return the commit SHA if *path* is a HuggingFace cache snapshot directory."""
    resolved = path.resolve()
    if resolved.parent.name == "snapshots":
        return resolved.name
    return None


def _safetensors_header(path: Path) -> bytes:
    """Read only the JSON header of a safetensors file (the first 8 bytes give its length)."""
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return f.read(header_len)


def _update_local_dir(h: Any, root: Path) -> None:
    """Feed a plain local checkpoint directory into *h* without reading weight contents.

    ``config.json`` and the shard index are hashed byte-for-byte; for every file the name,
    size, and ``st_mtime_ns`` are used, plus the safetensors header (which lists every
    tensor's name, dtype, shape, and offset) for ``*.safetensors``. That is kilobytes
    instead of gigabytes per rank per run. The mtime guards against a checkpoint that is
    retrained in place at the same path (same size and header, new weights) silently
    reusing stale log-probs; the tradeoff is that a *copy* of the directory misses the
    cache, because copying sets new mtimes.
    """
    files: list[Path] = []
    for name in ("config.json", "model.safetensors.index.json"):
        if (root / name).is_file():
            files.append(root / name)
    for pattern in _WEIGHT_FILE_PATTERNS:
        files.extend(sorted(root.glob(pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No config.json, shard index, or weight files found in '{root}'.")
    for path in files:
        st = path.stat()
        h.update(path.relative_to(root).as_posix().encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
        if path.suffix == ".safetensors":
            h.update(_safetensors_header(path))
        elif path.name in ("config.json", "model.safetensors.index.json"):
            h.update(path.read_bytes())


def checkpoint_cache_key(
    name_or_path: str,
    revision: str | None,
    dtype: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Compute a cheap, DeepSpeed-stage-independent cache key for a model checkpoint.

    TRL keys the precomputed reference-logprob cache (``*.npz`` next to the dataset cache)
    by ``hash_module(model)``, which hashes the in-memory ``state_dict()``. Under DeepSpeed
    ZeRO-3 the parameters are already partitioned when ``DPOTrainer.__init__`` computes it,
    so the same checkpoint hashes differently per ZeRO stage and per world size, and the
    cache is silently missed. This key instead identifies the checkpoint on disk:

    - Hub repo id: the resolved commit SHA of the snapshot (content-pinned by the Hub) —
      the same key the equivalent local snapshot dir gets, so prefetched and
      non-prefetched runs of the same checkpoint share the cache.
    - Local HF-cache snapshot directory (``.../snapshots/<sha>``, what prefetch substitutes
      into the config): the commit SHA from the path.
    - Any other local directory: ``config.json`` / shard-index bytes plus, per file, the
      name, size, mtime, and (for safetensors) the header. The mtime keeps a checkpoint
      retrained in place at the same path from reusing stale log-probs.

    *dtype* and *extra* are folded in as well — *extra* carries tokenization parameters that
    TRL applies at forward time (after the dataset fingerprint is fixed), notably
    ``max_length`` and ``truncation_mode``, which change the precomputed log-probs but not
    the dataset fingerprint.
    """
    h = hashlib.sha256()
    h.update(_CACHE_KEY_VERSION)
    h.update(dtype.encode())
    h.update(json.dumps(extra or {}, sort_keys=True, default=str).encode())
    path = Path(name_or_path)
    if path.is_dir():
        sha = _cache_snapshot_sha(path)
        if sha is not None:
            h.update(b"snapshot-sha\x00")
            h.update(sha.encode())
        else:
            h.update(b"local-dir")
            _update_local_dir(h, path)
    else:
        # Resolve to the commit SHA and use the same material as a local snapshot dir, so
        # `revision: null` (default branch), an explicit SHA of the same commit, and the
        # prefetched local path of the same snapshot all share one cache key.
        commit = _snapshot_commit(name_or_path, revision)
        h.update(b"snapshot-sha\x00")
        h.update(commit.encode())
    return h.hexdigest()


def install_stage_agnostic_hash_module(
    trainer_module: Any,
    name_or_path: str,
    revision: str | None,
    dtype: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Point ``trainer_module.hash_module`` at a checkpoint cache key and return that key.

    ``trainer_module`` is ``trl.trainer.dpo_trainer`` in production; ``DPOTrainer`` looks the
    name up as a module global when it builds the ``.npz`` cache file name, so replacing it
    there redirects the cache key without touching TRL's read/write logic.
    """
    model_hash = checkpoint_cache_key(name_or_path, revision, dtype, extra)

    def hash_module(module: Any) -> str:
        return model_hash

    trainer_module.hash_module = hash_module
    return model_hash


def _ref_source_and_revision(config: PostTrainingConfig) -> tuple[str, str | None]:
    """Return the ``(name_or_path, revision)`` whose cache key represents the reference model.

    TRL hashes ``self.ref_model or self.model``; with ``ref_model=None`` the implicit
    reference is built from the policy model's ``_name_or_path`` with ``model_init_kwargs``
    (including ``model.revision``), so that pair is used. An explicit
    ``dpo.ref_model_name_or_path`` may point at a different repo, so it gets its own
    ``dpo.ref_model_revision`` instead of inheriting ``model.revision``.
    """
    mc = config.dpo
    if mc.ref_model_name_or_path is not None:
        return mc.ref_model_name_or_path, mc.ref_model_revision
    return config.model.name_or_path, config.model.revision


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
        precompute_ref_batch_size=mc.precompute_ref_batch_size,
        model_init_kwargs=build_model_init_kwargs(config),
    )

    if mc.precompute_ref_log_probs:
        import trl.trainer.dpo_trainer as dpo_trainer_module

        ref_source, ref_revision = _ref_source_and_revision(config)
        model_hash = install_stage_agnostic_hash_module(
            dpo_trainer_module,
            ref_source,
            ref_revision,
            config.model.dtype,
            extra={
                # Applied by TRL at forward time, after the dataset fingerprint is fixed:
                # they change the precomputed log-probs, so they must change the cache key.
                "max_length": dpo_config.max_length,
                "truncation_mode": dpo_config.truncation_mode,
            },
        )
        logger.info(
            "precompute_ref_log_probs=true: reference log-prob cache key '%s' identifies the "
            "checkpoint '%s' (revision=%s) on disk and is shared across DeepSpeed ZeRO stages.",
            model_hash,
            ref_source,
            ref_revision or "main",
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
