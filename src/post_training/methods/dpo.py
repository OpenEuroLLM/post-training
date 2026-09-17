"""Direct preference optimisation (DPO) method."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from accelerate import PartialState
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from trl.trainer import dpo_trainer as trl_dpo_trainer

from post_training.data.loader import load_and_mix_datasets
from post_training.methods.common import (
    align_generation_eos,
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


def build_ref_logps_cache_key(config: PostTrainingConfig, run_name: str) -> str:
    """Return the reference-model half of TRL's reference log-prob cache key.

    ``dpo.ref_logps_cache_key`` is the entire key when set. Otherwise the key
    joins the model identity and the run name with hyphens.
    TRL adds the dataset fingerprint as the other half.

    With ``precompute_ref_log_probs=True`` TRL keeps no separate reference
    model; it treats the policy weights at step 0 as the reference. So the
    policy model identity names the weights the cached log-probs came from.
    """
    if config.dpo.ref_logps_cache_key is not None:
        return config.dpo.ref_logps_cache_key
    parts = [config.model.name_or_path, config.model.revision or "main", run_name]
    # Hyphens replace "/", "@" and the like, so the logged key pastes into YAML as is.
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts)).strip("-")


class StableCacheKeyDPOTrainer(DPOTrainer):
    """A :class:`DPOTrainer` whose reference log-prob cache ignores the ZeRO stage.

    TRL keys that cache on ``hash_module(model)``, which hashes the bytes of
    ``state_dict()``. Under ZeRO-3 the parameters are already sharded when the
    constructor runs, so one checkpoint hashes differently under stage 3 and
    stage 2. A cache that a stage-3 precompute run writes is then invisible to
    the stage-2 training run. This subclass substitutes a stage-independent
    key, so the two runs share one cache file.

    The substitution costs the staleness protection that the weight hash gives.
    Rebuild the cache, or set ``dpo.ref_logps_cache_key``, when the weights
    change but the model path does not.
    """

    def __init__(self, *args, ref_logps_cache_key: str, **kwargs):
        self._ref_logps_cache_key = ref_logps_cache_key
        # TRL precomputes inside the constructor, so the key must exist first.
        super().__init__(*args, **kwargs)

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        # `_fingerprint` is the other half of TRL's cache key. Log both halves so
        # two runs can be compared without reading the arrow cache directory.
        logger.info(
            "Reference log-prob cache key for the %s dataset: "
            "dataset_fingerprint=%s, ref_logps_cache_key=%s",
            name,
            dataset._fingerprint,
            self._ref_logps_cache_key,
        )
        original = trl_dpo_trainer.hash_module
        trl_dpo_trainer.hash_module = lambda _module: self._ref_logps_cache_key
        try:
            return super()._precompute_ref_logps(dataset, name, batch_size)
        finally:
            trl_dpo_trainer.hash_module = original


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

    trainer = StableCacheKeyDPOTrainer(
        model=config.model.name_or_path,
        ref_model=mc.ref_model_name_or_path,  # None → TRL creates implicit copy
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        callbacks=build_callbacks(config, run_dir),
        ref_logps_cache_key=build_ref_logps_cache_key(config, run_dir.name),
    )
    sanitize_generation_config(trainer)
    align_generation_eos(trainer)
    prioritize_metric_callbacks(trainer)
    return trainer
