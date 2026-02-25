#!/usr/bin/env python3
"""Main training entry point (method-agnostic).

Usage
-----
Always launch via ``accelerate launch`` so that DeepSpeed and multi-GPU /
multi-node setup are handled correctly — including single-GPU debug runs.

# SFT training
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --dynamo_backend=inductor \
    --use_deepspeed \
    --same_network \
    --rdzv_backend static \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/trl/sft.yaml \
    training.max_steps=10 \
    offline=true

# DPO training
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --dynamo_backend=inductor \
    --use_deepspeed \
    --same_network \
    --rdzv_backend static \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/trl/dpo.yaml \
    training.max_steps=10 \
    offline=true
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Ensure the project root is on ``sys.path`` so that ``post_training`` is
# importable when running directly (``python scripts/train.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.config import PostTrainingConfig
from post_training.utils.logging import setup_logging
from post_training.utils.paths import setup_run_directory

logger = logging.getLogger(__name__)


def _parse_args() -> tuple[str, list[str]]:
    """Parse ``--config`` and collect remaining args as OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Post-training launcher (SFT, DPO, …)",
        # Allow unknown args → they become OmegaConf dot-list overrides.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trl/sft.yaml",
        help="Path to the YAML config file.",
    )
    known, unknown = parser.parse_known_args()
    return known.config, unknown


def main() -> None:
    setup_logging()

    config_path, cli_overrides = _parse_args()
    logger.info("Loading config from %s", config_path)
    config = PostTrainingConfig.load(config_path, cli_overrides)

    # ── Offline mode ────────────────────────────────────────────────
    if config.offline:
        logger.info(
            "Offline mode ON — disabling all HuggingFace and wandb network calls."
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["WANDB_MODE"] = "offline"

    # Lazy import: must come after offline env vars are set so that
    # huggingface_hub caches HF_HUB_OFFLINE=1 on first import.
    from post_training.methods import build_trainer

    # ── Debug mode overrides ────────────────────────────────────────
    if config.debug.enabled:
        logger.info("Debug mode ON — training for %d steps.", config.training.max_steps)
        config.logging.report_to = "none"

    # ── Set up run directory ────────────────────────────────────────
    run_dir = setup_run_directory(config)
    logger.info("Run directory: %s", run_dir)

    # Freeze a copy of the config into the run directory.
    frozen_path = run_dir / "config.yaml"
    config.save(frozen_path)

    # ── Build trainer & launch ──────────────────────────────────────
    trainer = build_trainer(config, run_dir)

    # Auto-resume from the latest checkpoint if one exists.
    checkpoints_dir = run_dir / "checkpoints"
    existing = sorted(checkpoints_dir.glob("checkpoint-*"))
    resume_from = str(existing[-1]) if existing else None
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    train_result = trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Training complete.")

    # ── Log metrics ─────────────────────────────────────────────────
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
