#!/usr/bin/env python3
"""Submit a training job to SLURM.

Usage
-----
python scripts/submit.py --config configs/sft.yaml

Any extra arguments are forwarded as OmegaConf dot-list overrides, just
like ``scripts/train.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.config import PostTrainingConfig
from post_training.slurm.launcher import generate_and_submit
from post_training.utils.guardrails import run_guardrails
from post_training.utils.logging import setup_logging
from post_training.utils.paths import setup_run_directory
from post_training.utils.prefetch import prefetch_assets

logger = logging.getLogger(__name__)

_HF_CACHE_VARS = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
    }
)


def _apply_hf_env_from_file(env_file: str) -> None:
    """Parse HF cache vars from a shell env file and apply them to os.environ.

    Ensures prefetch_assets() downloads to the same cache root that the
    container will use (set via container.env_file sourced in the SLURM script).
    """
    path = Path(env_file)
    if not path.exists():
        logger.warning("container.env_file '%s' not found, skipping.", env_file)
        return

    parsed: dict[str, str] = {}
    export_re = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    with path.open() as f:
        for line in f:
            m = export_re.match(line.strip())
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip("\"'")
            value = re.sub(
                r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?",
                lambda mv: parsed.get(mv.group(1), os.environ.get(mv.group(1), "")),
                value,
            )
            parsed[key] = value

    applied = []
    for key in _HF_CACHE_VARS:
        if key in parsed:
            os.environ[key] = parsed[key]
            applied.append(f"{key}={parsed[key]}")
    if applied:
        logger.info("Applied HF cache vars from %s: %s", env_file, ", ".join(applied))


def _parse_args() -> tuple[str, list[str], bool]:
    parser = argparse.ArgumentParser(description="Submit a SLURM training job.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trl/sft.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive guardrails review and submit immediately.",
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Pass --tokenize-only to train.py — exits after trainer initialization.",
    )
    known, unknown = parser.parse_known_args()
    return known.config, unknown, known.confirm, known.tokenize_only


def main() -> None:
    setup_logging()

    config_path, cli_overrides, confirmed, tokenize_only = _parse_args()
    logger.info("Loading config from %s", config_path)
    config = PostTrainingConfig.load(config_path, cli_overrides)

    if tokenize_only:
        config.slurm.num_nodes = 1
        config.slurm.gpus_per_node = 1

    if config.offline:
        if config.container.env_file:
            _apply_hf_env_from_file(config.container.env_file)
        logger.info(
            "offline=True: pre-fetching models and datasets on the login node "
            "before submitting the job."
        )
        prefetch_assets(config)

    # Set up the run directory (so the SLURM script can reference it).
    run_dir = setup_run_directory(config)
    logger.info("Run directory: %s", run_dir)

    if not confirmed:
        run_guardrails(config, run_dir, tokenize_only=tokenize_only)

    # CRITICAL: Set run_name so it's preserved in the frozen config.
    # This ensures train.py uses the same directory when it loads the config.
    config.run_name = run_dir.name

    # Freeze a copy of the config.
    frozen = run_dir / "config.yaml"
    config.save(frozen)

    # Copy any backend-specific artifacts for reproducibility.
    from post_training.backend import get_backend

    get_backend(config.backend).post_freeze(config, run_dir)

    job_id = generate_and_submit(config, run_dir, str(frozen))
    logger.info("SLURM job submitted: %s", job_id)


if __name__ == "__main__":
    main()
