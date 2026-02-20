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
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.config import PostTrainingConfig
from post_training.slurm.launcher import generate_and_submit
from post_training.utils.logging import setup_logging
from post_training.utils.paths import setup_run_directory
from post_training.utils.prefetch import prefetch_assets

logger = logging.getLogger(__name__)


def _parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(description="Submit a SLURM training job.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft.yaml",
        help="Path to the YAML config file.",
    )
    known, unknown = parser.parse_known_args()
    return known.config, unknown


def main() -> None:
    setup_logging()

    config_path, cli_overrides = _parse_args()
    logger.info("Loading config from %s", config_path)
    config = PostTrainingConfig.load(config_path, cli_overrides)

    if config.offline:
        logger.info(
            "offline=True: pre-fetching models and datasets on the login node "
            "before submitting the job."
        )
        prefetch_assets(config)

    # Set up the run directory (so the SLURM script can reference it).
    run_dir = setup_run_directory(config)
    logger.info("Run directory: %s", run_dir)

    # CRITICAL: Set run_name so it's preserved in the frozen config.
    # This ensures train.py uses the same directory when it loads the config.
    config.run_name = run_dir.name

    # Freeze a copy of the config.
    frozen = run_dir / "config.yaml"
    config.save(frozen)

    job_id = generate_and_submit(config, run_dir, str(frozen))
    logger.info("SLURM job submitted: %s", job_id)


if __name__ == "__main__":
    main()
