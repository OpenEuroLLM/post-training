#!/usr/bin/env python3
"""Render the launch script of a run directory that already exists.

Usage
-----
python scripts/render_launch_script.py --run-dir outputs/<run_name>

The script reads the frozen ``<run_dir>/config.yaml`` and writes
``<run_dir>/slurm/launch.sh`` and ``<run_dir>/slurm/launch.env``. It does not
change the config, the frozen code or the checkpoints, and it does not submit
a job. An external orchestrator (for example an oellm-autoexp pool) runs the
launch script with one srun task for each node::

    srun --nodes=N --ntasks=N --ntasks-per-node=1 bash <run_dir>/slurm/launch.sh

Run this script from the root of the repository. A relative
``container.env_file`` resolves against the root.

For a new run, use ``python scripts/submit.py --prepare-only`` instead.
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
from post_training.slurm.launcher import render_trl_launch_script
from post_training.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def render_for_run_dir(run_dir: Path, *, tokenize_only: bool = False) -> Path:
    """Render ``launch.sh`` for *run_dir* from its frozen config and return its path."""
    run_dir = run_dir.resolve()
    frozen = run_dir / "config.yaml"
    if not frozen.is_file():
        raise FileNotFoundError(f"{frozen} not found. {run_dir} is not a prepared run directory.")

    config = PostTrainingConfig.load(str(frozen))
    if config.backend != "trl":
        raise ValueError(
            f"Launch scripts are only supported with backend=trl, got '{config.backend}'."
        )

    # The launch script runs the frozen code of the run directory.
    for required in (run_dir / "src" / "post_training", run_dir / "scripts" / "train.py"):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} not found. The run directory has no frozen code; "
                "prepare it with `python scripts/submit.py --prepare-only`."
            )

    # train.py derives the run directory from the config. If the two disagree (for
    # example after the directory moved), the job writes to a different directory.
    base = config.paths.debug_base if config.debug.enabled else config.paths.output_base
    expected = (Path(base) / (config.run_name or "")).resolve()
    if not config.run_name or expected != run_dir:
        raise ValueError(
            f"The frozen config points to the run directory {expected}, not to {run_dir}."
        )

    # Same as submit.py: the tokenization run uses one GPU.
    if tokenize_only:
        config.slurm.num_nodes = 1
        config.slurm.gpus_per_node = 1

    return render_trl_launch_script(config, run_dir, str(frozen), tokenize_only=tokenize_only)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Render the launch script of a run directory that already exists."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the run directory.")
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Pass --tokenize-only to train.py — exits after trainer initialization.",
    )
    args = parser.parse_args()

    launch = render_for_run_dir(args.run_dir, tokenize_only=args.tokenize_only)
    logger.info("No job was submitted. Run the launch script with one srun task for each node:")
    logger.info("    bash %s", launch)


if __name__ == "__main__":
    main()
