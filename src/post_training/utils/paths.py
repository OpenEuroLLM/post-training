"""Run directory creation and naming conventions.

The auto-generated run name encodes the **method**, **model**, and
**data configuration** so that experiments are easy to identify at a
glance::

    <method>-<model_short>-<dataset_short>-<timestamp>      # single dataset
    <method>-<model_short>-mix_<8-char-hash>-<timestamp>     # mixture
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig


def generate_run_name(config: PostTrainingConfig) -> str:
    """Create a human-readable run name from the config.

    If ``config.run_name`` is already set, return it unchanged.
    """
    if config.run_name:
        return config.run_name

    from post_training.backend import get_backend

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return get_backend(config.backend).generate_run_name(config, timestamp)


def setup_run_directory(config: PostTrainingConfig) -> Path:
    """Create the run output directory tree and return the run root.

    Directory layout::

        <base>/<run_name>/
            config.yaml
            checkpoints/
            inference_checkpoints/
            logs/
            slurm/

    In debug mode the base switches to ``paths.debug_base`` and an
    existing directory is optionally wiped when
    ``debug.override_existing`` is ``True``.
    """
    run_name = generate_run_name(config)

    if config.debug.enabled:
        base = Path(config.paths.debug_base)
    else:
        base = Path(config.paths.output_base)

    run_dir = base / run_name

    # Handle debug override.
    if config.debug.enabled and config.debug.override_existing and run_dir.exists():
        shutil.rmtree(run_dir)

    from post_training.backend import get_backend

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "slurm").mkdir(exist_ok=True)
    for subdir in get_backend(config.backend).run_dir_subdirs():
        (run_dir / subdir).mkdir(exist_ok=True)

    return run_dir
