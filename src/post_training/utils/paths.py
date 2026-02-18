"""Run directory creation and naming conventions.

The auto-generated run name encodes the **method**, **model**, and
**data configuration** so that experiments are easy to identify at a
glance::

    <method>-<model_short>-<dataset_short>-<timestamp>      # single dataset
    <method>-<model_short>-mix_<8-char-hash>-<timestamp>     # mixture
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig


def _shorten_model_name(name_or_path: str) -> str:
    """Derive a short, filesystem-safe identifier from a model name.

    Example
    -------
    >>> _shorten_model_name("swiss-ai/Apertus-8B-2509")
    'apertus-8b-2509'
    """
    # Take only the last segment of an org/model path.
    short = name_or_path.rsplit("/", 1)[-1]
    # Lowercase, keep alphanumerics and hyphens.
    short = re.sub(r"[^a-z0-9\-]", "", short.lower())
    return short


def _shorten_dataset_name(name: str) -> str:
    """Make a dataset name filesystem-safe."""
    return re.sub(r"[^a-z0-9_\-]", "_", name.lower())


def _dataset_mix_hash(datasets: list) -> str:
    """Compute an 8-character hash of the dataset mixture spec."""
    # Build a canonical, sorted representation.
    canonical = sorted(
        [{"name": d.name, "path": d.path, "weight": d.weight} for d in datasets],
        key=lambda x: x["name"],
    )
    digest = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
    return digest[:8]


def generate_run_name(config: PostTrainingConfig) -> str:
    """Create a human-readable run name from the config.

    If ``config.run_name`` is already set, return it unchanged.
    """
    if config.run_name:
        return config.run_name

    model_short = _shorten_model_name(config.model.name_or_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    datasets = config.data.datasets
    if len(datasets) == 1:
        ds_part = _shorten_dataset_name(datasets[0].name)
    else:
        ds_part = f"mix_{_dataset_mix_hash(datasets)}"

    return f"{config.method}-{model_short}-{ds_part}-{timestamp}"


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

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "inference_checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "slurm").mkdir(exist_ok=True)

    return run_dir
