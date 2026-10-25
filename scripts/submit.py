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
from post_training.slurm.launcher import REQUIRED_HF_CACHE_VARS, generate_and_submit
from post_training.utils.guardrails import run_guardrails
from post_training.utils.logging import setup_logging
from post_training.utils.paths import setup_run_directory

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
        raise FileNotFoundError(f"container.env_file '{env_file}' not found.")

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


def _check_required_hf_vars(config: PostTrainingConfig) -> None:
    """Refuse a container job that would stop at start for a missing HF cache variable.

    Call after _apply_hf_env_from_file(): os.environ then holds the submitting
    shell's values, which sbatch passes to the job, plus the env file's exports.
    The job script checks again at start, for differences on the compute node.
    """
    if config.container is None or not config.container.image:
        return
    missing = [var for var in REQUIRED_HF_CACHE_VARS if not os.environ.get(var)]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} not set. Export them in container.env_file "
            f"({config.container.env_file}) as 'export NAME=value' lines."
        )


def _prepare_llamafactory_tokenize_only(config: PostTrainingConfig) -> None:
    """Check llamafactory.tokenized_path for a --tokenize-only job and make it absolute.

    The job saves the tokenized data there. A relative path starts at the directory
    where submit.py runs, so a later training submit with the same value finds the data.
    """
    lf = config.llamafactory
    tokenized_path = lf.get("tokenized_path")
    if not isinstance(tokenized_path, str) or not tokenized_path:
        raise ValueError(
            "--tokenize-only with backend: llamafactory needs llamafactory.tokenized_path: "
            "the job saves the tokenized data there."
        )
    path = Path(tokenized_path).resolve()
    if path.is_dir() and any(path.iterdir()):
        raise ValueError(
            f"llamafactory.tokenized_path '{path}' already holds data, and LlamaFactory would "
            "load it. Delete it, or set a new tokenized_path."
        )
    lf["tokenized_path"] = str(path)


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
        help=(
            "Tokenize the datasets on 1 GPU and exit before training. TRL: passes "
            "--tokenize-only to train.py. LlamaFactory: saves to llamafactory.tokenized_path."
        ),
    )
    known, unknown = parser.parse_known_args()
    return known.config, unknown, known.confirm, known.tokenize_only


def main() -> None:
    setup_logging()

    config_path, cli_overrides, confirmed, tokenize_only = _parse_args()
    logger.info("Loading config from %s", config_path)
    config = PostTrainingConfig.load(config_path, cli_overrides)

    if tokenize_only:
        if config.backend == "llamafactory":
            _prepare_llamafactory_tokenize_only(config)
        config.slurm.num_nodes = 1
        config.slurm.gpus_per_node = 1

    if config.container is not None and config.container.env_file:
        _apply_hf_env_from_file(config.container.env_file)

    _check_required_hf_vars(config)

    prefetched = None
    if config.offline or config.prefetch_assets:
        logger.info(
            "offline=True: pre-fetching models and datasets on the login node "
            "before submitting the job."
        )
        # Lazy import: must come after _apply_hf_env_from_file so that
        # huggingface_hub and datasets read the correct HF_HOME on first import.
        from post_training.utils.prefetch import prefetch_assets

        prefetched = prefetch_assets(config)

    # Set up the run directory (so the SLURM script can reference it).
    run_dir = setup_run_directory(config, allow_override=True)
    logger.info("Run directory: %s", run_dir)

    if not confirmed:
        run_guardrails(config, run_dir, tokenize_only=tokenize_only)

    # CRITICAL: Set run_name so it's preserved in the frozen config.
    # This ensures train.py uses the same directory when it loads the config.
    config.run_name = run_dir.name

    # Substitute resolved local snapshot dirs for the model repo ids, now that
    # run-name generation and the guardrails review (both more readable with
    # the original repo id) are done. train.py then loads a local directory
    # path, so from_pretrained skips HF Hub's cache-resolution/filelock path
    # entirely — avoiding contention when every rank loads it at once.
    # LlamaFactory gets its snapshot paths in post_freeze() instead, so
    # config.yaml keeps the user's llamafactory block and TRL defaults.
    if prefetched is not None and config.backend == "trl":
        config.model.name_or_path = prefetched.model
        config.model.tokenizer_name_or_path = prefetched.tokenizer
        if prefetched.ref_model is not None:
            config.dpo.ref_model_name_or_path = prefetched.ref_model

    # Freeze a copy of the config.
    frozen = run_dir / "config.yaml"
    config.save(frozen)

    # Copy any backend-specific artifacts for reproducibility.
    from post_training.backend import get_backend

    get_backend(config.backend).post_freeze(config, run_dir, prefetched)

    job_id = generate_and_submit(
        config, run_dir, str(frozen.resolve()), tokenize_only=tokenize_only
    )
    logger.info("SLURM job submitted: %s", job_id)


if __name__ == "__main__":
    main()
