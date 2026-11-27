#!/usr/bin/env python3
"""Prepare a run directory for an external orchestrator. Submit nothing.

Usage
-----
python scripts/prepare_run.py --config configs/trl/sft.yaml run_name=my-run \
    [--num-nodes 8 --gpus-per-node 4] [--check] [key=value ...]

An external orchestrator (for example an oellm-autoexp backend) calls this
command each time it enqueues a run. The command asks no questions and can be
called again for the same run:

- A new run (no ``<run_dir>/config.yaml``): the full submission phase of
  ``scripts/submit.py --prepare-only``. It freezes the config and the code and
  writes ``<run_dir>/slurm/launch.sh``.
- A continued run (``<run_dir>/config.yaml`` exists): the frozen config and the
  frozen code stay as they are. Only ``launch.sh`` and ``launch.env`` are
  written again. A config that is different from the frozen one is an error.

``run_name`` is mandatory (in the YAML file or as an override). A generated
name has a time stamp, so each call would make a new run directory.

``--num-nodes`` and ``--gpus-per-node`` are for an orchestrator that owns these
two values. A different value in the YAML file is an error, and so is an
override of the same key.

``--check`` does all checks and writes nothing.

The command sets ``umask 002`` for its own process. The job can run as a
different Unix user (the owner of the allocation), which must be able to write
to the run directory.

The last line on stdout is the result, for the caller to parse::

    PREPARE_RUN_RESULT {"run_dir": "...", "launch_script": "...", "new_run": true, ...}

The exit code is 0 only if the run directory is ready (or, with ``--check``,
if all checks passed).

Run this script from the root of the repository, as ``scripts/submit.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
# The sibling scripts are not a package.
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import render_launch_script
import submit

from post_training.config import PostTrainingConfig
from post_training.utils.logging import setup_logging

logger = logging.getLogger(__name__)

RESULT_PREFIX = "PREPARE_RUN_RESULT "

# The orchestrator arguments and the config keys that they own.
_OWNED_KEYS = {"num_nodes": "slurm.num_nodes", "gpus_per_node": "slurm.gpus_per_node"}

# submit.py replaces a Hub repo id with its local snapshot directory before it
# freezes the config. For these keys a repo id is compared with that directory.
_PREFETCHED_KEYS = (
    "model.name_or_path",
    "model.tokenizer_name_or_path",
    "dpo.ref_model_name_or_path",
)


def _owned_overrides(config_path: str, overrides: list[str], args) -> list[str]:
    """Return the overrides for the values that the orchestrator owns, after the checks."""
    file_cfg = OmegaConf.load(config_path)
    override_keys = {item.split("=", 1)[0].lstrip("+") for item in overrides}

    owned = []
    for arg_name, key in _OWNED_KEYS.items():
        value = getattr(args, arg_name)
        if value is None:
            continue
        flag = "--" + arg_name.replace("_", "-")
        if key in override_keys:
            raise ValueError(f"{key} is set by {flag}. Remove the override of {key}.")
        in_file = OmegaConf.select(file_cfg, key)
        if in_file is not None and in_file != value:
            raise ValueError(
                f"{key} is {in_file} in {config_path}, but {flag} is {value}. "
                "Make the two values equal, or remove the key from the file."
            )
        owned.append(f"{key}={value}")
    return owned


def _run_dir_of(config: PostTrainingConfig) -> Path:
    """The run directory that setup_run_directory() makes for *config*."""
    base = config.paths.debug_base if config.debug.enabled else config.paths.output_base
    return Path(base).resolve() / config.run_name


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        flat: dict[str, object] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def _is_snapshot_of(frozen_value: object, repo_id: object) -> bool:
    """True if *frozen_value* is the local snapshot directory of the Hub repo *repo_id*.

    The Hub cache keeps the model ``org/name`` in ``models--org--name/snapshots/<hash>``.
    """
    if not isinstance(frozen_value, str) or not isinstance(repo_id, str):
        return False
    if Path(repo_id).exists():
        return False  # a local path is frozen unchanged, so it must be equal
    return "models--" + repo_id.replace("/", "--") in Path(frozen_value).parts


def _differences(frozen: PostTrainingConfig, config: PostTrainingConfig) -> list[str]:
    """Return one line for each value of *config* that the frozen config does not have."""
    old = _flatten(dataclasses.asdict(frozen))
    new = _flatten(dataclasses.asdict(config))

    # submit.py makes the base directory absolute before it freezes the config.
    base_key = "paths.debug_base" if config.debug.enabled else "paths.output_base"
    new[base_key] = str(Path(str(new[base_key])).resolve())

    lines = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) == new.get(key):
            continue
        if key in _PREFETCHED_KEYS and _is_snapshot_of(old.get(key), new.get(key)):
            continue
        lines.append(f"  {key}: frozen {old.get(key)!r}, now {new.get(key)!r}")
    return lines


def _check_continued_run(config: PostTrainingConfig, run_dir: Path) -> None:
    frozen_path = run_dir / "config.yaml"
    frozen = PostTrainingConfig.load(str(frozen_path))
    lines = _differences(frozen, config)
    if lines:
        raise ValueError(
            f"The config is different from the frozen config {frozen_path}:\n"
            + "\n".join(lines)
            + "\nA continued run uses the frozen config. Make the values equal, "
            "use the frozen config as --config, or use a new run_name."
        )


def _prepare_new_run(config_path: str, overrides: list[str], run_dir: Path) -> None:
    """Do the submission phase of submit.py --prepare-only, with no question."""
    argv = sys.argv
    sys.argv = ["submit.py", "--config", config_path, "--prepare-only", "--confirm", *overrides]
    try:
        submit.main()
    finally:
        sys.argv = argv
    for made in (run_dir / "config.yaml", run_dir / "slurm" / "launch.sh"):
        if not made.is_file():
            raise RuntimeError(f"The submission phase did not write {made}.")


def prepare(args, overrides: list[str]) -> dict[str, object]:
    """Check the config, prepare the run directory, and return the result."""
    overrides = [*overrides, *_owned_overrides(args.config, overrides, args)]
    config = PostTrainingConfig.load(args.config, overrides)

    if not config.run_name:
        raise ValueError(
            "run_name is mandatory. Set it in the config file or as the override run_name=<name>."
        )
    if config.backend != "trl" or config.container is None or not config.container.image:
        raise ValueError("A launch script is only supported with backend=trl and container.image.")
    if config.debug.enabled and config.debug.override_existing:
        raise ValueError(
            "debug.override_existing deletes a run directory that exists. "
            "Set it to false for a run that an orchestrator can start again."
        )
    # The same check as in submit.py, before any write.
    config.resolve_gradient_accumulation_steps(config.slurm.num_nodes * config.slurm.gpus_per_node)

    run_dir = _run_dir_of(config)
    new_run = not (run_dir / "config.yaml").is_file()

    if not new_run:
        _check_continued_run(config, run_dir)

    if not args.check:
        if new_run:
            _prepare_new_run(args.config, overrides, run_dir)
        else:
            render_launch_script.render_for_run_dir(run_dir)

    return {
        "run_dir": str(run_dir),
        "launch_script": str(run_dir / "slurm" / "launch.sh"),
        "config": str(run_dir / "config.yaml"),
        "new_run": new_run,
        "checked_only": args.check,
        "num_nodes": config.slurm.num_nodes,
        "gpus_per_node": config.slurm.gpus_per_node,
    }


def main() -> None:
    setup_logging()
    # The orchestrator can run the job as a different Unix user (the owner of the
    # allocation). Keep all that this process creates writable for the group, as
    # launch.sh does for the files of the job.
    os.umask(0o002)

    parser = argparse.ArgumentParser(
        description="Prepare a run directory for an external orchestrator. Submit nothing."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--num-nodes", type=int, default=None, help="Sets slurm.num_nodes.")
    parser.add_argument("--gpus-per-node", type=int, default=None, help="Sets slurm.gpus_per_node.")
    parser.add_argument("--check", action="store_true", help="Do all checks, write nothing.")
    args, overrides = parser.parse_known_args()

    try:
        result = prepare(args, overrides)
    except (ValueError, FileNotFoundError) as error:
        print(f"prepare_run.py: error: {error}", file=sys.stderr)
        sys.exit(1)

    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
