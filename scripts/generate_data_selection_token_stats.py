#!/usr/bin/env python3
"""Generate token stats for data-selection configs.

This is a development helper for producing a YAML file shaped like
``data-selection-datasets-stats.yaml``.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.config import PostTrainingConfig
from post_training.methods.common import build_tokenizer
from post_training.utils.logging import setup_logging
from post_training.utils.token_stats import count_token_stats_for_config, token_stats_as_dict

_DEFAULT_CONFIGS = (
    "configs/data_selection/lmsys-chat-1m-decontaminated.yaml",
    "configs/data_selection/orca-agentinstruct-1m-v1-decontaminated.yaml",
    "configs/data_selection/open-perfectblend-decontaminated.yaml",
    "configs/data_selection/smoltalk2-decontaminated.yaml",
    "configs/data_selection/nemotron-post-training-dataset-v2-decontaminated.yaml",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decontaminated data-selection token stats YAML."
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Config to process. Repeat to generate stats for selected configs only.",
    )
    parser.add_argument(
        "--output",
        default="data-selection-datasets-stats-decontaminated.yaml",
        help="Output YAML path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute configured splits even when they already exist in the output YAML.",
    )
    return parser.parse_args()


def _single_entry_config(config: PostTrainingConfig, entry: Any) -> PostTrainingConfig:
    single = copy.deepcopy(config)
    single.data.datasets = [entry]
    return single


def _load_stats_document(output: Path) -> dict:
    """Load an existing stats document, treating a missing/empty file as empty."""
    if not output.exists():
        return {}

    with output.open() as f:
        document = yaml.safe_load(f)

    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ValueError(f"Expected '{output}' to contain a YAML mapping, got {type(document)}.")
    return document


def _save_stats_document(output: Path, document: dict) -> None:
    """Atomically save the stats document."""
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(document, f, sort_keys=False)
    tmp.replace(output)


def _split_exists(document: dict, path: str, subset: str, split: str) -> bool:
    """Return whether a dataset/subset/split entry already exists."""
    return split in document.get(path, {}).get(subset, {})


def _set_split_stats(document: dict, path: str, subset: str, split: str, stats: dict) -> None:
    """Set stats for one dataset/subset/split entry."""
    document.setdefault(path, {}).setdefault(subset, {})[split] = stats


def generate_stats_document(
    config_paths: list[str],
    *,
    output: Path,
    force: bool,
) -> dict:
    """Generate the nested stats document for the requested config files."""
    document = _load_stats_document(output)

    for config_path in config_paths:
        config = PostTrainingConfig.load(config_path)
        tokenizer = build_tokenizer(config)

        for entry in config.data.datasets:
            subset = entry.subset or "default"
            if not force and _split_exists(document, entry.path, subset, entry.split):
                print(
                    "Skipping "
                    f"path={entry.path} subset={subset} split={entry.split}",
                    flush=True,
                )
                continue

            print(
                "Counting "
                f"path={entry.path} subset={subset} split={entry.split}",
                flush=True,
            )
            result = count_token_stats_for_config(
                _single_entry_config(config, entry),
                tokenizer=tokenizer,
            )

            stats = token_stats_as_dict(result.stats)
            _set_split_stats(document, entry.path, subset, entry.split, stats)
            _save_stats_document(output, document)
            print(f"Wrote {output}", flush=True)

    return document


def main() -> None:
    setup_logging()

    args = _parse_args()
    config_paths = args.configs or list(_DEFAULT_CONFIGS)
    output = Path(args.output)

    document = generate_stats_document(config_paths, output=output, force=args.force)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
