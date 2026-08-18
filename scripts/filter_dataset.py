#!/usr/bin/env python3
"""Drop rows whose assistant loss mask is all zero, then write sharded parquet.

A row is kept only if ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``
marks at least one token for the loss.  Rows with an all-zero mask contribute no
gradient during SFT, so they only waste tokens in a packed batch.

The number of output shards is inferred from the estimated size of the kept rows
and ``--max-shard-size``.

Usage
-----
    python scripts/filter_dataset.py \\
        --dataset allenai/Dolci-Instruct-SFT \\
        --tokenizer Qwen/Qwen3-8B \\
        --chat-template chatml \\
        --output data/dolci-instruct-sft-nonempty

    # Report only, write nothing.
    python scripts/filter_dataset.py --dataset ... --tokenizer ... --dry-run

The output directory holds ``train-00000-of-000NN.parquet`` files that
``load_dataset("<output>", split="train")`` reads directly.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.chat_templates.registry import (
    CHAT_TEMPLATES,
    get_chat_template,
    has_generation_markers,
)

_SIZE_UNITS = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}
_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")


def parse_size(value: str) -> int:
    """Return *value* (for example ``"500MB"``) as a number of bytes."""
    match = _SIZE_RE.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(f"Cannot parse size '{value}'.")
    number, unit = match.groups()
    unit = (unit or "B").upper()
    if unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(
            f"Unknown size unit '{unit}'. Use one of: {', '.join(_SIZE_UNITS)}."
        )
    size = int(float(number) * _SIZE_UNITS[unit])
    if size <= 0:
        raise argparse.ArgumentTypeError(f"Size must be positive, got '{value}'.")
    return size


def resolve_chat_template(spec: str | None) -> str | None:
    """Return the Jinja source for *spec*.

    *spec* is either a registry name (see ``CHAT_TEMPLATES``) or a path to a
    ``.jinja`` file.  ``None`` keeps the template that comes with the tokenizer.
    """
    if spec is None:
        return None
    path = Path(spec)
    if path.suffix == ".jinja" or path.exists():
        if not path.exists():
            raise FileNotFoundError(f"Chat template file not found: {path}")
        return path.read_text()
    return get_chat_template(spec)


# ``datasets`` >= 4 loads a directory or a repo id, but not a single data file.
# For a file the builder name comes from the suffix and the file goes to data_files.
_FILE_BUILDERS = {
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".arrow": "arrow",
    ".txt": "text",
}


def load_source(dataset: str, dataset_config: str | None, split: str, num_proc: int):
    """Load *dataset* from the Hub, a local directory, or a single data file."""
    path = Path(dataset)
    if path.is_file():
        builder = _FILE_BUILDERS.get(path.suffix)
        if builder is None:
            raise SystemExit(
                f"Cannot load the file '{path}'. Known suffixes: "
                f"{', '.join(sorted(_FILE_BUILDERS))}."
            )
        return load_dataset(
            builder,
            data_files={split: str(path)},
            split=split,
            num_proc=num_proc,
        )
    return load_dataset(dataset, dataset_config, split=split, num_proc=num_proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter out dataset rows with an all-zero assistant loss mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF repo id, local directory, or single data file (parquet/json/csv/arrow).",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset config/subset name.",
    )
    parser.add_argument("--split", default="train", help="Split to load.")
    parser.add_argument(
        "--messages-column",
        default="messages",
        help="Column that holds the list of chat messages.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF repo id or local path of the tokenizer.",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help=(
            "Chat template: a registry name "
            f"({', '.join(sorted(CHAT_TEMPLATES))}) or a path to a .jinja file. "
            "Omit it to keep the template of the tokenizer."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for the parquet shards. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=parse_size,
        default="500MB",
        help="Target size of one parquet shard. The shard count follows from it.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Fixed shard count. It overrides --max-shard-size.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        help="Worker processes for load_dataset() and map().",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of dropped rows to print.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the statistics only and write no files.",
    )
    args = parser.parse_args()
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run is set.")
    if args.num_shards is not None and args.num_shards < 1:
        parser.error("--num-shards must be at least 1.")
    return args


def infer_num_shards(dataset, max_shard_size: int) -> int:
    """Return the shard count that keeps each shard near *max_shard_size* bytes."""
    num_rows = len(dataset)
    if num_rows == 0:
        return 1
    nbytes = dataset.data.nbytes
    num_shards = math.ceil(nbytes / max_shard_size)
    return max(1, min(num_shards, num_rows))


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = resolve_chat_template(args.chat_template)
    if template is not None:
        tokenizer.chat_template = template
    if not has_generation_markers(tokenizer.chat_template):
        raise SystemExit(
            "The chat template has no {% generation %} markers, so every mask is "
            "all zero and the filter drops every row. Pass a template that has them."
        )

    dataset = load_source(args.dataset, args.dataset_config, args.split, args.num_proc)
    if args.messages_column not in dataset.column_names:
        raise SystemExit(
            f"Column '{args.messages_column}' is not in the dataset. "
            f"Available columns: {', '.join(dataset.column_names)}."
        )
    print(f"Loaded {len(dataset):,} rows from '{args.dataset}' (split={args.split}).")

    # remove_columns keeps the map cache small: it holds the flag alone, not a
    # copy of the data.
    in_loss = dataset.map(
        lambda row: {
            "in_loss": any(
                tokenizer.apply_chat_template(
                    row[args.messages_column],
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )["assistant_masks"]
            )
        },
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="masking",
    )["in_loss"]
    in_loss = list(in_loss)

    empty = [i for i, keep in enumerate(in_loss) if not keep]
    share = 100 * len(empty) / len(dataset) if len(dataset) else 0.0
    print(f"{len(dataset):,} rows, {len(empty):,} with an all-zero mask ({share:.3f}%)")
    for i in empty[: args.num_examples]:
        roles = " -> ".join(m["role"] for m in dataset[i][args.messages_column])
        print(f"  row {i}: {roles}")

    kept = dataset.select([i for i, keep in enumerate(in_loss) if keep])
    if args.dry_run:
        print(f"Dry run: {len(kept):,} of {len(dataset):,} rows pass the filter.")
        return

    num_shards = (
        args.num_shards
        if args.num_shards is not None
        else infer_num_shards(kept, args.max_shard_size)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(num_shards):
        path = args.output / f"{args.split}-{i:05d}-of-{num_shards:05d}.parquet"
        kept.shard(num_shards=num_shards, index=i, contiguous=True).to_parquet(path)
        print(f"wrote {path}")

    print(f"kept {len(kept):,} of {len(dataset):,} rows in {args.output} ({num_shards} shards)")


if __name__ == "__main__":
    main()
