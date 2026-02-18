#!/usr/bin/env python3
"""Inspect intermediate data values for debugging.

Each output stage is togglable via a CLI flag so you can focus on exactly
the information you need.

Usage
-----
python scripts/inspect_data.py --config configs/sft.yaml \\
    --show-raw --show-transformed --show-formatted --show-tokens \\
    --num-samples 3

Any extra arguments are forwarded as OmegaConf dot-list overrides::

    python scripts/inspect_data.py --config configs/sft.yaml \\
        --show-stats training.max_steps=5 offline=true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.config import PostTrainingConfig
from post_training.utils.logging import setup_logging

# ── Helpers ──────────────────────────────────────────────────────────────────

# Columns to check when extracting a message list from a row (in priority order).
_MESSAGE_COLUMNS = ("messages", "chosen", "conversations", "conversation", "dialog")


def _extract_messages(row: dict) -> list | None:
    """Return the first non-empty message list found in *row*, or ``None``."""
    for col in _MESSAGE_COLUMNS:
        msgs = row.get(col)
        if msgs and isinstance(msgs, list):
            return msgs
    return None


# ── Pretty printing helpers ─────────────────────────────────────────────────

_SEPARATOR = "─" * 72


def _print_header(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def _print_sample(idx: int, data: Any) -> None:
    print(f"\n  [Sample {idx}]")
    if isinstance(data, dict):
        for k, v in data.items():
            val_repr = repr(v) if not isinstance(v, str) else v
            # Truncate very long values.
            if len(str(val_repr)) > 500:
                val_repr = str(val_repr)[:500] + " …"
            print(f"    {k}: {val_repr}")
    else:
        print(f"    {data}")


# ── Argument parsing ────────────────────────────────────────────────────────


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse known args and collect remaining args as OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Inspect data pipeline outputs.",
        # Allow unknown args → they become OmegaConf dot-list overrides.
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument(
        "--num-samples", type=int, default=3, help="Number of samples to inspect."
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show raw dataset rows before any processing.",
    )
    parser.add_argument(
        "--show-transformed",
        action="store_true",
        help="Show rows after per-dataset transform.",
    )
    parser.add_argument(
        "--show-formatted",
        action="store_true",
        help="Show text after applying the chat template.",
    )
    parser.add_argument(
        "--show-tokens", action="store_true", help="Show token IDs and decoded tokens."
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Show dataset-level statistics."
    )

    args, unknown = parser.parse_known_args()

    # If nothing is toggled on, turn everything on by default.
    any_toggled = (
        args.show_raw
        or args.show_transformed
        or args.show_formatted
        or args.show_tokens
        or args.show_stats
    )
    if not any_toggled:
        args.show_raw = True
        args.show_transformed = True
        args.show_formatted = True
        args.show_tokens = True
        args.show_stats = True

    return args, unknown


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    setup_logging()

    args, cli_overrides = _parse_args()

    config = PostTrainingConfig.load(args.config, cli_overrides)

    # ── Lazy imports (heavy) ────────────────────────────────────────
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from post_training.chat_templates.registry import get_chat_template
    from post_training.data.transforms import get_transform

    n = args.num_samples

    for entry in config.data.datasets:
        print(f"\n{'═' * 72}")
        print(f"  Dataset: {entry.name}  ({entry.path}, split={entry.split})")
        print(f"{'═' * 72}")

        # Build kwargs for load_dataset.
        load_kwargs: dict = {}
        if getattr(entry, "subset", None) is not None:
            load_kwargs["name"] = entry.subset
        if getattr(entry, "data_dir", None) is not None:
            load_kwargs["data_dir"] = entry.data_dir

        ds = load_dataset(entry.path, split=entry.split, **load_kwargs)

        # ── Raw ─────────────────────────────────────────────────────
        if args.show_raw:
            _print_header("RAW (before transform)")
            for i in range(min(n, len(ds))):
                _print_sample(i, ds[i])

        # ── Transformed ─────────────────────────────────────────────
        if args.show_transformed and entry.transform is not None:
            _print_header(f"TRANSFORMED (via '{entry.transform}')")
            transform_fn = get_transform(entry.transform)
            for i in range(min(n, len(ds))):
                _print_sample(i, transform_fn(ds[i]))
        elif args.show_transformed and entry.transform is None:
            _print_header("TRANSFORMED (no transform — using raw)")
            for i in range(min(n, len(ds))):
                _print_sample(i, ds[i])

        # ── Formatted via chat template ─────────────────────────────
        if args.show_formatted:
            _print_header(f"FORMATTED (chat_template='{config.data.chat_template}')")
            tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
            template_str = get_chat_template(config.data.chat_template)
            tokenizer.chat_template = template_str

            for i in range(min(n, len(ds))):
                row = ds[i]
                if entry.transform is not None:
                    row = get_transform(entry.transform)(row)
                messages = _extract_messages(row)
                if messages is None:
                    print(
                        f"\n  [Sample {i}] <no message column found in: {list(row.keys())}>"
                    )
                    continue
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception as exc:
                    formatted = f"<ERROR applying template: {exc}>"
                print(f"\n  [Sample {i}]\n{formatted}")

        # ── Tokens ──────────────────────────────────────────────────
        if args.show_tokens:
            _print_header("TOKENS")
            tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
            template_str = get_chat_template(config.data.chat_template)
            tokenizer.chat_template = template_str

            for i in range(min(n, len(ds))):
                row = ds[i]
                if entry.transform is not None:
                    row = get_transform(entry.transform)(row)
                messages = _extract_messages(row)
                if messages is None:
                    print(f"\n  [Sample {i}] <no message column found>")
                    continue
                try:
                    token_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )
                except Exception as exc:
                    print(f"\n  [Sample {i}] <ERROR: {exc}>")
                    continue
                decoded = [tokenizer.decode([tid]) for tid in token_ids]
                print(f"\n  [Sample {i}]  length={len(token_ids)}")
                # Show first 40 tokens.
                preview = list(zip(token_ids[:40], decoded[:40]))
                for tid, tok in preview:
                    print(f"    {tid:>8}  {repr(tok)}")
                if len(token_ids) > 40:
                    print(f"    ... ({len(token_ids) - 40} more tokens)")

        # ── Stats ───────────────────────────────────────────────────
        if args.show_stats:
            _print_header("DATASET STATISTICS")
            print(f"  Total rows : {len(ds)}")
            print(f"  Columns    : {ds.column_names}")

            tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
            template_str = get_chat_template(config.data.chat_template)
            tokenizer.chat_template = template_str

            # Sample up to 200 rows for token-length stats.
            sample_size = min(200, len(ds))
            lengths: list[int] = []
            for i in range(sample_size):
                row = ds[i]
                if entry.transform is not None:
                    row = get_transform(entry.transform)(row)
                messages = _extract_messages(row)
                if messages is None:
                    continue
                try:
                    tids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )
                    lengths.append(len(tids))
                except Exception:
                    pass

            if lengths:
                avg = sum(lengths) / len(lengths)
                print(f"  Token length (sampled {sample_size} rows):")
                print(f"    min={min(lengths)}, max={max(lengths)}, mean={avg:.1f}")


if __name__ == "__main__":
    main()
