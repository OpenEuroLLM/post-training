"""
Inspect a pre-tokenized dataset by printing random samples.

Usage:
    python scripts/inspect_tokenized.py /path/to/tokenized/train --tokenizer Qwen/Qwen2.5-0.5B-Instruct
    python scripts/inspect_tokenized.py /path/to/tokenized/train --tokenizer Qwen/Qwen2.5-0.5B-Instruct --num-samples 5
    python scripts/inspect_tokenized.py /path/to/tokenized/train --tokenizer Qwen/Qwen2.5-0.5B-Instruct --indices 0,10,100
"""

import argparse
import random

from datasets import load_from_disk
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Inspect a pre-tokenized dataset")
    parser.add_argument("dataset_path", help="Path to the tokenized dataset directory")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path for decoding")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of random samples to print (default: 3)")
    parser.add_argument("--indices", type=str, help="Comma-separated list of specific indices to print (overrides --num-samples)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to display per sample (default: all)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print()

    # Determine which indices to print
    if args.indices:
        indices = [int(i.strip()) for i in args.indices.split(",")]
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    # Print samples
    for i, idx in enumerate(indices):
        print("=" * 80)
        print(f"Sample {i + 1} (index {idx})")
        print("=" * 80)

        sample = dataset[idx]
        input_ids = sample["input_ids"]

        # Optionally truncate for display
        if args.max_tokens and len(input_ids) > args.max_tokens:
            input_ids_display = input_ids[: args.max_tokens]
            truncated = True
        else:
            input_ids_display = input_ids
            truncated = False

        # Decode and print
        decoded = tokenizer.decode(input_ids_display)
        print(f"Length: {len(sample['input_ids'])} tokens")
        print("-" * 80)
        print(decoded)
        if truncated:
            print(f"\n... [truncated, showing first {args.max_tokens} of {len(sample['input_ids'])} tokens]")
        print()


if __name__ == "__main__":
    main()

