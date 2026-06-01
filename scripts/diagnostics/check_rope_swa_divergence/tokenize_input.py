"""Tokenize a Dolci row through our training-time chat template.

Writes a single .npz with the input_ids that both containers will consume,
guaranteeing identical input regardless of which container's transformers
or tokenizer would otherwise be picked up.

Intended to be run inside the post-training container (transformers 5.4.0)
so the chat-template + tokenizer match the SFT pipeline exactly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Make `post_training` importable so we get the exact training chat template.
REPO = Path(__file__).resolve().parents[3]  # .../post-training
sys.path.insert(0, str(REPO / "src"))

from transformers import AutoTokenizer
from post_training.chat_templates.registry import get_chat_template

DOLCI = Path(
    "/leonardo_scratch/large/userexternal/knikolao/propella_annotation/data/Dolci-Instruct-SFT-Decont/data"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="allenai/OLMo-3-7B-Instruct-SFT")
    ap.add_argument("--chat-template", default="olmo3-instruct-sft")
    ap.add_argument("--row-index", type=int, default=0,
                    help="Index into the first Dolci parquet shard.")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Truncate input to this many tokens. Longer = stronger "
                    "RoPE-vs-vanilla divergence (try 2048 or 4096 if 1024 is quiet).")
    ap.add_argument("--output",
                    default=str(Path(__file__).parent / "run" / "input.npz"))
    args = ap.parse_args()

    print(f"Tokenizer: {args.model_id}  chat_template={args.chat_template!r}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.chat_template = get_chat_template(args.chat_template)

    parquet_files = sorted(DOLCI.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet under {DOLCI}")
    row = pq.read_table(parquet_files[0]).to_pylist()[args.row_index]
    print(f"Loaded Dolci row {args.row_index}: {len(row['messages'])} messages, "
          f"roles={[m['role'] for m in row['messages']]}")

    enc = tok.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False,
    )
    if hasattr(enc, "input_ids"):
        ids = enc["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        input_ids = list(ids)
    else:
        input_ids = list(enc)
    print(f"Tokenized to {len(input_ids)} tokens (pre-truncate)")

    if args.max_tokens and len(input_ids) > args.max_tokens:
        input_ids = input_ids[: args.max_tokens]
        print(f"Truncated to {len(input_ids)} tokens")
    elif len(input_ids) < 256:
        print(f"WARNING: only {len(input_ids)} tokens. RoPE divergence grows "
              "with position; very short inputs may show only kernel noise. "
              "Consider --row-index N for a longer conversation.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, input_ids=np.array(input_ids, dtype=np.int64))
    print(f"Wrote {len(input_ids)} input_ids to {out_path}")


if __name__ == "__main__":
    main()