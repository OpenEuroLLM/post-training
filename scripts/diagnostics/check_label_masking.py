#!/usr/bin/env python3
"""Verify whether our production SFTConfig trains loss on user/system tokens.

Step 0 in the reproduction-gap investigation. Loads one row of
Dolci-Instruct-SFT-Decont, renders it through our configured chat template, and
shows:

  1. The labels TRL would produce under the production SFTConfig
     (`assistant_only_loss=False`, the current default) — i.e. labels = input_ids.
  2. The labels open-instruct's `sft_tulu_tokenize_and_truncate_v1` would
     produce — assistant content only, everything else masked to -100.
  3. What `apply_chat_template(..., return_assistant_tokens_mask=True)`
     returns for our template — this is the path TRL takes when you set
     `assistant_only_loss=True`. If the mask is all zeros, the template
     lacks `{% generation %}...{% endgeneration %}` markers and the flag
     would be a no-op.

Re-run with `--row N` to spot-check different examples. Heavier compute
(model load, GPU) is not needed — this is tokenizer-only.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from post_training.chat_templates.registry import CHAT_TEMPLATES, get_chat_template

DEFAULT_DATASET = (
    "/leonardo_work/OELLM_prod2026/users/knikolao/propella_annotation"
    "/data/Dolci-Instruct-SFT-Decont/data"
)
DEFAULT_TOKENIZER = "allenai/Olmo-3-7B-Think-SFT"
DEFAULT_TEMPLATE = "olmo3-instruct-sft"


def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def _tok_len(tokenizer, msgs, *, add_generation_prompt: bool, max_seq_length: int) -> int:
    return len(
        tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            return_dict=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=add_generation_prompt,
        )
    )


def classify_tokens_by_role(tokenizer, messages, max_seq_length: int) -> list[str]:
    """Assign a role to every token of apply_chat_template(messages).

    Walks message prefixes to find the token-index boundary of each message.
    Mirrors open-instruct's boundary accounting: when message i+1 is an
    assistant, the assistant prefix tokens that get emitted at the boundary
    of message i (e.g. `<|im_start|>assistant\\n<think>`) are charged to the
    *previous* role, not to "assistant".
    """
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    n_total = len(full)
    role_of: list[str | None] = [None] * n_total

    prev_end = 0
    for i, msg in enumerate(messages):
        role = msg["role"]
        next_is_assistant = (
            i + 1 < len(messages) and messages[i + 1]["role"] == "assistant"
        )
        end = _tok_len(
            tokenizer,
            messages[: i + 1],
            add_generation_prompt=next_is_assistant,
            max_seq_length=max_seq_length,
        )
        for j in range(prev_end, min(end, n_total)):
            role_of[j] = role
        prev_end = end
        if prev_end >= n_total:
            break

    return [r if r is not None else "unknown" for r in role_of]


def open_instruct_style_labels(tokenizer, messages, max_seq_length: int) -> list[int]:
    """Faithful reimplementation of open-instruct's sft_tulu_tokenize_and_truncate_v1.

    See open-instruct/open_instruct/dataset_transformation.py lines 1105-1176.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = list(input_ids)
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            continue
        start = 0 if i == 0 else _tok_len(
            tokenizer, messages[:i],
            add_generation_prompt=False, max_seq_length=max_seq_length,
        )
        next_is_assistant = (
            i + 1 < len(messages) and messages[i + 1]["role"] == "assistant"
        )
        end = _tok_len(
            tokenizer, messages[: i + 1],
            add_generation_prompt=next_is_assistant, max_seq_length=max_seq_length,
        )
        for j in range(start, min(end, len(labels))):
            labels[j] = -100
        if end >= max_seq_length:
            break
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--max-trace", type=int, default=80,
                        help="Number of tokens to print in the detailed trace")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    args = parser.parse_args()

    banner("check_label_masking — production SFTConfig vs open-instruct")
    print(f"tokenizer:        {args.tokenizer}")
    print(f"chat template:    {args.template}  ({CHAT_TEMPLATES[args.template]})")
    print(f"dataset:          {args.dataset}")
    print(f"row index:        {args.row}")
    print(f"max_seq_length:   {args.max_seq_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = get_chat_template(args.template)

    print("\nTRL/transformers versions:")
    try:
        import trl, transformers  # noqa: E401
        print(f"  trl:          {trl.__version__}")
        print(f"  transformers: {transformers.__version__}")
    except Exception as e:
        print(f"  (could not import: {e})")

    ds = load_dataset(args.dataset, split="train")
    row = ds[args.row]
    messages = row["messages"]
    print(f"\nrow turns:        {len(messages)} "
          f"({', '.join(m['role'] for m in messages)})")

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        truncation=True,
        max_length=args.max_seq_length,
        add_generation_prompt=False,
    )
    role_of = classify_tokens_by_role(tokenizer, messages, args.max_seq_length)
    if len(role_of) < len(input_ids):
        role_of = role_of + ["unknown"] * (len(input_ids) - len(role_of))
    role_of = role_of[: len(input_ids)]

    # ── (1) Production labels: SFTConfig defaults → labels = input_ids ───────
    banner("(1) PRODUCTION SFTConfig path  (assistant_only_loss=False default)")
    labels_prod = list(input_ids)  # this is what TRL does when no masking flag is set
    print(f"sequence length:  {len(input_ids)} tokens")

    tok_per_role = Counter(role_of)
    in_loss_prod = Counter(r for r, lab in zip(role_of, labels_prod) if lab != -100)
    print("\nPer-role tokens / tokens-in-loss under PRODUCTION labels:")
    print(f"  {'role':<14} {'tokens':>8} {'in_loss':>8} {'frac':>6}")
    for r in sorted(tok_per_role):
        n = tok_per_role[r]
        k = in_loss_prod.get(r, 0)
        print(f"  {r:<14} {n:>8} {k:>8} {(k/n if n else 0):>6.2f}")
    total = sum(in_loss_prod.values())
    print(f"  {'TOTAL':<14} {len(input_ids):>8} {total:>8} {total/len(input_ids):>6.2f}")

    # ── (2) Open-instruct labels: per-message masking → assistant content only
    banner("(2) OPEN-INSTRUCT path  (sft_tulu_tokenize_and_truncate_v1 reimpl)")
    labels_oi = open_instruct_style_labels(tokenizer, messages, args.max_seq_length)
    in_loss_oi = Counter(r for r, lab in zip(role_of, labels_oi) if lab != -100)
    print("Per-role tokens / tokens-in-loss under OPEN-INSTRUCT labels:")
    print(f"  {'role':<14} {'tokens':>8} {'in_loss':>8} {'frac':>6}")
    for r in sorted(tok_per_role):
        n = tok_per_role[r]
        k = in_loss_oi.get(r, 0)
        print(f"  {r:<14} {n:>8} {k:>8} {(k/n if n else 0):>6.2f}")
    total = sum(in_loss_oi.values())
    print(f"  {'TOTAL':<14} {len(input_ids):>8} {total:>8} {total/len(input_ids):>6.2f}")

    # ── (3) TRL's planned path when assistant_only_loss=True ────────────────
    banner("(3) TRL's `return_assistant_tokens_mask=True` path on our template")
    assistant_mask: list[int] | None = None
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=args.max_seq_length,
            add_generation_prompt=False,
        )
        am = out.get("assistant_masks")
        if am is None:
            print("apply_chat_template returned no `assistant_masks` field.")
            print("=> Template likely has no {% generation %} markers.")
        else:
            assistant_mask = list(am)
            n_assist_marked = int(sum(am))
            print(f"assistant_masks length:    {len(am)}")
            print(f"sum(assistant_masks):      {n_assist_marked}")
            print(f"fraction marked:           {n_assist_marked / max(len(am), 1):.2f}")
            if n_assist_marked == 0:
                print("=> Mask is ALL ZEROS. The configured template "
                      f"({args.template}) lacks")
                print("   {% generation %}...{% endgeneration %} markers TRL needs.")
                print("   Setting `assistant_only_loss=True` today would be a no-op")
                print("   (or worse — raise, since TRL warns 'at least one example")
                print("   has no assistant tokens').")
            else:
                print("=> Template DOES emit an assistant mask. Cross-check against")
                print("   open-instruct labels (section 2) below.")
    except Exception as e:  # noqa: BLE001
        print(f"apply_chat_template(return_assistant_tokens_mask=True) raised: "
              f"{type(e).__name__}: {e}")
        print("=> TRL would surface the same error if `assistant_only_loss=True`.")

    # ── (3b) Diff between TRL mask and open-instruct labels ─────────────────
    banner("(3b) Mismatches between TRL mask  vs  open-instruct labels")
    if assistant_mask is None or len(assistant_mask) != len(input_ids):
        print("No TRL mask available (or length mismatch); skipping diff.")
    else:
        # Build TRL-equivalent "in_loss" boolean: token contributes to loss iff
        # mask[i] == 1.  OI-equivalent: labels_oi[i] != -100.
        trl_in_loss = [bool(m) for m in assistant_mask]
        oi_in_loss = [lab != -100 for lab in labels_oi]
        mismatches = [i for i in range(len(input_ids)) if trl_in_loss[i] != oi_in_loss[i]]
        only_in_trl = [i for i in mismatches if trl_in_loss[i] and not oi_in_loss[i]]
        only_in_oi  = [i for i in mismatches if oi_in_loss[i] and not trl_in_loss[i]]
        print(f"total mismatches:         {len(mismatches)}")
        print(f"  in TRL mask only:       {len(only_in_trl)}")
        print(f"  in open-instruct only:  {len(only_in_oi)}")
        if mismatches:
            print(f"\nFirst {min(20, len(mismatches))} mismatches (showing ±0 context):")
            print(f"  {'idx':>5} {'tok_id':>7}  {'role':<10}  trl  oi   token")
            for i in mismatches[:20]:
                tok = repr(tokenizer.decode([input_ids[i]]))
                if len(tok) > 38:
                    tok = tok[:38] + "..."
                trl = "L" if trl_in_loss[i] else "."
                oi  = "L" if oi_in_loss[i] else "."
                print(f"  {i:>5} {input_ids[i]:>7}  {role_of[i]:<10}   {trl}    {oi}   {tok}")
            print("\nBoundary context (5 tokens before/after the first mismatch):")
            i0 = mismatches[0]
            for j in range(max(0, i0 - 5), min(len(input_ids), i0 + 6)):
                tok = repr(tokenizer.decode([input_ids[j]]))
                if len(tok) > 38:
                    tok = tok[:38] + "..."
                trl = "L" if trl_in_loss[j] else "."
                oi  = "L" if oi_in_loss[j] else "."
                marker = " <-- first mismatch" if j == i0 else ""
                print(f"  {j:>5} {input_ids[j]:>7}  {role_of[j]:<10}   {trl}    {oi}   {tok}{marker}")

    # ── (4) Token-level trace ───────────────────────────────────────────────
    banner(f"(4) Token-level trace (first {args.max_trace} tokens)")
    print(f"  {'idx':>5} {'tok_id':>7}  {'role':<10}  prod  oi   token")
    n_show = min(args.max_trace, len(input_ids))
    for i in range(n_show):
        tok = tokenizer.decode([input_ids[i]])
        tok_repr = repr(tok)
        if len(tok_repr) > 38:
            tok_repr = tok_repr[:38] + "..."
        prod = "L" if labels_prod[i] != -100 else "."
        oi = "L" if labels_oi[i] != -100 else "."
        print(f"  {i:>5} {input_ids[i]:>7}  {role_of[i]:<10}   {prod}    {oi}   {tok_repr}")
    if n_show < len(input_ids):
        print(f"  ... ({len(input_ids) - n_show} more tokens not shown)")
    print("  (legend: 'L' = label != -100, contributes to CE loss; '.' = masked)")

    # ── (5) Verdict ─────────────────────────────────────────────────────────
    banner("(5) Verdict")
    nonassist_tokens = sum(tok_per_role.get(r, 0) for r in tok_per_role if r != "assistant")
    nonassist_in_loss_prod = sum(
        in_loss_prod.get(r, 0) for r in in_loss_prod if r != "assistant"
    )
    if nonassist_in_loss_prod > 0:
        print(f"CONFIRMED: production pipeline puts {nonassist_in_loss_prod} of "
              f"{nonassist_tokens} non-assistant tokens into the CE loss on this row.")
        print("Open-instruct's recipe puts 0. This is the gap.")
    else:
        print("NOT CONFIRMED: production labels already mask non-assistant tokens.")
        print("Hypothesis is wrong — look elsewhere.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
