"""Verify the HF transformers Olmo3 RoPE-on-SWA bug by comparing teacher-forced
per-position log-probabilities between two forward paths on the same model
weights and the same byte-identical input_ids:

    HF transformers (training container, 5.4.0)
        - modeling_olmo3.Olmo3Model.forward computes ONE model-level rotary_emb
          and shares its YaRN-scaled (cos, sin) with all 32 layers, including
          the 24 SWA layers.  Comment at the top of Olmo3Model says SWA layers
          should NOT receive scaled RoPE — but the code does it anyway.
          -> BUGGY

    vLLM (eval container, 0.19.0)
        - olmo2.py serves Olmo3ForCausalLM.  At models/olmo2.py:141-150, for
          each layer it checks layer_types[layer_idx]; if SWA, it builds a
          per-layer rotary_emb with rope_type='default' (vanilla theta=500K),
          else with the YaRN scaling.
          -> CORRECT, matches OLMo-core

If both forwards produce the same per-position log-probabilities to within
numerical noise, the bug is harmless / doesn't fire on this input.  If they
diverge meaningfully (mean |diff| ≫ 1e-3 nats), the bug is real and the
trained HF weights are being evaluated under a different distribution than
they were trained on.

Designed so tokenization happens ONCE on the host venv (or any environment
with transformers).  The two GPU phases consume the saved input_ids file —
guaranteed identical input across containers.

Pipeline:
    --phase tokenize   tokenize the input prompt, save input_ids JSON
    --phase hf         load input_ids, HF forward, save hf_logprobs.json
    --phase vllm       load input_ids, vLLM forward, save vllm_logprobs.json
    --phase compare    diff the two logprob files, print verdict

Defaults assume all four files live in `--workdir` (default: this directory).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# ---- Constants matching the actual training pipeline ------------------------

# The Olmo3 model we're verifying against.  Using the canonical AllenAI
# checkpoint instead of one of ours: this isolates the bug from any
# training-quality artifact.  The bug fires for ANY Olmo3 model.
MODEL_ID = "allenai/OLMo-3-7B-Instruct-SFT"

# Tokenizer = same source as our production pipeline after fix #2 lands.
TOKENIZER_ID = "allenai/OLMo-3-7B-Instruct-SFT"

# A multi-paragraph prompt chosen to be long enough that YaRN scaling diverges
# noticeably from vanilla RoPE (YaRN's effect grows with position; at < ~1k
# tokens the divergence is tiny).  We repeat 8x to push into the regime where
# the bug should bite — roughly ~1500-2000 tokens after repetition.
_BASE = (
    "The OLMo-3 hybrid attention pattern interleaves sliding-window attention "
    "layers (which attend only to a local window of nearby tokens) with full "
    "attention layers (which attend to the entire sequence).  Three of every "
    "four layers use the sliding window of 4096 tokens.  Positional encoding "
    "is applied via Rotary Position Embeddings.  The OLMo-3 paper specifies "
    "that YaRN scaling — used to extend the model's effective context length "
    "from 8192 tokens to 65536 — is applied only to the full-attention layers; "
    "the sliding-window layers, which only ever see local positions within a "
    "4096-token window, retain the unscaled rotary base of theta = 500000.\n\n"
)
PROMPT = _BASE * 8

# Maximum number of input tokens to forward.  Keeps a single A100 forward
# comfortable (~14 GB for the 7B weights plus activations).  The bug's effect
# size grows with position, so longer is better — but past 4096 we'd also be
# hitting the sliding window in a less interesting way for this test.
MAX_INPUT_TOKENS = 4096


# ---- Phase 1: tokenize ------------------------------------------------------


def phase_tokenize(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    encoded = tok(PROMPT, add_special_tokens=True, return_tensors=None)
    ids = encoded["input_ids"]
    if len(ids) > MAX_INPUT_TOKENS:
        ids = ids[:MAX_INPUT_TOKENS]

    out_path = Path(args.output or args.workdir) / "input_ids.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "tokenizer_id": TOKENIZER_ID,
                "prompt_chars": len(PROMPT),
                "n_tokens": len(ids),
                "input_ids": list(map(int, ids)),
            },
            f,
            indent=2,
        )
    print(f"[tokenize] wrote {len(ids)} input_ids → {out_path}")


# ---- Phase 2: HF forward (training container) ------------------------------


def phase_hf(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    in_path = Path(args.input or args.workdir) / "input_ids.json"
    with open(in_path) as f:
        prep = json.load(f)
    ids = prep["input_ids"]

    print(f"[hf] transformers version =", __import__("transformers").__version__)
    print(f"[hf] loading {MODEL_ID} on cuda in bfloat16, attn_impl=flash_attention_2")

    # Matched against our production training: bf16 weights.  FA2 instead of
    # the training default FA3, because FA3 in this codepath has been observed
    # to suppress small numerical mismatches that we explicitly want to see.
    # The RoPE bug is structural, not numerical — both impls should reveal it.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits[0].float()  # (seq_len, vocab); upcast for numerical stability of log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)

    # log P(ids[i+1] | ids[0..i]) — teacher-forced. Aligns with vLLM's
    # prompt_logprobs (which conditions ids[i] on ids[0..i-1]).
    per_pos = []
    for i in range(len(ids) - 1):
        per_pos.append(
            {
                "position": i,
                "context_last_id": int(ids[i]),
                "next_token_id": int(ids[i + 1]),
                "next_token_logprob": float(log_probs[i, ids[i + 1]].item()),
            }
        )

    out_path = Path(args.output or args.workdir) / "hf_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
                "n_tokens": len(ids),
                "input_ids": list(map(int, ids)),
                "per_position_logprobs": per_pos,
            },
            f,
            indent=2,
        )
    print(f"[hf] wrote {len(per_pos)} per-position logprobs → {out_path}")


# ---- Phase 3: vLLM forward (eval container) --------------------------------


def phase_vllm(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    in_path = Path(args.input or args.workdir) / "input_ids.json"
    with open(in_path) as f:
        prep = json.load(f)
    ids = prep["input_ids"]

    print(f"[vllm] vllm version =", __import__("vllm").__version__)
    print(f"[vllm] loading {MODEL_ID}")

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enforce_eager=True,            # disable CUDA graph capture for determinism
        gpu_memory_utilization=0.50,   # 7B in bf16 ≈ 14GB; leave plenty for KV
        max_model_len=max(len(ids) + 16, 8192),
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,                  # generation length doesn't matter; we want prompt logprobs
        prompt_logprobs=1,             # logprob of the actual prompt token at each position
    )
    outputs = llm.generate(prompt_token_ids=[ids], sampling_params=sp)
    pps = outputs[0].prompt_logprobs   # list aligned with input_ids; pps[0] is None

    # pps[i] is dict[token_id -> Logprob]; we want the logprob of the actual
    # token at position i (= ids[i]).  Position 0 has no condition, so skipped.
    per_pos = []
    for i, position_dict in enumerate(pps):
        if i == 0 or position_dict is None:
            continue
        actual_id = ids[i]
        entry = position_dict.get(actual_id)
        if entry is None:
            # Shouldn't happen — vLLM always includes the actual token
            continue
        per_pos.append(
            {
                "position": i - 1,           # align with HF indexing
                "context_last_id": int(ids[i - 1]),
                "next_token_id": int(actual_id),
                "next_token_logprob": float(entry.logprob),
            }
        )

    out_path = Path(args.output or args.workdir) / "vllm_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "dtype": "bfloat16",
                "n_tokens": len(ids),
                "input_ids": list(map(int, ids)),
                "per_position_logprobs": per_pos,
            },
            f,
            indent=2,
        )
    print(f"[vllm] wrote {len(per_pos)} per-position logprobs → {out_path}")


# ---- Phase 4: compare ------------------------------------------------------


def phase_compare(args: argparse.Namespace) -> None:
    hf_path = Path(args.hf or (Path(args.workdir) / "hf_logprobs.json"))
    vl_path = Path(args.vllm or (Path(args.workdir) / "vllm_logprobs.json"))
    with open(hf_path) as f:
        hf = json.load(f)
    with open(vl_path) as f:
        vl = json.load(f)

    assert hf["input_ids"] == vl["input_ids"], (
        "Input IDs differ between HF and vLLM — tokenization is not controlled! "
        "Re-run with the same input_ids.json fed to both."
    )

    hf_lps = {p["position"]: p for p in hf["per_position_logprobs"]}
    vl_lps = {p["position"]: p for p in vl["per_position_logprobs"]}
    common = sorted(set(hf_lps) & set(vl_lps))

    diffs = []
    for pos in common:
        h = hf_lps[pos]
        v = vl_lps[pos]
        assert h["next_token_id"] == v["next_token_id"]
        diff = h["next_token_logprob"] - v["next_token_logprob"]
        diffs.append((pos, diff, h["next_token_logprob"], v["next_token_logprob"]))

    abs_diffs = [abs(d) for _, d, _, _ in diffs]
    sum_diff = sum(d for _, d, _, _ in diffs)

    print(f"Compared {len(diffs)} positions on input of {hf['n_tokens']} tokens.")
    print()
    print(f"Per-position |HF - vLLM| log-prob differences (nats):")
    print(f"    mean   = {statistics.mean(abs_diffs):.4e}")
    print(f"    median = {statistics.median(abs_diffs):.4e}")
    print(f"    p95    = {statistics.quantiles(abs_diffs, n=20)[18]:.4e}")
    print(f"    max    = {max(abs_diffs):.4e}")
    print()
    print(f"Cumulative diff (sum_i HF - vLLM): {sum_diff:+.3f} nats over {len(diffs)} positions")
    print(f"    ≈ {sum_diff / len(diffs):+.4e} nats/token average")
    print()

    # Position-binned: bug effect should grow with position (YaRN diverges from
    # vanilla more at later positions).  Plot the divergence vs position so we
    # can see whether the diff is structural (grows) or just noise (flat).
    n_bins = min(20, len(diffs))
    bin_size = max(1, len(diffs) // n_bins)
    print("Position-binned mean |HF - vLLM| log-prob diff (bars scaled to max):")
    bin_means = []
    for bi in range(n_bins):
        lo, hi = bi * bin_size, (bi + 1) * bin_size
        if lo >= len(diffs):
            break
        b = [abs_diffs[i] for i in range(lo, min(hi, len(diffs)))]
        if b:
            bin_means.append((diffs[lo][0], statistics.mean(b)))
    max_bm = max(m for _, m in bin_means) if bin_means else 1.0
    for start_pos, mean_abs in bin_means:
        bar_len = int(50 * mean_abs / max_bm) if max_bm > 0 else 0
        bar = "█" * bar_len
        print(f"    pos>={start_pos:5d}  mean|diff|={mean_abs:.4e}  {bar}")

    print()
    avg = statistics.mean(abs_diffs)
    print("VERDICT:")
    if avg < 1e-3:
        print("    HF and vLLM forwards are numerically indistinguishable on this input")
        print("    (mean |diff| < 1e-3 nats). Either the RoPE-on-SWA bug doesn't fire")
        print("    on this input, or its effect is below noise. Consider re-running")
        print("    with a longer input (the bug grows with position).")
    elif avg < 1e-2:
        print("    Small but real divergence (mean |diff| ~ 1e-3 to 1e-2 nats). Bug")
        print("    fires but impact per token is modest. Cumulative effect across")
        print("    20+ training-token sequences could still be significant.")
    else:
        print(f"    Significant divergence (mean |diff| = {avg:.2e} nats >> 1e-2).")
        print("    The bug clearly fires. HF and vLLM are producing materially")
        print("    different distributions on the same model + same input. Our")
        print("    trained weights have been optimised against HF's distribution")
        print("    and are being evaluated against vLLM's — a train/eval mismatch.")


# ---- entrypoint ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--phase",
        choices=["tokenize", "hf", "vllm", "compare"],
        required=True,
    )
    ap.add_argument(
        "--workdir",
        default=str(Path(__file__).parent / "rope_swa_verification"),
        help="Directory for the four pipeline artifacts. Default: %(default)s",
    )
    ap.add_argument("--input", help="Override input file path (for hf/vllm phases).")
    ap.add_argument("--output", help="Override output file path (for tokenize/hf/vllm phases).")
    ap.add_argument("--hf", help="hf_logprobs.json path (compare phase).")
    ap.add_argument("--vllm", help="vllm_logprobs.json path (compare phase).")
    args = ap.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    {
        "tokenize": phase_tokenize,
        "hf": phase_hf,
        "vllm": phase_vllm,
        "compare": phase_compare,
    }[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
