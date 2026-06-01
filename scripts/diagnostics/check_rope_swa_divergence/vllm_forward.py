"""vLLM forward — the path with the correct per-layer rotary_emb.

vLLM maps Olmo3ForCausalLM -> olmo2.py, which constructs a per-layer
rotary_emb: SWA layers get vanilla RoPE (no YaRN), full-attn layers get
the YaRN-scaled config. This matches OLMo-core and is the eval-time
behavior all our direct-battle and ELO runs go through.

Run inside the vLLM eval container (vllm 0.19.0, transformers 4.57.6).
"""
import argparse
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="allenai/OLMo-3-7B-Instruct-SFT")
    ap.add_argument("--input",
                    default=str(Path(__file__).parent / "run" / "input.npz"))
    ap.add_argument("--output",
                    default=str(Path(__file__).parent / "run" / "vllm_logprobs.npz"))
    ap.add_argument("--top-k", type=int, default=5,
                    help="vLLM prompt_logprobs=K; the actual token's logprob is "
                    "always included in addition to the top-K-1.")
    args = ap.parse_args()

    data = np.load(args.input)
    input_ids = data["input_ids"].tolist()
    print(f"[vllm_forward] Loaded {len(input_ids)} input tokens")

    print(f"[vllm_forward] Loading {args.model_id} via vLLM...")
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=max(2048, len(input_ids) + 64),
        tensor_parallel_size=1,
    )
    print(f"[vllm_forward] Forwarding (prompt_logprobs={args.top_k}) ...")

    sampling = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=args.top_k,
    )
    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=input_ids)],
        sampling_params=sampling,
        use_tqdm=False,
    )
    plp = outputs[0].prompt_logprobs  # list of len(input_ids); [0] is None.

    actual_logprob = np.full(len(input_ids), np.nan, dtype=np.float32)
    top1_token = np.full(len(input_ids), -1, dtype=np.int64)
    top1_logprob = np.full(len(input_ids), np.nan, dtype=np.float32)

    for i, lp_dict in enumerate(plp):
        if lp_dict is None:
            continue
        # vLLM always includes the actual token's logprob in the dict.
        actual = input_ids[i]
        if actual in lp_dict:
            actual_logprob[i] = float(lp_dict[actual].logprob)
        # Top-1 token (highest logprob across the dict).
        best_tid, best_lp = -1, -float("inf")
        for tid, lp in lp_dict.items():
            v = float(lp.logprob)
            if v > best_lp:
                best_lp, best_tid = v, tid
        top1_token[i] = best_tid
        top1_logprob[i] = best_lp

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        input_ids=data["input_ids"],
        actual_logprob=actual_logprob,
        top1_token=top1_token,
        top1_logprob=top1_logprob,
    )
    print(f"[vllm_forward] Wrote logprobs to {out_path}")


if __name__ == "__main__":
    main()
