"""HF transformers forward — the path with the model-level single rotary_emb.

Loads Olmo3, prints a few sanity-check lines about the bug surface (single
model-level rotary_emb shared by all 32 layers, including 24 SWA ones),
then runs one forward on the shared input_ids and saves the full logits.

Run inside the post-training container (transformers 5.4.0).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="allenai/OLMo-3-7B-Instruct-SFT")
    ap.add_argument("--input",
                    default=str(Path(__file__).parent / "run" / "input.npz"))
    ap.add_argument("--output",
                    default=str(Path(__file__).parent / "run" / "hf_logits.npz"))
    ap.add_argument("--attn", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
                    help="HF attn implementation. 'eager' minimizes kernel-impl "
                    "noise vs vLLM. The RoPE bug fires regardless.")
    ap.add_argument("--apply-patch", action="store_true",
                    help="Install patch_olmo3_swa_rope.install() before loading "
                    "the model so SWA layers get unscaled RoPE (matches vLLM/OLMo-core).")
    ap.add_argument("--bench-iters", type=int, default=20,
                    help="Forward-pass timing iterations after the main logits "
                    "computation. 0 disables benchmarking.")
    args = ap.parse_args()

    if args.apply_patch:
        sys.path.insert(0, str(Path(__file__).parent))
        import patch_olmo3_swa_rope
        patch_olmo3_swa_rope.install()
        print("[hf_forward] PATCH INSTALLED: Olmo3Model now uses per-layer RoPE "
              "(SWA layers vanilla, full-attn YaRN-scaled).")

    print(f"[hf_forward] Loading {args.model_id}  attn={args.attn}  dtype=bf16  "
          f"patch={args.apply_patch}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    ).cuda()
    model.eval()

    # Sanity-print the bug surface so the operator can verify we're testing
    # what we think we're testing.
    cfg = model.config
    layer_types = cfg.layer_types
    n_swa = sum(1 for t in layer_types if t == "sliding_attention")
    rope_scaling = getattr(cfg, "rope_scaling", None) or {}
    rope_theta = (rope_scaling.get("rope_theta")
                  if isinstance(rope_scaling, dict)
                  else getattr(cfg, "rope_theta", None))
    print(f"  config class:        {type(cfg).__name__}")
    print(f"  num_hidden_layers:   {cfg.num_hidden_layers}")
    print(f"  layer_types[:8]:     {layer_types[:8]}")
    print(f"  SWA layers:          {n_swa}/{len(layer_types)}")
    print(f"  rope_scaling:        {rope_scaling}")
    print(f"  rope_theta:          {rope_theta}")

    # Bug surface: stock HF has ONE model-level rotary_emb shared by all 32 layers.
    # The patch (mirroring upstream PR #45945) replaces it with a ModuleDict
    # `rotary_embs` keyed by layer type: SWA gets vanilla RoPE, full-attn keeps YaRN.
    def _scale(r):
        s = getattr(r, "attention_scaling", None)
        if s is None:
            s = getattr(r, "scaling_factor", None)
        return s
    rotary_embs = getattr(model.model, "rotary_embs", None)
    if rotary_embs is not None:
        full = rotary_embs["full_attention"]
        swa = rotary_embs["sliding_attention"]
        print(f"  rotary_embs[full]:   {type(full).__name__}  "
              f"rope_type={getattr(full, 'rope_type', '?')}  scaling={_scale(full)}")
        print(f"  rotary_embs[swa]:    {type(swa).__name__}  "
              f"rope_type={getattr(swa, 'rope_type', '?')}  scaling={_scale(swa)}  "
              "← patched. SWA layers use this (vanilla RoPE).")
    else:
        rope = model.model.rotary_emb
        print(f"  rotary_emb (full):   {type(rope).__name__}  "
              f"rope_type={getattr(rope, 'rope_type', '?')}  scaling={_scale(rope)}")
        print(f"  rotary_embs:         <none>  ← SWA layers share scaled rotary_emb (BUG)")

    # Load shared input and forward.
    data = np.load(args.input)
    input_ids = torch.tensor(data["input_ids"], dtype=torch.long).unsqueeze(0).cuda()
    print(f"[hf_forward] Forwarding {input_ids.shape[1]} tokens ...")
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0].float().cpu().numpy()
    print(f"  logits shape: {logits.shape}")

    # Speed benchmark: do N warmup + N timed forwards on the same input so we
    # can measure whether the patch (2x rotary_emb computation + per-layer
    # dispatch) costs us throughput.
    import time
    N_WARMUP = 5
    N_TIMED = args.bench_iters
    timings_ms = []
    if N_TIMED > 0:
        print(f"[hf_forward] Benchmark: {N_WARMUP} warmup + {N_TIMED} timed forwards ...")
        with torch.no_grad():
            for _ in range(N_WARMUP):
                model(input_ids)
            torch.cuda.synchronize()
            for _ in range(N_TIMED):
                t0 = time.perf_counter()
                model(input_ids)
                torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(timings_ms)
        print(f"  forward time / iter (ms): "
              f"median={np.median(arr):.2f}  "
              f"min={arr.min():.2f}  "
              f"mean={arr.mean():.2f}  "
              f"p90={np.percentile(arr, 90):.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        logits=logits,
        input_ids=data["input_ids"],
        timings_ms=np.array(timings_ms, dtype=np.float64),
        patched=bool(args.apply_patch),
        attn=args.attn,
        seq_len=int(input_ids.shape[1]),
    )
    print(f"[hf_forward] Wrote logits + benchmark to {out_path}")


if __name__ == "__main__":
    main()
