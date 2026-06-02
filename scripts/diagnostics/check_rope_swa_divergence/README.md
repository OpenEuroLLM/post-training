# check_rope_swa_divergence

Cross-container forward-pass divergence diagnostic for the RoPE-on-SWA-layers
bug in HF transformers' `Olmo3Model`.

## What it does

Loads `allenai/OLMo-3-7B-Instruct-SFT` in both containers, runs the SAME
tokenized input through each, and compares log-probabilities position-by-position:

- **HF (training container, transformers 5.4.0)** — a single model-level
  `Olmo3RotaryEmbedding` produces YaRN-scaled `(cos, sin)` that is shared by
  all 32 layers, including the 24 SWA layers. (BUGGY per the comment in HF's
  own code; see `modular_olmo3.py` above `class Olmo3Model`.)
- **vLLM (eval container, vllm 0.19.0)** — `model_executor/models/olmo2.py`
  (serves `Olmo3ForCausalLM`) constructs a per-layer `rotary_emb`. SWA layers
  get vanilla RoPE (theta=500K, `rope_type="default"`); full-attention layers
  get the YaRN scaling. (CORRECT; matches OLMo-core.)

All other pipeline components are forced equal:

- same `model_id`, same shared input tokens (saved to disk by `tokenize.py`),
  same `bfloat16` dtype, same `olmo3-instruct-sft` chat template at tokenize
  time, HF uses `attn_implementation="eager"` to minimize kernel-impl noise,
  vLLM uses `enforce_eager=True`.

If HF and vLLM forwards diverge on this controlled comparison, the only
remaining difference is the RoPE-on-SWA wiring — i.e., the bug fires.

## Run

```bash
sbatch scripts/diagnostics/run_check_rope_swa_divergence.sh
```

1 node × 1 A100, `boost_qos_dbg`, ~5-10 min wall.

## Interpreting `compare.py` output

- **Mean |Δlogprob| < 0.05 nats** across positions: numerical kernel noise
  only. Bug not firing on this input — possibly because positions are too
  early for YaRN-vs-vanilla to diverge meaningfully. Try a longer input.
- **Mean |Δlogprob| > 0.2 nats** and rising with position: bug fires.
  Magnitude growing with position is consistent with RoPE phase
  accumulation.
- **Top-1 token disagreement > 5%**: the bug changes which token the model
  thinks is most likely at many positions — strong evidence the SFT regime
  is materially different from the eval regime.

## Files

- `tokenize_input.py` — pre-tokenize a Dolci row through our chat template, save
  to `run/input.npz`. Both containers consume the same file.
- `hf_forward.py` — HF model forward, save logits to `run/hf_logits.npz`.
- `vllm_forward.py` — vLLM forward, save top-K logprobs + actual-token
  logprobs to `run/vllm_logprobs.npz`.
- `compare.py` — diff stats, position-binned.