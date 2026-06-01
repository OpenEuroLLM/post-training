# Claude Context — Post-Training Framework

## Project

Fine-tuning **allenai/Olmo-3-7B-Think-SFT** on **Dolci-Instruct-SFT** using this TRL-based post-training framework on CINECA Leonardo (A100 SXM 64GB nodes).

## Cluster: Leonardo (CINECA)

- **Container**: `/leonardo_work/OELLM_prod2026/container_images/post-training-flash-attn-3.sif` — PyTorch 2.9, CUDA 13.0, FA3, TRL 0.29.1, transformers 5.4.0, DeepSpeed 0.18.9
- **HF cache**: `$HF_HOME=/leonardo_scratch/large/userexternal/knikolao/huggingface` (set in `~/.bashrc`)
- **Env file**: `env/leonardo.env` — HF paths, `NCCL_IB_HCA=mlx5` (enables all 4 IB HCAs on compute nodes, verified against [CINECA docs](https://eurohpcsupport.eu/best-practice/best-practice-guide-to-running-ai-workloads-on-leonardo-cineca/))
- **Bare-metal venv** (`.venv/`) exists but has a CUDA driver mismatch with Leonardo's compute nodes — use the container for all training runs
- **SLURM QoS options**: `boost_qos_dbg` (30 min, 4 nodes max), default (24h, 64 nodes), `boost_qos_lprod` (96h, 8 nodes, lower priority)

## Running

```bash
# Debug (4 nodes, 100 steps, ~15 min)
python scripts/submit.py --config configs/trl/dolci_instruct_sft_container_debug.yaml

# Production (4 nodes, 2 epochs, ~19h)
python scripts/submit.py --config configs/trl/dolci_instruct_sft_container.yaml
```

Logs land in `outputs/<debug/>/sft-.../slurm/slurm-<jobid>.out`.

## Validated performance (4 nodes × 4 A100s, 32k seq_len, ZeRO-2)

- **Production run** (Dolci-Instruct-SFT-Decont, 2 epochs, 3,408 steps, 3.49B tokens): ~45.6k tok/sec, ~2.8k tok/sec/GPU, ~39% MFU, **21.2h wall time**
- **Debug run** (100K rows, no WandB logging): ~51k tok/sec, ~3.2k tok/sec/GPU, ~46% MFU, ~20s/step
- Checkpoints save cleanly via HF Trainer's native path

## Key fixes in this branch

### `dac4c64` — fix: warmup bug, missing cpus_per_task, and run-dir wipe

- **warmup_steps truncation**: `warmup_ratio` (float 0.03) was passed to `warmup_steps` (int), silently disabling warmup. Now both fields are forwarded correctly. (`config.py`, `common.py`)
- **Container template missing `--cpus-per-task`**: SLURM allocated ~15 GB instead of ~480 GB host RAM. This was the root cause of all checkpoint save OOMs. (`job_trl_container.sh.jinja`, `launcher.py`)
- **`CUDA_DEVICE_MAX_CONNECTIONS=1` removed**: Megatron-LM tensor-parallel optimization that serializes CUDA streams and hurts ZeRO-2's `overlap_comm`. (`job_trl_container.sh.jinja`)
- **Run-dir wipe race**: `setup_run_directory` could delete the run dir when called from `train.py` on compute nodes during a running job. Now gated via `allow_override` (only `submit.py` can wipe). (`paths.py`, `submit.py`)
- **New config fields**: `qos`, `mem`, `save_strategy`, `warmup_steps` added for flexible SLURM and training control. (`config.py`, `common.py`, templates, `launcher.py`)
- **PYTHONPATH in container template**: ensures local `src/` takes priority over the container's baked-in `post_training` package. (`job_trl_container.sh.jinja`)

### `506660d` — feat: multi-node Dolci SFT on Leonardo + wandb offline support

- **`main_process_first()` wrapper**: `load_and_mix_datasets()` is wrapped in `PartialState().main_process_first()` to prevent HF datasets cache races on Lustre in multi-rank runs. (`sft.py`, `dpo.py`)
- **`_sanitize_generation_config()`**: Olmo-3 Think models ship `temperature/top_p` with `do_sample=False`, crashing `transformers >= 5.x` strict validation on `save_pretrained()`. We set `do_sample=True` in-memory. AllenAI's open-instruct solves the same issue by stripping the params instead (see their `model_utils.py`). (`sft.py`)
- **WANDB_DIR in container template**: wandb offline runs now persist to `<run_dir>/wandb/` on shared storage instead of being lost on compute node local scratch. (`job_trl_container.sh.jinja`)
- **Leonardo env**: `NCCL_IB_HCA=mlx5`, HF cache paths, offline mode, `WANDB_PROJECT`. (`env/leonardo.env`)
- **Dolci configs**: production (4 nodes, 2 epochs) and debug (4 nodes, 100 steps), matching OLMo-3 paper Table 47 hyperparameters (lr=8e-5, linear schedule to 0, adam beta2=0.95, warmup_ratio=0.03, 32K seq len). Uses decontaminated dataset. (`configs/trl/`)
- **CLAUDE.md**: project documentation with validated throughput numbers

### Pending — feat: Olmo3 SWA RoPE patch (mirrors HF transformers PR #45945)

- **Root-cause**: HF transformers ≤ 5.4.x's `Olmo3Model.forward` computes a single model-level `Olmo3RotaryEmbedding` whose YaRN-scaled `(cos, sin)` is fed to all 32 layers — including the 24 sliding-window-attention (SWA) layers, which were trained with vanilla (unscaled) RoPE in OLMo-core. Introduced by the "Refactor RoPE for layer types" transformers PR #39847 (2025-10-17); fixed by community PR #45945 (still draft, no maintainer review). vLLM and OLMo-core both do per-layer RoPE correctly. Cross-container diagnostic shows mean |Δlogprob| of 0.0743 nats vs vLLM (3.22% top-1 disagreement) on a Dolci row through stock HF — the bug fires.
- **`post_training.patches.olmo3_swa_rope.install()`**: monkey-patches `Olmo3RotaryEmbedding.__init__` to accept a `rope_type` kwarg, replaces `Olmo3Model.rotary_emb` with `rotary_embs = nn.ModuleDict({"sliding_attention": Olmo3RotaryEmbedding(rope_type="default"), "full_attention": Olmo3RotaryEmbedding(...)})`, and dispatches per-layer in `forward` via `position_embeddings_mapping[layer_type]`. Byte-for-byte mirror of PR #45945. Closes the divergence to 0.0049 nats (kernel noise floor). (`src/post_training/patches/olmo3_swa_rope.py`)
- **Auto-install on SFT**: `methods/sft.py` calls `maybe_install_olmo3_swa_rope(config.model.name_or_path)` before `SFTTrainer` constructs the model. Skips for non-Olmo3 models; skips if the installed transformers already carries the fix (`inspect.getsource(Olmo3Model.__init__)` contains `"rotary_embs"`) — so the patch self-retires the day upstream merges. (`src/post_training/methods/sft.py`, `src/post_training/patches/`)
- **Empirical lift on a 4h test run** (`outputs/sft-olmo-3-7b-think-sft-dolci_instruct_sft-20260527-153634/`): step-1 loss 0.983 vs the May 22 bundle's 1.224 at identical LR; step-500 alpaca-eval win-rate 36.5% vs May 22's 28.8% (+7.7 pp); arena-hard-v2.0 step-500 35.6% vs May 22's 25.4% (+10.2 pp). Arena-hard moves more than alpaca-eval — exactly the "long/hard-prompt" signature the SWA mis-rotation predicted. See [REPRODUCTION_GAP_INVESTIGATION.md Session 5](../outputs/sft-olmo-3-7b-think-sft-dolci_instruct_sft-20260413-144518/REPRODUCTION_GAP_INVESTIGATION.md) for the full narrative.
- **`handle_signal` gate**: SLURM template's wall-time signal handler now no-ops when `max_failures <= 1`, so single-shot test runs die cleanly at the wall instead of auto-requeueing. Production runs (`max_failures: 3`) keep their existing auto-resume behavior. (`job_trl_container.sh.jinja`)
- **Short test config**: `configs/trl/dolci_instruct_sft_container_short.yaml` — mirrors the production config exactly (so TRL's tokenized-dataset cache hits the May 22 fingerprint) but caps wall at 4h, sets `max_failures: 1`, and writes inference checkpoints every 50 steps. Critically keeps `num_train_epochs: 2` so the LR schedule (warmup ≈ 100 steps, linear decay over ~3354 steps) is byte-identical to production — checkpoint at step N is on the same LR curve as step N of prior runs, making matched-step comparison valid.

### Pending — feat: RoPE-on-SWA cross-container divergence diagnostic

- **`scripts/diagnostics/check_rope_swa_divergence/`**: harness that loads `allenai/OLMo-3-7B-Instruct-SFT` in both the training container (HF transformers 5.4.0) and the eval container (vLLM 0.19.0), runs the same Dolci-row token stream through both, and diffs per-position log-probabilities. Position-binned breakdown shows the [64, 256) range is where YaRN-vs-vanilla rotation difference fires hardest. Used to (a) prove the bug fires on real Dolci data; (b) verify each patch revision (deepcopy-style → upstream-PR-style → package-shim) lands at the same kernel-noise floor.
- **`scripts/diagnostics/run_check_rope_swa_divergence.sh`**: 1 node × 1 A100, `boost_qos_dbg`, ~10 min wall. Runs the full pipeline (tokenize → HF stock forward → HF patched forward → vLLM forward → compare) and prints a V1+V2 verdict (bug fires? does the patch close it?).
- **`scripts/diagnostics/verify_rope_swa_bug.py`** + `run_verify_rope_swa_bug.sh`: earlier single-container check; superseded by the cross-container harness but kept for reference.

## Dataset

- **Path**: `/leonardo_scratch/large/userexternal/knikolao/propella_annotation/data/Dolci-Instruct-SFT-Decont/data`
- **Format**: parquet with `messages` column (role/content/function_calls/functions), ~143k rows
- **Chat template**: `olmo3` (includes `<think>` tag in generation prompt)
