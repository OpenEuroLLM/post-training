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

### Pending commit — feat: multi-node support and Dolci Instruct configs

- **`main_process_first()` wrapper**: `load_and_mix_datasets()` is wrapped in `PartialState().main_process_first()` to prevent HF datasets cache races on Lustre in multi-rank runs. (`sft.py`, `dpo.py`)
- **`_sanitize_generation_config()`**: Olmo-3 Think models ship `temperature/top_p` with `do_sample=False`, crashing `transformers >= 5.x` strict validation on `save_pretrained()`. We set `do_sample=True` in-memory. AllenAI's open-instruct solves the same issue by stripping the params instead (see their `model_utils.py`). (`sft.py`)
- **Leonardo env**: `NCCL_IB_HCA=mlx5`, HF cache paths, offline mode. (`env/leonardo.env`)
- **Dolci configs**: production (4 nodes, 2 epochs) and debug (4 nodes, 100 steps). (`configs/trl/`)

## Dataset

- **Path**: `/leonardo_scratch/large/userexternal/knikolao/propella_annotation/data/Dolci-Instruct-SFT-Decont/data`
- **Format**: parquet with `messages` column (role/content/function_calls/functions), ~143k rows
- **Chat template**: `olmo3` (includes `<think>` tag in generation prompt)
