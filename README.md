# Post-Training

A modular framework for post-training large language models, supporting **SFT** (supervised fine-tuning) and **DPO** (direct preference optimisation). Built on [TRL](https://github.com/huggingface/trl) with DeepSpeed ZeRO and multi-node SLURM support.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Local SFT training (single-node)

```bash
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --dynamo_backend=inductor \
    --use_deepspeed \
    --same_network \
    --rdzv_backend static \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/sft.yaml \
    training.max_steps=100 \
    offline=true
```

### 3. Local DPO training (single-node)

```bash
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --dynamo_backend=inductor \
    --use_deepspeed \
    --same_network \
    --rdzv_backend static \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/dpo.yaml \
    training.max_steps=100 \
    offline=true
```

### 4. SLURM submission

```bash
python scripts/submit.py --config configs/sft.yaml training.max_steps=100 offline=true
python scripts/submit.py --config configs/dpo.yaml training.max_steps=100 offline=true
```

### 5. Switch DeepSpeed ZeRO stage (example)

```bash
scripts/train.py --config configs/sft.yaml deepspeed.config_path=configs/deepspeed/zero3.yaml
```

> [!NOTE]
> The `--mixed_precision` flag passed to `accelerate launch` must match `model.dtype` in your config.

## Table of Contents

- [Project Structure](#project-structure)
- [Configuration: one YAML + CLI overrides](#configuration-one-yaml--cli-overrides)
- [Run outputs & directory layout](#run-outputs--directory-layout)
- [Configuration reference (by YAML section)](#configuration-reference-by-yaml-section)

## Project Structure

```
post-training/
├── configs/
│   ├── sft.yaml                  # SFT example config
│   ├── dpo.yaml                  # DPO example config
│   └── deepspeed/
│       ├── zero2.yaml            # DeepSpeed ZeRO Stage 2 config
│       └── zero3.yaml            # DeepSpeed ZeRO Stage 3 config
├── src/post_training/
│   ├── config.py                 # OmegaConf dataclass schema + validation
│   ├── methods/                  # Trainer builders (SFT/DPO)
│   ├── data/                     # Dataset loading, transforms, mixing
│   ├── chat_templates/           # Chat template registry + Jinja templates
│   ├── callbacks/                # Custom callbacks (e.g., inference checkpoints)
│   ├── slurm/                    # SLURM script rendering + submission
│   └── utils/                    # Logging + run directory utilities
├── scripts/
│   ├── train.py                  # Training entrypoint (supports CLI overrides)
│   ├── submit.py                 # SLURM submission entrypoint
│   └── inspect_data.py           # Data pipeline debugger
└── pyproject.toml
```

## Configuration: one YAML + CLI overrides

All runs are configured via a **single YAML file** passed with `--config` (e.g. `configs/sft.yaml` or `configs/dpo.yaml`).

Both `scripts/train.py` and `scripts/submit.py` accept additional arguments as **OmegaConf dot-notation overrides** (unknown args are forwarded to the config loader). This makes it easy to run quick experiments without copying YAML files.

Examples:

```bash
# Change model and training length
scripts/train.py --config configs/sft.yaml model.name_or_path=org/model training.max_steps=200

# Disable SFT packing (enabled by default)
scripts/train.py --config configs/sft.yaml sft.packing=false

# Token-based training length (ONLY valid when sft.packing=true)
scripts/train.py --config configs/sft.yaml training.num_training_tokens=500000000

# Switch DeepSpeed config
scripts/train.py --config configs/sft.yaml deepspeed.config_path=configs/deepspeed/zero3.yaml
```

## Run outputs & directory layout

Each run writes into a run directory created by `setup_run_directory()`.

- **Base path**: `paths.output_base` (or `paths.debug_base` if `debug.enabled=true`)
- **Run name**: auto-generated from method/model/dataset mix, unless `run_name` is provided

Directory layout:

```text
<paths.output_base>/<run_name>/
  config.yaml
  checkpoints/
    checkpoint-*/
  inference_checkpoints/
    step-*/
  logs/
  slurm/
    job.sh
    slurm-<jobid>.out
    slurm-<jobid>.err
    failure_count
```

What each artifact is for:

- `config.yaml`: frozen config for the run (includes CLI overrides).
- `checkpoints/`: full TRL checkpoints (`checkpoint-*`) used for resume.
- `inference_checkpoints/`: lightweight model+tokenizer saves (`step-*`) for inference/eval.
- `logs/`: TensorBoard logs (via `TENSORBOARD_LOGGING_DIR`).
- `slurm/`: generated SLURM script and SLURM stdout/stderr logs.

Auto-resume:

- On startup, `scripts/train.py` scans `checkpoints/checkpoint-*` and resumes from the latest checkpoint if present.

## Configuration reference (by YAML section)

This section pairs each feature with its configuration keys and explains the impact.

### Top-level

- `method`: selects trainer (`sft` or `dpo`).
- `run_name`: optional explicit run name; otherwise auto-generated.
- `offline`: when true, disables Hugging Face + wandb network calls. This must be set to true when training on a cluster where the internet is not reachable from the compute nodes.

### `model`

- `model.name_or_path`: Hugging Face model ID or local path.
- `model.attn_implementation`: attention backend (e.g. `flash_attention_3`).
- `model.dtype`: must match `accelerate launch --mixed_precision`.

### `training`

**Training length (mutually exclusive)**

You must specify **exactly one** of:

- `training.max_steps`: explicit optimizer steps.
- `training.num_training_samples`: derives steps as `ceil(samples / effective_batch_size)`.
- `training.num_training_tokens`: derives steps as `ceil(tokens / (effective_batch_size * sft.max_seq_length))`.
  - Only valid when `method: sft` and `sft.packing: true`.

**Batch sizing**

- `training.effective_batch_size`: total samples per optimizer step across all GPUs.
- `training.per_device_train_batch_size`: per-GPU microbatch.
- Gradient accumulation is derived to match the effective batch size.

**Performance / memory**

- `training.gradient_checkpointing`: reduces activation memory at the cost of compute.
- `training.use_liger_kernel`: enables Liger kernels (if installed) for performance.

**Optimization**

- `training.learning_rate`, `training.warmup_ratio`
- `training.lr_scheduler_type`, `training.lr_scheduler_kwargs.min_lr_rate`

### `sft` (SFT-only)

- `sft.max_seq_length`: maximum sequence length.
- `sft.packing`: when true, packs multiple examples into sequences (enables `training.num_training_tokens`).

### `dpo` (DPO-only)

- `dpo.beta`: DPO beta.
- `dpo.loss_type`: loss type (e.g. `sigmoid`).
- `dpo.ref_model_name_or_path`: optional explicit ref model (null => implicit copy).
- `dpo.max_seq_length`: maximum sequence length.

### `checkpointing`

- Full checkpoints (`checkpoints/checkpoint-*`):
  - `checkpointing.save_steps`
  - `checkpointing.save_total_limit`
- Inference checkpoints (`inference_checkpoints/step-*`):
  - `checkpointing.inference_checkpoint_steps`
    - When set to `null` (or a non-positive value via CLI overrides), inference
      checkpointing is disabled and no `inference_checkpoints/` directory is
      created.
  - `checkpointing.inference_checkpoint_path`

### `data`

- `data.chat_template`: selects a chat template from the registry.
- `data.num_proc`: workers for `datasets.map/filter` (auto-capped).
- `data.datasets`: list of dataset entries:
  - `name`, `path`, `subset`, `split`, `weight`, `transform`

Custom transforms:

- Define a function in `src/post_training/data/transforms.py` with signature
  `fn(example: dict[str, Any]) -> dict[str, Any]` and register it:

  ```python
  from post_training.data.transforms import register_transform

  @register_transform("my_transform")
  def my_transform(example: dict[str, Any]) -> dict[str, Any]:
      return {
          "messages": [
              {"role": "user", "content": example["prompt"]},
              {"role": "assistant", "content": example["answer"]},
          ]
      }
  ```

- Reference it in your YAML:

  ```yaml
  data:
    datasets:
      - name: "my_dataset"
        path: "org/my-dataset"
        split: "train"
        weight: 1.0
        transform: "my_transform"
  ```

Data pipeline debugging:

```bash
python scripts/inspect_data.py --config configs/sft.yaml --num-samples 5
```

Advanced: one-off transforms in `scripts/train.py`:

- You can also define experiment-specific transforms directly in `scripts/train.py`
  as long as they are registered before the config is loaded and the trainer is
  built:

  ```python
  from post_training.data.transforms import register_transform

  @register_transform("my_experiment_transform")
  def my_experiment_transform(example: dict[str, Any]) -> dict[str, Any]:
      # Custom logic here...
      return {"messages": example["messages"]}
  ```

- Then reference it from YAML or via CLI:

  ```bash
  scripts/train.py \
    --config configs/sft.yaml \
    data.datasets[0].transform=my_experiment_transform
  ```

### `deepspeed`

- `deepspeed.config_path`: path to DeepSpeed YAML.
  - `configs/deepspeed/zero2.yaml` (ZeRO-2)
  - `configs/deepspeed/zero3.yaml` (ZeRO-3)

### `accelerate`

These values are used by the SLURM launcher/template and can guide your `accelerate launch` flags:

- `accelerate.mixed_precision`
- `accelerate.dynamo_backend`
- `accelerate.use_deepspeed`
- `accelerate.deepspeed_multinode_launcher`
- `accelerate.same_network`
- `accelerate.rdzv_backend`

### `logging`

- `logging.report_to`: list of integrations (e.g. `['wandb', 'tensorboard']`) or set to `['none']`.
- `logging.wandb_project`
- `logging.logging_steps`
- `logging.include_num_input_tokens_seen`

### `slurm`

- `slurm.partition`
- `slurm.num_nodes`
- `slurm.gpus_per_node`
- `slurm.cpus_per_gpu`
- `slurm.wall_time`
- `slurm.signal_time_seconds`: triggers signal-based requeue before wall time expires.
- `slurm.max_failures`: retry limit on failure.

Self-healing behavior (high level):

1. Sends `SIGUSR1` before wall time.
2. Requeues so training resumes from latest checkpoint.
3. Resubmits on failure up to `slurm.max_failures`.

### `paths`

- `paths.output_base`: base directory for run outputs.
- `paths.debug_base`: base directory used when `debug.enabled=true`.

### `debug`

- `debug.enabled`: enables debug mode behavior.
- `debug.override_existing`: if true, can wipe an existing debug run directory before starting.


