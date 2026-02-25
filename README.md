# Post-Training Framework

A modular, configuration-driven framework for **SFT** (Supervised Fine-Tuning) and **DPO** (Direct Preference Optimization). Built on **TRL**, **DeepSpeed**, and **Accelerate** with multi-node **SLURM** support.

This repo supports two training backends:
- **TRL** -- SFT and DPO via `accelerate launch`
- **LlamaFactory** -- SFT, DPO, long-context tuning via Singularity containers

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration Philosophy](#configuration-philosophy)
- [Feature Guide](#feature-guide)
  - [Training Methods](#training-methods)
  - [Data Pipeline](#data-pipeline)
  - [Training Length](#training-length)
  - [Infrastructure & Compute](#infrastructure--compute)
  - [Checkpointing](#checkpointing)
  - [Environment Modes](#environment-modes)
  - [Logging & Experiment Tracking](#logging--experiment-tracking)
- [Run Outputs & Directory Layout](#run-outputs--directory-layout)

## ⚡ Quick Start

### Installation

This project uses `uv` for dependency management. To create the Python environment, run:

```bash
uv sync
```

### Local Training (Single-Node)

To run training locally, use `accelerate launch`. You must specify the distributed flags explicitly.

#### SFT example

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
    --config configs/trl/sft.yaml \
    training.max_steps=100 \
    offline=true
```

#### DPO example

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
    --config configs/trl/dpo.yaml \
    training.max_steps=100 \
    offline=true
```

> [!NOTE]
> The `--mixed_precision` flag passed to `accelerate launch` must match `model.dtype` in your config.

### SLURM Submission (Multi-Node)

For cluster environments, use the submission script. It auto-generates a SLURM batch script based on your YAML configuration and submits it.

- SLURM job template: `src/post_training/slurm/job.sh.jinja`

```bash
python scripts/submit.py --config configs/trl/sft.yaml
```

## 📂 Project Structure

```text
post-training/
├── configs/
│   ├── trl/
│   │   └── sft.yaml              # TRL SFT example config
│   ├── llamafactory/
│   │   └── long-context.yaml     # LlamaFactory long-context SFT config
│   └── deepspeed/
│       ├── zero2.yaml            # DeepSpeed ZeRO Stage 2 config
│       ├── zero3.yaml            # DeepSpeed ZeRO Stage 3 config
│       └── z3_partial_offload.json  # ZeRO Stage 3 with CPU offloading
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
│   ├── data.py                   # Data pipeline debugger + token-stats
│   └── wb.py                  # Weights & Biases utilities
└── pyproject.toml
```

## 🛠 Configuration Philosophy

### The golden rule

All run configuration lives in a **single YAML file**.

You do not need to edit Python scripts to change hyperparameters, models, or data mixtures.

- Override any YAML value via the CLI using **dot-notation**
- Or create a new YAML config specific to your run

### Example: overriding the config via CLI

```bash
scripts/train.py \
    --config configs/trl/sft.yaml \
    model.name_or_path="meta-llama/Llama-3.1-8B" \
    training.learning_rate=5e-6 \
    sft.packing=false
```

## 🧩 Feature Guide

### 1. Training Methods

Select your training strategy using `method`.

- **SFT (Supervised Fine-Tuning)**
  - **Key**: `method: "sft"`
  - **Packing**: set `sft.packing: true` to pack multiple short examples into a single sequence (recommended for efficiency)
  - **Sequence length**: controlled by `sft.max_seq_length`

- **DPO (Direct Preference Optimization)**
  - **Key**: `method: "dpo"`
  - **Loss type**: set `dpo.loss_type` (e.g., `sigmoid`, `hinge`, `ipo`)
  - **Reference model**: set `dpo.ref_model_name_or_path`
    - If `null`, TRL creates an implicit copy of the active model
    - If using **ZeRO Stage 3**, consider specifying the reference model explicitly (implicit copy creation can be unstable with Stage 3)

### 2. Data Pipeline

The data pipeline is modularized into four distinct stages.

#### A. Dataset registry & mixing

Define multiple datasets in `data.datasets`. The loader automatically interleaves them based on the `weight` parameter (normalized automatically).

```yaml
data:
  datasets:
    - name: "my_dataset"
      path: "org/dataset"
      split: "train"
      weight: 1.0  # Mixing weight (normalized automatically)
```

#### B. Data transformations

Raw datasets often come in varying formats. Transforms normalize them into a standard `messages` list format before templating.

- **Config**: `transform: "transform_name"` (in the dataset entry)
- **Registry**: `src/post_training/data/transforms.py`
- **Customization**: decorate a function with `@register_transform("name")` to add your own logic

Example (normalize raw fields into `messages`):

```python
from post_training.data.transforms import register_transform

@register_transform("my_transform")
def my_transform(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }
```

#### C. Chat templates

Templates convert the list of messages into a single string for the model.

- **Config**: `data.chat_template: "name"`
- **Source**: Jinja files located in `src/post_training/chat_templates/templates/`

#### D. Data inspection

Use the data script to debug the pipeline stages (Raw → Transformed → Formatted → Tokenized) and to compute token statistics.

```bash
python scripts/data.py --config configs/trl/sft.yaml --show-formatted --num-samples 3
python scripts/data.py --config configs/trl/sft.yaml token-stats
```

### 3. Training Length

You must specify exactly one determining factor for training duration in the `training` section:

- **Step-based**: `training.max_steps` (fixed number of optimizer steps)
- **Sample-based**: `training.num_training_samples` (steps = `ceil(samples / global_batch_size)`)
- **Token-based**: `training.num_training_tokens` (steps based on total token count)
  - Only valid when `method: "sft"` and `sft.packing: true`

### 4. Infrastructure & Compute

- **DeepSpeed**: configured via `deepspeed.config_path` (e.g., `configs/deepspeed/zero3.yaml`)
- **Accelerate flags**: the `accelerate` section in the YAML mirrors the CLI flags required for multi-node setups (`mixed_precision`, `dynamo_backend`, `rdzv_backend`, etc.).
  These are used by the SLURM launcher to generate the correct job script.
- **Self-healing**: the SLURM launcher (`src/post_training/slurm/`) supports auto-requeueing.
  - `slurm.signal_time_seconds` ensures the job saves a checkpoint and requeues itself before the wall time expires

### 5. Checkpointing

#### Resume checkpoints (full training state)

- **What**: full training state (optimizer + model)
- **Location**: `checkpoints/checkpoint-*`
- **Logic**: training automatically resumes from the latest checkpoint found here

#### Inference checkpoints (lightweight)

- **What**: model + tokenizer only
- **Location**: `inference_checkpoints/step-*`
- **Config**: `checkpointing.inference_checkpoint_steps` (set to `null` to disable)

### 6. Environment Modes

- **Offline**: `offline: true`  
  Disables Hugging Face Hub / Weights & Biases network calls (essential for air-gapped nodes).
- **Debug**: `debug.enabled: true`  
  Forces `report_to: none`, uses a separate output directory, and allows overwriting existing runs.

### 7. Logging & Experiment Tracking

The framework supports multiple logging backends and handles offline environments (e.g., air-gapped clusters).

#### SLURM Logs
For multi-node runs, SLURM output and error logs are stored within each run's specific directory:
- `<run_directory>/slurm/slurm-<job_id>.out`: Standard output (including console logs and progress bars)
- `<run_directory>/slurm/slurm-<job_id>.err`: Standard error (including stack traces and warnings)

#### Weights & Biases (WandB)
- **Online**: Logs are streamed directly to the WandB cloud. The project name is controlled by `logging.wandb_project`.
- **Offline**: When `offline: true` is set, WandB logs are saved locally to the `wandb/` directory in the project root.

#### Syncing Offline Runs
To upload offline runs to the cloud (e.g., from a login node with internet access), use the utility script:

```bash
# Interactive mode - view and select runs to sync
python scripts/wb.py sync --interactive

# Sync a specific run by its training run name
python scripts/wb.py sync --run-name <run_name>
```

## 📦 Run Outputs & Directory Layout

Each run generates a unique directory based on `paths.output_base` (or `paths.debug_base`) and a run name auto-generated from the model, method, and dataset mix.

```text
<output_base>/<run_name>/
├── config.yaml               # Frozen configuration for reproducibility
├── checkpoints/              # Full TRL training state (resumable)
│   └── checkpoint-500/
├── inference_checkpoints/    # Lightweight model + tokenizer only
│   └── step-500/
├── logs/                     # TensorBoard / Weights & Biases logs
└── slurm/                    # SLURM artifacts
    ├── job.sh                # The generated submission script
    ├── slurm-<id>.out        # Standard output
    ├── slurm-<id>.err        # Standard error
    └── failure_count         # Tracks retries for self-healing
```

## LlamaFactory Backend

An alternative backend using [LlamaFactory](https://github.com/hiyouga/LlamaFactory) for training, running inside a Singularity container.

### Setup

1. Build the Singularity container:
   ```bash
   singularity build --fakeroot llamafactory.sif containers/llamafactory_jupiter.def
   ```
2. Set the container path in `env/jupiter.env`:
   ```bash
   export CONTAINER=/path/to/llamafactory.sif
   ```

### Long-Context SFT (example)

```bash
python scripts/submit.py --config configs/llamafactory/long-context.yaml
```

- Config: `configs/llamafactory/long-context.yaml`
- DeepSpeed: `configs/deepspeed/z3_partial_offload.json`
- Dataset registry: `data/llamafactory/dataset_info.json`

## 📘 Configuration Reference: `configs/trl/sft.yaml`

Full reference configuration for the default SFT setup:

```yaml
# ============================================================================
# SFT (Supervised Fine-Tuning) Configuration
# ============================================================================
# Override any value via CLI dot-notation:
#   accelerate launch \
#      --num_machines 1 \
#      --num_processes 4 \
#      --dynamo_backend=inductor \
#      --use_deepspeed \
#      --same_network \
#      --rdzv_backend static \
#      --mixed_precision bf16 \
#      scripts/train.py \
#      --config configs/trl/sft.yaml \
#      training.max_steps=100 \
#      offline=true
# ============================================================================

method: sft
backend: trl
run_name: null                               # auto-generated from model + datasets if null
offline: false                               # set true to disable all HuggingFace / wandb network calls

# -- Model -------------------------------------------------------------------
model:
  name_or_path: "allenai/Olmo-3-1025-7B"
  attn_implementation: "flash_attention_3"
  dtype: "bfloat16"

# -- Training hyper-parameters -----------------------------------------------
training:
  max_steps: null                            # Set explicitly, OR use num_training_samples below
  num_training_samples: null                 # If set: max_steps = ceil(num_samples / effective_batch_size)
  # num_training_tokens: null                # Only valid when sft.packing=true (max_steps = ceil(tokens / (effective_batch_size * sft.max_seq_length)))

  learning_rate: 2.0e-5
  effective_batch_size: 32                   # per_device * grad_accum * world_size
  per_device_train_batch_size: 8
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine_with_min_lr"
  lr_scheduler_kwargs:
    min_lr_rate: 0.1
  gradient_checkpointing: true
  bf16: true
  seed: 42
  use_liger_kernel: true

# -- SFT method parameters ---------------------------------------------------
sft:
  max_seq_length: 4096
  packing: true

# -- Checkpointing -----------------------------------------------------------
checkpointing:
  save_steps: 200
  save_total_limit: 2                        # Full checkpoints to keep
  inference_checkpoint_steps: 157            # Minimal inference model interval (set to null to disable)
  inference_checkpoint_path: "inference_checkpoints"   # Relative to run dir

# -- Data mix ----------------------------------------------------------------
data:
  chat_template: "olmo3"                     # Name from chat template registry
  num_proc: null                             # null = auto-detect, capped at 32
  datasets:
    - name: "nemotron_pt_v2"
      path: "nvidia/Nemotron-Post-Training-Dataset-v2"
      split: "stem"
      weight: 1.0
      transform: null                        # null = already conversational

# -- DeepSpeed ---------------------------------------------------------------
deepspeed:
  config_path: "configs/deepspeed/zero2.yaml"

# -- Accelerate launch flags (explicit multi-node control) -------------------
accelerate:
  mixed_precision: "bf16"
  use_deepspeed: true
  deepspeed_multinode_launcher: "standard"   # "standard" | "pdsh" | etc.
  same_network: true                         # All nodes on same network
  rdzv_backend: "static"                     # "static" | "c10d" | "etcd"
  dynamo_backend: "inductor"                 # "inductor" | "no" | etc.

# -- Logging & tracking ------------------------------------------------------
logging:
  report_to:
    - "wandb"
    - "tensorboard"
  wandb_project: "sft-training"
  logging_steps: 1
  include_num_input_tokens_seen: "non_padding"

# -- SLURM -------------------------------------------------------------------
slurm:
  partition: "booster"
  num_nodes: 1
  gpus_per_node: 4
  cpus_per_gpu: 32
  wall_time: "02:00:00"
  job_name: "sft-training"
  signal_time_seconds: 300                   # SIGUSR1 sent this many seconds before timeout to trigger self-healing
  max_failures: 3                            # Self-healing retry limit

# -- Debug mode --------------------------------------------------------------
debug:
  enabled: false
  override_existing: false

# -- Output paths -------------------------------------------------------------
paths:
  output_base: "outputs"
  debug_base: "outputs/debug"
```


