# OpenEuroLLM Post-Training

A shared pipeline for post-training large language models across different compute clusters as part of the OpenEuroLLM initiative. Built on top of [TRL](https://github.com/huggingface/trl), it currently supports supervised fine-tuning (SFT) and direct preference optimization (DPO) with multi-node distributed training via Hugging Face Accelerate and FSDP v2.

---

- [OpenEuroLLM Post-Training](#openeurollm-post-training)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Caching Resources](#caching-resources)
    - [Running Training](#running-training)
    - [Generating Accelerate Configs](#generating-accelerate-configs)
  - [Repository Structure](#repository-structure)
  - [Contributing](#contributing)

---

## Getting Started

### Prerequisites

Ensure you have `uv` installed on your system. If you don't have it yet, you can install it following the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation

```bash
uv sync
```

## Usage

### Caching Resources

Before running training on a cluster, download and cache all required Hugging Face resources (models, tokenizers, datasets) specified in your config:

```bash
python scripts/data/cache.py configs/sft/tulu3.yaml --workers 8
```

### Running Training

1. Create a config file for your run in `configs/sft/` or `configs/dpo/`. See `configs/sft/tulu3.yaml` for reference.

2. Update `slurm/submit_multinode.sh` with your cluster settings (partition, number of GPUs, number of nodes, and environment).

3. Submit the job:

```bash
# sbatch slurm/submit_multinode.sh <task> <accelerate_config> <training_config>
sbatch slurm/submit_multinode.sh sft configs/accelerate/fsdp2.yaml configs/sft/tulu3.yaml
sbatch slurm/submit_multinode.sh dpo configs/accelerate/fsdp2.yaml configs/dpo/tulu3.yaml
```

### Generating Accelerate Configs

```bash
cd configs/accelerate
accelerate config --config_file my_config.yaml
```

## Repository Structure

```
configs/
  accelerate/   - Distributed training configs (FSDP v2, etc.)
  sft/          - SFT training configs
  dpo/          - DPO training configs
scripts/
  train/        - Training scripts (sft.py, dpo.py)
  data/         - Data utilities (caching, downloading)
slurm/          - Slurm job submission scripts
```

## Contributing

This repository is meant to be a **single, well-maintained pipeline** with a consistent interface for all post-training workflows. Before adding code, please consider:

- **Does it fit the existing structure?** New training methods, configs, and data utilities are welcome as long as they follow the patterns already in place.
- **Is it general-purpose?** One-off experiments, personal helper scripts, and cluster-specific hacks should live elsewhere. Only add code that other contributors can reuse.
- **Does it work with the existing tooling?** Training scripts should be launchable through the same Slurm and Accelerate workflow. Configs should follow the established YAML format.

Proposals for new training paradigms or improvements to existing workflows are always welcome — open an issue to discuss before submitting a PR.
