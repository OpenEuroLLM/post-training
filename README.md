# Post-training

This repo supports two training backends:
- **TRL** -- SFT and DPO via `accelerate launch`
- **LlamaFactory** -- SFT, DPO, long-context, reasoning-SFT via Singularity containers

## Project Structure

```
post-training/
├── configs/
│   ├── trl/                      # TRL configs
│   │   ├── accelerate/
│   │   ├── sft/
│   │   └── dpo/
│   └── llamafactory/             # LlamaFactory configs
│       └── long-context.yaml
├── containers/                   # Singularity/Apptainer definition files
│   └── llamafactory_jupiter.def
├── env/                          # Cluster-specific environment files
│   └── jupiter.env
├── scripts/
│   └── trl/                      # TRL helper scripts
│       ├── data/
│       └── train/
└── slurm/
    ├── trl/                      # TRL SLURM scripts
    │   ├── download_dataset.sh
    │   └── submit_multinode.sh
    └── llamafactory/             # LlamaFactory SLURM scripts
        └── submit.sh
```

---

## TRL Workflows

### Prerequisites

Ensure you have uv installed on your system. If you don't have it yet, you can install it following the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install trl==0.27.0
```

### Running SFT

```bash
sbatch slurm/trl/submit_multinode.sh sft configs/trl/accelerate/fsdp2.yaml configs/trl/sft/tulu3.yaml
```

---

## LlamaFactory Workflows

LlamaFactory is installed from source inside the Singularity container at build time.

### Building the Container

```bash
singularity build --fakeroot /e/scratch/jureap59/raj3/containers/llamafactory_jupiter.sif containers/llamafactory_jupiter.def
```

### Submitting a Job

```bash
sbatch slurm/llamafactory/submit.sh configs/llamafactory/long-context.yaml
```

Override SLURM defaults at submission time:

```bash
sbatch --nodes=4 --time=48:00:00 slurm/llamafactory/submit.sh configs/llamafactory/long-context.yaml
```

### Adding a New Workflow

1. Create a YAML config under `configs/llamafactory/`.
2. Submit with `sbatch slurm/llamafactory/submit.sh configs/llamafactory/<config>.yaml`.

No changes to the SLURM script or container are needed -- only the config differs between workflows.

### Cluster Environment

Cluster-specific paths (container location, HF cache, bind-mount dirs) are defined in `env/jupiter.env`. To use a different cluster, create a new env file (e.g. `env/other_cluster.env`) and update the source line in the SLURM script.
