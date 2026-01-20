# LLM Post-training With TRL

## Prerequisites

Ensure you have conda installed on your system. If you don't have it yet, you can install Miniconda or Anaconda following the [official conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Installation

Create and activate a conda environment, then install TRL using pip:

```bash
# Create a new conda environment with Python
conda create -n trl python=3.14

# Activate the environment
conda activate trl

# Install TRL using pip
pip install trl==0.27.0
```

## Running SFT

First, create the config file for your finetuning run in `configs/sft`. See `configs/sft/tulu3.yaml` for reference.

To submit a multi-node job using FSDP v2, first update `submit_multinode` with your run configurations (partition, number of gpus, number of nodes, $GPUS_PER_NODE, and environment settings). Then, submit the job:
```
# sbatch slurm/submit_multinode.sh sft <accelerate_config> <sft_config>
sbatch slurm/submit_multinode.sh sft configs/accelerate/fsdp2.yaml configs/sft/tulu3.yaml
```

More configs for `accelerate` can be generated using `accelerate config` and responding to the prompts. For example:
```
cd configs/accelerate
accelerate config --config_file my_accelerate_config.yaml
```
