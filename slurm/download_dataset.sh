#!/bin/bash

#SBATCH --job-name=download-dataset
#SBATCH -D .
#SBATCH --partition=xxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/O-%x.%j
#SBATCH --error=logs/E-%x.%j

set -x -e

if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument (dataset ID)" >&2
    echo "Usage: sbatch download_dataset.sh <dataset_id>" >&2
    echo "Example: sbatch download_dataset.sh allenai/tulu-3-sft-mixture" >&2
    exit 1
fi

DATASET=$1

######################
### Set environment ###
######################
source .venv/bin/activate
######################

echo "START TIME: $(date)"
echo "Downloading dataset: $DATASET with $SLURM_CPUS_PER_TASK workers"

python scripts/download.py "$DATASET" --workers "$SLURM_CPUS_PER_TASK"

echo "END TIME: $(date)"
