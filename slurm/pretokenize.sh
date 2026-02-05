#!/bin/bash

#SBATCH --job-name=pretokenize
#SBATCH -D .
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/O-%x.%j
#SBATCH --error=logs/E-%x.%j

set -x -e

if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument (config path)" >&2
    echo "Usage: sbatch pretokenize.sh <config_path>" >&2
    echo "Example: sbatch pretokenize.sh configs/datasets/tulu3.yaml" >&2
    exit 1
fi

CONFIG=$1

######################
### Set environment ###
######################
source .venv/bin/activate
######################

echo "START TIME: $(date)"
echo "Pretokenizing with config: $CONFIG using $SLURM_CPUS_PER_TASK processes"

python scripts/data/pretokenize.py "$CONFIG" --num-proc "$SLURM_CPUS_PER_TASK"

echo "END TIME: $(date)"
