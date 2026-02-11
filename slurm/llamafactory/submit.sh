#!/bin/bash

#SBATCH --job-name=llamafactory
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --account=jureap59
#SBATCH --partition=booster
#SBATCH --time=24:00:00
#SBATCH --threads-per-core=1
#SBATCH --output=logs/llamafactory_%j.out

set -euo pipefail

# --- Usage check ---
if [ $# -ne 1 ]; then
    echo "Usage: sbatch slurm/llamafactory/submit.sh <llamafactory_config_path>" >&2
    echo "Example: sbatch slurm/llamafactory/submit.sh configs/llamafactory/long-context.yaml" >&2
    exit 1
fi

CONFIG=$1

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG" >&2
    exit 1
fi

# --- Source cluster environment ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
source "${REPO_DIR}/env/jupiter.env"

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "=========================================="

# --- Distributed setup ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent hosts "$MASTER_ADDR" | awk '{print $1}')
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 2000))
echo "MASTER_ADDR:MASTER_PORT set to: ${MASTER_ADDR}:${MASTER_PORT}"

export WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))
export RANK=${SLURM_NODEID}

# NCCL settings for distributed training stability
export NCCL_IB_TIMEOUT=120
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

echo "Starting training..."
echo "NNODES=$SLURM_NNODES, GPUS_PER_NODE=$GPUS_PER_NODE, WORLD_SIZE=$WORLD_SIZE"
echo "=========================================="

srun --export=ALL --wait=60 --kill-on-bad-exit=1 \
  singularity exec --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
  --bind "${DATA_DIR}:${DATA_DIR}" \
  "$CONTAINER" \
  bash -lc "
    set -e

    # Prevent host Python packages from interfering
    export PYTHONPATH=\"\"
    export PYTHONNOUSERSITE=1

    export NODE_RANK=\"\$SLURM_NODEID\"
    export NNODES=\"\$SLURM_NNODES\"
    export NPROC_PER_NODE=${GPUS_PER_NODE}
    export HF_HOME=${HF_HOME}
    export HF_HUB_CACHE=${HF_HUB_CACHE}
    export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
    export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}
    export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
    export ALLOW_EXTRA_ARGS=1

    cd ${REPO_DIR}

    llamafactory-cli train ${CONFIG}
  "
