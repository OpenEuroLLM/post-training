#!/bin/bash

#SBATCH --job-name=multinode-sft
#SBATCH -D .
#SBATCH --partition=xxx
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=16          # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/O-%x.%j
#SBATCH --error=logs/E-%x.%j

set -x -e

if [ $# -ne 3 ]; then
    echo "Error: Expected 3 arguments (task, accelerate config, script config file)" >&2
    exit 1
fi

TASK=$1
ACCELERATE_CONFIG=$2
CONFIG_FILE=$3

# Check if the task is either "sft" or "dpo"
if [[ "$TASK" != "sft" && "$TASK" != "dpo" ]]; then
    echo "Error: Task must be either 'sft' or 'dpo'" >&2
    exit 1
fi

######################
### Set environment ###
######################
source .venv/bin/activate
export GPUS_PER_NODE=4
######################

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=9001
######################

export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend static \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    "

export SCRIPT="scripts/${TASK}.py"
export SCRIPT_ARGS="--config $CONFIG_FILE"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
echo $CMD

echo "START TIME: $(date)"
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
echo "END TIME: $(date)"