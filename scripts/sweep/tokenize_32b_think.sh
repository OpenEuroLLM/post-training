#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=tokenize-32b-think
#SBATCH --output=outputs/slurm_logs/tokenize-32b-think_%j.out
#SBATCH --error=outputs/slurm_logs/tokenize-32b-think_%j.err

mkdir -p outputs/slurm_logs

export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
source ~/.bashrc
echo $HF_HOME
srun --ntasks=1 --gpus-per-node=1 bash -c '
module purge
module load Stages/2026 GCC/14.3.0 CUDA/13
export SHARED_MAMBA_ENV=$PROJECT/miniforge3
export ENV_NAME=post-train
export SHARED_ENV=$PROJECT/envs/$ENV_NAME
source ${SHARED_MAMBA_ENV}/bin/activate ${SHARED_ENV}
export DS_SKIP_COPY_ENV=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train.py \
    --config configs/trl/sft_throughput_32b_think.yaml \
    --tokenize-only
'
