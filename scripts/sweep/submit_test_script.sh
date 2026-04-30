#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=sft-tp-1n-4k
#SBATCH --output=outputs/slurm_logs/sft-tp-1n-4k_%j.out
#SBATCH --error=outputs/slurm_logs/sft-tp-1n-4k_%j.err

export WANDB_RUN_GROUP="sft-throughput-sweep"
export WANDB_TAGS="nodes=1,gpus=4,seq_len=4096,pbs=8,gas=1,eff_bs=32"

export MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=$PROJECT/envs/$ENV_NAME && source ${SHARED_MAMBA_ENV}/bin/activate ${SHARED_ENV} && accelerate launch --num_processes 4 --num_machines 1 --machine_rank $SLURM_NODEID --main_process_ip $MASTER_IP --main_process_port 29500 --mixed_precision bf16 --rdzv_backend static scripts/train.py --config configs/trl/sft_throughput.yaml run_name=sft-tp-1n-4k sft.max_seq_length=4096 training.per_device_train_batch_size=8 training.effective_batch_size=32 deepspeed.config_path=configs/deepspeed/zero2.yaml slurm.num_nodes=1'