#!/usr/bin/env bash
# DPO throughput sweep: 4 node counts × 4 seq lengths = 16 SLURM jobs.
#
# Effective batch size is fixed at 128. Per-device batch size is memory-
# constrained (halves as seq length doubles) and capped so GAS >= 1.
# Each run logs nodes/seq/pbs to W&B via WANDB_TAGS and WANDB_RUN_GROUP.
#
# Usage:
#   bash scripts/sweep/sweep_throughput.sh              # submit all 16 jobs
#   bash scripts/sweep/sweep_throughput.sh --dry-run    # print job scripts, no submit

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Sweep parameters ──────────────────────────────────────────────────────────
NODES_LIST=(1 2 4 8)
SEQ_LENGTHS=(2048 4096 8192 16384)
MAX_PBS=(8 4 2 1)           # max per-device batch size for each seq length
EFFECTIVE_BS=128
GPUS_PER_NODE=4
WALL_TIME="00:45:00"
PARTITION="booster"
CONFIG="configs/trl/dpo_throughput.yaml"
DEEPSPEED_CONFIG="configs/deepspeed/zero3.yaml"
WANDB_GROUP="dpo-throughput-sweep"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p outputs/slurm_logs

# ── Helpers ───────────────────────────────────────────────────────────────────
compute_pbs() {
    local max_pbs=$1 world_size=$2
    local cap=$(( EFFECTIVE_BS / world_size ))
    local pbs=$(( max_pbs < cap ? max_pbs : cap ))
    echo $(( pbs < 1 ? 1 : pbs ))
}

compute_gas() {
    local pbs=$1 world_size=$2
    echo $(( EFFECTIVE_BS / (pbs * world_size) ))
}

# ── Print plan ────────────────────────────────────────────────────────────────
total=$(( ${#NODES_LIST[@]} * ${#SEQ_LENGTHS[@]} ))
echo ""
echo "=== DPO Throughput Sweep (ZeRO-3) ==="
echo "    Effective batch size : ${EFFECTIVE_BS} (fixed)"
echo "    Wall time per job    : ${WALL_TIME}"
echo "    Total jobs           : ${total}"
echo ""
printf "  %-5s  %-5s  %-7s  %-4s  %-4s  %s\n" "Nodes" "GPUs" "SeqLen" "PBS" "GAS" "Job Name"
printf "  %-5s  %-5s  %-7s  %-4s  %-4s  %s\n" "-----" "-----" "-------" "----" "----" "--------"

for nodes in "${NODES_LIST[@]}"; do
    world_size=$(( nodes * GPUS_PER_NODE ))
    for i in "${!SEQ_LENGTHS[@]}"; do
        seq=${SEQ_LENGTHS[$i]}
        pbs=$(compute_pbs "${MAX_PBS[$i]}" "$world_size")
        gas=$(compute_gas "$pbs" "$world_size")
        seq_tag="$(( seq / 1024 ))k"
        job_name="dpo-tp-${nodes}n-${seq_tag}"
        printf "  %-5s  %-5s  %-7s  %-4s  %-4s  %s\n" \
            "$nodes" "$world_size" "$seq" "$pbs" "$gas" "$job_name"
    done
done
echo ""

if [[ "$DRY_RUN" -eq 0 ]]; then
    read -rp "Submit all ${total} jobs? [y/N]: " confirm
    [[ "${confirm,,}" =~ ^(y|yes)$ ]] || { echo "Aborted."; exit 1; }
    echo ""
fi

# ── Submit ────────────────────────────────────────────────────────────────────
for nodes in "${NODES_LIST[@]}"; do
    world_size=$(( nodes * GPUS_PER_NODE ))
    for i in "${!SEQ_LENGTHS[@]}"; do
        seq=${SEQ_LENGTHS[$i]}
        pbs=$(compute_pbs "${MAX_PBS[$i]}" "$world_size")
        gas=$(compute_gas "$pbs" "$world_size")
        seq_tag="$(( seq / 1024 ))k"
        job_name="dpo-tp-${nodes}n-${seq_tag}"

        job_script=$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --time=${WALL_TIME}
#SBATCH --job-name=${job_name}
#SBATCH --output=outputs/slurm_logs/${job_name}_%j.out
#SBATCH --error=outputs/slurm_logs/${job_name}_%j.err

export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_TAGS="nodes=${nodes},gpus=${world_size},seq_len=${seq},pbs=${pbs},gas=${gas},eff_bs=${EFFECTIVE_BS}"

# Compute master IP in the batch script header where SLURM_JOB_NODELIST is
# guaranteed to be set, then export so srun tasks inherit it.
export MASTER_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=\$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=\$PROJECT/envs/\$ENV_NAME && source \${SHARED_MAMBA_ENV}/bin/activate \${SHARED_ENV} && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --rdzv_backend static scripts/train.py --config ${CONFIG} run_name=${job_name} dpo.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${EFFECTIVE_BS} deepspeed.config_path=${DEEPSPEED_CONFIG} slurm.num_nodes=${nodes}'
EOF
)

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "=== ${job_name} ==="
            echo "$job_script"
            echo ""
        else
            job_id=$(echo "$job_script" | sbatch --parsable)
            echo "Submitted ${job_name} → job ${job_id}"
        fi
    done
done

[[ "$DRY_RUN" -eq 1 ]] && echo "[dry-run] No jobs submitted." || echo "Done."
