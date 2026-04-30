#!/usr/bin/env bash
# OLMo-3 32B Think SFT throughput sweep: 5 node counts × 2 seq lengths × 2 scaling modes.
#
# Weak scaling  (--weak):   pbs=1, GAS=2 fixed → eff_bs doubles with world size.
# Strong scaling (--strong): pbs=1, GAS varies → eff_bs=1024 fixed across all node counts.
# Default: both modes are submitted.
#
# Usage:
#   bash scripts/sweep/sweep_32b_think_throughput.sh              # submit all jobs (both modes)
#   bash scripts/sweep/sweep_32b_think_throughput.sh --weak       # weak scaling only
#   bash scripts/sweep/sweep_32b_think_throughput.sh --strong     # strong scaling only
#   bash scripts/sweep/sweep_32b_think_throughput.sh --dry-run    # print, no submit

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Sweep parameters ──────────────────────────────────────────────────────────
NODES_LIST=(16 32 64 128 256)
SEQ_LENGTHS=(16384 32768)
GPUS_PER_NODE=4
PARTITION="booster"

# Wall time scales with node count (32B init is slow at large scale)
wall_time_for_nodes() {
    case "$1" in
        16|32)  echo "02:00:00" ;;
        64)     echo "02:30:00" ;;
        128)    echo "03:00:00" ;;
        256)    echo "04:00:00" ;;
        *)      echo "03:00:00" ;;
    esac
}
EXCLUDE_NODES="jpbo-061-14"           # node with stale NFS handle
CONFIG="configs/trl/sft_throughput_32b_think.yaml"
DEEPSPEED_CONFIG="configs/deepspeed/zero3.yaml"

# Weak scaling: GAS fixed, eff_bs = pbs * GAS * world_size
WEAK_GAS=2

# Strong scaling: eff_bs fixed, GAS = STRONG_EFF_BS / world_size
STRONG_EFF_BS=1024

# ── Argument parsing ──────────────────────────────────────────────────────────
RUN_WEAK=1
RUN_STRONG=1
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --weak)    RUN_STRONG=0 ;;
        --strong)  RUN_WEAK=0 ;;
        --dry-run) DRY_RUN=1 ;;
    esac
done

mkdir -p outputs/slurm_logs

# ── Helpers ───────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo "=== OLMo-3 32B Think SFT Throughput Sweep (ZeRO-3) ==="
    echo "    Excluded nodes  : ${EXCLUDE_NODES}"
    echo "    Wall time       : scales with node count"
    echo ""
    printf "  %-8s  %-5s  %-5s  %-7s  %-4s  %-4s  %-8s  %-10s  %s\n" "Scaling" "Nodes" "GPUs" "SeqLen" "PBS" "GAS" "Eff BS" "Wall Time" "Job Name"
    printf "  %-8s  %-5s  %-5s  %-7s  %-4s  %-4s  %-8s  %-10s  %s\n" "--------" "-----" "-----" "-------" "----" "----" "------" "---------" "--------"
}

submit_job() {
    local scaling=$1 nodes=$2 seq=$3 pbs=$4 gas=$5 eff_bs=$6
    local world_size=$(( nodes * GPUS_PER_NODE ))
    local seq_tag="$(( seq / 1024 ))k"
    local job_name="olmo3-32b-${scaling}-${nodes}n-${seq_tag}"
    local wandb_group="sft-throughput-32b-think-${scaling}"
    local wandb_project="sft-throughput-32b-${scaling}"
    local wall_time
    wall_time=$(wall_time_for_nodes "$nodes")

    printf "  %-8s  %-5s  %-5s  %-7s  %-4s  %-4s  %-8s  %-10s  %s\n" \
        "$scaling" "$nodes" "$world_size" "$seq" "$pbs" "$gas" "$eff_bs" "$wall_time" "$job_name"

    [[ "$DRY_RUN" -eq 1 ]] && return

    local job_script
    job_script=$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --time=${wall_time}
#SBATCH --job-name=${job_name}
#SBATCH --output=outputs/slurm_logs/${job_name}_%j.out
#SBATCH --error=outputs/slurm_logs/${job_name}_%j.err
#SBATCH --exclude=${EXCLUDE_NODES}

export WANDB_RUN_GROUP="${wandb_group}"
export WANDB_TAGS="scaling=${scaling},nodes=${nodes},gpus=${world_size},seq_len=${seq},pbs=${pbs},gas=${gas},eff_bs=${eff_bs}"

export MASTER_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=\$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=\$PROJECT/envs/\$ENV_NAME && source \${SHARED_MAMBA_ENV}/bin/activate \${SHARED_ENV} && export DS_SKIP_COPY_ENV=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --dynamo_backend no --rdzv_backend static scripts/train.py --config ${CONFIG} run_name=${job_name} sft.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${eff_bs} deepspeed.config_path=${DEEPSPEED_CONFIG} slurm.num_nodes=${nodes} logging.wandb_project=${wandb_project}'
EOF
)
    local job_id
    job_id=$(echo "$job_script" | sbatch --parsable)
    echo "    → submitted job ${job_id}"
}

# ── Plan & submit ─────────────────────────────────────────────────────────────
total=0
[[ "$RUN_WEAK"   -eq 1 ]] && total=$(( total + ${#NODES_LIST[@]} * ${#SEQ_LENGTHS[@]} ))
[[ "$RUN_STRONG" -eq 1 ]] && total=$(( total + ${#NODES_LIST[@]} * ${#SEQ_LENGTHS[@]} ))

echo "    Total jobs      : ${total}"
print_header

for nodes in "${NODES_LIST[@]}"; do
    world_size=$(( nodes * GPUS_PER_NODE ))
    for seq in "${SEQ_LENGTHS[@]}"; do
        if [[ "$RUN_WEAK" -eq 1 ]]; then
            eff_bs=$(( 1 * WEAK_GAS * world_size ))
            submit_job "weak" "$nodes" "$seq" 1 "$WEAK_GAS" "$eff_bs"
        fi
        if [[ "$RUN_STRONG" -eq 1 ]]; then
            gas=$(( STRONG_EFF_BS / world_size ))
            [[ "$gas" -lt 1 ]] && gas=1
            submit_job "strong" "$nodes" "$seq" 1 "$gas" "$STRONG_EFF_BS"
        fi
    done
done

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] No jobs submitted."
else
    [[ "$DRY_RUN" -eq 0 ]] && echo "Done. Submitted ${total} jobs."
fi
