#!/usr/bin/env bash
# Round-2 resubmit: targeted resubmission of jobs that failed in the initial sweep.
#
# Fixes applied vs the original sweep scripts:
#
#   (1) 7B strong-256n GAS < 1 bug
#         STRONG_EFF_BS=1024 with pbs=2 × 1024 GPUs → GAS=0.5 → ValueError.
#         Fix: when GAS < 1, clamp to 1 and set eff_bs = pbs × world_size.
#
#   (2) TRITON_CACHE_DIR per node (32B jobs)
#         Without this, all nodes race to update the same autotune pickle on a
#         shared NFS mount, producing spurious FileNotFoundError noise.
#         Fix: export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOBID}_${SLURM_NODEID}
#
#   (3) 32B × 32k OOM → ncclUnhandledCudaError
#         All 32k 32B jobs OOM'd: ZeRO-3 all-gather buffers (500 MB) + 32k-seq
#         activations exceed GH200 96 GB HBM.
#         Fix: zero3_lowmem.yaml shrinks prefetch bucket (50 MB) and
#         max_live_parameters (200 M), reducing peak buffer memory.
#
#   (4) 32B 16n wall-time too short
#         strong-16n-{16k,32k} hit TIME LIMIT; step time ~450 s × 50 steps = 6+ h.
#         Fix: extend wall time to 08:00:00 for 16n 32B jobs.
#
#   (5) Expanded node exclusion list
#         jpbo-061-14 (original) + jpbo-018-36 (Bus error in 7B-strong-256n-32k).
#
# Usage:
#   bash scripts/sweep/resubmit_r2.sh           # submit all failed jobs
#   bash scripts/sweep/resubmit_r2.sh --dry-run # print plan, no submit
#   bash scripts/sweep/resubmit_r2.sh --7b      # 7B jobs only
#   bash scripts/sweep/resubmit_r2.sh --32b     # 32B jobs only

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Shared settings ───────────────────────────────────────────────────────────
GPUS_PER_NODE=4
PARTITION="booster"
EXCLUDE_NODES="jpbo-061-14,jpbo-018-36"

CONFIG_7B="configs/trl/sft_throughput.yaml"
CONFIG_32B="configs/trl/sft_throughput_32b_think.yaml"
DS_ZERO2="configs/deepspeed/zero2.yaml"
DS_ZERO3="configs/deepspeed/zero3.yaml"
DS_ZERO3_LOWMEM="configs/deepspeed/zero3_lowmem.yaml"

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=0
RUN_7B=1
RUN_32B=1
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --7b)      RUN_32B=0 ;;
        --32b)     RUN_7B=0  ;;
    esac
done

mkdir -p outputs/slurm_logs

TOTAL_SUBMITTED=0

# ── Helper ────────────────────────────────────────────────────────────────────
submit() {
    # submit <job_name> <sbatch_script_body>
    local job_name=$1
    local job_script=$2

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] would submit: ${job_name}"
        return
    fi

    local job_id
    job_id=$(echo "$job_script" | sbatch --parsable)
    echo "  submitted ${job_name} → job ${job_id}"
    TOTAL_SUBMITTED=$(( TOTAL_SUBMITTED + 1 ))
}

# ─────────────────────────────────────────────────────────────────────────────
# 7B RESUBMITS (4 jobs)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RUN_7B" -eq 1 ]]; then
    echo ""
    echo "=== 7B resubmits ==="
    echo "    Config  : ${CONFIG_7B}  (ZeRO-2)"
    echo "    Fixes   : GAS<1 bug (strong-256n), expanded exclude list"
    echo ""

    # ── (a) weak-16n-32k: transient torchelastic tmp dir missing ─────────────
    # Plain retry; root cause was a missing /e/scratch tmp dir on one run.
    (
    nodes=16; seq=32768; pbs=2; gas=2; eff_bs=$(( pbs * gas * nodes * GPUS_PER_NODE ))
    world_size=$(( nodes * GPUS_PER_NODE ))
    job_name="r2-olmo3-7b-weak-${nodes}n-32k"
    submit "$job_name" "$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=${job_name}
#SBATCH --output=outputs/slurm_logs/${job_name}_%j.out
#SBATCH --error=outputs/slurm_logs/${job_name}_%j.err
#SBATCH --exclude=${EXCLUDE_NODES}

export WANDB_RUN_GROUP="sft-throughput-7b-weak"
export WANDB_TAGS="scaling=weak,nodes=${nodes},gpus=${world_size},seq_len=${seq},pbs=${pbs},gas=${gas},eff_bs=${eff_bs}"
export MASTER_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=\$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=\$PROJECT/envs/\$ENV_NAME && source \${SHARED_MAMBA_ENV}/bin/activate \${SHARED_ENV} && export DS_SKIP_COPY_ENV=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --dynamo_backend no --rdzv_backend static scripts/train.py --config ${CONFIG_7B} run_name=${job_name} sft.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${eff_bs} deepspeed.config_path=${DS_ZERO2} slurm.num_nodes=${nodes} logging.wandb_project=sft-throughput-7b-weak'
EOF
)"
    )

    # ── (b) weak-256n-32k: task launch timeout (255/256 tasks started) ────────
    # Root cause: one bad node couldn't start srun task. exclude list.
    (
    nodes=256; seq=32768; pbs=1; gas=2; eff_bs=$(( pbs * gas * nodes * GPUS_PER_NODE ))
    world_size=$(( nodes * GPUS_PER_NODE ))
    job_name="r2-olmo3-7b-weak-${nodes}n-32k"
    submit "$job_name" "$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --job-name=${job_name}
#SBATCH --output=outputs/slurm_logs/${job_name}_%j.out
#SBATCH --error=outputs/slurm_logs/${job_name}_%j.err
#SBATCH --exclude=${EXCLUDE_NODES}

export WANDB_RUN_GROUP="sft-throughput-7b-weak"
export WANDB_TAGS="scaling=weak,nodes=${nodes},gpus=${world_size},seq_len=${seq},pbs=${pbs},gas=${gas},eff_bs=${eff_bs}"
export MASTER_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=\$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=\$PROJECT/envs/\$ENV_NAME && source \${SHARED_MAMBA_ENV}/bin/activate \${SHARED_ENV} && export DS_SKIP_COPY_ENV=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --dynamo_backend no --rdzv_backend static scripts/train.py --config ${CONFIG_7B} run_name=${job_name} sft.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${eff_bs} deepspeed.config_path=${DS_ZERO2} slurm.num_nodes=${nodes} logging.wandb_project=sft-throughput-7b-weak'
EOF
)"
    )

    # ── (c) strong-256n-16k: GAS < 1 bug ──────────────────────────────────────
    # Original: STRONG_EFF_BS=1024, pbs=2, world_size=1024 → GAS=0.5 → ValueError.
    # Fix: clamp GAS=1 and set eff_bs = pbs × world_size = 2048.
    (
    nodes=256; seq=16384; pbs=2
    world_size=$(( nodes * GPUS_PER_NODE ))
    gas=1
    eff_bs=$(( pbs * gas * world_size ))   # 2 × 1 × 1024 = 2048
    job_name="r2-olmo3-7b-strong-${nodes}n-16k"
    submit "$job_name" "$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --job-name=${job_name}
#SBATCH --output=outputs/slurm_logs/${job_name}_%j.out
#SBATCH --error=outputs/slurm_logs/${job_name}_%j.err
#SBATCH --exclude=${EXCLUDE_NODES}

export WANDB_RUN_GROUP="sft-throughput-7b-strong"
export WANDB_TAGS="scaling=strong,nodes=${nodes},gpus=${world_size},seq_len=${seq},pbs=${pbs},gas=${gas},eff_bs=${eff_bs}"
export MASTER_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=\$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=\$PROJECT/envs/\$ENV_NAME && source \${SHARED_MAMBA_ENV}/bin/activate \${SHARED_ENV} && export DS_SKIP_COPY_ENV=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --dynamo_backend no --rdzv_backend static scripts/train.py --config ${CONFIG_7B} run_name=${job_name} sft.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${eff_bs} deepspeed.config_path=${DS_ZERO2} slurm.num_nodes=${nodes} logging.wandb_project=sft-throughput-7b-strong'
EOF
)"
    )

    # ── (d) strong-256n-32k: NFS stale file handle on jpbo-018-36 ─────────────
    # jpbo-018-36 now excluded. Plain retry otherwise.
    (
    nodes=256; seq=32768; pbs=1
    world_size=$(( nodes * GPUS_PER_NODE ))
    gas=1
    eff_bs=$(( pbs * gas * world_size ))   # GAS=1 (STRONG_EFF_BS/world_size < 1 anyway)
    job_name="r2-olmo3-7b-strong-${nodes}n-32k"
    submit "$job_name" "$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --job-name=${job_name}
#SBATCH --output=outputs/slurm_logs/${job_name}_%j.out
#SBATCH --error=outputs/slurm_logs/${job_name}_%j.err
#SBATCH --exclude=${EXCLUDE_NODES}

export WANDB_RUN_GROUP="sft-throughput-7b-strong"
export WANDB_TAGS="scaling=strong,nodes=${nodes},gpus=${world_size},seq_len=${seq},pbs=${pbs},gas=${gas},eff_bs=${eff_bs}"
export MASTER_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)

srun bash -c 'source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=\$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=\$PROJECT/envs/\$ENV_NAME && source \${SHARED_MAMBA_ENV}/bin/activate \${SHARED_ENV} && export DS_SKIP_COPY_ENV=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --dynamo_backend no --rdzv_backend static scripts/train.py --config ${CONFIG_7B} run_name=${job_name} sft.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${eff_bs} deepspeed.config_path=${DS_ZERO2} slurm.num_nodes=${nodes} logging.wandb_project=sft-throughput-7b-strong'
EOF
)"
    )
fi

# ─────────────────────────────────────────────────────────────────────────────
# 32B RESUBMITS (13 jobs)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RUN_32B" -eq 1 ]]; then
    echo ""
    echo "=== 32B resubmits ==="
    echo "    Config      : ${CONFIG_32B}  (ZeRO-3)"
    echo "    32k DS cfg  : ${DS_ZERO3_LOWMEM}  (reduced all-gather buffers)"
    echo "    16k DS cfg  : ${DS_ZERO3}  (standard)"
    echo "    Fixes       : TRITON_CACHE_DIR, lowmem ZeRO-3 for 32k, extended wall-time for 16n"
    echo ""

    # Shared srun prefix for all 32B jobs (includes TRITON fix)
    SRUN_32B_PREFIX='source ~/.bashrc && module purge && module load Stages/2026 GCC/14.3.0 CUDA/13 && export SHARED_MAMBA_ENV=$PROJECT/miniforge3 && export ENV_NAME=post-train && export SHARED_ENV=$PROJECT/envs/$ENV_NAME && source ${SHARED_MAMBA_ENV}/bin/activate ${SHARED_ENV} && export DS_SKIP_COPY_ENV=1 && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOBID}_${SLURM_NODEID}'

    submit_32b() {
        local scaling=$1 nodes=$2 seq=$3 pbs=$4 gas=$5 eff_bs=$6 ds_cfg=$7 wall_time=$8
        local world_size=$(( nodes * GPUS_PER_NODE ))
        local seq_tag="$(( seq / 1024 ))k"
        local job_name="r2-olmo3-32b-${scaling}-${nodes}n-${seq_tag}"
        local wandb_project="sft-throughput-32b-${scaling}"
        local wandb_group="sft-throughput-32b-think-${scaling}"

        submit "$job_name" "$(cat <<EOF
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

srun bash -c '${SRUN_32B_PREFIX} && accelerate launch --num_processes ${world_size} --num_machines ${nodes} --machine_rank \$SLURM_NODEID --main_process_ip \$MASTER_IP --main_process_port 29500 --mixed_precision bf16 --dynamo_backend no --rdzv_backend static scripts/train.py --config ${CONFIG_32B} run_name=${job_name} sft.max_seq_length=${seq} training.per_device_train_batch_size=${pbs} training.effective_batch_size=${eff_bs} deepspeed.config_path=${ds_cfg} slurm.num_nodes=${nodes} logging.wandb_project=${wandb_project}'
EOF
)"
    }

    # ── 32B × 32k: OOM → use zero3_lowmem.yaml ───────────────────────────────
    # All 32k jobs OOM'd: ncclUnhandledCudaError wrapping 'out of memory'.
    # zero3_lowmem reduces prefetch bucket 500M→50M and max_live_params 1B→200M.
    echo "  -- 32k OOM retries (zero3_lowmem) --"
    for nodes in 32 64 128 256; do
        seq=32768; pbs=1
        world_size=$(( nodes * GPUS_PER_NODE ))
        # weak: GAS=2
        gas=2; eff_bs=$(( pbs * gas * world_size ))
        submit_32b "weak" "$nodes" "$seq" "$pbs" "$gas" "$eff_bs" "$DS_ZERO3_LOWMEM" "03:00:00"
        # strong: GAS = 1024/world_size (min 1)
        gas=$(( 1024 / world_size )); [[ "$gas" -lt 1 ]] && gas=1
        eff_bs=$(( pbs * gas * world_size ))
        submit_32b "strong" "$nodes" "$seq" "$pbs" "$gas" "$eff_bs" "$DS_ZERO3_LOWMEM" "03:00:00"
    done

    # ── 32B 16n: TIME LIMIT → extend wall time ────────────────────────────────
    # strong-16n-16k hit limit at 25/50 steps (~225 s/step × 50 = 3 h).
    # strong-16n-32k hit limit at 14/50 steps (~450 s/step × 50 = 6.25 h).
    # Giving 8 h covers both with margin.
    echo "  -- 16n time-limit retries (wall 08:00:00) --"
    for seq in 16384 32768; do
        nodes=16; pbs=1; world_size=$(( nodes * GPUS_PER_NODE ))
        # strong only (weak-16n succeeded)
        gas=$(( 1024 / world_size )); [[ "$gas" -lt 1 ]] && gas=1
        eff_bs=$(( pbs * gas * world_size ))
        ds_cfg="$DS_ZERO3"
        [[ "$seq" -eq 32768 ]] && ds_cfg="$DS_ZERO3_LOWMEM"
        submit_32b "strong" "$nodes" "$seq" "$pbs" "$gas" "$eff_bs" "$ds_cfg" "08:00:00"
    done

    # ── 32B NFS/node failures: plain retries ──────────────────────────────────
    # strong-64n-16k: stale file handle on wandb tmp
    # weak-256n-16k:  ModuleNotFoundError on jpbo-011-05 (IPv6 network error)
    # (weak-32n-32k is already covered above in the 32k OOM section)
    echo "  -- NFS/node failure retries --"

    nodes=64; seq=16384; pbs=1; world_size=$(( nodes * GPUS_PER_NODE ))
    gas=$(( 1024 / world_size )); [[ "$gas" -lt 1 ]] && gas=1
    eff_bs=$(( pbs * gas * world_size ))
    submit_32b "strong" "$nodes" "$seq" "$pbs" "$gas" "$eff_bs" "$DS_ZERO3" "02:30:00"

    nodes=256; seq=16384; pbs=1; world_size=$(( nodes * GPUS_PER_NODE ))
    gas=2; eff_bs=$(( pbs * gas * world_size ))
    submit_32b "weak" "$nodes" "$seq" "$pbs" "$gas" "$eff_bs" "$DS_ZERO3" "04:00:00"
fi

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] No jobs submitted."
else
    echo "Done. Submitted ${TOTAL_SUBMITTED} jobs."
fi
