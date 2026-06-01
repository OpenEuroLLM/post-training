#!/bin/bash
#
# Run verify_rope_swa_bug.py end-to-end:
#   1) tokenize the prompt on the host venv (one-time, deterministic)
#   2) HF forward in the training container (transformers 5.4 — buggy)
#   3) vLLM forward in the eval container (vllm 0.19 — correct)
#   4) compare per-position logprobs, print verdict
#
# Submit from any login node with:
#   sbatch post-training/scripts/diagnostics/run_verify_rope_swa_bug.sh
#
# Uses boost_qos_dbg (30 min, 1 node, 1 GPU). The actual GPU compute is
# usually <10 min total. The script auto-creates a workdir under
# .../diagnostics/rope_swa_verification/<jobid>/ so multiple runs don't
# clobber each other.

#SBATCH --job-name=verify-rope-swa
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=oellm_prod2026
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO=/leonardo_scratch/large/userexternal/knikolao/post_trainig_olmo_trl
SCRIPT=$REPO/post-training/scripts/diagnostics/verify_rope_swa_bug.py
WORKDIR=$REPO/post-training/scripts/diagnostics/rope_swa_verification/${SLURM_JOB_ID:-local}
mkdir -p "$WORKDIR"

# Offline mode so neither phase tries to reach huggingface.co
export HF_HOME=/leonardo_scratch/large/userexternal/knikolao/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TRAINING_SIF=/leonardo_work/OELLM_prod2026/container_images/post-training-flash-attn-3.sif
EVAL_SIF=/leonardo_scratch/large/userexternal/knikolao/containers/vllm-v0.19.0.sif

# Common binds so both containers can read the HF cache + this repo
BINDS="--bind /leonardo_scratch/large/userexternal/knikolao:/leonardo_scratch/large/userexternal/knikolao"
BINDS="$BINDS --bind /leonardo_work/OELLM_prod2026:/leonardo_work/OELLM_prod2026"

ENVS="--env HF_HOME=$HF_HOME"
ENVS="$ENVS --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
ENVS="$ENVS --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
ENVS="$ENVS --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

echo "============================================================"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Workdir: $WORKDIR"
echo "============================================================"

# -----------------------------------------------------------------
# Phase 1: tokenize ONCE so HF and vLLM consume byte-identical input_ids.
# We use the training container's transformers (5.4) for tokenize so the
# Olmo3Config path is exercised; the actual tokenizer behavior is the same
# in the eval container's transformers (4.57). Either works for tokenization.
# -----------------------------------------------------------------
echo ""
echo "[1/4] Tokenize prompt (training container, deterministic, identical input for both forwards)"
singularity exec $BINDS $ENVS "$TRAINING_SIF" \
    python "$SCRIPT" --phase tokenize --workdir "$WORKDIR"

# -----------------------------------------------------------------
# Phase 2: HF forward through the BUGGY path
# Training container has transformers 5.4 with the single-rotary_emb bug.
# -----------------------------------------------------------------
echo ""
echo "[2/4] HF forward (training container, transformers 5.4 — the BUGGY path)"
singularity exec --nv $BINDS $ENVS "$TRAINING_SIF" \
    python "$SCRIPT" --phase hf --workdir "$WORKDIR"

# -----------------------------------------------------------------
# Phase 3: vLLM forward through the CORRECT per-layer-RoPE path
# Eval container has vllm 0.19.0 which dispatches RoPE per layer
# (model_executor/models/olmo2.py:141-150).
# -----------------------------------------------------------------
echo ""
echo "[3/4] vLLM forward (eval container, vllm 0.19 — the CORRECT path)"
singularity exec --nv $BINDS $ENVS "$EVAL_SIF" \
    python "$SCRIPT" --phase vllm --workdir "$WORKDIR"

# -----------------------------------------------------------------
# Phase 4: compare
# Either container can do this — it's pure JSON diffing. Use the eval one.
# -----------------------------------------------------------------
echo ""
echo "[4/4] Compare"
singularity exec $BINDS $ENVS "$EVAL_SIF" \
    python "$SCRIPT" --phase compare --workdir "$WORKDIR"

echo ""
echo "============================================================"
echo "Artifacts: $WORKDIR/"
echo "  input_ids.json       (shared tokenization, fed to both forwards)"
echo "  hf_logprobs.json     (HF transformers 5.4 — buggy path)"
echo "  vllm_logprobs.json   (vllm 0.19 — correct path)"
echo "============================================================"
