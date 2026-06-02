#!/bin/bash
# Cross-container RoPE-on-SWA divergence diagnostic.
#
# Runs the same Olmo-3-7B-Instruct-SFT model + same tokenized input through
# both containers and diffs the per-position log-probabilities. The only
# structural difference between the two forwards is the RoPE-on-SWA-layers
# wiring: HF (transformers 5.4) applies YaRN-scaled (cos,sin) to all 32
# layers including the 24 SWA layers; vLLM (0.19.0) gives SWA layers
# vanilla RoPE per OLMo-core's reference. Divergence beyond kernel-impl
# noise IS the bug firing.
#
# Usage:
#   sbatch scripts/diagnostics/run_check_rope_swa_divergence.sh
#
#SBATCH --account=oellm_prod2026
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=rope-swa-check
#SBATCH --output=/leonardo_scratch/large/userexternal/knikolao/post_trainig_olmo_trl/post-training/scripts/diagnostics/check_rope_swa_divergence/run/%x-%j.out

set -euo pipefail

REPO=/leonardo_scratch/large/userexternal/knikolao/post_trainig_olmo_trl/post-training
DIAG=$REPO/scripts/diagnostics/check_rope_swa_divergence

TRAIN_SIF=/leonardo_work/OELLM_prod2026/container_images/post-training-flash-attn-3.sif
EVAL_SIF=/leonardo_scratch/large/userexternal/knikolao/containers/vllm-v0.19.0.sif

BIND="--bind /leonardo_scratch/large/userexternal/knikolao:/leonardo_scratch/large/userexternal/knikolao --bind /leonardo_work/OELLM_prod2026:/leonardo_work/OELLM_prod2026"
ENV="--env HF_HOME=/leonardo_scratch/large/userexternal/knikolao/huggingface --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env PYTHONPATH=$REPO/src"

cd "$DIAG"
mkdir -p run

echo "=============================================="
echo " [1/4] Tokenize Dolci row (post-training container)"
echo "       — applies olmo3-instruct-sft chat template,"
echo "         saves input_ids to run/input.npz"
echo "=============================================="
singularity exec --nv $BIND $ENV "$TRAIN_SIF" python3 tokenize_input.py

echo
echo "=============================================="
echo " [2/5] HF forward STOCK (post-training container, transformers 5.4.0)"
echo "       — single model-level rotary_emb shared by all 32 layers (BUG)"
echo "=============================================="
singularity exec --nv $BIND $ENV "$TRAIN_SIF" python3 hf_forward.py

echo
echo "=============================================="
echo " [3/5] HF forward PATCHED (post-training container)"
echo "       — patch_olmo3_swa_rope.install():  SWA layers get vanilla RoPE,"
echo "         full-attn layers keep YaRN-scaled — mirrors vLLM/OLMo-core"
echo "=============================================="
singularity exec --nv $BIND $ENV "$TRAIN_SIF" python3 hf_forward.py \
    --apply-patch \
    --output run/hf_logits_patched.npz

echo
echo "=============================================="
echo " [4/5] vLLM forward (eval container, vllm 0.19.0)"
echo "       — per-layer rotary_emb: SWA vanilla, full-attn YaRN-scaled"
echo "=============================================="
singularity exec --nv $BIND $ENV "$EVAL_SIF" python3 vllm_forward.py

echo
echo "=============================================="
echo " [5/5] Compare (V1: stock vs vLLM,  V2: patched vs vLLM)"
echo "=============================================="
singularity exec --nv $BIND $ENV "$TRAIN_SIF" python3 compare.py

echo
echo "Diagnostic outputs in: $DIAG/run/"
