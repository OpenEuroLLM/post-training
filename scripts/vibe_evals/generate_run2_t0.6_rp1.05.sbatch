#!/bin/bash
# Response generation for ablation run 2 with temperature 0.6 and repetition penalty 1.05 (LUMI, one MI250X GCD).
# Submit from the repository root: sbatch scripts/vibe_evals/generate_run2_t0.6_rp1.05.sbatch

#SBATCH --job-name=vibe-evals-run2-t0.6-rp1.05
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=vibe-evals-run2-t0.6-rp1.05-%j.out

set -euo pipefail

CKPT="/scratch/project_465002530/users/krishnak/mn5-checkpoints/sft-ablation-run2-no-think/inference_checkpoints/step-28000"
OUTPUT="results/multilingual_responses_run2_t0.6_rp1.05.json"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CONTAINER="${TRL_CONTAINER:-/scratch/project_465002530/containers/post-training-rocm7.2.4-py3.12-torch2.9.1-trl1.7.0-olmo-patched.sif}"

cd "$REPO"
[ -f scripts/generate_responses.py ] || { echo "Submit this job from the repository root."; exit 1; }
[ -d "$CKPT" ] || { echo "Checkpoint directory not found: $CKPT"; exit 1; }
[ ! -e "$OUTPUT" ] || { echo "Output file exists already: $OUTPUT"; exit 1; }

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Checkpoint: $CKPT"
echo "Output:     $OUTPUT"

srun singularity exec \
    --bind "$REPO" \
    --bind "$CKPT" \
    --env PYTHONPATH="$REPO/src" \
    --env HF_HUB_OFFLINE=1 \
    "$CONTAINER" \
    python -u scripts/generate_responses.py \
        --model "$CKPT" \
        --prompts data/eval_prompts/multilingual_prompts.json \
        --output "$OUTPUT" \
        --temperature 0.6 \
        --repetition-penalty 1.05

echo "Job finished at $(date)"
