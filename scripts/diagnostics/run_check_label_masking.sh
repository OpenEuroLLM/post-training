#!/bin/bash
# Submit the label-masking diagnostic on a single Leonardo node via debug QoS.
#
# Usage:
#   cd post-training
#   mkdir -p outputs/diagnostics      # one-time; sbatch won't create it
#   sbatch scripts/diagnostics/run_check_label_masking.sh
#   sbatch scripts/diagnostics/run_check_label_masking.sh --row 42
#
# Logs land in outputs/diagnostics/slurm-<jobid>.{out,err}.
# Runs inside the production training container so the TRL/transformers
# versions match what trained the run we're diagnosing.

#SBATCH --job-name=check-label-masking
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=outputs/diagnostics/slurm-%j.out
#SBATCH --error=outputs/diagnostics/slurm-%j.err

set -euo pipefail

# SLURM copies the submission script to /var/spool before running, so
# ${BASH_SOURCE[0]} doesn't point at the repo. Use SLURM_SUBMIT_DIR (set by
# sbatch to the cwd of submission); fall back to a manual run from the repo.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

# Source HF cache paths, offline flags, etc.
# shellcheck disable=SC1091
source env/leonardo.env

CONTAINER=/leonardo_work/OELLM_prod2026/container_images/post-training-flash-attn-3.sif

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Repo dir:      ${REPO_DIR}"
echo "Container:     ${CONTAINER}"
echo "Forwarded args: $*"
echo "=========================================="

# Pass any sbatch-time positional args (e.g. "--row 42") through to the python
# script. The leading `_` becomes $0 inside `bash -lc`, so "$@" inside the
# heredoc expands to just $1, $2, ...
singularity exec --nv \
  --bind /leonardo_scratch/large/userexternal/knikolao:/leonardo_scratch/large/userexternal/knikolao \
  --bind /leonardo_work/OELLM_prod2026:/leonardo_work/OELLM_prod2026 \
  "${CONTAINER}" \
  bash -lc '
    set -euo pipefail
    export PATH=/usr/local/bin:/usr/bin:/bin
    export PYTHONPATH='"${REPO_DIR}"'/src
    export PYTHONNOUSERSITE=1
    export HF_HOME='"${HF_HOME}"'
    export HF_HUB_CACHE='"${HF_HUB_CACHE}"'
    export HUGGINGFACE_HUB_CACHE='"${HUGGINGFACE_HUB_CACHE}"'
    export HF_DATASETS_CACHE='"${HF_DATASETS_CACHE}"'
    export HF_DATASETS_OFFLINE='"${HF_DATASETS_OFFLINE}"'
    export TRANSFORMERS_OFFLINE='"${TRANSFORMERS_OFFLINE}"'
    export HF_HUB_OFFLINE='"${HF_HUB_OFFLINE}"'
    export SSL_CERT_FILE='"${SSL_CERT_FILE}"'
    cd '"${REPO_DIR}"'
    python scripts/diagnostics/check_label_masking.py "$@"
  ' _ "$@"
