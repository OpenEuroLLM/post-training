#!/usr/bin/env bash
#SBATCH --job-name=data-selection-token-stats
#SBATCH --account=ehpc715
#SBATCH --partition=acc_ehpc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=data-selection-token-stats-%j.out
#SBATCH --error=data-selection-token-stats-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/gpfs/scratch/ehpc715/frei058666/tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export HF_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${VENV_PYTHON}" ]; then
    echo "Expected project Python at ${VENV_PYTHON}, but it is not executable." >&2
    exit 1
fi

echo "Running on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "Offline env: HF_OFFLINE=${HF_OFFLINE}, HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"

"${VENV_PYTHON}" scripts/generate_data_selection_token_stats.py \
    --output data-selection-datasets-stats-decontaminated.yaml
