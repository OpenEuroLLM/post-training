#!/usr/bin/env bash
set -euo pipefail

# Download the datasets/model needed by scripts/generate_data_selection_token_stats.py
# into a shared Hugging Face cache that compute nodes can read offline.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -x "${REPO_ROOT}/.venv/bin/hf" ]; then
    HF_CLI="${REPO_ROOT}/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
elif [ -x "${HOME}/.local/bin/hf" ]; then
    HF_CLI="${HOME}/.local/bin/hf"
elif [ -x "${HOME}/bin/hf" ]; then
    HF_CLI="${HOME}/bin/hf"
else
    echo "Could not find the Hugging Face CLI 'hf' in .venv/bin, PATH, ~/.local/bin, or ~/bin." >&2
    echo "Install it in the project venv or user bin before running this script." >&2
    exit 127
fi

echo "Repo root: ${REPO_ROOT}"
echo "hf CLI: ${HF_CLI}"

download_dataset() {
    local repo_id="$1"
    echo
    echo "Downloading dataset repo: ${repo_id}"
    "${HF_CLI}" download "${repo_id}" \
        --repo-type dataset
}

download_model() {
    local repo_id="$1"
    echo
    echo "Downloading model repo: ${repo_id}"
    "${HF_CLI}" download "${repo_id}" \
        --repo-type model
}

download_model "allenai/Olmo-3-1025-7B"

download_dataset "openeurollm/lmsys-chat-1m-decontaminated"
download_dataset "openeurollm/orca-agentinstruct-1M-v1-decontaminated"
download_dataset "openeurollm/open-perfectblend-decontaminated"
download_dataset "openeurollm/smoltalk2-decontaminated"
download_dataset "openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated"

echo
echo "HF cache download complete."
