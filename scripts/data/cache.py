"""Download and cache all HuggingFace resources (models, tokenizers, datasets) referenced in a YAML config.

Usage:
    python scripts/data/cache.py configs/sft/tulu3.yaml
    python scripts/data/cache.py configs/dpo/tulu3.yaml --workers 8
    python scripts/data/cache.py configs/sft/tulu3.yaml configs/dpo/tulu3.yaml
"""

import argparse

import yaml
from huggingface_hub import scan_cache_dir, snapshot_download

# Config keys that reference HuggingFace model repos
MODEL_KEYS = {"model_name_or_path", "chat_template_path"}
# Config keys that reference HuggingFace dataset repos
DATASET_KEYS = {"dataset_name"}


def find_repos(config):
    """Extract unique HF repo IDs from a config dict, searching nested dicts."""
    models = set()
    datasets = set()

    def _scan(d):
        if not isinstance(d, dict):
            return
        for key, value in d.items():
            if isinstance(value, str):
                if key in MODEL_KEYS:
                    models.add(value)
                elif key in DATASET_KEYS:
                    datasets.add(value)
            elif isinstance(value, dict):
                _scan(value)

    _scan(config)
    return models, datasets


def main():
    parser = argparse.ArgumentParser(description="Download and cache HF resources from YAML configs")
    parser.add_argument("configs", nargs="+", help="Path(s) to YAML config files")
    parser.add_argument("--workers", type=int, default=4, help="Number of download workers (default: 4)")
    args = parser.parse_args()

    # Collect all repos across all config files
    models = set()
    datasets = set()
    for config_path in args.configs:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m, d = find_repos(config)
        models |= m
        datasets |= d

    if not models and not datasets:
        print("No HuggingFace resources found in config(s).")
        return

    repos = [(repo_id, "model") for repo_id in sorted(models)] + [
        (repo_id, "dataset") for repo_id in sorted(datasets)
    ]

    for repo_id, repo_type in repos:
        print(f"[downloading]  {repo_type:>7}  {repo_id}")
        path = snapshot_download(repo_id=repo_id, repo_type=repo_type, max_workers=args.workers)
        print(f"[done]         {repo_type:>7}  {repo_id}  ({path})")


if __name__ == "__main__":
    main()
