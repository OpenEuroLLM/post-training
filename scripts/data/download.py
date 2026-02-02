"""Download datasets from Hugging Face using snapshot_download."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download, scan_cache_dir


def get_cache_path(repo_id: str) -> Path | None:
    """Check if dataset is already in the cache."""
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "dataset":
            return repo.repo_path
    return None

def main():
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face")
    parser.add_argument("dataset", help="Dataset ID (e.g., allenai/tulu-3-sft-mixture)")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of download workers (default: 4)",
    )
    args = parser.parse_args()

    cache_path = get_cache_path(args.dataset)

    if cache_path is not None:
        print(f"Dataset '{args.dataset}' is already downloaded at: {cache_path}")
    else:
        print(f"Downloading dataset '{args.dataset}' with {args.workers} workers...")
        path = snapshot_download(
            repo_id=args.dataset,
            repo_type="dataset",
            max_workers=args.workers,
        )
        print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
