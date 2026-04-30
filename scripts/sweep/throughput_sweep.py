#!/usr/bin/env python3
"""Submit DPO throughput benchmark jobs across different node counts.

Each job uses the same per-device batch size; effective_batch_size is scaled
so that gradient_accumulation_steps=1 on every run.  This isolates raw GPU
throughput from accumulation overhead, making tokens/sec and MFU directly
comparable across node counts.

Usage
-----
# Sweep 1, 2, and 4 nodes (default)
python scripts/sweep/throughput_sweep.py

# Custom node counts and config
python scripts/sweep/throughput_sweep.py --nodes 1 2 4 8 --config configs/trl/dpo_throughput.yaml

# Skip confirmation prompt
python scripts/sweep/throughput_sweep.py --nodes 1 2 4 --confirm

# Extra config overrides (forwarded to every job)
python scripts/sweep/throughput_sweep.py --nodes 1 2 dpo.max_seq_length=4096
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.backend import get_backend
from post_training.config import PostTrainingConfig
from post_training.slurm.launcher import generate_and_submit
from post_training.utils.logging import setup_logging
from post_training.utils.paths import setup_run_directory

logger = logging.getLogger(__name__)


def _parse_args() -> tuple[str, list[int], bool, list[str]]:
    parser = argparse.ArgumentParser(
        description="Submit DPO throughput benchmark jobs across node counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/trl/dpo_throughput.yaml",
        help="Base YAML config (default: configs/trl/dpo_throughput.yaml).",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        metavar="N",
        help="Node counts to sweep (default: 1 2 4).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    known, cli_overrides = parser.parse_known_args()
    return known.config, known.nodes, known.confirm, cli_overrides


def _build_job_configs(
    config_path: str,
    node_counts: list[int],
    cli_overrides: list[str],
) -> list[tuple[int, int, PostTrainingConfig]]:
    """Return (num_nodes, world_size, config) for each node count."""
    jobs = []
    for num_nodes in node_counts:
        config = PostTrainingConfig.load(config_path, cli_overrides)
        config.slurm.num_nodes = num_nodes
        world_size = num_nodes * config.slurm.gpus_per_node
        # GAS=1: scale effective_batch_size = per_device * world_size.
        config.training.effective_batch_size = (
            config.training.per_device_train_batch_size * world_size
        )
        config.slurm.job_name = f"dpo-throughput-{num_nodes}n"
        jobs.append((num_nodes, world_size, config))
    return jobs


def _print_plan(jobs: list[tuple[int, int, PostTrainingConfig]]) -> None:
    print("\n=== DPO Throughput Sweep Plan ===\n")
    header = f"  {'Nodes':>5}  {'GPUs':>5}  {'EffBatch':>8}  {'GAS':>4}  {'Steps':>6}  {'MaxSeqLen':>9}  Job Name"
    print(header)
    print(f"  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*30}")
    for num_nodes, world_size, cfg in jobs:
        bs = cfg.training.per_device_train_batch_size
        eff = cfg.training.effective_batch_size
        gas = eff // (bs * world_size)
        print(
            f"  {num_nodes:>5}  {world_size:>5}  {eff:>8}  {gas:>4}"
            f"  {cfg.training.max_steps:>6}  {cfg.dpo.max_seq_length:>9}  {cfg.slurm.job_name}"
        )
    print()


def _print_results(results: list[tuple[int, int, str, Path]]) -> None:
    print("\n=== Submitted Jobs ===\n")
    print(f"  {'Nodes':>5}  {'GPUs':>5}  {'Job ID':>10}  Run Directory")
    print(f"  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*55}")
    for num_nodes, world_size, job_id, run_dir in results:
        print(f"  {num_nodes:>5}  {world_size:>5}  {job_id:>10}  {run_dir}")
    print()
    print("Monitor throughput/tokens_per_sec and throughput/mfu in TensorBoard:")
    for _, _, _, run_dir in results:
        print(f"  tensorboard --logdir {run_dir / 'logs'}")
    print()


def main() -> None:
    setup_logging()
    config_path, node_counts, confirmed, cli_overrides = _parse_args()

    jobs = _build_job_configs(config_path, node_counts, cli_overrides)
    _print_plan(jobs)

    if not confirmed:
        answer = input("Submit all jobs? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(1)
        print()

    results: list[tuple[int, int, str, Path]] = []
    for num_nodes, world_size, config in jobs:
        run_dir = setup_run_directory(config)
        config.run_name = run_dir.name
        frozen = run_dir / "config.yaml"
        config.save(frozen)

        get_backend(config.backend).post_freeze(config, run_dir)

        job_id = generate_and_submit(config, run_dir, str(frozen))
        results.append((num_nodes, world_size, job_id, run_dir))
        logger.info("Submitted %d-node job %s → %s", num_nodes, job_id, run_dir)

    _print_results(results)


if __name__ == "__main__":
    main()
