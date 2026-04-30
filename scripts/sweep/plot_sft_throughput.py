#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb",
#   "matplotlib",
#   "numpy",
# ]
# ///
"""Post-add throughput/tokens_per_sec to wandb runs and plot weak/strong scaling figures.

Each SLURM job spawns one wandb run per node with identical metrics; this script
deduplicates by keeping the run with the most logged steps per (scaling, nodes, seq_len),
back-calculates tokens/sec from throughput/tflops_per_gpu, posts it to run.summary,
then saves one figure per (scaling × seq_len) combination.

Usage
-----
    python scripts/sweep/plot_sft_throughput.py                        # update + plot
    python scripts/sweep/plot_sft_throughput.py --plot-only            # skip summary update
    python scripts/sweep/plot_sft_throughput.py --dry-run              # print plan, no I/O
    python scripts/sweep/plot_sft_throughput.py --project sft-tp-7b    # override project name
    python scripts/sweep/plot_sft_throughput.py --scaling weak         # weak only
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb

# ── Constants ──────────────────────────────────────────────────────────────────
ENTITY          = "openeurollm-project"
PROJECT         = "sft-throughput-jupiter"
ACTIVE_PARAMS   = 7_298_011_136   # MFUCallback log: num_params=7298011136 (7.30B)
PEAK_TFLOPS_GPU = 989.0           # GH200 BF16 peak
FLOPS_COEFF     = 6.0             # SFT: 6N per token
WARMUP_SKIP     = 3               # drop first N log steps before taking median


# ── Wandb helpers ──────────────────────────────────────────────────────────────

def parse_tags(run) -> dict[str, str]:
    """Tags are stored as 'key=value' strings; return as a dict."""
    return {k: v for tag in run.tags if "=" in tag for k, v in [tag.split("=", 1)]}


def fetch_and_deduplicate(api: wandb.Api, project: str) -> dict[tuple, wandb.apis.public.Run]:
    """Return the most-complete run per (scaling, nodes, seq_len)."""
    runs = api.runs(f"{ENTITY}/{project}")
    groups: dict[tuple, list] = defaultdict(list)

    for run in runs:
        tags = parse_tags(run)
        scaling = tags.get("scaling")
        nodes   = tags.get("nodes")
        seq_len = tags.get("seq_len")
        if not (scaling and nodes and seq_len):
            continue
        key = (scaling, int(nodes), int(seq_len))
        groups[key].append(run)

    return {
        key: max(runs_list, key=lambda r: r.lastHistoryStep or 0)
        for key, runs_list in groups.items()
    }


def median_metric(run, key: str, skip: int = WARMUP_SKIP) -> float | None:
    hist = run.history(keys=[key], pandas=False)
    vals = [row[key] for row in hist if row.get(key) is not None]
    if not vals:
        return None
    trimmed = vals[skip:] or vals
    return statistics.median(trimmed)


def tflops_to_total_tps(tflops_per_gpu: float, world_size: int) -> float:
    tps_per_gpu = tflops_per_gpu * 1e12 / (FLOPS_COEFF * ACTIVE_PARAMS)
    return tps_per_gpu * world_size


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_scaling(
    data: dict[int, dict[str, float]],
    seq_len: int,
    scaling: str,
) -> plt.Figure:
    """
    data maps num_gpus → {"tokens_per_sec": float, "tflops_per_gpu": float}.
    """
    gpus      = sorted(data)
    tps       = [data[g]["tokens_per_sec"]  for g in gpus]
    tflops    = [data[g]["tflops_per_gpu"]  for g in gpus]
    tps_per_g = [data[g]["tokens_per_sec"] / g for g in gpus]

    # Optimal (linear) scaling anchored at the smallest GPU count
    optimal    = [tps[0] * (g / gpus[0]) for g in gpus]
    efficiency = [100 * t / o for t, o in zip(tps, optimal)]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        f"Token Throughput  OLMo-3 7B Think SFT  ·  {scaling.title()} Scaling  ·  seq={seq_len // 1024}k",
        fontsize=12, fontweight="bold",
    )

    x    = np.arange(len(gpus))
    xlbl = [str(g) for g in gpus]

    # ── Top panel: total tokens/sec + efficiency ───────────────────────────────
    ax_eff = ax_top.twinx()

    bars = ax_top.bar(x, tps, 0.5, color="#4472C4", alpha=0.85, label="Measured Tok/s")
    ax_top.plot(x, optimal, "r--", marker="s", ms=5, lw=1.5, label="Optimal scaling")
    ax_eff.plot(x, efficiency, color="crimson", linestyle=":", marker="o", ms=5,
                lw=1.5, label="Efficiency (%)")

    for bar, v in zip(bars, tps):
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.03,
            f"{v / 1e6:.2f}M",
            ha="center", va="bottom", fontsize=8,
        )

    ax_top.set_xticks(x); ax_top.set_xticklabels(xlbl)
    ax_top.set_ylabel("Tokens / second")
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax_eff.set_ylabel("Efficiency (%)")
    ax_eff.set_ylim(0, 115)

    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_eff.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    # ── Bottom panel: TFLOPs/GPU + tokens/s per GPU ────────────────────────────
    ax_tpg = ax_bot.twinx()

    bars2 = ax_bot.bar(x, tflops, 0.5, color="#70AD47", alpha=0.85, label="TFLOPs/GPU")
    ax_tpg.plot(x, tps_per_g, color="darkgreen", linestyle="--", marker="D",
                ms=6, lw=1.5, label="Tokens/s per GPU")

    for bar, v in zip(bars2, tflops):
        ax_bot.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.03,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax_bot.set_xticks(x); ax_bot.set_xticklabels(xlbl)
    ax_bot.set_xlabel("Number of GPUs")
    ax_bot.set_ylabel("TFLOPs / GPU")
    ax_tpg.set_ylabel("Tokens/s per GPU")
    ax_tpg.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.1f}k"))

    h1, l1 = ax_bot.get_legend_handles_labels()
    h2, l2 = ax_tpg.get_legend_handles_labels()
    ax_bot.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project",    default=PROJECT, help="WandB project name")
    parser.add_argument("--scaling",    default=None, choices=["weak", "strong"],
                        help="Filter to one scaling mode (default: both)")
    parser.add_argument("--plot-only",  action="store_true",
                        help="Skip run.summary update; use cached tokens_per_sec if present")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print plan without posting to wandb or writing files")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    api = wandb.Api()
    print(f"Fetching runs from {ENTITY}/{args.project} …")
    best = fetch_and_deduplicate(api, args.project)
    print(f"  {len(best)} unique (scaling, nodes, seq_len) configs found\n")

    # ── Collect metrics per config ─────────────────────────────────────────────
    # key: (scaling, world_size, seq_len)  →  {"tokens_per_sec": …, "tflops_per_gpu": …}
    collected: dict[tuple, dict[str, float]] = {}

    for (scaling, nodes, seq_len), run in sorted(best.items()):
        if args.scaling and scaling != args.scaling:
            continue

        tags       = parse_tags(run)
        world_size = int(tags.get("gpus", nodes * 4))

        # tflops: always fetch from history (needed for bottom panel)
        med_tflops = median_metric(run, "throughput/tflops_per_gpu")
        if med_tflops is None:
            print(f"  SKIP {run.name}: no tflops data in history")
            continue

        # always back-calculate from tflops_per_gpu — existing wandb tokens_per_sec
        # is per-node local throughput, not total across all GPUs
        total_tps = tflops_to_total_tps(med_tflops, world_size)
        if not args.plot_only:
            if args.dry_run:
                print(f"  [dry-run] {run.name}: would set tokens_per_sec={total_tps:.0f}")
            else:
                run.summary.update({"throughput/tokens_per_sec_total": round(total_tps)})

        print(
            f"  {scaling:6s}  {nodes:3d}n  {world_size:4d} GPUs  "
            f"seq={seq_len//1024:2d}k  →  {total_tps/1e6:.3f}M tok/s  "
            f"{med_tflops:.1f} TFLOPs/GPU"
        )
        collected[(scaling, world_size, seq_len)] = {
            "tokens_per_sec": total_tps,
            "tflops_per_gpu": med_tflops,
        }

    if not collected:
        print("\nNo data collected — check project name and tags.")
        return

    # ── Save figures ───────────────────────────────────────────────────────────
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    scalings = sorted({k[0] for k in collected})
    seq_lens = sorted({k[2] for k in collected})

    for sc in scalings:
        for sl in seq_lens:
            data = {gpus: m for (s, gpus, seq), m in collected.items() if s == sc and seq == sl}
            if not data:
                continue

            fig = plot_scaling(data, sl, sc)
            out = Path(args.output_dir) / f"sft_throughput_{sc}_{sl//1024}k.png"

            if args.dry_run:
                print(f"\n  [dry-run] figure for {sc}/{sl//1024}k → {out}")
            else:
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"\n  Saved → {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
