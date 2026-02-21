#!/usr/bin/env python3
"""Wandb utility script for managing offline runs.

Usage
-----
# Sync a specific run (non-interactive)
python scripts/wb.py sync --run-name sft-olmo-3-1025-7b-nemotron_pt_v2-20260218-172238
python scripts/wb.py sync --wandb-folder offline-run-20260218_172330-7ukg3tch

# Interactive mode - choose which runs to sync
python scripts/wb.py sync --interactive
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is on ``sys.path`` so that ``post_training`` is
# importable when running directly (``python scripts/wb.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from post_training.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def extract_run_name_from_log(debug_log: Path) -> Optional[str]:
    """Extract run_name from wandb debug.log file.

    The run_name is stored in the config dictionary logged to debug.log.
    """
    if not debug_log.exists():
        return None

    try:
        with open(debug_log) as f:
            for line in f:
                # Look for run_name in the config dictionary
                if "'run_name':" in line:
                    match = re.search(r"'run_name':\s*'([^']+)'", line)
                    if match:
                        return match.group(1)
    except Exception as e:
        logger.debug("Error reading debug.log: %s", e)

    return None


def map_wandb_runs_to_training_runs(
    wandb_dir: Path, outputs_dir: Path
) -> list[dict[str, str]]:
    """Map wandb offline runs to training output runs.

    Returns a list of mappings with keys:
    - wandb_folder: name of the wandb run folder
    - run_id: wandb run ID
    - run_name: training run name (from outputs folder)
    - output_dir: path to the output directory (if exists)
    """
    mappings = []

    # Find all wandb run directories
    wandb_runs = []
    for item in wandb_dir.iterdir():
        if item.is_dir() and item.name.startswith("offline-run-"):
            wandb_runs.append(item)

    logger.info("Found %d wandb runs", len(wandb_runs))

    # Extract run names from each wandb run
    for run_dir in sorted(wandb_runs):
        run_id = run_dir.name.split("-")[-1]  # Extract run ID from folder name
        debug_log = run_dir / "logs" / "debug.log"

        run_name = extract_run_name_from_log(debug_log)

        if run_name:
            # Check if corresponding output directory exists
            output_dir = outputs_dir / run_name
            output_exists = output_dir.exists()

            mappings.append(
                {
                    "wandb_folder": run_dir.name,
                    "run_id": run_id,
                    "run_name": run_name,
                    "output_dir": str(output_dir) if output_exists else None,
                    "output_exists": output_exists,
                }
            )
        else:
            logger.warning("Could not extract run_name from %s", run_dir.name)

    return mappings


def sync_run(wandb_folder: str, wandb_dir: Path) -> bool:
    """Sync a wandb offline run to wandb cloud.

    Args:
        wandb_folder: Name of the wandb run folder (e.g., "offline-run-20260218_172330-7ukg3tch")
        wandb_dir: Path to the wandb directory

    Returns:
        True if sync was successful, False otherwise
    """
    run_path = wandb_dir / wandb_folder

    if not run_path.exists():
        logger.error("Wandb run folder not found: %s", run_path)
        return False

    logger.info("Syncing wandb run: %s", wandb_folder)

    try:
        # Use wandb CLI to sync the offline run
        result = subprocess.run(
            ["wandb", "sync", str(run_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Successfully synced: %s", wandb_folder)
            return True
        else:
            logger.error("Failed to sync %s: %s", wandb_folder, result.stderr)
            return False
    except FileNotFoundError:
        logger.error("wandb CLI not found. Please install wandb: pip install wandb")
        return False
    except Exception as e:
        logger.error("Error syncing %s: %s", wandb_folder, e)
        return False


def sync_command(args: argparse.Namespace) -> None:
    """Handle the sync command."""
    wandb_dir = Path(args.wandb_dir)
    outputs_dir = Path(args.outputs_dir)

    if not wandb_dir.exists():
        logger.error("Wandb directory not found: %s", wandb_dir)
        sys.exit(1)

    # Default to interactive mode if no option is provided
    if args.interactive or (not args.run_name and not args.wandb_folder):
        # Interactive mode: show all mappings and let user choose
        mappings = map_wandb_runs_to_training_runs(wandb_dir, outputs_dir)

        if not mappings:
            logger.warning("No wandb runs found or no run names could be extracted.")
            return

        print("\n" + "=" * 80)
        print("Wandb Runs → Training Runs Mapping")
        print("=" * 80)
        print()

        for i, mapping in enumerate(mappings, 1):
            status = "✓" if mapping["output_exists"] else "✗"
            print(f"{i:2d}. {status} {mapping['wandb_folder']}")
            print(f"    → run_name: {mapping['run_name']}")
            if mapping["output_exists"]:
                print(f"    → output: {mapping['output_dir']}")
            else:
                print("    → output: (not found)")
            print()

        print("=" * 80)
        print(
            "Enter run numbers to sync (comma-separated, e.g., 1,3,5) or 'all' for all:"
        )
        selection = input("> ").strip()

        if selection.lower() == "all":
            selected_indices = list(range(len(mappings)))
        else:
            try:
                selected_indices = [int(x.strip()) - 1 for x in selection.split(",")]
                # Validate indices
                selected_indices = [
                    i for i in selected_indices if 0 <= i < len(mappings)
                ]
            except ValueError:
                logger.error(
                    "Invalid selection. Please enter numbers separated by commas."
                )
                return

        if not selected_indices:
            logger.warning("No valid runs selected.")
            return

        print(f"\nSyncing {len(selected_indices)} run(s)...")
        print()

        success_count = 0
        for idx in selected_indices:
            mapping = mappings[idx]
            if sync_run(mapping["wandb_folder"], wandb_dir):
                success_count += 1

        print(f"\n{'=' * 80}")
        print(f"Successfully synced {success_count}/{len(selected_indices)} run(s)")
        print("=" * 80)

    elif args.run_name:
        # Non-interactive mode: sync by training run name
        mappings = map_wandb_runs_to_training_runs(wandb_dir, outputs_dir)

        # Find the wandb folder for this run name
        matching_mapping = None
        for mapping in mappings:
            if mapping["run_name"] == args.run_name:
                matching_mapping = mapping
                break

        if not matching_mapping:
            logger.error("No wandb run found for training run: %s", args.run_name)
            logger.info("Available run names:")
            for mapping in mappings:
                logger.info("  - %s", mapping["run_name"])
            sys.exit(1)

        success = sync_run(matching_mapping["wandb_folder"], wandb_dir)
        sys.exit(0 if success else 1)

    elif args.wandb_folder:
        # Non-interactive mode: sync by wandb folder name
        success = sync_run(args.wandb_folder, wandb_dir)
        sys.exit(0 if success else 1)

    else:
        # This should not happen due to the default logic above, but keep as safety check
        logger.error(
            "Invalid arguments. Use --interactive, --run-name, or --wandb-folder."
        )
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Wandb utility script for managing offline runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="wandb",
        help="Path to the wandb directory (default: wandb)",
    )

    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Path to the outputs directory (default: outputs)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync offline wandb runs to wandb cloud",
        description="Sync offline wandb runs to wandb cloud. Interactive mode is the default. "
        "Use --run-name or --wandb-folder for non-interactive mode.",
    )

    sync_group = sync_parser.add_mutually_exclusive_group(required=False)
    sync_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: show all runs and let user choose which to sync (default)",
    )
    sync_group.add_argument(
        "--run-name",
        type=str,
        help="Training run name (from outputs folder) to sync",
    )
    sync_group.add_argument(
        "--wandb-folder",
        type=str,
        help="Wandb run folder name (e.g., offline-run-20260218_172330-7ukg3tch) to sync",
    )

    args = parser.parse_args()

    if args.command == "sync":
        sync_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
