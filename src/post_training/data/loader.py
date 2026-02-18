"""Dataset loading, transformation, and mixing.

The main entry point is :func:`load_and_mix_datasets` which reads the
``data`` section of the config, loads each dataset, applies per-dataset
transforms, and interleaves them according to the specified weights.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional

from datasets import Dataset, interleave_datasets, load_dataset

from post_training.data.transforms import get_transform

if TYPE_CHECKING:
    from post_training.config import DataConfig

logger = logging.getLogger(__name__)

_MAX_NUM_PROC = 32


def _resolve_num_proc(configured: int | None) -> int:
    """Return the number of worker processes for ``.map()`` / ``.filter()``.

    When *configured* is ``None`` (the default), auto-detect from
    ``os.cpu_count()`` but cap at ``_MAX_NUM_PROC`` to avoid process
    explosion on large HPC nodes.  An explicit value is still clamped to the
    available CPU count so we never request more workers than cores.
    """
    available = os.cpu_count() or 1
    if configured is not None:
        return min(configured, available)
    return min(available, _MAX_NUM_PROC)


def load_and_mix_datasets(
    config: "DataConfig",
    row_filter: Optional[Callable[[dict], bool]] = None,
) -> Dataset:
    """Load, transform, and optionally filter/mix datasets.

    Parameters
    ----------
    config:
        The ``data`` section of :class:`PostTrainingConfig`.
    row_filter:
        Optional predicate applied after transforms to exclude invalid
        rows.  Each training method passes its own filter (e.g. SFT
        checks ``messages``, DPO checks ``chosen`` / ``rejected``).
        When ``None``, no filtering is applied.

    Returns
    -------
    datasets.Dataset
        A single (possibly interleaved) dataset ready for the trainer.
    """
    entries = config.datasets
    if not entries:
        raise ValueError("No datasets specified in data.datasets.")

    num_proc = _resolve_num_proc(config.num_proc)
    logger.info("Dataset processing will use num_proc=%d", num_proc)

    loaded_datasets: list[Dataset] = []
    weights: list[float] = []

    for entry in entries:
        logger.info(
            "Loading dataset '%s' from '%s' (data_dir=%s, subset=%s, split=%s, "
            "weight=%s, transform=%s)",
            entry.name,
            entry.path,
            entry.data_dir,
            entry.subset,
            entry.split,
            entry.weight,
            entry.transform,
        )

        # Build kwargs for load_dataset, only passing optional params when set.
        load_kwargs: dict = {}
        if entry.data_dir is not None:
            load_kwargs["data_dir"] = entry.data_dir
        if entry.subset is not None:
            load_kwargs["name"] = entry.subset

        ds = load_dataset(entry.path, split=entry.split, **load_kwargs)

        # Apply optional per-dataset transform.
        if entry.transform is not None:
            try:
                transform_fn = get_transform(entry.transform)
            except KeyError as exc:
                raise KeyError(
                    f"Unknown transform '{entry.transform}' for dataset '{entry.name}'. "
                    "Define it in 'post_training.data.transforms' using "
                    "@register_transform, or update your 'data.datasets[].transform' "
                    "value. Original error: "
                    f"{exc}"
                ) from exc

            logger.info(
                "Applying transform '%s' to dataset '%s'.",
                entry.transform,
                entry.name,
            )
            ds = ds.map(transform_fn, num_proc=num_proc)

        # Apply method-specific row filter (e.g. SFT checks for non-empty
        # "messages", DPO checks for non-empty "chosen" / "rejected").
        if row_filter is not None:
            ds = ds.filter(row_filter, num_proc=num_proc)

        loaded_datasets.append(ds)
        weights.append(entry.weight)

    # If there is only a single dataset, return it directly.
    if len(loaded_datasets) == 1:
        return loaded_datasets[0]

    # Normalise weights so they sum to 1.
    total = sum(weights)
    probabilities = [w / total for w in weights]

    logger.info(
        "Interleaving %d datasets with probabilities %s",
        len(loaded_datasets),
        probabilities,
    )
    return interleave_datasets(loaded_datasets, probabilities=probabilities)
