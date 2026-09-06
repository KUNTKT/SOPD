"""Task-cluster bootstrap utilities."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np


def task_cluster_bootstrap(
    values: Mapping[str, float] | Mapping[Any, float],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap the mean of per-task values with task resampling.

    values: mapping task_id -> scalar (e.g. accuracy or paired delta)
    Returns mean / 95% percentile CI.
    """
    keys = list(values.keys())
    arr = np.asarray([float(values[k]) for k in keys], dtype=np.float64)
    n = int(arr.shape[0])
    if n == 0:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_tasks": 0,
            "n_boot": int(n_boot),
        }

    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means[i] = float(arr[idx].mean())

    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_tasks": n,
        "n_boot": int(n_boot),
    }
