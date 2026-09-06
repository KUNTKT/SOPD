"""Task-clustered bootstrap."""

from __future__ import annotations

import random
from typing import Callable, Sequence


def clustered_bootstrap_mean(
    values_by_task: dict[str, list[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Resample tasks with replacement; within-task values averaged then overall mean."""
    task_ids = list(values_by_task.keys())
    if not task_ids:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_tasks": 0}

    def task_means(ids: Sequence[str]) -> float:
        acc = []
        for tid in ids:
            vals = values_by_task[tid]
            if vals:
                acc.append(sum(vals) / len(vals))
        return sum(acc) / len(acc) if acc else float("nan")

    point = task_means(task_ids)
    rng = random.Random(seed)
    boots = []
    n = len(task_ids)
    for _ in range(n_boot):
        sample = [task_ids[rng.randrange(n)] for _ in range(n)]
        boots.append(task_means(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {
        "mean": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_tasks": n,
        "n_boot": n_boot,
    }


def paired_mean_diff_bootstrap(
    a_by_task: dict[str, list[float]],
    b_by_task: dict[str, list[float]],
    *,
    n_boot: int = 2000,
    seed: int = 1,
) -> dict:
    """Bootstrap mean(a)-mean(b) with shared task resampling."""
    tasks = sorted(set(a_by_task) & set(b_by_task))
    diffs: dict[str, list[float]] = {}
    for t in tasks:
        va = a_by_task[t]
        vb = b_by_task[t]
        # use mean per task
        if va and vb:
            diffs[t] = [sum(va) / len(va) - sum(vb) / len(vb)]
    return clustered_bootstrap_mean(diffs, n_boot=n_boot, seed=seed)
