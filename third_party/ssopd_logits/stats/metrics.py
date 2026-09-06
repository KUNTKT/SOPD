"""Shared metric helpers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


def wilson_interval(
    successes: int, n: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used where the count is small enough that a point estimate invites
    over-reading: the cold-start mixed-group rate read 5/60 and 9/40 for the same
    checkpoint, so a gate compared against the point estimate alone would have
    been decided by which task subset happened to be probed.

    Preferred over the normal approximation because it stays inside [0, 1] and
    does not degenerate at 0 or n successes, both of which occur here.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def rate(flags: Iterable[bool]) -> float:
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags) if flags else float("nan")


def label_counts(labels: Iterable[str]) -> dict[str, int]:
    return dict(Counter(labels))
