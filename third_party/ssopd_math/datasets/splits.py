"""Problem split assignment for SSOPD vector_fit / selection / confirm."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


def build_split_manifest(
    problem_ids: Iterable[str],
    vector_fit_frac: float = 0.6,
    selection_frac: float = 0.2,
    confirm_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    ids = sorted(str(x) for x in problem_ids)
    n = len(ids)
    if n == 0:
        return {
            "seed": int(seed),
            "fractions": {
                "vector_fit": float(vector_fit_frac),
                "selection": float(selection_frac),
                "confirm": float(confirm_frac),
            },
            "counts": {"vector_fit": 0, "selection": 0, "confirm": 0},
            "problem_splits": {},
        }

    total = float(vector_fit_frac) + float(selection_frac) + float(confirm_frac)
    vf = float(vector_fit_frac) / total
    sf = float(selection_frac) / total
    # remainder goes to confirm

    rng = random.Random(int(seed))
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n_fit = int(round(n * vf))
    n_sel = int(round(n * sf))
    # keep totals exact
    if n_fit + n_sel > n:
        n_sel = max(0, n - n_fit)
    n_conf = n - n_fit - n_sel

    fit_ids = shuffled[:n_fit]
    sel_ids = shuffled[n_fit : n_fit + n_sel]
    conf_ids = shuffled[n_fit + n_sel :]

    problem_splits = {}
    for pid in fit_ids:
        problem_splits[pid] = "vector_fit"
    for pid in sel_ids:
        problem_splits[pid] = "selection"
    for pid in conf_ids:
        problem_splits[pid] = "confirm"

    return {
        "seed": int(seed),
        "fractions": {
            "vector_fit": float(vector_fit_frac),
            "selection": float(selection_frac),
            "confirm": float(confirm_frac),
        },
        "counts": {
            "vector_fit": len(fit_ids),
            "selection": len(sel_ids),
            "confirm": len(conf_ids),
        },
        "problem_splits": problem_splits,
    }


def save_splits(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
