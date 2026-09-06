"""Task-balanced paired difference capability directions.

full2k_fix: random_unit uses rng.gauss.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np

from ssopd_math.verifier.reward import NEGATIVE, POSITIVE


def normalize_direction(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def paired_diff_per_problem(
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
) -> np.ndarray | None:
    if not positives or not negatives:
        return None
    pos = np.mean(np.stack(positives), axis=0)
    neg = np.mean(np.stack(negatives), axis=0)
    return pos - neg


def compute_macro_direction(per_problem: dict[str, np.ndarray]) -> np.ndarray:
    if not per_problem:
        return np.zeros(1, dtype=np.float64)
    return np.mean(np.stack(list(per_problem.values())), axis=0)


def _group_records(records: list[dict[str, Any]], layer: int, hidden_key: str):
    by_problem: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"pos": [], "neg": []}
    )
    fallback_count = 0
    dim: int | None = None
    for rec in records:
        st = rec.get("verification_status")
        if st not in (POSITIVE, NEGATIVE):
            continue
        hs = rec.get(hidden_key) or {}
        if layer not in hs:
            # also allow string keys
            if str(layer) in hs:
                vec = np.asarray(hs[str(layer)], dtype=np.float64)
            else:
                continue
        else:
            vec = np.asarray(hs[layer], dtype=np.float64)
        if dim is None:
            dim = int(vec.shape[0])
        if rec.get("representation_fallback"):
            fallback_count += 1
        pid = str(rec["problem_id"])
        if st == POSITIVE:
            by_problem[pid]["pos"].append(vec)
        else:
            by_problem[pid]["neg"].append(vec)
    return by_problem, fallback_count, dim


def build_directions_from_records(
    records: list[dict[str, Any]],
    *,
    layer: int,
    position_key: str = "reasoning_end",
    variant: str = "task_balanced_paired",
    seed: int = 0,
    hidden_key: str = "hidden_states",
) -> dict[str, Any]:
    del position_key
    rng = random.Random(int(seed))
    by_problem, fallback_count, dim = _group_records(records, layer, hidden_key)
    if dim is None:
        dim = 1

    per_problem: dict[str, np.ndarray] = {}
    for pid, g in by_problem.items():
        diff = paired_diff_per_problem(g["pos"], g["neg"])
        if diff is not None:
            per_problem[pid] = diff

    if variant == "task_balanced_paired":
        v_cap = compute_macro_direction(per_problem) if per_problem else np.zeros(dim)
    elif variant == "reasoning_mean":
        all_pos = [v for g in by_problem.values() for v in g["pos"]]
        all_neg = [v for g in by_problem.values() for v in g["neg"]]
        if not all_pos or not all_neg:
            v_cap = np.zeros(dim, dtype=np.float64)
        else:
            v_cap = np.mean(np.stack(all_pos), axis=0) - np.mean(np.stack(all_neg), axis=0)
    elif variant == "instance_level":
        diffs: list[np.ndarray] = []
        for g in by_problem.values():
            for p in g["pos"]:
                for n in g["neg"]:
                    diffs.append(p - n)
        v_cap = np.mean(np.stack(diffs), axis=0) if diffs else np.zeros(dim, dtype=np.float64)
    elif variant == "shuffled_label":
        labeled: list[tuple[np.ndarray, int]] = []
        for g in by_problem.values():
            for v in g["pos"]:
                labeled.append((v, 1))
            for v in g["neg"]:
                labeled.append((v, 0))
        rng.shuffle(labeled)
        if not labeled:
            v_cap = np.zeros(dim, dtype=np.float64)
        else:
            mid = max(1, len(labeled) // 2)
            pos = [v for v, _ in labeled[:mid]]
            neg = [v for v, _ in labeled[mid:]] or pos
            v_cap = np.mean(np.stack(pos), axis=0) - np.mean(np.stack(neg), axis=0)
    elif variant == "random_unit":
        v_cap = normalize_direction(
            np.asarray([rng.gauss(0.0, 1.0) for _ in range(dim)], dtype=np.float64)
        )
        return {
            "v_cap": v_cap.astype(np.float32),
            "v_hat": v_cap.astype(np.float32),
            "n_problems": 0,
            "fallback_count": fallback_count,
        }
    else:
        raise ValueError(f"unknown direction variant: {variant}")

    v_hat = normalize_direction(v_cap)
    return {
        "v_cap": np.asarray(v_cap, dtype=np.float32),
        "v_hat": np.asarray(v_hat, dtype=np.float32),
        "n_problems": len(per_problem),
        "fallback_count": fallback_count,
    }
