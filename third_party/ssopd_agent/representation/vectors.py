"""Paired mixed-task capability vectors for EXP04.

Main site is pre_call_reasoning only. Fallback, translator, and parse-failure
rows are excluded from the primary estimator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from ssopd_agent.labeling.tool_selection import NEGATIVE, POSITIVE

SITE_PRE_CALL = 0
EXCLUDED_TOOLS = frozenset({"translator"})
MIN_TOOL_SUPPORT = 3


def unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit_vector(a), unit_vector(b)
    return float(np.dot(ua, ub))


def is_valid_row(row: dict[str, Any]) -> bool:
    if row.get("tool_selection_label") not in {POSITIVE, NEGATIVE}:
        return False
    if row.get("tool_name_gold") in EXCLUDED_TOOLS:
        return False
    site = (row.get("sites") or {}).get("pre_call_reasoning") or {}
    if site.get("representation_fallback"):
        return False
    return True


def mixed_units(
    rows: list[dict[str, Any]],
    *,
    held_out_templates: Iterable[str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Pairing key = (task_id, tool_name_gold). Same task and same target tool."""
    held = set(held_out_templates or [])
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for i, row in enumerate(rows):
        if not is_valid_row(row):
            continue
        tmpl = row.get("template_id")
        if tmpl in held:
            continue
        key = (str(row["task_id"]), str(row["tool_name_gold"]))
        slot = buckets.setdefault(
            key,
            {
                "task_id": key[0],
                "tool_name_gold": key[1],
                "template_id": tmpl,
                "pos_idx": [],
                "neg_idx": [],
            },
        )
        if row["tool_selection_label"] == POSITIVE:
            slot["pos_idx"].append(i)
        else:
            slot["neg_idx"].append(i)
    return {
        k: v
        for k, v in buckets.items()
        if len(v["pos_idx"]) > 0 and len(v["neg_idx"]) > 0
    }


def task_deltas(
    acts: np.ndarray,
    units: dict[tuple[str, str], dict[str, Any]],
    layer_i: int,
) -> dict[tuple[str, str], np.ndarray]:
    """δ_x = mean(h+) - mean(h-) at pre_call_reasoning, float32."""
    out = {}
    h = np.asarray(acts[:, layer_i, SITE_PRE_CALL, :], dtype=np.float32)
    for key, slot in units.items():
        pos = h[slot["pos_idx"]].mean(axis=0)
        neg = h[slot["neg_idx"]].mean(axis=0)
        out[key] = pos - neg
    return out


def v_task(deltas: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    if not deltas:
        raise ValueError("empty mixed set")
    return np.mean(np.stack(list(deltas.values()), axis=0), axis=0)


def tool_groups(units: dict[tuple[str, str], dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    by: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, slot in units.items():
        by[slot["tool_name_gold"]].append(key)
    return dict(by)


def eligible_tools(groups: dict[str, list], min_support: int = MIN_TOOL_SUPPORT) -> list[str]:
    return sorted(g for g, keys in groups.items() if len(keys) >= min_support)


def v_macro(
    deltas: dict[tuple[str, str], np.ndarray],
    units: dict[tuple[str, str], dict[str, Any]],
    min_support: int = MIN_TOOL_SUPPORT,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    groups = tool_groups(units)
    elig = eligible_tools(groups, min_support)
    v_g = {}
    for g in elig:
        v_g[g] = np.mean(np.stack([deltas[k] for k in groups[g]], axis=0), axis=0)
    if not v_g:
        raise ValueError("no eligible tools for macro estimator")
    v = np.mean(np.stack(list(v_g.values()), axis=0), axis=0)
    return v, v_g, elig


def bootstrap_cosines(
    deltas: dict[tuple[str, str], np.ndarray],
    v_full: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    keys = list(deltas.keys())
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(n_boot):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        stacked = np.stack([deltas[keys[i]] for i in draw], axis=0)
        vb = stacked.mean(axis=0)
        cosines.append(cosine(v_full, vb))
    arr = np.asarray(cosines, dtype=np.float32)
    return {
        "n_boot": n_boot,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
