"""Task-isolated linear probes for EXP05 readout (not layer selection)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ssopd_agent.labeling.tool_selection import NEGATIVE, POSITIVE
from ssopd_agent.representation.vectors import (
    SITE_PRE_CALL,
    is_valid_row,
    mixed_units,
    task_deltas,
    unit_vector,
    v_task,
)


def eligible_indices(
    rows: list[dict[str, Any]],
    *,
    held_out_templates: Iterable[str] | None = None,
) -> list[int]:
    held = set(held_out_templates or [])
    out = []
    for i, row in enumerate(rows):
        if not is_valid_row(row):
            continue
        if row.get("template_id") in held:
            continue
        out.append(i)
    return out


def labels_from_rows(rows: list[dict[str, Any]], idxs: list[int]) -> np.ndarray:
    y = []
    for i in idxs:
        lab = rows[i]["tool_selection_label"]
        if lab == POSITIVE:
            y.append(1)
        elif lab == NEGATIVE:
            y.append(0)
        else:
            raise ValueError(lab)
    return np.asarray(y, dtype=np.int64)


def features_layer(acts: np.ndarray, idxs: list[int], layer_i: int) -> np.ndarray:
    return np.asarray(acts[idxs, layer_i, SITE_PRE_CALL, :], dtype=np.float32)


def meta_feature(rows: list[dict[str, Any]], idxs: list[int], key: str) -> np.ndarray:
    vals = []
    for i in idxs:
        if key == "context_length":
            vals.append(float(rows[i].get("context_length") or 0))
        elif key == "step_id":
            vals.append(float(rows[i].get("step_id") or 0))
        elif key == "trajectory_length":
            vals.append(float(rows[i].get("trajectory_length") or 0))
        else:
            raise KeyError(key)
    return np.asarray(vals, dtype=np.float32)[:, None]


def task_ids(rows: list[dict[str, Any]], idxs: list[int]) -> list[str]:
    return [str(rows[i]["task_id"]) for i in idxs]


def unique_tasks(rows: list[dict[str, Any]], idxs: list[int]) -> list[str]:
    seen = []
    for tid in task_ids(rows, idxs):
        if tid not in seen:
            seen.append(tid)
    return seen


def split_tasks(
    tasks: list[str],
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[set[str], set[str]]]:
    """Deterministic task folds. Each fold returns (train_tasks, test_tasks)."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    rng = np.random.default_rng(seed)
    order = list(tasks)
    rng.shuffle(order)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, t in enumerate(order):
        folds[i % n_folds].append(t)
    out = []
    for k in range(n_folds):
        test = set(folds[k])
        train = set(order) - test
        if not train or not test:
            raise RuntimeError("empty fold after task split")
        out.append((train, test))
    return out


def indices_for_tasks(rows: list[dict[str, Any]], idxs: list[int], tasks: set[str]) -> list[int]:
    return [i for i in idxs if str(rows[i]["task_id"]) in tasks]


def fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(np.unique(y_train)) < 2:
        raise ValueError("train set needs both classes")
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_test)
    clf = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=int(seed),
    )
    clf.fit(Xt, y_train)
    proba = clf.predict_proba(Xv)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    info = {
        "n_features": int(X_train.shape[1]),
        "coef_l2": float(np.linalg.norm(clf.coef_.ravel())),
        "intercept": float(clf.intercept_[0]),
    }
    return pred, proba.astype(np.float64), info


def score_binary(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, proba))
    return out


def shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(y, copy=True)
    rng.shuffle(out)
    return out


def projection_scores(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    vhat = unit_vector(v)
    return (X.astype(np.float32) @ vhat).astype(np.float64)


def train_only_v_task(
    acts: np.ndarray,
    rows: list[dict[str, Any]],
    train_idxs: list[int],
    layer_i: int,
    held_out_templates: Iterable[str] | None,
) -> np.ndarray | None:
    """Recompute v_task on train rows only to avoid test leakage."""
    # Remap mixed_units indices to local positions inside a view of train rows.
    train_rows = [rows[i] for i in train_idxs]
    units = mixed_units(train_rows, held_out_templates=held_out_templates)
    if not units:
        return None
    # acts for train rows in the same order as train_rows
    sub = np.asarray(acts[train_idxs], dtype=np.float16)
    deltas = task_deltas(sub, units, layer_i)
    if not deltas:
        return None
    return v_task(deltas)


def mean_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ["accuracy", "balanced_accuracy", "auroc"]
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) == m.get(k)]  # drop nan
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    out["n_folds"] = len(fold_metrics)
    return out


def inventory(rows: list[dict[str, Any]], idxs: list[int]) -> dict[str, Any]:
    by_tool: dict[str, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    by_task: dict[str, set[str]] = defaultdict(set)
    for i in idxs:
        r = rows[i]
        lab = r["tool_selection_label"]
        tool = str(r["tool_name_gold"])
        if lab == POSITIVE:
            by_tool[tool]["pos"] += 1
            by_task[r["task_id"]].add("pos")
        else:
            by_tool[tool]["neg"] += 1
            by_task[r["task_id"]].add("neg")
    return {
        "n_rows": len(idxs),
        "n_tasks": len({rows[i]["task_id"] for i in idxs}),
        "n_pos": sum(1 for i in idxs if rows[i]["tool_selection_label"] == POSITIVE),
        "n_neg": sum(1 for i in idxs if rows[i]["tool_selection_label"] == NEGATIVE),
        "by_tool": dict(sorted(by_tool.items())),
        "n_mixed_tasks": sum(1 for labs in by_task.values() if labs == {"pos", "neg"}),
    }
