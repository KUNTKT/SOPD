#!/usr/bin/env python3
"""NPM-faithful retrieval (task text) + PCA synthesis (archive)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from uce_library import extract_goal_from_prompt, jaccard, tokenize


def pca_steering(h_pos: np.ndarray, h_neg: np.ndarray) -> np.ndarray:
    """First PC of centered (h+ − h−), flipped to agree with mean(h+)−mean(h−)."""
    if h_pos.ndim != 2 or h_neg.ndim != 2 or h_pos.shape != h_neg.shape:
        raise ValueError("h_pos/h_neg must be [n, d] and same shape")
    diffs = h_pos.astype(np.float64) - h_neg.astype(np.float64)
    diffs = diffs - diffs.mean(axis=0, keepdims=True)
    if diffs.shape[0] == 1:
        v = diffs[0]
    else:
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
        v = vt[0]
    sign = float((h_pos.mean(0) - h_neg.mean(0)).astype(np.float64) @ v)
    if sign < 0:
        v = -v
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros(h_pos.shape[1], dtype=np.float32)
    return (v / n).astype(np.float32)


class FaithfulNpmMemory:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        self.layers = [int(x) for x in data["layers"].tolist()]
        self.task_ids = [str(x) for x in data["task_ids"].tolist()]
        self.goals = [str(x) for x in data["goals"].tolist()]
        self.kinds = [str(x) for x in data["kinds"].tolist()]
        self.h_pos = {int(li): data[f"h_pos_{li}"].astype(np.float32) for li in self.layers}
        self.h_neg = {int(li): data[f"h_neg_{li}"].astype(np.float32) for li in self.layers}
        self._by_task: dict[str, list[int]] = defaultdict(list)
        self._goal_toks: list[frozenset[str]] = []
        for i, (tid, goal) in enumerate(zip(self.task_ids, self.goals)):
            self._by_task[tid].append(i)
            self._goal_toks.append(tokenize(goal))
        self._task_goal: dict[str, frozenset[str]] = {}
        for tid, idxs in self._by_task.items():
            toks: set[str] = set()
            for i in idxs:
                toks |= set(self._goal_toks[i])
            self._task_goal[tid] = frozenset(toks)

    def retrieve_tasks(self, goal: str, top_k: int) -> list[str]:
        q = tokenize(goal)
        scored = [(jaccard(q, toks), tid) for tid, toks in self._task_goal.items()]
        scored.sort(reverse=True)
        return [tid for _, tid in scored[:top_k]]

    def synthesize(self, goal: str, top_k: int = 8) -> dict[int, np.ndarray]:
        tasks = self.retrieve_tasks(goal, top_k)
        idxs = [i for tid in tasks for i in self._by_task.get(tid, [])]
        if not idxs:
            idxs = list(range(min(8, len(self.task_ids))))
        out: dict[int, np.ndarray] = {}
        for li in self.layers:
            hp = self.h_pos[li][idxs]
            hn = self.h_neg[li][idxs]
            out[int(li)] = pca_steering(hp, hn)
        return out


def rec_goal(rec: dict[str, Any]) -> str:
    g = str(rec.get("goal") or "").strip()
    if g:
        return g
    return extract_goal_from_prompt(str(rec.get("prompt_text") or ""))
