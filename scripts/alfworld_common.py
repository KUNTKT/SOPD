#!/usr/bin/env python3
"""Shared utilities for ALFWorld steering experiments (archive; read-only SSOPD)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ARCHIVE = Path(__file__).resolve().parents[1]
_THIRD = ARCHIVE / "third_party"
_SSOPD_ENV = os.environ.get("SSOPD_ROOT")
SSOPD_ROOT = Path(_SSOPD_ENV) if _SSOPD_ENV else (_THIRD if (_THIRD / "ssopd_logits").exists() else Path("/scratch/ktang115/SSOPD"))
for p in (
    str(ARCHIVE / "scripts"),
    str(SSOPD_ROOT),
    str(SSOPD_ROOT / "ssopd_logits") if (SSOPD_ROOT / "ssopd_logits").exists() else str(SSOPD_ROOT),
    str(_THIRD),
    str(_THIRD / "ssopd_logits"),
):
    if p and p not in sys.path:
        sys.path.insert(0, p)

from environments.alfworld_adapter import (  # noqa: E402
    ALFWORLD_FEWSHOT_PROMPT,
    AlfWorldEnv,
    load_alfworld_tasks,
    partition_alfworld_tasks,
    tasks_fingerprint,
)
from rollout.multistep import group_by_task, mixed_groups, rollout_summary  # noqa: E402

DEFAULT_MODEL = os.environ.get("SSOPD_MODEL", "/scratch/ktang115/models/Qwen3-1.7B")
DEFAULT_DATA_ROOT = os.environ.get("ALFWORLD_DATA", "/scratch/ktang115/cache/alfworld")
PARTITION_SEED = 1010
PARTITION_FRACTIONS = {"sft": 0.4, "audit_select": 0.3, "audit_confirm": 0.3}


def _rewrite_paths(obj: Any) -> Any:
    """Map cluster absolute paths to this checkout / env vars."""
    mapping = {
        "/home/yiyangba/ssopd_paper_archive": str(ARCHIVE),
        "/scratch/ktang115/models/Qwen3-1.7B": DEFAULT_MODEL,
        "/scratch/ktang115/models/Qwen3-8B": os.environ.get("SSOPD_MODEL_8B", "/scratch/ktang115/models/Qwen3-8B"),
        "/scratch/ktang115/cache/alfworld": DEFAULT_DATA_ROOT,
        "/home/yiyangba/ssopd_paper_archive/data/ssopd05_agent_ssopd/uce_library.json": str(
            ARCHIVE / "artifacts/uce/uce_library.json"
        ),
        "/home/yiyangba/ssopd_paper_archive/data/ssopd05_agent_ssopd/uce_library_evolved.json": str(
            ARCHIVE / "artifacts/uce/uce_library_evolved.json"
        ),
        "/home/yiyangba/ssopd_paper_archive/data/ssopd05_alfworld/directions.npz": str(
            ARCHIVE / "artifacts/directions/directions.npz"
        ),
    }

    def rec(x: Any) -> Any:
        if isinstance(x, str):
            out = x
            for old, new in mapping.items():
                out = out.replace(old, new)
            return os.path.expandvars(out)
        if isinstance(x, list):
            return [rec(i) for i in x]
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        return x

    return rec(obj)


def load_yaml_cfg(path: Path) -> dict[str, Any]:
    import yaml

    return _rewrite_paths(yaml.safe_load(path.read_text()))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def ensure_alfworld_env(cfg: dict[str, Any]) -> None:
    import os

    root = cfg.get("env", {}).get("data_root", DEFAULT_DATA_ROOT)
    os.environ.setdefault("ALFWORLD_DATA", str(root))


def load_eval_tasks(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Holdout tasks (e.g. valid_unseen) — no train partition."""
    env = cfg.get("env", {})
    split = str(env.get("eval_split", "valid_unseen"))
    limit = env.get("eval_limit")
    data_root = env.get("data_root", DEFAULT_DATA_ROOT)
    tasks = load_alfworld_tasks(
        split=split,
        task_types=tuple(env.get("task_types", (1, 2, 3, 4, 5, 6))),
        root=data_root,
        limit=int(limit) if limit is not None else None,
    )
    meta = {
        "data_root": data_root,
        "split": split,
        "limit": limit,
        "n_tasks": len(tasks),
        "fingerprint": tasks_fingerprint(tasks),
    }
    return tasks, meta


def load_task_pools(
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    limit: int = 1200,
    split: str = "train",
    partition_seed: int = PARTITION_SEED,
    fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    fractions = fractions or PARTITION_FRACTIONS
    tasks = load_alfworld_tasks(
        split=split,
        task_types=(1, 2, 3, 4, 5, 6),
        root=data_root,
        limit=limit,
    )
    pools = partition_alfworld_tasks(tasks, fractions=fractions, seed=partition_seed)
    meta = {
        "data_root": data_root,
        "limit": limit,
        "split": split,
        "partition_seed": partition_seed,
        "fractions": fractions,
        "fingerprint": tasks_fingerprint(tasks),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
    }
    return {"pools": pools, "meta": meta}


def make_env_factory(cfg: dict[str, Any]):
    rollout = cfg.get("rollout", {})
    prompt = cfg.get("prompt", {})

    def factory():
        return AlfWorldEnv(
            max_steps=int(rollout.get("max_steps", 40)),
            history_window=int(prompt.get("history_window", 12)),
            expert_plan=False,
        )

    return factory


def trajectory_metrics(records: list[dict]) -> dict[str, Any]:
    """Extended rollout summary for steering experiments."""
    base = rollout_summary(records)
    n = len(records)
    hit_max = 0
    n_dp = 0
    n_adm = 0
    n_parse = 0
    for rec in records:
        if str(rec.get("termination_reason")) == "max_steps":
            hit_max += 1
        for dp in rec.get("decision_points") or []:
            n_dp += 1
            if dp.get("parse_ok"):
                n_parse += 1
                n_adm += 1
            elif dp.get("failure_reason") != "not_admissible" and dp.get("predicted_tool"):
                n_parse += 1
    base["hit_max_steps_rate"] = hit_max / n if n else 0.0
    base["admissible_action_rate"] = n_adm / n_dp if n_dp else 0.0
    base["parse_ok_rate"] = n_parse / n_dp if n_dp else 0.0
    base["mean_tokens"] = sum(int(r.get("token_count") or 0) for r in records) / n if n else 0.0
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        tt = str(rec.get("task_type") or "unknown")
        by_type.setdefault(tt, []).append(rec)
    base["by_task_type"] = {
        tt: {
            "n": len(rows),
            "success_rate": sum(1 for r in rows if r.get("episode_success")) / len(rows),
            "admissible_rate": rollout_summary(rows).get("admissible_rate", 0.0),
        }
        for tt, rows in by_type.items()
    }
    return base


def paired_task_bootstrap(records_a: list[dict], records_b: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Task-level bootstrap for success_rate difference (records_b - records_a)."""
    import random

    ga = group_by_task(records_a)
    gb = group_by_task(records_b)
    common = sorted(set(ga) & set(gb))
    if not common:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_tasks": 0}

    def task_delta(task_id: str) -> float:
        ra = ga[task_id]
        rb = gb[task_id]
        sa = sum(1 for r in ra if r.get("episode_success")) / len(ra)
        sb = sum(1 for r in rb if r.get("episode_success")) / len(rb)
        return sb - sa

    rng = random.Random(seed)
    deltas = [task_delta(t) for t in common]
    mean = sum(deltas) / len(deltas)
    boots = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_tasks": len(common)}


def amplification_ratio(delta_episode: float, delta_admissible: float, eps: float = 1e-6) -> float | None:
    if abs(delta_admissible) < eps:
        return None
    return delta_episode / delta_admissible
