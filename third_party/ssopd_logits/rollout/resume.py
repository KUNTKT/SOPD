"""Resume helpers for incremental JSONL rollout collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_trajectory_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["trajectory_id"]): r for r in load_jsonl(path) if "trajectory_id" in r}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(path: Path) -> set[str]:
    return set(index_by_trajectory_id(path).keys())
