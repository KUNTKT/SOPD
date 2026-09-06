"""Resume helpers for rollout jsonl files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_existing_rollouts(path: Path | str) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[str(rec["trajectory_id"])] = rec
    return out


def append_rollout(path: Path | str, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
