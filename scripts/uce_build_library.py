#!/usr/bin/env python3
"""B0: instance-level UCE library from select-pool success trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, load_yaml_cfg  # noqa: E402
from uce_library import UceLibrary, extract_goal_from_prompt, make_entry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_uce_alfworld.yaml"),
    )
    args = ap.parse_args()
    cfg = load_yaml_cfg(Path(args.config))
    src = Path(cfg["paths"]["rollouts_select"])
    out = Path(cfg["paths"]["uce_library"])

    lib = UceLibrary()
    n_success = 0
    n_skip = 0
    by_type: Counter[str] = Counter()
    with src.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("episode_success"):
                continue
            n_success += 1
            goal = extract_goal_from_prompt(str(rec.get("prompt_text") or ""))
            actions = rec.get("actions") or []
            entry = make_entry(
                entry_id=f"sel_{n_success:04d}_{rec.get('task_id', i)}",
                task_type=str(rec.get("task_type") or "unknown"),
                goal=goal,
                actions=actions,
                source_task_id=str(rec.get("task_id") or ""),
                usage=1,
            )
            if not entry["lines"]:
                n_skip += 1
                continue
            lib.add(entry)
            by_type[entry["task_type"]] += 1

    lib.save(
        out,
        n_success_scanned=n_success,
        n_skip_empty=n_skip,
        source=str(src),
        by_task_type=dict(by_type),
        note="instance-level; no type-majority skeleton",
    )
    dump_json(
        Path(cfg["paths"]["reports_dir"]) / "b0_library_summary.json",
        {
            "n_success_scanned": n_success,
            "n_entries": len(lib.entries),
            "n_skip_empty": n_skip,
            "by_task_type": dict(by_type),
            "library": str(out),
        },
    )
    print(
        f"wrote {out} n_entries={len(lib.entries)} "
        f"n_success={n_success} skip={n_skip} types={dict(by_type)}"
    )


if __name__ == "__main__":
    main()
