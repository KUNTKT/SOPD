#!/usr/bin/env python3
"""Pre-fill a smoke_resume checkpoint with existing no-think clean per_problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="results_confirm.json with per_problem.clean")
    ap.add_argument("--dst", required=True, help="checkpoint_confirm.json to write")
    ap.add_argument("--experiment-id", required=True)
    args = ap.parse_args()
    src = json.loads(Path(args.src).read_text())
    clean = src["per_problem"]["clean"]
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "split": "confirm",
                "phase": "clean",
                "clean_per_problem": clean,
                "steered_per_problem": {},
            },
            indent=2,
        )
    )
    print(f"wrote {dst} clean_n={len(clean)}")


if __name__ == "__main__":
    main()
