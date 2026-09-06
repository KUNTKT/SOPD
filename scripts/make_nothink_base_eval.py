#!/usr/bin/env python3
"""Adapt no-think confirm clean into q4 --reuse-base-eval format."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    src = json.loads(
        Path("/home/yiyangba/ssopd_paper_archive/reports/ssopd03_math_L14_nothink_confirm/results_confirm.json").read_text()
    )
    out = Path("/home/yiyangba/ssopd_paper_archive/reports/ssopd04_nothink_base_eval.json")
    clean = src["clean"]
    per = src["per_problem"]["clean"]
    payload = {
        "base_eval": {
            "n": clean["n"],
            "accuracy": clean["accuracy"],
            "parse_rate": clean["parse_rate"],
            "mean_length": clean["mean_length"],
            "wall_s": 0.0,
            "source": "ssopd03_math_L14_nothink_confirm clean",
        },
        "base_per_problem": per,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} acc={clean['accuracy']:.4f} n={clean['n']}")


if __name__ == "__main__":
    main()
