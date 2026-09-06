#!/usr/bin/env python3
"""Merge D4 + weak-α confirm results and rewrite L1/L2 verdict."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/yiyangba/ssopd_paper_archive/gsm8k")
D4 = BASE / "reports/ssopd03_qwen3_1_7b_gsm8k_confirm_alpha/results_confirm.json"
WEAK = BASE / "reports/ssopd03_qwen3_1_7b_gsm8k_confirm_weak_alpha/results_confirm.json"
MERGED = BASE / "reports/ssopd03_qwen3_1_7b_gsm8k_confirm_merged/results_confirm.json"


def main() -> None:
    a = json.loads(D4.read_text())
    b = json.loads(WEAK.read_text())
    steered = dict(a["per_problem"]["steered"])
    steered.update(b["per_problem"]["steered"])
    merged = dict(a)
    merged["per_problem"] = {
        "clean": a["per_problem"]["clean"],
        "steered": steered,
    }
    merged["selection_curve"] = (a.get("selection_curve") or []) + [
        row for row in (b.get("selection_curve") or []) if row.get("alpha") not in (0.0, 0)
    ]
    merged["experiment_id"] = "ssopd03_qwen3_1_7b_gsm8k_confirm_merged"
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    MERGED.write_text(json.dumps(merged, indent=2))
    cmd = [
        sys.executable,
        str(BASE / "scripts/analyze_gsm8k_L1_L2.py"),
        "--results",
        str(MERGED),
        "--out-json",
        str(BASE / "reports/gsm8k_L1_L2_length_analysis.json"),
        "--out-md",
        str(BASE / "reports/gsm8k_L1_L2_verdict.md"),
        "--mt",
        "2048",
    ]
    subprocess.check_call(cmd)
    print("merged alphas", sorted(steered, key=float))


if __name__ == "__main__":
    main()
