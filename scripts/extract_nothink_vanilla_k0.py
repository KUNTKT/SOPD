#!/usr/bin/env python3
"""Take k=0 no-think rollouts as vanilla teacher (1-shot, then keep correct)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/ktang115/SSOPD")
from ssopd_math.verifier.reward import POSITIVE


def main() -> None:
    src = Path("/home/yiyangba/ssopd_paper_archive/data/ssopd01_qwen3_1_7b_nothink_vf1200/rollouts.jsonl")
    out_dir = Path("/home/yiyangba/ssopd_paper_archive/reports/ssopd04_teacher_nothink_vanilla_k0")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        extra = r.get("extra") or {}
        k = extra.get("group_index", r.get("group_index"))
        if int(k) != 0:
            continue
        correct = int(r.get("verification_status") == POSITIVE)
        rows.append(
            {
                "problem_id": str(r["problem_id"]),
                "trajectory_id": r.get("trajectory_id") or f"{r['problem_id']}::k0",
                "split": "vector_fit",
                "role": "vanilla_teacher",
                "problem": r.get("problem"),
                "prompt_user": r.get("prompt_user") or extra.get("prompt_user") or r.get("problem"),
                "prompt_text": r.get("prompt_text"),
                "completion_text": r.get("completion_text"),
                "gold_answer": r["gold_answer"],
                "verification_status": r.get("verification_status"),
                "correct": correct,
                "parse_ok": int(bool(extra.get("parse_ok") or r.get("parse_ok"))),
                "completion_token_count": int(r.get("token_count") or len(r.get("completion_token_ids") or [])),
                "group_index": 0,
                "enable_thinking": False,
            }
        )
    all_path = out_dir / "teacher_trajectories.jsonl"
    cor_path = out_dir / "teacher_correct.jsonl"
    with all_path.open("w") as fa, cor_path.open("w") as fc:
        for r in rows:
            fa.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["correct"]:
                fc.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    c = sum(r["correct"] for r in rows)
    mean_len = sum(r["completion_token_count"] for r in rows) / max(n, 1)
    meta = {
        "n": n,
        "n_correct": c,
        "accuracy": c / max(n, 1),
        "mean_length": mean_len,
        "source": str(src),
        "protocol": "k0_one_shot_nothink",
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
