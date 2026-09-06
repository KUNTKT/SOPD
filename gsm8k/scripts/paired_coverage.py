#!/usr/bin/env python3
"""GSM8K vector_fit paired coverage (why n_paired=98)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ssopd_math.verifier.reward import NEGATIVE, POSITIVE

BASE = Path("/home/yiyangba/ssopd_paper_archive/gsm8k")


def main() -> None:
    splits = json.loads((BASE / "data/ssopd02_qwen3_1_7b_gsm8k_1k/splits.json").read_text())
    vf = {pid for pid, s in splits["problem_splits"].items() if s == "vector_fit"}
    by: dict[str, list[str]] = defaultdict(list)
    for line in (BASE / "data/ssopd01_qwen3_1_7b_gsm8k_1k/rollouts.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pid = str(r["problem_id"])
        if pid in vf:
            by[pid].append(str(r.get("verification_status", "")))

    mixed = all_pos = all_neg = other = 0
    n_pos = n_neg = 0
    for sts in by.values():
        n_pos += sts.count(POSITIVE)
        n_neg += sts.count(NEGATIVE)
        has_p = POSITIVE in sts
        has_n = NEGATIVE in sts
        if has_p and has_n:
            mixed += 1
        elif has_p:
            all_pos += 1
        elif has_n:
            all_neg += 1
        else:
            other += 1

    report = {
        "vector_fit_problems": len(by),
        "mixed_paired": mixed,
        "all_positive": all_pos,
        "all_negative": all_neg,
        "other": other,
        "traj_positive": n_pos,
        "traj_negative": n_neg,
        "success_rate_vf": n_pos / max(n_pos + n_neg + other, 1),
        "note": (
            "task_balanced_paired 只在同一题同时有 POSITIVE 与 NEGATIVE 时才贡献方向。"
            "GSM8K 正确率高 → 大量 all-positive → paired 远小于 600。"
        ),
    }
    out = BASE / "reports/gsm8k_paired_coverage.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = BASE / "reports/gsm8k_paired_coverage.md"
    md.write_text(
        "\n".join(
            [
                "# GSM8K vector_fit paired 覆盖",
                "",
                f"- vector_fit 题数: **{report['vector_fit_problems']}**",
                f"- 可配对（同题既对又错）: **{mixed}**",
                f"- 全对: {all_pos} · 全错: {all_neg} · 其它: {other}",
                f"- 轨迹 POSITIVE/NEGATIVE: {n_pos}/{n_neg}",
                "",
                report["note"],
                "",
            ]
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/scratch/ktang115/SSOPD")
    main()
