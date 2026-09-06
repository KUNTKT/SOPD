#!/usr/bin/env python3
"""Update ALL_RESULTS + injection research notes from unseen ablation results."""

from __future__ import annotations

import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
RESULTS = ARCHIVE / "reports/ssopd05_alfworld_unseen_ablation/results.json"
ALL_RESULTS = ARCHIVE / "ALL_RESULTS.md"
NOTES = ARCHIVE / "notes/alfworld_steering_injection_research.md"


def fmt_grid_row(rid: str, payload: dict) -> str:
    grid = payload.get("grid", {})
    m0 = grid.get("0.0", {}).get("metrics", {})
    best_a = payload.get("best_alpha", 0.0)
    mb = grid.get(str(best_a), {}).get("metrics", {})
    return (
        f"| {rid} | {payload.get('inject_style')} | "
        f"{m0.get('success_rate', 0)*100:.1f}% | "
        f"{mb.get('success_rate', 0)*100:.1f}% @ α={best_a} | "
        f"{payload.get('best_gain_pp_vs_alpha0', 0):+.2f} pp |"
    )


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}")
    data = json.loads(RESULTS.read_text())
    ana = data.get("analysis", {})
    gate = ana.get("gate", {})
    r0 = data.get("r0", {}).get("conditions", {})
    r1 = data.get("r1", {})

    lines = [
        "## valid_unseen 注入位点消融（R0/R1）",
        "",
        f"- 评测池：valid_unseen，n={data.get('task_meta', {}).get('n_tasks', '?')} tasks",
        f"- R0 赢家：**{ana.get('winner_rid')}** (`{ana.get('winner_style')}`) α={ana.get('winner_alpha')}",
        f"- vs R0a α=0：Δsuccess **{ana.get('winner_delta_pp_vs_r0a_alpha0', 0):+.2f} pp**",
        f"- Gate（+3pp, CI_lo>0, amp>1）：**{'PASS' if gate.get('gate_pass') else 'FAIL'}**",
        f"  - Δsuccess={gate.get('delta_success_pp', 0):+.2f} pp，bootstrap CI [{gate.get('bootstrap', {}).get('ci_lo', 0)*100:.2f}, {gate.get('bootstrap', {}).get('ci_hi', 0)*100:.2f}] pp",
        f"  - 放大比={gate.get('amplification_ratio')}",
        "",
        "| ID | inject_style | α=0 success | best success | Δ vs α=0 |",
        "|----|--------------|-------------|--------------|----------|",
    ]
    for rid in sorted(r0.keys()):
        lines.append(fmt_grid_row(rid, r0[rid]))
    lines.extend(["", "### R1", ""])
    if r1.get("R1a", {}).get("metrics"):
        m = r1["R1a"]["metrics"]
        lines.append(f"- R1a step-admissible v：success {m.get('success_rate', 0)*100:.1f}%")
    elif r1.get("R1a", {}).get("skipped"):
        lines.append(f"- R1a：skipped ({r1['R1a'].get('reason')})")
    if r1.get("R1b", {}).get("metrics"):
        m = r1["R1b"]["metrics"]
        lines.append(f"- R1b CAST-lite：success {m.get('success_rate', 0)*100:.1f}%")
    lines.append(f"- 产物：`reports/ssopd05_alfworld_unseen_ablation/`")
    block = "\n".join(lines) + "\n"

    # ALL_RESULTS
    text = ALL_RESULTS.read_text()
    marker = "## valid_unseen 注入位点消融（R0/R1）"
    if marker in text:
        pre, _, post = text.partition(marker)
        # drop old block until next ## section
        if "\n## " in post:
            _, _, rest = post.partition("\n## ")
            text = pre + block + "\n## " + rest
        else:
            text = pre + block
    else:
        # insert after ssopd05 confirm section
        anchor = "- 产物：`reports/ssopd05_alfworld_confirm/`"
        if anchor in text:
            text = text.replace(anchor, anchor + "\n" + block.rstrip())
        else:
            text = text.rstrip() + "\n\n" + block

    ALL_RESULTS.write_text(text)

    # Research notes results section
    notes = NOTES.read_text()
    res_marker = "## 结果"
    res_body = (
        "## 结果\n\n"
        f"完整 JSON：`reports/ssopd05_alfworld_unseen_ablation/results.json`\n\n"
        f"- Gate: **{'PASS' if gate.get('gate_pass') else 'FAIL'}**\n"
        f"- Winner: {ana.get('winner_rid')} / {ana.get('winner_style')} / α={ana.get('winner_alpha')}\n"
    )
    if res_marker in notes:
        pre, _, _ = notes.partition(res_marker)
        notes = pre.rstrip() + "\n\n" + res_body
    else:
        notes = notes.rstrip() + "\n\n" + res_body
    NOTES.write_text(notes)
    print("Updated", ALL_RESULTS, "and", NOTES)


if __name__ == "__main__":
    main()
