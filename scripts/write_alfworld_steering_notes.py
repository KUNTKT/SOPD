#!/usr/bin/env python3
"""Write ALFWorld steering notes + ALL_RESULTS snippet from experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARCHIVE = Path("/home/yiyangba/ssopd_paper_archive")


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt_pp(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):+.2f}"


def main() -> None:
    smoke_base = load(ARCHIVE / "reports/ssopd05_alfworld_smoke/smoke_results.json")
    smoke_cs = load(ARCHIVE / "reports/ssopd05_alfworld_smoke/smoke_results_coldstart.json")
    p1 = load(ARCHIVE / "reports/ssopd05_alfworld_confirm/p1_rollout_summary.json")
    fit = load(ARCHIVE / "data/ssopd05_alfworld/fit_meta.json")
    p3 = load(ARCHIVE / "reports/ssopd05_alfworld_confirm/results_confirm.json")
    if p3 is None:
        raise SystemExit("missing results_confirm.json; P3 not finished")

    pb = p3.get("phase_b") or {}
    analysis = p3.get("analysis") or {}
    delta_ep = pb.get("delta_success_pp")
    delta_adm = pb.get("delta_admissible_pp")
    amp = pb.get("amplification_ratio")
    boot = pb.get("bootstrap") or {}
    style = analysis.get("selected_style")
    alpha = analysis.get("selected_alpha")
    phase_a_gain = analysis.get("phase_a_best_gain_pp")
    boot_lo = boot.get("ci_lo")
    boot_hi = boot.get("ci_hi")
    confirm_sig = boot_lo is not None and boot_hi is not None and boot_lo > 0

    if confirm_sig and amp is not None and float(amp) > 1.5 and (delta_ep or 0) >= 5.0:
        verdict = "H1 PASS（全量 confirm）：端到端增益显著且大于逐步 admissible 增益，支持多轮放大。"
    elif (delta_ep or 0) >= 5.0 and confirm_sig:
        verdict = "部分 PASS：端到端 +5 pp 但放大比未分离 admissible/episode。"
    elif phase_a_gain and float(phase_a_gain) >= 5.0 and not confirm_sig:
        verdict = (
            f"Phase A（80 task）探 α 见 +{float(phase_a_gain):.2f} pp，但 **全量 confirm358** "
            f"仅 {fmt_pp(delta_ep)} pp，bootstrap CI [{boot_lo:.3f}, {boot_hi:.3f}] 含 0。"
            "不支持干净的 superlinear 放大；负 α 方向与 MATH no-think 一致。"
        )
    elif delta_ep is not None and abs(float(delta_ep)) <= 3.5:
        verdict = "H0：全量 confirm 增益与 MATH no-think 同级，不支持 superlinear 放大。"
    else:
        verdict = "结果混合：见 Phase A/B 分解。"

    lines = [
        "# ALFWorld 多轮注入放大实验",
        "",
        "协议：Qwen3-1.7B + few-shot ALFWorld，sft 池 planner 轨迹 coldstart LoRA，",
        "`audit_select` 拟合 L14 `v_cap`，`audit_confirm` 一次性 α 扫。无蒸馏。",
        "",
        "## P0 Base vs coldstart",
        "",
        f"- 裸跑（20 task × K=2）：success **{(smoke_base or {}).get('metrics', {}).get('success_rate', 0)*100:.2f}%**，",
        f"admissible {(smoke_base or {}).get('metrics', {}).get('admissible_action_rate', 0):.3f} → GATE FAIL",
        f"- Qwen3 coldstart LoRA 后：success **{(smoke_cs or {}).get('metrics', {}).get('success_rate', 0)*100:.2f}%**，",
        f"admissible {(smoke_cs or {}).get('metrics', {}).get('admissible_action_rate', 0):.3f} → GATE PASS",
        f"- adapter: `data/ssopd05_alfworld_coldstart/coldstart_lora`",
        "",
        "## P1 Fit-pool rollouts",
        "",
        f"- pool=`audit_select`，n_tasks={p1['metrics']['n_tasks'] if p1 else '—'}，K=4",
        f"- success **{p1['metrics']['success_rate']*100:.2f}%**，admissible {p1['metrics']['admissible_action_rate']:.3f}" if p1 else "",
        f"- paired / mixed groups: **{p1.get('n_paired_tasks') if p1 else '—'}**",
        "",
        "## P2 Direction",
        "",
        f"- n_paired_used={fit.get('n_paired_used') if fit else '—'}",
        f"- ‖v_cap‖={fit.get('v_cap_norm') if fit else '—'}",
        f"- cosine vs MATH no-think v: {fit.get('cosine_to_math_nothink') if fit else '—'}",
        "- 提取：每条轨迹 4 个决策点 prompt 末 token mean-pool（step-0 同构会导致 v=0）",
        "",
        "## P3 Confirm",
        "",
        f"- selected inject_style=`{style}` α={alpha}",
        f"- Phase A（80 task）best gain: {fmt_pp(phase_a_gain)} pp（α={alpha}，**非 holdout 全量**）",
        "",
        "### Phase B（full audit_confirm n=358）",
        "",
        "| | success | admissible | mean_steps |",
        "|--|--------:|-----------:|-----------:|",
        f"| α=0 | {(pb.get('baseline') or {}).get('success_rate', 0)*100:.2f}% | {(pb.get('baseline') or {}).get('admissible_action_rate', 0):.3f} | {(pb.get('baseline') or {}).get('mean_steps', 0):.1f} |",
        f"| steered | {(pb.get('steered') or {}).get('success_rate', 0)*100:.2f}% | {(pb.get('steered') or {}).get('admissible_action_rate', 0):.3f} | {(pb.get('steered') or {}).get('mean_steps', 0):.1f} |",
        f"| Δ | {fmt_pp(delta_ep)} pp | {fmt_pp(delta_adm)} pp | — |",
        "",
        f"- amplification_ratio = Δepisode / Δadmissible = **{amp if amp is not None else '—'}**",
        f"- task-level bootstrap mean={boot.get('mean')} CI=[{boot.get('ci_lo')}, {boot.get('ci_hi')}] n={boot.get('n_tasks')}",
        f"- confirm 显著性（CI_lo>0）: **{confirm_sig}**",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## MATH 对照",
        "",
        "| 设定 | proxy | primary | 放大比 |",
        "|------|-------|---------|--------|",
        "| MATH no-think confirm400 | — | +3.00 pp | 1×（单轮） |",
        f"| ALFWorld confirm | {fmt_pp(delta_adm)} pp admissible | {fmt_pp(delta_ep)} pp episode | {amp if amp is not None else '—'} |",
        "",
    ]
    notes = ARCHIVE / "notes/alfworld_steering_result.md"
    notes.write_text("\n".join(lines))
    print(f"wrote {notes}", flush=True)

    # Patch ALL_RESULTS.md: replace or append section.
    ar = ARCHIVE / "ALL_RESULTS.md"
    text = ar.read_text() if ar.exists() else ""
    marker = "## ALFWorld 多轮注入（ssopd05）"
    section = "\n".join(
        [
            marker,
            "",
            f"- Coldstart LoRA 后 select success **{p1['metrics']['success_rate']*100:.2f}%**（P1），paired={p1.get('n_paired_tasks')}。" if p1 else "",
            f"- L14 v_cap ‖v‖={fit.get('v_cap_norm'):.2f}，cos(MATH-nothink)={fit.get('cosine_to_math_nothink'):.3f}。" if fit else "",
            f"- Confirm：style=`{style}` α={alpha}，Δepisode **{fmt_pp(delta_ep)} pp**，Δadmissible **{fmt_pp(delta_adm)} pp**，amplification={amp}。",
            f"- {verdict}",
            f"- 产物：`reports/ssopd05_alfworld_confirm/` · `notes/alfworld_steering_result.md` · `data/ssopd05_alfworld/directions.npz`",
            "",
        ]
    )
    if marker in text:
        # replace from marker to next ## or EOF
        i = text.index(marker)
        j = text.find("\n## ", i + 1)
        text = text[:i] + section + (text[j:] if j != -1 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    ar.write_text(text)
    print(f"updated {ar}", flush=True)


if __name__ == "__main__":
    main()
