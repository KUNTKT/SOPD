#!/usr/bin/env python3
"""Write D0/D1 gate results into archive notes. Does not edit the Cursor plan file."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ARCHIVE = Path("/home/yiyangba/ssopd_paper_archive")


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.1f}%"


def _pp(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):+.1f} pp"


def _ci(boot: dict | None) -> str:
    if not boot:
        return "n/a"
    return f"[{100.0 * float(boot.get('ci_lo', 0)):.1f}, {100.0 * float(boot.get('ci_hi', 0)):.1f}]"


def upsert_section(path: Path, heading: str, body: str) -> None:
    text = path.read_text() if path.exists() else ""
    start = text.find(heading)
    if start < 0:
        path.write_text(text.rstrip() + "\n\n" + heading + "\n\n" + body.rstrip() + "\n")
        return
    nxt = text.find("\n## ", start + len(heading))
    if nxt < 0:
        new = text[:start] + heading + "\n\n" + body.rstrip() + "\n"
    else:
        new = text[:start] + heading + "\n\n" + body.rstrip() + "\n" + text[nxt:]
    path.write_text(new)


def ledger_d0() -> dict:
    summary = json.loads((ARCHIVE / "reports/uce_opd/summary.json").read_text())
    results = summary.get("results") or {}
    gate = summary.get("gate") or {}
    passed = bool(gate.get("gate_pass"))
    verdict = "PASS" if passed else "FAIL"
    boots = gate.get("bootstrap") or {}
    rows = []
    for key, label in [
        ("H0", "H0（复用）"),
        ("vanilla_sft", "Vanilla-LoRA"),
        ("no_pi_opd", "No-PI OPD"),
        ("uce_sft", "UCE-SFT（不得叫 OPD）"),
        ("uce_opd", "UCE-OPD"),
        ("B_uce", "B-uce（复用，只作 R_int）"),
    ]:
        rec = results.get(key) or {}
        rows.append(f"| {label} | {_pct(rec.get('success_rate'))} |")
    body = f"""更新：{date.today().isoformat()}。评测：valid_unseen 100，无记忆、HF generate、只加载蒸馏 LoRA。

| 条件 | success |
|------|---------|
{chr(10).join(rows)}

- ΔH0 = {_pp(gate.get("delta_h0_pp"))}；vs H0 CI {_ci(boots.get("vs_h0"))}
- vs Vanilla {_pp(gate.get("delta_vanilla_pp"))}；vs No-PI {_pp(gate.get("delta_no_pi_pp"))}；vs UCE-SFT {_pp(gate.get("delta_sft_pp"))}（只作文）
- R_int = {gate.get("R_int")}
- **Gate D0：{verdict}。** {"进入 D1。" if passed else "不调 KL/lr/epoch/rank，不进 D1，不宣称 memory 可内化。"}
- 产物：`reports/uce_opd/summary.json` · `paired_bootstrap.json` · `eval_*.jsonl`
"""
    upsert_section(ARCHIVE / "notes/uce_opd_plan.md", "## D0 结果", body)
    claim = ARCHIVE / "notes/agent_ssopd_claim.md"
    claim_text = claim.read_text()
    marker = "<!-- UCE_OPD_D0 -->"
    block = (
        f"{marker}\n**UCE-OPD Gate D0：{verdict}。** "
        f"UCE-OPD {_pct((results.get('uce_opd') or {}).get('success_rate'))}，"
        f"ΔH0 {_pp(gate.get('delta_h0_pp'))}，CI {_ci(boots.get('vs_h0'))}。"
        + (
            "过门，继续记忆—参数交替。"
            if passed
            else "未过门；项目记为 UCE 与 parameterization 的机制负结果。不进 D1。"
        )
        + "\n"
    )
    if marker in claim_text:
        pre, rest = claim_text.split(marker, 1)
        nxt = rest.find("\n## ")
        claim.write_text(pre + block + (rest[nxt:] if nxt >= 0 else ""))
    else:
        claim.write_text(claim_text.rstrip() + "\n\n" + block)
    upsert_section(ARCHIVE / "ALL_RESULTS.md", "## UCE-OPD（Self-Amortized Procedural Memory）", body)
    table_line = (
        f"| ALFWorld agent：UCE-OPD 内化 | **{verdict}** "
        f"{_pct((results.get('uce_opd') or {}).get('success_rate'))} "
        f"（ΔH0 {_pp(gate.get('delta_h0_pp'))}） |"
    )
    allr = (ARCHIVE / "ALL_RESULTS.md").read_text()
    if "ALFWorld agent：UCE-OPD 内化" not in allr:
        allr = allr.replace(
            "| ALFWorld agent：NPM / 忠实 NPM / H4 logit / H5 动作分布 | **FAIL**（steering 三层否证；未蒸馏） |\n",
            "| ALFWorld agent：NPM / 忠实 NPM / H4 logit / H5 动作分布 | **FAIL**（steering 三层否证；未蒸馏） |\n"
            + table_line
            + "\n",
        )
        (ARCHIVE / "ALL_RESULTS.md").write_text(allr)
    return {"stage": "d0", "gate_pass": passed, "gate": gate}


def ledger_d1() -> dict:
    summary = json.loads((ARCHIVE / "reports/uce_opd/d1/summary.json").read_text())
    results = summary.get("results") or {}
    gate = summary.get("gate") or {}
    passed = bool(gate.get("gate_pass"))
    verdict = "PASS" if passed else "FAIL"
    boots = gate.get("bootstrap") or {}
    rows = []
    for key, label in [
        ("theta1", "θ1（D0 UCE-OPD，无记忆）"),
        ("theta2_m0", "θ2 Fixed-memory M0"),
        ("theta2_m1", "θ2 Re-evolved M1"),
        ("vanilla_cont", "Continued Vanilla"),
    ]:
        rec = results.get(key) or {}
        rows.append(f"| {label} | {_pct(rec.get('success_rate'))} |")
    body = f"""更新：{date.today().isoformat()}。教师仍为冻结 θ0，只换 memory。评测无记忆。

| 条件 | success |
|------|---------|
{chr(10).join(rows)}

- S_θ2,M1 − S_θ1 = {_pp(gate.get("delta_theta1_pp"))}；CI {_ci(boots.get("theta2_m1_vs_theta1"))}
- vs Fixed M0 {_pp(gate.get("delta_m0_pp"))}
- M1：{json.dumps(summary.get("m1") or {}, ensure_ascii=False)}
- **Gate D1：{verdict}。**
- 产物：`reports/uce_opd/d1/summary.json`
"""
    upsert_section(ARCHIVE / "notes/uce_opd_plan.md", "## D1 结果", body)
    upsert_section(ARCHIVE / "ALL_RESULTS.md", "## UCE-OPD D1（记忆—参数交替）", body)
    return {"stage": "d1", "gate_pass": passed, "gate": gate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["d0", "d1"], required=True)
    args = ap.parse_args()
    payload = ledger_d0() if args.stage == "d0" else ledger_d1()
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
