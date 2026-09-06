#!/usr/bin/env python3
"""probe_confirm evaluation and T0-A / T0-B gate. Then stop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from agent_uce_opd_lib import build_student_agent  # noqa: E402
from alfworld_common import dump_json, ensure_alfworld_env, trajectory_metrics  # noqa: E402
from tame_opd_lib import (  # noqa: E402
    RUN_SEEDS,
    evaluate_gates,
    load_manifest,
    load_tame_cfg,
    per_seed_success,
    sub_seed,
    task_mean_paired_bootstrap,
    task_mean_success,
    tasks_of,
    with_rollout_seed,
)


def _eval(agent, tasks, cfg, out_path: Path, seed: int, resume: bool) -> list[dict]:
    cfg_s = with_rollout_seed(cfg, sub_seed(seed, "evaluation"))
    return run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg_s,
        memory=None,
        alpha=0.0,
        top_k=8,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_path,
        resume=resume,
        dataset_split="probe_confirm",
        workflow_fn=None,
    )


def _load_recs(path: Path) -> list[dict]:
    from alfworld_common import load_jsonl

    return load_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    cfg = load_tame_cfg(args.config)
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    adapters = Path(cfg["paths"]["adapters_dir"])
    manifest = load_manifest(reports / "task_manifest.json")
    tasks = tasks_of(manifest, "probe_confirm")
    teacher = json.loads((reports / "teacher_eval.json").read_text())
    probe = json.loads((reports / "probe_meta_select.json").read_text())
    eligible = list(teacher.get("eligible") or ["M0"])
    teach_pick = probe.get("teach_pick")
    teacher_pick = teacher.get("teacher_pick")

    if probe.get("skipped") or eligible == ["M0"] or teach_pick is None:
        j_m0 = teacher.get("results", {}).get("M0", {}).get("mean_success") or 0.0
        gate = evaluate_gates(
            teach_pick=None,
            teacher_pick=None,
            eligible=["M0"],
            delta_m0=0.0,
            ci_lo_m0=0.0,
            delta_teacher=0.0,
            ci_lo_teacher=0.0,
            n_pos_seeds_vs_m0=0,
            teach_vs_init=0.0,
            teacher_j_teach=j_m0,
            teacher_j_m0=j_m0,
            min_delta_m0_pp=float(cfg["gate"]["min_delta_m0_pp"]),
            teacher_slack_pp=float(cfg["gate"]["teacher_slack_pp"]),
        )
        dump_json(reports / "T0_gate.json", gate)
        _write_summary(reports, cfg, teacher, probe, gate, extra={"note": "not identifiable before confirm"})
        _append_all_results(Path(cfg["paths"]["archive"]) / "ALL_RESULTS.md", gate)
        print(json.dumps(gate, indent=2), flush=True)
        return

    names = []
    for n in (teach_pick, teacher_pick, "M0"):
        if n and n not in names:
            names.append(n)
    recs = {"init": {}, **{n: {} for n in names}}
    for s in RUN_SEEDS:
        agent = build_student_agent(cfg, distill_path=adapters / f"init_seed{s}", trainable=False)
        recs["init"][s] = _eval(agent, tasks, cfg, reports / "confirm" / f"init_seed{s}.jsonl", s, args.resume)
        agent.close()
        for name in names:
            agent = build_student_agent(cfg, distill_path=adapters / f"probe_{name}_seed{s}", trainable=False)
            recs[name][s] = _eval(agent, tasks, cfg, reports / "confirm" / f"{name}_seed{s}.jsonl", s, args.resume)
            agent.close()

    means = {k: task_mean_success(v) for k, v in recs.items()}
    boot_m0 = task_mean_paired_bootstrap(
        means["M0"],
        means[teach_pick],
        n_boot=int(cfg["gate"]["n_boot"]),
        seed=int(cfg["gate"]["bootstrap_seed"]),
    )
    if teacher_pick and teacher_pick != teach_pick:
        boot_t = task_mean_paired_bootstrap(
            means[teacher_pick],
            means[teach_pick],
            n_boot=int(cfg["gate"]["n_boot"]),
            seed=int(cfg["gate"]["bootstrap_seed"]),
        )
    else:
        boot_t = {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_win": 0, "n_loss": 0, "n_tie": 0}

    seed_teach = {s: per_seed_success(recs[teach_pick][s]) for s in RUN_SEEDS}
    seed_m0 = {s: per_seed_success(recs["M0"][s]) for s in RUN_SEEDS}
    seed_init = {s: per_seed_success(recs["init"][s]) for s in RUN_SEEDS}
    n_pos = sum(1 for s in RUN_SEEDS if seed_teach[s] > seed_m0[s])
    j_teach = sum(seed_teach.values()) / 3
    j_m0 = sum(seed_m0.values()) / 3
    j_init = sum(seed_init.values()) / 3
    j_teacher_pick = sum(per_seed_success(recs[teacher_pick][s]) for s in RUN_SEEDS) / 3 if teacher_pick else j_m0

    gate = evaluate_gates(
        teach_pick=teach_pick,
        teacher_pick=teacher_pick,
        eligible=eligible,
        delta_m0=j_teach - j_m0,
        ci_lo_m0=boot_m0["ci_lo"],
        delta_teacher=j_teach - j_teacher_pick,
        ci_lo_teacher=boot_t["ci_lo"],
        n_pos_seeds_vs_m0=n_pos,
        teach_vs_init=j_teach - j_init,
        teacher_j_teach=teacher["results"][teach_pick]["mean_success"],
        teacher_j_m0=teacher["results"]["M0"]["mean_success"],
        min_delta_m0_pp=float(cfg["gate"]["min_delta_m0_pp"]),
        teacher_slack_pp=float(cfg["gate"]["teacher_slack_pp"]),
    )
    gate["confirm"] = {
        "success_mean": {k: sum(per_seed_success(recs[k][s]) for s in RUN_SEEDS) / 3 for k in recs},
        "per_seed": {
            "teach_pick": seed_teach,
            "M0": seed_m0,
            "init": seed_init,
            "teacher_pick": {s: per_seed_success(recs[teacher_pick][s]) for s in RUN_SEEDS} if teacher_pick else {},
        },
        "bootstrap_vs_m0": boot_m0,
        "bootstrap_vs_teacher": boot_t,
        "invalid_action_rate": {
            k: trajectory_metrics([r for s in RUN_SEEDS for r in recs[k][s]]).get("admissible_action_rate")
            for k in recs
        },
        "mean_steps": {
            k: sum(int(r.get("trajectory_length") or 0) for s in RUN_SEEDS for r in recs[k][s])
            / max(1, sum(len(recs[k][s]) for s in RUN_SEEDS))
            for k in recs
        },
        "ranking_meta_select": {
            "U_teach": probe.get("U_teach"),
            "U_teacher": {c: teacher["results"][c]["U_teacher"] for c in eligible if c in teacher["results"]},
        },
        "ranking_probe_confirm": {
            k: sum(per_seed_success(recs[k][s]) for s in RUN_SEEDS) / 3 for k in names
        },
    }
    dump_json(reports / "T0_gate.json", gate)
    _write_summary(reports, cfg, teacher, probe, gate)
    _append_all_results(Path(cfg["paths"]["archive"]) / "ALL_RESULTS.md", gate)
    print(json.dumps({k: gate[k] for k in ("T0_A", "T0_B", "method_gate_pass", "claim", "teach_pick", "teacher_pick")}, indent=2), flush=True)


def _write_summary(reports: Path, cfg, teacher, probe, gate, extra=None) -> None:
    lines = [
        "# TAME-OPD Gate T0 summary",
        "",
        "T0 PASS only means brief no-workflow student utility can serve as a memory selection target.",
        "It is not a complete scalable TAME method.",
        "",
        f"- teach_pick: `{gate.get('teach_pick')}`",
        f"- teacher_pick: `{gate.get('teacher_pick')}`",
        f"- eligible: `{gate.get('eligible')}`",
        f"- T0-A: `{gate['T0_A']}`",
        f"- T0-B: `{gate['T0_B']}`",
        f"- method_gate_pass: `{gate.get('method_gate_pass')}`",
        "",
        gate.get("claim", ""),
        "",
        "Stopped after Gate T0. No multi-round training.",
        "",
        "Mutation audit: M1-M4 fallback rates exceeded 5% after deterministic",
        "revert of instance-number leaks (no regeneration). Population-level",
        "teaching-utility selection is not identifiable.",
    ]
    if extra:
        lines.append("")
        lines.append(json.dumps(extra, indent=2))
    (reports / "T0_summary.md").write_text("\n".join(lines) + "\n")


def _append_all_results(path: Path, gate: dict) -> None:
    marker = "## TAME-OPD Gate T0"
    block = (
        f"\n{marker}\n\n"
        "更新：2026-09-05。train-only manifest；valid_unseen 未打开。\n\n"
        f"- teach_pick: `{gate.get('teach_pick')}` · teacher_pick: `{gate.get('teacher_pick')}`\n"
        f"- T0-A: `{gate.get('T0_A')}`\n"
        f"- T0-B: `{gate.get('T0_B')}`\n"
        f"- method_gate_pass: `{gate.get('method_gate_pass')}`\n"
        f"- 主张：{gate.get('claim')}\n"
        "- 产物：`reports/tame_opd/` · `notes/tame_opd_plan.md`\n"
        "- 停止。不自动多轮训练。不把结果写成完整 TAME。\n"
    )
    text = path.read_text() if path.exists() else ""
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n" + block)


if __name__ == "__main__":
    main()
