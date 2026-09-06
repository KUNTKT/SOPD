#!/usr/bin/env python3
"""A2: Collect steered / vanilla teacher trajectories for Agent-SSOPD distill."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_jsonl,
    load_task_pools,
    load_yaml_cfg,
    trajectory_metrics,
)
from agent_ssopd_teacher_eval import NpmMemory, alpha_tag, run_npm_episodes  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_alfworld.yaml"),
    )
    ap.add_argument("--teacher-key", default=None, help="e.g. npm:-1.5 from a1 analysis")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    a1 = json.loads(Path(cfg["paths"]["reports_dir"], "a1_results.json").read_text())
    gate = a1.get("analysis", {})
    if not gate.get("gate_pass"):
        raise SystemExit(f"A1 gate failed: {gate}; refuse distill collection")

    teacher_key = args.teacher_key or gate["best_key"]
    # Parse "npm:-1.5" or "npm_gated:1.0"
    if ":" not in teacher_key:
        raise SystemExit(f"bad teacher key {teacher_key}")
    tag, alpha_s = teacher_key.rsplit(":", 1)
    alpha = float(alpha_s)
    use_gate = tag == "npm_gated"
    if tag not in ("npm", "npm_gated"):
        raise SystemExit(f"collect only supports npm teachers, got {tag}")

    limit = int(cfg.get("distill", {}).get("select_limit", 120))
    pools = load_task_pools(limit=max(limit * 3, 400))
    tasks = pools["pools"]["audit_select"][:limit]
    memory = NpmMemory(Path(cfg["paths"]["memory_dir"]) / "npm_memory.npz")
    out_dir = Path(cfg["paths"]["reports_dir"]) / "a2_collect"
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(cfg)
    # Vanilla
    print(f"collect vanilla n={len(tasks)}", flush=True)
    t0 = time.time()
    vanilla = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=None,
        alpha=0.0,
        top_k=int(cfg["steering"].get("top_k", 8)),
        use_gate=False,
        gate_threshold=float(cfg["steering"].get("gate_threshold", 0.45)),
        static_vector=None,
        out_path=out_dir / "vanilla.jsonl",
        resume=args.resume,
        dataset_split="audit_select",
    )
    print("vanilla", trajectory_metrics(vanilla)["success_rate"], flush=True)

    print(f"collect steered {teacher_key} n={len(tasks)}", flush=True)
    steered = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=memory,
        alpha=alpha,
        top_k=int(cfg["steering"].get("top_k", 8)),
        use_gate=use_gate,
        gate_threshold=float(cfg["steering"].get("gate_threshold", 0.45)),
        static_vector=None,
        out_path=out_dir / f"steered_{tag}_{alpha_tag(alpha)}.jsonl",
        resume=args.resume,
        dataset_split="audit_select",
    )
    print("steered", trajectory_metrics(steered)["success_rate"], flush=True)
    agent.close()

    def to_sft(recs: list[dict], role: str) -> list[dict]:
        rows = []
        for r in recs:
            if not r.get("episode_success"):
                continue
            # Pack multi-step as concatenated assistant actions for SFT.
            actions = r.get("actions") or []
            prompt = r.get("prompt_text") or ""
            completion = "\n".join(str(a) for a in actions)
            rows.append(
                {
                    "problem_id": r["task_id"],
                    "trajectory_id": r["trajectory_id"],
                    "prompt_text": prompt,
                    "completion_text": completion,
                    "correct": 1,
                    "role": role,
                    "episode_success": True,
                }
            )
        return rows

    vanilla_ok = to_sft(vanilla, "vanilla_teacher")
    steered_ok = to_sft(steered, "steered_teacher")
    dump_json(out_dir / "vanilla_correct.json", vanilla_ok)
    dump_json(out_dir / "steered_correct.json", steered_ok)
    # also jsonl
    from rollout.resume import append_jsonl

    for name, rows in (("vanilla_correct.jsonl", vanilla_ok), ("steered_correct.jsonl", steered_ok)):
        p = out_dir / name
        p.write_text("")
        for row in rows:
            append_jsonl(p, row)

    meta = {
        "teacher_key": teacher_key,
        "n_tasks": len(tasks),
        "vanilla_success": trajectory_metrics(vanilla)["success_rate"],
        "steered_success": trajectory_metrics(steered)["success_rate"],
        "n_vanilla_correct": len(vanilla_ok),
        "n_steered_correct": len(steered_ok),
        "wall_s": time.time() - t0,
    }
    dump_json(out_dir / "collect_meta.json", meta)
    print("COLLECT_DONE", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
