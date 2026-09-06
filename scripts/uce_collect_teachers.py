#!/usr/bin/env python3
"""Collect vanilla vs winner-teacher trajectories for UCE distill (per-step SFT)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import NpmMemory, run_npm_episodes  # noqa: E402
from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_jsonl,
    load_task_pools,
    load_yaml_cfg,
    trajectory_metrics,
)
from rollout.resume import append_jsonl  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402
from uce_eval import make_workflow_fn  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def strip_workflow(obs: str) -> str:
    text = str(obs or "")
    if text.startswith("Suggested workflow"):
        idx = text.find("\n\n")
        if idx >= 0:
            return text[idx + 2 :]
    return text


def pack_step_rows(recs: list[dict], agent, role: str) -> list[dict]:
    rows: list[dict] = []
    for rec in recs:
        if not rec.get("episode_success"):
            continue
        for dp in rec.get("decision_points") or []:
            raw = strip_workflow(str(dp.get("prefix_text") or ""))
            if not raw:
                continue
            action = str(
                dp.get("raw_action_text")
                or dp.get("action_text")
                or dp.get("predicted_tool")
                or ""
            )
            if not action:
                continue
            prompt = agent._prompt_text(raw)
            rows.append(
                {
                    "problem_id": rec["task_id"],
                    "trajectory_id": rec.get("trajectory_id"),
                    "step_index": dp.get("step_index"),
                    "prompt_text": prompt,
                    "completion_text": action,
                    "correct": 1,
                    "role": role,
                    "episode_success": True,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    for row in rows:
        append_jsonl(path, row)


def evolve_task_ids(cfg: dict) -> set[str]:
    path = Path(cfg["paths"]["evolve_jsonl"])
    return {str(r["task_id"]) for r in load_jsonl(path)}


def collect_pool(cfg: dict) -> list[dict]:
    evolve_n = int(cfg["env"].get("evolve_n", 80))
    limit = int(cfg.get("distill", {}).get("select_limit", 120))
    pools = load_task_pools(
        data_root=str(cfg["env"]["data_root"]),
        limit=1200,
        split="train",
        partition_seed=int(cfg.get("random_seed", 1010)),
    )
    audit = list(pools["pools"]["audit_select"])
    skip = {str(t["task_id"]) for t in audit[:evolve_n]}
    skip |= evolve_task_ids(cfg)
    rest = [t for t in audit if str(t["task_id"]) not in skip]
    return rest[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_uce_npm_hybrid.yaml"),
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    h3 = json.loads(Path(cfg["paths"]["reports_dir"], "results.json").read_text())
    gate = h3.get("analysis") or {}
    teacher = str(gate.get("teacher") or "uce")
    teacher_alpha = float(gate.get("teacher_alpha") or 0.0)
    if teacher not in {"uce", "uce_npm"}:
        raise SystemExit(f"bad teacher {teacher}")

    tasks = collect_pool(cfg)
    out_dir = Path(cfg["paths"]["collect_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = UceLibrary.load(Path(cfg["paths"]["uce_library_evolved"]))
    memory = None
    if teacher == "uce_npm":
        memory = NpmMemory(Path(cfg["paths"]["memory_dir"]) / "npm_memory.npz")
    workflow_fn = make_workflow_fn(lib)

    print(f"collect n={len(tasks)} teacher={teacher} alpha={teacher_alpha}", flush=True)
    agent = build_agent(cfg)
    t0 = time.time()

    print("vanilla (no workflow, alpha=0)", flush=True)
    vanilla = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=None,
        alpha=0.0,
        top_k=8,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_dir / "vanilla.jsonl",
        resume=args.resume,
        dataset_split="audit_select_distill",
        workflow_fn=None,
    )
    print("vanilla", trajectory_metrics(vanilla)["success_rate"], flush=True)

    print(f"teacher {teacher}", flush=True)
    steered = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=memory,
        alpha=teacher_alpha,
        top_k=int(cfg["steering"].get("top_k", 8)),
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_dir / f"teacher_{teacher}.jsonl",
        resume=args.resume,
        dataset_split="audit_select_distill",
        workflow_fn=workflow_fn,
    )
    print("teacher", trajectory_metrics(steered)["success_rate"], flush=True)

    vanilla_ok = pack_step_rows(vanilla, agent, "vanilla_teacher")
    teacher_ok = pack_step_rows(steered, agent, f"{teacher}_teacher")
    write_jsonl(out_dir / "vanilla_correct.jsonl", vanilla_ok)
    write_jsonl(out_dir / "teacher_correct.jsonl", teacher_ok)
    dump_json(out_dir / "vanilla_correct.json", vanilla_ok)
    dump_json(out_dir / "teacher_correct.json", teacher_ok)

    meta: dict[str, Any] = {
        "teacher": teacher,
        "teacher_alpha": teacher_alpha,
        "n_tasks": len(tasks),
        "task_ids": [t["task_id"] for t in tasks],
        "vanilla_success": trajectory_metrics(vanilla)["success_rate"],
        "teacher_success": trajectory_metrics(steered)["success_rate"],
        "n_vanilla_steps": len(vanilla_ok),
        "n_teacher_steps": len(teacher_ok),
        "wall_s": time.time() - t0,
    }
    dump_json(out_dir / "collect_meta.json", meta)
    agent.close()
    print("COLLECT_DONE", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
