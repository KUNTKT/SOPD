#!/usr/bin/env python3
"""H1/H2: textual workflow ± NPM retrieval teacher on valid_unseen."""

from __future__ import annotations

import argparse
import json
import shutil
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
    load_eval_tasks,
    load_jsonl,
    load_yaml_cfg,
    paired_task_bootstrap,
    trajectory_metrics,
)
from steerable_alfworld_agent import build_agent  # noqa: E402


def load_workflows(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    bank = payload.get("workflows") or payload
    out: dict[str, str] = {}
    for tt, item in bank.items():
        if isinstance(item, dict):
            out[str(tt)] = str(item["text"])
        else:
            out[str(tt)] = str(item)
    return out


def slim_metrics(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "by_task_type"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_hybrid.yaml"),
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    steer = cfg["steering"]
    workflows = load_workflows(Path(cfg["paths"]["workflows"]))
    memory = NpmMemory(Path(cfg["paths"]["memory_dir"]) / "npm_memory.npz")
    tasks, task_meta = load_eval_tasks(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    split = str(cfg["env"].get("eval_split", "valid_unseen"))
    alpha = float(steer.get("hybrid_alpha", 1.0))
    top_k = int(steer.get("top_k", 8))
    min_pp = float(steer.get("gate_min_delta_pp", 5.0))

    def workflow_fn(task: dict) -> str:
        tt = str(task.get("task_type") or "")
        return workflows.get(tt) or workflows.get("pick_and_place_simple", "")

    agent = build_agent(cfg)
    conditions: dict[str, Any] = {}
    recs_cache: dict[str, list[dict]] = {}

    # H0: reuse A1 same-runner base
    h0_src = Path(cfg["paths"]["h0_base"])
    h0_dst = reports / "H0_base.jsonl"
    print("H0 reuse custom base", flush=True)
    if not h0_src.exists():
        raise SystemExit(f"missing H0 base {h0_src}")
    recs0 = load_jsonl(h0_src)
    if len(recs0) < len(tasks):
        raise SystemExit(f"H0 base n={len(recs0)} < {len(tasks)}")
    h0_dst.parent.mkdir(parents=True, exist_ok=True)
    if not h0_dst.exists():
        shutil.copy2(h0_src, h0_dst)
    m0 = trajectory_metrics(recs0)
    conditions["H0"] = {"metrics": slim_metrics(m0), "out": str(h0_dst), "gain_pp": 0.0}
    recs_cache["H0"] = recs0
    print(f"  success={m0['success_rate']:.3f}", flush=True)

    # H1: workflow only
    print("H1 workflow-only", flush=True)
    t0 = time.time()
    recs1 = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=None,
        alpha=0.0,
        top_k=top_k,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=reports / "H1_workflow.jsonl",
        resume=args.resume,
        dataset_split=split,
        workflow_fn=workflow_fn,
    )
    m1 = trajectory_metrics(recs1)
    conditions["H1"] = {
        "metrics": slim_metrics(m1),
        "wall_s": time.time() - t0,
        "out": str(reports / "H1_workflow.jsonl"),
        "gain_pp": (m1["success_rate"] - m0["success_rate"]) * 100,
    }
    recs_cache["H1"] = recs1
    print(f"  success={m1['success_rate']:.3f} gain_pp={conditions['H1']['gain_pp']:+.1f}", flush=True)

    # H2: workflow + NPM α=+1.0, no gate
    print(f"H2 workflow+NPM alpha={alpha}", flush=True)
    t1 = time.time()
    recs2 = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=memory,
        alpha=alpha,
        top_k=top_k,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=reports / "H2_workflow_npm.jsonl",
        resume=args.resume,
        dataset_split=split,
        workflow_fn=workflow_fn,
    )
    m2 = trajectory_metrics(recs2)
    conditions["H2"] = {
        "metrics": slim_metrics(m2),
        "wall_s": time.time() - t1,
        "out": str(reports / "H2_workflow_npm.jsonl"),
        "gain_pp": (m2["success_rate"] - m0["success_rate"]) * 100,
    }
    recs_cache["H2"] = recs2
    print(f"  success={m2['success_rate']:.3f} gain_pp={conditions['H2']['gain_pp']:+.1f}", flush=True)

    agent.close()

    best = max(("H1", "H2"), key=lambda k: conditions[k]["gain_pp"])
    boot = paired_task_bootstrap(recs0, recs_cache[best])
    h2_vs_h1 = paired_task_bootstrap(recs1, recs2)
    gate = {
        "best_key": best,
        "delta_success_pp": conditions[best]["gain_pp"],
        "bootstrap_vs_H0": boot,
        "h2_minus_h1_pp": (m2["success_rate"] - m1["success_rate"]) * 100,
        "bootstrap_h2_vs_h1": h2_vs_h1,
        "gate_pass": conditions[best]["gain_pp"] >= min_pp and boot["ci_lo"] > 0,
        "min_delta_pp": min_pp,
    }
    payload = {"task_meta": task_meta, "conditions": conditions, "analysis": gate}
    dump_json(reports / "results.json", payload)
    print("HYBRID_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
