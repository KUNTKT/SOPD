#!/usr/bin/env python3
"""H3: UCE-evolved workflow + NPM steering additivity on valid_unseen."""

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

from agent_ssopd_teacher_eval import NpmMemory, alpha_tag, run_npm_episodes  # noqa: E402
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
from uce_eval import make_workflow_fn, slim_metrics  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


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
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    min_pp = float(cfg["steering"].get("gate_min_delta_pp", 5.0))
    split = str(cfg["env"].get("eval_split", "valid_unseen"))
    alphas = [float(a) for a in cfg["steering"]["alphas"]]
    top_k = int(cfg["steering"].get("top_k", 8))
    thr = float(cfg["steering"].get("gate_threshold", 0.45))

    tasks, task_meta = load_eval_tasks(cfg)
    lib = UceLibrary.load(Path(cfg["paths"]["uce_library_evolved"]))
    memory = NpmMemory(Path(cfg["paths"]["memory_dir"]) / "npm_memory.npz")

    recs_h0 = load_jsonl(Path(cfg["paths"]["h0_base"]))
    recs_uce = load_jsonl(Path(cfg["paths"]["b_uce"]))
    if len(recs_h0) < len(tasks) or len(recs_uce) < len(tasks):
        raise SystemExit("missing H0 or B_uce records")
    h0_dst = reports / "H0_base.jsonl"
    uce_dst = reports / "B_uce.jsonl"
    if not h0_dst.exists():
        shutil.copy2(Path(cfg["paths"]["h0_base"]), h0_dst)
    if not uce_dst.exists():
        shutil.copy2(Path(cfg["paths"]["b_uce"]), uce_dst)

    m0 = trajectory_metrics(recs_h0)
    mu = trajectory_metrics(recs_uce)
    conditions: dict[str, Any] = {
        "H0": {"metrics": slim_metrics(m0), "out": str(h0_dst), "gain_vs_h0_pp": 0.0},
        "B_uce": {
            "metrics": slim_metrics(mu),
            "out": str(uce_dst),
            "gain_vs_h0_pp": (mu["success_rate"] - m0["success_rate"]) * 100,
            "gain_vs_uce_pp": 0.0,
        },
    }
    recs_cache = {"H0": recs_h0, "B_uce": recs_uce}
    print(
        f"H0={m0['success_rate']:.3f} B_uce={mu['success_rate']:.3f}",
        flush=True,
    )

    print("loading agent...", flush=True)
    agent = build_agent(cfg)
    print("agent ready", flush=True)
    workflow_fn = make_workflow_fn(lib)

    for alpha in alphas:
        key = f"H3_{alpha_tag(alpha)}"
        print(f"{key} UCE+NPM", flush=True)
        t0 = time.time()
        recs = run_npm_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            memory=memory,
            alpha=alpha,
            top_k=top_k,
            use_gate=False,
            gate_threshold=thr,
            static_vector=None,
            out_path=reports / f"{key}.jsonl",
            resume=args.resume,
            dataset_split=split,
            workflow_fn=workflow_fn,
        )
        metrics = trajectory_metrics(recs)
        conditions[key] = {
            "metrics": slim_metrics(metrics),
            "wall_s": time.time() - t0,
            "out": str(reports / f"{key}.jsonl"),
            "alpha": alpha,
            "gain_vs_h0_pp": (metrics["success_rate"] - m0["success_rate"]) * 100,
            "gain_vs_uce_pp": (metrics["success_rate"] - mu["success_rate"]) * 100,
        }
        recs_cache[key] = recs
        print(
            f"  success={metrics['success_rate']:.3f} "
            f"vs_uce={conditions[key]['gain_vs_uce_pp']:+.1f} pp",
            flush=True,
        )

    agent.close()
    h3_keys = [k for k in conditions if k.startswith("H3_")]
    best = max(h3_keys, key=lambda k: conditions[k]["gain_vs_uce_pp"])
    boot = paired_task_bootstrap(recs_uce, recs_cache[best])
    gate = {
        "best_key": best,
        "delta_vs_uce_pp": conditions[best]["gain_vs_uce_pp"],
        "delta_vs_h0_pp": conditions[best]["gain_vs_h0_pp"],
        "bootstrap_vs_uce": boot,
        "gate_pass": conditions[best]["gain_vs_uce_pp"] >= min_pp and boot["ci_lo"] > 0,
        "min_delta_pp": min_pp,
        "teacher": "uce_npm" if (
            conditions[best]["gain_vs_uce_pp"] >= min_pp and boot["ci_lo"] > 0
        ) else "uce",
        "teacher_alpha": (
            conditions[best]["alpha"]
            if conditions[best]["gain_vs_uce_pp"] >= min_pp and boot["ci_lo"] > 0
            else 0.0
        ),
    }
    payload = {"task_meta": task_meta, "conditions": conditions, "analysis": gate}
    dump_json(reports / "results.json", payload)
    print("H3_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
