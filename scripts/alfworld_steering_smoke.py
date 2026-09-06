#!/usr/bin/env python3
"""P0: ALFWorld steering smoke + base policy gate (archive)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, ensure_alfworld_env, load_task_pools, load_yaml_cfg, make_env_factory, trajectory_metrics  # noqa: E402
from rollout.multistep import collect_episodes  # noqa: E402
from steerable_alfworld_agent import build_agent, build_rollout_agent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_smoke.yaml"),
    )
    ap.add_argument("--adapter-path", default=None, help="optional LoRA after coldstart")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    if args.adapter_path:
        cfg.setdefault("model", {})["adapter_path"] = args.adapter_path
    cfg.setdefault("rollout", {})["backend"] = args.backend

    smoke = cfg.get("smoke", {})
    n_sel = int(smoke.get("n_select", 10))
    n_conf = int(smoke.get("n_confirm", 10))

    payload = load_task_pools(
        data_root=cfg["env"]["data_root"],
        limit=int(cfg["env"].get("limit", 1200)),
        split=cfg["env"].get("split", "train"),
    )
    pools = payload["pools"]
    tasks = pools["audit_select"][:n_sel] + pools["audit_confirm"][:n_conf]
    print(f"smoke tasks={len(tasks)} select={n_sel} confirm={n_conf}", flush=True)

    out = Path(cfg["paths"]["rollouts"])
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    agent = build_rollout_agent(cfg) if args.backend == "vllm" else build_agent(cfg)
    t0 = time.time()
    n_done = [0]

    def _progress(rec: dict) -> None:
        n_done[0] += 1
        if n_done[0] % 5 == 0:
            print(
                f"  rollout {n_done[0]} task={rec.get('task_id')} "
                f"success={rec.get('episode_success')} steps={rec.get('trajectory_length')}",
                flush=True,
            )

    records = collect_episodes(
        tasks,
        agent,
        make_env_factory(cfg),
        n_rollouts=int(cfg["rollout"]["n_rollouts"]),
        dataset_split="smoke",
        base_seed=int(cfg["rollout"]["seed"]),
        out_path=out,
        resume=False,
        max_steps=int(cfg["rollout"]["max_steps"]),
        batch_size=int(cfg["rollout"]["batch_size"]),
        n_env_workers=int(cfg["rollout"].get("env_workers", 4)),
        progress_cb=_progress,
    )
    agent.close() if hasattr(agent, "close") else None
    metrics = trajectory_metrics(records)
    gate_ok = (
        metrics["success_rate"] >= float(smoke.get("gate_min_success", 0.05))
        and metrics["admissible_action_rate"] >= float(smoke.get("gate_min_admissible", 0.35))
    )
    result = {
        "stage": "P0_smoke",
        "metrics": metrics,
        "gate_pass": gate_ok,
        "gate_thresholds": {
            "min_success": smoke.get("gate_min_success", 0.05),
            "min_admissible": smoke.get("gate_min_admissible", 0.35),
        },
        "partition_meta": payload["meta"],
        "wall_s": time.time() - t0,
        "n_tasks": len(tasks),
        "adapter_path": cfg.get("model", {}).get("adapter_path"),
    }
    dump_json(reports / "smoke_results.json", result)
    dump_json(reports / "partition_meta.json", payload["meta"])
    print("SMOKE", json.dumps(result, indent=2), flush=True)
    if not gate_ok:
        print(
            "GATE_FAIL: run Qwen3 coldstart before P1 "
            "(wrap ssopd_logits/experiments/alfworld_coldstart.py with model.path=Qwen3-1.7B)",
            flush=True,
        )
        sys.exit(2)
    print("GATE_PASS", flush=True)


if __name__ == "__main__":
    main()
