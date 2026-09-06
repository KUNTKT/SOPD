#!/usr/bin/env python3
"""P1: Collect ALFWorld rollouts on audit_select for direction fitting (archive)."""

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
from rollout.multistep import collect_episodes, mixed_groups  # noqa: E402
from steerable_alfworld_agent import build_rollout_agent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_confirm.yaml"),
    )
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--n-rollouts", type=int, default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    if args.adapter_path:
        cfg.setdefault("model", {})["adapter_path"] = args.adapter_path

    payload = load_task_pools(
        data_root=cfg["env"]["data_root"],
        limit=int(cfg["env"].get("limit", 1200)),
    )
    tasks = payload["pools"]["audit_select"]
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]
    n_rollouts = int(args.n_rollouts or cfg["rollout"].get("n_rollouts", 4))
    out = Path(cfg["paths"]["rollouts_select"])
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    print(
        f"P1 rollouts pool=audit_select tasks={len(tasks)} K={n_rollouts} out={out}",
        flush=True,
    )
    agent = build_rollout_agent(cfg)
    cfg_rollout = dict(cfg)
    cfg_rollout["rollout"] = dict(cfg["rollout"])
    cfg_rollout["rollout"]["n_rollouts"] = n_rollouts

    t0 = time.time()
    n_done = [0]

    def _progress(rec: dict) -> None:
        n_done[0] += 1
        if n_done[0] % 20 == 0:
            print(
                f"  rollout {n_done[0]} task={rec.get('task_id')} "
                f"success={rec.get('episode_success')}",
                flush=True,
            )

    records = collect_episodes(
        tasks,
        agent,
        make_env_factory(cfg),
        n_rollouts=n_rollouts,
        dataset_split="audit_select",
        base_seed=int(cfg["rollout"]["seed"]),
        out_path=out,
        resume=bool(args.resume),
        max_steps=int(cfg["rollout"]["max_steps"]),
        batch_size=int(cfg["rollout"]["batch_size"]),
        n_env_workers=int(cfg["rollout"].get("env_workers", 6)),
        progress_cb=_progress,
    )
    agent.close() if hasattr(agent, "close") else None

    mixed = mixed_groups(records)
    metrics = trajectory_metrics(records)
    meta = {
        "stage": "P1_rollouts",
        "metrics": metrics,
        "n_mixed_groups": len(mixed),
        "n_paired_tasks": len(mixed),
        "wall_s": time.time() - t0,
        "out": str(out),
        "partition_meta": payload["meta"],
    }
    dump_json(reports / "p1_rollout_summary.json", meta)
    print("P1_DONE", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
