#!/usr/bin/env python3
"""Collect student / θ0 / UCE-teacher rollouts for UCE-OPD."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from agent_uce_opd_lib import (  # noqa: E402
    apply_workflow,
    build_student_agent,
    load_opd_cfg,
    retrieve_readonly,
    strip_workflow,
    task_splits,
)
from alfworld_common import dump_json, ensure_alfworld_env, trajectory_metrics  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402
from uce_eval import make_workflow_fn  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def annotate_student_rows(recs: list[dict], lib: UceLibrary, tasks: list[dict]) -> list[dict]:
    by_id = {str(t["task_id"]): t for t in tasks}
    for rec in recs:
        task = by_id[str(rec["task_id"])]
        wf, eid, score = retrieve_readonly(lib, task)
        rec["workflow_text"] = wf
        rec["workflow_id"] = eid
        rec["retrieval_score"] = score
        for dp in rec.get("decision_points") or []:
            base = strip_workflow(str(dp.get("prefix_text") or ""))
            dp["prefix_base"] = base
            dp["prefix_uce"] = apply_workflow(base, wf)
            dp["workflow_text"] = wf
    return recs


def collect_episodes(cfg: dict, agent, tasks: list[dict], out_path: Path, *, workflow_fn, split: str, resume: bool):
    return run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=None,
        alpha=0.0,
        top_k=8,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_path,
        resume=resume,
        dataset_split=split,
        workflow_fn=workflow_fn,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["student", "vanilla_theta0", "uce_teacher"], required=True)
    ap.add_argument("--adapter", default=None, help="distill LoRA for student collect")
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_opd_cfg(args.config)
    ensure_alfworld_env(cfg)
    splits = task_splits(cfg)
    tasks = list(splits["distill"])
    if args.limit_tasks:
        tasks = tasks[: int(args.limit_tasks)]
    lib = UceLibrary.load(Path(cfg["paths"]["uce_library_evolved"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg_agent = copy.deepcopy(cfg)
    if args.mode == "student" and args.adapter:
        # Eval/collect student: merge happens in train loader; here load distill on merged θ0 via helper.
        agent = build_student_agent(cfg_agent, distill_path=Path(args.adapter), trainable=False)
    else:
        agent = build_agent(cfg_agent)
    agent.clear_steering()

    wf = make_workflow_fn(lib) if args.mode == "uce_teacher" else None
    t0 = time.time()
    recs = collect_episodes(
        cfg,
        agent,
        tasks,
        out,
        workflow_fn=wf,
        split=f"audit_select_{args.mode}",
        resume=args.resume,
    )
    if args.mode != "uce_teacher":
        recs = annotate_student_rows(recs, lib, tasks)
        # rewrite with annotations
        out.write_text("")
        from rollout.resume import append_jsonl

        for rec in recs:
            append_jsonl(out, rec)
    m = trajectory_metrics(recs)
    meta = {
        "mode": args.mode,
        "n": len(recs),
        "success_rate": m["success_rate"],
        "wall_s": time.time() - t0,
        "out": str(out),
        "task_ids": [t["task_id"] for t in tasks],
    }
    dump_json(out.with_suffix(".meta.json"), meta)
    print(json.dumps(meta, indent=2))
    agent.close()


if __name__ == "__main__":
    main()
