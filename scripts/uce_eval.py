#!/usr/bin/env python3
"""B-ret / B-uce: instance-level UCE workflow retrieve ± evolve on ALFWorld."""

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

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_eval_tasks,
    load_jsonl,
    load_task_pools,
    load_yaml_cfg,
    paired_task_bootstrap,
    trajectory_metrics,
)
from steerable_alfworld_agent import build_agent  # noqa: E402
from uce_library import UceLibrary, make_entry  # noqa: E402


def slim_metrics(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "by_task_type"}


def make_workflow_fn(lib: UceLibrary, hits: list[dict] | None = None):
    def workflow_fn(task: dict) -> str:
        text, eid, score = lib.retrieve(task)
        if hits is not None:
            hits.append(
                {
                    "task_id": task.get("task_id"),
                    "task_type": task.get("task_type"),
                    "entry_id": eid,
                    "score": score,
                }
            )
        return text

    return workflow_fn


def eval_condition(
    *,
    name: str,
    agent,
    tasks: list[dict],
    cfg: dict,
    lib: UceLibrary,
    out_path: Path,
    split: str,
    resume: bool,
) -> tuple[list[dict], dict, list[dict]]:
    hits: list[dict] = []
    t0 = time.time()
    recs = run_npm_episodes(
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
        workflow_fn=make_workflow_fn(lib, hits),
    )
    metrics = trajectory_metrics(recs)
    info = {
        "metrics": slim_metrics(metrics),
        "wall_s": time.time() - t0,
        "out": str(out_path),
        "n_retrieve_hits": len(hits),
        "mean_retrieve_score": (
            sum(h["score"] for h in hits) / len(hits) if hits else 0.0
        ),
    }
    return recs, info, hits


def evolve_library(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    lib: UceLibrary,
    out_path: Path,
    resume: bool,
) -> dict[str, Any]:
    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume evolve skip {out_path.name} n={len(recs)}", flush=True)
            n_ok = sum(1 for r in recs if r.get("episode_success"))
            return {
                "n": len(recs),
                "n_success": n_ok,
                "n_fail": len(recs) - n_ok,
                "n_added": 0,
                "n_pruned": 0,
                "resumed": True,
                "out": str(out_path),
            }

    hits_by_task: dict[str, tuple[str | None, float]] = {}

    def workflow_fn(task: dict) -> str:
        text, eid, score = lib.retrieve(task)
        hits_by_task[str(task.get("task_id"))] = (eid, score)
        return text

    recs = run_npm_episodes(
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
        resume=False,
        dataset_split="audit_select_evolve",
        workflow_fn=workflow_fn,
    )
    n_added = 0
    n_ok = 0
    for rec, task in zip(recs, tasks):
        tid = str(task.get("task_id"))
        eid, _ = hits_by_task.get(tid, (None, 0.0))
        ok = bool(rec.get("episode_success"))
        if ok:
            n_ok += 1
            lib.bump(eid, +1)
            lib.add(
                make_entry(
                    entry_id=f"evo_{tid}",
                    task_type=str(task.get("task_type") or "unknown"),
                    goal=str(task.get("goal") or ""),
                    actions=rec.get("actions") or [],
                    source_task_id=tid,
                    usage=1,
                )
            )
            n_added += 1
        else:
            lib.bump(eid, -1)
    n_pruned = lib.prune()
    return {
        "n": len(recs),
        "n_success": n_ok,
        "n_fail": len(recs) - n_ok,
        "n_added": n_added,
        "n_pruned": n_pruned,
        "resumed": False,
        "out": str(out_path),
        "n_entries_after": len(lib.entries),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_uce_alfworld.yaml"),
    )
    ap.add_argument("--phase", choices=["ret", "uce", "all"], default="all")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    min_pp = float(cfg["steering"].get("gate_min_delta_pp", 5.0))
    split = str(cfg["env"].get("eval_split", "valid_unseen"))

    tasks, task_meta = load_eval_tasks(cfg)
    lib = UceLibrary.load(Path(cfg["paths"]["uce_library"]))

    h0_src = Path(cfg["paths"]["h0_base"])
    if not h0_src.exists():
        raise SystemExit(f"missing H0 base {h0_src}")
    recs0 = load_jsonl(h0_src)
    if len(recs0) < len(tasks):
        raise SystemExit(f"H0 base n={len(recs0)} < {len(tasks)}")
    h0_dst = reports / "H0_base.jsonl"
    if not h0_dst.exists():
        shutil.copy2(h0_src, h0_dst)
    m0 = trajectory_metrics(recs0)
    conditions: dict[str, Any] = {
        "H0": {"metrics": slim_metrics(m0), "out": str(h0_dst), "gain_pp": 0.0}
    }
    recs_cache: dict[str, list[dict]] = {"H0": recs0}
    print(f"H0 success={m0['success_rate']:.3f}", flush=True)

    print("loading agent (HF + coldstart LoRA)...", flush=True)
    agent = build_agent(cfg)
    print("agent ready", flush=True)

    if args.phase in {"ret", "all"}:
        print("B-ret instance retrieve (frozen)", flush=True)
        recs_ret, info_ret, hits_ret = eval_condition(
            name="B_ret",
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            lib=lib,
            out_path=reports / "B_ret.jsonl",
            split=split,
            resume=args.resume,
        )
        info_ret["gain_pp"] = (info_ret["metrics"]["success_rate"] - m0["success_rate"]) * 100
        conditions["B_ret"] = info_ret
        recs_cache["B_ret"] = recs_ret
        dump_json(reports / "B_ret_hits.json", hits_ret)
        print(
            f"  success={info_ret['metrics']['success_rate']:.3f} "
            f"gain_pp={info_ret['gain_pp']:+.1f}",
            flush=True,
        )

    if args.phase in {"uce", "all"}:
        evolve_n = int(cfg["env"].get("evolve_n", 80))
        evo_path = Path(cfg["paths"]["uce_library_evolved"])
        evolve_jsonl = reports / "B_uce_evolve.jsonl"
        if args.resume and evo_path.exists() and evolve_jsonl.exists():
            recs_evo = load_jsonl(evolve_jsonl)
            if len(recs_evo) >= evolve_n:
                lib = UceLibrary.load(evo_path)
                evo_stats = json.loads(
                    (reports / "b_uce_evolve_stats.json").read_text()
                ) if (reports / "b_uce_evolve_stats.json").exists() else {
                    "n": len(recs_evo),
                    "resumed": True,
                    "n_entries_after": len(lib.entries),
                }
                print(
                    f"B-uce reuse evolved library n_entries={len(lib.entries)}",
                    flush=True,
                )
            else:
                evo_stats = None
        else:
            evo_stats = None
        if evo_stats is None:
            pools = load_task_pools(
                data_root=str(cfg["env"]["data_root"]),
                limit=1200,
                split="train",
                partition_seed=int(cfg.get("random_seed", 1010)),
            )
            evolve_tasks = list(pools["pools"]["audit_select"][:evolve_n])
            print(f"B-uce evolve n={len(evolve_tasks)}", flush=True)
            evo_stats = evolve_library(
                agent=agent,
                tasks=evolve_tasks,
                cfg=cfg,
                lib=lib,
                out_path=evolve_jsonl,
                resume=False,
            )
            lib.save(evo_path, evolve=evo_stats, source=str(cfg["paths"]["uce_library"]))
            dump_json(reports / "b_uce_evolve_stats.json", evo_stats)
        print(f"  evolve {json.dumps(evo_stats)}", flush=True)

        print("B-uce frozen eval", flush=True)
        recs_uce, info_uce, hits_uce = eval_condition(
            name="B_uce",
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            lib=lib,
            out_path=reports / "B_uce.jsonl",
            split=split,
            resume=args.resume,
        )
        info_uce["gain_pp"] = (info_uce["metrics"]["success_rate"] - m0["success_rate"]) * 100
        info_uce["evolve"] = evo_stats
        conditions["B_uce"] = info_uce
        recs_cache["B_uce"] = recs_uce
        dump_json(reports / "B_uce_hits.json", hits_uce)
        print(
            f"  success={info_uce['metrics']['success_rate']:.3f} "
            f"gain_pp={info_uce['gain_pp']:+.1f}",
            flush=True,
        )

    agent.close()

    scored = [k for k in ("B_ret", "B_uce") if k in conditions]
    if not scored:
        raise SystemExit("no B conditions ran")
    best = max(scored, key=lambda k: conditions[k]["gain_pp"])
    boot = paired_task_bootstrap(recs0, recs_cache[best])
    gate = {
        "best_key": best,
        "delta_success_pp": conditions[best]["gain_pp"],
        "bootstrap_vs_H0": boot,
        "gate_pass": conditions[best]["gain_pp"] >= min_pp and boot["ci_lo"] > 0,
        "min_delta_pp": min_pp,
        "distill": "skip_unless_gate",
    }
    for k in scored:
        if k == best:
            continue
        gate[f"bootstrap_{k}_vs_H0"] = paired_task_bootstrap(recs0, recs_cache[k])
    payload = {"task_meta": task_meta, "conditions": conditions, "analysis": gate}
    dump_json(reports / "results.json", payload)
    print("UCE_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
