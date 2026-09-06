#!/usr/bin/env python3
"""D1 memory–parameter alternation. Run only after Gate D0 PASS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from agent_uce_opd_eval import eval_adapter  # noqa: E402
from agent_uce_opd_lib import build_student_agent, load_opd_cfg, task_splits  # noqa: E402
from agent_uce_opd_train import train_opd, train_sft  # noqa: E402
from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_jsonl,
    paired_task_bootstrap,
    trajectory_metrics,
)
from uce_eval import slim_metrics  # noqa: E402
from uce_library import UceLibrary, make_entry  # noqa: E402


def require_d0_pass(reports: Path) -> dict:
    summary = json.loads((reports / "summary.json").read_text())
    if not summary.get("gate", {}).get("gate_pass"):
        raise SystemExit("Gate D0 FAIL; D1 is not authorized")
    return summary


def build_m1_library(recs: list[dict], tasks: list[dict]) -> tuple[UceLibrary, dict]:
    by_id = {str(t["task_id"]): t for t in tasks}
    lib = UceLibrary()
    n_ok = 0
    n_skip = 0
    for rec in recs:
        if not rec.get("episode_success"):
            continue
        n_ok += 1
        task = by_id.get(str(rec.get("task_id")), {})
        entry = make_entry(
            entry_id=f"m1_{rec.get('task_id')}",
            task_type=str(task.get("task_type") or rec.get("task_type") or "unknown"),
            goal=str(task.get("goal") or rec.get("goal") or ""),
            actions=rec.get("actions") or [],
            source_task_id=str(rec.get("task_id") or ""),
            usage=1,
        )
        if not entry["lines"]:
            n_skip += 1
            continue
        lib.add(entry)
    stats = {
        "n_rollouts": len(recs),
        "n_success": n_ok,
        "n_skip_empty": n_skip,
        "n_entries": len(lib.entries),
        "note": "fresh library from theta1 memory-free successes; no M0 copy; no usage bump",
    }
    return lib, stats


def collect_theta1_evolve(cfg, adapter: Path, tasks: list[dict], out_path: Path, resume: bool) -> list[dict]:
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= len(tasks):
            print(f"  reuse M1 rollouts n={len(recs)}", flush=True)
            return recs[: len(tasks)]
    agent = build_student_agent(cfg, distill_path=adapter, trainable=False)
    agent.clear_steering()
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
        dataset_split="audit_select_evolve_m1",
        workflow_fn=None,
    )
    agent.close()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_opd_cfg(args.config)
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    adapters = Path(cfg["paths"]["adapters_dir"])
    d1_rep = reports / "d1"
    d1_rep.mkdir(parents=True, exist_ok=True)
    d0 = require_d0_pass(reports)
    splits = task_splits(cfg)
    evolve = list(splits["evolve"])
    distill = list(splits["distill"])
    eval_tasks = list(splits["eval"])
    theta1 = adapters / "uce_opd"
    if not (theta1 / "adapter_config.json").exists():
        raise SystemExit(f"missing θ1 adapter {theta1}")

    print("D1 collect θ1 memory-free on evolve 80", flush=True)
    m1_jsonl = d1_rep / "m1_theta1_evolve.jsonl"
    recs_m1 = collect_theta1_evolve(cfg, theta1, evolve, m1_jsonl, args.resume)
    lib_m1, m1_stats = build_m1_library(recs_m1, evolve)
    lib_m1.save(
        d1_rep / "uce_library_m1.json",
        **m1_stats,
        source="theta1_memory_free_evolve80",
    )
    dump_json(d1_rep / "m1_build.json", m1_stats)
    print(json.dumps(m1_stats), flush=True)

    lib_m0 = UceLibrary.load(Path(cfg["paths"]["uce_library_evolved"]))

    print("D1 Re-evolved M1 OPD (θ1 → θ2)", flush=True)
    stats_m1 = train_opd(
        cfg,
        tasks=distill,
        lib=lib_m1,
        adapter_out=adapters / "d1_theta2_m1",
        reports=d1_rep / "theta2_m1",
        teacher_uce=True,
        resume=args.resume,
        limit_tasks=None,
        init_adapter=theta1,
    )
    dump_json(d1_rep / "train_theta2_m1.json", stats_m1)

    print("D1 Fixed-memory M0 continuation", flush=True)
    stats_m0 = train_opd(
        cfg,
        tasks=distill,
        lib=lib_m0,
        adapter_out=adapters / "d1_theta2_m0",
        reports=d1_rep / "theta2_m0",
        teacher_uce=True,
        resume=args.resume,
        limit_tasks=None,
        init_adapter=theta1,
    )
    dump_json(d1_rep / "train_theta2_m0.json", stats_m0)

    print("D1 Continued Vanilla", flush=True)
    van_recs = load_jsonl(reports / "collect_vanilla_theta0.jsonl")
    stats_van = train_sft(
        cfg,
        van_recs,
        adapters / "d1_vanilla_cont",
        uce=False,
        init_adapter=adapters / "vanilla_sft",
    )
    dump_json(d1_rep / "train_vanilla_cont.json", stats_van)

    recs_th1 = load_jsonl(reports / "eval_uce_opd.jsonl")
    names = [
        ("theta2_m1", adapters / "d1_theta2_m1", d1_rep / "eval_theta2_m1.jsonl"),
        ("theta2_m0", adapters / "d1_theta2_m0", d1_rep / "eval_theta2_m0.jsonl"),
        ("vanilla_cont", adapters / "d1_vanilla_cont", d1_rep / "eval_vanilla_cont.jsonl"),
    ]
    recs_by = {"theta1": recs_th1}
    results = {
        "theta1": {
            "success_rate": trajectory_metrics(recs_th1)["success_rate"],
            "metrics": slim_metrics(trajectory_metrics(recs_th1)),
        },
        "d0": {k: v for k, v in d0.get("results", {}).items() if k in {"H0", "uce_opd"}},
    }
    for name, adapter, outp in names:
        print(f"eval {name}", flush=True)
        recs = eval_adapter(cfg, adapter, eval_tasks, outp, args.resume)
        recs_by[name] = recs
        m = trajectory_metrics(recs)
        results[name] = {"success_rate": m["success_rate"], "metrics": slim_metrics(m)}
        print(f"  {name} success={m['success_rate']:.3f}", flush=True)

    dcfg = cfg["distill"]
    n_boot = int(dcfg.get("n_boot", 10000))
    seed = int(dcfg.get("bootstrap_seed", 0))
    boots = {
        "theta2_m1_vs_theta1": paired_task_bootstrap(
            recs_by["theta1"], recs_by["theta2_m1"], n_boot=n_boot, seed=seed
        ),
        "theta2_m1_vs_theta2_m0": paired_task_bootstrap(
            recs_by["theta2_m0"], recs_by["theta2_m1"], n_boot=n_boot, seed=seed
        ),
    }
    dump_json(d1_rep / "paired_bootstrap.json", boots)
    s1 = results["theta1"]["success_rate"]
    s2m1 = results["theta2_m1"]["success_rate"]
    s2m0 = results["theta2_m0"]["success_rate"]
    min_pp = float(dcfg.get("gate_min_delta_pp", 2.0))
    d_pp = (s2m1 - s1) * 100
    gate = {
        "delta_theta1_pp": d_pp,
        "delta_m0_pp": (s2m1 - s2m0) * 100,
        "bootstrap": boots,
        "min_delta_pp": min_pp,
        "gate_pass": bool(
            d_pp >= min_pp and boots["theta2_m1_vs_theta1"]["ci_lo"] > 0 and s2m1 > s2m0
        ),
        "eval_protocol": "memory_free_hf_generate",
    }
    payload = {"results": results, "gate": gate, "m1": m1_stats}
    dump_json(d1_rep / "summary.json", payload)
    print(json.dumps(gate, indent=2))
    print("Gate D1", "PASS" if gate["gate_pass"] else "FAIL")


if __name__ == "__main__":
    main()
