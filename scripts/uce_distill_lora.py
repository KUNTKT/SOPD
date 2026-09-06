#!/usr/bin/env python3
"""Distill UCE/hybrid teacher vs vanilla; eval with custom runner (no collect_episodes)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_distill_lora import train_lora  # noqa: E402
from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
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
from uce_eval import slim_metrics  # noqa: E402


def eval_adapter(cfg: dict, adapter: Path, out_jsonl: Path, resume: bool) -> list[dict]:
    cfg2 = json.loads(json.dumps(cfg))
    cfg2.setdefault("model", {})["adapter_path"] = str(adapter)
    agent = build_agent(cfg2)
    tasks, _ = load_eval_tasks(cfg2)
    recs = run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg2,
        memory=None,
        alpha=0.0,
        top_k=8,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_jsonl,
        resume=resume,
        dataset_split=str(cfg2["env"].get("eval_split", "valid_unseen")),
        workflow_fn=None,
    )
    agent.close()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_uce_npm_hybrid.yaml"),
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    h3 = json.loads(Path(cfg["paths"]["reports_dir"], "results.json").read_text())
    collect_dir = Path(cfg["paths"]["collect_dir"])
    out_dir = Path(cfg["paths"]["distill_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((collect_dir / "collect_meta.json").read_text())

    results: dict = {}
    recs_by: dict[str, list] = {}
    for name, jsonl in (
        ("teacher", collect_dir / "teacher_correct.jsonl"),
        ("vanilla", collect_dir / "vanilla_correct.jsonl"),
    ):
        adapter = out_dir / f"lora_{name}"
        if not args.skip_train or not adapter.exists():
            print(f"train LoRA from {jsonl}", flush=True)
            results[f"train_{name}"] = train_lora(cfg, jsonl, adapter)
        print(f"eval {name} (custom runner, no workflow)", flush=True)
        recs = eval_adapter(cfg, adapter, out_dir / f"eval_{name}.jsonl", args.resume)
        recs_by[name] = recs
        metrics = trajectory_metrics(recs)
        results[f"eval_{name}"] = slim_metrics(metrics)
        print(f"  success={metrics['success_rate']:.3f}", flush=True)

    recs0 = load_jsonl(Path(cfg["paths"]["h0_base"]))
    m0 = trajectory_metrics(recs0)
    teacher_m = results["eval_teacher"]
    vanilla_m = results["eval_vanilla"]
    min_pp = float(cfg["steering"].get("distill_min_delta_pp", 2.0))
    boot_h0 = paired_task_bootstrap(recs0, recs_by["teacher"])
    boot_van = paired_task_bootstrap(recs_by["vanilla"], recs_by["teacher"])
    delta_h0 = (teacher_m["success_rate"] - m0["success_rate"]) * 100
    delta_van = (teacher_m["success_rate"] - vanilla_m["success_rate"]) * 100
    gate = {
        "teacher": meta.get("teacher"),
        "teacher_alpha": meta.get("teacher_alpha"),
        "delta_vs_h0_pp": delta_h0,
        "delta_vs_vanilla_lora_pp": delta_van,
        "bootstrap_vs_h0": boot_h0,
        "bootstrap_vs_vanilla": boot_van,
        "gate_pass": (
            delta_h0 >= min_pp
            and boot_h0["ci_lo"] > 0
            and teacher_m["success_rate"] >= vanilla_m["success_rate"]
        ),
        "min_delta_pp": min_pp,
        "h0_success": m0["success_rate"],
        "teacher_lora_success": teacher_m["success_rate"],
        "vanilla_lora_success": vanilla_m["success_rate"],
    }
    payload = {
        "h3_analysis": h3.get("analysis"),
        "collect": meta,
        "results": results,
        "analysis": gate,
    }
    dump_json(out_dir / "results.json", payload)
    dump_json(Path(cfg["paths"]["reports_dir"]) / "distill_results.json", payload)
    print("DISTILL_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
