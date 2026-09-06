#!/usr/bin/env python3
"""Eval memory-free students and Gate D0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from agent_uce_opd_lib import build_student_agent, load_opd_cfg, task_splits  # noqa: E402
from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_jsonl,
    paired_task_bootstrap,
    trajectory_metrics,
)
from uce_eval import slim_metrics  # noqa: E402


def eval_adapter(cfg, adapter: Path, tasks: list[dict], out_jsonl: Path, resume: bool) -> list[dict]:
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
        out_path=out_jsonl,
        resume=resume,
        dataset_split=str(cfg["env"].get("eval_split", "valid_unseen")),
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
    dcfg = cfg["distill"]
    splits = task_splits(cfg)
    tasks = splits["eval"]

    recs0 = load_jsonl(Path(cfg["paths"]["h0_base"]))
    recs_uce = load_jsonl(Path(cfg["paths"]["b_uce"]))
    m0 = trajectory_metrics(recs0)
    mu = trajectory_metrics(recs_uce)

    names = [
        ("uce_opd", adapters / "uce_opd", reports / "eval_uce_opd.jsonl"),
        ("vanilla_sft", adapters / "vanilla_sft", reports / "eval_vanilla.jsonl"),
        ("no_pi_opd", adapters / "no_pi_opd", reports / "eval_no_pi_opd.jsonl"),
        ("uce_sft", adapters / "uce_sft", reports / "eval_uce_sft.jsonl"),
    ]
    recs_by: dict[str, list] = {}
    results = {
        "H0": {"success_rate": m0["success_rate"], "metrics": slim_metrics(m0)},
        "B_uce": {"success_rate": mu["success_rate"], "metrics": slim_metrics(mu)},
    }
    for name, adapter, outp in names:
        if not (adapter / "adapter_config.json").exists():
            print(f"skip missing {adapter}", flush=True)
            continue
        print(f"eval {name}", flush=True)
        recs = eval_adapter(cfg, adapter, tasks, outp, args.resume)
        recs_by[name] = recs
        m = trajectory_metrics(recs)
        results[name] = {"success_rate": m["success_rate"], "metrics": slim_metrics(m)}
        print(f"  {name} success={m['success_rate']:.3f}", flush=True)

    boot_seed = int(dcfg.get("bootstrap_seed", 0))
    n_boot = int(dcfg.get("n_boot", 10000))
    boots = {}
    if "uce_opd" in recs_by:
        boots["vs_h0"] = paired_task_bootstrap(recs0, recs_by["uce_opd"], n_boot=n_boot, seed=boot_seed)
        if "vanilla_sft" in recs_by:
            boots["vs_vanilla"] = paired_task_bootstrap(
                recs_by["vanilla_sft"], recs_by["uce_opd"], n_boot=n_boot, seed=boot_seed
            )
        if "no_pi_opd" in recs_by:
            boots["vs_no_pi"] = paired_task_bootstrap(
                recs_by["no_pi_opd"], recs_by["uce_opd"], n_boot=n_boot, seed=boot_seed
            )
        if "uce_sft" in recs_by:
            boots["vs_uce_sft"] = paired_task_bootstrap(
                recs_by["uce_sft"], recs_by["uce_opd"], n_boot=n_boot, seed=boot_seed
            )
    dump_json(reports / "paired_bootstrap.json", boots)

    sopd = results.get("uce_opd", {}).get("success_rate")
    svan = results.get("vanilla_sft", {}).get("success_rate")
    snopi = results.get("no_pi_opd", {}).get("success_rate")
    ssft = results.get("uce_sft", {}).get("success_rate")
    s0 = m0["success_rate"]
    su = mu["success_rate"]
    min_pp = float(dcfg.get("gate_min_delta_pp", 2.0))
    d_h0 = None if sopd is None else (sopd - s0) * 100
    r_int = None
    if sopd is not None and abs(su - s0) > 1e-12:
        r_int = (sopd - s0) / (su - s0)
    gate = {
        "delta_h0_pp": d_h0,
        "delta_vanilla_pp": None if sopd is None or svan is None else (sopd - svan) * 100,
        "delta_no_pi_pp": None if sopd is None or snopi is None else (sopd - snopi) * 100,
        "delta_sft_pp": None if sopd is None or ssft is None else (sopd - ssft) * 100,
        "R_int": r_int,
        "bootstrap": boots,
        "min_delta_pp": min_pp,
        "gate_pass": False,
    }
    if sopd is not None and svan is not None and snopi is not None and boots.get("vs_h0"):
        gate["gate_pass"] = bool(
            d_h0 is not None
            and d_h0 >= min_pp
            and boots["vs_h0"]["ci_lo"] > 0
            and sopd >= svan
            and sopd > snopi
        )
    payload = {"results": results, "gate": gate}
    dump_json(reports / "summary.json", payload)
    print(json.dumps(gate, indent=2))
    print("Gate D0", "PASS" if gate["gate_pass"] else "FAIL")


if __name__ == "__main__":
    main()
