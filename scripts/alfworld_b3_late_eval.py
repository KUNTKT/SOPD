#!/usr/bin/env python3
"""B3: late-divergent v vs episode mean-pool v on valid_unseen (archive)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import (  # noqa: E402
    amplification_ratio,
    dump_json,
    ensure_alfworld_env,
    load_eval_tasks,
    load_jsonl,
    load_yaml_cfg,
    paired_task_bootstrap,
    trajectory_metrics,
)
from alfworld_injection_sweep import load_directions, load_or_run  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:+.1f}".replace("+", "p").replace("-", "m")


def maybe_reuse_r0(src: Path, dst: Path, expected: int, resume: bool) -> list[dict] | None:
    if not resume or not src.exists():
        return None
    recs = load_jsonl(src)
    if len(recs) < expected:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or len(load_jsonl(dst)) < expected:
        shutil.copy2(src, dst)
        print(f"  reuse R0a {src.name} -> {dst.name}", flush=True)
    return load_jsonl(dst)


def evaluate_gate(
    *,
    baseline_recs: list[dict],
    best_recs: list[dict],
    baseline_metrics: dict,
    best_metrics: dict,
    min_delta_pp: float,
) -> dict[str, Any]:
    boot = paired_task_bootstrap(baseline_recs, best_recs)
    delta_ep = float(best_metrics["success_rate"]) - float(baseline_metrics["success_rate"])
    delta_adm = float(best_metrics["admissible_action_rate"]) - float(
        baseline_metrics["admissible_action_rate"]
    )
    return {
        "delta_success_pp": delta_ep * 100,
        "delta_admissible_pp": delta_adm * 100,
        "amplification_ratio": amplification_ratio(delta_ep, delta_adm),
        "bootstrap": boot,
        "gate_pass": delta_ep * 100 >= min_delta_pp and boot["ci_lo"] > 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_b3_late.yaml"),
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    cfg.setdefault("rollout", {})["backend"] = "hf"

    steer = cfg["steering"]
    layer = int(steer["layer"])
    style = str(steer.get("inject_style", "decision_point"))
    key = str(steer["direction_key"])
    alphas = [float(a) for a in steer["alpha_grid"]]
    v_episode = load_directions(Path(steer["directions_episode"]), key)
    v_late = load_directions(Path(steer["directions_late"]), key)

    tasks, task_meta = load_eval_tasks(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    r0_reuse = Path(cfg["paths"].get("r0_reuse_dir") or "")

    agent = build_agent(cfg)
    vectors = {"v_episode": v_episode, "v_late": v_late}
    conditions: dict[str, Any] = {}
    recs_cache: dict[str, list[dict]] = {}

    # Shared α=0 baseline (no vector dependence)
    alpha0_out = reports / "shared" / f"{alpha_tag(0.0)}.jsonl"
    print(f"B3 shared alpha=0 n={len(tasks)}", flush=True)
    recs0 = None
    if r0_reuse.is_dir():
        recs0 = maybe_reuse_r0(
            r0_reuse / f"{alpha_tag(0.0)}.jsonl", alpha0_out, len(tasks), args.resume
        )
    if recs0 is None:
        t0 = time.time()
        recs0 = load_or_run(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            alpha=0.0,
            inject_style=style,
            vector=v_episode,
            layer=layer,
            out_path=alpha0_out,
            resume=args.resume,
        )
        wall0 = time.time() - t0
    else:
        wall0 = 0.0
    m0 = trajectory_metrics(recs0)
    conditions["alpha0"] = {"metrics": m0, "wall_s": wall0, "out": str(alpha0_out)}
    recs_cache["alpha0"] = recs0
    print(f"  success={m0['success_rate']:.3f} adm={m0['admissible_action_rate']:.3f}", flush=True)

    for vname, vector in vectors.items():
        vdir = reports / vname
        vdir.mkdir(parents=True, exist_ok=True)
        grid: dict[str, Any] = {"0.0": {"metrics": m0, "out": str(alpha0_out), "shared": True}}
        for alpha in alphas:
            if abs(alpha) < 1e-12:
                continue
            out = vdir / f"{alpha_tag(alpha)}.jsonl"
            print(f"B3 {vname} style={style} alpha={alpha} n={len(tasks)}", flush=True)
            recs = None
            if vname == "v_episode" and r0_reuse.is_dir():
                recs = maybe_reuse_r0(
                    r0_reuse / f"{alpha_tag(alpha)}.jsonl", out, len(tasks), args.resume
                )
            if recs is None:
                t0 = time.time()
                recs = load_or_run(
                    agent=agent,
                    tasks=tasks,
                    cfg=cfg,
                    alpha=alpha,
                    inject_style=style,
                    vector=vector,
                    layer=layer,
                    out_path=out,
                    resume=args.resume,
                )
                wall = time.time() - t0
            else:
                wall = 0.0
            metrics = trajectory_metrics(recs)
            grid[str(alpha)] = {"metrics": metrics, "wall_s": wall, "out": str(out)}
            recs_cache[f"{vname}:{alpha}"] = recs
            print(
                f"  success={metrics['success_rate']:.3f} "
                f"adm={metrics['admissible_action_rate']:.3f}",
                flush=True,
            )

        # Best non-zero alpha vs alpha0
        best_alpha = None
        best_gain = -1e9
        for a in alphas:
            if abs(a) < 1e-12:
                continue
            gain = float(grid[str(a)]["metrics"]["success_rate"]) - float(m0["success_rate"])
            if gain > best_gain:
                best_gain = gain
                best_alpha = a
        conditions[vname] = {
            "grid": grid,
            "best_alpha": best_alpha,
            "best_gain_pp_vs_alpha0": best_gain * 100,
        }

    agent.close()

    # Gate: v_late best vs v_episode @ -1.5
    ep_alpha = -1.5 if "-1.5" in conditions["v_episode"]["grid"] else conditions["v_episode"]["best_alpha"]
    late_best = conditions["v_late"]["best_alpha"]
    ep_recs = recs_cache.get(f"v_episode:{ep_alpha}", [])
    late_recs = recs_cache.get(f"v_late:{late_best}", [])
    ep_m = conditions["v_episode"]["grid"][str(ep_alpha)]["metrics"]
    late_m = conditions["v_late"]["grid"][str(late_best)]["metrics"]
    gate = evaluate_gate(
        baseline_recs=ep_recs,
        best_recs=late_recs,
        baseline_metrics=ep_m,
        best_metrics=late_m,
        min_delta_pp=float(steer.get("gate_min_delta_pp", 3.0)),
    )
    vs_alpha0 = evaluate_gate(
        baseline_recs=recs0,
        best_recs=late_recs,
        baseline_metrics=m0,
        best_metrics=late_m,
        min_delta_pp=float(steer.get("gate_min_delta_pp", 3.0)),
    )

    payload = {
        "task_meta": task_meta,
        "conditions": conditions,
        "analysis": {
            "inject_style": style,
            "episode_ref_alpha": ep_alpha,
            "late_best_alpha": late_best,
            "gate_vs_episode_m15": gate,
            "late_vs_alpha0": vs_alpha0,
        },
    }
    dump_json(reports / "results.json", payload)
    print("B3_DONE", json.dumps(payload["analysis"], indent=2), flush=True)


if __name__ == "__main__":
    main()
