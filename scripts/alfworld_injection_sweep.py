#!/usr/bin/env python3
"""P3: ALFWorld confirm-pool steering alpha sweep (archive)."""

from __future__ import annotations

import argparse
import json
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
    load_task_pools,
    load_yaml_cfg,
    make_env_factory,
    paired_task_bootstrap,
    trajectory_metrics,
)
from rollout.multistep import collect_episodes  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402


def load_directions(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    if key not in data.files:
        raise KeyError(f"{key} not in {path}")
    return np.asarray(data[key], dtype=np.float32)


def load_or_run(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    alpha: float,
    inject_style: str,
    vector: np.ndarray,
    layer: int,
    out_path: Path,
    resume: bool,
    dataset_split: str | None = None,
) -> list[dict]:
    from alfworld_common import load_jsonl

    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume skip {out_path.name} n={len(recs)}", flush=True)
            return recs
    return run_condition(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        alpha=alpha,
        inject_style=inject_style,
        vector=vector,
        layer=layer,
        out_path=out_path,
        resume=resume,
        dataset_split=dataset_split,
    )


def run_condition(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    alpha: float,
    inject_style: str,
    vector: np.ndarray,
    layer: int,
    out_path: Path,
    resume: bool,
    dataset_split: str | None = None,
) -> list[dict]:
    agent.clear_steering()
    if alpha != 0.0:
        agent.set_steering(
            layer=layer,
            vector=vector,
            alpha=alpha,
            inject_style=inject_style,  # type: ignore[arg-type]
        )
    n_done = [0]

    def _progress(rec: dict) -> None:
        n_done[0] += 1
        if n_done[0] % 20 == 0:
            print(
                f"  rec {n_done[0]} success={rec.get('episode_success')} "
                f"steps={rec.get('trajectory_length')}",
                flush=True,
            )

    return collect_episodes(
        tasks,
        agent,
        make_env_factory(cfg),
        n_rollouts=1,
        dataset_split=dataset_split or str(cfg.get("env", {}).get("eval_split", "audit_confirm")),
        base_seed=int(cfg["rollout"]["seed"]) + int(abs(alpha) * 100),
        out_path=out_path,
        resume=resume,
        max_steps=int(cfg["rollout"]["max_steps"]),
        batch_size=int(cfg["rollout"]["batch_size"]),
        n_env_workers=int(cfg["rollout"].get("env_workers", 6)),
        progress_cb=_progress,
    )


def pick_best_alpha(grid_results: dict[str, dict]) -> tuple[str, float, dict]:
    best_key = "0.0"
    best_gain = -1e9
    baseline = grid_results.get("0.0", {})
    b_succ = baseline.get("metrics", {}).get("success_rate", 0.0)
    for key, payload in grid_results.items():
        if key == "0.0":
            continue
        m = payload.get("metrics", {})
        gain = m.get("success_rate", 0.0) - b_succ
        if gain > best_gain:
            best_gain = gain
            best_key = key
    return best_key, best_gain, grid_results[best_key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_confirm.yaml"),
    )
    ap.add_argument("--phase", choices=["A", "B", "all"], default="all")
    ap.add_argument("--inject-style", choices=["decision_point", "generation_wide", "both"], default="both")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--adapter-path", default=None)
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    cfg.setdefault("rollout", {})["backend"] = "hf"
    if args.adapter_path:
        cfg.setdefault("model", {})["adapter_path"] = args.adapter_path

    steer = cfg["steering"]
    vector = load_directions(Path(steer["directions"]), steer["direction_key"])
    layer = int(steer["layer"])
    alphas = [float(a) for a in steer["alpha_grid"]]
    styles = (
        ["decision_point", "generation_wide"]
        if args.inject_style == "both"
        else [args.inject_style]
    )

    payload = load_task_pools(limit=int(cfg["env"]["limit"]))
    confirm_all = payload["pools"]["audit_confirm"]
    phase_a_n = int(steer.get("phase_a_tasks", 80))
    tasks_a = confirm_all[:phase_a_n]
    tasks_b = confirm_all if steer.get("phase_b_full", True) else confirm_all[:phase_a_n]

    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    agent = build_agent(cfg)

    all_results: dict[str, Any] = {"phase_a": {}, "phase_b": {}, "analysis": {}}
    best_style = "decision_point"
    best_alpha = 0.0
    best_phase_a_gain = -1e9

    for style in styles:
        style_dir = reports / f"phase_a_{style}"
        style_dir.mkdir(parents=True, exist_ok=True)
        grid: dict[str, dict] = {}
        if args.phase in ("A", "all"):
            for alpha in alphas:
                tag = f"alpha_{alpha:+.1f}".replace("+", "p").replace("-", "m")
                out = style_dir / f"{tag}.jsonl"
                if out.exists() and args.resume and alpha != 0.0:
                    pass
                print(f"PhaseA style={style} alpha={alpha} tasks={len(tasks_a)}", flush=True)
                t0 = time.time()
                recs = load_or_run(
                    agent=agent,
                    tasks=tasks_a,
                    cfg=cfg,
                    alpha=alpha,
                    inject_style=style,
                    vector=vector,
                    layer=layer,
                    out_path=out,
                    resume=args.resume,
                )
                metrics = trajectory_metrics(recs)
                grid[str(alpha)] = {
                    "metrics": metrics,
                    "wall_s": time.time() - t0,
                    "n_tasks": len(tasks_a),
                    "out": str(out),
                }
                print(f"  success={metrics['success_rate']:.3f} adm={metrics['admissible_action_rate']:.3f}", flush=True)

            best_key, best_gain, best_payload = pick_best_alpha(grid)
            all_results["phase_a"][style] = {"grid": grid, "best_alpha": float(best_key), "best_gain_pp": best_gain * 100}
            if best_gain > best_phase_a_gain:
                best_phase_a_gain = best_gain
                best_alpha = float(best_key)
                best_style = style

    if args.phase in ("B", "all"):
        phase_b_dir = reports / f"phase_b_{best_style}"
        phase_b_dir.mkdir(parents=True, exist_ok=True)
        base_out = phase_b_dir / "alpha_0.0.jsonl"
        best_out = phase_b_dir / f"alpha_{best_alpha:+.1f}.jsonl".replace("+", "p").replace("-", "m")
        print(f"PhaseB style={best_style} alpha=0 tasks={len(tasks_b)}", flush=True)
        recs0 = load_or_run(
            agent=agent,
            tasks=tasks_b,
            cfg=cfg,
            alpha=0.0,
            inject_style=best_style,
            vector=vector,
            layer=layer,
            out_path=base_out,
            resume=args.resume,
        )
        print(f"PhaseB style={best_style} alpha={best_alpha} tasks={len(tasks_b)}", flush=True)
        recs1 = load_or_run(
            agent=agent,
            tasks=tasks_b,
            cfg=cfg,
            alpha=best_alpha,
            inject_style=best_style,
            vector=vector,
            layer=layer,
            out_path=best_out,
            resume=args.resume,
        )
        m0 = trajectory_metrics(recs0)
        m1 = trajectory_metrics(recs1)
        boot = paired_task_bootstrap(recs0, recs1)
        delta_ep = m1["success_rate"] - m0["success_rate"]
        delta_adm = m1["admissible_action_rate"] - m0["admissible_action_rate"]
        all_results["phase_b"] = {
            "inject_style": best_style,
            "best_alpha": best_alpha,
            "baseline": m0,
            "steered": m1,
            "delta_success_pp": delta_ep * 100,
            "delta_admissible_pp": delta_adm * 100,
            "amplification_ratio": amplification_ratio(delta_ep, delta_adm),
            "bootstrap": boot,
            "n_tasks": len(tasks_b),
        }

    agent.close()
    all_results["analysis"] = {
        "selected_style": best_style,
        "selected_alpha": best_alpha,
        "phase_a_best_gain_pp": best_phase_a_gain * 100,
        "h1_episode_gain_ge_5pp": best_phase_a_gain * 100 >= 5.0,
    }
    dump_json(reports / "results_confirm.json", all_results)
    print("P3_DONE", json.dumps(all_results["analysis"], indent=2), flush=True)


if __name__ == "__main__":
    main()
