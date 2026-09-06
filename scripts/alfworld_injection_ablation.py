#!/usr/bin/env python3
"""R0/R1: ALFWorld valid_unseen injection-site ablation (archive)."""

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
    load_eval_tasks,
    load_jsonl,
    load_yaml_cfg,
    make_env_factory,
    paired_task_bootstrap,
    trajectory_metrics,
)
from alfworld_injection_sweep import load_directions, load_or_run  # noqa: E402
from rollout.multistep import episode_seed, _decision_point  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402

R0_STYLES_DEFAULT = {
    "R0a": "decision_point",
    "R0b": "prefill_decode",
    "R0c": "decode_only",
    "R0d": "generation_wide",
    "R0e": "action_boundary",
}


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:+.1f}".replace("+", "p").replace("-", "m")


def pick_best_vs_baseline(
    grid: dict[str, dict],
    *,
    baseline_key: str = "0.0",
) -> tuple[str, float, dict]:
    baseline = grid.get(baseline_key, {}).get("metrics", {})
    b_succ = float(baseline.get("success_rate", 0.0))
    best_key = baseline_key
    best_gain = -1e9
    best_payload = grid.get(baseline_key, {})
    for key, payload in grid.items():
        if key == baseline_key:
            continue
        m = payload.get("metrics", {})
        gain = float(m.get("success_rate", 0.0)) - b_succ
        if gain > best_gain:
            best_gain = gain
            best_key = key
            best_payload = payload
    return best_key, best_gain, best_payload


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
    amp = amplification_ratio(delta_ep, delta_adm)
    passed = (
        delta_ep * 100 >= min_delta_pp
        and boot["ci_lo"] > 0
        and amp is not None
        and amp > 1.0
    )
    return {
        "delta_success_pp": delta_ep * 100,
        "delta_admissible_pp": delta_adm * 100,
        "amplification_ratio": amp,
        "bootstrap": boot,
        "gate_pass": passed,
    }


def run_r0(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    vector: np.ndarray,
    layer: int,
    styles: dict[str, str],
    alphas: list[float],
    reports: Path,
    resume: bool,
) -> dict[str, Any]:
    r0_dir = reports / "r0"
    r0_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"conditions": {}, "meta": {"n_tasks": len(tasks)}}

    for rid, inject_style in styles.items():
        style_dir = r0_dir / rid
        style_dir.mkdir(parents=True, exist_ok=True)
        grid: dict[str, dict] = {}
        recs_by_alpha: dict[str, list[dict]] = {}
        for alpha in alphas:
            out = style_dir / f"{alpha_tag(alpha)}.jsonl"
            print(f"R0 {rid} style={inject_style} alpha={alpha} n={len(tasks)}", flush=True)
            t0 = time.time()
            recs = load_or_run(
                agent=agent,
                tasks=tasks,
                cfg=cfg,
                alpha=alpha,
                inject_style=inject_style,
                vector=vector,
                layer=layer,
                out_path=out,
                resume=resume,
            )
            metrics = trajectory_metrics(recs)
            grid[str(alpha)] = {
                "metrics": metrics,
                "wall_s": time.time() - t0,
                "out": str(out),
            }
            recs_by_alpha[str(alpha)] = recs
            print(
                f"  success={metrics['success_rate']:.3f} "
                f"adm={metrics['admissible_action_rate']:.3f}",
                flush=True,
            )
        best_alpha, best_gain, _ = pick_best_vs_baseline(grid)
        results["conditions"][rid] = {
            "inject_style": inject_style,
            "grid": grid,
            "best_alpha": float(best_alpha),
            "best_gain_pp_vs_alpha0": best_gain * 100,
            "recs_best_alpha": recs_by_alpha.get(best_alpha, []),
            "recs_alpha0": recs_by_alpha.get("0.0", []),
        }
    return results


def select_r0_winner(r0: dict[str, Any], *, baseline_rid: str = "R0a") -> tuple[str, str, float]:
    baseline = r0["conditions"][baseline_rid]
    b0 = baseline["grid"]["0.0"]["metrics"]
    best_rid = baseline_rid
    best_alpha = float(baseline["best_alpha"])
    best_delta = -1e9
    for rid, payload in r0["conditions"].items():
        if rid == baseline_rid:
            continue
        alpha = str(payload["best_alpha"])
        m = payload["grid"][alpha]["metrics"]
        delta = float(m["success_rate"]) - float(b0["success_rate"])
        if delta > best_delta:
            best_delta = delta
            best_rid = rid
            best_alpha = float(payload["best_alpha"])
    return best_rid, str(best_alpha), best_delta


def run_cast_lite(
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
) -> list[dict]:
    from rollout.resume import append_jsonl
    from rollout.trajectory import empty_trajectory

    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume skip cast_lite {out_path.name} n={len(recs)}", flush=True)
            return recs

    agent.clear_steering()
    if alpha != 0.0:
        agent.set_steering(
            layer=layer,
            vector=vector,
            alpha=alpha,
            inject_style=inject_style,  # type: ignore[arg-type]
        )

    rollout = cfg["rollout"]
    max_steps = int(rollout["max_steps"])
    base_seed = int(rollout["seed"])
    split = str(cfg.get("env", {}).get("eval_split", "valid_unseen"))
    env_factory = make_env_factory(cfg)
    records: list[dict] = []

    for i, task in enumerate(tasks, 1):
        env = env_factory()
        tid = str(task["task_id"])
        traj_id = f"{tid}:cast:0"
        seed = episode_seed(base_seed, tid, 0)
        obs = env.reset(task)
        last_admissible = True
        actions: list[str] = []
        decisions: list[dict] = []
        token_count = 0
        termination = "max_steps"
        episode_success = False
        first_prompt = ""

        for step_i in range(max_steps):
            state = env.current_state()

            def _gate(la: bool = last_admissible) -> bool:
                return not la

            agent.set_steer_gate(_gate)
            gen = agent.generate(obs, seed=seed + step_i * 17)
            agent.set_steer_gate(None)
            if step_i == 0:
                first_prompt = gen.prompt_text or obs
            action_text = gen.text
            token_count += int(gen.completion_token_count)
            actions.append(action_text)
            verdict = env.verify_decision(state, action_text)
            out = env.step(action_text)
            dp = _decision_point(
                decision_id=f"{traj_id}:{step_i}",
                step_index=step_i,
                prefix_text=obs,
                prompt_text=gen.prompt_text or obs,
                state=state,
                verdict=verdict,
                action_text=action_text,
            )
            decisions.append(dp)
            adm = bool(verdict.get("parse_ok")) and verdict.get("failure_reason") != "not_admissible"
            last_admissible = adm
            if out.terminal:
                termination = str(out.failure_reason or "success")
                episode_success = bool(out.success)
                obs = out.observation
                break
            obs = out.observation

        rec = empty_trajectory(
            task_id=tid,
            trajectory_id=traj_id,
            dataset_split=split,
            task_type=task.get("task_type"),
            goal=task.get("goal"),
        )
        rec.update(
            {
                "episode_success": episode_success,
                "termination_reason": termination,
                "trajectory_length": len(actions),
                "actions": actions,
                "decision_points": decisions,
                "token_count": token_count,
                "prompt_text": first_prompt,
            }
        )
        records.append(rec)
        if i % 20 == 0:
            print(f"  cast_lite {i}/{len(tasks)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")
    for rec in records:
        append_jsonl(out_path, rec)
    return records


def run_r1(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    layer: int,
    winner_rid: str,
    winner_style: str,
    winner_alpha: float,
    baseline_vector: np.ndarray,
    reports: Path,
    resume: bool,
) -> dict[str, Any]:
    steer = cfg["steering"]
    r1_dir = reports / "r1"
    r1_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"winner": {"rid": winner_rid, "style": winner_style, "alpha": winner_alpha}}

    # R1a: step-admissible refit vector
    r1a_path = Path(steer["directions_r1a"])
    if r1a_path.exists():
        v_r1a = load_directions(r1a_path, steer["direction_key"])
        out_path = r1_dir / "R1a_step_admissible.jsonl"
        print(f"R1a style={winner_style} alpha={winner_alpha}", flush=True)
        recs = load_or_run(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            alpha=winner_alpha,
            inject_style=winner_style,
            vector=v_r1a,
            layer=layer,
            out_path=out_path,
            resume=resume,
        )
        out["R1a"] = {"metrics": trajectory_metrics(recs), "out": str(out_path)}
    else:
        out["R1a"] = {"skipped": True, "reason": f"missing {r1a_path}"}

    # R1b: CAST-lite on winner style
    cast_alpha = float(steer.get("cast_lite_alpha", winner_alpha))
    cast_out = r1_dir / f"R1b_cast_lite_{alpha_tag(cast_alpha)}.jsonl"
    print(f"R1b CAST-lite style={winner_style} alpha={cast_alpha}", flush=True)
    recs_cast = run_cast_lite(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        alpha=cast_alpha,
        inject_style=winner_style,
        vector=baseline_vector,
        layer=layer,
        out_path=cast_out,
        resume=resume,
    )
    out["R1b"] = {"metrics": trajectory_metrics(recs_cast), "out": str(cast_out)}

    # Always-on winner for comparison
    win_out = r1_dir / f"winner_always_on_{alpha_tag(winner_alpha)}.jsonl"
    recs_win = load_or_run(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        alpha=winner_alpha,
        inject_style=winner_style,
        vector=baseline_vector,
        layer=layer,
        out_path=win_out,
        resume=resume,
    )
    out["winner_always_on"] = {"metrics": trajectory_metrics(recs_win), "out": str(win_out)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_unseen_ablation.yaml"),
    )
    ap.add_argument("--phase", choices=["r0", "r1", "all"], default="all")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--fit-r1a", action="store_true", help="Run fit_alfworld_directions step_admissible first")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    cfg.setdefault("rollout", {})["backend"] = "hf"

    steer = cfg["steering"]
    vector = load_directions(Path(steer["directions"]), steer["direction_key"])
    layer = int(steer["layer"])
    alphas = [float(a) for a in steer["alpha_grid"]]
    styles = dict(steer.get("r0_styles") or R0_STYLES_DEFAULT)

    tasks, task_meta = load_eval_tasks(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    if args.fit_r1a and args.phase in ("r1", "all"):
        import subprocess

        fit_script = SCRIPT_DIR / "fit_alfworld_directions.py"
        # Fit R1a after R0 so long extraction does not block injection ablation.
        fit_after_r0 = True
    else:
        fit_after_r0 = False
        fit_script = None

    agent = build_agent(cfg)
    payload: dict[str, Any] = {"task_meta": task_meta, "r0": {}, "r1": {}, "analysis": {}}

    if args.phase in ("r0", "all"):
        payload["r0"] = run_r0(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            vector=vector,
            layer=layer,
            styles=styles,
            alphas=alphas,
            reports=reports,
            resume=args.resume,
        )
        winner_rid, winner_alpha_s, winner_delta = select_r0_winner(payload["r0"])
        winner_style = styles[winner_rid]
        winner_alpha = float(winner_alpha_s)
        baseline = payload["r0"]["conditions"]["R0a"]
        b_recs = baseline["recs_alpha0"]
        w_payload = payload["r0"]["conditions"][winner_rid]
        w_recs = w_payload["recs_best_alpha"]
        w_alpha = str(w_payload["best_alpha"])
        w_metrics = w_payload["grid"][w_alpha]["metrics"]
        b_metrics = baseline["grid"]["0.0"]["metrics"]
        gate = evaluate_gate(
            baseline_recs=b_recs,
            best_recs=w_recs,
            baseline_metrics=b_metrics,
            best_metrics=w_metrics,
            min_delta_pp=float(steer.get("gate_min_delta_pp", 3.0)),
        )
        payload["analysis"] = {
            "winner_rid": winner_rid,
            "winner_style": winner_style,
            "winner_alpha": winner_alpha,
            "winner_delta_pp_vs_r0a_alpha0": winner_delta * 100,
            "gate": gate,
        }

    if fit_after_r0 and fit_script is not None:
        import subprocess

        fit_proc = subprocess.run(
            [
                str(cfg.get("python", sys.executable)),
                str(fit_script),
                "--config",
                args.config,
                "--pairing",
                "step_admissible",
                "--site",
                "action_boundary",
                "--output-key-suffix",
                "r1a_step_admissible",
                "--skip-smoke",
            ],
            check=False,
        )
        if fit_proc.returncode != 0:
            print("WARN: fit_r1a failed; R1a will be skipped", flush=True)

    if args.phase in ("r1", "all"):
        if not payload.get("analysis"):
            payload["analysis"] = {
                "winner_rid": "R0b",
                "winner_style": styles.get("R0b", "prefill_decode"),
                "winner_alpha": -1.5,
            }
        ana = payload["analysis"]
        payload["r1"] = run_r1(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            layer=layer,
            winner_rid=str(ana["winner_rid"]),
            winner_style=str(ana["winner_style"]),
            winner_alpha=float(ana["winner_alpha"]),
            baseline_vector=vector,
            reports=reports,
            resume=args.resume,
        )

    agent.close()
    dump_json(reports / "results.json", payload)
    print("ABLATION_DONE", json.dumps(payload.get("analysis", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
