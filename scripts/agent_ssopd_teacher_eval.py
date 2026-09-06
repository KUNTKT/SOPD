#!/usr/bin/env python3
"""A1: NPM-lite retrieval Super-Self Teacher eval on valid_unseen (archive)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_eval_tasks,
    load_jsonl,
    load_yaml_cfg,
    make_env_factory,
    paired_task_bootstrap,
    trajectory_metrics,
)
from alfworld_injection_sweep import load_directions  # noqa: E402
from rollout.multistep import _decision_point, episode_seed  # noqa: E402
from rollout.resume import append_jsonl  # noqa: E402
from rollout.trajectory import empty_trajectory  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402


class NpmMemory:
    def __init__(self, path: Path):
        data = np.load(path)
        self.queries = data["queries"].astype(np.float32)
        self.deltas = data["deltas"].astype(np.float32)
        # Normalize queries for cosine retrieval.
        qn = np.linalg.norm(self.queries, axis=1, keepdims=True).clip(min=1e-8)
        self.queries = self.queries / qn
        self.probe_coef = data["probe_coef"].astype(np.float32)
        self.probe_intercept = float(data["probe_intercept"][0])
        self.layer = int(data["layer"][0]) if "layer" in data.files else 14

    def synthesize(self, query: np.ndarray, top_k: int) -> np.ndarray:
        q = query.astype(np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self.queries @ q
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        w = sims[idx]
        w = np.maximum(w, 0.0)
        if float(w.sum()) < 1e-8:
            w = np.ones_like(w)
        w = w / w.sum()
        v = (self.deltas[idx] * w[:, None]).sum(axis=0)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return np.zeros_like(v)
        return (v / n).astype(np.float32)

    def error_prob(self, hidden: np.ndarray) -> float:
        x = hidden.astype(np.float32).reshape(-1)
        z = float(self.probe_coef @ x + self.probe_intercept)
        # Stable sigmoid
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:+.1f}".replace("+", "p").replace("-", "m")


def apply_workflow(obs: str, workflow_text: str | None) -> str:
    if not workflow_text:
        return obs
    return f"{workflow_text}\n\n{obs}"


def run_npm_episodes(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    memory: NpmMemory | None,
    alpha: float,
    top_k: int,
    use_gate: bool,
    gate_threshold: float,
    static_vector: np.ndarray | None,
    out_path: Path,
    resume: bool,
    dataset_split: str,
    workflow_fn: Any | None = None,
) -> list[dict]:
    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume skip {out_path.name} n={len(recs)}", flush=True)
            return recs

    layer = int(cfg["steering"]["layer"])
    style = str(cfg["steering"].get("inject_style", "decision_point"))
    max_steps = int(cfg["rollout"]["max_steps"])
    base_seed = int(cfg["rollout"]["seed"])
    env_factory = make_env_factory(cfg)

    agent.clear_steering()
    if abs(alpha) > 1e-12:
        # Placeholder vector; updated every step for NPM / static.
        init_v = static_vector if static_vector is not None else np.zeros(2048, dtype=np.float32)
        if memory is not None and init_v is not None and float(np.linalg.norm(init_v)) < 1e-8:
            init_v = memory.deltas[0]
        agent.set_steering(
            layer=layer,
            vector=init_v if init_v is not None else memory.deltas[0],  # type: ignore[arg-type]
            alpha=alpha,
            inject_style=style,  # type: ignore[arg-type]
        )

    records: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    for i, task in enumerate(tasks, 1):
        env = env_factory()
        tid = str(task["task_id"])
        traj_id = f"{tid}:npm:0"
        seed = episode_seed(base_seed, tid, 0)
        raw_obs = env.reset(task)
        wf = workflow_fn(task) if workflow_fn is not None else None
        obs = apply_workflow(raw_obs, wf)
        actions: list[str] = []
        decisions: list[dict] = []
        token_count = 0
        termination = "max_steps"
        episode_success = False
        first_prompt = ""
        n_steered = 0
        episode_v: np.ndarray | None = None

        for step_i in range(max_steps):
            state = env.current_state()
            do_steer = abs(alpha) > 1e-12

            if abs(alpha) > 1e-12 and memory is not None and (
                episode_v is None or use_gate
            ):
                prompt = agent._prompt_text(obs)
                h = agent.forward_prompt_last_hidden(prompt, layer).numpy()
                if episode_v is None:
                    episode_v = memory.synthesize(h, top_k)
                if use_gate:
                    do_steer = memory.error_prob(h) >= gate_threshold

            if abs(alpha) > 1e-12:
                if memory is not None:
                    v = episode_v
                else:
                    v = static_vector
                assert v is not None
                agent.update_steer_vector(torch.tensor(v, dtype=torch.float32), alpha=alpha)
                agent.set_steer_gate(lambda ds=do_steer: bool(ds))
                if do_steer:
                    n_steered += 1
            else:
                agent.set_steer_gate(None)

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
            dp["steered"] = bool(do_steer and abs(alpha) > 1e-12)
            decisions.append(dp)
            if out.terminal:
                termination = str(out.failure_reason or "success")
                episode_success = bool(out.success)
                break
            obs = apply_workflow(out.observation, wf)

        rec = empty_trajectory(
            task_id=tid,
            trajectory_id=traj_id,
            dataset_split=dataset_split,
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
                "n_steered_steps": n_steered,
            }
        )
        records.append(rec)
        append_jsonl(out_path, rec)
        if i % 10 == 0:
            print(
                f"  rec {i}/{len(tasks)} success={episode_success} "
                f"steered_steps={n_steered}",
                flush=True,
            )
    return records


def maybe_copy(src: Path, dst: Path, expected: int) -> list[dict] | None:
    if not src.exists():
        return None
    recs = load_jsonl(src)
    if len(recs) < expected:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or len(load_jsonl(dst)) < expected:
        shutil.copy2(src, dst)
        print(f"  reuse {src} -> {dst.name}", flush=True)
    return load_jsonl(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_alfworld.yaml"),
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    steer = cfg["steering"]
    memory = NpmMemory(Path(cfg["paths"]["memory_dir"]) / "npm_memory.npz")
    static_v = load_directions(Path(steer["directions_static"]), steer["direction_key"])
    tasks, task_meta = load_eval_tasks(cfg)
    reports = Path(cfg["paths"]["reports_dir"]) / "a1_teacher"
    reports.mkdir(parents=True, exist_ok=True)
    split = str(cfg["env"].get("eval_split", "valid_unseen"))
    alphas = [float(a) for a in steer["alpha_grid"]]
    top_k = int(steer.get("top_k", 8))
    thr = float(steer.get("gate_threshold", 0.45))
    min_pp = float(steer.get("gate_min_delta_pp", 5.0))

    agent = build_agent(cfg)
    conditions: dict[str, Any] = {}
    recs_cache: dict[str, list[dict]] = {}

    # Baseline α=0
    out0 = reports / "base_alpha0.jsonl"
    print("A1 base alpha=0", flush=True)
    recs0 = maybe_copy(Path(cfg["paths"].get("r0_alpha0", "")), out0, len(tasks))
    if recs0 is None:
        t0 = time.time()
        recs0 = run_npm_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            memory=None,
            alpha=0.0,
            top_k=top_k,
            use_gate=False,
            gate_threshold=thr,
            static_vector=None,
            out_path=out0,
            resume=args.resume,
            dataset_split=split,
        )
        wall0 = time.time() - t0
    else:
        wall0 = 0.0
    m0 = trajectory_metrics(recs0)
    conditions["base"] = {"metrics": m0, "wall_s": wall0, "out": str(out0)}
    recs_cache["base"] = recs0
    print(f"  success={m0['success_rate']:.3f}", flush=True)

    # Static CAA @ -1.5 (reuse R0a if present)
    out_st = reports / "static_m1.5.jsonl"
    print("A1 static CAA alpha=-1.5", flush=True)
    recs_st = maybe_copy(Path(cfg["paths"].get("r0_static_m15", "")), out_st, len(tasks))
    if recs_st is None:
        t0 = time.time()
        recs_st = run_npm_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            memory=None,
            alpha=-1.5,
            top_k=top_k,
            use_gate=False,
            gate_threshold=thr,
            static_vector=static_v,
            out_path=out_st,
            resume=args.resume,
            dataset_split=split,
        )
        wall_st = time.time() - t0
    else:
        wall_st = 0.0
    m_st = trajectory_metrics(recs_st)
    conditions["static_m1.5"] = {"metrics": m_st, "wall_s": wall_st, "out": str(out_st)}
    recs_cache["static_m1.5"] = recs_st
    print(f"  success={m_st['success_rate']:.3f}", flush=True)

    # NPM-lite with / without gate across alpha grid
    best_key = None
    best_gain = -1e9
    for gated in (False, True):
        tag = "npm_gated" if gated else "npm"
        for alpha in alphas:
            key = f"{tag}:{alpha}"
            out = reports / f"{tag}_{alpha_tag(alpha)}.jsonl"
            print(f"A1 {key}", flush=True)
            t0 = time.time()
            recs = run_npm_episodes(
                agent=agent,
                tasks=tasks,
                cfg=cfg,
                memory=memory,
                alpha=alpha,
                top_k=top_k,
                use_gate=gated,
                gate_threshold=thr,
                static_vector=None,
                out_path=out,
                resume=args.resume,
                dataset_split=split,
            )
            metrics = trajectory_metrics(recs)
            gain = float(metrics["success_rate"]) - float(m0["success_rate"])
            conditions[key] = {
                "metrics": metrics,
                "wall_s": time.time() - t0,
                "out": str(out),
                "gain_pp": gain * 100,
            }
            recs_cache[key] = recs
            print(f"  success={metrics['success_rate']:.3f} gain_pp={gain*100:+.1f}", flush=True)
            if gain > best_gain:
                best_gain = gain
                best_key = key

    agent.close()
    assert best_key is not None
    boot = paired_task_bootstrap(recs0, recs_cache[best_key])
    gate = {
        "best_key": best_key,
        "delta_success_pp": best_gain * 100,
        "bootstrap": boot,
        "gate_pass": best_gain * 100 >= min_pp and boot["ci_lo"] > 0,
        "min_delta_pp": min_pp,
    }
    payload = {
        "task_meta": task_meta,
        "conditions": {
            k: {kk: vv for kk, vv in v.items() if kk != "metrics"}
            | {"metrics": {mk: mv for mk, mv in v["metrics"].items() if mk != "by_task_type"}}
            for k, v in conditions.items()
        },
        "analysis": gate,
    }
    # Store full metrics without by_task_type bloat already done
    dump_json(reports / "results.json", payload)
    dump_json(Path(cfg["paths"]["reports_dir"]) / "a1_results.json", payload)
    print("A1_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
