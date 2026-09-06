#!/usr/bin/env python3
"""N1/N2: NPM-faithful (task Jaccard + PCA + 3-layer prefill_decode + KL α)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import apply_workflow  # noqa: E402
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
from npm_faithful_lib import FaithfulNpmMemory, rec_goal  # noqa: E402
from rollout.multistep import _decision_point, episode_seed  # noqa: E402
from rollout.resume import append_jsonl  # noqa: E402
from rollout.trajectory import empty_trajectory  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402
from uce_eval import make_workflow_fn, slim_metrics  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def to_torch_vs(vs: dict[int, Any]) -> dict[int, torch.Tensor]:
    return {int(k): torch.tensor(v, dtype=torch.float32) for k, v in vs.items()}


def _complete(path: Path, expected: int) -> list[dict] | None:
    if not path.exists():
        return None
    recs = load_jsonl(path)
    if len(recs) >= expected:
        return recs
    return None


def run_plain_episodes(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    workflow_fn: Any | None,
    out_path: Path,
    resume: bool,
    dataset_split: str,
    tag: str,
) -> list[dict]:
    expected = len(tasks)
    if resume:
        done = _complete(out_path, expected)
        if done is not None:
            print(f"  resume skip {out_path.name} n={len(done)}", flush=True)
            return done

    max_steps = int(cfg["rollout"]["max_steps"])
    base_seed = int(cfg["rollout"]["seed"])
    env_factory = make_env_factory(cfg)
    agent.clear_steering()

    records: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    for i, task in enumerate(tasks, 1):
        env = env_factory()
        tid = str(task["task_id"])
        traj_id = f"{tid}:{tag}:0"
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

        for step_i in range(max_steps):
            state = env.current_state()
            gen = agent.generate(obs, seed=seed + step_i * 17)
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
            dp["steered"] = False
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
            }
        )
        records.append(rec)
        append_jsonl(out_path, rec)
        if i % 10 == 0:
            print(f"  rec {i}/{len(tasks)} success={episode_success}", flush=True)
    return records


def run_faithful_episodes(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    memory: FaithfulNpmMemory,
    workflow_fn: Any | None,
    out_path: Path,
    resume: bool,
    dataset_split: str,
) -> list[dict]:
    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume skip {out_path.name} n={len(recs)}", flush=True)
            return recs

    style = str(cfg["steering"].get("inject_style", "prefill_decode"))
    top_k = int(cfg["steering"].get("top_k", 8))
    cands = [float(a) for a in cfg["steering"]["alpha_candidates"]]
    eps = float(cfg["steering"].get("kl_eps", 0.2))
    max_steps = int(cfg["rollout"]["max_steps"])
    base_seed = int(cfg["rollout"]["seed"])
    env_factory = make_env_factory(cfg)

    records: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    for i, task in enumerate(tasks, 1):
        env = env_factory()
        tid = str(task["task_id"])
        traj_id = f"{tid}:npmf:0"
        seed = episode_seed(base_seed, tid, 0)
        raw_obs = env.reset(task)
        wf = workflow_fn(task) if workflow_fn is not None else None
        obs = apply_workflow(raw_obs, wf)
        goal = str(task.get("goal") or rec_goal({"prompt_text": raw_obs}) or "")
        vs = memory.synthesize(goal, top_k=top_k)
        tv = to_torch_vs(vs)
        agent.set_steering_layers(vectors=tv, alpha=cands[0], inject_style=style)  # type: ignore[arg-type]
        prompt0 = agent._prompt_text(obs)
        alpha, kl = agent.choose_alpha_kl(prompt0, cands, eps=eps)
        agent.steer_alpha = alpha

        actions: list[str] = []
        decisions: list[dict] = []
        token_count = 0
        termination = "max_steps"
        episode_success = False
        first_prompt = ""

        for step_i in range(max_steps):
            state = env.current_state()
            gen = agent.generate(obs, seed=seed + step_i * 17)
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
            dp["steered"] = True
            dp["alpha"] = alpha
            decisions.append(dp)
            if out.terminal:
                termination = str(out.failure_reason or "success")
                episode_success = bool(out.success)
                break
            obs = apply_workflow(out.observation, wf)

        agent.clear_steering()
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
                "alpha_used": alpha,
                "kl_probe": kl,
            }
        )
        records.append(rec)
        append_jsonl(out_path, rec)
        if i % 10 == 0:
            print(
                f"  rec {i}/{len(tasks)} success={episode_success} α={alpha:.1f} kl={kl:.3f}",
                flush=True,
            )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_npm_faithful.yaml"),
    )
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
    memory = FaithfulNpmMemory(Path(cfg["paths"]["faithful_memory"]))
    h0_src = Path(cfg["paths"].get("h0_base") or reports / "H0_base.jsonl")
    uce_src = Path(cfg["paths"].get("b_uce") or reports / "B_uce.jsonl")
    h0_dst = reports / "H0_base.jsonl"
    uce_dst = reports / "B_uce.jsonl"
    recs_h0 = _complete(h0_src, len(tasks)) or _complete(h0_dst, len(tasks))
    recs_uce = _complete(uce_src, len(tasks)) or _complete(uce_dst, len(tasks))
    if recs_h0 is not None and h0_src != h0_dst and not h0_dst.exists():
        shutil.copy2(h0_src, h0_dst)
    if recs_uce is not None and uce_src != uce_dst and not uce_dst.exists():
        shutil.copy2(uce_src, uce_dst)

    lib = UceLibrary.load(Path(cfg["paths"]["uce_library_evolved"]))
    print("loading agent...", flush=True)
    agent = build_agent(cfg)
    print("agent ready", flush=True)

    conditions: dict[str, Any] = {}
    recs_cache: dict[str, list[dict]] = {}

    if recs_h0 is None:
        print("H0 base (same runner, no NPM)", flush=True)
        t_h0 = time.time()
        recs_h0 = run_plain_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            workflow_fn=None,
            out_path=h0_dst,
            resume=args.resume,
            dataset_split=split,
            tag="h0",
        )
        wall_h0 = time.time() - t_h0
    else:
        wall_h0 = 0.0
    m0 = trajectory_metrics(recs_h0)
    conditions["H0"] = {
        "metrics": slim_metrics(m0),
        "gain_vs_h0_pp": 0.0,
        "wall_s": wall_h0,
    }
    recs_cache["H0"] = recs_h0
    print(f"H0={m0['success_rate']:.3f}", flush=True)

    if recs_uce is None:
        print("B-uce (frozen UCE, no NPM)", flush=True)
        t_u = time.time()
        recs_uce = run_plain_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            workflow_fn=make_workflow_fn(lib),
            out_path=uce_dst,
            resume=args.resume,
            dataset_split=split,
            tag="uce",
        )
        wall_u = time.time() - t_u
    else:
        wall_u = 0.0
    mu = trajectory_metrics(recs_uce)
    conditions["B_uce"] = {
        "metrics": slim_metrics(mu),
        "gain_vs_h0_pp": (mu["success_rate"] - m0["success_rate"]) * 100,
        "wall_s": wall_u,
    }
    recs_cache["B_uce"] = recs_uce
    print(f"B_uce={mu['success_rate']:.3f}", flush=True)

    print("N1 faithful NPM (no UCE)", flush=True)
    t0 = time.time()
    recs_n1 = run_faithful_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=memory,
        workflow_fn=None,
        out_path=reports / "N1_npm.jsonl",
        resume=args.resume,
        dataset_split=split,
    )
    m1 = trajectory_metrics(recs_n1)
    conditions["N1"] = {
        "metrics": slim_metrics(m1),
        "wall_s": time.time() - t0,
        "gain_vs_h0_pp": (m1["success_rate"] - m0["success_rate"]) * 100,
        "mean_alpha": float(sum(r.get("alpha_used") or 0 for r in recs_n1) / max(len(recs_n1), 1)),
    }
    recs_cache["N1"] = recs_n1
    print(f"  success={m1['success_rate']:.3f} vs_h0={conditions['N1']['gain_vs_h0_pp']:+.1f}", flush=True)

    print("N2 faithful NPM + UCE", flush=True)
    t1 = time.time()
    recs_n2 = run_faithful_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=memory,
        workflow_fn=make_workflow_fn(lib),
        out_path=reports / "N2_npm_uce.jsonl",
        resume=args.resume,
        dataset_split=split,
    )
    m2 = trajectory_metrics(recs_n2)
    conditions["N2"] = {
        "metrics": slim_metrics(m2),
        "wall_s": time.time() - t1,
        "gain_vs_h0_pp": (m2["success_rate"] - m0["success_rate"]) * 100,
        "gain_vs_uce_pp": (m2["success_rate"] - mu["success_rate"]) * 100,
        "mean_alpha": float(sum(r.get("alpha_used") or 0 for r in recs_n2) / max(len(recs_n2), 1)),
    }
    recs_cache["N2"] = recs_n2
    print(
        f"  success={m2['success_rate']:.3f} vs_uce={conditions['N2']['gain_vs_uce_pp']:+.1f}",
        flush=True,
    )
    agent.close()

    boot_n1 = paired_task_bootstrap(recs_h0, recs_n1)
    boot_n2 = paired_task_bootstrap(recs_uce, recs_n2)
    pass_n1 = conditions["N1"]["gain_vs_h0_pp"] >= min_pp and boot_n1["ci_lo"] > 0
    pass_n2 = conditions["N2"]["gain_vs_uce_pp"] >= min_pp and boot_n2["ci_lo"] > 0
    gate = {
        "n1_vs_h0_pp": conditions["N1"]["gain_vs_h0_pp"],
        "n2_vs_uce_pp": conditions["N2"]["gain_vs_uce_pp"],
        "bootstrap_n1_vs_h0": boot_n1,
        "bootstrap_n2_vs_uce": boot_n2,
        "n1_pass": pass_n1,
        "n2_pass": pass_n2,
        "gate_pass": bool(pass_n1 or pass_n2),
        "min_delta_pp": min_pp,
        "distill": "only_if_gate_and_teacher_has_npm",
    }
    payload = {
        "task_meta": task_meta,
        "model": {
            "name": cfg.get("model", {}).get("name"),
            "path": cfg.get("model", {}).get("path"),
            "adapter_path": cfg.get("model", {}).get("adapter_path"),
            "layers": cfg.get("steering", {}).get("layers"),
        },
        "conditions": conditions,
        "analysis": gate,
    }
    dump_json(reports / "results.json", payload)
    print("NPM_FAITHFUL_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
