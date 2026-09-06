"""On-policy rollout collection (no activation steering)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from decisions.schema import build_decision_point
from environments.synthetic_tool_env import SyntheticToolEnv
from rollout.parser import parse_tool_call
from rollout.resume import append_jsonl, completed_ids, load_jsonl
from rollout.trajectory import empty_trajectory, prompt_hash


def rollout_seed(base_seed: int, task_id: str, group_index: int) -> int:
    raw = f"{base_seed}:{task_id}:{group_index}".encode()
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def _finalize_trajectory(
    *,
    task: dict[str, Any],
    trajectory_id: str,
    seed: int,
    dataset_split: str,
    env: SyntheticToolEnv,
    actions: list[str],
    observations: list[str],
    decisions: list[dict[str, Any]],
    full_parts: list[str],
    token_count: int,
    first_prompt: str,
    termination: str,
    agent: Any,
) -> dict[str, Any]:
    episode_success = termination == "success"
    meta = getattr(agent, "meta", None)
    traj = empty_trajectory(
        task_id=task["task_id"],
        trajectory_id=trajectory_id,
        dataset_split=dataset_split,
        prompt_text=first_prompt or observations[0],
        full_text="\n".join(full_parts),
        actions=actions,
        observations=observations,
        decision_points=decisions,
        final_reward=1.0 if episode_success else 0.0,
        episode_success=episode_success,
        token_count=token_count,
        tool_call_count=len(actions),
        trajectory_length=len(actions),
        termination_reason=termination,
        model_name=getattr(meta, "model_name", "") if meta else "",
        tokenizer_hash=getattr(meta, "tokenizer_hash", "") if meta else "",
        prompt_hash=prompt_hash(first_prompt or observations[0]),
        random_seed=seed,
    )
    traj["chat_template_hash"] = getattr(meta, "chat_template_hash", "") if meta else ""
    return traj


def run_single_rollout(
    agent: Any,
    task: dict[str, Any],
    *,
    trajectory_id: str,
    seed: int,
    dataset_split: str,
    max_steps: int = 3,
) -> dict[str, Any]:
    env = SyntheticToolEnv()
    obs = env.reset(task)
    actions: list[str] = []
    observations: list[str] = [obs]
    decisions: list[dict[str, Any]] = []
    full_parts: list[str] = [obs]
    token_count = 0
    termination = "max_steps"
    first_prompt = ""

    for step_i in range(max_steps):
        if env._terminal:
            break
        state = env.current_state()
        gen = agent.generate(obs, seed=seed + step_i * 17)
        if step_i == 0:
            first_prompt = gen.prompt_text or obs
        action_text = gen.text
        token_count += int(gen.completion_token_count)
        actions.append(action_text)
        full_parts.append(action_text)

        parsed = parse_tool_call(action_text)
        dp = build_decision_point(
            decision_id=f"{trajectory_id}_d{step_i}",
            state=state,
            action={
                "tool_name": parsed.tool_name,
                "arguments": parsed.arguments,
                "parse_ok": parsed.parse_ok,
            },
            prefix_text=obs,
            task=task,
        )
        decisions.append(dp)

        out = env.step(action_text)
        observations.append(out.tool_result)
        full_parts.append(out.tool_result)
        if out.terminal:
            if out.success and env.step_index >= len(task["gold_plan"]):
                termination = "success"
            else:
                termination = out.failure_reason or "terminal"
            break
        obs = out.observation

    return _finalize_trajectory(
        task=task,
        trajectory_id=trajectory_id,
        seed=seed,
        dataset_split=dataset_split,
        env=env,
        actions=actions,
        observations=observations,
        decisions=decisions,
        full_parts=full_parts,
        token_count=token_count,
        first_prompt=first_prompt,
        termination=termination,
        agent=agent,
    )


def collect_rollouts(
    tasks: list[dict[str, Any]],
    agent: Any,
    *,
    n_rollouts: int,
    dataset_split: str,
    base_seed: int,
    out_path: Any,
    resume: bool = True,
    max_steps: int = 3,
    batch_size: int = 8,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    env_factory: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect rollouts with optional batched first-step generation."""
    make_env = env_factory or SyntheticToolEnv
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume or not path.exists():
        path.write_text("")
        done: set[str] = set()
        records: list[dict[str, Any]] = []
    else:
        records = load_jsonl(path)
        done = {str(r["trajectory_id"]) for r in records}

    # Build pending work items
    pending: list[dict[str, Any]] = []
    for task in tasks:
        tid = str(task["task_id"])
        for k in range(n_rollouts):
            traj_id = f"{tid}::k{k}"
            if traj_id in done:
                continue
            pending.append(
                {
                    "task": task,
                    "trajectory_id": traj_id,
                    "seed": rollout_seed(base_seed, tid, k),
                    "k": k,
                }
            )

    use_batch = batch_size > 1 and callable(getattr(agent, "generate_batch", None))
    i = 0
    while i < len(pending):
        chunk = pending[i : i + (batch_size if use_batch else 1)]
        i += len(chunk)

        # Initialize envs
        states: list[dict[str, Any]] = []
        for item in chunk:
            env = make_env()
            obs = env.reset(item["task"])
            states.append(
                {
                    "item": item,
                    "env": env,
                    "obs": obs,
                    "actions": [],
                    "observations": [obs],
                    "decisions": [],
                    "full_parts": [obs],
                    "token_count": 0,
                    "first_prompt": "",
                    "termination": "max_steps",
                    "active": True,
                }
            )

        for step_i in range(max_steps):
            active_idx = [j for j, s in enumerate(states) if s["active"]]
            if not active_idx:
                break
            obs_list = [states[j]["obs"] for j in active_idx]
            seeds = [states[j]["item"]["seed"] + step_i * 17 for j in active_idx]
            if use_batch and len(obs_list) > 1:
                gens = agent.generate_batch(obs_list, seed=seeds[0])
            else:
                gens = [
                    agent.generate(obs_list[t], seed=seeds[t]) for t in range(len(obs_list))
                ]

            for local_t, j in enumerate(active_idx):
                s = states[j]
                gen = gens[local_t]
                if step_i == 0:
                    s["first_prompt"] = gen.prompt_text or s["obs"]
                action_text = gen.text
                s["token_count"] += int(gen.completion_token_count)
                s["actions"].append(action_text)
                s["full_parts"].append(action_text)

                env = s["env"]
                state = env.current_state()
                if getattr(env, "decision_after_step", False):
                    out = env.step(action_text)
                    action_dict = env.parse_action(action_text)
                    action_dict = dict(action_dict)
                    if getattr(env, "_last_eval", None) is not None:
                        action_dict["tests_pass"] = bool(env._last_eval.get("all_pass"))
                        action_dict["failure_reason"] = out.failure_reason
                    dp = build_decision_point(
                        decision_id=f"{s['item']['trajectory_id']}_d{step_i}",
                        state=state,
                        action=action_dict,
                        prefix_text=s["obs"],
                        task=s["item"]["task"],
                    )
                else:
                    if hasattr(env, "parse_action"):
                        action_dict = env.parse_action(action_text)
                    else:
                        parsed = parse_tool_call(action_text)
                        action_dict = {
                            "tool_name": parsed.tool_name,
                            "arguments": parsed.arguments,
                            "parse_ok": parsed.parse_ok,
                        }
                    dp = build_decision_point(
                        decision_id=f"{s['item']['trajectory_id']}_d{step_i}",
                        state=state,
                        action=action_dict,
                        prefix_text=s["obs"],
                        task=s["item"]["task"],
                    )
                    out = env.step(action_text)
                s["decisions"].append(dp)
                s["observations"].append(out.tool_result)
                s["full_parts"].append(out.tool_result)
                if out.terminal:
                    task = s["item"]["task"]
                    if out.success and env.step_index >= len(task["gold_plan"]):
                        s["termination"] = "success"
                    else:
                        s["termination"] = out.failure_reason or "terminal"
                    s["active"] = False
                else:
                    s["obs"] = out.observation

        for s in states:
            rec = _finalize_trajectory(
                task=s["item"]["task"],
                trajectory_id=s["item"]["trajectory_id"],
                seed=s["item"]["seed"],
                dataset_split=dataset_split,
                env=s["env"],
                actions=s["actions"],
                observations=s["observations"],
                decisions=s["decisions"],
                full_parts=s["full_parts"],
                token_count=s["token_count"],
                first_prompt=s["first_prompt"],
                termination=s["termination"],
                agent=agent,
            )
            append_jsonl(path, rec)
            records.append(rec)
            done.add(rec["trajectory_id"])
            if progress_cb:
                progress_cb(rec)

    return records
