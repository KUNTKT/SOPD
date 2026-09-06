"""Environment-agnostic multi-step rollout collection.

`rollout/generate.py` is hard-wired to the single-/few-step synthetic tool env and
its XML parser. This module takes any `AgentEnvironment` that exposes
`current_state()` and does episode-level generation with a growing context, which
is what ALFWorld needs. Trajectory records use the same schema, so downstream
peer alignment and statistics are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from rollout.resume import append_jsonl, load_jsonl
from stats.metrics import wilson_interval
from rollout.trajectory import (
    empty_decision,
    empty_trajectory,
    prompt_hash,
    state_fingerprint,
)

EnvFactory = Callable[[], Any]


def episode_seed(base_seed: int, task_id: str, group_index: int) -> int:
    import hashlib

    raw = f"{base_seed}:{task_id}:{group_index}".encode()
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def _decision_point(
    *,
    decision_id: str,
    step_index: int,
    prefix_text: str,
    prompt_text: str,
    state: dict[str, Any],
    verdict: dict[str, Any],
    action_text: str,
) -> dict[str, Any]:
    dp = empty_decision(
        decision_id=decision_id,
        step_index=step_index,
        prefix_text=prefix_text,
        state=state,
        state_fingerprint=state_fingerprint(state),
        predicted_tool=verdict.get("predicted_tool"),
        predicted_arguments=dict(verdict.get("predicted_arguments") or {}),
        gold_tool=verdict.get("gold_tool"),
        gold_arguments=dict(verdict.get("gold_arguments") or {}),
        tool_correct=bool(verdict.get("tool_correct")),
        argument_exact=bool(verdict.get("argument_exact")),
        argument_semantic=bool(verdict.get("argument_semantic")),
        parse_ok=bool(verdict.get("parse_ok")),
        decision_label=str(verdict.get("decision_label", "UNSCORABLE")),
        failure_reason=verdict.get("failure_reason"),
        is_premature_downstream_action=bool(
            verdict.get("is_premature_downstream_action")
        ),
        valid_tools=list(state.get("valid_tools") or []),
    )
    # Exact templated prefix so a later audit can rescore this decision without
    # re-deriving the chat template.
    dp["prompt_text"] = prompt_text
    dp["raw_action_text"] = action_text
    return dp


def run_episode(
    agent: Any,
    env: Any,
    task: dict[str, Any],
    *,
    trajectory_id: str,
    seed: int,
    dataset_split: str,
    max_steps: int = 40,
) -> dict[str, Any]:
    obs = env.reset(task)
    first_prompt = ""
    actions: list[str] = []
    observations: list[str] = [obs]
    decisions: list[dict[str, Any]] = []
    full_parts: list[str] = [obs]
    token_count = 0
    termination = "max_steps"
    episode_success = False

    for step_i in range(max_steps):
        state = env.current_state()
        gen = agent.generate(obs, seed=seed + step_i * 17)
        if step_i == 0:
            first_prompt = gen.prompt_text or obs
        action_text = gen.text
        token_count += int(gen.completion_token_count)
        actions.append(action_text)
        full_parts.append(action_text)

        verdict = env.verify_decision(state, action_text)
        decisions.append(
            _decision_point(
                decision_id=f"{trajectory_id}_d{step_i}",
                step_index=step_i,
                prefix_text=obs,
                prompt_text=gen.prompt_text or obs,
                state=state,
                verdict=verdict,
                action_text=action_text,
            )
        )

        out = env.step(action_text)
        observations.append(out.tool_result)
        full_parts.append(out.tool_result)
        if out.terminal:
            episode_success = bool(out.success)
            termination = "success" if episode_success else (
                out.failure_reason or "terminal"
            )
            break
        obs = out.observation

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
    traj["task_type"] = task.get("task_type")
    traj["env_source"] = task.get("source")
    return traj


def _finalize(
    slot: dict[str, Any],
    *,
    dataset_split: str,
    agent: Any,
) -> dict[str, Any]:
    task = slot["task"]
    meta = getattr(agent, "meta", None)
    first_prompt = slot["first_prompt"] or slot["observations"][0]
    traj = empty_trajectory(
        task_id=task["task_id"],
        trajectory_id=slot["trajectory_id"],
        dataset_split=dataset_split,
        prompt_text=first_prompt,
        full_text="\n".join(slot["full_parts"]),
        actions=slot["actions"],
        observations=slot["observations"],
        decision_points=slot["decisions"],
        final_reward=1.0 if slot["episode_success"] else 0.0,
        episode_success=slot["episode_success"],
        token_count=slot["token_count"],
        tool_call_count=len(slot["actions"]),
        trajectory_length=len(slot["actions"]),
        termination_reason=slot["termination"],
        model_name=getattr(meta, "model_name", "") if meta else "",
        tokenizer_hash=getattr(meta, "tokenizer_hash", "") if meta else "",
        prompt_hash=prompt_hash(first_prompt),
        random_seed=slot["seed"],
    )
    traj["chat_template_hash"] = getattr(meta, "chat_template_hash", "") if meta else ""
    traj["task_type"] = task.get("task_type")
    traj["env_source"] = task.get("source")
    return traj


def collect_episodes(
    tasks: Iterable[dict[str, Any]],
    agent: Any,
    env_factory: EnvFactory,
    *,
    n_rollouts: int,
    dataset_split: str,
    base_seed: int,
    out_path: Any,
    resume: bool = True,
    max_steps: int = 40,
    batch_size: int = 16,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    propose_cb: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    n_env_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run `batch_size` episodes in lockstep so generation can be batched.

    Multi-step episodes are long; generating one step at a time for one episode
    at a time wastes almost all of the available throughput. Envs advance in
    lockstep and every still-active env contributes one prompt per batch.

    `propose_cb` may replace the sampled action, which is how EXP12 samples from
    `mu = (1-eps) p_theta + eps p_peer` without touching the policy. Whatever it
    returns under `"proposal"` is recorded on the decision point, because an
    update that cannot tell which actions came from the mixture cannot correct
    for having sampled off-policy.

    `n_env_workers` above 1 moves environment stepping into worker processes.
    ALFWorld's engine is CPU-bound Python, so without this the GPU idles through
    every step. Results are unchanged: a slot stays pinned to one worker and
    replies are reordered into request order.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume or not path.exists():
        path.write_text("")
        records: list[dict[str, Any]] = []
        done: set[str] = set()
    else:
        records = load_jsonl(path)
        done = {str(r["trajectory_id"]) for r in records}

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
                    "seed": episode_seed(base_seed, tid, k),
                }
            )

    use_batch = batch_size > 1 and callable(getattr(agent, "generate_batch", None))
    width = batch_size if use_batch else 1

    pool = None
    if n_env_workers > 1:
        from rollout.env_pool import EnvPool

        pool = EnvPool(env_factory, n_workers=min(n_env_workers, width))

    try:
        cursor = 0
        while cursor < len(pending):
            chunk = pending[cursor : cursor + width]
            cursor += len(chunk)

            slots: list[dict[str, Any]] = []
            if pool is not None:
                observations = pool.reset(
                    list(range(len(chunk))), [item["task"] for item in chunk]
                )
                envs: list[Any] = [None] * len(chunk)
            else:
                envs = [env_factory() for _ in chunk]
                observations = [
                    env.reset(item["task"]) for env, item in zip(envs, chunk)
                ]
            for index, (item, env, obs) in enumerate(zip(chunk, envs, observations)):
                slots.append(
                    {
                        **item,
                        "index": index,
                        "env": env,
                        "obs": obs,
                        "actions": [],
                        "executed": [],
                        "observations": [obs],
                        "decisions": [],
                        "full_parts": [obs],
                        "token_count": 0,
                        "first_prompt": "",
                        "termination": "max_steps",
                        "episode_success": False,
                        "active": True,
                    }
                )

            try:
                for step_i in range(max_steps):
                    active = [s for s in slots if s["active"]]
                    if not active:
                        break
                    prompts = [s["obs"] for s in active]
                    seeds = [s["seed"] + step_i * 17 for s in active]
                    if use_batch and len(prompts) > 1:
                        # Per-request seeds, not one seed for the batch. Sibling
                        # rollouts of a task share a prompt at step 0, so a shared
                        # seed makes them the same trajectory and no group can ever
                        # be mixed.
                        gens = agent.generate_batch(prompts, seeds=seeds)
                    else:
                        gens = [
                            agent.generate(prompts[t], seed=seeds[t])
                            for t in range(len(prompts))
                        ]

                    # The proposal hook needs `A(s)` before it can substitute an
                    # action, so the state is fetched up front only when a hook is
                    # installed. Otherwise the step call returns it for free.
                    pre_states: list[dict[str, Any]] | None = None
                    if propose_cb is not None:
                        if pool is not None:
                            pre_states = pool.state([s["index"] for s in active])
                        else:
                            pre_states = [s["env"].current_state() for s in active]

                    actions: list[str] = []
                    proposals: list[dict[str, Any] | None] = []
                    for i, (slot, gen) in enumerate(zip(active, gens)):
                        if step_i == 0:
                            slot["first_prompt"] = gen.prompt_text or slot["obs"]
                        slot["token_count"] += int(gen.completion_token_count)
                        action_text = gen.text
                        proposal = None
                        if propose_cb is not None and pre_states is not None:
                            proposal = propose_cb(
                                {
                                    "task": slot["task"],
                                    "trajectory_id": slot["trajectory_id"],
                                    "step_index": step_i,
                                    "observation": slot["obs"],
                                    "admissible": list(
                                        pre_states[i].get("valid_tools") or []
                                    ),
                                    "executed": list(slot["executed"]),
                                    "sampled_text": action_text,
                                }
                            )
                            if proposal and proposal.get("action_text"):
                                action_text = str(proposal["action_text"])
                        actions.append(action_text)
                        proposals.append(proposal)

                    if pool is not None:
                        outs = pool.step([s["index"] for s in active], actions)
                    else:
                        outs = []
                        for slot, action_text in zip(active, actions):
                            env = slot["env"]
                            state = env.current_state()
                            verdict = env.verify_decision(state, action_text)
                            out = env.step(action_text)
                            outs.append(
                                {
                                    "state": state,
                                    "verdict": verdict,
                                    "observation": out.observation,
                                    "tool_result": out.tool_result,
                                    "success": bool(out.success),
                                    "terminal": bool(out.terminal),
                                    "failure_reason": out.failure_reason,
                                }
                            )

                    for slot, gen, action_text, proposal, out in zip(
                        active, gens, actions, proposals, outs
                    ):
                        slot["actions"].append(action_text)
                        slot["full_parts"].append(action_text)
                        verdict = out["verdict"]
                        dp = _decision_point(
                            decision_id=f"{slot['trajectory_id']}_d{step_i}",
                            step_index=step_i,
                            prefix_text=slot["obs"],
                            prompt_text=gen.prompt_text or slot["obs"],
                            state=out["state"],
                            verdict=verdict,
                            action_text=action_text,
                        )
                        if proposal:
                            dp["proposal"] = proposal
                        slot["decisions"].append(dp)
                        if verdict.get("predicted_tool"):
                            slot["executed"].append(str(verdict["predicted_tool"]))

                        slot["observations"].append(out["tool_result"])
                        slot["full_parts"].append(out["tool_result"])
                        if out["terminal"]:
                            slot["episode_success"] = bool(out["success"])
                            slot["termination"] = (
                                "success"
                                if out["success"]
                                else (out["failure_reason"] or "terminal")
                            )
                            slot["active"] = False
                        else:
                            slot["obs"] = out["observation"]
            finally:
                if pool is not None:
                    pool.close_slots([s["index"] for s in slots])
                else:
                    for slot in slots:
                        close = getattr(slot["env"], "close", None)
                        if callable(close):
                            close()

            for slot in slots:
                rec = _finalize(slot, dataset_split=dataset_split, agent=agent)
                append_jsonl(path, rec)
                records.append(rec)
                done.add(rec["trajectory_id"])
                if progress_cb:
                    progress_cb(rec)
    finally:
        if pool is not None:
            pool.close()

    return records


def group_by_task(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(str(rec["task_id"]), []).append(rec)
    return groups


def mixed_groups(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Tasks with at least one success and one failure — SGCD's `0 < sum R < n`.

    A homogeneous group carries no sibling contrast and must never enter a PACW
    update.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_id, recs in group_by_task(records).items():
        wins = [r for r in recs if r.get("episode_success")]
        losses = [r for r in recs if not r.get("episode_success")]
        if wins and losses:
            out[task_id] = {"success": wins, "failure": losses}
    return out


def rollout_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n_trajectories": 0,
            "success_rate": 0.0,
            "admissible_rate": 0.0,
            "mean_steps": 0.0,
            "n_tasks": 0,
            "n_mixed_groups": 0,
            "mixed_group_rate": 0.0,
            "mixed_group_rate_ci": [0.0, 1.0],
        }
    n_dp = sum(len(r.get("decision_points") or []) for r in records)
    n_ok = sum(
        1
        for r in records
        for dp in (r.get("decision_points") or [])
        if dp.get("parse_ok")
    )
    groups = group_by_task(records)
    mixed = mixed_groups(records)
    # The task is the independent unit, so the mixed-group rate is a binomial
    # proportion over tasks. It is reported with an interval because it is a gate
    # quantity and the counts involved are small.
    lo, hi = wilson_interval(len(mixed), len(groups))
    return {
        "n_trajectories": n,
        "success_rate": sum(1 for r in records if r.get("episode_success")) / n,
        "admissible_rate": (n_ok / n_dp) if n_dp else 0.0,
        "mean_steps": sum(int(r.get("trajectory_length") or 0) for r in records) / n,
        "n_tasks": len(groups),
        "n_mixed_groups": len(mixed),
        "mixed_group_rate": len(mixed) / len(groups) if groups else 0.0,
        "mixed_group_rate_ci": [lo, hi],
    }
