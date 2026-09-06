#!/usr/bin/env python3
"""DVPD S0: pair, first full-action fork, remain-budget no-workflow continuations, gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import ensure_alfworld_env, make_env_factory  # noqa: E402
from dvpd_s0_lib import (  # noqa: E402
    WORKFLOW_HEAD,
    assert_clean_prompt,
    assert_replay_aligned,
    delta_q,
    dump_json,
    evaluate_gate_s0,
    fingerprint_at,
    first_full_action_diverge,
    h_remain,
    load_cfg,
    load_jsonl,
    prefix_commands,
    predicted_tool_at,
    q_hat,
    refuse_forbidden_split,
    replicate_terminal,
    retrieve_readonly,
    sample_remaining_tasks,
    suffix_seed,
    workflow_hash,
)
from rollout.multistep import _decision_point, episode_seed  # noqa: E402
from rollout.trajectory import empty_trajectory, state_fingerprint  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_fast_agent(cfg: dict):
    """vLLM + coldstart LoRA. Per-request seeds, batched decode, ~0.35 GPU mem."""
    from environments.alfworld_adapter import ALFWORLD_FEWSHOT_PROMPT
    from models.vllm_agent import VLLMAgent

    vcfg = cfg.get("vllm") or {}
    agent = VLLMAgent(
        cfg["model"]["path"],
        max_new_tokens=int(cfg["rollout"].get("max_new_tokens", 32)),
        temperature=float(cfg["rollout"].get("temperature", 1.0)),
        top_p=float(cfg["rollout"].get("top_p", 1.0)),
        gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.35)),
        max_model_len=int(vcfg.get("max_model_len", 4096)),
        system_prompt=ALFWORLD_FEWSHOT_PROMPT,
        seed=int(cfg["rollout"].get("seed", 2020)),
        lora_path=cfg["model"].get("adapter_path"),
        enable_sleep_mode=False,
        enforce_eager=True,
        max_lora_rank=int(vcfg.get("max_lora_rank", 32)),
    )

    def _prompt_text(observation: str) -> str:
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": observation},
        ]
        try:
            return agent.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return agent.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    agent._prompt_text = _prompt_text  # type: ignore[method-assign]
    return agent


def close_fast_agent(agent) -> None:
    llm = getattr(agent, "llm", None)
    if llm is not None:
        del agent.llm
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def force_only(
    env_factory,
    task: dict,
    prefix: list[str],
    forced: str,
    original_fp: str,
    other_pre_fp: str | None,
) -> dict[str, Any]:
    env = env_factory()
    replay_prefix(env, task, prefix)
    pre_state = env.current_state()
    pre_fp = state_fingerprint(pre_state)
    assert_replay_aligned(original_fp, pre_fp, other_pre_fp or pre_fp)
    admissible = list(pre_state.get("valid_tools") or [])
    accepted = forced in admissible
    out = env.step(forced)
    return {
        "accepted": accepted,
        "pre_fp": pre_fp,
        "immediate": bool(out.terminal),
        "reward": 1 if out.success else 0,
        "obs": None if out.terminal else out.observation,
        "env": None if out.terminal else env,
    }


def continue_fork_batched(
    *,
    agent,
    env_factory,
    task: dict,
    prefix: list[str],
    a_m: str,
    a_0: str,
    t_star: int,
    pair_i: int,
    k_n: int,
    cfg: dict,
    workflow_text: str,
    wf_hash: str,
    original_fp: str,
    other_pre_fp: str,
) -> dict[str, Any]:
    remain = h_remain(int(cfg["rollout"]["max_steps"]), t_star)
    probe_m = force_only(env_factory, task, prefix, a_m, original_fp, other_pre_fp)
    probe_0 = force_only(env_factory, task, prefix, a_0, original_fp, probe_m["pre_fp"])
    if probe_m["immediate"] and probe_0["immediate"]:
        return {
            "accepted_m": probe_m["accepted"],
            "accepted_0": probe_0["accepted"],
            "immediate_m": True,
            "immediate_0": True,
            "pre_fp": probe_m["pre_fp"],
            "outcomes_m": replicate_terminal(probe_m["reward"], k_n),
            "outcomes_0": replicate_terminal(probe_0["reward"], k_n),
        }
    tid = str(task["task_id"])
    slots: list[dict[str, Any]] = []
    for arm, forced, probe in (("m", a_m, probe_m), ("0", a_0, probe_0)):
        if probe["immediate"]:
            for k in range(k_n):
                slots.append({"arm": arm, "k": k, "alive": False, "reward": int(probe["reward"]), "j": 0})
            continue
        for k in range(k_n):
            env = env_factory()
            replay_prefix(env, task, prefix)
            out = env.step(forced)
            if out.terminal:
                slots.append({"arm": arm, "k": k, "alive": False, "reward": 1 if out.success else 0, "j": 0})
                continue
            assert_clean_prompt(out.observation, workflow_text=workflow_text, wf_hash=wf_hash)
            slots.append(
                {
                    "arm": arm,
                    "k": k,
                    "alive": True,
                    "reward": 0,
                    "j": 0,
                    "env": env,
                    "obs": out.observation,
                }
            )
    while any(s["alive"] for s in slots):
        live = [s for s in slots if s["alive"]]
        obs = [s["obs"] for s in live]
        seeds = [suffix_seed(tid, pair_i, int(s["k"])) + int(s["j"]) * 17 for s in live]
        for o in obs:
            assert_clean_prompt(o, workflow_text=workflow_text, wf_hash=wf_hash)
        gens = agent.generate_batch(obs, seeds=seeds)
        for s, gen in zip(live, gens):
            assert_clean_prompt(gen.prompt_text or s["obs"], workflow_text=workflow_text, wf_hash=wf_hash)
            step_out = s["env"].step(gen.text)
            s["j"] += 1
            if step_out.terminal:
                s["alive"] = False
                s["reward"] = 1 if step_out.success else 0
            elif s["j"] >= remain:
                s["alive"] = False
                s["reward"] = 0
            else:
                s["obs"] = step_out.observation
    outcomes_m = [int(s["reward"]) for s in slots if s["arm"] == "m"]
    outcomes_0 = [int(s["reward"]) for s in slots if s["arm"] == "0"]
    return {
        "accepted_m": probe_m["accepted"],
        "accepted_0": probe_0["accepted"],
        "immediate_m": bool(probe_m["immediate"]),
        "immediate_0": bool(probe_0["immediate"]),
        "pre_fp": probe_m["pre_fp"],
        "outcomes_m": outcomes_m,
        "outcomes_0": outcomes_0,
    }


def spawn_slots(
    *,
    env_factory,
    task: dict,
    prefix: list[str],
    forced: str,
    probe: dict,
    arm: str,
    k_n: int,
    tid: str,
    pair_i: int,
    remain: int,
    workflow_text: str,
    wf_hash: str,
    fork_key: tuple,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    if probe["immediate"]:
        for k in range(k_n):
            slots.append(
                {
                    "fork_key": fork_key,
                    "arm": arm,
                    "k": k,
                    "alive": False,
                    "reward": int(probe["reward"]),
                    "j": 0,
                    "remain": remain,
                    "tid": tid,
                    "pair_i": pair_i,
                    "workflow_text": workflow_text,
                    "wf_hash": wf_hash,
                }
            )
        return slots
    for k in range(k_n):
        env = env_factory()
        replay_prefix(env, task, prefix)
        out = env.step(forced)
        if out.terminal:
            slots.append(
                {
                    "fork_key": fork_key,
                    "arm": arm,
                    "k": k,
                    "alive": False,
                    "reward": 1 if out.success else 0,
                    "j": 0,
                    "remain": remain,
                    "tid": tid,
                    "pair_i": pair_i,
                    "workflow_text": workflow_text,
                    "wf_hash": wf_hash,
                }
            )
            continue
        assert_clean_prompt(out.observation, workflow_text=workflow_text, wf_hash=wf_hash)
        slots.append(
            {
                "fork_key": fork_key,
                "arm": arm,
                "k": k,
                "alive": True,
                "reward": 0,
                "j": 0,
                "remain": remain,
                "tid": tid,
                "pair_i": pair_i,
                "env": env,
                "obs": out.observation,
                "workflow_text": workflow_text,
                "wf_hash": wf_hash,
            }
        )
    return slots


def step_slots(agent, slots: list[dict[str, Any]]) -> None:
    while any(s["alive"] for s in slots):
        live = [s for s in slots if s["alive"]]
        obs = [s["obs"] for s in live]
        seeds = [suffix_seed(s["tid"], s["pair_i"], int(s["k"])) + int(s["j"]) * 17 for s in live]
        for s, o in zip(live, obs):
            assert_clean_prompt(o, workflow_text=s["workflow_text"], wf_hash=s["wf_hash"])
        gens = agent.generate_batch(obs, seeds=seeds)
        for s, gen in zip(live, gens):
            assert_clean_prompt(gen.prompt_text or s["obs"], workflow_text=s["workflow_text"], wf_hash=s["wf_hash"])
            step_out = s["env"].step(gen.text)
            s["j"] += 1
            if step_out.terminal:
                s["alive"] = False
                s["reward"] = 1 if step_out.success else 0
            elif s["j"] >= s["remain"]:
                s["alive"] = False
                s["reward"] = 0
            else:
                s["obs"] = step_out.observation


def collect_episode(
    *,
    agent,
    env_factory,
    task: dict,
    cfg: dict,
    pair_i: int,
    workflow_text: str | None,
    dataset_split: str,
) -> dict:
    max_steps = int(cfg["rollout"]["max_steps"])
    base_seed = int(cfg["rollout"]["seed"])
    tid = str(task["task_id"])
    seed = episode_seed(base_seed, tid, int(pair_i))
    env = env_factory()
    raw_obs = env.reset(task)
    wf = workflow_text or None
    obs = f"{wf}\n\n{raw_obs}" if wf else raw_obs
    actions: list[str] = []
    decisions: list[dict] = []
    episode_success = False
    termination = "max_steps"
    first_prompt = ""
    traj_id = f"{tid}:dvpd:{pair_i}"
    for step_i in range(max_steps):
        state = env.current_state()
        gen = agent.generate(obs, seed=seed + step_i * 17)
        if step_i == 0:
            first_prompt = gen.prompt_text or obs
        action_text = gen.text
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
        if out.terminal:
            termination = str(out.failure_reason or "success")
            episode_success = bool(out.success)
            break
        raw_obs = out.observation
        obs = f"{wf}\n\n{raw_obs}" if wf else raw_obs
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
            "prompt_text": first_prompt,
            "pair_i": int(pair_i),
            "random_seed": seed,
        }
    )
    return rec


def replay_prefix(env, task: dict, commands: list[str]) -> str:
    raw = env.reset(task)
    for cmd in commands:
        out = env.step(cmd)
        if out.terminal:
            return raw
        raw = out.observation
    return raw


def force_and_maybe_continue(
    *,
    agent,
    env_factory,
    task: dict,
    prefix: list[str],
    forced: str,
    t_star: int,
    k: int,
    pair_i: int,
    cfg: dict,
    workflow_text: str,
    wf_hash: str,
    original_fp: str,
    other_pre_fp: str | None,
) -> dict[str, Any]:
    h_max = int(cfg["rollout"]["max_steps"])
    remain = h_remain(h_max, t_star)
    env = env_factory()
    replay_prefix(env, task, prefix)
    pre_state = env.current_state()
    pre_fp = state_fingerprint(pre_state)
    assert_replay_aligned(original_fp, pre_fp, other_pre_fp or pre_fp)
    admissible = list(pre_state.get("valid_tools") or [])
    accepted = forced in admissible
    verdict = env.verify_decision(pre_state, forced)
    out = env.step(forced)
    if out.terminal:
        reward = 1 if out.success else 0
        return {
            "reward": reward,
            "accepted": accepted,
            "pre_fp": pre_fp,
            "immediate": True,
            "outcomes": replicate_terminal(reward, 1),
        }
    raw_obs = out.observation
    assert_clean_prompt(raw_obs, workflow_text=workflow_text, wf_hash=wf_hash)
    seed0 = suffix_seed(str(task["task_id"]), pair_i, k)
    success = False
    for j in range(remain):
        assert_clean_prompt(raw_obs, workflow_text=workflow_text, wf_hash=wf_hash)
        gen = agent.generate(raw_obs, seed=seed0 + j * 17)
        assert_clean_prompt(gen.prompt_text or raw_obs, workflow_text=workflow_text, wf_hash=wf_hash)
        step_out = env.step(gen.text)
        if step_out.terminal:
            success = bool(step_out.success)
            break
        raw_obs = step_out.observation
    return {
        "reward": 1 if success else 0,
        "accepted": accepted,
        "pre_fp": pre_fp,
        "immediate": False,
        "outcomes": [1 if success else 0],
        "predicted_tool": verdict.get("predicted_tool"),
    }


def append_all_results(archive: Path, gate: dict) -> None:
    path = archive / "ALL_RESULTS.md"
    marker = "## DVPD Gate S0"
    block = (
        f"\n{marker}\n\n"
        "更新：2026-09-05。部署态反事实可迁移信号。未训练。valid_unseen 未打开。\n\n"
        f"- Gate S0: `{'PASS' if gate.get('pass') else 'FAIL'}` reasons=`{gate.get('reasons')}`\n"
        f"- n_fork={gate.get('n_fork_states')} n_tasks={gate.get('n_tasks')} "
        f"task_macro={gate.get('task_macro_mean')} CI=[{gate.get('ci_lo')}, {gate.get('ci_hi')}]\n"
        f"- signed_large={gate.get('signed_large_count')} abs_large={gate.get('abs_large_count')} "
        f"neg_large={gate.get('negative_large_count')} accept={gate.get('accept_rate')}\n"
        f"- 主张：{gate.get('claim')}\n"
        f"- 范围：{gate.get('scope')}\n"
        "- 产物：`reports/dvpd_s0/` · `notes/dvpd_s0_plan.md`\n"
        "- 停止。不自动训练，不调 K/阈值。\n"
    )
    text = path.read_text() if path.exists() else ""
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n" + block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    refuse_forbidden_split(str((cfg.get("env") or {}).get("eval_split") or ""))
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    data = Path(cfg["paths"]["data_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    split = sample_remaining_tasks(cfg)
    dump_json(reports / "task_split.json", split)
    by_id = {t["task_id"]: t for t in split["tasks"]}
    pair_seeds = [int(x) for x in split["pair_seeds"]]
    k_n = int(cfg["K"])

    lib = UceLibrary.load(Path(cfg["paths"]["uce_library_evolved"]))
    retrieve_hits = []
    for t in split["tasks"]:
        text, eid, score = retrieve_readonly(lib, t)
        retrieve_hits.append(
            {
                "task_id": t["task_id"],
                "entry_id": eid,
                "score": score,
                "workflow_hash": workflow_hash(text),
            }
        )
    dump_json(reports / "retrieve_hits.json", retrieve_hits)
    wf_by_task = {}
    for t in split["tasks"]:
        text, eid, score = retrieve_readonly(lib, t)
        wf_by_task[t["task_id"]] = {
            "text": text,
            "entry_id": eid,
            "hash": workflow_hash(text),
        }
        prev = next(h for h in retrieve_hits if h["task_id"] == t["task_id"])
        if prev["workflow_hash"] != wf_by_task[t["task_id"]]["hash"] or prev["entry_id"] != eid:
            raise RuntimeError("retrieve not stable across a second pass")

    pairs_path = data / "pairs.jsonl"
    done_pairs = {}
    if args.resume and pairs_path.exists():
        for rec in load_jsonl(pairs_path):
            done_pairs[(rec["task_id"], int(rec["pair_i"]))] = rec

    need_pairs = [(tid, p) for tid in split["ids"] for p in pair_seeds if (tid, p) not in done_pairs]
    if need_pairs:
        from steerable_alfworld_agent import build_agent

        print(f"pair {len(need_pairs)} remaining of {len(split['ids']) * len(pair_seeds)}", flush=True)
        agent = build_agent(cfg)
        agent.clear_steering()
        env_factory = make_env_factory(cfg)
        t0 = time.time()
        for i, (tid, pair_i) in enumerate(need_pairs, 1):
            task = by_id[tid]
            wf = wf_by_task[tid]
            base = collect_episode(
                agent=agent,
                env_factory=env_factory,
                task=task,
                cfg=cfg,
                pair_i=pair_i,
                workflow_text=None,
                dataset_split="dvpd_s0_base",
            )
            uce = collect_episode(
                agent=agent,
                env_factory=env_factory,
                task=task,
                cfg=cfg,
                pair_i=pair_i,
                workflow_text=wf["text"] or None,
                dataset_split="dvpd_s0_uce",
            )
            rec = {
                "task_id": tid,
                "pair_i": pair_i,
                "task_type": task.get("task_type"),
                "workflow_hash": wf["hash"],
                "workflow_id": wf["entry_id"],
                "base": base,
                "uce": uce,
            }
            _append(pairs_path, rec)
            done_pairs[(tid, pair_i)] = rec
            if i % 5 == 0 or i == len(need_pairs):
                print(f"  pair {i}/{len(need_pairs)} wall={time.time()-t0:.0f}s", flush=True)
        agent.close()

    forks_path = data / "forks.jsonl"
    done_forks = {}
    if args.resume and forks_path.exists():
        for rec in load_jsonl(forks_path):
            done_forks[(rec["task_id"], int(rec["pair_i"]))] = rec

    need_forks = []
    for tid in split["ids"]:
        for pair_i in pair_seeds:
            if (tid, pair_i) in done_forks:
                continue
            pair = done_pairs[(tid, pair_i)]
            t_star = first_full_action_diverge(pair["base"], pair["uce"])
            if t_star is None:
                done_forks[(tid, pair_i)] = {
                    "task_id": tid,
                    "pair_i": pair_i,
                    "legal": False,
                    "reason": "no_full_action_diverge",
                }
                _append(forks_path, done_forks[(tid, pair_i)])
                continue
            need_forks.append((tid, pair_i, t_star, pair))

    if need_forks:
        print(f"continue {len(need_forks)} legal forks (vLLM group-batched)", flush=True)
        agent = build_fast_agent(cfg)
        env_factory = make_env_factory(cfg)
        t0 = time.time()
        group_n = int((cfg.get("vllm") or {}).get("fork_group", 6))
        done_n = 0
        for start in range(0, len(need_forks), group_n):
            chunk = need_forks[start : start + group_n]
            slots: list[dict[str, Any]] = []
            meta: dict[tuple, dict[str, Any]] = {}
            for tid, pair_i, t_star, pair in chunk:
                task = by_id[tid]
                wf = wf_by_task[tid]
                prefix = prefix_commands(pair["base"], t_star)
                a0 = predicted_tool_at(pair["base"], t_star)
                am = predicted_tool_at(pair["uce"], t_star)
                orig_fp = fingerprint_at(pair["base"], t_star)
                orig_fp_u = fingerprint_at(pair["uce"], t_star)
                env = env_factory()
                replay_prefix(env, task, prefix)
                pre_fp = state_fingerprint(env.current_state())
                try:
                    assert_replay_aligned(orig_fp, pre_fp, orig_fp_u or pre_fp)
                except AssertionError as exc:
                    rec = {
                        "task_id": tid,
                        "pair_i": pair_i,
                        "legal": False,
                        "reason": f"replay_mismatch:{exc}",
                        "t_star": t_star,
                    }
                    _append(forks_path, rec)
                    done_forks[(tid, pair_i)] = rec
                    continue
                remain = h_remain(int(cfg["rollout"]["max_steps"]), t_star)
                probe_m = force_only(env_factory, task, prefix, am, orig_fp, pre_fp)
                probe_0 = force_only(env_factory, task, prefix, a0, orig_fp, probe_m["pre_fp"])
                key = (tid, pair_i)
                meta[key] = {
                    "task": task,
                    "wf": wf,
                    "t_star": t_star,
                    "remain": remain,
                    "a_M": am,
                    "a_0": a0,
                    "orig_fp": orig_fp,
                    "probe_m": probe_m,
                    "probe_0": probe_0,
                }
                slots.extend(
                    spawn_slots(
                        env_factory=env_factory,
                        task=task,
                        prefix=prefix,
                        forced=am,
                        probe=probe_m,
                        arm="m",
                        k_n=k_n,
                        tid=tid,
                        pair_i=pair_i,
                        remain=remain,
                        workflow_text=wf["text"],
                        wf_hash=wf["hash"],
                        fork_key=key,
                    )
                )
                slots.extend(
                    spawn_slots(
                        env_factory=env_factory,
                        task=task,
                        prefix=prefix,
                        forced=a0,
                        probe=probe_0,
                        arm="0",
                        k_n=k_n,
                        tid=tid,
                        pair_i=pair_i,
                        remain=remain,
                        workflow_text=wf["text"],
                        wf_hash=wf["hash"],
                        fork_key=key,
                    )
                )
            if slots:
                print(f"  group live={sum(1 for s in slots if s['alive'])} forks={len(meta)}", flush=True)
                step_slots(agent, slots)
            by_key: dict[tuple, list[dict]] = {}
            for s in slots:
                by_key.setdefault(s["fork_key"], []).append(s)
            for key, rows in by_key.items():
                tid, pair_i = key
                info = meta[key]
                outcomes_m = [int(s["reward"]) for s in rows if s["arm"] == "m"]
                outcomes_0 = [int(s["reward"]) for s in rows if s["arm"] == "0"]
                rec = {
                    "task_id": tid,
                    "pair_i": pair_i,
                    "task_type": info["task"].get("task_type"),
                    "legal": True,
                    "t_star": info["t_star"],
                    "H_remain": info["remain"],
                    "a_M": info["a_M"],
                    "a_0": info["a_0"],
                    "workflow_hash": info["wf"]["hash"],
                    "workflow_id": info["wf"]["entry_id"],
                    "replay_fp": info["probe_m"]["pre_fp"],
                    "original_fp": info["orig_fp"],
                    "accepted_m": bool(info["probe_m"]["accepted"]),
                    "accepted_0": bool(info["probe_0"]["accepted"]),
                    "immediate_m": bool(info["probe_m"]["immediate"]),
                    "immediate_0": bool(info["probe_0"]["immediate"]),
                    "outcomes_m": outcomes_m,
                    "outcomes_0": outcomes_0,
                    "q_m": q_hat(outcomes_m),
                    "q_0": q_hat(outcomes_0),
                    "delta_q": delta_q(outcomes_m, outcomes_0),
                }
                _append(forks_path, rec)
                done_forks[key] = rec
                done_n += 1
                print(
                    f"  fork {done_n}/{len(need_forks)} t*={info['t_star']} dQ={rec['delta_q']:.3f} "
                    f"wall={time.time()-t0:.0f}s",
                    flush=True,
                )
        close_fast_agent(agent)

    states = [r for r in done_forks.values() if r.get("legal")]
    dump_json(reports / "fork_states.json", {"n": len(states), "states": states})
    gate = evaluate_gate_s0(states, cfg)
    dump_json(reports / "S0_gate.json", gate)
    lines = [
        "# DVPD Gate S0 summary",
        "",
        "On base/UCE full-action divergence states only. Not all task states.",
        "Not a trained method. Not semantic preservation. Not a SERL/MOPD comparison.",
        "",
        f"- pass: `{gate['pass']}`",
        f"- reasons: `{gate['reasons']}`",
        f"- n_fork_states: `{gate['n_fork_states']}` (need ≥50)",
        f"- n_tasks T_D: `{gate['n_tasks']}` (need ≥30)",
        f"- task_macro_mean: `{gate['task_macro_mean']:.4f}` CI95 [{gate['ci_lo']:.4f}, {gate['ci_hi']:.4f}]",
        f"- signed_large ΔQ≥0.25: `{gate['signed_large_count']}`",
        f"- abs_large (aux): `{gate['abs_large_count']}`",
        f"- negative_large (aux): `{gate['negative_large_count']}`",
        f"- forced accept_rate: `{gate['accept_rate']:.3f}`",
        "",
        gate["claim"],
        "",
        gate["scope"],
        "",
        "Stopped after Gate S0. No training.",
    ]
    (reports / "S0_summary.md").write_text("\n".join(lines) + "\n")
    append_all_results(Path(cfg["paths"]["archive"]), gate)
    print(json.dumps({k: gate[k] for k in ("pass", "reasons", "n_fork_states", "n_tasks", "task_macro_mean", "ci_lo", "signed_large_count", "accept_rate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
