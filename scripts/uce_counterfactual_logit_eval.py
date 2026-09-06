#!/usr/bin/env python3
"""H4 teacher eval: UCE counterfactual action-logit steering."""

from __future__ import annotations

import argparse
import json
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
    trajectory_metrics,
)
from rollout.multistep import _decision_point, episode_seed  # noqa: E402
from rollout.resume import append_jsonl  # noqa: E402
from rollout.trajectory import empty_trajectory  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402
from uce_counterfactual_logit import (  # noqa: E402
    aggregate_token_diags,
    extract_workflow_block,
    generate_counterfactual_action,
    paired_bootstrap_strict,
    sha256_file,
    strict_paired_ids,
    workflow_sha256,
)
from uce_eval import slim_metrics  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def retrieve_readonly(lib: UceLibrary, task: dict) -> tuple[str, str | None, float]:
    return lib.retrieve(task)


def b_uce_workflow_map(recs: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in recs:
        tid = str(rec["task_id"])
        dps = rec.get("decision_points") or []
        blob = ""
        if dps:
            blob = extract_workflow_block(str(dps[0].get("prefix_text") or ""))
            if not blob:
                blob = extract_workflow_block(str(dps[0].get("prompt_text") or ""))
        if not blob:
            blob = extract_workflow_block(str(rec.get("prompt_text") or ""))
        out[tid] = blob
    return out


def assert_workflow_parity(
    tasks: list[dict],
    recs_uce: list[dict],
    lib: UceLibrary,
    reports: Path,
) -> dict[str, str]:
    gold = b_uce_workflow_map(recs_uce)
    mapping: dict[str, str] = {}
    mismatches: list[dict[str, str]] = []
    for task, rec in zip(tasks, recs_uce):
        tid = str(task["task_id"])
        if tid != str(rec["task_id"]):
            raise SystemExit(f"task order mismatch at {tid} vs {rec['task_id']}")
        text, eid, score = retrieve_readonly(lib, task)
        h_ret = workflow_sha256(text)
        h_gold = workflow_sha256(gold.get(tid, ""))
        mapping[tid] = h_ret
        if h_ret != h_gold:
            mismatches.append(
                {
                    "task_id": tid,
                    "entry_id": str(eid),
                    "score": str(score),
                    "h_retrieve": h_ret,
                    "h_buce": h_gold,
                }
            )
    payload = {
        "n_tasks": len(tasks),
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:20],
        "task_id_to_workflow_sha256": mapping,
    }
    dump_json(reports / "H4_workflow_hashes.json", payload)
    if mismatches:
        raise SystemExit(f"workflow hash mismatch n={len(mismatches)} see H4_workflow_hashes.json")
    print(f"workflow parity ok n={len(tasks)}", flush=True)
    return mapping


def run_h4_episodes(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    lib: UceLibrary,
    target_kl: float,
    out_path: Path,
    resume: bool,
    dataset_split: str,
    tag: str,
) -> list[dict]:
    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume skip {out_path.name} n={len(recs)}", flush=True)
            return recs

    max_steps = int(cfg["rollout"]["max_steps"])
    base_seed = int(cfg["rollout"]["seed"])
    steer = cfg.get("steering", {})
    env_factory = make_env_factory(cfg)
    agent.clear_steering()
    records: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    for i, task in enumerate(tasks, 1):
        t_ep = time.time()
        env = env_factory()
        tid = str(task["task_id"])
        traj_id = f"{tid}:{tag}:0"
        seed = episode_seed(base_seed, tid, 0)
        wf_text, wf_id, wf_score = retrieve_readonly(lib, task)
        raw_obs = env.reset(task)
        base_obs = raw_obs
        uce_obs = apply_workflow(raw_obs, wf_text)
        actions: list[str] = []
        decisions: list[dict] = []
        token_count = 0
        termination = "max_steps"
        episode_success = False
        first_prompt = ""
        ep_agg: list[dict[str, float]] = []
        invalid = 0

        for step_i in range(max_steps):
            state = env.current_state()
            step_seed = seed + step_i * 17
            gen = generate_counterfactual_action(
                agent,
                base_obs,
                uce_obs,
                target_kl=float(target_kl),
                step_seed=step_seed,
                max_new_tokens=int(cfg["rollout"]["max_new_tokens"]),
                beta_max=float(steer.get("beta_max", 4.0)),
                bisection_steps=int(steer.get("bisection_steps", 12)),
                kl_temperature=float(steer.get("kl_temperature", 1.0)),
                sample_temperature=float(cfg["rollout"].get("temperature", 1.0)),
                top_p=float(cfg["rollout"].get("top_p", 1.0)),
            )
            if step_i == 0:
                first_prompt = gen.prompt_text_uce
            action_text = gen.text
            token_count += len(gen.token_ids)
            actions.append(action_text)
            verdict = env.verify_decision(state, action_text)
            if not verdict.get("parse_ok"):
                invalid += 1
            out = env.step(action_text)
            agg = aggregate_token_diags(gen.token_diags)
            ep_agg.append(agg)
            dp = _decision_point(
                decision_id=f"{traj_id}:{step_i}",
                step_index=step_i,
                prefix_text=uce_obs,
                prompt_text=gen.prompt_text_uce,
                state=state,
                verdict=verdict,
                action_text=action_text,
            )
            dp["target_kl"] = float(target_kl)
            dp["mean_beta"] = agg["mean_beta"]
            dp["first_action_kl_vs_uce"] = agg["first_action_kl_vs_uce"]
            decisions.append(dp)
            if out.terminal:
                termination = str(out.failure_reason or "success")
                episode_success = bool(out.success)
                break
            raw_obs = out.observation
            base_obs = raw_obs
            uce_obs = apply_workflow(raw_obs, wf_text)

        def mean_key(k: str) -> float:
            if not ep_agg:
                return 0.0
            return float(sum(a[k] for a in ep_agg) / len(ep_agg))

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
                "success": episode_success,
                "termination_reason": termination,
                "trajectory_length": len(actions),
                "actions": actions,
                "decision_points": decisions,
                "token_count": token_count,
                "prompt_text": first_prompt,
                "seed": seed,
                "workflow_id": wf_id,
                "workflow_hash": workflow_sha256(wf_text),
                "retrieval_score": wf_score,
                "target_kl": float(target_kl),
                "mean_beta": mean_key("mean_beta"),
                "max_beta": max((a["max_beta"] for a in ep_agg), default=0.0),
                "mean_achieved_kl_vs_uce": mean_key("mean_achieved_kl_vs_uce"),
                "first_action_kl_vs_uce": ep_agg[0]["first_action_kl_vs_uce"] if ep_agg else 0.0,
                "first_action_kl_vs_base": ep_agg[0]["first_action_kl_vs_base"] if ep_agg else 0.0,
                "protocol_token_kl": mean_key("protocol_token_kl"),
                "command_token_kl": mean_key("command_token_kl"),
                "action_top1_flip_rate_vs_uce": mean_key("action_top1_flip_rate_vs_uce"),
                "action_top1_flip_rate_vs_base": mean_key("action_top1_flip_rate_vs_base"),
                "target_unreachable_rate": mean_key("target_unreachable_rate"),
                "mean_uce_entropy": mean_key("mean_uce_entropy"),
                "mean_teacher_entropy": mean_key("mean_teacher_entropy"),
                "invalid_action_count": invalid,
                "episode_latency": time.time() - t_ep,
            }
        )
        records.append(rec)
        append_jsonl(out_path, rec)
        if i % 10 == 0:
            print(
                f"  rec {i}/{len(tasks)} success={episode_success} "
                f"β={rec['mean_beta']:.2f} kl={rec['command_token_kl']:.4f}",
                flush=True,
            )
    return records


def hf_completion_ids(agent, observation: str, seed: int) -> list[int]:
    prompt = agent._prompt_text(observation)
    encoded = agent.tokenizer(prompt, return_tensors="pt")
    device = agent._resolve_device()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    plen = int(encoded["input_ids"].shape[1])
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    gen_kwargs = {
        "max_new_tokens": agent.max_new_tokens,
        "do_sample": agent.do_sample,
        "pad_token_id": agent.tokenizer.pad_token_id,
        "eos_token_id": agent.tokenizer.eos_token_id,
        "use_cache": agent.use_cache,
    }
    if agent.do_sample:
        gen_kwargs["temperature"] = agent.temperature
        gen_kwargs["top_p"] = agent.top_p
    out = agent.model.generate(**encoded, **gen_kwargs)
    ids = out[0, plen:].tolist()
    pad_id = agent.tokenizer.pad_token_id
    if pad_id is not None:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return [int(x) for x in ids]


def run_decoder_parity(
    *,
    agent,
    tasks: list[dict],
    cfg: dict,
    lib: UceLibrary,
    reports: Path,
) -> dict[str, Any]:
    n = int(cfg["steering"].get("parity_n", 10))
    subset = tasks[:n]
    env_factory = make_env_factory(cfg)
    base_seed = int(cfg["rollout"]["seed"])
    max_steps = int(cfg["rollout"]["max_steps"])
    n_steps = 0
    n_match = 0
    n_mismatch = 0
    examples: list[dict[str, Any]] = []
    agent.clear_steering()
    for task in subset:
        env = env_factory()
        tid = str(task["task_id"])
        seed = episode_seed(base_seed, tid, 0)
        wf_text, _, _ = retrieve_readonly(lib, task)
        raw = env.reset(task)
        for step_i in range(max_steps):
            uce_obs = apply_workflow(raw, wf_text)
            base_obs = raw
            step_seed = seed + step_i * 17
            hf_ids = hf_completion_ids(agent, uce_obs, step_seed)
            cf = generate_counterfactual_action(
                agent,
                base_obs,
                uce_obs,
                target_kl=0.0,
                step_seed=step_seed,
                max_new_tokens=int(cfg["rollout"]["max_new_tokens"]),
                sample_temperature=float(cfg["rollout"].get("temperature", 1.0)),
                top_p=float(cfg["rollout"].get("top_p", 1.0)),
            )
            n_steps += 1
            ok = cf.token_ids == hf_ids
            if ok:
                n_match += 1
            else:
                n_mismatch += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "task_id": tid,
                            "step": step_i,
                            "hf": hf_ids[:16],
                            "cf": cf.token_ids[:16],
                            "hf_text": agent.tokenizer.decode(hf_ids, skip_special_tokens=True),
                            "cf_text": cf.text,
                        }
                    )
            # advance env with HF action so both see same future prefixes
            action = agent.tokenizer.decode(hf_ids, skip_special_tokens=True)
            out = env.step(action)
            if out.terminal:
                break
            raw = out.observation
    token_match = n_mismatch == 0 and n_steps > 0
    payload = {
        "n_tasks": n,
        "n_steps": n_steps,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "token_match": token_match,
        "examples": examples,
    }
    dump_json(reports / "H4_decoder_parity.json", payload)
    print(f"decoder parity token_match={token_match} {n_match}/{n_steps}", flush=True)
    return payload


def diagnose_fail(h4: dict, ctrl: dict, recs_h4: list[dict]) -> str:
    dkl = float(h4.get("command_token_kl") or 0.0)
    flip = float(h4.get("action_top1_flip_rate_vs_uce") or 0.0)
    unreach = float(h4.get("target_unreachable_rate") or 0.0)
    inv = float(h4.get("invalid_action_count") or 0.0)
    ent_t = float(h4.get("mean_teacher_entropy") or 0.0)
    ent_u = float(ctrl.get("mean_teacher_entropy") or ctrl.get("mean_uce_entropy") or 0.0)
    ds = float(h4.get("success_rate", 0) - ctrl.get("success_rate", 0))
    if dkl < 0.005 and unreach > 0.8:
        return "steering_insufficient_kl"
    if inv > 5 or (ent_u and ent_t > ent_u * 1.5):
        return "invalid_or_entropy_worse"
    if flip > 0.02 and ds <= 0:
        return "changed_action_no_success_gain"
    if ds <= 0:
        return "amplified_wrong_workflow_or_no_gain"
    return "delta_positive_but_under_gate"


def mean_diag_field(recs: list[dict], key: str) -> float:
    if not recs:
        return 0.0
    return float(sum(float(r.get(key) or 0.0) for r in recs) / len(recs))


def pack_condition(name: str, recs: list[dict], wall: float, target_kl: float) -> dict[str, Any]:
    m = trajectory_metrics(recs)
    info = {
        "name": name,
        "target_kl": target_kl,
        "metrics": slim_metrics(m),
        "success_rate": m["success_rate"],
        "wall_s": wall,
        "mean_beta": mean_diag_field(recs, "mean_beta"),
        "max_beta": max((float(r.get("max_beta") or 0) for r in recs), default=0.0),
        "mean_achieved_kl_vs_uce": mean_diag_field(recs, "mean_achieved_kl_vs_uce"),
        "first_action_kl_vs_uce": mean_diag_field(recs, "first_action_kl_vs_uce"),
        "first_action_kl_vs_base": mean_diag_field(recs, "first_action_kl_vs_base"),
        "protocol_token_kl": mean_diag_field(recs, "protocol_token_kl"),
        "command_token_kl": mean_diag_field(recs, "command_token_kl"),
        "action_top1_flip_rate_vs_uce": mean_diag_field(recs, "action_top1_flip_rate_vs_uce"),
        "action_top1_flip_rate_vs_base": mean_diag_field(recs, "action_top1_flip_rate_vs_base"),
        "target_unreachable_rate": mean_diag_field(recs, "target_unreachable_rate"),
        "mean_uce_entropy": mean_diag_field(recs, "mean_uce_entropy"),
        "mean_teacher_entropy": mean_diag_field(recs, "mean_teacher_entropy"),
        "invalid_action_count": mean_diag_field(recs, "invalid_action_count"),
        "episode_latency": mean_diag_field(recs, "episode_latency"),
    }
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_uce_counterfactual_logit.yaml"),
    )
    ap.add_argument(
        "--phase",
        choices=["math", "parity", "smoke", "eval", "all"],
        default="all",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    if args.phase in {"math", "all"}:
        from test_uce_counterfactual_logit import main as math_main

        math_main()
        if args.phase == "math":
            return

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    split = str(cfg["env"].get("eval_split", "valid_unseen"))
    tasks, task_meta = load_eval_tasks(cfg)
    recs_h0 = load_jsonl(Path(cfg["paths"]["h0_base"]))
    recs_uce = load_jsonl(Path(cfg["paths"]["b_uce"]))
    if len(recs_h0) < len(tasks) or len(recs_uce) < len(tasks):
        raise SystemExit("missing H0 or B_uce")
    recs_h0 = recs_h0[: len(tasks)]
    recs_uce = recs_uce[: len(tasks)]
    strict_paired_ids(recs_uce, [{"task_id": t["task_id"]} for t in tasks])

    lib_path = Path(cfg["paths"]["uce_library_evolved"])
    lib_hash = sha256_file(lib_path)
    lib = UceLibrary.load(lib_path)
    dump_json(
        reports / "H4_library_meta.json",
        {"library_file_sha256": lib_hash, "n_entries": len(lib.entries), "path": str(lib_path)},
    )
    wf_map = assert_workflow_parity(tasks, recs_uce, lib, reports)

    print("loading agent...", flush=True)
    agent = build_agent(cfg)
    agent.clear_steering()
    print("agent ready", flush=True)

    parity = {"token_match": False}
    if args.phase in {"parity", "all"}:
        parity = run_decoder_parity(
            agent=agent, tasks=tasks, cfg=cfg, lib=lib, reports=reports
        )
        if args.phase == "parity":
            agent.close()
            return

    if args.phase in {"smoke", "all"}:
        n_smoke = int(cfg["steering"].get("smoke_n", 10))
        print(f"smoke n={n_smoke} target_kl=0", flush=True)
        t0 = time.time()
        recs_smoke = run_h4_episodes(
            agent=agent,
            tasks=tasks[:n_smoke],
            cfg=cfg,
            lib=lib,
            target_kl=0.0,
            out_path=reports / "H4_smoke.jsonl",
            resume=args.resume,
            dataset_split=split,
            tag="h4smoke",
        )
        print(
            f"  smoke done n={len(recs_smoke)} wall={time.time()-t0:.1f}s "
            f"success={sum(1 for r in recs_smoke if r.get('episode_success'))}",
            flush=True,
        )
        if args.phase == "smoke":
            agent.close()
            return

    use_control = not bool(parity.get("token_match"))
    print(f"causal_control={'H4-control' if use_control else 'B-uce'} token_match={parity.get('token_match')}", flush=True)

    conditions: dict[str, Any] = {
        "H0": {"success_rate": trajectory_metrics(recs_h0)["success_rate"], "metrics": slim_metrics(trajectory_metrics(recs_h0))},
        "B_uce": {"success_rate": trajectory_metrics(recs_uce)["success_rate"], "metrics": slim_metrics(trajectory_metrics(recs_uce))},
    }
    recs_cache: dict[str, list[dict]] = {"H0": recs_h0, "B_uce": recs_uce}

    if use_control:
        print("H4-control target_kl=0", flush=True)
        t0 = time.time()
        recs_ctrl = run_h4_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            lib=lib,
            target_kl=0.0,
            out_path=reports / "H4_control.jsonl",
            resume=args.resume,
            dataset_split=split,
            tag="h4ctrl",
        )
        conditions["H4_control"] = pack_condition("H4_control", recs_ctrl, time.time() - t0, 0.0)
        recs_cache["H4_control"] = recs_ctrl
        print(f"  control success={conditions['H4_control']['success_rate']:.3f}", flush=True)
        recs_primary = recs_ctrl
        primary_name = "H4_control"
    else:
        recs_primary = recs_uce
        primary_name = "B_uce"

    cond_cfgs = list(cfg.get("conditions") or [])
    name_to_file = {
        "uce_cf_logit_delta002": "H4_delta002.jsonl",
        "uce_cf_logit_delta005": "H4_delta005.jsonl",
    }
    for c in cond_cfgs:
        cname = str(c["name"])
        dkl = float(c["target_kl"])
        print(f"{cname} target_kl={dkl}", flush=True)
        t0 = time.time()
        recs = run_h4_episodes(
            agent=agent,
            tasks=tasks,
            cfg=cfg,
            lib=lib,
            target_kl=dkl,
            out_path=reports / name_to_file[cname],
            resume=args.resume,
            dataset_split=split,
            tag=cname,
        )
        conditions[cname] = pack_condition(cname, recs, time.time() - t0, dkl)
        recs_cache[cname] = recs
        print(f"  success={conditions[cname]['success_rate']:.3f}", flush=True)

    agent.close()

    n_boot = int(cfg["steering"].get("n_boot", 10000))
    bseed = int(cfg["steering"].get("bootstrap_seed", 0))
    min_pp = float(cfg["steering"].get("gate_min_delta_pp", 5.0))
    boots: dict[str, Any] = {}
    best_name = None
    best_delta = -1e9
    for c in cond_cfgs:
        cname = str(c["name"])
        recs = recs_cache[cname]
        boot95 = paired_bootstrap_strict(recs_primary, recs, n_boot=n_boot, seed=bseed, ci=0.95)
        boot975 = paired_bootstrap_strict(recs_primary, recs, n_boot=n_boot, seed=bseed, ci=0.975)
        boot_uce = paired_bootstrap_strict(recs_uce, recs, n_boot=n_boot, seed=bseed, ci=0.95)
        boot_h0 = paired_bootstrap_strict(recs_h0, recs, n_boot=n_boot, seed=bseed, ci=0.95)
        boots[cname] = {
            "vs_primary": {"name": primary_name, "ci95": boot95, "ci975_bonferroni": boot975},
            "vs_B_uce": boot_uce,
            "vs_H0": boot_h0,
        }
        if boot95["delta_pp"] > best_delta:
            best_delta = boot95["delta_pp"]
            best_name = cname

    assert best_name is not None
    best_boot = boots[best_name]["vs_primary"]["ci975_bonferroni"]
    gate_pass = best_delta >= min_pp and float(best_boot["ci_lo"]) > 0
    fail_kind = None
    if not gate_pass:
        fail_kind = diagnose_fail(
            conditions[best_name],
            conditions.get(primary_name) or conditions["B_uce"],
            recs_cache[best_name],
        )

    analysis = {
        "primary_control": primary_name,
        "decoder_token_match": bool(parity.get("token_match")),
        "best_condition": best_name,
        "best_delta_pp": best_delta,
        "gate_pass": gate_pass,
        "min_delta_pp": min_pp,
        "fail_kind": fail_kind,
        "distill": "skip_unless_gate",
        "library_file_sha256": lib_hash,
        "n_workflow_hashes": len(wf_map),
    }
    dump_json(reports / "H4_paired_bootstrap.json", boots)
    dump_json(
        reports / "H4_diagnostics.json",
        {
            "parity": parity,
            "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "metrics"} for k, v in conditions.items()},
        },
    )
    payload = {
        "task_meta": task_meta,
        "conditions": conditions,
        "analysis": analysis,
        "bootstrap": boots,
    }
    dump_json(reports / "H4_summary.json", payload)
    dump_json(reports / "results.json", payload)
    print("H4_DONE", json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
