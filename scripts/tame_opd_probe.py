#!/usr/bin/env python3
"""Teacher eval, shared student collect, and fixed-budget single-refresh OPD probe."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from agent_uce_opd_lib import (  # noqa: E402
    AdapterSwitch,
    build_student_agent,
    build_theta0_agent,
    completion_ids,
    completion_logits,
    mean_step_kl,
    supervised_mask,
)
from alfworld_common import dump_json, ensure_alfworld_env, load_jsonl, trajectory_metrics  # noqa: E402
from tame_opd_lib import (  # noqa: E402
    CANDIDATE_ORDER,
    RUN_SEEDS,
    annotate_episode,
    argmax_with_tiebreak,
    as_uce_library,
    fake_whole_transcript_prepend,
    load_library,
    load_manifest,
    load_tame_cfg,
    plan_probe_budget,
    retrieve_once,
    sub_seed,
    tasks_of,
    teacher_prefix,
    truncate_mask,
    with_rollout_seed,
)
from uce_whole_action import strip_workflow  # noqa: E402
from uce_eval import make_workflow_fn  # noqa: E402


def _rollout(agent, tasks, cfg, out_path: Path, workflow_fn, split: str, resume: bool):
    return run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=None,
        alpha=0.0,
        top_k=8,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_path,
        resume=resume,
        dataset_split=split,
        workflow_fn=workflow_fn,
    )


def smoke_b_uce_logits(cfg, reports: Path) -> dict:
    """Mandatory B-uce prompt/logit equivalence before probe updates."""
    b_uce = Path(cfg["paths"]["b_uce"])
    rec = None
    with b_uce.open() as f:
        for line in f:
            rec = json.loads(line)
            if len(rec.get("decision_points") or []) >= 3:
                break
    if rec is None:
        raise SystemExit("B_uce.jsonl missing a multi-step episode")
    dps = rec["decision_points"]
    first = str(dps[0]["prefix_text"])
    mid = str(dps[len(dps) // 2]["prefix_text"])
    wf = first.split("\n\n", 1)[0]
    rec_first = teacher_prefix(strip_workflow(first), wf)
    rec_mid = teacher_prefix(strip_workflow(mid), wf)
    if rec_first != first or rec_mid != mid:
        raise SystemExit("TAME teacher prefix != B-uce prefix_text")
    fake = fake_whole_transcript_prepend(strip_workflow(first), strip_workflow(mid), wf)
    if fake == mid:
        raise SystemExit("fake whole-transcript prepend unexpectedly matched B-uce")
    agent = build_theta0_agent(cfg)
    tok = agent.tokenizer
    device = agent._resolve_device()

    def ids_of(obs: str):
        return tok(agent._prompt_text(obs), return_tensors="pt").input_ids.to(device)

    i_first = ids_of(first)
    i_mid = ids_of(mid)
    if not torch.equal(i_first, ids_of(rec_first)) or not torch.equal(i_mid, ids_of(rec_mid)):
        agent.close()
        raise SystemExit("B-uce vs TAME input_ids mismatch")
    if torch.equal(i_mid, ids_of(fake)):
        agent.close()
        raise SystemExit("fake builder input_ids matched B-uce mid-step")
    with torch.no_grad():
        z_b = agent.model(input_ids=i_first, use_cache=False).logits[0, 0].float()
        z_t = agent.model(input_ids=ids_of(rec_first), use_cache=False).logits[0, 0].float()
    max_abs = float((z_b - z_t).abs().max())
    agent.close()
    if max_abs > 1e-4:
        raise SystemExit(f"first-step logits differ max_abs={max_abs}")
    payload = {"ok": True, "first_step_logits_max_abs": max_abs, "n_mid_step": len(dps) // 2}
    dump_json(reports / "b_uce_prefix_smoke.json", payload)
    print(json.dumps(payload), flush=True)
    return payload


def _valid_candidates(freeze: dict) -> list[str]:
    valid = freeze.get("valid") or {}
    return [c for c in CANDIDATE_ORDER if valid.get(c, c == "M0")]


def teacher_eval(cfg, manifest, reports: Path, resume: bool) -> dict:
    freeze = json.loads((reports / "population_freeze.json").read_text())
    valid = _valid_candidates(freeze)
    if valid == ["M0"]:
        payload = {
            "skipped_gpu": True,
            "reason": "only_M0_valid_after_mutation_audit",
            "results": {
                "M0": {"mean_success": None, "U_teacher": None, "constraint_ok": True},
                "empty": {"mean_success": None},
            },
            "eligible": ["M0"],
            "teacher_pick": None,
            "identifiable": False,
            "valid": freeze.get("valid"),
        }
        dump_json(reports / "teacher_eval.json", payload)
        print(json.dumps({"teacher": "skipped", "reason": payload["reason"]}), flush=True)
        return payload
    tasks = tasks_of(manifest, "meta_select")
    cand_root = Path(cfg["paths"]["candidates_dir"])
    libs = {cid: as_uce_library(load_library(cand_root / cid / "library.json")) for cid in _valid_candidates(freeze)}
    libs["empty"] = None
    agent = build_theta0_agent(cfg)
    results = {}
    for cid, lib in [("empty", None)] + [(c, libs[c]) for c in CANDIDATE_ORDER if c in libs]:
        per_seed = {}
        for s in RUN_SEEDS:
            ev_seed = sub_seed(s, "evaluation")
            cfg_s = with_rollout_seed(cfg, ev_seed)
            out = reports / "teacher" / f"{cid}_seed{s}.jsonl"
            wf = None if lib is None else make_workflow_fn(lib)
            recs = _rollout(agent, tasks, cfg_s, out, wf, "meta_select", resume)
            per_seed[str(s)] = {
                "success_rate": trajectory_metrics(recs)["success_rate"],
                "n": len(recs),
            }
        mean_j = sum(v["success_rate"] for v in per_seed.values()) / len(per_seed)
        results[cid] = {"per_seed": per_seed, "mean_success": mean_j}
    agent.close()
    j_empty = results["empty"]["mean_success"]
    j_m0 = results["M0"]["mean_success"]
    slack = float(cfg["gate"]["teacher_slack_pp"]) / 100.0
    eligible = []
    for cid in _valid_candidates(freeze):
        j = results[cid]["mean_success"]
        results[cid]["U_teacher"] = j - j_empty
        results[cid]["constraint_ok"] = j >= j_m0 - slack
        if results[cid]["constraint_ok"]:
            eligible.append(cid)
    if not eligible:
        eligible = ["M0"]
        results["M0"]["constraint_ok"] = True
    scores = {c: results[c]["U_teacher"] for c in eligible}
    teacher_pick = argmax_with_tiebreak(scores, eligible)
    if eligible == ["M0"]:
        teacher_pick = None
    payload = {
        "results": results,
        "eligible": eligible,
        "teacher_pick": teacher_pick,
        "identifiable": teacher_pick is not None and eligible != ["M0"],
    }
    dump_json(reports / "teacher_eval.json", payload)
    return payload


def _ensure_init_adapter(cfg, seed: int, adapters: Path) -> Path:
    dest = adapters / f"init_seed{seed}"
    if (dest / "adapter_config.json").exists():
        return dest
    torch.manual_seed(sub_seed(seed, "init"))
    agent = build_student_agent(cfg, distill_path=None, trainable=True)
    dest.mkdir(parents=True, exist_ok=True)
    agent.model.save_pretrained(str(dest))
    agent.tokenizer.save_pretrained(str(dest))
    agent.close()
    return dest


def collect_student(cfg, manifest, reports: Path, adapters: Path, seed: int, resume: bool) -> list[dict]:
    tasks = tasks_of(manifest, "inner_train")
    init = _ensure_init_adapter(cfg, seed, adapters)
    agent = build_student_agent(cfg, distill_path=init, trainable=False)
    cfg_s = with_rollout_seed(cfg, sub_seed(seed, "rollout"))
    out = reports / "student" / f"inner_train_seed{seed}.jsonl"
    recs = _rollout(agent, tasks, cfg_s, out, None, "inner_train", resume)
    agent.close()
    return recs


def annotate_for_library(recs: list[dict], lib, tasks: list[dict]) -> list[dict]:
    by_id = {str(t["task_id"]): t for t in tasks}
    out = []
    for rec in recs:
        wf, eid, score = retrieve_once(lib, by_id[str(rec["task_id"])])
        out.append(annotate_episode(rec, wf, workflow_id=eid, score=score))
    return out


def shared_decision_points(recs: list[dict], tok) -> list[dict]:
    steps = []
    for rec in recs:
        for dp in rec.get("decision_points") or []:
            action = str(dp.get("raw_action_text") or dp.get("action_text") or "")
            base = str(dp.get("prefix_base") or "")
            if not base or not action:
                continue
            cids = completion_ids(tok, action)
            mask = supervised_mask(tok, cids)
            n = sum(1 for b in mask if b)
            if n <= 0:
                continue
            steps.append(
                {
                    "task_id": rec.get("task_id"),
                    "step_index": dp.get("step_index"),
                    "prefix_base": base,
                    "prefix_uce": dp.get("prefix_uce"),
                    "action": action,
                    "n_supervised": n,
                    "cids": cids,
                    "mask": mask,
                }
            )
    steps.sort(key=lambda x: (str(x["task_id"]), int(x["step_index"])))
    return steps


def run_probe_update(cfg, steps: list[dict], init: Path, out_adapter: Path, seed: int) -> dict:
    pcfg = cfg["probe"]
    keep = plan_probe_budget(
        [int(s["n_supervised"]) for s in steps],
        max_steps=int(pcfg["optimizer_steps"]),
        max_tokens=int(pcfg["max_supervised_tokens_per_seed"]),
    )
    rng = random.Random(sub_seed(seed, "shuffle"))
    order = list(range(len(steps)))
    rng.shuffle(order)
    # Re-plan after shuffle: budget applies to shuffled visit order.
    shuffled = [steps[i] for i in order]
    keep = plan_probe_budget(
        [int(s["n_supervised"]) for s in shuffled],
        max_steps=int(pcfg["optimizer_steps"]),
        max_tokens=int(pcfg["max_supervised_tokens_per_seed"]),
    )
    if out_adapter.exists() and (out_adapter / "adapter_config.json").exists():
        return {
            "adapter": str(out_adapter),
            "n_steps_used": sum(1 for k in keep if k > 0),
            "n_tokens_used": sum(keep),
            "resumed": True,
        }
    if out_adapter.exists():
        shutil.rmtree(out_adapter)
    shutil.copytree(init, out_adapter)
    agent = build_student_agent(cfg, distill_path=out_adapter, trainable=True)
    switch = AdapterSwitch(agent.model)
    tok = agent.tokenizer
    device = agent._resolve_device()
    opt = torch.optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad],
        lr=float(pcfg["learning_rate"]),
    )
    t0 = time.time()
    n_steps = 0
    n_tokens = 0
    losses = []
    for step, k in zip(shuffled, keep):
        if k <= 0:
            continue
        mask = truncate_mask(step["mask"], k)
        cids = step["cids"]
        switch.teacher()
        with torch.no_grad():
            t_logits = completion_logits(agent.model, tok, agent._prompt_text(step["prefix_uce"]), cids, device=device)
        switch.student()
        s_logits = completion_logits(agent.model, tok, agent._prompt_text(step["prefix_base"]), cids, device=device)
        loss = mean_step_kl(
            s_logits,
            t_logits,
            cids,
            mask,
            student_temperature=float(pcfg["student_temperature"]),
            teacher_temperature=float(pcfg["teacher_temperature"]),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(agent.model.parameters(), float(pcfg["gradient_clip"]))
        opt.step()
        n_steps += 1
        n_tokens += k
        losses.append(float(loss.detach()))
        del s_logits, t_logits, loss
        if n_steps % 8 == 0:
            print(f"  probe step={n_steps} tokens={n_tokens} loss={losses[-1]:.4f}", flush=True)
    agent.model.save_pretrained(str(out_adapter))
    tok.save_pretrained(str(out_adapter))
    agent.close()
    return {
        "adapter": str(out_adapter),
        "n_steps_used": n_steps,
        "n_tokens_used": n_tokens,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "wall_s": time.time() - t0,
        "resumed": False,
    }


def eval_nowf(cfg, agent, tasks, out_path: Path, seed: int, resume: bool) -> list[dict]:
    cfg_s = with_rollout_seed(cfg, sub_seed(seed, "evaluation"))
    return _rollout(agent, tasks, cfg_s, out_path, None, "meta_select", resume)


def probe_all(cfg, manifest, reports: Path, resume: bool) -> dict:
    freeze = json.loads((reports / "population_freeze.json").read_text())
    teacher = json.loads((reports / "teacher_eval.json").read_text())
    eligible = list(teacher["eligible"])
    if eligible == ["M0"] or teacher.get("teacher_pick") is None and eligible == ["M0"]:
        payload = {
            "skipped": True,
            "reason": teacher.get("reason") or "not_identifiable_only_M0",
            "eligible": eligible,
            "teach_pick": None,
        }
        dump_json(reports / "probe_meta_select.json", payload)
        return payload
    smoke_b_uce_logits(cfg, reports)
    adapters = Path(cfg["paths"]["adapters_dir"])
    tasks_inner = tasks_of(manifest, "inner_train")
    tasks_meta = tasks_of(manifest, "meta_select")
    cand_root = Path(cfg["paths"]["candidates_dir"])
    init_rates = {}
    cand_rates = {c: {} for c in eligible}
    budgets = {}
    for s in RUN_SEEDS:
        init = _ensure_init_adapter(cfg, s, adapters)
        recs = collect_student(cfg, manifest, reports, adapters, s, resume)
        agent_init = build_student_agent(cfg, distill_path=init, trainable=False)
        init_recs = eval_nowf(cfg, agent_init, tasks_meta, reports / "eval_init" / f"meta_seed{s}.jsonl", s, resume)
        init_rates[str(s)] = trajectory_metrics(init_recs)["success_rate"]
        tok = agent_init.tokenizer
        agent_init.close()
        token_used = None
        steps_used = None
        for cid in eligible:
            lib = as_uce_library(load_library(cand_root / cid / "library.json"))
            annotated = annotate_for_library(recs, lib, tasks_inner)
            steps = shared_decision_points(annotated, tok)
            out_ad = adapters / f"probe_{cid}_seed{s}"
            stats = run_probe_update(cfg, steps, init, out_ad, s)
            if token_used is None:
                token_used = stats["n_tokens_used"]
                steps_used = stats["n_steps_used"]
            elif stats["n_tokens_used"] != token_used or stats["n_steps_used"] != steps_used:
                if not stats.get("resumed"):
                    raise RuntimeError(
                        f"probe budget mismatch {cid} seed={s}: "
                        f"{stats['n_tokens_used']}/{stats['n_steps_used']} vs {token_used}/{steps_used}"
                    )
            budgets[f"{cid}:{s}"] = stats
            agent_j = build_student_agent(cfg, distill_path=out_ad, trainable=False)
            recs_j = eval_nowf(cfg, agent_j, tasks_meta, reports / "eval_probe" / f"{cid}_meta_seed{s}.jsonl", s, resume)
            cand_rates[cid][str(s)] = trajectory_metrics(recs_j)["success_rate"]
            agent_j.close()
    j_init = sum(init_rates.values()) / len(init_rates)
    u_teach = {}
    for cid in eligible:
        j = sum(cand_rates[cid].values()) / len(cand_rates[cid])
        u_teach[cid] = j - j_init
    teach_pick = argmax_with_tiebreak(u_teach, eligible)
    if eligible == ["M0"]:
        teach_pick = None
    payload = {
        "init_success": init_rates,
        "init_mean": j_init,
        "candidate_success": cand_rates,
        "U_teach": u_teach,
        "teach_pick": teach_pick,
        "eligible": eligible,
        "budgets": {k: {kk: vv for kk, vv in v.items() if kk != "adapter"} | {"adapter": v.get("adapter")} for k, v in budgets.items()},
    }
    dump_json(reports / "probe_meta_select.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--phase", choices=["teacher", "probe", "all"], default="all")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    cfg = load_tame_cfg(args.config)
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    manifest = load_manifest(reports / "task_manifest.json")
    if args.phase in {"teacher", "all"}:
        print("teacher eval", flush=True)
        teacher_eval(cfg, manifest, reports, args.resume)
    if args.phase in {"probe", "all"}:
        print("probe", flush=True)
        probe_all(cfg, manifest, reports, args.resume)


if __name__ == "__main__":
    main()
