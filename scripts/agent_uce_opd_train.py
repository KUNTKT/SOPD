#!/usr/bin/env python3
"""Train UCE-OPD / No-PI OPD / success-only SFT LoRA from collected rollouts."""

from __future__ import annotations

import argparse
import gc
import json
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
    apply_workflow,
    build_student_agent,
    build_theta0_agent,
    completion_ids,
    completion_logits,
    load_opd_cfg,
    mean_step_kl,
    retrieve_readonly,
    strip_workflow,
    supervised_mask,
    task_splits,
)
from alfworld_common import dump_json, ensure_alfworld_env, load_jsonl, trajectory_metrics  # noqa: E402
from rollout.resume import append_jsonl  # noqa: E402
from uce_eval import make_workflow_fn  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def sft_rows_from_success(recs: list[dict], *, uce: bool) -> list[dict]:
    rows: list[dict] = []
    for rec in recs:
        if not rec.get("episode_success"):
            continue
        for dp in rec.get("decision_points") or []:
            prefix = str(dp.get("prefix_uce") if uce else (dp.get("prefix_base") or dp.get("prefix_text") or ""))
            if uce and not dp.get("prefix_uce"):
                prefix = apply_workflow(strip_workflow(str(dp.get("prefix_text") or "")), rec.get("workflow_text"))
            action = str(dp.get("raw_action_text") or "")
            if not prefix or not action:
                continue
            rows.append({"prefix": prefix, "action": action})
    return rows


def opd_episodes(recs: list[dict]) -> list[dict]:
    """All episodes, success and failure."""
    out = []
    for rec in recs:
        dps = []
        for dp in rec.get("decision_points") or []:
            base = str(dp.get("prefix_base") or strip_workflow(dp.get("prefix_text") or ""))
            uce = str(dp.get("prefix_uce") or apply_workflow(base, rec.get("workflow_text")))
            action = str(dp.get("raw_action_text") or "")
            if base and action:
                dps.append({"prefix_base": base, "prefix_uce": uce, "action": action})
        if dps:
            out.append({"task_id": rec.get("task_id"), "steps": dps})
    return out


def collect_current_student(cfg, agent, tasks, lib, out_path: Path, resume: bool) -> list[dict]:
    recs = run_npm_episodes(
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
        dataset_split="audit_select_student",
        workflow_fn=None,
    )
    by_id = {str(t["task_id"]): t for t in tasks}
    out_path.write_text("")
    for rec in recs:
        wf, eid, score = retrieve_readonly(lib, by_id[str(rec["task_id"])])
        rec["workflow_text"] = wf
        rec["workflow_id"] = eid
        rec["retrieval_score"] = score
        for dp in rec.get("decision_points") or []:
            base = strip_workflow(str(dp.get("prefix_text") or ""))
            dp["prefix_base"] = base
            dp["prefix_uce"] = apply_workflow(base, wf)
            dp["workflow_text"] = wf
        append_jsonl(out_path, rec)
    return recs


def train_sft(cfg, recs, adapter_out: Path, *, uce: bool, init_adapter: Path | None = None) -> dict:
    rows = sft_rows_from_success(recs, uce=uce)
    if not rows:
        raise SystemExit("no success rows for SFT")
    agent = build_student_agent(cfg, distill_path=init_adapter, trainable=True)
    tok = agent.tokenizer
    device = agent._resolve_device()
    dcfg = cfg["distill"]
    opt = torch.optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad],
        lr=float(dcfg["learning_rate"]),
    )
    losses: list[float] = []
    n_tokens = 0
    t0 = time.time()
    for ep in range(int(dcfg["epochs"])):
        for row in rows:
            prompt = agent._prompt_text(row["prefix"])
            cids = completion_ids(tok, row["action"])
            mask = supervised_mask(tok, cids)
            logits = completion_logits(agent.model, tok, prompt, cids, device=device)
            logp = torch.nn.functional.log_softmax(
                logits.float() / float(dcfg["student_temperature"]), dim=-1
            )
            picked = []
            for k, keep in enumerate(mask):
                if keep:
                    picked.append(-logp[k, int(cids[k])])
                    n_tokens += 1
            if not picked:
                continue
            loss = torch.stack(picked).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(agent.model.parameters(), float(dcfg["gradient_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
            if len(losses) % 20 == 0:
                print(f"  sft step={len(losses)} loss={losses[-1]:.4f}", flush=True)
    adapter_out.mkdir(parents=True, exist_ok=True)
    agent.model.save_pretrained(str(adapter_out))
    tok.save_pretrained(str(adapter_out))
    agent.close()
    return {
        "n_rows": len(rows),
        "n_steps": len(losses),
        "n_action_tokens": n_tokens,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "wall_s": time.time() - t0,
        "adapter": str(adapter_out),
    }


def train_opd(
    cfg,
    *,
    tasks: list[dict],
    lib: UceLibrary,
    adapter_out: Path,
    reports: Path,
    teacher_uce: bool,
    resume: bool,
    limit_tasks: int | None,
    init_adapter: Path | None = None,
) -> dict:
    if limit_tasks:
        tasks = tasks[: int(limit_tasks)]
    print(
        f"train_opd teacher_uce={teacher_uce} n_tasks={len(tasks)} "
        f"init={init_adapter} out={adapter_out}",
        flush=True,
    )
    agent = build_student_agent(cfg, distill_path=init_adapter, trainable=True)
    switch = AdapterSwitch(agent.model)
    tok = agent.tokenizer
    device = agent._resolve_device()
    dcfg = cfg["distill"]
    opt = torch.optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad],
        lr=float(dcfg["learning_rate"]),
    )
    t0 = time.time()
    epoch_logs = []
    n_opt = 0
    n_tokens = 0
    teacher_grad_hits = 0
    for ep in range(int(dcfg["epochs"])):
        collect_path = reports / f"student_epoch{ep+1}.jsonl"
        existing = load_jsonl(collect_path) if collect_path.exists() else []
        if len(existing) >= len(tasks):
            print(f"  reuse collect {collect_path.name} n={len(existing)}", flush=True)
            recs = existing[: len(tasks)]
        else:
            switch.student()
            agent.model.eval()
            recs = collect_current_student(cfg, agent, tasks, lib, collect_path, resume=False)
        gc.collect()
        torch.cuda.empty_cache()
        agent.model.train()
        switch.student()
        episodes = opd_episodes(recs)
        ep_losses: list[float] = []
        for epi in episodes:
            valid = []
            for step in epi["steps"]:
                cids = completion_ids(tok, step["action"])
                mask = supervised_mask(tok, cids)
                if any(mask):
                    valid.append((step, cids, mask))
            if not valid:
                continue
            n_steps = len(valid)
            opt.zero_grad(set_to_none=True)
            ep_loss_acc = 0.0
            for step, cids, mask in valid:
                student_obs = step["prefix_base"]
                teacher_obs = step["prefix_uce"] if teacher_uce else step["prefix_base"]
                sp = agent._prompt_text(student_obs)
                tp = agent._prompt_text(teacher_obs)
                switch.teacher()
                with torch.no_grad():
                    t_logits = completion_logits(agent.model, tok, tp, cids, device=device)
                switch.student()
                s_logits = completion_logits(agent.model, tok, sp, cids, device=device)
                if any(
                    n.startswith("base_model.model") and p.requires_grad
                    for n, p in agent.model.named_parameters()
                    if "lora_" not in n
                ):
                    teacher_grad_hits += 1
                loss_step = mean_step_kl(
                    s_logits,
                    t_logits,
                    cids,
                    mask,
                    student_temperature=float(dcfg["student_temperature"]),
                    teacher_temperature=float(dcfg["teacher_temperature"]),
                )
                n_tokens += sum(1 for m in mask if m)
                scaled = loss_step / n_steps
                scaled.backward()
                ep_loss_acc += float(loss_step.detach())
                del s_logits, t_logits, loss_step, scaled
            if n_opt == 0:
                gnorm = sum(
                    float(p.grad.detach().abs().sum())
                    for p in agent.model.parameters()
                    if p.grad is not None
                )
                print(f"  first_backward lora_grad_abs={gnorm}", flush=True)
                if gnorm <= 0:
                    raise RuntimeError("student LoRA got zero gradients")
            clip_grad_norm_(agent.model.parameters(), float(dcfg["gradient_clip"]))
            opt.step()
            n_opt += 1
            ep_losses.append(ep_loss_acc / n_steps)
            if n_opt % 10 == 0:
                print(f"  opd ep{ep+1} step={n_opt} loss={ep_losses[-1]:.4f}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        epoch_logs.append(
            {
                "epoch": ep + 1,
                "n_episodes": len(episodes),
                "success_rate": trajectory_metrics(recs)["success_rate"],
                "mean_loss": sum(ep_losses) / max(len(ep_losses), 1),
                "first_loss": ep_losses[0] if ep_losses else None,
                "last_loss": ep_losses[-1] if ep_losses else None,
                "n_opt_steps": len(ep_losses),
            }
        )
        dump_json(reports / f"epoch_{ep+1}_metrics.json", epoch_logs[-1])
        print(json.dumps(epoch_logs[-1]), flush=True)
    adapter_out.mkdir(parents=True, exist_ok=True)
    agent.model.save_pretrained(str(adapter_out))
    tok.save_pretrained(str(adapter_out))
    agent.close()
    return {
        "epochs": epoch_logs,
        "n_opt_steps": n_opt,
        "n_action_tokens": n_tokens,
        "teacher_grad_hits": teacher_grad_hits,
        "wall_s": time.time() - t0,
        "adapter": str(adapter_out),
        "teacher_uce": teacher_uce,
    }


def check_merge(cfg) -> dict:
    """θ0 is defined as merged coldstart. Two merges must match; unmerged Peft is only noted."""
    from copy import deepcopy

    from steerable_alfworld_agent import build_agent

    a_m1 = build_theta0_agent(cfg)
    a_m2 = build_theta0_agent(cfg)
    cfg_peft = deepcopy(cfg)
    a_peft = build_agent(cfg_peft)
    obs = "Task: put a mug on the counter.\n\nAdmissible actions: go to countertop 1"
    p = a_m1._prompt_text(obs)
    ids = a_m1.tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        z1 = a_m1.model(input_ids=ids.input_ids.to(a_m1._resolve_device()), use_cache=False).logits[0, -1].float()
        z2 = a_m2.model(input_ids=ids.input_ids.to(a_m2._resolve_device()), use_cache=False).logits[0, -1].float()
        zp = a_peft.model(input_ids=a_peft.tokenizer(a_peft._prompt_text(obs), return_tensors="pt").input_ids.to(a_peft._resolve_device()), use_cache=False).logits[0, -1].float()
    merge_vs_merge = float((z1 - z2).abs().max())
    merge_vs_peft = float((z1 - zp).abs().max())
    a_m1.close()
    a_m2.close()
    a_peft.close()
    return {
        "theta0_definition": "merged_coldstart",
        "merge_vs_merge_max_abs": merge_vs_merge,
        "merge_vs_unmerged_peft_max_abs": merge_vs_peft,
        "ok": merge_vs_merge < 1e-3,
        "note": "H0/B-uce used unmerged Peft; students train/eval on merged θ0 + new LoRA",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--mode",
        choices=["uce_opd", "no_pi_opd", "vanilla_sft", "uce_sft", "check_merge"],
        required=True,
    )
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--source-jsonl", default=None, help="for SFT: existing rollouts")
    ap.add_argument("--init-adapter", default=None, help="continue from an existing distill LoRA")
    ap.add_argument("--library", default=None, help="teacher UCE library json")
    ap.add_argument("--adapter-out", default=None)
    ap.add_argument("--reports-subdir", default=None)
    ap.add_argument("--tag", default=None, help="override reports/train json tag")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_opd_cfg(args.config)
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    adapters = Path(cfg["paths"]["adapters_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    adapters.mkdir(parents=True, exist_ok=True)

    if args.mode == "check_merge":
        payload = check_merge(cfg)
        dump_json(reports / "merge_check.json", payload)
        print(json.dumps(payload, indent=2))
        if not payload["ok"]:
            raise SystemExit("merge not numerically close; use dual-adapter")
        return

    splits = task_splits(cfg)
    tasks = list(splits["distill"])
    if args.limit_tasks:
        tasks = tasks[: int(args.limit_tasks)]

    if args.mode in {"vanilla_sft", "uce_sft"}:
        src = Path(args.source_jsonl or "")
        if not src.exists():
            raise SystemExit("SFT needs --source-jsonl")
        recs = load_jsonl(src)
        tag = args.tag or ("vanilla_sft" if args.mode == "vanilla_sft" else "uce_sft")
        out = Path(args.adapter_out) if args.adapter_out else adapters / tag
        init = Path(args.init_adapter) if args.init_adapter else None
        stats = train_sft(cfg, recs, out, uce=(args.mode == "uce_sft"), init_adapter=init)
        dump_json(reports / f"train_{tag}.json", stats)
        print(json.dumps(stats, indent=2))
        return

    tag = args.tag or args.mode
    lib_path = Path(args.library) if args.library else Path(cfg["paths"]["uce_library_evolved"])
    lib = UceLibrary.load(lib_path)
    out = Path(args.adapter_out) if args.adapter_out else adapters / tag
    sub = reports / (args.reports_subdir or tag)
    init = Path(args.init_adapter) if args.init_adapter else None
    stats = train_opd(
        cfg,
        tasks=tasks,
        lib=lib,
        adapter_out=out,
        reports=sub,
        teacher_uce=(args.mode == "uce_opd"),
        resume=args.resume,
        limit_tasks=args.limit_tasks,
        init_adapter=init,
    )
    dump_json(reports / f"train_{tag}.json", stats)
    print(json.dumps({k: stats[k] for k in ("n_opt_steps", "n_action_tokens", "wall_s", "adapter")}, indent=2))


if __name__ == "__main__":
    main()
