#!/usr/bin/env python3
"""A0/A1: BFCL tool-call propensity steering on Qwen3-1.7B (no ALFWorld LoRA)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, load_jsonl, load_yaml_cfg  # noqa: E402
from bfcl_propensity_lib import (  # noqa: E402
    BFCL_SYSTEM,
    build_splits,
    fit_v_tool,
    max_tool_prefix_logp,
    paired_f1_bootstrap,
    score_completion,
    subset_metrics,
    user_prompt,
)
from rollout.resume import append_jsonl  # noqa: E402
from steerable_alfworld_agent import SteerableAlfworldAgent  # noqa: E402


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:+.1f}".replace("+", "p").replace("-", "m")


def build_bfcl_agent(cfg: dict[str, Any]) -> SteerableAlfworldAgent:
    model = cfg["model"]
    rollout = cfg["rollout"]
    chat_kwargs: dict[str, Any] = {}
    if model.get("enable_thinking") is False:
        chat_kwargs["enable_thinking"] = False
    return SteerableAlfworldAgent(
        model["path"],
        system_prompt=BFCL_SYSTEM,
        device=model.get("device", "cuda"),
        torch_dtype=model.get("torch_dtype", "bfloat16"),
        local_files_only=bool(model.get("local_files_only", True)),
        max_new_tokens=int(rollout.get("max_new_tokens", 256)),
        temperature=float(rollout.get("temperature", 1.0)),
        top_p=float(rollout.get("top_p", 1.0)),
        do_sample=bool(rollout.get("do_sample", False)),
        use_cache=bool(model.get("use_cache", True)),
        adapter_path=None,
        chat_template_kwargs=chat_kwargs or None,
    )


def run_tasks(
    *,
    agent: SteerableAlfworldAgent,
    tasks: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_path: Path,
    resume: bool,
    layer: int,
    collect_hidden: bool,
    collect_logp: bool,
    alpha: float,
    vector: np.ndarray | None,
) -> list[dict[str, Any]]:
    expected = len(tasks)
    if resume and out_path.exists():
        recs = load_jsonl(out_path)
        if len(recs) >= expected:
            print(f"  resume skip {out_path.name} n={len(recs)}", flush=True)
            return recs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")
    seed = int(cfg["rollout"]["seed"])
    style = "decision_point"
    agent.clear_steering()
    if vector is not None and abs(alpha) > 1e-12:
        agent.set_steering(
            layer=layer,
            vector=torch.tensor(vector, dtype=torch.float32),
            alpha=alpha,
            inject_style=style,  # type: ignore[arg-type]
        )
    recs: list[dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        obs = user_prompt(task)
        prompt = agent._prompt_text(obs)
        hidden = None
        logp = None
        if collect_hidden:
            hidden = agent.forward_prompt_last_hidden(prompt, layer).numpy()
        if collect_logp:
            logp = max_tool_prefix_logp(agent, prompt)
        gen = agent.generate(obs, seed=seed + i)
        scored = score_completion(task, gen.text)
        rec = {
            **scored,
            "alpha": alpha,
            "prompt_text": prompt,
            "completion": gen.text,
            "token_count": int(gen.completion_token_count),
            "tool_prefix_logp": logp,
        }
        if hidden is not None:
            rec["hidden"] = hidden.astype(np.float32).tolist()
        recs.append(rec)
        append_jsonl(out_path, rec)
        if i % 10 == 0 or i == expected:
            print(
                f"  {out_path.name} {i}/{expected} called={scored['called']} "
                f"should={scored['should_call']}",
                flush=True,
            )
    agent.clear_steering()
    return recs


def strip_hidden(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "hidden"} for r in recs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_bfcl_propensity.yaml"),
    )
    ap.add_argument("--phase", choices=["fit", "eval", "all"], default="all")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    data_dir = Path(cfg["paths"]["data_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    splits = build_splits(cfg)
    dump_json(
        data_dir / "splits.json",
        {
            "select_ids": [t["task_id"] for t in splits["select"]],
            "eval_ids": [t["task_id"] for t in splits["eval"]],
            "split_seed": cfg["bfcl"]["split_seed"],
            "note": "new seed 2026; do not reuse burned exp_bfcl_confirm IDs",
        },
    )
    layer = int(cfg["bfcl"]["layer"])
    parse_min = float(cfg["bfcl"]["parse_ok_min"])
    alphas = [float(a) for a in cfg["bfcl"]["alphas"]]
    min_df1 = float(cfg["bfcl"]["gate_min_delta_f1"])

    agent = build_bfcl_agent(cfg)

    fit_summary: dict[str, Any]
    v_path = data_dir / "v_tool.npy"
    if args.phase in {"fit", "all"}:
        print("A0 select rollout + fit v_tool", flush=True)
        t0 = time.time()
        recs_sel = run_tasks(
            agent=agent,
            tasks=splits["select"],
            cfg=cfg,
            out_path=data_dir / "select_alpha0.jsonl",
            resume=args.resume,
            layer=layer,
            collect_hidden=True,
            collect_logp=True,
            alpha=0.0,
            vector=None,
        )
        metrics_sel = subset_metrics(recs_sel)
        parse_ok_rel = float(metrics_sel["parse_ok_relevance"])
        called = [bool(r["called"]) for r in recs_sel]
        hiddens = [np.array(r["hidden"], dtype=np.float32) for r in recs_sel if "hidden" in r]
        v = fit_v_tool(hiddens, called) if len(hiddens) == len(called) else None
        kill = parse_ok_rel < parse_min or v is None
        fit_summary = {
            "metrics_select": metrics_sel,
            "n_called": int(sum(called)),
            "n_not_called": int(len(called) - sum(called)),
            "parse_ok_relevance": parse_ok_rel,
            "parse_ok_min": parse_min,
            "v_norm": float(np.linalg.norm(v)) if v is not None else 0.0,
            "killed": kill,
            "kill_reason": (
                "parse_ok_relevance<min"
                if parse_ok_rel < parse_min
                else ("empty_called_or_not_called" if v is None else None)
            ),
            "wall_s": time.time() - t0,
        }
        dump_json(reports / "a0_fit.json", fit_summary)
        if v is not None:
            np.save(v_path, v)
        print("A0", json.dumps(fit_summary, indent=2), flush=True)
        if kill:
            payload = {
                "phase": "A0",
                "killed": True,
                "fit": fit_summary,
                "analysis": {
                    "gate_pass": False,
                    "reason": "uninformative_parse_or_empty_contrast",
                    "distill": "skipped",
                },
            }
            dump_json(reports / "results.json", payload)
            agent.close()
            print("A_KILLED", json.dumps(payload["analysis"], indent=2), flush=True)
            return
    else:
        fit_summary = json.loads((reports / "a0_fit.json").read_text())
        if fit_summary.get("killed"):
            print("A0 previously killed; skip eval", flush=True)
            agent.close()
            return

    if args.phase in {"eval", "all"}:
        if not v_path.exists():
            raise SystemExit(f"missing {v_path}")
        v = np.load(v_path).astype(np.float32)
        conditions: dict[str, Any] = {}
        recs_by: dict[str, list[dict]] = {}
        for alpha in alphas:
            key = alpha_tag(alpha)
            print(f"A1 eval {key}", flush=True)
            t0 = time.time()
            recs = run_tasks(
                agent=agent,
                tasks=splits["eval"],
                cfg=cfg,
                out_path=reports / f"{key}.jsonl",
                resume=args.resume,
                layer=layer,
                collect_hidden=False,
                collect_logp=True,
                alpha=alpha,
                vector=v if abs(alpha) > 1e-12 else None,
            )
            metrics = subset_metrics(recs)
            mean_logp = float(
                np.mean([r["tool_prefix_logp"] for r in recs if r.get("tool_prefix_logp") is not None])
            ) if recs else 0.0
            conditions[key] = {
                "metrics": metrics,
                "mean_tool_prefix_logp": mean_logp,
                "wall_s": time.time() - t0,
            }
            recs_by[key] = recs
            print(
                f"  f1={metrics['f1']:.3f} call_rel={metrics['call_rate_relevance']:.3f} "
                f"call_irr={metrics['call_rate_irrelevance']:.3f} "
                f"tool_correct={metrics['tool_correct_relevance']:.3f}",
                flush=True,
            )

        base_key = alpha_tag(0.0)
        base = conditions[base_key]["metrics"]
        best_key = max(
            (k for k in conditions if k != base_key),
            key=lambda k: conditions[k]["metrics"]["f1"],
        )
        delta_f1 = conditions[best_key]["metrics"]["f1"] - base["f1"]
        boot = paired_f1_bootstrap(recs_by[base_key], recs_by[best_key])
        dlogp = (
            conditions[best_key]["mean_tool_prefix_logp"]
            - conditions[base_key]["mean_tool_prefix_logp"]
        )
        gate = {
            "best_key": best_key,
            "delta_f1": delta_f1,
            "bootstrap_f1": boot,
            "delta_tool_correct_relevance": (
                conditions[best_key]["metrics"]["tool_correct_relevance"]
                - base["tool_correct_relevance"]
            ),
            "delta_tool_prefix_logp": dlogp,
            "gate_pass": delta_f1 >= min_df1 and boot["ci_lo"] > 0,
            "min_delta_f1": min_df1,
            "distill": "skip_unless_gate",
        }
        payload = {
            "fit": fit_summary,
            "conditions": conditions,
            "analysis": gate,
        }
        dump_json(reports / "results.json", payload)
        print("A1_DONE", json.dumps(gate, indent=2), flush=True)

    agent.close()


if __name__ == "__main__":
    main()
