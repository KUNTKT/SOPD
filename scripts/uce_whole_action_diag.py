#!/usr/bin/env python3
"""Offline H5 diagnostic: score admissible actions on frozen H4-control prefixes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, load_jsonl, load_yaml_cfg  # noqa: E402
from rollout.resume import append_jsonl  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402
from uce_whole_action import (  # noqa: E402
    extract_command,
    match_admissible,
    score_actions,
    strip_workflow,
    summarize_pair,
)


def _finite(xs: list[float]) -> list[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def _mean(xs: list[float]) -> float | None:
    vals = _finite(xs)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _median(xs: list[float]) -> float | None:
    vals = _finite(xs)
    if not vals:
        return None
    return float(statistics.median(vals))


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for f in flags if f) / len(flags))


def iter_steps(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in recs:
        tid = str(rec.get("task_id") or "")
        task_type = rec.get("task_type")
        for dp in rec.get("decision_points") or []:
            state = dp.get("state") or {}
            actions = list(state.get("valid_tools") or [])
            prefix_uce = str(dp.get("prefix_text") or "")
            if not actions or not prefix_uce:
                continue
            rows.append(
                {
                    "task_id": tid,
                    "task_type": task_type or state.get("task_type"),
                    "step_index": int(dp.get("step_index") or 0),
                    "decision_id": dp.get("decision_id"),
                    "prefix_uce": prefix_uce,
                    "prefix_base": strip_workflow(prefix_uce),
                    "actions": actions,
                    "executed": extract_command(
                        str(dp.get("raw_action_text") or dp.get("predicted_tool") or "")
                    ),
                }
            )
    return rows


def pack_done_key(row: dict[str, Any]) -> str:
    return f"{row['task_id']}:{row['step_index']}:{row.get('decision_id') or ''}"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for rec in load_jsonl(path):
        keys.add(str(rec.get("key") or pack_done_key(rec)))
    return keys


def score_step(agent, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    sc = cfg.get("scoring", {})
    tau = float(sc.get("tau_a", 1.0))
    beta_max = float(sc.get("beta_max", 4.0))
    batch = int(sc.get("batch_size", 8))
    targets = tuple(float(x) for x in sc.get("kl_targets", [0.02, 0.05]))
    actions = list(row["actions"])
    s0, tok0 = score_actions(agent, row["prefix_base"], actions, batch_size=batch)
    sp, tokp = score_actions(agent, row["prefix_uce"], actions, batch_size=batch)
    if any(not math.isfinite(x) for x in s0 + sp):
        raise RuntimeError(f"non-finite score {row['task_id']} step {row['step_index']}")
    executed = match_admissible(row["executed"], actions) or row["executed"]
    summary = summarize_pair(
        actions,
        s0,
        sp,
        executed=executed,
        first_cmd_token_0=tok0,
        first_cmd_token_plus=tokp,
        tau_a=tau,
        beta_max=beta_max,
        kl_targets=targets,
    )
    out = {
        "key": pack_done_key(row),
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "step_index": row["step_index"],
        "decision_id": row.get("decision_id"),
        "n_actions": len(actions),
        "s0": s0,
        "s_plus": sp,
        "first_cmd_token_0": tok0,
        "first_cmd_token_plus": tokp,
        **summary,
    }
    return out


def aggregate(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    sc = cfg.get("scoring", {})
    gate_top1 = float(sc.get("gate_top1_diff", 0.10))
    gate_js = float(sc.get("gate_median_js", 0.01))
    js = [float(r["js"]) for r in rows]
    top1 = [bool(r["whole_action_top1_diff"]) for r in rows]
    tok = [bool(r["first_cmd_token_top1_diff"]) for r in rows]
    missed = [bool(r["token_same_action_diff"]) for r in rows]
    word = [bool(r["first_word_top1_diff"]) for r in rows]
    ranks = [float(r["rank_change"]) for r in rows if r.get("rank_change") is not None]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    early: list[dict[str, Any]] = []
    late: list[dict[str, Any]] = []
    for r in rows:
        by_type[str(r.get("task_type") or "unknown")].append(r)
        if int(r.get("step_index") or 0) <= 2:
            early.append(r)
        else:
            late.append(r)

    def slice_stats(chunk: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(chunk),
            "top1_diff": _rate([bool(r["whole_action_top1_diff"]) for r in chunk]),
            "median_js": _median([float(r["js"]) for r in chunk]),
            "mean_js": _mean([float(r["js"]) for r in chunk]),
            "token_top1_diff": _rate([bool(r["first_cmd_token_top1_diff"]) for r in chunk]),
            "token_same_action_diff": _rate([bool(r["token_same_action_diff"]) for r in chunk]),
        }

    reach: dict[str, list[bool]] = defaultdict(list)
    kmax: list[float] = []
    for r in rows:
        rec_reach = r.get("reach") or {}
        for key, info in rec_reach.items():
            reach[key].append(not bool(info.get("unreachable")))
            if key == "0.05":
                kmax.append(float(info.get("k_max") or 0.0))

    top1_rate = _rate(top1)
    med_js = _median(js)
    signal_pass = bool(top1_rate >= gate_top1 and med_js is not None and med_js > gate_js)
    return {
        "n_states": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "mean_js": _mean(js),
        "median_js": med_js,
        "whole_action_top1_diff": top1_rate,
        "first_cmd_token_top1_diff": _rate(tok),
        "first_word_top1_diff": _rate(word),
        "token_same_action_diff": _rate(missed),
        "mean_rank_change": _mean(ranks),
        "median_rank_change": _median(ranks),
        "reach_frac": {k: _rate(v) for k, v in sorted(reach.items())},
        "median_kmax": _median(kmax),
        "early_step_0_2": slice_stats(early),
        "later_steps": slice_stats(late),
        "by_task_type": {k: slice_stats(v) for k, v in sorted(by_type.items())},
        "gate": {
            "top1_threshold": gate_top1,
            "median_js_threshold": gate_js,
            "top1_pass": bool(top1_rate >= gate_top1),
            "js_pass": bool(med_js is not None and med_js > gate_js),
            "signal_pass": signal_pass,
        },
        "verdict": "PASS" if signal_pass else "FAIL",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_uce_whole_action.yaml"),
    )
    ap.add_argument("--limit-steps", type=int, default=None)
    ap.add_argument("--math-only", action="store_true")
    args = ap.parse_args()

    if args.math_only:
        from test_uce_whole_action import main as math_main

        math_main()
        return

    cfg = load_yaml_cfg(Path(args.config))
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    src = Path(cfg["paths"]["h4_control"])
    recs = load_jsonl(src)
    steps = iter_steps(recs)
    if args.limit_steps is not None:
        steps = steps[: int(args.limit_steps)]
    out_steps = reports / "H5_offline_steps.jsonl"
    done = load_done(out_steps)
    pending = [s for s in steps if pack_done_key(s) not in done]
    print(f"steps={len(steps)} pending={len(pending)} done={len(done)}", flush=True)

    if pending:
        agent = build_agent(cfg)
        agent.clear_steering()
        t0 = time.time()
        if not out_steps.exists():
            out_steps.write_text("")
        for i, row in enumerate(pending, 1):
            rec = score_step(agent, row, cfg)
            append_jsonl(out_steps, rec)
            if i % 20 == 0 or i == len(pending):
                print(
                    f"  {i}/{len(pending)} {row['task_id']}#{row['step_index']} "
                    f"js={rec['js']:.4f} top1_diff={rec['whole_action_top1_diff']}",
                    flush=True,
                )
        print(f"scored {len(pending)} in {time.time() - t0:.1f}s", flush=True)
        del agent

    scored = [r for r in load_jsonl(out_steps) if pack_done_key(r) in {pack_done_key(s) for s in steps}]
    summary = aggregate(scored, cfg)
    summary["source"] = str(src)
    summary["n_source_trajectories"] = len(recs)
    dump_json(reports / "H5_offline_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_states", "whole_action_top1_diff", "median_js", "verdict", "gate")}, indent=2))
    print(f"wrote {reports / 'H5_offline_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
