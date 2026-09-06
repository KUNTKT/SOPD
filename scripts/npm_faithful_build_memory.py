#!/usr/bin/env python3
"""N0: task-keyed inter/intra pair bank at configured mid layers (NPM-faithful)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, ensure_alfworld_env, load_jsonl, load_yaml_cfg  # noqa: E402
from fit_alfworld_directions import (  # noqa: E402
    _prompt_at_step,
    find_first_prompt_diverge_step,
    pick_success_fail_pair,
)
from npm_faithful_lib import rec_goal  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402


def cfg_layers(cfg: dict) -> tuple[int, ...]:
    raw = cfg.get("steering", {}).get("layers", [13, 14, 15])
    return tuple(int(x) for x in raw)


def _is_admissible(dp: dict) -> bool:
    return bool(dp.get("parse_ok")) and dp.get("failure_reason") != "not_admissible"


def extract_layers(agent, prompt: str, layers: tuple[int, ...]) -> dict[int, np.ndarray]:
    hs = agent.forward_prompt_last_hidden_layers(prompt, list(layers))
    return {int(k): v.numpy().astype(np.float32) for k, v in hs.items()}


def build_inter(agent, records: list[dict], layers: tuple[int, ...]) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[str(r["task_id"])].append(r)
    out: list[dict] = []
    tasks = sorted(by_task)
    for i, tid in enumerate(tasks, 1):
        pair = pick_success_fail_pair(by_task[tid])
        if pair is None:
            continue
        s_rec, f_rec = pair
        t_star = find_first_prompt_diverge_step(s_rec, f_rec)
        if t_star is None:
            continue
        sp = _prompt_at_step(s_rec, t_star)
        fp = _prompt_at_step(f_rec, t_star)
        if not sp or not fp or sp == fp:
            continue
        if i % 20 == 0 or i == len(tasks):
            print(f"inter {i}/{len(tasks)} t={t_star}", flush=True)
        hp = extract_layers(agent, sp, layers)
        hn = extract_layers(agent, fp, layers)
        out.append(
            {
                "kind": "inter",
                "task_id": tid,
                "goal": rec_goal(s_rec) or rec_goal(f_rec),
                "h_pos": hp,
                "h_neg": hn,
            }
        )
    return out


def build_intra(
    agent, records: list[dict], layers: tuple[int, ...], *, max_each: int = 3
) -> list[dict]:
    out: list[dict] = []
    fails = [r for r in records if not r.get("episode_success")]
    for i, rec in enumerate(fails, 1):
        if i % 50 == 0 or i == len(fails):
            print(f"intra {i}/{len(fails)}", flush=True)
        dps = rec.get("decision_points") or []
        pos = [dp for dp in dps if dp.get("prompt_text") and _is_admissible(dp)][:max_each]
        neg = [dp for dp in dps if dp.get("prompt_text") and not _is_admissible(dp)][:max_each]
        if not pos or not neg:
            continue
        acc_p = {li: [] for li in layers}
        acc_n = {li: [] for li in layers}
        for dp in pos:
            h = extract_layers(agent, str(dp["prompt_text"]), layers)
            for li in layers:
                acc_p[li].append(h[li])
        for dp in neg:
            h = extract_layers(agent, str(dp["prompt_text"]), layers)
            for li in layers:
                acc_n[li].append(h[li])
        out.append(
            {
                "kind": "intra",
                "task_id": str(rec["task_id"]),
                "goal": rec_goal(rec),
                "h_pos": {li: np.mean(acc_p[li], axis=0).astype(np.float32) for li in layers},
                "h_neg": {li: np.mean(acc_n[li], axis=0).astype(np.float32) for li in layers},
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_npm_faithful.yaml"),
    )
    args = ap.parse_args()
    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    layers = cfg_layers(cfg)
    records = load_jsonl(Path(cfg["paths"]["rollouts_select"]))
    print(f"loaded n={len(records)} layers={list(layers)}", flush=True)
    agent = build_agent(cfg)
    t0 = time.time()
    inter = build_inter(agent, records, layers)
    intra = build_intra(agent, records, layers)
    agent.close()
    entries = inter + intra
    if not entries:
        raise SystemExit("no pairs")
    payload = {
        "layers": np.asarray(layers, dtype=np.int32),
        "task_ids": np.asarray([e["task_id"] for e in entries], dtype=object),
        "goals": np.asarray([e["goal"] for e in entries], dtype=object),
        "kinds": np.asarray([e["kind"] for e in entries], dtype=object),
    }
    for li in layers:
        payload[f"h_pos_{li}"] = np.stack([e["h_pos"][li] for e in entries]).astype(np.float32)
        payload[f"h_neg_{li}"] = np.stack([e["h_neg"][li] for e in entries]).astype(np.float32)
    out = Path(cfg["paths"]["faithful_memory"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    summary = {
        "n_inter": len(inter),
        "n_intra": len(intra),
        "n_total": len(entries),
        "n_tasks": len({e["task_id"] for e in entries}),
        "layers": list(layers),
        "model": cfg.get("model", {}).get("name"),
        "wall_s": time.time() - t0,
        "memory": str(out),
    }
    dump_json(Path(cfg["paths"]["reports_dir"]) / "n0_memory_summary.json", summary)
    print("N0_DONE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
