#!/usr/bin/env python3
"""A0: Build NPM-lite inter/intra contrastive memory for Agent-SSOPD (archive)."""

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
    _extract_at_prompt,
    _prompt_at_step,
    find_first_diverge_step,
    find_first_prompt_diverge_step,
    pick_success_fail_pair,
    step_admissible_label,
)
from ssopd_math.verifier.reward import NEGATIVE, POSITIVE  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402

LAYER = 14


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def _is_admissible(dp: dict) -> bool:
    return bool(dp.get("parse_ok")) and dp.get("failure_reason") != "not_admissible"


def build_inter_entries(agent, records: list[dict], layer: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[str(r["task_id"])].append(r)
    entries: list[dict] = []
    tasks = sorted(by_task.keys())
    for i, tid in enumerate(tasks, 1):
        pair = pick_success_fail_pair(by_task[tid])
        if pair is None:
            continue
        s_rec, f_rec = pair
        t_act = find_first_diverge_step(s_rec, f_rec)
        t_star = find_first_prompt_diverge_step(s_rec, f_rec)
        if t_star is None:
            continue
        sp = _prompt_at_step(s_rec, t_star)
        fp = _prompt_at_step(f_rec, t_star)
        if not sp or not fp or sp == fp:
            continue
        if i % 20 == 0 or i == len(tasks):
            print(f"inter {i}/{len(tasks)} t_prompt={t_star}", flush=True)
        hs, _ = _extract_at_prompt(agent, sp, layer, "agg")
        hf, _ = _extract_at_prompt(agent, fp, layer, "agg")
        delta = _unit(hs.astype(np.float64) - hf.astype(np.float64))
        if float(np.linalg.norm(delta)) < 1e-8:
            continue
        # Query key: mean of the two diverge-site hiddens (task-conditioned).
        q = _unit(0.5 * (hs.astype(np.float64) + hf.astype(np.float64)))
        entries.append(
            {
                "kind": "inter",
                "task_id": tid,
                "t_act": t_act,
                "t_prompt": t_star,
                "query": q,
                "delta": delta,
                "goal": str(s_rec.get("goal") or f_rec.get("goal") or ""),
            }
        )
    return entries


def build_intra_entries(agent, records: list[dict], layer: int, *, max_pairs_per_traj: int = 4) -> list[dict]:
    entries: list[dict] = []
    fails = [r for r in records if not r.get("episode_success")]
    for i, rec in enumerate(fails, 1):
        if i % 50 == 0 or i == len(fails):
            print(f"intra {i}/{len(fails)}", flush=True)
        dps = rec.get("decision_points") or []
        pos = [dp for dp in dps if dp.get("prompt_text") and _is_admissible(dp)]
        neg = [dp for dp in dps if dp.get("prompt_text") and not _is_admissible(dp)]
        if not pos or not neg:
            continue
        # Pair earliest pos with earliest neg (stable), up to max_pairs_per_traj.
        n = min(len(pos), len(neg), max_pairs_per_traj)
        for j in range(n):
            pp = str(pos[j]["prompt_text"])
            nprompt = str(neg[j]["prompt_text"])
            hs, _ = _extract_at_prompt(agent, pp, layer, "agg")
            hf, _ = _extract_at_prompt(agent, nprompt, layer, "agg")
            delta = _unit(hs.astype(np.float64) - hf.astype(np.float64))
            if float(np.linalg.norm(delta)) < 1e-8:
                continue
            q = _unit(0.5 * (hs.astype(np.float64) + hf.astype(np.float64)))
            entries.append(
                {
                    "kind": "intra",
                    "task_id": str(rec["task_id"]),
                    "trajectory_id": str(rec.get("trajectory_id")),
                    "step_pos": int(pos[j].get("step_index", j)),
                    "step_neg": int(neg[j].get("step_index", j)),
                    "query": q,
                    "delta": delta,
                    "goal": str(rec.get("goal") or ""),
                }
            )
    return entries


def fit_error_probe(agent, records: list[dict], layer: int, *, max_steps: int = 4000) -> dict:
    """Linear probe: L14 last-token -> P(step inadmissible). ASA-lite gate.

    Uses sklearn if available; otherwise ridge closed-form on ±1 labels mapped to
    a sigmoid score at inference (coef from least squares).
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    n = 0
    for rec in records:
        for dp in rec.get("decision_points") or []:
            p = dp.get("prompt_text")
            if not p:
                continue
            h, _ = _extract_at_prompt(agent, str(p), layer, "agg")
            xs.append(h.astype(np.float32))
            ys.append(0 if _is_admissible(dp) else 1)
            n += 1
            if n >= max_steps:
                break
        if n >= max_steps:
            break
    X = np.stack(xs, axis=0).astype(np.float64)
    y = np.asarray(ys, dtype=np.float64)
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X, y.astype(np.int64))
        coef = clf.coef_.astype(np.float32).reshape(-1)
        intercept = float(clf.intercept_[0])
        acc = float(clf.score(X, y.astype(np.int64)))
    except Exception:
        # Ridge on centered features; targets in {0,1}
        Xc = X - X.mean(axis=0, keepdims=True)
        lam = 1.0
        xtx = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
        coef = np.linalg.solve(xtx, Xc.T @ (y - y.mean())).astype(np.float32)
        intercept = float(y.mean() - X.mean(axis=0) @ coef)
        pred = ((Xc @ coef + y.mean()) >= 0.5).astype(np.float64)
        acc = float((pred == y).mean())
    return {
        "coef": coef,
        "intercept": intercept,
        "train_acc": acc,
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_alfworld.yaml"),
    )
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    rollouts = Path(cfg["paths"]["rollouts_select"])
    out_dir = Path(cfg["paths"]["memory_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(rollouts)
    print(f"loaded rollouts n={len(records)}", flush=True)
    agent = build_agent(cfg)
    t0 = time.time()
    inter = build_inter_entries(agent, records, LAYER)
    intra = build_intra_entries(agent, records, LAYER)
    print(f"inter={len(inter)} intra={len(intra)}", flush=True)

    probe = {}
    if not args.skip_probe:
        print("fitting error probe...", flush=True)
        probe = fit_error_probe(agent, records, LAYER)
        print(f"probe train_acc={probe['train_acc']:.3f} n={probe['n']}", flush=True)
    agent.close()

    entries = inter + intra
    if not entries:
        raise SystemExit("no memory entries built")
    queries = np.stack([e["query"] for e in entries], axis=0).astype(np.float32)
    deltas = np.stack([e["delta"] for e in entries], axis=0).astype(np.float32)
    kinds = np.asarray([0 if e["kind"] == "inter" else 1 for e in entries], dtype=np.int8)
    meta_rows = [
        {
            "kind": e["kind"],
            "task_id": e["task_id"],
            "goal": e.get("goal", ""),
            **{k: e[k] for k in ("t_act", "t_prompt", "step_pos", "step_neg", "trajectory_id") if k in e},
        }
        for e in entries
    ]
    np.savez_compressed(
        out_dir / "npm_memory.npz",
        queries=queries,
        deltas=deltas,
        kinds=kinds,
        layer=np.asarray([LAYER], dtype=np.int32),
        probe_coef=probe.get("coef", np.zeros(queries.shape[1], dtype=np.float32)),
        probe_intercept=np.asarray([probe.get("intercept", 0.0)], dtype=np.float32),
    )
    dump_json(out_dir / "memory_meta.json", meta_rows)
    summary = {
        "n_inter": len(inter),
        "n_intra": len(intra),
        "n_total": len(entries),
        "layer": LAYER,
        "probe": {k: probe[k] for k in ("train_acc", "n", "pos_rate", "intercept") if k in probe},
        "wall_s": time.time() - t0,
        "memory_npz": str(out_dir / "npm_memory.npz"),
    }
    dump_json(out_dir / "build_summary.json", summary)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    dump_json(reports / "a0_memory_summary.json", summary)
    print("A0_DONE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
