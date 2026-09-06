#!/usr/bin/env python3
"""P2: Fit L14 v_cap from ALFWorld audit_select rollouts (archive)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
for p in (str(SCRIPT_DIR), str(SSOPD_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from alfworld_common import dump_json, ensure_alfworld_env, load_jsonl, load_yaml_cfg  # noqa: E402
from ssopd_math.nxt.paired_direction import build_directions_from_records, cosine  # noqa: E402
from ssopd_math.verifier.reward import NEGATIVE, POSITIVE  # noqa: E402
from steerable_alfworld_agent import build_agent  # noqa: E402

LAYER = 14
KEY = "layer_14_task_balanced_paired_v_cap"
KEY_HAT = "layer_14_task_balanced_paired_v_hat"


def pair_stats(records: list[dict]) -> dict:
    by: dict[str, set[bool]] = defaultdict(set)
    for r in records:
        by[str(r["task_id"])].add(bool(r.get("episode_success")))
    n = len(by)
    paired = sum(1 for s in by.values() if True in s and False in s)
    return {
        "n_tasks": n,
        "n_paired": paired,
        "n_trajectories": len(records),
        "success_rate": sum(1 for r in records if r.get("episode_success")) / max(len(records), 1),
    }


def decision_prompts(rec: dict, *, max_n: int = 4) -> list[str]:
    dps = rec.get("decision_points") or []
    prompts: list[str] = []
    for dp in dps:
        p = dp.get("prompt_text")
        if p:
            prompts.append(p)
    if not prompts:
        p = rec.get("prompt_text")
        return [p] if p else []
    if len(prompts) <= max_n:
        return prompts
    # Keep first, last, and evenly spaced middle steps so success/fail diverge.
    idxs = sorted({0, len(prompts) - 1, *[int(i * (len(prompts) - 1) / (max_n - 1)) for i in range(max_n)]})
    return [prompts[i] for i in idxs]


def step_admissible_label(dp: dict) -> str:
    adm = bool(dp.get("parse_ok")) and dp.get("failure_reason") != "not_admissible"
    return POSITIVE if adm else NEGATIVE


def _action_at_step(rec: dict, step: int) -> str:
    dps = rec.get("decision_points") or []
    for dp in dps:
        if int(dp.get("step_index", -1)) == step:
            return str(dp.get("raw_action_text") or dp.get("predicted_tool") or "")
    actions = rec.get("actions") or []
    if 0 <= step < len(actions):
        return str(actions[step])
    return ""


def _prompt_at_step(rec: dict, step: int) -> str | None:
    dps = rec.get("decision_points") or []
    for dp in dps:
        if int(dp.get("step_index", -1)) == step:
            p = dp.get("prompt_text")
            return str(p) if p else None
    if step == 0:
        p = rec.get("prompt_text")
        return str(p) if p else None
    return None


def find_first_diverge_step(s_rec: dict, f_rec: dict) -> int:
    """First step where success/fail trajectories take different actions."""
    s_n = len(s_rec.get("decision_points") or s_rec.get("actions") or [])
    f_n = len(f_rec.get("decision_points") or f_rec.get("actions") or [])
    n = max(s_n, f_n)
    for t in range(n):
        if _action_at_step(s_rec, t) != _action_at_step(f_rec, t):
            return t
    return 0


def find_first_prompt_diverge_step(s_rec: dict, f_rec: dict) -> int | None:
    """First step whose prompt_text differs (history already diverged).

    Action divergence at t* leaves decision_points[t*].prompt_text identical
    (same prefix). The first non-zero contrastive site is usually t*+1.
    """
    s_n = len(s_rec.get("decision_points") or [])
    f_n = len(f_rec.get("decision_points") or [])
    for t in range(max(s_n, f_n)):
        sp = _prompt_at_step(s_rec, t)
        fp = _prompt_at_step(f_rec, t)
        if sp and fp and sp != fp:
            return t
    return None


def pick_success_fail_pair(records: list[dict]) -> tuple[dict, dict] | None:
    succ = sorted(
        [r for r in records if r.get("episode_success")],
        key=lambda r: str(r.get("trajectory_id", "")),
    )
    fail = sorted(
        [r for r in records if not r.get("episode_success")],
        key=lambda r: str(r.get("trajectory_id", "")),
    )
    if not succ or not fail:
        return None
    return succ[0], fail[0]


def _extract_at_prompt(agent, prompt: str, layer: int, site: str) -> tuple[np.ndarray, int]:
    if site == "action_boundary":
        off = agent._action_boundary_offset(prompt)
        ids = agent.tokenizer.encode(prompt, add_special_tokens=False)
        pos = min(off, max(len(ids) - 1, 0))
        return agent.forward_prompt_index_hidden(prompt, layer, pos).numpy(), pos
    return agent.forward_prompt_last_hidden(prompt, layer).numpy(), -1


def extract_late_divergent_records(
    agent,
    records: list[dict],
    layer: int,
    *,
    site: str = "agg",
) -> tuple[list[dict], dict]:
    """One success+fail pair per task at first action-diverge step."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[str(r["task_id"])].append(r)

    out: list[dict] = []
    t_stars: list[int] = []
    t_acts: list[int] = []
    paired_tasks = sorted(by_task.keys())
    for i, tid in enumerate(paired_tasks, 1):
        pair = pick_success_fail_pair(by_task[tid])
        if pair is None:
            continue
        s_rec, f_rec = pair
        t_act = find_first_diverge_step(s_rec, f_rec)
        t_star = find_first_prompt_diverge_step(s_rec, f_rec)
        if t_star is None:
            continue
        s_prompt = _prompt_at_step(s_rec, t_star)
        f_prompt = _prompt_at_step(f_rec, t_star)
        if not s_prompt or not f_prompt or s_prompt == f_prompt:
            continue
        if i % 20 == 0 or i == len(paired_tasks):
            print(
                f"extract late_divergent {i}/{len(paired_tasks)} "
                f"t_act={t_act} t_prompt={t_star}",
                flush=True,
            )
        s_vec, s_pos = _extract_at_prompt(agent, s_prompt, layer, site)
        f_vec, f_pos = _extract_at_prompt(agent, f_prompt, layer, site)
        t_stars.append(t_star)
        t_acts.append(t_act)
        for rec, vec, pos, st in (
            (s_rec, s_vec, s_pos, POSITIVE),
            (f_rec, f_vec, f_pos, NEGATIVE),
        ):
            out.append(
                {
                    "problem_id": tid,
                    "trajectory_id": rec["trajectory_id"],
                    "verification_status": st,
                    "representation_fallback": False,
                    "token_position": pos,
                    "n_sites": 1,
                    "t_star": t_star,
                    "t_act": t_act,
                    "hidden_states": {layer: vec},
                }
            )
    meta = {
        "n_pairs_extracted": len(t_stars),
        "mean_t_star": float(sum(t_stars) / len(t_stars)) if t_stars else 0.0,
        "mean_t_act": float(sum(t_acts) / len(t_acts)) if t_acts else 0.0,
        "frac_t0": float(sum(1 for t in t_stars if t == 0) / len(t_stars)) if t_stars else 0.0,
        "t_star_hist": {str(k): t_stars.count(k) for k in sorted(set(t_stars))},
        "extract_site_note": "first prompt_text diverge (typically t_act+1)",
    }
    return out, meta


def extract_hidden_records(
    agent,
    records: list[dict],
    layer: int,
    *,
    pairing: str = "episode",
    site: str = "agg",
    max_n: int = 4,
) -> list[dict]:
    if pairing == "late_divergent":
        out, _ = extract_late_divergent_records(agent, records, layer, site=site)
        return out

    out: list[dict] = []
    n = len(records)
    for i, rec in enumerate(records, 1):
        if i % 50 == 0 or i == n:
            print(f"extract hidden {i}/{n}", flush=True)
        if pairing == "step_admissible":
            dps = rec.get("decision_points") or []
            for dp in dps:
                p = dp.get("prompt_text")
                if not p:
                    continue
                if site == "action_boundary":
                    off = agent._action_boundary_offset(p)
                    ids = agent.tokenizer.encode(p, add_special_tokens=False)
                    pos = min(off, max(len(ids) - 1, 0))
                    vec = agent.forward_prompt_index_hidden(p, layer, pos).numpy()
                else:
                    vec = agent.forward_prompt_last_hidden(p, layer).numpy()
                out.append(
                    {
                        "problem_id": f"{rec['task_id']}:{dp.get('step_index', 0)}",
                        "trajectory_id": rec["trajectory_id"],
                        "verification_status": step_admissible_label(dp),
                        "representation_fallback": False,
                        "token_position": pos if site == "action_boundary" else -1,
                        "n_sites": 1,
                        "hidden_states": {layer: vec},
                    }
                )
            continue

        prompts = decision_prompts(rec, max_n=max_n)
        if not prompts:
            continue
        if site == "action_boundary":
            vecs = []
            for p in prompts:
                off = agent._action_boundary_offset(p)
                ids = agent.tokenizer.encode(p, add_special_tokens=False)
                pos = min(off, max(len(ids) - 1, 0))
                vecs.append(agent.forward_prompt_index_hidden(p, layer, pos).numpy())
            vec = np.mean(np.stack(vecs, axis=0), axis=0)
        else:
            vecs = [agent.forward_prompt_last_hidden(p, layer).numpy() for p in prompts]
            vec = np.mean(np.stack(vecs, axis=0), axis=0)
        st = POSITIVE if rec.get("episode_success") else NEGATIVE
        out.append(
            {
                "problem_id": rec["task_id"],
                "trajectory_id": rec["trajectory_id"],
                "verification_status": st,
                "representation_fallback": False,
                "token_position": -1,
                "n_sites": len(prompts),
                "hidden_states": {layer: vec},
            }
        )
    return out


def extract_hidden_records_legacy(agent, records: list[dict], layer: int) -> list[dict]:
    return extract_hidden_records(agent, records, layer, pairing="episode", site="agg")


def direction_smoke(agent, vector: np.ndarray, tasks: list[dict], cfg: dict) -> dict:
    from alfworld_common import make_env_factory, trajectory_metrics  # noqa: E402
    from rollout.multistep import collect_episodes  # noqa: E402

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="alfw_dir_smoke_")) / "rollouts.jsonl"
    results = {}
    for alpha in (-1.0, 0.0, 1.0):
        agent.clear_steering()
        if alpha != 0.0:
            agent.set_steering(
                layer=LAYER,
                vector=vector,
                alpha=alpha,
                inject_style="decision_point",
            )
        recs = collect_episodes(
            tasks[: min(20, len(tasks))],
            agent,
            make_env_factory(cfg),
            n_rollouts=1,
            dataset_split="dir_smoke",
            base_seed=999,
            out_path=tmp,
            resume=False,
            max_steps=int(cfg["rollout"]["max_steps"]),
            batch_size=4,
            n_env_workers=2,
        )
        results[str(alpha)] = trajectory_metrics(recs)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_confirm.yaml"),
    )
    ap.add_argument(
        "--rollouts",
        default=None,
    )
    ap.add_argument("--min-paired", type=int, default=50)
    ap.add_argument(
        "--ref-directions",
        default="/home/yiyangba/ssopd_paper_archive/data/ssopd02_qwen3_1_7b_nothink/directions.npz",
    )
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument(
        "--pairing",
        choices=["episode", "step_admissible", "late_divergent"],
        default="episode",
    )
    ap.add_argument("--site", choices=["agg", "action_boundary"], default="agg")
    ap.add_argument("--output-key-suffix", default="")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    rollouts_path = Path(args.rollouts or cfg["paths"]["rollouts_select"])
    out_dir = Path(cfg["paths"]["directions"]).parent
    records = load_jsonl(rollouts_path)
    stats = pair_stats(records)
    print("pair_stats", json.dumps(stats), flush=True)
    if stats["n_paired"] < args.min_paired:
        raise SystemExit(f"paired={stats['n_paired']} < min={args.min_paired}; increase K or tasks")

    by_task: dict[str, set[bool]] = defaultdict(set)
    for r in records:
        by_task[str(r["task_id"])].add(bool(r.get("episode_success")))
    paired_ids = {tid for tid, s in by_task.items() if True in s and False in s}
    subset = [r for r in records if str(r["task_id"]) in paired_ids]

    agent = build_agent(cfg)
    t0 = time.time()
    late_meta: dict = {}
    if args.pairing == "late_divergent":
        hidden_recs, late_meta = extract_late_divergent_records(
            agent, subset, LAYER, site=args.site
        )
    else:
        hidden_recs = extract_hidden_records(
            agent,
            subset if args.pairing == "episode" else records,
            LAYER,
            pairing=args.pairing,
            site=args.site,
        )
    built = build_directions_from_records(
        hidden_recs, layer=LAYER, variant="task_balanced_paired", seed=42
    )
    v_cap = np.asarray(built["v_cap"], dtype=np.float32)
    v_hat = np.asarray(built["v_hat"], dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_key_suffix:
        out_name = f"directions_{args.output_key_suffix}.npz"
        key_cap = f"layer_{LAYER}_task_balanced_paired_v_cap"
        key_hat = f"layer_{LAYER}_task_balanced_paired_v_hat"
    else:
        out_name = "directions.npz"
        key_cap = KEY
        key_hat = KEY_HAT
    np.savez(out_dir / out_name, **{key_cap: v_cap, key_hat: v_hat})

    cos_math = None
    ref_path = Path(args.ref_directions)
    if ref_path.exists():
        ref = np.load(ref_path)
        if KEY in ref.files:
            cos_math = float(cosine(v_cap, np.asarray(ref[KEY], dtype=np.float64)))

    cos_episode = None
    episode_path = Path(cfg["paths"].get("directions_episode") or (out_dir / "directions.npz"))
    if episode_path.exists() and args.pairing != "episode":
        ep = np.load(episode_path)
        if KEY in ep.files:
            cos_episode = float(cosine(v_cap, np.asarray(ep[KEY], dtype=np.float64)))

    dir_smoke = {}
    if not args.skip_smoke:
        from alfworld_common import load_task_pools  # noqa: E402

        limit = int(cfg.get("env", {}).get("limit") or cfg.get("env", {}).get("eval_limit") or 1200)
        tasks = load_task_pools(limit=limit)["pools"]["audit_select"]
        dir_smoke = direction_smoke(agent, v_cap, tasks, cfg)
    agent.close()

    meta = {
        **stats,
        "n_paired_used": int(built["n_problems"]),
        "fallback_count": int(built["fallback_count"]),
        "v_cap_norm": float(np.linalg.norm(v_cap)),
        "cosine_to_math_nothink": cos_math,
        "cosine_to_episode_v": cos_episode,
        "direction_key": key_cap,
        "pairing": args.pairing,
        "site": args.site,
        "output_npz": str(out_dir / out_name),
        "layer": LAYER,
        "extract_wall_s": time.time() - t0,
        "direction_smoke": dir_smoke,
        **late_meta,
    }
    dump_json(out_dir / "fit_meta.json", meta)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    dump_json(reports / "p2_fit_summary.json", meta)
    print("FIT", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
