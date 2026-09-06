#!/usr/bin/env python3
"""DVPD S0: exclusion union, full-action diverge, remain budget, task-macro Gate."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKFLOW_HEAD = "Suggested workflow"
FORBIDDEN_SPLITS = ("valid_unseen",)
TRAIN_POOLS = ("sft", "audit_select", "audit_confirm")
MANIFEST_ID_KEYS = (
    "excluded_uce_evolution",
    "inner_train",
    "meta_select",
    "probe_confirm",
)
PAIR_SEEDS = (0, 1)


def load_cfg(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    default = Path(__file__).resolve().parents[1] / "configs/experiment_dvpd_s0.yaml"
    return yaml.safe_load(Path(path or default).read_text())


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def ids_hash(ids: list[str] | set[str]) -> str:
    return sha256_text("\n".join(sorted(str(x) for x in ids)))


def refuse_forbidden_split(split: str | None) -> None:
    name = str(split or "")
    if name in FORBIDDEN_SPLITS or name.startswith("valid_unseen"):
        raise SystemExit(f"forbidden split {name!r}: valid_unseen must not be opened")


def refuse_holdout_ids(ids: list[str] | set[str], holdout: set[str]) -> None:
    leak = sorted(set(str(x) for x in ids) & holdout)
    if leak:
        raise SystemExit(f"holdout ids leaked into train split: {leak[:8]}")


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def dump_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _ids_from_jsonl(path: Path) -> list[str]:
    return sorted({str(r["task_id"]) for r in load_jsonl(path) if r.get("task_id")})


def compute_exclusion(cfg: dict[str, Any]) -> dict[str, Any]:
    refuse_forbidden_split(str((cfg.get("env") or {}).get("eval_split") or ""))
    paths = cfg["paths"]
    prov = json.loads(Path(paths["provenance"]).read_text())
    splits = prov.get("splits") or {}
    if "eval_ids" in splits or "eval" in splits:
        pass
    manifest = json.loads(Path(paths["tame_manifest"]).read_text())
    mid = manifest.get("ids") or {}
    sources: dict[str, list[str]] = {
        "evolve_jsonl": _ids_from_jsonl(Path(paths["evolve_jsonl"])),
        "provenance_evolve_ids": sorted(str(x) for x in (splits.get("evolve_ids") or [])),
        "provenance_distill_ids": sorted(str(x) for x in (splits.get("distill_ids") or [])),
    }
    for key in MANIFEST_ID_KEYS:
        sources[f"manifest_{key}"] = sorted(str(x) for x in (mid.get(key) or []))
    union: set[str] = set()
    source_meta = {}
    for name, ids in sources.items():
        union |= set(ids)
        source_meta[name] = {"n": len(ids), "hash": ids_hash(ids)}
    return {
        "sources": source_meta,
        "union_ids": sorted(union),
        "n_excluded_union": len(union),
        "union_hash": ids_hash(union),
    }


def load_train_universe(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    refuse_forbidden_split(str((cfg.get("env") or {}).get("eval_split") or ""))
    cached = json.loads(Path(cfg["paths"]["task_splits_cache"]).read_text())
    universe: list[dict[str, Any]] = []
    for name in TRAIN_POOLS:
        rows = cached.get(name)
        if rows is None:
            raise SystemExit(f"task_splits_cache missing train pool {name}")
        for t in rows:
            split = str(t.get("split") or t.get("dataset_split") or "train")
            refuse_forbidden_split(split)
            rec = dict(t)
            rec["source_pool"] = name
            universe.append(rec)
    return universe


def sample_remaining_tasks(cfg: dict[str, Any]) -> dict[str, Any]:
    excl = compute_exclusion(cfg)
    excluded = set(excl["union_ids"])
    universe = load_train_universe(cfg)
    remaining = [t for t in universe if str(t["task_id"]) not in excluded]
    remaining.sort(key=lambda t: str(t["task_id"]))
    n = int(cfg["sample_n"])
    seed = int(cfg["sample_seed"])
    if len(remaining) < n:
        raise SystemExit(f"not enough remaining train tasks: {len(remaining)} < {n}")
    layers: dict[str, list[dict]] = defaultdict(list)
    for t in remaining:
        layers[str(t.get("task_type") or "unknown")].append(t)
    keys = sorted(layers)
    counts = {k: len(layers[k]) for k in keys}
    total = sum(counts.values())
    alloc = {k: int(n * counts[k] / total) for k in keys}
    while sum(alloc.values()) < n:
        for k in keys:
            if alloc[k] < counts[k]:
                alloc[k] += 1
                if sum(alloc.values()) >= n:
                    break
    while sum(alloc.values()) > n:
        for k in reversed(keys):
            if alloc[k] > 0:
                alloc[k] -= 1
                break
    rng = random.Random(seed)
    picked: list[dict] = []
    layer_actual = {}
    leftover = 0
    for k in keys:
        bucket = list(layers[k])
        bucket.sort(key=lambda x: str(x["task_id"]))
        rng.shuffle(bucket)
        take = min(alloc[k], len(bucket))
        leftover += alloc[k] - take
        layers[k] = bucket[take:]
        picked.extend(bucket[:take])
        layer_actual[k] = {"planned": alloc[k], "actual": take, "available": counts[k]}
    if leftover:
        for k in keys:
            extra = layers[k][:leftover]
            if not extra:
                continue
            picked.extend(extra)
            layers[k] = layers[k][len(extra) :]
            layer_actual[k]["actual"] += len(extra)
            leftover -= len(extra)
            if leftover <= 0:
                break
    picked.sort(key=lambda t: str(t["task_id"]))
    picked = picked[:n]
    slim = [
        {
            "task_id": str(t["task_id"]),
            "task_type": t.get("task_type"),
            "goal": t.get("goal"),
            "game_file": t.get("game_file"),
            "split": t.get("split") or "train",
            "source_pool": t.get("source_pool"),
        }
        for t in picked
    ]
    for row in slim:
        refuse_forbidden_split(row.get("split"))
    return {
        "sample_seed": seed,
        "n": len(slim),
        "pair_seeds": list(cfg.get("pair_seeds") or PAIR_SEEDS),
        "exclusion_sources": excl["sources"],
        "n_excluded_union": excl["n_excluded_union"],
        "union_hash": excl["union_hash"],
        "n_remaining": len(remaining),
        "layer_actual": layer_actual,
        "tasks": slim,
        "ids": [s["task_id"] for s in slim],
    }


def retrieve_readonly(lib: Any, task: dict) -> tuple[str, str | None, float]:
    before_usage = [e.get("usage") for e in lib.entries]
    before_ids = [e.get("id") for e in lib.entries]
    text, eid, score = lib.retrieve(task)
    after_usage = [e.get("usage") for e in lib.entries]
    after_ids = [e.get("id") for e in lib.entries]
    if before_usage != after_usage or before_ids != after_ids:
        raise RuntimeError("UCE retrieve mutated usage or entry order")
    return text, eid, score


def workflow_hash(text: str) -> str:
    return sha256_text(text or "")


def predicted_tool_at(rec: dict, step: int) -> str:
    for dp in rec.get("decision_points") or []:
        if int(dp.get("step_index", -1)) == step:
            return str(dp.get("predicted_tool") or "").strip()
    return ""


def fingerprint_at(rec: dict, step: int) -> str:
    for dp in rec.get("decision_points") or []:
        if int(dp.get("step_index", -1)) == step:
            return str(dp.get("state_fingerprint") or "")
    return ""


def first_full_action_diverge(base_rec: dict, uce_rec: dict) -> int | None:
    nb = len(base_rec.get("decision_points") or [])
    nu = len(uce_rec.get("decision_points") or [])
    n = min(nb, nu)
    for t in range(n):
        a = predicted_tool_at(base_rec, t)
        b = predicted_tool_at(uce_rec, t)
        if not a or not b:
            return None
        if a != b:
            return t
    return None


def prefix_commands(rec: dict, t_star: int) -> list[str]:
    return [predicted_tool_at(rec, t) for t in range(t_star)]


def h_remain(h_max: int, t_star: int) -> int:
    return max(0, int(h_max) - (int(t_star) + 1))


def replicate_terminal(reward: int | float, k: int) -> list[int]:
    return [1 if reward else 0] * int(k)


def suffix_seed(task_id: str, pair_i: int, k: int) -> int:
    raw = f"dvpd-suffix:{task_id}:{int(pair_i)}:{int(k)}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def prompt_has_workflow(text: str, *, workflow_text: str = "", wf_hash: str = "") -> bool:
    blob = str(text or "")
    if WORKFLOW_HEAD in blob:
        return True
    if wf_hash and wf_hash in blob:
        return True
    wt = str(workflow_text or "").strip()
    if wt and wt in blob:
        return True
    return False


def assert_clean_prompt(text: str, *, workflow_text: str = "", wf_hash: str = "") -> None:
    if prompt_has_workflow(text, workflow_text=workflow_text, wf_hash=wf_hash):
        raise AssertionError("continuation prompt contains workflow")


def assert_replay_aligned(original_fp: str, replay_fp: str, other_fp: str) -> None:
    if not original_fp or original_fp != replay_fp:
        raise AssertionError("replay fingerprint != original first-diverge fingerprint")
    if replay_fp != other_fp:
        raise AssertionError("the two arms differ before the forced action")


def q_hat(outcomes: list[int]) -> float:
    if not outcomes:
        return 0.0
    return sum(int(x) for x in outcomes) / len(outcomes)


def delta_q(outcomes_m: list[int], outcomes_0: list[int]) -> float:
    return q_hat(outcomes_m) - q_hat(outcomes_0)


def group_by_task(states: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for s in states:
        by[str(s["task_id"])].append(s)
    return dict(by)


def task_means(states: list[dict]) -> dict[str, float]:
    by = group_by_task(states)
    out = {}
    for tid, rows in by.items():
        out[tid] = sum(float(r["delta_q"]) for r in rows) / len(rows)
    return out


def task_macro_mean(states: list[dict]) -> float:
    means = task_means(states)
    if not means:
        return 0.0
    return sum(means.values()) / len(means)


def bootstrap_task_macro(
    states: list[dict],
    *,
    n_boot: int = 10000,
    seed: int = 7070,
) -> dict[str, float]:
    means = task_means(states)
    tasks = sorted(means)
    if not tasks:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_tasks": 0}
    rng = random.Random(int(seed))
    vals = [means[t] for t in tasks]
    boots = []
    n = len(vals)
    for _ in range(int(n_boot)):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(len(boots) - 1, int(0.975 * n_boot))]
    return {
        "mean": sum(vals) / n,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_tasks": n,
        "n_boot": int(n_boot),
    }


def force_accept_rate(states: list[dict]) -> float:
    n = 0
    ok = 0
    for s in states:
        n += 2
        ok += int(bool(s.get("accepted_m")))
        ok += int(bool(s.get("accepted_0")))
    return ok / n if n else 0.0


def evaluate_gate_s0(states: list[dict], cfg: dict | None = None) -> dict[str, Any]:
    g = (cfg or {}).get("gate") or {}
    min_states = int(g.get("min_fork_states", 50))
    min_tasks = int(g.get("min_tasks", 30))
    mean_min = float(g.get("mean_min", 0.10))
    large_pos_min = int(g.get("large_pos_min", 30))
    thr = float(g.get("large_delta", 0.25))
    accept_min = float(g.get("accept_min", 0.95))
    n_boot = int((cfg or {}).get("n_boot") or 10000)
    boot_seed = int((cfg or {}).get("bootstrap_seed") or 7070)
    boot = bootstrap_task_macro(states, n_boot=n_boot, seed=boot_seed)
    pos_large = sum(1 for s in states if float(s.get("delta_q") or 0) >= thr)
    abs_large = sum(1 for s in states if abs(float(s.get("delta_q") or 0)) >= thr)
    neg_large = sum(1 for s in states if float(s.get("delta_q") or 0) <= -thr)
    accept = force_accept_rate(states)
    reasons = []
    if len(states) < min_states:
        reasons.append("n_fork_states")
    if boot["n_tasks"] < min_tasks:
        reasons.append("n_tasks")
    if boot["mean"] < mean_min:
        reasons.append("task_macro_mean")
    if boot["ci_lo"] <= 0:
        reasons.append("ci_lo")
    if pos_large < large_pos_min:
        reasons.append("signed_large")
    if accept < accept_min:
        reasons.append("accept_rate")
    passed = not reasons
    claim = (
        "Gate S0 PASS: on base/UCE full-action divergence states, UCE actions "
        "have positive no-workflow continuation value. Not a claim over all "
        "task states. Not a trained method. Not a comparison to SERL/MOPD/π-Distill."
        if passed
        else "Gate S0 FAIL. Stop this method line. Do not retune K or thresholds. Do not train."
    )
    return {
        "pass": passed,
        "reasons": reasons,
        "n_fork_states": len(states),
        "n_tasks": boot["n_tasks"],
        "task_macro_mean": boot["mean"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "signed_large_count": pos_large,
        "abs_large_count": abs_large,
        "negative_large_count": neg_large,
        "accept_rate": accept,
        "claim": claim,
        "scope": (
            "On task-states where base and UCE first diverge in the full command, "
            "the UCE action has positive no-workflow continuation value."
        ),
    }
