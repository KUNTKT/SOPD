"""On-policy rollout collection without steering."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ssopd_math.datasets.math_dataset import problem_rollout_seed
from ssopd_math.rollout.resume import append_rollout, load_existing_rollouts
from ssopd_math.rollout.trajectory import TrajectoryRecord, trajectory_id
from ssopd_math.verifier.reward import verify_completion


def _record_from_generation(
    problem: dict[str, Any],
    k: int,
    tid: str,
    seed: int,
    gen: Any,
    model_name: str,
    dataset_version: str,
    wall_time_s: float,
) -> dict[str, Any]:
    ver = verify_completion(gen.text, problem["gold_answer"])
    rec = TrajectoryRecord(
        problem_id=str(problem["problem_id"]),
        trajectory_id=tid,
        problem=str(problem["problem"]),
        prompt_text=gen.prompt_text,
        completion_text=gen.text,
        prompt_token_ids=list(gen.prompt_token_ids),
        completion_token_ids=list(gen.completion_token_ids),
        full_token_ids=list(gen.full_token_ids),
        parsed_answer=ver.get("parsed_answer"),
        gold_answer=str(problem["gold_answer"]),
        final_reward=float(ver.get("final_reward", 0.0)),
        verification_status=str(ver["verification_status"]),
        token_count=len(gen.completion_token_ids),
        termination_reason=getattr(gen, "termination_reason", "eos"),
        random_seed=int(seed),
        model_name=model_name,
        dataset_version=dataset_version,
        extra={
            "parse_ok": bool(ver.get("parse_ok", False)),
            "prompt_user": problem.get("prompt_user", problem["problem"]),
            "group_index": int(k),
            "wall_time_s": float(wall_time_s),
        },
    )
    return rec.to_dict()


def _error_record(
    problem: dict[str, Any],
    k: int,
    tid: str,
    seed: int,
    exc: Exception,
    model_name: str,
    dataset_version: str,
) -> dict[str, Any]:
    return {
        "problem_id": str(problem["problem_id"]),
        "trajectory_id": tid,
        "problem": str(problem["problem"]),
        "prompt_text": "",
        "completion_text": "",
        "prompt_token_ids": [],
        "completion_token_ids": [],
        "full_token_ids": [],
        "parsed_answer": None,
        "gold_answer": str(problem["gold_answer"]),
        "final_reward": 0.0,
        "verification_status": "ERROR",
        "token_count": 0,
        "termination_reason": "error",
        "random_seed": int(seed),
        "model_name": model_name,
        "dataset_version": dataset_version,
        "extra": {
            "error": str(exc),
            "prompt_user": problem.get("prompt_user", problem["problem"]),
            "group_index": int(k),
        },
    }


def collect_rollouts(
    problems: list[dict[str, Any]],
    model: Any,
    group_size: int,
    random_seed: int,
    model_name: str,
    dataset_version: str,
    out_path: str | Path | None,
    resume: bool = True,
    progress_cb: Callable[[str, str], None] | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(out_path) if out_path else None
    existing = load_existing_rollouts(path) if path and resume else {}
    records = list(existing.values())
    seen = set(existing.keys())

    if path and not (resume and path.exists()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    pending: list[tuple[dict[str, Any], int, str, int]] = []
    for problem in problems:
        pid = str(problem["problem_id"])
        for k in range(group_size):
            tid = trajectory_id(pid, k)
            if tid in seen:
                continue
            seed = problem_rollout_seed(random_seed, pid, k)
            pending.append((problem, k, tid, seed))

    use_batch = hasattr(model, "generate_batch")
    chunk_size = int(batch_size or group_size or 1)

    i = 0
    while i < len(pending):
        chunk = pending[i : i + chunk_size]
        i += len(chunk)
        if use_batch and len(chunk) > 1:
            t0 = time.time()
            try:
                gens = model.generate_batch(
                    [p.get("prompt_user", p["problem"]) for p, _, _, _ in chunk],
                    [p["gold_answer"] for p, _, _, _ in chunk],
                    [s for _, _, _, s in chunk],
                )
                elapsed = time.time() - t0
                per = elapsed / max(len(chunk), 1)
                for (problem, k, tid, seed), gen in zip(chunk, gens):
                    rec = _record_from_generation(
                        problem, k, tid, seed, gen, model_name, dataset_version, per
                    )
                    records.append(rec)
                    seen.add(tid)
                    if path:
                        append_rollout(path, rec)
                    if progress_cb:
                        progress_cb(tid, rec["verification_status"])
            except Exception as exc:
                for problem, k, tid, seed in chunk:
                    rec = _error_record(problem, k, tid, seed, exc, model_name, dataset_version)
                    records.append(rec)
                    seen.add(tid)
                    if path:
                        append_rollout(path, rec)
                    if progress_cb:
                        progress_cb(tid, rec["verification_status"])
        else:
            for problem, k, tid, seed in chunk:
                t0 = time.time()
                try:
                    gen = model.generate(
                        problem.get("prompt_user", problem["problem"]),
                        problem["gold_answer"],
                        seed=seed,
                    )
                    rec = _record_from_generation(
                        problem,
                        k,
                        tid,
                        seed,
                        gen,
                        model_name,
                        dataset_version,
                        time.time() - t0,
                    )
                except Exception as exc:
                    rec = _error_record(problem, k, tid, seed, exc, model_name, dataset_version)
                records.append(rec)
                seen.add(tid)
                if path:
                    append_rollout(path, rec)
                if progress_cb:
                    progress_cb(tid, rec["verification_status"])

    return records


def summarize_rollouts(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    parse_ok = sum(1 for r in records if r.get("extra", {}).get("parse_ok") or r.get("parse_ok"))
    success = sum(1 for r in records if r.get("verification_status") == "POSITIVE")
    by_problem: dict[str, list[str]] = {}
    for r in records:
        by_problem.setdefault(str(r["problem_id"]), []).append(r.get("verification_status", ""))

    mixed = success_only = fail_only = parse_fail_only = 0
    for statuses in by_problem.values():
        has_pos = "POSITIVE" in statuses
        has_neg = any(s != "POSITIVE" for s in statuses)
        if has_pos and has_neg:
            mixed += 1
        elif has_pos and not has_neg:
            success_only += 1
        elif not has_pos and any(s == "NEGATIVE" for s in statuses):
            fail_only += 1
        elif not has_pos:
            parse_fail_only += 1

    schema_ok = all("trajectory_id" in r and "full_token_ids" in r for r in records[: min(n, 32)])
    return {
        "n_records": n,
        "n_problems": len(by_problem),
        "parse_rate": parse_ok / max(n, 1),
        "success_rate": success / max(n, 1),
        "mixed_problems": mixed,
        "success_only": success_only,
        "fail_only": fail_only,
        "parse_fail_only": parse_fail_only,
        "mean_tokens": sum(r.get("token_count", 0) for r in records) / max(n, 1),
        "schema_ok": bool(schema_ok),
    }
