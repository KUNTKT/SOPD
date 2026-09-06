#!/usr/bin/env python3
"""GSM8K SSOPD01 rollouts with HF generate_batch API fix.

Upstream collect_rollouts passes a seed list as a 3rd positional arg, but
HFMathModel.generate_batch(prompts, golds, *, seed=...) rejects that and
every trajectory becomes ERROR. This wrapper uses keyword seed= per batch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SSOPD = Path("/scratch/ktang115/SSOPD")
if str(SSOPD) not in sys.path:
    sys.path.insert(0, str(SSOPD))

from ssopd_math.datasets.math_dataset import problem_rollout_seed
from ssopd_math.experiments.common import (
    collect_metadata,
    dump_json,
    ensure_dir,
    load_config,
    resolve_path,
    write_yaml,
)
from ssopd_math.models.math_model import HFMathModel
from ssopd_math.rollout.generator import _error_record, _record_from_generation, summarize_rollouts
from ssopd_math.rollout.resume import append_rollout, load_existing_rollouts
from ssopd_math.rollout.trajectory import trajectory_id


def _pass_criteria(stats: dict, cfg: dict) -> tuple[str, list[str]]:
    pc = cfg.get("ssopd01", {}).get("pass_criteria", {})
    reasons: list[str] = []
    if stats.get("n_records", 0) < stats.get("n_expected", 0):
        reasons.append("incomplete")
    if stats.get("parse_rate", 0.0) < float(pc.get("min_parse_rate", 0.0)):
        reasons.append("parse_rate")
    if stats.get("mixed_problems", 0) < int(pc.get("min_mixed_problems", 0)):
        reasons.append("mixed_problems")
    sr = stats.get("success_rate", 0.0)
    lo = float(pc.get("success_rate_lo", 0.0))
    hi = float(pc.get("success_rate_hi", 1.0))
    if not (lo <= sr <= hi):
        reasons.append("success_rate")
    if float(stats.get("complete", 0)) < float(pc.get("min_completion_rate", 1.0)):
        reasons.append("completion_rate")
    return ("PASS" if not reasons else "FAIL"), reasons


def collect_rollouts_hf(
    problems: list[dict],
    model: HFMathModel,
    group_size: int,
    random_seed: int,
    model_name: str,
    dataset_version: str,
    out_path: Path,
    resume: bool,
    batch_size: int,
) -> list[dict]:
    existing = load_existing_rollouts(out_path) if resume else {}
    records = list(existing.values())
    seen = set(existing.keys())
    if not (resume and out_path.exists()):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")

    pending: list[tuple[dict, int, str, int]] = []
    for problem in problems:
        pid = str(problem["problem_id"])
        for k in range(group_size):
            tid = trajectory_id(pid, k)
            if tid in seen:
                continue
            seed = problem_rollout_seed(random_seed, pid, k)
            pending.append((problem, k, tid, seed))

    print(f"pending={len(pending)} existing={len(seen)} batch={batch_size}", flush=True)
    i = 0
    while i < len(pending):
        chunk = pending[i : i + batch_size]
        i += len(chunk)
        prompts = [p.get("prompt_user", p["problem"]) for p, _, _, _ in chunk]
        golds = [p["gold_answer"] for p, _, _, _ in chunk]
        # HFMathModel uses one seed per batch; use first trajectory seed.
        seed0 = int(chunk[0][3])
        t0 = time.time()
        try:
            gens = model.generate_batch(prompts, golds, seed=seed0)
            elapsed = time.time() - t0
            per = elapsed / max(len(chunk), 1)
            for (problem, k, tid, seed), gen in zip(chunk, gens):
                rec = _record_from_generation(
                    problem, k, tid, seed, gen, model_name, dataset_version, per
                )
                records.append(rec)
                seen.add(tid)
                append_rollout(out_path, rec)
                if len(records) % 64 == 0 or i >= len(pending):
                    print(
                        f"done {tid} status={rec['verification_status']} "
                        f"progress={len(seen)}/{len(seen)+len(pending)-i}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"batch ERROR at i={i}: {exc}", flush=True)
            for problem, k, tid, seed in chunk:
                rec = _error_record(problem, k, tid, seed, exc, model_name, dataset_version)
                records.append(rec)
                seen.add(tid)
                append_rollout(out_path, rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    reports_dir = ensure_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]))
    rollouts_path = resolve_path(cfg, cfg["paths"]["rollouts"])
    problems_path = resolve_path(cfg, cfg["paths"]["problems"])

    payload = json.loads(problems_path.read_text())
    problems = list(payload["problems"] if isinstance(payload, dict) else payload)
    n_problems = int(cfg["rollout"]["n_problems"])
    problems = problems[:n_problems]
    group_size = int(cfg["rollout"]["group_size"])
    batch_size = int(cfg["rollout"].get("batch_size", 16))
    resume = bool(cfg["rollout"].get("resume", True))
    seed = int(cfg.get("random_seed", 42))
    mc = cfg["model"]

    print(
        f"GSM8K SSOPD01 n={len(problems)} group={group_size} batch={batch_size} resume={resume}",
        flush=True,
    )
    t0 = time.time()
    model = HFMathModel(
        mc["path"],
        device=mc.get("device", "cuda"),
        torch_dtype=mc.get("torch_dtype", "bfloat16"),
        local_files_only=bool(mc.get("local_files_only", True)),
        use_cache=bool(mc.get("use_cache", True)),
        max_new_tokens=int(mc.get("max_new_tokens", 2048)),
        temperature=float(mc.get("temperature", 1.0)),
        top_p=float(mc.get("top_p", 0.95)),
        do_sample=bool(mc.get("do_sample", True)),
    )
    try:
        records = collect_rollouts_hf(
            problems,
            model,
            group_size,
            seed,
            mc["name"],
            cfg.get("dataset_version", "gsm8k_v1"),
            rollouts_path,
            resume,
            batch_size,
        )
    finally:
        model.close()

    stats = summarize_rollouts(records)
    stats["n_expected"] = n_problems * group_size
    stats["complete"] = float(stats.get("n_records", 0) >= stats["n_expected"])
    status, fail_reasons = _pass_criteria(stats, cfg)
    results = {
        "stats": stats,
        "status": status,
        "fail_reasons": fail_reasons,
        "metadata": collect_metadata(cfg, __file__),
        "wall_clock_s": time.time() - t0,
        "note": "HF generate_batch seed= kwarg fix",
    }
    dump_json(reports_dir / "results.json", results)
    write_yaml(reports_dir / "config.yaml", cfg)
    md = resolve_path(cfg, cfg["paths"].get("report_md", reports_dir / "report.md"))
    md.write_text(
        "\n".join(
            [
                "# SSOPD01 GSM8K Rollouts",
                "",
                f"- status: **{status}**",
                f"- records: {stats.get('n_records')} / {stats['n_expected']}",
                f"- parse_rate: {stats.get('parse_rate', 0):.4f}",
                f"- success_rate: {stats.get('success_rate', 0):.4f}",
                f"- mixed_problems: {stats.get('mixed_problems', 0)}",
                f"- wall_s: {results['wall_clock_s']:.1f}",
                "",
            ]
        )
    )
    print(f"SSOPD01 done status={status} records={stats.get('n_records')}", flush=True)


if __name__ == "__main__":
    main()
