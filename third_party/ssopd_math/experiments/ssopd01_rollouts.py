#!/usr/bin/env python3
"""SSOPD01: on-policy rollout collection for MATH."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from ssopd_math.datasets.math_dataset import load_math_problems
from ssopd_math.experiments.common import (
    collect_metadata,
    dump_json,
    ensure_dir,
    load_config,
    resolve_path,
    write_yaml,
)
from ssopd_math.rollout.generator import collect_rollouts, summarize_rollouts


def load_rollout_model(cfg: dict):
    backend = str(cfg.get("rollout", {}).get("backend", "vllm")).lower()
    mc = cfg["model"]
    if backend == "vllm":
        from ssopd_math.models.vllm_math_model import VLLMMathModel

        vllm_cfg = cfg.get("rollout", {}).get("vllm", {})
        return VLLMMathModel(
            mc["path"],
            max_new_tokens=int(mc.get("max_new_tokens", 1024)),
            temperature=float(mc.get("temperature", 1.0)),
            top_p=float(mc.get("top_p", 0.95)),
            dtype=mc.get("torch_dtype", "bfloat16"),
            gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.85)),
            max_model_len=int(vllm_cfg.get("max_model_len", 4096)),
            seed=int(cfg.get("random_seed", 42)),
            enforce_eager=bool(vllm_cfg.get("enforce_eager", True)),
            local_files_only=bool(mc.get("local_files_only", True)),
        )
    from ssopd_math.models.math_model import HFMathModel

    return HFMathModel(
        mc["path"],
        device=mc.get("device", "cuda"),
        torch_dtype=mc.get("torch_dtype", "bfloat16"),
        local_files_only=bool(mc.get("local_files_only", True)),
        use_cache=bool(mc.get("use_cache", False)),
        max_new_tokens=int(mc.get("max_new_tokens", 1024)),
        temperature=float(mc.get("temperature", 1.0)),
        top_p=float(mc.get("top_p", 0.95)),
        do_sample=bool(mc.get("do_sample", True)),
    )


def _load_problems(cfg: dict) -> list[dict]:
    paths = cfg["paths"]
    problems_path = resolve_path(cfg, paths.get("problems"))
    if problems_path.exists():
        payload = json.loads(problems_path.read_text())
        if isinstance(payload, dict) and "problems" in payload:
            return list(payload["problems"])
        if isinstance(payload, list):
            return payload
    ds = cfg["dataset"]
    ro = cfg["rollout"]
    return load_math_problems(
        source=ds["source"],
        split=ds.get("split_name", "train"),
        cache_dir=ds.get("cache_dir"),
        n_problems=int(ro["n_problems"]),
        seed=int(cfg.get("random_seed", 42)),
        instruction_suffix=ds.get("instruction_suffix", ""),
    )


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    reports_dir = ensure_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]))
    rollouts_path = resolve_path(cfg, cfg["paths"]["rollouts"])
    ensure_dir(rollouts_path.parent)

    ro = cfg["rollout"]
    n_problems = int(ro["n_problems"])
    group_size = int(ro["group_size"])
    batch_size = int(ro.get("batch_size", group_size))
    resume = bool(ro.get("resume", True))
    seed = int(cfg.get("random_seed", 42))

    problems = _load_problems(cfg)[:n_problems]
    print(
        f"SSOPD01 start n_problems={len(problems)} group_size={group_size} "
        f"resume={resume} backend={ro.get('backend', 'vllm')}",
        flush=True,
    )

    t0 = time.time()
    model = load_rollout_model(cfg)
    try:
        records = collect_rollouts(
            problems,
            model,
            group_size,
            seed,
            cfg["model"]["name"],
            cfg.get("dataset_version", "math_lighteval_v1"),
            rollouts_path,
            resume,
            batch_size=batch_size,
            progress_cb=lambda tid, status: print(f"done {tid} status={status}", flush=True),
        )
    finally:
        if hasattr(model, "close"):
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
    }
    dump_json(reports_dir / "results.json", results)
    write_yaml(reports_dir / "config.yaml", cfg)
    md = resolve_path(cfg, cfg["paths"].get("report_md", reports_dir / "report.md"))
    md.write_text(
        "\n".join(
            [
                "# SSOPD01 Rollouts",
                "",
                f"- status: **{status}**",
                f"- records: {stats.get('n_records')} / {stats['n_expected']}",
                f"- parse_rate: {stats.get('parse_rate', 0):.4f}",
                f"- success_rate: {stats.get('success_rate', 0):.4f}",
                f"- mixed_problems: {stats.get('mixed_problems', 0)}",
                "",
            ]
        )
    )
    print(f"SSOPD01 done status={status} records={stats.get('n_records')}", flush=True)


if __name__ == "__main__":
    main()
