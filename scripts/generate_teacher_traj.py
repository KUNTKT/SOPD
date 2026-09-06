#!/usr/bin/env python3
"""Generate teacher trajectories (optional steering) with per-batch resume.

Designed for A100-80GB: large generate_batch + checkpoint after every batch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
if str(SSOPD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSOPD_ROOT))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.verifier.reward import POSITIVE, verify_completion


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_per_problem(records: list[dict], problem_ids: set[str]) -> list[dict]:
    out, seen = [], set()
    for r in records:
        pid = str(r["problem_id"])
        if pid not in problem_ids or pid in seen:
            continue
        seen.add(pid)
        out.append(r)
    return out


def sanitize(s: str) -> str:
    return (
        str(s)
        .replace("\u2028", " ")
        .replace("\u2029", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


def rewrite_traj(path: Path, records: list[dict], done: dict[str, dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            pid = str(r["problem_id"])
            if pid in done:
                f.write(json.dumps(done[pid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/scratch/ktang115/models/Qwen3-8B")
    ap.add_argument("--rollouts", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd01_qwen3_1_7b_scale2k/rollouts.jsonl")
    ap.add_argument("--splits", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd02_qwen3_1_7b_scale2k/splits.json")
    ap.add_argument("--split", default="vector_fit")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-steer", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "teacher_trajectories.jsonl"
    ckpt_path = out_dir / "checkpoint.json"
    log_path = out_dir / "teacher_gen.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    splits = json.loads(Path(args.splits).read_text())
    ids = sorted(pid for pid, s in splits["problem_splits"].items() if s == args.split)
    records = one_per_problem(load_jsonl(Path(args.rollouts)), set(ids))
    records = sorted(records, key=lambda r: str(r["problem_id"]))

    done: dict[str, dict] = {}
    if args.resume and ckpt_path.exists():
        done = json.loads(ckpt_path.read_text()).get("per_problem", {})
        log(f"resume checkpoint {len(done)}/{len(records)}")
    elif args.resume and traj_path.exists():
        for r in load_jsonl(traj_path):
            done[str(r["problem_id"])] = r
        log(f"resume jsonl {len(done)}/{len(records)}")

    pending = [r for r in records if str(r["problem_id"]) not in done]
    log(
        f"TEACHER GEN model={args.model_path} split={args.split} n={len(records)} "
        f"pending={len(pending)} batch={args.batch_size} mt={args.max_new_tokens} "
        f"steer={not args.no_steer}"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "teacher_model": args.model_path,
                "student_eval_model": "/scratch/ktang115/models/Qwen3-1.7B",
                "split": args.split,
                "n_problems": len(records),
                "steer": not args.no_steer,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    if not pending:
        log("nothing pending")
        return

    model = HFMathModel(
        args.model_path,
        device="cuda",
        torch_dtype="bfloat16",
        local_files_only=True,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
    )
    seed0 = int(args.seed) + 30_000
    t0 = time.time()
    teacher_name = Path(args.model_path).name

    i = 0
    while i < len(pending):
        chunk = pending[i : i + args.batch_size]
        i += len(chunk)
        prompts = [r.get("prompt_user", r["problem"]) for r in chunk]
        golds = [r["gold_answer"] for r in chunk]
        gens = model.generate_batch(prompts, golds, seed=seed0 + i)
        for rec, gen, gold in zip(chunk, gens, golds):
            ver = verify_completion(gen.text, gold)
            pid = str(rec["problem_id"])
            done[pid] = {
                "problem_id": pid,
                "trajectory_id": f"{pid}::teacher_{teacher_name}_vanilla",
                "split": args.split,
                "role": "strong_vanilla_teacher",
                "teacher_model": args.model_path,
                "problem": sanitize(rec["problem"]),
                "prompt_user": sanitize(rec.get("prompt_user", rec["problem"])),
                "prompt_text": sanitize(gen.prompt_text or ""),
                "completion_text": gen.text or "",
                "gold_answer": str(gold),
                "verification_status": ver["verification_status"],
                "correct": int(ver["verification_status"] == POSITIVE),
                "parse_ok": int(bool(ver.get("parse_ok", False))),
                "extracted_answer": ver.get("extracted_answer"),
                "completion_token_count": len(gen.completion_token_ids),
                "hit_max_tokens": int(len(gen.completion_token_ids) >= args.max_new_tokens),
            }
        rewrite_traj(traj_path, records, done)
        atomic_write_json(ckpt_path, {"per_problem": done, "n": len(done)})
        correct = sum(v["correct"] for v in done.values())
        mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        log(
            f"teacher {len(done)}/{len(records)} acc={correct/max(len(done),1):.4f} "
            f"peak_alloc_gb={mem:.1f}"
        )

    correct = sum(v["correct"] for v in done.values())
    hit = sum(v["hit_max_tokens"] for v in done.values())
    summary = {
        "n": len(done),
        "accuracy": correct / max(len(done), 1),
        "n_correct": correct,
        "parse_rate": sum(v["parse_ok"] for v in done.values()) / max(len(done), 1),
        "hit_max_frac": hit / max(len(done), 1),
        "mean_length": float(np.mean([v["completion_token_count"] for v in done.values()]))
        if done
        else 0.0,
        "wall_s": time.time() - t0,
        "traj_path": str(traj_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    correct_path = out_dir / "teacher_correct.jsonl"
    with correct_path.open("w", encoding="utf-8") as f:
        for r in records:
            row = done.get(str(r["problem_id"]))
            if row and row["correct"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if ckpt_path.exists():
        ckpt_path.unlink()
    log(
        f"TEACHER GEN done n={summary['n']} acc={summary['accuracy']:.4f} "
        f"n_correct={correct} hit_max={summary['hit_max_frac']:.3f} "
        f"wall_h={summary['wall_s']/3600:.2f} wrote {traj_path}"
    )
    model.close()


if __name__ == "__main__":
    main()
