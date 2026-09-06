#!/usr/bin/env python3
"""Generate L14 steered teacher trajectories on vector_fit problems (Q4 MVP)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
if str(SSOPD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSOPD_ROOT))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.verifier.reward import POSITIVE, verify_completion
import ssopd_math.models.math_model as math_model_mod
import ssopd_math.nxt.positions as positions_mod


def patch_disable_thinking() -> None:
    def build_prompt_text_no_think(tokenizer, prompt_user: str, system_prompt: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or positions_mod.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    positions_mod.build_prompt_text = build_prompt_text_no_think
    math_model_mod.build_prompt_text = build_prompt_text_no_think


def resolve_cfg_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="vector_fit")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()
    if args.no_thinking:
        patch_disable_thinking()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["paths"]["root"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "teacher_trajectories.jsonl"
    meta_path = out_dir / "meta.json"
    ckpt_path = out_dir / "checkpoint.json"
    log_path = out_dir / "teacher_gen.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    ss = cfg["ssopd03"]
    layer = int(ss["layer_index"])
    alpha = float(args.alpha if args.alpha is not None else ss["alpha_grid"][0])
    batch_size = int(args.batch_size or ss.get("audit_batch_size") or 48)
    key = str(ss["direction_key"])
    steer_mode = str(ss.get("inject_mode", "contrastive_cap"))

    splits = json.loads(resolve_cfg_path(root, cfg["paths"]["splits"]).read_text())
    ids = sorted(pid for pid, s in splits["problem_splits"].items() if s == args.split)
    records = one_per_problem(load_jsonl(resolve_cfg_path(root, cfg["paths"]["rollouts"])), set(ids))
    records = sorted(records, key=lambda r: str(r["problem_id"]))

    done: dict[str, dict] = {}
    if args.resume and ckpt_path.exists():
        done = json.loads(ckpt_path.read_text()).get("per_problem", {})
        log(f"resume: have {len(done)}/{len(records)}")

    z = np.load(resolve_cfg_path(root, cfg["paths"]["directions"]))
    direction = torch.tensor(z[key], dtype=torch.float32, device=cfg["model"].get("device", "cuda"))
    vnorm = float(direction.norm().item())
    mc = cfg["model"]

    log(
        f"TEACHER GEN start split={args.split} n={len(records)} layer={layer} "
        f"key={key} alpha={alpha} ||v||={vnorm:.4f} batch={batch_size} "
        f"mt={mc.get('max_new_tokens')} no_thinking={args.no_thinking}"
    )

    meta_path.write_text(
        json.dumps(
            {
                "experiment_id": cfg.get("experiment_id"),
                "split": args.split,
                "n_problems": len(records),
                "layer": layer,
                "direction_key": key,
                "direction_norm": vnorm,
                "alpha": alpha,
                "steer_mode": steer_mode,
                "max_new_tokens": int(mc.get("max_new_tokens", 2048)),
                "batch_size": batch_size,
                "seed": int(cfg.get("random_seed", 42)),
                "no_thinking": bool(args.no_thinking),
            },
            indent=2,
        )
    )

    model = HFMathModel(
        mc["path"],
        device=mc.get("device", "cuda"),
        torch_dtype=mc.get("torch_dtype", "bfloat16"),
        local_files_only=True,
        use_cache=True,
        max_new_tokens=int(mc.get("max_new_tokens", 2048)),
        temperature=float(mc.get("temperature", 1.0)),
        top_p=float(mc.get("top_p", 0.95)),
        do_sample=bool(mc.get("do_sample", True)),
    )
    sample = model.build_prompt(records[0].get("prompt_user", records[0]["problem"]))
    empty_think = "<think>\n\n</think>" in sample
    log(f"prompt_check no_thinking={args.no_thinking} empty_think_prefill={empty_think} tail={sample[-120:]!r}")
    if args.no_thinking and not empty_think:
        raise RuntimeError("enable_thinking=False patch failed")
    seed = int(cfg.get("random_seed", 42)) + 20_000
    t0 = time.time()
    n = len(records)

    for start in range(0, n, batch_size):
        chunk = [
            r for r in records[start : start + batch_size] if str(r["problem_id"]) not in done
        ]
        if chunk:
            prompts = [r.get("prompt_user", r["problem"]) for r in chunk]
            golds = [r["gold_answer"] for r in chunk]
            gens = model.generate_batch(
                prompts,
                golds,
                seed=seed + start,
                steer_layer=layer,
                steer_direction=direction,
                steer_alpha=alpha,
                steer_mode=steer_mode,
            )
            for rec, gen, gold in zip(chunk, gens, golds):
                ver = verify_completion(gen.text, gold)
                pid = str(rec["problem_id"])
                row = {
                    "problem_id": pid,
                    "trajectory_id": f"{pid}::teacher_L{layer}_a{alpha}",
                    "split": args.split,
                    "role": "steered_teacher",
                    "layer": layer,
                    "alpha": alpha,
                    "direction_key": key,
                    "problem": rec["problem"],
                    "prompt_user": rec.get("prompt_user", rec["problem"]),
                    "prompt_text": gen.prompt_text,
                    "completion_text": gen.text,
                    "gold_answer": gold,
                    "verification_status": ver["verification_status"],
                    "correct": int(ver["verification_status"] == POSITIVE),
                    "parse_ok": int(bool(ver.get("parse_ok", False))),
                    "extracted_answer": ver.get("extracted_answer"),
                    "completion_token_count": len(gen.completion_token_ids),
                    "hit_max_tokens": int(
                        len(gen.completion_token_ids) >= int(mc.get("max_new_tokens", 2048))
                    ),
                }
                done[pid] = row
            # rewrite jsonl atomically from done (order by records)
            tmp = traj_path.with_suffix(".tmp")
            with tmp.open("w") as f:
                for r in records:
                    pid = str(r["problem_id"])
                    if pid in done:
                        f.write(json.dumps(done[pid], ensure_ascii=False) + "\n")
            tmp.replace(traj_path)
            ckpt_path.write_text(json.dumps({"per_problem": done}, ensure_ascii=False))

        done_n = min(start + batch_size, n)
        correct = sum(v["correct"] for v in done.values())
        mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        log(
            f"teacher {len(done)}/{n} (batch_end={done_n}) "
            f"acc={correct/max(len(done),1):.4f} peak_alloc_gb={mem:.1f}"
        )

    correct = sum(v["correct"] for v in done.values())
    hit = sum(v["hit_max_tokens"] for v in done.values())
    summary = {
        "n": len(done),
        "accuracy": correct / max(len(done), 1),
        "parse_rate": sum(v["parse_ok"] for v in done.values()) / max(len(done), 1),
        "hit_max_frac": hit / max(len(done), 1),
        "mean_length": float(np.mean([v["completion_token_count"] for v in done.values()]))
        if done
        else 0.0,
        "wall_s": time.time() - t0,
        "traj_path": str(traj_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if ckpt_path.exists():
        ckpt_path.unlink()
    log(
        f"TEACHER GEN done n={summary['n']} acc={summary['accuracy']:.4f} "
        f"hit_max={summary['hit_max_frac']:.3f} wall_h={summary['wall_s']/3600:.2f} wrote {traj_path}"
    )
    model.close()


if __name__ == "__main__":
    main()
