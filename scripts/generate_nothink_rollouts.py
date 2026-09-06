#!/usr/bin/env python3
"""No-think on-policy rollouts for vector_fit (archive wrapper; does not modify SSOPD)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
if str(SSOPD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSOPD_ROOT))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.rollout.generator import collect_rollouts, summarize_rollouts
import ssopd_math.models.math_model as math_model_mod
import ssopd_math.nxt.positions as positions_mod


class BatchSeedCompat:
    """collect_rollouts passes seeds as a 3rd positional list; HFMathModel wants seed=int."""

    def __init__(self, inner: HFMathModel) -> None:
        self.inner = inner

    def generate_batch(self, prompts, golds, seeds=None, **kwargs):
        seed = seeds[0] if isinstance(seeds, (list, tuple)) and seeds else seeds
        return self.inner.generate_batch(prompts, golds, seed=seed, **kwargs)

    def build_prompt(self, prompt_user: str) -> str:
        return self.inner.build_prompt(prompt_user)

    def close(self) -> None:
        self.inner.close()


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_per_problem(records: list[dict], problem_ids: set[str]) -> list[dict]:
    out, seen = [], set()
    for r in records:
        pid = str(r["problem_id"])
        if pid not in problem_ids or pid in seen:
            continue
        seen.add(pid)
        out.append(
            {
                "problem_id": pid,
                "problem": r.get("problem") or r.get("prompt_user"),
                "prompt_user": r.get("prompt_user") or r.get("extra", {}).get("prompt_user") or r["problem"],
                "gold_answer": r["gold_answer"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd02_qwen3_1_7b_scale2k/splits.json")
    ap.add_argument("--src-rollouts", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd01_qwen3_1_7b_scale2k/rollouts.jsonl")
    ap.add_argument("--out", default="/home/yiyangba/ssopd_paper_archive/data/ssopd01_qwen3_1_7b_nothink_vf1200/rollouts.jsonl")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    patch_disable_thinking()
    splits = json.loads(Path(args.splits).read_text())
    vf_ids = {pid for pid, s in splits["problem_splits"].items() if s == "vector_fit"}
    problems = one_per_problem(load_jsonl(Path(args.src_rollouts)), vf_ids)
    problems = sorted(problems, key=lambda p: p["problem_id"])
    if args.max_problems is not None:
        problems = problems[: int(args.max_problems)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"nothink rollouts n_problems={len(problems)} K={args.group_size} "
        f"batch={args.batch_size} out={out}",
        flush=True,
    )

    inner = HFMathModel(
        "/scratch/ktang115/models/Qwen3-1.7B",
        device="cuda",
        torch_dtype="bfloat16",
        local_files_only=True,
        use_cache=True,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
    )
    model = BatchSeedCompat(inner)
    sample = model.build_prompt(problems[0]["prompt_user"])
    if "<think>\n\n</think>" not in sample:
        raise RuntimeError("enable_thinking=False patch failed")
    print(f"prompt_check empty_think_prefill=True tail={sample[-80:]!r}", flush=True)

    t0 = time.time()
    n_done = [0]

    def progress(tid: str, status: str) -> None:
        n_done[0] += 1
        if n_done[0] % 48 == 0 or n_done[0] == 1:
            print(f"rollout {n_done[0]} last={tid} status={status} wall_s={time.time()-t0:.0f}", flush=True)

    records = collect_rollouts(
        problems,
        model,
        group_size=args.group_size,
        random_seed=args.seed,
        model_name="Qwen3-1.7B",
        dataset_version="math_lighteval_v1",
        out_path=out,
        resume=True,
        progress_cb=progress,
        batch_size=args.batch_size,
    )
    summary = summarize_rollouts(records)
    summary["wall_s"] = time.time() - t0
    summary["enable_thinking"] = False
    summary["group_size"] = args.group_size
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps(summary), flush=True)
    model.close()


if __name__ == "__main__":
    main()
