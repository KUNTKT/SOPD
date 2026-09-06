#!/usr/bin/env python3
"""Prepare GSM8K problems.json for SSOPD-style rollouts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def gold_from_gsm8k_answer(answer: str) -> str:
    text = str(answer).strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    # strip common currency/comma
    text = text.replace(",", "").strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="/home/yiyangba/hf_cache/datasets")
    args = ap.parse_args()

    import os
    import sys

    os.environ.setdefault("HF_HOME", "/home/yiyangba/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    sys.path = [p for p in sys.path if "ssopd_math/datasets" not in p.replace("\\", "/")]
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=args.cache_dir)
    idxs = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(idxs)
    idxs = idxs[: int(args.n_problems)]

    problems = []
    for j, idx in enumerate(idxs):
        ex = ds[int(idx)]
        problem = str(ex["question"]).strip()
        gold = gold_from_gsm8k_answer(ex["answer"])
        problems.append(
            {
                "problem_id": f"gsm8k_train_{idx:05d}",
                "index": j,
                "hf_index": int(idx),
                "problem": problem,
                "prompt_user": problem + " " + INSTRUCTION,
                "gold_answer": gold,
                "solution_ref": str(ex["answer"]),
                "dataset_split": "train",
                "source": "openai/gsm8k",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"problems": problems}, indent=2))
    print(f"wrote {out} n={len(problems)} sample_gold={problems[0]['gold_answer']}")


if __name__ == "__main__":
    main()
