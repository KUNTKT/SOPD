#!/usr/bin/env python3
"""Qwen3 ALFWorld coldstart: SFT only, reusing existing planner expert pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SSOPD_LOGITS = Path("/scratch/ktang115/SSOPD/ssopd_logits")
for p in (str(SCRIPT_DIR), str(SSOPD_LOGITS.parent), str(SSOPD_LOGITS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from alfworld_common import dump_json, ensure_alfworld_env, load_task_pools, load_yaml_cfg  # noqa: E402
from environments.alfworld_adapter import ALFWORLD_FEWSHOT_PROMPT  # noqa: E402
from train.expert_trajectories import load_sft_pairs  # noqa: E402
from train.sft import run_lora_sft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_ssopd05_alfworld_smoke.yaml"),
    )
    ap.add_argument(
        "--expert-pairs",
        default="/scratch/ktang115/SSOPD/ssopd_logits/data/alfworld_coldstart/expert_sft_pairs.jsonl",
    )
    ap.add_argument(
        "--out-adapter",
        default="/home/yiyangba/ssopd_paper_archive/data/ssopd05_alfworld_coldstart/coldstart_lora",
    )
    ap.add_argument("--max-pairs", type=int, default=6000)
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    pools = load_task_pools(limit=int(cfg["env"]["limit"]))["pools"]
    allowed = {str(t["task_id"]) for t in pools["sft"]}
    pairs = load_sft_pairs(Path(args.expert_pairs))
    kept = [row for row in pairs if str(row.get("task_id")) in allowed]
    if args.max_pairs:
        kept = kept[: int(args.max_pairs)]
    print(f"sft pairs kept={len(kept)} from allowed sft pool={len(allowed)}", flush=True)
    if not kept:
        raise SystemExit("no expert pairs in sft pool")

    out = run_lora_sft(
        model_path=cfg["model"]["path"],
        pairs=kept,
        system_prompt=ALFWORLD_FEWSHOT_PROMPT,
        out_dir=Path(args.out_adapter),
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        epochs=1,
        batch_size=2,
        grad_accum=8,
        lr=1e-4,
        max_length=4096,
        seed=1010,
        checkpoint_fractions=(0.7,),
    )
    meta = {k: v for k, v in out.items() if k != "history"}
    meta["n_pairs"] = len(kept)
    dump_json(Path(args.out_adapter).parent / "coldstart_meta.json", meta)
    print("COLDSTART_SFT", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
