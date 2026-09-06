#!/usr/bin/env python3
"""Fit L14 task_balanced_paired v_cap from GSM8K rollouts + splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SSOPD = Path("/scratch/ktang115/SSOPD")
if str(SSOPD) not in sys.path:
    sys.path.insert(0, str(SSOPD))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.nxt.extract import attach_block_output_hooks, remove_hooks
from ssopd_math.nxt.paired_direction import build_directions_from_records
from ssopd_math.nxt.positions import locate_sites
from ssopd_math.verifier.reward import NEGATIVE, POSITIVE

LAYER = 14


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def extract_hidden_records(model: HFMathModel, records: list[dict], layer_indices: list[int]) -> list[dict]:
    out = []
    device = model.device
    target = [r for r in records if r.get("verification_status") in (POSITIVE, NEGATIVE)]
    for n_done, rec in enumerate(target, 1):
        if n_done % 50 == 0 or n_done == len(target):
            print(f"extract hidden {n_done}/{len(target)}", flush=True)
        pack = locate_sites(
            model.tokenizer,
            rec.get("prompt_user") or rec["problem"],
            rec["completion_text"],
        )
        pos_site = pack.sites["reasoning_end"]
        pos = min(pos_site.token_position, len(rec["full_token_ids"]) - 1)
        fallback = bool(pos_site.representation_fallback or (not pack.marker_found))
        ids = rec["full_token_ids"]
        input_ids = torch.tensor([ids], device=device)
        attn = torch.ones_like(input_ids)
        cache, handles = attach_block_output_hooks(model.model, layer_indices)
        try:
            with torch.no_grad():
                model.model(input_ids, attn, use_cache=False)
        finally:
            remove_hooks(handles)
        hidden_states = {
            layer: cache[layer][0, pos].float().cpu().numpy()
            for layer in layer_indices
            if layer in cache
        }
        out.append(
            {
                "problem_id": rec["problem_id"],
                "trajectory_id": rec["trajectory_id"],
                "verification_status": rec["verification_status"],
                "representation_fallback": fallback,
                "token_position": pos,
                "hidden_states": hidden_states,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default="/scratch/ktang115/models/Qwen3-1.7B")
    ap.add_argument("--split-filter", default="vector_fit", help="Only use this split for fit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(args.splits).read_text())
    keep = {
        pid
        for pid, s in splits["problem_splits"].items()
        if s == args.split_filter
    }
    records = [
        r
        for r in load_jsonl(Path(args.rollouts))
        if str(r["problem_id"]) in keep
        and r.get("verification_status") in (POSITIVE, NEGATIVE)
    ]
    print(f"fit records={len(records)} split={args.split_filter}", flush=True)

    model = HFMathModel(
        args.model_path,
        device="cuda",
        torch_dtype="bfloat16",
        local_files_only=True,
        use_cache=False,
        max_new_tokens=64,
    )
    t0 = time.time()
    try:
        hidden = extract_hidden_records(model, records, [LAYER])
    finally:
        model.close()

    dirs = build_directions_from_records(
        hidden,
        layer=LAYER,
        variant="task_balanced_paired",
        seed=42,
    )
    key = f"layer_{LAYER}_task_balanced_paired_v_cap"
    key_hat = f"layer_{LAYER}_task_balanced_paired_v_hat"
    v_cap = np.asarray(dirs["v_cap"], dtype=np.float32)
    v_hat = np.asarray(dirs["v_hat"], dtype=np.float32)
    payload = {key: v_cap, key_hat: v_hat}
    meta = {
        "layer": LAYER,
        "n_hidden": len(hidden),
        "n_paired": int(dirs["n_problems"]),
        "fallback_count": int(dirs["fallback_count"]),
        "direction_key": key,
        "norm": float(np.linalg.norm(v_cap)),
        "wall_s": time.time() - t0,
    }
    np.savez(out_dir / "directions.npz", **payload)
    (out_dir / "fit_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"wrote directions paired={meta['n_paired']} ||v||={meta['norm']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
