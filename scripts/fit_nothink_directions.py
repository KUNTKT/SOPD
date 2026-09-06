#!/usr/bin/env python3
"""Fit L14 v_cap from no-think rollouts (archive; does not write into SSOPD)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
if str(SSOPD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSOPD_ROOT))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.nxt.extract import attach_block_output_hooks, remove_hooks
from ssopd_math.nxt.paired_direction import build_directions_from_records, cosine
from ssopd_math.nxt.positions import locate_sites
from ssopd_math.verifier.reward import NEGATIVE, POSITIVE
import ssopd_math.models.math_model as math_model_mod
import ssopd_math.nxt.positions as positions_mod

LAYER = 14
KEY = "layer_14_task_balanced_paired_v_cap"
KEY_HAT = "layer_14_task_balanced_paired_v_hat"


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


def pair_stats(records: list[dict]) -> dict:
    by: dict[str, set[str]] = defaultdict(set)
    for r in records:
        st = r.get("verification_status")
        if st in (POSITIVE, NEGATIVE):
            by[str(r["problem_id"])].add(st)
    n = len(by)
    paired = sum(1 for s in by.values() if POSITIVE in s and NEGATIVE in s)
    pos_only = sum(1 for s in by.values() if POSITIVE in s and NEGATIVE not in s)
    neg_only = sum(1 for s in by.values() if NEGATIVE in s and POSITIVE not in s)
    return {
        "n_problems": n,
        "n_paired": paired,
        "pos_only": pos_only,
        "neg_only": neg_only,
        "n_traj": len(records),
        "success_rate": sum(1 for r in records if r.get("verification_status") == POSITIVE) / max(len(records), 1),
    }


def extract_hidden_records(model: HFMathModel, records: list[dict], layer_indices: list[int]) -> list[dict]:
    out: list[dict] = []
    device = model.device
    n_target = sum(1 for r in records if r.get("verification_status") in (POSITIVE, NEGATIVE))
    n_done = 0
    for rec in records:
        if rec.get("verification_status") not in (POSITIVE, NEGATIVE):
            continue
        n_done += 1
        if n_done % 50 == 0 or n_done == n_target:
            print(f"extract hidden {n_done}/{n_target}", flush=True)
        pack = locate_sites(
            model.tokenizer,
            rec.get("prompt_user") or rec.get("extra", {}).get("prompt_user") or rec["problem"],
            rec["completion_text"],
        )
        pos_site = pack.sites["reasoning_end"]
        pos = min(pos_site.token_position, len(rec["full_token_ids"]) - 1)
        fallback = bool(pos_site.representation_fallback or (not pack.marker_found))
        ids = rec["full_token_ids"]
        input_ids = __import__("torch").tensor([ids], device=device)
        attn = __import__("torch").ones_like(input_ids)
        cache, handles = attach_block_output_hooks(model.model, layer_indices)
        try:
            with __import__("torch").no_grad():
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
    ap.add_argument(
        "--rollouts",
        default="/home/yiyangba/ssopd_paper_archive/data/ssopd01_qwen3_1_7b_nothink_vf1200/rollouts.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/yiyangba/ssopd_paper_archive/data/ssopd02_qwen3_1_7b_nothink",
    )
    ap.add_argument("--min-paired", type=int, default=80)
    ap.add_argument("--ref-directions", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd02_qwen3_1_7b_scale2k/directions.npz")
    args = ap.parse_args()

    patch_disable_thinking()
    records = load_jsonl(Path(args.rollouts))
    stats = pair_stats(records)
    print("pair_stats", json.dumps(stats), flush=True)
    if stats["n_paired"] < args.min_paired:
        raise SystemExit(
            f"paired={stats['n_paired']} < min_paired={args.min_paired}; increase K or use scheme-3"
        )

    by_st: dict[str, set[str]] = defaultdict(set)
    for r in records:
        st = r.get("verification_status")
        if st in (POSITIVE, NEGATIVE):
            by_st[str(r["problem_id"])].add(st)
    paired_ids = {pid for pid, s in by_st.items() if POSITIVE in s and NEGATIVE in s}
    subset = [
        r
        for r in records
        if str(r["problem_id"]) in paired_ids
        and r.get("verification_status") in (POSITIVE, NEGATIVE)
        and r.get("full_token_ids")
    ]
    print(f"extract subset paired_problems={len(paired_ids)} traj={len(subset)}", flush=True)
    model = HFMathModel(
        "/scratch/ktang115/models/Qwen3-1.7B",
        device="cuda",
        torch_dtype="bfloat16",
        local_files_only=True,
        use_cache=False,
        max_new_tokens=1024,
    )
    t0 = time.time()
    hidden_recs = extract_hidden_records(model, subset, [LAYER])
    print(f"extract done n={len(hidden_recs)} wall_s={time.time()-t0:.1f}", flush=True)
    model.close()

    built = build_directions_from_records(
        hidden_recs, layer=LAYER, variant="task_balanced_paired", seed=42
    )
    v_cap = np.asarray(built["v_cap"], dtype=np.float32)
    v_hat = np.asarray(built["v_hat"], dtype=np.float32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "directions.npz", **{KEY: v_cap, KEY_HAT: v_hat})

    cos = None
    ref_path = Path(args.ref_directions)
    if ref_path.exists():
        ref = np.load(ref_path)
        if KEY in ref.files:
            cos = float(cosine(v_cap, np.asarray(ref[KEY], dtype=np.float64)))

    meta = {
        **stats,
        "n_paired_used": int(built["n_problems"]),
        "fallback_count": int(built["fallback_count"]),
        "v_cap_norm": float(np.linalg.norm(v_cap)),
        "cosine_to_thinking_on": cos,
        "direction_key": KEY,
        "layer": LAYER,
        "enable_thinking": False,
        "extract_wall_s": time.time() - t0,
    }
    (out_dir / "fit_meta.json").write_text(json.dumps(meta, indent=2))
    print("FIT", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
