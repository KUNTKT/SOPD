#!/usr/bin/env python3
"""Build vector_fit / selection / confirm splits from GSM8K rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SSOPD = Path("/scratch/ktang115/SSOPD")
if str(SSOPD) not in sys.path:
    sys.path.insert(0, str(SSOPD))

from ssopd_math.datasets.splits import build_split_manifest, save_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vector-fit-frac", type=float, default=0.6)
    ap.add_argument("--selection-frac", type=float, default=0.2)
    ap.add_argument("--confirm-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = sorted(
        {
            str(json.loads(line)["problem_id"])
            for line in Path(args.rollouts).read_text().splitlines()
            if line.strip()
        }
    )
    manifest = build_split_manifest(
        ids,
        vector_fit_frac=args.vector_fit_frac,
        selection_frac=args.selection_frac,
        confirm_frac=args.confirm_frac,
        seed=args.seed,
    )
    save_splits(args.out, manifest)
    print(f"wrote {args.out} counts={manifest['counts']} n_ids={len(ids)}", flush=True)


if __name__ == "__main__":
    main()
