#!/usr/bin/env python3
"""Build the TAME-OPD train-only task manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, ensure_alfworld_env  # noqa: E402
from tame_opd_lib import build_manifest, load_tame_cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_tame_cfg(args.config)
    ensure_alfworld_env(cfg)
    payload = build_manifest(cfg)
    out = Path(cfg["paths"]["reports_dir"]) / "task_manifest.json"
    dump_json(out, payload)
    print(
        f"wrote {out} hash={payload['manifest_hash']} "
        f"n={payload['n']} forbidden={payload['forbidden_splits']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
