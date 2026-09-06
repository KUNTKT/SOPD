#!/usr/bin/env python3
"""TAME-OPD Gate T0 orchestrator. Stops after the gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def _run(py: str, script: str, extra: list[str]) -> None:
    cmd = [py, str(SCRIPT_DIR / script), *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiment_tame_opd.yaml"))
    ap.add_argument(
        "--phase",
        choices=["splits", "rewrite", "teacher", "probe", "gate", "all", "test"],
        default="all",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    py = str(cfg.get("python") or sys.executable)
    extra = ["--config", args.config]
    resume = ["--resume"] if args.resume else ["--no-resume"]
    if args.phase == "test":
        _run(py, "test_tame_opd.py", [])
        return
    if args.phase in {"splits", "all"}:
        _run(py, "tame_opd_splits.py", extra)
    if args.phase in {"rewrite", "all"}:
        _run(py, "tame_opd_rewrite.py", extra + resume)
    if args.phase in {"teacher", "all"}:
        _run(py, "tame_opd_probe.py", extra + resume + ["--phase", "teacher"])
    if args.phase in {"probe", "all"}:
        _run(py, "tame_opd_probe.py", extra + resume + ["--phase", "probe"])
    if args.phase in {"gate", "all"}:
        _run(py, "tame_opd_gate.py", extra + resume)
    print("TAME-OPD Gate T0 finished (no multi-round training).", flush=True)


if __name__ == "__main__":
    main()
