#!/usr/bin/env python3
"""Skip A2 distill when A1 gate fails; write skip marker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, load_yaml_cfg  # noqa: E402


def main() -> None:
    cfg = load_yaml_cfg(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_alfworld.yaml")
    a1_path = Path(cfg["paths"]["reports_dir"]) / "a1_results.json"
    a1 = json.loads(a1_path.read_text())
    gate = a1.get("analysis", {})
    payload = {
        "skipped": True,
        "reason": "A1 gate_pass=False; refuse distill per pre-registration",
        "a1_analysis": gate,
    }
    out = Path(cfg["paths"]["reports_dir"]) / "a2_results.json"
    dump_json(out, payload)
    print("A2_SKIPPED", json.dumps(payload, indent=2))
    if gate.get("gate_pass"):
        raise SystemExit("A1 passed — run agent_ssopd_collect_teacher.py + agent_ssopd_distill_lora.py")


if __name__ == "__main__":
    main()
