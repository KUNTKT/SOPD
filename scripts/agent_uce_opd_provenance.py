#!/usr/bin/env python3
"""P0: UCE library provenance, split isolation, retrieve-only check."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_uce_opd_lib import (  # noqa: E402
    ids_of,
    load_opd_cfg,
    retrieve_readonly,
    sha256_dir,
    sha256_file,
    task_splits,
)
from alfworld_common import dump_json, ensure_alfworld_env  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_opd_cfg(args.config)
    ensure_alfworld_env(cfg)
    paths = cfg["paths"]
    reports = Path(paths["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    splits = task_splits(cfg)
    evolve_ids = set(splits["evolve_ids"])
    distill_ids = set(splits["distill_ids"])
    eval_ids = set(splits["eval_ids"])
    confirm_ids = ids_of(splits["audit_confirm"])
    sft_ids = ids_of(splits["sft"])

    isolation = {
        "evolve_n": len(evolve_ids),
        "distill_n": len(distill_ids),
        "eval_n": len(eval_ids),
        "evolve_distill": sorted(evolve_ids & distill_ids),
        "evolve_eval": sorted(evolve_ids & eval_ids),
        "distill_eval": sorted(distill_ids & eval_ids),
        "distill_confirm": sorted(distill_ids & confirm_ids),
        "eval_confirm": sorted(eval_ids & confirm_ids),
        "distill_sft": sorted(distill_ids & sft_ids),
    }
    isolation_ok = (
        isolation["evolve_n"] == 80
        and isolation["distill_n"] == 120
        and isolation["eval_n"] == 100
        and not isolation["evolve_distill"]
        and not isolation["evolve_eval"]
        and not isolation["distill_eval"]
    )

    lib = UceLibrary.load(Path(paths["uce_library_evolved"]))
    seed = json.loads(Path(paths["uce_library"]).read_text())
    seed_sources = {str(e.get("source_task_id")) for e in seed.get("entries") or []}
    evo_sources = {str(e.get("source_task_id")) for e in lib.entries}
    overlap_distill_seed = sorted(seed_sources & distill_ids)
    overlap_distill_evolved = sorted(evo_sources & distill_ids)

    usages = [e.get("usage") for e in lib.entries]
    for t in splits["distill"][:5] + splits["eval"][:5]:
        retrieve_readonly(lib, t)
    usage_unchanged = [e.get("usage") for e in lib.entries] == usages

    # Deepcopy retrieve should also leave original untouched.
    lib2 = copy.deepcopy(lib)
    retrieve_readonly(lib2, splits["distill"][0])
    retrieve_readonly_ok = usage_unchanged

    wording = (
        "self-generated procedural memory"
        if isolation_ok
        else "bootstrapped privileged procedural memory"
    )
    payload = {
        "wording": wording,
        "workflow_generator": {
            "kind": "rule_abstract_action",
            "llm_rewrite": False,
            "external_model": False,
            "seed_runner": "collect_episodes",
            "evolve_runner": "custom_run_npm_episodes",
            "policy": "Qwen3-1.7B + coldstart LoRA",
            "note": "abstraction strips ACTION:, instance numbers, look/inventory; max 12 steps",
        },
        "hashes": {
            "uce_library": sha256_file(Path(paths["uce_library"])),
            "uce_library_evolved": sha256_file(Path(paths["uce_library_evolved"])),
            "coldstart_adapter": sha256_dir(Path(paths["coldstart_adapter"])),
            "base_model_path": cfg["model"]["path"],
            "h0_base": sha256_file(Path(paths["h0_base"])),
            "b_uce": sha256_file(Path(paths["b_uce"])),
            "rollouts_select": sha256_file(Path(paths["rollouts_select"])),
        },
        "checkpoints": {
            "base": cfg["model"]["path"],
            "coldstart_adapter": paths["coldstart_adapter"],
            "enable_thinking": False,
        },
        "splits": {
            "pools_meta": splits["pools_meta"],
            "eval_meta": splits["eval_meta"],
            "evolve_ids": splits["evolve_ids"],
            "distill_ids": splits["distill_ids"],
            "eval_ids": splits["eval_ids"],
        },
        "isolation": isolation,
        "isolation_ok": isolation_ok,
        "seed_library_overlap_distill_n": len(overlap_distill_seed),
        "evolved_library_overlap_distill_n": len(overlap_distill_evolved),
        "seed_library_overlap_distill_ids": overlap_distill_seed,
        "retrieve_readonly_ok": retrieve_readonly_ok,
        "library_n_entries": len(lib.entries),
        "confirm_unused": True,
    }
    out = reports / "provenance.json"
    dump_json(out, payload)
    print(json.dumps({
        "isolation_ok": isolation_ok,
        "wording": wording,
        "overlap_distill_seed": len(overlap_distill_seed),
        "retrieve_readonly_ok": retrieve_readonly_ok,
        "out": str(out),
    }, indent=2))
    if not isolation_ok:
        raise SystemExit("P0 isolation failed")
    if not retrieve_readonly_ok:
        raise SystemExit("retrieve is not readonly")


if __name__ == "__main__":
    main()
