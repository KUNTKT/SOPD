#!/usr/bin/env python3
"""CPU tests for T1-V patch schema, leaks, sampling, and Gate V."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tame_t1v_lib import (  # noqa: E402
    LIBRARY_SEEDS,
    aggregate_error_stats,
    canonical_hash,
    delexicalize,
    disagreement_matrix,
    diversity_ok,
    editor_user_prompt,
    evaluate_gate_v,
    extract_json_object,
    per_seed_metrics,
    process_generation,
    sample_60,
    seed_passes,
    usage_bucket,
    validate_patch,
)


def _assert(c: bool, msg: str) -> None:
    if not c:
        raise AssertionError(msg)


def test_delex_and_stats_no_obs() -> None:
    _assert(delexicalize("take apple 1 from table 2") == "take apple from table", delexicalize("take apple 1 from table 2"))
    recs = [
        {
            "task_type": "pick_and_place_simple",
            "task_id": "alfw_deadbeef",
            "goal": "put apple 1 on table 2",
            "decision_points": [
                {
                    "action_text": "take apple 1",
                    "verdict": {"failure_reason": "not_admissible"},
                    "state": {"feedback": "Nothing happens. The drawer 1 is closed."},
                },
                {"action_text": "take apple 1", "verdict": {}, "state": {"feedback": "ok"}},
            ],
        }
    ]
    stats = aggregate_error_stats(recs)
    prompt = editor_user_prompt(["go to table", "take apple"], stats["pick_and_place_simple"])
    _assert("apple 1" not in prompt, prompt)
    _assert("alfw_deadbeef" not in prompt, prompt)
    _assert("put apple 1" not in prompt, prompt)
    _assert("ACTION_REJECTED" in prompt or "CONTAINER_CLOSED" in prompt, prompt)
    print("ok delex")


def test_schema_and_render() -> None:
    lines = ["go to table", "take apple", "go to fridge", "open fridge"]
    ok, err = extract_json_object('{"delete_rules":[4],"rewrite_rules":[],"add_rules":[]}')
    _assert(err is None and validate_patch(ok, 4) is None, err)
    bad = {"delete_rules": [2], "rewrite_rules": [{"rule_id": 2, "new_text": "x"}], "add_rules": []}
    _assert(validate_patch(bad, 4) == "delete_rewrite_overlap", validate_patch(bad, 4))
    extra = {"delete_rules": [], "rewrite_rules": [], "add_rules": [], "note": "no"}
    _assert(validate_patch(extra, 4) == "additional_properties", validate_patch(extra, 4))
    wipe = {"delete_rules": [1, 2, 3, 4], "rewrite_rules": [], "add_rules": []}
    _assert(validate_patch(wipe, 4) == "deleted_all", validate_patch(wipe, 4))
    prose, e = extract_json_object('Here you go\n{"delete_rules":[],"rewrite_rules":[],"add_rules":[]}\nThanks')
    _assert(e == "prose_around_json", e)
    print("ok schema")


def test_process_leak_noop() -> None:
    lines = ["go to table", "take apple"]
    orig = "1. go to table\n2. take apple"
    empty = process_generation(
        '{"delete_rules":[],"rewrite_rules":[],"add_rules":[]}', lines, orig
    )
    _assert(empty["schema_ok"] and empty["accepted"] and empty["noop"] and not empty["fallback"], empty)
    leak = process_generation(
        '{"delete_rules":[],"rewrite_rules":[{"rule_id":2,"new_text":"take apple 1"}],"add_rules":[]}',
        lines,
        orig,
    )
    _assert(leak["raw_leak"] and leak["fallback"] and leak["fallback_reason"] == "raw_leak", leak)
    _assert(leak["schema_ok"] and not leak["accepted"], leak)
    resid = process_generation(
        '{"delete_rules":[],"rewrite_rules":[{"rule_id":2,"new_text":"take fruit"}],"add_rules":[]}',
        lines,
        orig,
    )
    _assert(resid["accepted"] and not resid["residual_leak"], resid)
    good = process_generation(
        '{"delete_rules":[],"rewrite_rules":[{"rule_id":1,"new_text":"approach the table"}],"add_rules":[]}',
        lines,
        orig,
    )
    _assert(good["real_modify"] and good["accepted"] and not good["fallback"], good)
    _assert(good["canonical_hash"] != canonical_hash(lines), good)
    print("ok process")


def test_sample_buckets() -> None:
    _assert(usage_bucket(0) == "0", usage_bucket(0))
    _assert(usage_bucket(2) == "1-2", usage_bucket(2))
    _assert(usage_bucket(5) == ">=3", usage_bucket(5))
    entries = []
    types = ["pick_and_place_simple", "look_at_obj_in_light", "pick_two_obj_and_place"]
    for i in range(90):
        entries.append(
            {
                "id": f"e{i:03d}",
                "task_type": types[i % 3],
                "usage": [0, 1, 4][i % 3],
                "lines": ["go to table", "take apple"],
                "text": "wf",
            }
        )
    a = sample_60(entries, 12, 4040)
    b = sample_60(entries, 12, 4040)
    _assert(a["ids"] == b["ids"], "sample not reproducible")
    _assert(len(a["ids"]) == 12, a)
    _assert(all("bucket" in e and "type" in e and "parent_hash" in e for e in a["entries"]), a["entries"][0])
    tiny = [
        {"id": "only_a", "task_type": "aaa", "usage": 0, "lines": ["go"], "text": "wf", "parent_hash": "p"},
    ]
    for i in range(20):
        tiny.append(
            {
                "id": f"b{i:02d}",
                "task_type": "bbb",
                "usage": 5,
                "lines": ["go"],
                "text": "wf",
                "parent_hash": "p",
            }
        )
    c = sample_60(tiny, 5, 4040)
    _assert(len(c["ids"]) == 5, c)
    _assert(c["layer_actual"]["aaa|0"]["planned"] >= 1, c["layer_actual"])
    _assert(c["layer_actual"]["aaa|0"]["actual"] == 1, c["layer_actual"])
    print("ok sample")


def test_gate_math() -> None:
    def row(*, schema=True, fb=False, modify=True, raw=False, resid=False):
        return {
            "schema_ok": schema,
            "fallback": fb,
            "real_modify": modify and not fb,
            "raw_leak": raw,
            "accepted": (not fb) and schema,
            "residual_leak": resid,
        }

    good_rows = [row() for _ in range(60)]
    m = per_seed_metrics(good_rows)
    ok, reasons = seed_passes(m)
    _assert(ok, (m, reasons))
    bad_rows = [row(fb=True, modify=False, schema=False) for _ in range(10)] + [row() for _ in range(50)]
    mb = per_seed_metrics(bad_rows)
    okb, _ = seed_passes(mb)
    _assert(not okb, mb)

    by = {}
    for i in range(60):
        eid = f"e{i}"
        by[eid] = {
            11: {"real_modify": True, "fallback": False, "canonical_hash": "a"},
            12: {"real_modify": True, "fallback": False, "canonical_hash": "b"},
            13: {"real_modify": True, "fallback": False, "canonical_hash": "c"},
            14: {"real_modify": False, "fallback": False, "canonical_hash": "orig"},
        }
    d = diversity_ok(by)
    _assert(d["pass"] and d["n_ok"] == 60, d)
    weak = {eid: {11: {"real_modify": False, "fallback": False, "canonical_hash": "o"},
                  12: {"real_modify": False, "fallback": False, "canonical_hash": "o"},
                  13: {"real_modify": False, "fallback": False, "canonical_hash": "o"},
                  14: {"real_modify": True, "fallback": False, "canonical_hash": "x"}} for eid in by}
    dw = diversity_ok(weak)
    _assert(not dw["pass"], dw)
    mat = disagreement_matrix(by)
    _assert(mat["11vs12"] == 1.0, mat)
    per = {s: m for s in LIBRARY_SEEDS}
    g = evaluate_gate_v(per, d, 0.95)
    _assert(g["pass"], g)
    g2 = evaluate_gate_v(per, dw, 0.95)
    _assert(not g2["pass"] and "diversity" in g2["reasons"], g2)
    print("ok gate")


def main() -> None:
    test_delex_and_stats_no_obs()
    test_schema_and_render()
    test_process_leak_noop()
    test_sample_buckets()
    test_gate_math()
    print("ALL T1-V CPU TESTS PASSED")


if __name__ == "__main__":
    main()
