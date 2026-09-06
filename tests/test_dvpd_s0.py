#!/usr/bin/env python3
"""CPU tests for DVPD Gate S0."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dvpd_s0_lib import (  # noqa: E402
    WORKFLOW_HEAD,
    assert_clean_prompt,
    assert_replay_aligned,
    bootstrap_task_macro,
    compute_exclusion,
    delta_q,
    evaluate_gate_s0,
    first_full_action_diverge,
    h_remain,
    load_cfg,
    predicted_tool_at,
    prompt_has_workflow,
    refuse_forbidden_split,
    refuse_holdout_ids,
    replicate_terminal,
    retrieve_readonly,
    sample_remaining_tasks,
    suffix_seed,
    task_macro_mean,
    workflow_hash,
)
from uce_library import UceLibrary  # noqa: E402


def _assert(c: bool, msg: str) -> None:
    if not c:
        raise AssertionError(msg)


def _rec(tools: list[str], fps: list[str] | None = None) -> dict:
    dps = []
    for i, cmd in enumerate(tools):
        dps.append(
            {
                "step_index": i,
                "predicted_tool": cmd,
                "raw_action_text": f"ACTION: {cmd} extra-token-{i}",
                "state_fingerprint": (fps or [f"fp{i}"])[i] if fps else f"fp{i}",
            }
        )
    return {"decision_points": dps, "task_id": "alfw_test"}


def test_exclusion_and_forbid() -> None:
    cfg = load_cfg()
    excl = compute_exclusion(cfg)
    _assert(excl["n_excluded_union"] > 0, excl)
    _assert("evolve_jsonl" in excl["sources"], excl["sources"])
    _assert("provenance_distill_ids" in excl["sources"], excl["sources"])
    _assert(excl["sources"]["manifest_inner_train"]["n"] == 80, excl["sources"])
    union = set(excl["union_ids"])
    for name, meta in excl["sources"].items():
        _assert(meta["n"] >= 0, name)
    try:
        refuse_forbidden_split("valid_unseen")
        raise AssertionError("should refuse valid_unseen")
    except SystemExit:
        pass
    try:
        refuse_holdout_ids(["alfw_aaa"], {"alfw_aaa"})
        raise AssertionError("should refuse holdout leak")
    except SystemExit:
        pass
    split = sample_remaining_tasks(cfg)
    _assert(split["n"] == 60, split["n"])
    _assert(split["n_excluded_union"] == excl["n_excluded_union"], split)
    _assert(not (set(split["ids"]) & union), "sampled excluded id")
    _assert("eval_ids" not in split, split.keys())
    print("ok exclusion")


def test_retrieve_readonly_order() -> None:
    lib = UceLibrary(
        [
            {
                "id": "e1",
                "task_type": "pick_and_place_simple",
                "goal_tokens": ["apple", "table"],
                "text": f"{WORKFLOW_HEAD} A",
                "usage": 3,
            },
            {
                "id": "e2",
                "task_type": "pick_and_place_simple",
                "goal_tokens": ["mug", "shelf"],
                "text": f"{WORKFLOW_HEAD} B",
                "usage": 1,
            },
        ]
    )
    t1 = {"task_type": "pick_and_place_simple", "goal": "put apple on table"}
    t2 = {"task_type": "pick_and_place_simple", "goal": "put mug on shelf"}
    a1 = retrieve_readonly(lib, t1)
    b1 = retrieve_readonly(lib, t2)
    lib2 = UceLibrary([dict(e) for e in lib.entries])
    b2 = retrieve_readonly(lib2, t2)
    a2 = retrieve_readonly(lib2, t1)
    _assert(a1[:2] == a2[:2] and b1[:2] == b2[:2], (a1, a2, b1, b2))
    _assert([e["usage"] for e in lib.entries] == [3, 1], lib.entries)
    _assert([e["id"] for e in lib.entries] == ["e1", "e2"], lib.entries)
    print("ok retrieve")


def test_diverge_command_not_token() -> None:
    base = _rec(["go to table", "take apple"])
    same_cmd = _rec(["go to table", "take apple"])
    _assert(first_full_action_diverge(base, same_cmd) is None, "token-only must not count")
    _assert(predicted_tool_at(base, 1) == "take apple", predicted_tool_at(base, 1))
    uce = _rec(["go to table", "take mug"])
    _assert(first_full_action_diverge(base, uce) == 1, first_full_action_diverge(base, uce))
    empty = _rec(["go to table", ""])
    _assert(first_full_action_diverge(base, empty) is None, "empty tool illegal")
    print("ok diverge")


def test_remain_and_copy() -> None:
    _assert(h_remain(40, 0) == 39, h_remain(40, 0))
    _assert(h_remain(40, 39) == 0, h_remain(40, 39))
    outs = replicate_terminal(1, 8)
    _assert(outs == [1] * 8, outs)
    _assert(len(replicate_terminal(0, 8)) == 8, "K must stay 8")
    print("ok remain")


def test_clean_prompt_and_fp() -> None:
    wf = f"{WORKFLOW_HEAD} from a similar solved instance"
    h = workflow_hash(wf)
    _assert(prompt_has_workflow(f"{wf}\n\nTask:", workflow_text=wf, wf_hash=h), "head")
    try:
        assert_clean_prompt(f"hash={h}", wf_hash=h)
        raise AssertionError("hash should fail")
    except AssertionError:
        pass
    assert_clean_prompt("Task: put apple\nAdmissible: go to table")
    assert_replay_aligned("abc", "abc", "abc")
    try:
        assert_replay_aligned("abc", "abc", "xyz")
        raise AssertionError("arms must match before force")
    except AssertionError:
        pass
    print("ok prompt/fp")


def test_rejected_stays_in_delta() -> None:
    d = delta_q([1, 1, 0, 0, 0, 0, 0, 0], [0] * 8)
    state = {"task_id": "t1", "delta_q": d, "accepted_m": False, "accepted_0": True}
    _assert(abs(d - 0.25) < 1e-9, d)
    g = evaluate_gate_s0([state], {"gate": {"min_fork_states": 1, "min_tasks": 1, "mean_min": 0.0, "large_pos_min": 1, "accept_min": 0.0}})
    _assert(g["n_fork_states"] == 1, g)
    _assert(g["signed_large_count"] == 1, g)
    print("ok rejected-in-delta")


def test_task_macro_and_gate() -> None:
    states = [
        {"task_id": "a", "delta_q": 1.0, "accepted_m": True, "accepted_0": True},
        {"task_id": "a", "delta_q": 0.0, "accepted_m": True, "accepted_0": True},
        {"task_id": "b", "delta_q": 0.0, "accepted_m": True, "accepted_0": True},
    ]
    _assert(abs(task_macro_mean(states) - 0.25) < 1e-9, task_macro_mean(states))
    boot = bootstrap_task_macro(states, n_boot=200, seed=1)
    _assert(boot["n_tasks"] == 2, boot)
    _assert(suffix_seed("t", 0, 1) == suffix_seed("t", 0, 1), "suffix not stable")
    _assert(suffix_seed("t", 0, 1) != suffix_seed("t", 0, 2), "suffix k")
    pos = [{"task_id": f"t{i}", "delta_q": 0.5, "accepted_m": True, "accepted_0": True} for i in range(30)]
    extra = [{"task_id": f"u{i}", "delta_q": 0.5, "accepted_m": True, "accepted_0": True} for i in range(20)]
    good = evaluate_gate_s0(pos + extra, {"n_boot": 200, "bootstrap_seed": 1, "gate": {"min_fork_states": 50, "min_tasks": 30, "mean_min": 0.10, "large_pos_min": 30, "accept_min": 0.95}})
    _assert(good["pass"], good)
    neg = [{**s, "delta_q": -0.5} for s in pos + extra]
    bad = evaluate_gate_s0(neg, {"n_boot": 200, "bootstrap_seed": 1, "gate": {"min_fork_states": 50, "min_tasks": 30, "mean_min": 0.10, "large_pos_min": 30, "accept_min": 0.95}})
    _assert(not bad["pass"] and "signed_large" in bad["reasons"], bad)
    _assert(bad["negative_large_count"] == 50, bad)
    print("ok gate")


def main() -> None:
    test_exclusion_and_forbid()
    test_retrieve_readonly_order()
    test_diverge_command_not_token()
    test_remain_and_copy()
    test_clean_prompt_and_fp()
    test_rejected_stays_in_delta()
    test_task_macro_and_gate()
    print("ALL DVPD S0 CPU TESTS PASSED")


if __name__ == "__main__":
    main()
