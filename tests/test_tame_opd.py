#!/usr/bin/env python3
"""CPU unit tests for TAME-OPD Gate T0 contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tame_opd_lib import (  # noqa: E402
    FORBIDDEN_SPLITS,
    apply_deterministic_fallback,
    argmax_with_tiebreak,
    audit_library,
    build_m0_library,
    entry_seed,
    evaluate_gates,
    fake_whole_transcript_prepend,
    novel_leak_hits,
    parse_rewrite_lines,
    plan_probe_budget,
    refuse_forbidden_split,
    sample_audit_ids,
    sub_seed,
    task_mean_paired_bootstrap,
    teacher_prefix,
    truncate_mask,
)
from tame_opd_lib import apply_workflow, strip_workflow  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_forbidden_split() -> None:
    try:
        refuse_forbidden_split("valid_unseen")
    except SystemExit:
        print("ok forbidden_split")
        return
    raise AssertionError("valid_unseen must SystemExit")


def test_entry_and_sub_seeds() -> None:
    a = entry_seed(11, "sel_0001_alfw_c45354f088")
    b = entry_seed(11, "sel_0001_alfw_c45354f088")
    c = entry_seed(12, "sel_0001_alfw_c45354f088")
    _assert(a == b, "entry seed unstable")
    _assert(a != c, "library seed must change entry seed")
    digest_prefix = __import__("hashlib").sha256(b"sel_0001_alfw_c45354f088").hexdigest()[:16]
    expect = (11 + int(digest_prefix, 16)) % (2**31)
    _assert(a == expect, f"entry seed formula {a} != {expect}")
    s1 = sub_seed(2020, "init")
    s2 = sub_seed(2020, "rollout")
    _assert(s1 != s2, "roles must be independent")
    _assert(sub_seed(2020, "init") == s1, "sub seed unstable")
    print("ok seeds")


def test_teacher_prefix_b_uce() -> None:
    path = ROOT / "reports/uce_alfworld/B_uce.jsonl"
    rec = None
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            dps = rec.get("decision_points") or []
            if len(dps) >= 3:
                break
    _assert(rec is not None and len(rec.get("decision_points") or []) >= 3, "need a mid-step B-uce episode")
    dps = rec["decision_points"]
    first = dps[0]["prefix_text"]
    mid = dps[len(dps) // 2]["prefix_text"]
    _assert(first.startswith("Suggested workflow"), first[:40])
    wf = first.split("\n\n", 1)[0]
    rec_first = teacher_prefix(strip_workflow(first), wf)
    rec_mid = teacher_prefix(strip_workflow(mid), wf)
    _assert(rec_first == first, "first-step teacher prefix != B-uce")
    _assert(rec_mid == mid, "mid-step teacher prefix != B-uce")
    fake = fake_whole_transcript_prepend(strip_workflow(first), strip_workflow(mid), wf)
    _assert(fake != mid, "fake whole-transcript prepend must differ at mid-step")
    print("ok b_uce_prefix")


def test_tokenizer_input_ids() -> None:
    path = ROOT / "data/ssopd05_alfworld_coldstart/coldstart_lora"
    try:
        from transformers import AutoTokenizer
    except Exception:
        print("skip tokenizer (transformers missing)")
        return
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    rec = None
    with (ROOT / "reports/uce_alfworld/B_uce.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if len(rec.get("decision_points") or []) >= 3:
                break
    dps = rec["decision_points"]
    first = dps[0]["prefix_text"]
    mid = dps[len(dps) // 2]["prefix_text"]
    wf = first.split("\n\n", 1)[0]

    def ids(obs: str) -> list[int]:
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": obs},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    _assert(ids(first) == ids(teacher_prefix(strip_workflow(first), wf)), "first input_ids")
    _assert(ids(mid) == ids(teacher_prefix(strip_workflow(mid), wf)), "mid input_ids")
    fake = fake_whole_transcript_prepend(strip_workflow(first), strip_workflow(mid), wf)
    _assert(ids(fake) != ids(mid), "fake builder input_ids must differ")
    print("ok input_ids")


def test_leak_only_novel() -> None:
    orig = "Suggested workflow\n1. go to drawer\n2. take apple from drawer"
    rw_ok = "Suggested workflow\n1. go to drawer\n2. take apple from drawer\n3. close drawer"
    rw_bad = "Suggested workflow\n1. go to drawer\n2. take apple 1 from drawer"
    _assert(novel_leak_hits(rw_ok, orig) == [], novel_leak_hits(rw_ok, orig))
    hits = novel_leak_hits(rw_bad, orig)
    _assert(any("instance" in h for h in hits), hits)
    already = "Suggested workflow\n1. take apple 1 from drawer"
    _assert(novel_leak_hits(already, already) == [], "original instance is not a leak")
    print("ok leak_novel")


def test_fallback_and_invalid() -> None:
    m0 = build_m0_library(
        {
            "entries": [
                {
                    "id": "e1",
                    "task_type": "pick_and_place_simple",
                    "goal": "put apple on table",
                    "goal_tokens": ["apple", "table"],
                    "lines": ["go to table", "take apple"],
                    "text": "Suggested workflow from a similar solved instance (adapt names to the current admissible list):\n1. go to table\n2. take apple",
                    "source_task_id": "t1",
                    "usage": 1,
                }
            ]
        }
    )
    bad = {
        "candidate_id": "M1",
        "entries": [
            {
                **m0["entries"][0],
                "text": "put apple 1 on table 2 and alfw_deadbeef",
                "lines": ["put apple 1 on table 2"],
                "fallback": False,
            }
        ],
    }
    audit = audit_library(bad, m0, whitelist={"go", "take", "put"}, extra_goals=[], extra_pairs=[], fallback_rate_max=0.05)
    _assert(not audit["entries"][0]["ok"], audit)
    reverted = apply_deterministic_fallback(bad, m0, audit)
    _assert(reverted["entries"][0]["reverted"] is True, reverted["entries"][0])
    _assert("apple 1" not in reverted["entries"][0]["text"], reverted["entries"][0]["text"])
    audit2 = audit_library(reverted, m0, whitelist={"go", "take", "put"}, fallback_rate_max=0.05)
    _assert(audit2["fallback_rate"] > 0.05, audit2["fallback_rate"])
    _assert(audit2["valid"] is False, "fallback>5% must invalidate")
    print("ok fallback_invalid")


def test_tiebreak_and_gates() -> None:
    _assert(argmax_with_tiebreak({"M0": 0.1, "M1": 0.1}, ["M0", "M1"]) == "M0", "tie -> M0")
    _assert(argmax_with_tiebreak({"M2": 0.4, "M1": 0.2}, ["M1", "M2"]) == "M2", "higher wins")
    g = evaluate_gates(
        teach_pick=None,
        teacher_pick=None,
        eligible=["M0"],
        delta_m0=0.0,
        ci_lo_m0=0.0,
        delta_teacher=0.0,
        ci_lo_teacher=0.0,
        n_pos_seeds_vs_m0=0,
        teach_vs_init=0.0,
        teacher_j_teach=0.4,
        teacher_j_m0=0.4,
    )
    _assert(g["T0_B"]["status"] == "not_identifiable", g)
    _assert(g["method_gate_pass"] is False, g)
    g2 = evaluate_gates(
        teach_pick="M1",
        teacher_pick="M1",
        eligible=["M0", "M1"],
        delta_m0=0.05,
        ci_lo_m0=0.01,
        delta_teacher=0.0,
        ci_lo_teacher=0.0,
        n_pos_seeds_vs_m0=3,
        teach_vs_init=0.05,
        teacher_j_teach=0.4,
        teacher_j_m0=0.4,
    )
    _assert(g2["T0_B"]["status"] == "not_identifiable", g2)
    g3 = evaluate_gates(
        teach_pick="M2",
        teacher_pick="M1",
        eligible=["M0", "M1", "M2"],
        delta_m0=0.04,
        ci_lo_m0=0.01,
        delta_teacher=0.02,
        ci_lo_teacher=0.005,
        n_pos_seeds_vs_m0=2,
        teach_vs_init=0.03,
        teacher_j_teach=0.41,
        teacher_j_m0=0.40,
    )
    _assert(g3["T0_A"]["pass"] is True, g3)
    _assert(g3["T0_B"]["pass"] is True, g3)
    _assert(g3["method_gate_pass"] is True, g3)
    print("ok gates")


def test_budget_truncation() -> None:
    keep = plan_probe_budget([100, 100, 100], max_steps=64, max_tokens=250)
    _assert(keep[:3] == [100, 100, 50], keep)
    _assert(sum(keep) == 250, keep)
    mask = [False, True, True, True, False]
    trunc = truncate_mask(mask, 2)
    _assert(trunc == [False, True, True, False, False], trunc)
    print("ok budget")


def test_task_mean_bootstrap() -> None:
    a = {"t1": 0.0, "t2": 1.0, "t3": 0.0}
    b = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    boot = task_mean_paired_bootstrap(a, b, n_boot=200, seed=0)
    _assert(boot["n_tasks"] == 3, boot)
    _assert(boot["mean"] > 0, boot)
    _assert(boot["n_win"] == 2, boot)
    print("ok bootstrap")


def test_parse_rewrite() -> None:
    raw = "1. go to fridge\n2. take apple\n3. look\n\nThanks"
    lines = parse_rewrite_lines(raw)
    _assert(lines == ["go to fridge", "take apple"], lines)
    print("ok parse")


def test_sample_ids() -> None:
    entries = [{"id": f"e{i}", "task_type": "pick_and_place_simple" if i % 2 == 0 else "look_at_obj_in_light"} for i in range(20)]
    ids = sample_audit_ids(entries, 6, seed=11)
    _assert(len(ids) == 6, ids)
    _assert(len(set(ids)) == 6, ids)
    print("ok sample")


def main() -> None:
    _assert(list(FORBIDDEN_SPLITS) == ["valid_unseen"], FORBIDDEN_SPLITS)
    test_forbidden_split()
    test_entry_and_sub_seeds()
    test_teacher_prefix_b_uce()
    test_tokenizer_input_ids()
    test_leak_only_novel()
    test_fallback_and_invalid()
    test_tiebreak_and_gates()
    test_budget_truncation()
    test_task_mean_bootstrap()
    test_parse_rewrite()
    test_sample_ids()
    print("ALL TAME-OPD T0 CPU TESTS PASSED")


if __name__ == "__main__":
    main()
