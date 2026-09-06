#!/usr/bin/env python3
"""CPU unit tests for H5 whole-action scoring math (no model)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uce_counterfactual_logit import kl_logp  # noqa: E402
from uce_whole_action import (  # noqa: E402
    action_kl_at_beta,
    apply_workflow,
    extract_command,
    js_divergence,
    match_admissible,
    rank_of,
    scores_to_logq,
    steered_logq,
    strip_workflow,
    summarize_pair,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_workflow_inverse() -> None:
    wf = (
        "Suggested workflow from a similar solved instance "
        "(adapt names to the current admissible list):\n"
        "1. go to armchair\n"
        "2. take box from armchair"
    )
    obs = "Task: place a salt shaker\n\nObservation: you see a table"
    stacked = apply_workflow(obs, wf)
    _assert(strip_workflow(stacked) == obs, f"inverse failed:\n{strip_workflow(stacked)!r}")
    _assert(strip_workflow(obs) == obs, "bare obs must stay")
    print("ok workflow_inverse")


def test_beta0_is_uce() -> None:
    torch.manual_seed(0)
    s0 = torch.randn(7)
    sp = torch.randn(7)
    log_qt = steered_logq(s0, sp, 0.0, tau_a=1.0)
    log_qp = scores_to_logq(sp, 1.0)
    _assert(torch.allclose(log_qt, log_qp, atol=1e-6), "beta=0 must be q+")
    _assert(float(kl_logp(log_qt, log_qp)) < 1e-8, "KL(qT||q+)=0 at beta=0")
    print("ok beta0")


def test_js_shift_zero() -> None:
    torch.manual_seed(1)
    s = torch.randn(5)
    log_p = scores_to_logq(s, 1.0)
    log_q = scores_to_logq(s + 3.0, 1.0)
    js = js_divergence(log_p, log_q)
    _assert(js < 1e-6, f"constant shift JS should be 0, got {js}")
    s2 = s + torch.tensor([4.0, 0, 0, 0, 0])
    js2 = js_divergence(scores_to_logq(s, 1.0), scores_to_logq(s2, 1.0))
    _assert(js2 > 0.01, f"moved mass should have JS>0, got {js2}")
    print("ok js_shift", js2)


def test_length_norm_ranking() -> None:
    # Two actions: same total logp, different lengths → shorter wins after mean.
    actions = ["go to a", "go to fridge 1 extra"]
    s0 = [-2.0, -2.0]
    # UCE prefers the longer string if we used sum; mean prefers equal.
    # Make per-token means differ: short -1.0, long -0.4 → long wins.
    s_plus = [-1.0, -0.4]
    rec = summarize_pair(
        actions,
        s0,
        s_plus,
        executed="go to a",
        first_cmd_token_0=10,
        first_cmd_token_plus=10,
    )
    _assert(rec["top1_uce"] == "go to fridge 1 extra", rec)
    _assert(rec["whole_action_top1_diff"] is True, rec)
    _assert(rec["token_same_action_diff"] is True, rec)
    _assert(rec["rank_base"] == 1, rec)
    _assert(rec["rank_uce"] == 2, rec)
    _assert(rec["rank_change"] == -1, rec)
    print("ok length_norm_rank")


class _Tok:
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        table = {10: "ACTION", 11: ":", 12: " look", 13: " go", 14: " to"}
        return "".join(table[int(i)] for i in ids)


def test_colon_mask_keeps_fused_verb() -> None:
    from uce_whole_action import command_content_mask

    tok = _Tok()
    look = command_content_mask(tok, [10, 11, 12])
    _assert(look == [False, False, True], look)
    go = command_content_mask(tok, [10, 11, 13, 14])
    _assert(go == [False, False, True, True], go)
    print("ok colon_mask")


def test_extract_and_match() -> None:
    acts = ["go to fridge 1", "take egg from fridge 1"]
    _assert(extract_command("ACTION: go to fridge 1") == "go to fridge 1", "extract")
    _assert(match_admissible("GO TO FRIDGE 1", acts) == "go to fridge 1", "match")
    print("ok extract_match")


def test_reach_zero_direction() -> None:
    s = torch.zeros(4)
    kl = action_kl_at_beta(s, s + 1.0, 4.0)
    _assert(kl < 1e-6, f"shift-only action KL should be 0, got {kl}")
    print("ok reach_zero")


def main() -> None:
    test_workflow_inverse()
    test_beta0_is_uce()
    test_js_shift_zero()
    test_length_norm_ranking()
    test_colon_mask_keeps_fused_verb()
    test_extract_and_match()
    test_reach_zero_direction()
    print("ALL_MATH_OK")


if __name__ == "__main__":
    main()
