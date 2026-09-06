#!/usr/bin/env python3
"""Unit tests for UCE counterfactual logit steering (CPU, no model)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uce_counterfactual_logit import (  # noqa: E402
    extract_workflow_block,
    guided_logp,
    is_protocol_text,
    kl_at_beta,
    kl_logp,
    log_softmax_temp,
    next_token_is_protocol,
    solve_beta,
    workflow_sha256,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_protocol_mask() -> None:
    _assert(next_token_is_protocol("") is True, "empty")
    _assert(next_token_is_protocol("ACT") is True, "partial ACTION")
    _assert(next_token_is_protocol("ACTION:") is True, "colon still protocol")
    _assert(next_token_is_protocol("ACTION: ") is False, "space done → command")
    _assert(next_token_is_protocol("ACTION: go") is False, "command started")
    _assert(is_protocol_text("ACTION: ") is True, "text still format")
    _assert(is_protocol_text("ACTION: go") is False, "text has command")
    print("ok protocol_mask")


def test_constant_shift_invariant() -> None:
    torch.manual_seed(0)
    z0 = torch.randn(64)
    zp = z0 + 5.0
    k = kl_at_beta(z0, zp, 4.0)
    _assert(k < 1e-6, f"shift should be ~0 KL, got {k}")
    beta, ach, kmax, unreach = solve_beta(z0, zp, 0.02)
    _assert(unreach is True, "unreachable when distributions equal")
    _assert(beta == 0.0, f"zero direction must beta=0, got {beta}")
    _assert(ach == 0.0, f"ach={ach}")
    _assert(kmax < 1e-6, f"kmax={kmax}")
    print("ok constant_shift")


def test_kl_bisection_monotone() -> None:
    torch.manual_seed(1)
    z0 = torch.randn(128)
    zp = z0 + torch.randn(128) * 2.0
    kls = [kl_at_beta(z0, zp, b) for b in (0.0, 0.5, 1.0, 2.0, 4.0)]
    for a, b in zip(kls, kls[1:]):
        _assert(b + 1e-6 >= a, f"KL not monotone: {kls}")
    target = 0.05
    beta, ach, kmax, unreach = solve_beta(z0, zp, target, bisection_steps=12)
    if kmax >= target - 1e-4:
        _assert(abs(ach - target) < 0.01, f"ach={ach} target={target} kmax={kmax}")
        _assert(unreach is False, "should reach")
    beta2, ach2, _, un2 = solve_beta(z0, zp, 50.0, beta_max=4.0)
    _assert(un2 is True, "huge target unreachable")
    _assert(abs(beta2 - 4.0) < 1e-9, f"beta_max used, got {beta2}")
    print("ok kl_bisection", {"beta": beta, "ach": ach, "kmax": kmax})


def test_beta0_equals_uce() -> None:
    torch.manual_seed(2)
    z0 = torch.randn(32)
    zp = torch.randn(32)
    log_pt = guided_logp(z0, zp, 0.0)
    lp = log_softmax_temp(zp, 1.0)
    _assert(torch.allclose(log_pt, lp, atol=1e-6), "beta=0 must equal UCE")
    _assert(float(kl_logp(log_pt, lp)) < 1e-8, "KL(pT||pUCE)=0 at beta=0")
    print("ok beta0")


def test_nan_safe_equal() -> None:
    z = torch.zeros(16)
    beta, ach, kmax, unreach = solve_beta(z, z, 0.05)
    _assert(not any(map(lambda x: x != x, [beta, ach, kmax])), "NaN")
    _assert(unreach is True, "equal logits unreachable")
    print("ok nan_safe")


def test_workflow_extract() -> None:
    block = (
        "Suggested workflow from a similar solved instance "
        "(adapt names to the current admissible list):\n"
        "1. go to armchair\n"
        "2. take box from armchair"
    )
    prompt = block + "\n\nTask: place a salt shaker\n\nObservation: hi"
    got = extract_workflow_block(prompt)
    _assert(workflow_sha256(got) == workflow_sha256(block), f"hash mismatch\n{got!r}")
    print("ok workflow_extract")


def main() -> None:
    test_protocol_mask()
    test_constant_shift_invariant()
    test_kl_bisection_monotone()
    test_beta0_equals_uce()
    test_nan_safe_equal()
    test_workflow_extract()
    print("ALL_MATH_OK")


if __name__ == "__main__":
    main()
