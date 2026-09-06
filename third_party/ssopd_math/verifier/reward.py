"""Completion verification rewards for SSOPD math."""

from __future__ import annotations

from typing import Any

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
PARSE_FAILURE = "PARSE_FAILURE"


def verify_completion(text: str, gold_answer: str) -> dict[str, Any]:
    """Parse \\boxed{} answer and compare to gold.

    Returns dict with verification_status in {POSITIVE, NEGATIVE, PARSE_FAILURE}
    and parse_ok bool.
    """
    from ssopd_math.verifier.math_answer import (
        extract_boxed_answer,
        is_equiv,
        parse_final_answer,
    )

    parsed = None
    try:
        parsed = extract_boxed_answer(text)
    except Exception:
        parsed = None
    if parsed is None:
        try:
            parsed = parse_final_answer(text)
        except Exception:
            parsed = None

    if parsed is None or str(parsed).strip() == "":
        return {
            "verification_status": PARSE_FAILURE,
            "parse_ok": False,
            "parsed_answer": None,
            "gold_answer": gold_answer,
        }

    ok = False
    try:
        ok = bool(is_equiv(str(parsed), str(gold_answer)))
    except Exception:
        ok = str(parsed).strip() == str(gold_answer).strip()

    return {
        "verification_status": POSITIVE if ok else NEGATIVE,
        "parse_ok": True,
        "parsed_answer": parsed,
        "gold_answer": gold_answer,
    }
