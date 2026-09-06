"""Sandbox execution for LiveCodeBench submissions."""

from __future__ import annotations

import multiprocessing
import re
from typing import Any


def _strip_thinking(text: str) -> str:
    tag = "redacted_thinking"
    pattern = rf"<{tag}>.*?</{tag}>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str | None:
    if not text:
        return None
    for src in (text, _strip_thinking(text)):
        if "```python" in src:
            block = src.split("```python", 1)[-1].split("```", 1)[0]
            if block.strip():
                return block.strip()
        if "```" in src:
            block = src.split("```", 1)[-1].split("```", 1)[0]
            if block.strip():
                return block.strip()
    cleaned = _strip_thinking(text)
    for marker in ("class Solution", "def solve(", "if __name__"):
        idx = cleaned.rfind(marker)
        if idx >= 0:
            snippet = cleaned[idx:].strip()
            if snippet:
                return snippet
    return None


def _run_test_worker(in_outs: dict[str, Any], code: str, debug: bool, out_q) -> None:
    try:
        from rollout.lcb_testing import run_test

        res, metadata = run_test(in_outs, test=code, debug=debug, timeout=6)
        out_q.put((res, metadata))
    except Exception as exc:
        out_q.put((None, {"error": repr(exc)}))


def _mp_context():
    if __import__("sys").platform == "win32":
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context("fork")


def run_lcb_tests(
    test_cases: dict[str, Any],
    code: str,
    *,
    timeout_per_case: int = 6,
    global_timeout: int | None = None,
    use_subprocess: bool = True,
) -> dict[str, Any]:
    """Execute code against LCB test cases; return rich feedback for peer contrast."""
    n = len(test_cases.get("inputs") or [])
    if not code or not n:
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_total": n,
            "first_fail_index": 0 if n else None,
            "feedback": "no_code_or_no_tests",
            "results": [],
            "metadata": {},
        }

    if not use_subprocess:
        return _format_results(*_run_inline(test_cases, code, timeout_per_case))

    gt = global_timeout or max(30, (timeout_per_case + 1) * n + 10)
    ctx = _mp_context()
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(
        target=_run_test_worker,
        args=(test_cases, code, False, q),
    )
    p.start()
    p.join(timeout=gt)
    if p.is_alive():
        p.kill()
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_total": n,
            "first_fail_index": 0,
            "feedback": "global_timeout",
            "results": [-1] * n,
            "metadata": {},
        }
    if q.empty():
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_total": n,
            "first_fail_index": 0,
            "feedback": "sandbox_empty_result",
            "results": [-1] * n,
            "metadata": {},
        }
    results, metadata = q.get()
    if results is None:
        err = metadata.get("error", "sandbox_error")
        tb = metadata.get("traceback", "")
        feedback = f"execution_error: {err}"
        if tb:
            feedback += f"\n{tb[:800]}"
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_total": n,
            "first_fail_index": 0,
            "feedback": feedback,
            "results": [],
            "metadata": metadata,
        }

    return _format_results(results, metadata, n)


def _run_inline(
    test_cases: dict[str, Any],
    code: str,
    timeout_per_case: int,
) -> tuple[list[Any] | None, dict[str, Any]]:
    from rollout.lcb_testing import run_test

    try:
        return run_test(test_cases, test=code, timeout=timeout_per_case)
    except Exception as exc:
        return None, {"error": repr(exc)}


def _format_results(
    results: list[Any] | None,
    metadata: dict[str, Any],
    n: int | None = None,
) -> dict[str, Any]:
    if results is None:
        err = metadata.get("error", "sandbox_error")
        tb = metadata.get("traceback", "")
        feedback = f"execution_error: {err}"
        if tb:
            feedback += f"\n{tb[:800]}"
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_total": n or 0,
            "first_fail_index": 0,
            "feedback": feedback,
            "results": [],
            "metadata": metadata,
        }

    n_total = n if n is not None else len(results)
    bool_results = [r is True for r in results]
    n_pass = sum(bool_results)
    first_fail = next((i for i, ok in enumerate(bool_results) if not ok), None)
    if n_pass == n_total and n_total > 0:
        feedback = f"All {n_total} tests passed."
    else:
        fi = first_fail if first_fail is not None else 0
        feedback = f"Failed test {fi + 1}/{n_total}."
        if metadata and isinstance(metadata, dict):
            if metadata.get("error"):
                feedback += f" Error: {metadata.get('error')}"
            if metadata.get("traceback"):
                feedback += f"\n{str(metadata.get('traceback'))[:600]}"

    return {
        "all_pass": n_pass == n_total and n_total > 0,
        "n_pass": n_pass,
        "n_total": n_total,
        "first_fail_index": first_fail,
        "feedback": feedback,
        "results": results,
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def format_env_feedback(eval_out: dict[str, Any]) -> str:
    """Structured environment feedback appended after submission (MOPD rich feedback)."""
    lines = [
        "[ENVIRONMENT FEEDBACK]",
        eval_out.get("feedback") or "No feedback.",
        f"Tests passed: {eval_out.get('n_pass', 0)}/{eval_out.get('n_total', 0)}",
        "[END ENVIRONMENT FEEDBACK]",
    ]
    return "\n".join(lines)
