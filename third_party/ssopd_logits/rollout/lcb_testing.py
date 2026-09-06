"""Minimal LiveCodeBench test runner (no pyext / verl dependency)."""

from __future__ import annotations

import ast
import io
import json
import signal
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any
from unittest.mock import mock_open, patch


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame) -> None:  # noqa: ARG001
    raise _Timeout("time limit exceeded")


def _truncate(text: str, limit: int = 200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalize_dict_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            nk: Any = int(k) if isinstance(k, str) and k.isdigit() else k
            out[nk] = _normalize_dict_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_dict_keys(x) for x in obj]
    return obj


def _outputs_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, list) and expected:
        if actual == expected[0]:
            return True
        try:
            if isinstance(actual, list) and actual and isinstance(actual[0], tuple):
                if [list(x) for x in actual] == expected[0]:
                    return True
        except Exception:
            pass
    return False


def _compare_stdin_output(actual_lines: list[str], expected_raw: str) -> bool:
    expected_lines = expected_raw.splitlines()
    if actual_lines == expected_lines:
        return True
    if actual_lines == [expected_raw]:
        return True
    stripped_actual = [ln.strip() for ln in actual_lines if ln.strip()]
    stripped_expected = [ln.strip() for ln in expected_lines if ln.strip()]
    if stripped_actual == stripped_expected:
        return True
    if stripped_actual == [expected_raw.strip()]:
        return True
    return False


def _compile_callable(code: str, fn_name: str | None):
    namespace: dict[str, Any] = {}
    prelude = (
        "import sys, json, math, itertools, collections, heapq, bisect, "
        "functools, random, string, re\n"
        "from typing import *\n"
    )
    exec(prelude + code, namespace)  # noqa: S102
    if fn_name:
        if "Solution" in namespace:
            return getattr(namespace["Solution"](), fn_name)
        if fn_name in namespace:
            return namespace[fn_name]
        raise AttributeError(f"callable {fn_name!r} not found")
    return namespace.get("code")


def _wrap_stdin_code(code: str) -> str:
    try:
        tree = ast.parse(code)
        last = tree.body[-1]
        if isinstance(last, ast.If):
            cond = ast.unparse(last.test).strip()
            if cond == "__name__ == '__main__'":
                code = ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last.body)
    except Exception:
        pass

    lines = code.split("\n")
    body: list[str] = []
    for line in lines:
        if line.startswith("from ") or line.startswith("import "):
            body.append(line)
        else:
            body.append("\t" + line)
    wrapped = (
        "import sys\n"
        "def code():\n"
        + "\n".join(body)
        + "\n"
    )
    return wrapped


def _call_stdin_method(method, stdin_text: str) -> str:
    if isinstance(stdin_text, list):
        stdin_text = "\n".join(stdin_text)
    line_iter = iter(stdin_text.split("\n"))

    @patch("builtins.open", mock_open(read_data=stdin_text))
    @patch("sys.stdin", StringIO(stdin_text))
    @patch("sys.stdin.readline", lambda *args: next(line_iter))
    @patch("sys.stdin.readlines", lambda *args: stdin_text.split("\n"))
    @patch("sys.stdin.read", lambda *args: stdin_text)
    def _inner(_method):
        try:
            return _method()
        except SystemExit:
            return None

    return _inner(method)


def _run_stdin_case(code: str, stdin_text: str, expected: str, timeout: int) -> tuple[bool | int, dict[str, Any]]:
    wrapped = _wrap_stdin_code(code)
    namespace: dict[str, Any] = {}
    try:
        exec(wrapped, namespace)  # noqa: S102
        fn = namespace["code"]
    except Exception as exc:
        return -2, {"error": repr(exc), "traceback": traceback.format_exc()}

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    buf = io.StringIO()
    try:
        signal.alarm(timeout)
        with redirect_stdout(buf):
            _call_stdin_method(fn, stdin_text)
        signal.alarm(0)
    except _Timeout:
        signal.alarm(0)
        return -1, {"error": "time limit exceeded"}
    except StopIteration:
        signal.alarm(0)
        return -1, {"error": "stdin exhausted"}
    except Exception as exc:
        signal.alarm(0)
        return -1, {"error": repr(exc), "traceback": traceback.format_exc()}
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    actual = buf.getvalue().splitlines()
    if _compare_stdin_output(actual, expected):
        return True, {}
    return False, {
        "output": _truncate(buf.getvalue()),
        "expected": _truncate(expected),
        "inputs": _truncate(stdin_text),
        "error_message": "Wrong Answer",
    }


def _run_call_case(
    method,
    input_text: str,
    expected_raw: str,
    timeout: int,
) -> tuple[bool | int, dict[str, Any]]:
    try:
        args = [json.loads(line) for line in input_text.split("\n") if line.strip()]
        expected = json.loads(expected_raw)
    except Exception as exc:
        return -2, {"error": f"bad test case json: {exc!r}"}

    args = [_normalize_dict_keys(a) for a in args]
    expected = _normalize_dict_keys(expected)

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(timeout)
        output = method(*args)
        signal.alarm(0)
    except _Timeout:
        signal.alarm(0)
        return -1, {"error": "time limit exceeded"}
    except Exception as exc:
        signal.alarm(0)
        return -1, {"error": repr(exc), "traceback": traceback.format_exc()}
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    if _outputs_match(output, expected):
        return True, {}
    return False, {
        "output": _truncate(json.dumps(output, default=str)),
        "expected": _truncate(expected_raw),
        "inputs": _truncate(input_text),
        "error_message": "Wrong Answer",
    }


def run_test(
    in_outs: dict[str, Any],
    *,
    test: str,
    timeout: int = 6,
    debug: bool = False,  # noqa: ARG001
) -> tuple[list[bool | int], dict[str, Any]]:
    """Run LCB-style tests; returns per-case bool/-1/-2 and failure metadata."""
    inputs = in_outs.get("inputs") or []
    outputs = in_outs.get("outputs") or []
    fn_name = in_outs.get("fn_name")
    if not inputs or len(inputs) != len(outputs):
        return [], {"error": "invalid test_cases"}

    results: list[bool | int] = []
    fn_name = fn_name or None
    call_based = fn_name is not None

    method = None
    if call_based:
        try:
            method = _compile_callable(test, fn_name)
        except Exception as exc:
            return [-2], {"error": repr(exc), "traceback": traceback.format_exc()}

    for index, inp in enumerate(inputs):
        exp = outputs[index]
        if call_based:
            ok, meta = _run_call_case(method, str(inp), str(exp), timeout)
        else:
            ok, meta = _run_stdin_case(test, str(inp), str(exp), timeout)
        results.append(ok)
        if ok is not True:
            return results, meta

    return results, {}


if sys.platform == "win32":
    # SIGALRM unavailable on Windows; tests still run without per-case alarm.
    def run_test(  # type: ignore[no-redef]
        in_outs: dict[str, Any],
        *,
        test: str,
        timeout: int = 6,
        debug: bool = False,
    ) -> tuple[list[bool | int], dict[str, Any]]:
        inputs = in_outs.get("inputs") or []
        outputs = in_outs.get("outputs") or []
        fn_name = in_outs.get("fn_name")
        if not inputs or len(inputs) != len(outputs):
            return [], {"error": "invalid test_cases"}

        results: list[bool | int] = []
        call_based = fn_name is not None
        method = None
        if call_based:
            try:
                method = _compile_callable(test, fn_name)
            except Exception as exc:
                return [-2], {"error": repr(exc), "traceback": traceback.format_exc()}

        for index, inp in enumerate(inputs):
            exp = outputs[index]
            if call_based:
                ok, meta = _run_call_case(method, str(inp), str(exp), timeout)
            else:
                ok, meta = _run_stdin_case(test, str(inp), str(exp), timeout)
            results.append(ok)
            if ok is not True:
                return results, meta
        return results, {}
