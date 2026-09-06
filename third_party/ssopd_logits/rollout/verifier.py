"""Decision-level verification: tool vs argument correctness separated."""

from __future__ import annotations

from typing import Any

from environments.tool_registry import resolve_tool_name


DECISION_LABELS = ("POSITIVE", "NEGATIVE", "PARSE_FAILURE", "UNSCORABLE")


def _norm_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v).strip().lower()


def argument_exact_match(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    if set(gold.keys()) != set(pred.keys()):
        # Allow pred to have only gold keys subset equality on shared
        if not gold:
            return not pred
    for k, gv in gold.items():
        if k not in pred:
            return False
        if _norm_val(pred[k]) != _norm_val(gv):
            return False
    return True


def argument_semantic_match(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    """Loose match: all gold keys present with string-equal normalized values."""
    if not gold:
        return True
    for k, gv in gold.items():
        if k not in pred:
            return False
        if _norm_val(pred[k]) != _norm_val(gv):
            return False
    return True


def argument_options_match(
    options: dict[str, list[Any]] | None,
    pred: dict[str, Any],
) -> bool:
    """BFCL-style match: each gold key has a list of acceptable values.

    Empty-string options allow the argument to be omitted.
    """
    if not options:
        return True
    for k, allowed in options.items():
        allowed_list = list(allowed or [])
        allow_missing = any(a == "" or a is None for a in allowed_list)
        concrete = [a for a in allowed_list if a != "" and a is not None]
        if k not in pred:
            if allow_missing or not concrete:
                continue
            return False
        pv = _norm_val(pred[k])
        if concrete and any(pv == _norm_val(a) for a in concrete):
            continue
        if allow_missing and pv == "":
            continue
        return False
    return True


def verify_decision_fields(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return decision verification fields without mutating state."""
    parse_ok = bool(action.get("parse_ok", action.get("tool_name") is not None))
    valid_tools = list(state.get("valid_tools") or [])
    is_bfcl = bool(task and task.get("source") == "bfcl")
    pred_tool = resolve_tool_name(
        action.get("tool_name"),
        valid_tools or None,
        keep_unknown=False,
    )
    if pred_tool is None and action.get("tool_name") and parse_ok:
        raw = str(action["tool_name"]).strip()
        if is_bfcl:
            # Open vocabulary: wrong invented names count as NEGATIVE, not PARSE_FAILURE.
            pred_tool = raw
        elif valid_tools and any(t.lower() == raw.lower() for t in valid_tools):
            for t in valid_tools:
                if t.lower() == raw.lower():
                    pred_tool = t
                    break
    pred_args = dict(action.get("arguments") or {})
    gold_tool = state.get("gold_tool")
    gold_args = dict(state.get("gold_arguments") or {})
    arg_options = state.get("argument_options")
    if arg_options is None and task is not None:
        plan = task.get("gold_plan") or []
        step_i = int(state.get("step_index", 0))
        if 0 <= step_i < len(plan):
            arg_options = plan[step_i].get("argument_options")

    premature = False
    premature_tool = None
    if task is not None:
        premature_tool = task.get("premature_downstream_tool")
    if premature_tool and pred_tool == premature_tool and state.get("step_index", 0) == 0:
        # Calling the downstream tool before completing prior dependency
        if gold_tool != premature_tool:
            premature = True

    if not parse_ok or pred_tool is None:
        return {
            "parse_ok": False,
            "predicted_tool": pred_tool,
            "predicted_arguments": pred_args,
            "gold_tool": gold_tool,
            "gold_arguments": gold_args,
            "tool_correct": False,
            "argument_exact": False,
            "argument_semantic": False,
            "decision_label": "PARSE_FAILURE",
            "failure_reason": "parse_failure",
            "is_premature_downstream_action": premature,
        }

    if gold_tool is None:
        return {
            "parse_ok": True,
            "predicted_tool": pred_tool,
            "predicted_arguments": pred_args,
            "gold_tool": None,
            "gold_arguments": gold_args,
            "tool_correct": False,
            "argument_exact": False,
            "argument_semantic": False,
            "decision_label": "UNSCORABLE",
            "failure_reason": "missing_gold",
            "is_premature_downstream_action": premature,
        }

    is_lcb = bool(task and task.get("source") == "lcb")
    if is_lcb:
        tests_pass = action.get("tests_pass")
        if tests_pass is True:
            return {
                "parse_ok": True,
                "predicted_tool": pred_tool,
                "predicted_arguments": pred_args,
                "gold_tool": gold_tool,
                "gold_arguments": gold_args,
                "tool_correct": True,
                "argument_exact": True,
                "argument_semantic": True,
                "decision_label": "POSITIVE",
                "failure_reason": None,
                "is_premature_downstream_action": False,
            }
        if tests_pass is False:
            return {
                "parse_ok": True,
                "predicted_tool": pred_tool,
                "predicted_arguments": pred_args,
                "gold_tool": gold_tool,
                "gold_arguments": gold_args,
                "tool_correct": False,
                "argument_exact": False,
                "argument_semantic": False,
                "decision_label": "NEGATIVE",
                "failure_reason": action.get("failure_reason") or "tests_failed",
                "is_premature_downstream_action": False,
            }

    tool_correct = pred_tool == gold_tool
    if tool_correct and arg_options:
        arg_exact = argument_options_match(arg_options, pred_args)
        arg_sem = arg_exact
    else:
        arg_exact = argument_exact_match(gold_args, pred_args) if tool_correct else False
        arg_sem = argument_semantic_match(gold_args, pred_args) if tool_correct else False

    if premature and not tool_correct:
        label = "NEGATIVE"
        reason = "premature_downstream_action"
    elif tool_correct:
        label = "POSITIVE"
        reason = None
    else:
        label = "NEGATIVE"
        reason = "wrong_tool"

    return {
        "parse_ok": True,
        "predicted_tool": pred_tool,
        "predicted_arguments": pred_args,
        "gold_tool": gold_tool,
        "gold_arguments": gold_args,
        "tool_correct": tool_correct,
        "argument_exact": arg_exact,
        "argument_semantic": arg_sem,
        "decision_label": label,
        "failure_reason": reason,
        "is_premature_downstream_action": premature,
    }
