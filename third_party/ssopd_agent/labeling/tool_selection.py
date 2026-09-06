"""Frozen EXP02 Tool-Selection labels. Not episode reward.

Mutually exclusive decision labels:
  POSITIVE_TOOL_SELECTION  parsed and tool == gold
  NEGATIVE_TOOL_SELECTION  parsed and tool != gold
  PARSE_FAILURE            not parsed as a known tool
  UNSCORABLE               off-plan; no unique remaining gold action

Parse failures are never folded into NEGATIVE_TOOL_SELECTION.
Argument correctness is stored separately and never flips the tool label.
"""

from __future__ import annotations

from typing import Any

from ssopd_agent.environments.tool_use_env import TOOL_SPECS, parse_tool_call

LABEL_VERSION = "tool_select_decision_v2"

POSITIVE = "POSITIVE_TOOL_SELECTION"
NEGATIVE = "NEGATIVE_TOOL_SELECTION"
PARSE_FAILURE = "PARSE_FAILURE"
UNSCORABLE = "UNSCORABLE"
DECISION_LABELS = (POSITIVE, NEGATIVE, PARSE_FAILURE, UNSCORABLE)


def episode_label(record: dict[str, Any]) -> tuple[str, str]:
    success = bool(record.get("success") or (record.get("final_reward", 0) > 0))
    if success:
        return "Y+", "episode_reward>0: every gold tool in the plan was selected"
    return "Y-", "episode_reward<=0: plan not completed with all gold tools"


def _gold_plan(record: dict[str, Any]) -> list[str]:
    steps = (record.get("task") or {}).get("steps") or []
    if steps:
        return [str(s["gold_tool"]) for s in steps]
    return [str(g) for g in (record.get("gold_tools") or [])]


def _gold_args_at(record: dict[str, Any], step_id: int) -> dict[str, Any]:
    steps = (record.get("task") or {}).get("steps") or []
    if step_id < len(steps):
        return dict(steps[step_id].get("gold_args") or {})
    return {}


def argument_flags(
    parsed_args: dict[str, Any],
    gold_args: dict[str, Any],
    tool_name: str | None,
) -> tuple[bool, bool]:
    required = list(TOOL_SPECS.get(tool_name or "", {}).get("args") or gold_args.keys())
    if not required:
        required = list(gold_args.keys())
    valid = bool(required) and all(k in parsed_args for k in required)
    exact = True
    if not gold_args:
        exact = False
    else:
        for key, val in gold_args.items():
            if key not in parsed_args:
                exact = False
                break
            if str(parsed_args[key]).strip().lower() != str(val).strip().lower():
                exact = False
                break
    return valid, exact


def _is_premature(selected: str, gold_now: str, plan: list[str], step_id: int) -> bool:
    if selected == gold_now:
        return False
    later = plan[step_id + 1 :]
    return selected in later


def label_decision(
    *,
    parsed_ok: bool,
    selected_tool: str | None,
    gold_tool: str,
    plan: list[str],
    step_id: int,
    on_plan: bool,
) -> tuple[str, str | None, str]:
    if not on_plan:
        return (
            UNSCORABLE,
            "unscoreable_off_plan",
            "prior decision left the gold plan; no unique counterfactual gold remains",
        )
    if not parsed_ok or not selected_tool or selected_tool not in TOOL_SPECS:
        return (
            PARSE_FAILURE,
            "parse_error",
            f"action did not parse as a known tool (got {selected_tool!r}); gold={gold_tool}",
        )
    if selected_tool == gold_tool:
        return (
            POSITIVE,
            None,
            f"parsed tool {selected_tool} equals gold {gold_tool}",
        )
    if _is_premature(selected_tool, gold_tool, plan, step_id):
        return (
            NEGATIVE,
            "premature_downstream_action",
            f"parsed {selected_tool} skips gold {gold_tool} and jumps to a later plan tool",
        )
    return (
        NEGATIVE,
        "wrong_tool",
        f"parsed {selected_tool} != gold {gold_tool}",
    )


def label_record(record: dict[str, Any]) -> dict[str, Any]:
    y_label, y_rationale = episode_label(record)
    plan = _gold_plan(record)
    golds = list(record.get("gold_tools") or plan)
    selected_stored = list(record.get("selected_tools") or [])
    actions = list(record.get("actions") or [])

    decisions: list[dict[str, Any]] = []
    on_plan = True
    for t in range(len(actions)):
        gold = str(golds[t] if t < len(golds) else (plan[t] if t < len(plan) else ""))
        parsed_name, parsed_args, parsed_ok = parse_tool_call(actions[t])
        # Prefer live reparse; fall back to stored name if reparse fails but store had a known tool.
        selected = parsed_name if parsed_ok else (
            selected_stored[t] if t < len(selected_stored) and selected_stored[t] in TOOL_SPECS else parsed_name
        )
        parsed_for_label = bool(selected in TOOL_SPECS) if selected else False
        label, fail_reason, rationale = label_decision(
            parsed_ok=parsed_for_label,
            selected_tool=selected if parsed_for_label else None,
            gold_tool=gold,
            plan=plan,
            step_id=t,
            on_plan=on_plan,
        )
        gold_args = _gold_args_at(record, t)
        arg_valid, arg_exact = argument_flags(parsed_args, gold_args, selected if parsed_for_label else None)
        if label != POSITIVE:
            # Args only evaluated when a known tool was parsed.
            if label == PARSE_FAILURE:
                arg_valid, arg_exact = False, False
        if label == POSITIVE and fail_reason is None and not arg_exact:
            rationale = rationale + "; argument_exact=False (ignored for tool label)"

        decisions.append(
            {
                "task_id": record["task_id"],
                "trajectory_id": record["trajectory_id"],
                "step_id": t,
                "kind": record.get("kind"),
                "gold_tool": gold,
                "gold_plan": plan,
                "selected_tool": selected,
                "parsed": parsed_for_label,
                "tool_selection_label": label,
                "argument_valid": bool(arg_valid),
                "argument_exact": bool(arg_exact),
                "downstream_success": y_label == "Y+",
                "failure_reason": fail_reason,
                "rationale": rationale,
                "episode_label": y_label,
                "episode_success": y_label == "Y+",
                "n_steps_gold": int(record.get("n_steps_gold") or len(plan) or 1),
                "on_plan": on_plan,
                "label_version": LABEL_VERSION,
            }
        )
        if label in {NEGATIVE, PARSE_FAILURE}:
            on_plan = False
        elif label == UNSCORABLE:
            on_plan = False

    counts = {k: sum(d["tool_selection_label"] == k for d in decisions) for k in DECISION_LABELS}
    episode = {
        "task_id": record["task_id"],
        "trajectory_id": record["trajectory_id"],
        "label": y_label,
        "rationale": y_rationale,
        "final_reward": record.get("final_reward"),
        "success": y_label == "Y+",
        "kind": record.get("kind"),
        "n_decisions": len(decisions),
        "n_positive": counts[POSITIVE],
        "n_negative": counts[NEGATIVE],
        "n_parse_failure": counts[PARSE_FAILURE],
        "n_unscorable": counts[UNSCORABLE],
        "termination_reason": record.get("termination_reason"),
        "label_version": LABEL_VERSION,
    }
    return {"episode": episode, "decisions": decisions}


def episode_projected_decision_label(episode_label_value: str) -> str:
    """Forbidden primary rule: copy episode outcome onto every decision."""
    return POSITIVE if episode_label_value == "Y+" else NEGATIVE


def parse_rate_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Clarify EXP01 parse-rate denominators. No relabeling of EXP01 PASS."""
    n_traj = len(records)
    n_steps = 0
    n_step_parsed = 0
    n_traj_all_parsed = 0
    n_traj_first_parsed = 0
    n_traj_stored_parse_ok = 0
    n_term_parse = 0
    for rec in records:
        actions = list(rec.get("actions") or [])
        step_ok = []
        for act in actions:
            name, _args, ok = parse_tool_call(act)
            parsed = bool(ok and name in TOOL_SPECS)
            n_steps += 1
            n_step_parsed += int(parsed)
            step_ok.append(parsed)
        if step_ok and all(step_ok):
            n_traj_all_parsed += 1
        if step_ok and step_ok[0]:
            n_traj_first_parsed += 1
        if rec.get("parse_ok"):
            n_traj_stored_parse_ok += 1
        if rec.get("termination_reason") == "parse_error":
            n_term_parse += 1
    return {
        "n_trajectories": n_traj,
        "n_steps": n_steps,
        "step_parse_rate": n_step_parsed / max(n_steps, 1),
        "step_parse_ok": n_step_parsed,
        "trajectory_parse_rate": n_traj_all_parsed / max(n_traj, 1),
        "trajectory_all_steps_parsed": n_traj_all_parsed,
        "first_step_parse_rate": n_traj_first_parsed / max(n_traj, 1),
        "exp01_stored_parse_ok_rate": n_traj_stored_parse_ok / max(n_traj, 1),
        "n_termination_parse_error": n_term_parse,
        "one_minus_term_parse_over_traj": (n_traj - n_term_parse) / max(n_traj, 1),
        "note": (
            "EXP01 parse_rate=0.836 is exp01_stored_parse_ok_rate (first-step). "
            "143 parse_error is termination_reason, so (800-143)/800=0.82125 "
            "equals 1 - trajectory termination parse rate, not step parse rate."
        ),
    }


def check_invariants(episodes: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_traj: dict[str, list[dict[str, Any]]] = {}
    for d in decisions:
        by_traj.setdefault(d["trajectory_id"], []).append(d)

    n_parse_as_neg = 0
    n_pos_not_gold = 0
    n_neg_is_gold = 0
    n_neg_unparsed = 0
    n_y_plus_has_neg = 0
    n_y_plus_has_parse = 0
    n_pos_in_y_minus = 0
    n_unscorable_on_plan = 0
    n_disagree_projection = 0
    n_label_outside = 0
    n_pos_arg_wrong = 0

    for ep in episodes:
        decs = sorted(by_traj.get(ep["trajectory_id"], []), key=lambda x: x["step_id"])
        if ep["label"] == "Y+" and any(d["tool_selection_label"] == NEGATIVE for d in decs):
            n_y_plus_has_neg += 1
        if ep["label"] == "Y+" and any(d["tool_selection_label"] == PARSE_FAILURE for d in decs):
            n_y_plus_has_parse += 1
        if ep["label"] == "Y-" and any(d["tool_selection_label"] == POSITIVE for d in decs):
            n_pos_in_y_minus += 1
        for d in decs:
            lab = d["tool_selection_label"]
            if lab not in DECISION_LABELS:
                n_label_outside += 1
            if lab == PARSE_FAILURE and lab == NEGATIVE:
                n_parse_as_neg += 1
            if lab == NEGATIVE and not d.get("parsed"):
                n_neg_unparsed += 1
            gold_match = bool(d.get("parsed") and d.get("selected_tool") == d.get("gold_tool"))
            if lab == POSITIVE and not gold_match:
                n_pos_not_gold += 1
            if lab == NEGATIVE and gold_match:
                n_neg_is_gold += 1
            if lab == UNSCORABLE and d.get("on_plan"):
                n_unscorable_on_plan += 1
            projected = episode_projected_decision_label(ep["label"])
            if lab in {POSITIVE, NEGATIVE} and lab != projected:
                n_disagree_projection += 1
            if lab == POSITIVE and not d.get("argument_exact"):
                n_pos_arg_wrong += 1

    invariants_ok = (
        n_parse_as_neg == 0
        and n_pos_not_gold == 0
        and n_neg_is_gold == 0
        and n_neg_unparsed == 0
        and n_y_plus_has_neg == 0
        and n_y_plus_has_parse == 0
        and n_unscorable_on_plan == 0
        and n_label_outside == 0
    )
    return {
        "n_episodes": len(episodes),
        "n_decisions": len(decisions),
        "n_parse_folded_into_negative": n_parse_as_neg,
        "n_positive_not_gold": n_pos_not_gold,
        "n_negative_is_gold": n_neg_is_gold,
        "n_negative_unparsed": n_neg_unparsed,
        "n_Y+_with_NEGATIVE": n_y_plus_has_neg,
        "n_Y+_with_PARSE_FAILURE": n_y_plus_has_parse,
        "n_POSITIVE_in_Y-": n_pos_in_y_minus,
        "n_unscorable_while_on_plan": n_unscorable_on_plan,
        "n_disagree_vs_episode_projection": n_disagree_projection,
        "n_positive_argument_inexact": n_pos_arg_wrong,
        "n_label_outside_schema": n_label_outside,
        "invariants_ok": invariants_ok,
    }


def per_tool_table(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from collections import defaultdict

    by_tool: dict[str, dict[str, Any]] = {}
    task_labels: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for d in decisions:
        tool = str(d.get("gold_tool") or "unknown")
        slot = by_tool.setdefault(
            tool,
            {POSITIVE: 0, NEGATIVE: 0, PARSE_FAILURE: 0, UNSCORABLE: 0, "n": 0},
        )
        lab = d["tool_selection_label"]
        slot[lab] = slot.get(lab, 0) + 1
        slot["n"] += 1
        if lab in {POSITIVE, NEGATIVE}:
            task_labels[tool][d["task_id"]].add(lab)
    out = {}
    for tool, slot in sorted(by_tool.items()):
        mixed = sum(1 for labs in task_labels[tool].values() if POSITIVE in labs and NEGATIVE in labs)
        out[tool] = {
            "positive": slot.get(POSITIVE, 0),
            "negative": slot.get(NEGATIVE, 0),
            "parse_failure": slot.get(PARSE_FAILURE, 0),
            "unscorable": slot.get(UNSCORABLE, 0),
            "n": slot["n"],
            "mixed_tasks": mixed,
        }
    return out
