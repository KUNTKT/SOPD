"""Trajectory and decision-point schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TRAJECTORY_REQUIRED_KEYS = (
    "task_id",
    "trajectory_id",
    "dataset_split",
    "prompt_text",
    "full_text",
    "actions",
    "observations",
    "decision_points",
    "final_reward",
    "episode_success",
    "token_count",
    "tool_call_count",
    "trajectory_length",
    "termination_reason",
    "model_name",
    "tokenizer_hash",
    "prompt_hash",
    "random_seed",
)

DECISION_REQUIRED_KEYS = (
    "decision_id",
    "step_index",
    "prefix_text",
    "state",
    "state_fingerprint",
    "predicted_tool",
    "predicted_arguments",
    "gold_tool",
    "gold_arguments",
    "tool_correct",
    "argument_exact",
    "argument_semantic",
    "parse_ok",
    "decision_label",
    "failure_reason",
    "is_premature_downstream_action",
    "valid_tools",
)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def state_fingerprint(state: dict[str, Any]) -> str:
    """Canonical fingerprint of decision-comparable state fields."""
    payload = {
        "task_id": state.get("task_id"),
        "step_index": state.get("step_index"),
        "gold_tool": state.get("gold_tool"),
        "completed_tools": state.get("completed_tools", []),
        "subgoal": state.get("subgoal"),
        # observation text hashed to keep fingerprint stable-size
        "observation_hash": hashlib.sha256(
            str(state.get("observation", "")).encode("utf-8")
        ).hexdigest()[:16],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def empty_trajectory(**kwargs: Any) -> dict[str, Any]:
    base = {
        "task_id": "",
        "trajectory_id": "",
        "dataset_split": "train",
        "prompt_text": "",
        "full_text": "",
        "actions": [],
        "observations": [],
        "decision_points": [],
        "final_reward": 0.0,
        "episode_success": False,
        "token_count": 0,
        "tool_call_count": 0,
        "trajectory_length": 0,
        "termination_reason": "",
        "model_name": "",
        "tokenizer_hash": "",
        "prompt_hash": "",
        "random_seed": 0,
    }
    base.update(kwargs)
    return base


def empty_decision(**kwargs: Any) -> dict[str, Any]:
    base = {
        "decision_id": "",
        "step_index": 0,
        "prefix_text": "",
        "state": {},
        "state_fingerprint": "",
        "predicted_tool": None,
        "predicted_arguments": {},
        "gold_tool": None,
        "gold_arguments": {},
        "tool_correct": False,
        "argument_exact": False,
        "argument_semantic": False,
        "parse_ok": False,
        "decision_label": "UNSCORABLE",
        "failure_reason": None,
        "is_premature_downstream_action": False,
        "valid_tools": [],
    }
    base.update(kwargs)
    return base


def validate_schema(obj: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [k for k in required if k not in obj]


def validate_trajectory(traj: dict[str, Any]) -> list[str]:
    missing = validate_schema(traj, TRAJECTORY_REQUIRED_KEYS)
    for i, dp in enumerate(traj.get("decision_points") or []):
        dp_missing = validate_schema(dp, DECISION_REQUIRED_KEYS)
        missing.extend([f"decision_points[{i}].{k}" for k in dp_missing])
    return missing
