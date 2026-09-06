"""Deterministic synthetic Tool-Selection environment."""

from __future__ import annotations

import copy
from typing import Any

from environments.base_env import AgentEnvironment, StepOutput
from environments.task_generator import build_agent_prompt
from environments.tool_registry import TOOL_NAMES, resolve_tool_name
from rollout.parser import parse_tool_call
from rollout.verifier import verify_decision_fields


class SyntheticToolEnv(AgentEnvironment):
    def __init__(self) -> None:
        self.task: dict[str, Any] | None = None
        self.step_index: int = 0
        self.history: list[dict[str, Any]] = []
        self.observations: list[str] = []
        self._terminal: bool = False
        self._last_observation: str = ""

    def reset(self, task: dict[str, Any]) -> str:
        self.task = copy.deepcopy(task)
        self.step_index = 0
        self.history = []
        self.observations = []
        self._terminal = False
        prompt = build_agent_prompt(self.task)
        self._last_observation = prompt
        self.observations.append(prompt)
        return prompt

    def current_state(self) -> dict[str, Any]:
        assert self.task is not None
        plan = self.task["gold_plan"]
        gold = plan[self.step_index] if self.step_index < len(plan) else None
        return {
            "task_id": self.task["task_id"],
            "step_index": self.step_index,
            "completed_tools": [h.get("tool_name") for h in self.history],
            "observation": self._last_observation,
            "gold_tool": gold["tool_name"] if gold else None,
            "gold_arguments": dict(gold.get("arguments", {})) if gold else {},
            "argument_options": dict(gold.get("argument_options") or {}) if gold else {},
            "subgoal": gold.get("subgoal") if gold else None,
            "valid_tools": list(self.task.get("valid_tools", TOOL_NAMES)),
            "n_steps": self.task.get("n_steps", len(plan)),
            "premature_downstream_tool": self.task.get("premature_downstream_tool"),
            "history": copy.deepcopy(self.history),
            "source": self.task.get("source"),
        }

    def valid_actions(self, state: dict[str, Any] | None = None) -> list[str]:
        if state is not None and "valid_tools" in state:
            return list(state["valid_tools"])
        if self.task is None:
            return list(TOOL_NAMES)
        return list(self.task.get("valid_tools", TOOL_NAMES))

    def parse_action(self, action: str) -> dict[str, Any]:
        parsed = parse_tool_call(action)
        return {
            "tool_name": parsed.tool_name,
            "arguments": parsed.arguments,
            "parse_ok": parsed.parse_ok,
        }

    def verify_decision(self, state: dict[str, Any], action: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(action, str):
            parsed = parse_tool_call(action)
            action_dict = {
                "tool_name": parsed.tool_name,
                "arguments": parsed.arguments,
                "parse_ok": parsed.parse_ok,
            }
        else:
            action_dict = action
        return verify_decision_fields(state, action_dict, task=self.task)

    def verify_trajectory(self, trajectory: dict[str, Any]) -> bool:
        return bool(trajectory.get("episode_success", False))

    def reward(self, trajectory: dict[str, Any]) -> float:
        return 1.0 if self.verify_trajectory(trajectory) else 0.0

    def step(self, action: str) -> StepOutput:
        if self.task is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminal:
            return StepOutput(
                observation=self._last_observation,
                tool_name=None,
                tool_arguments={},
                tool_result="",
                success=False,
                failure_reason="already_terminal",
                terminal=True,
                decision_label="UNSCORABLE",
            )

        state = self.current_state()
        parsed = parse_tool_call(action)
        action_dict = {
            "tool_name": parsed.tool_name,
            "arguments": parsed.arguments,
            "parse_ok": parsed.parse_ok,
        }
        verdict = self.verify_decision(state, action_dict)

        valid = list(self.task.get("valid_tools") or TOOL_NAMES)
        tool_name = (
            resolve_tool_name(
                parsed.tool_name,
                valid,
                keep_unknown=self.task.get("source") == "bfcl",
            )
            if parsed.parse_ok
            else None
        )
        tool_result = ""
        success = False
        failure_reason = verdict.get("failure_reason")
        terminal = False

        if not parsed.parse_ok:
            failure_reason = failure_reason or "parse_failure"
            tool_result = f"ERROR: {failure_reason}"
            terminal = True
            self._terminal = True
        elif verdict["decision_label"] == "POSITIVE":
            gold = self.task["gold_plan"][self.step_index]
            tool_result = str(gold.get("expected_result", "OK"))
            # BFCL episode success prefers AST-style arg match when options exist.
            if self.task.get("source") == "bfcl" and gold.get("argument_options"):
                success = bool(verdict.get("argument_exact"))
            else:
                success = True
            if not success and self.task.get("source") == "bfcl":
                failure_reason = "argument_mismatch"
                tool_result = f"ERROR: {failure_reason}"
                terminal = True
                self._terminal = True
            else:
                self.history.append(
                    {
                        "tool_name": tool_name,
                        "arguments": dict(parsed.arguments),
                        "result": tool_result,
                        "step_index": self.step_index,
                    }
                )
                self.step_index += 1
                if self.step_index >= len(self.task["gold_plan"]):
                    terminal = True
                    self._terminal = True
                    tool_result = tool_result + "\nTASK_COMPLETE"
                else:
                    next_obs = (
                        f"Tool result: {tool_result}\n"
                        f"Continue the task. Remaining steps: "
                        f"{len(self.task['gold_plan']) - self.step_index}."
                    )
                    self._last_observation = next_obs
                    self.observations.append(next_obs)
        else:
            # Wrong tool or premature / bad args — episode fails for mock simplicity
            failure_reason = failure_reason or "wrong_tool"
            tool_result = f"ERROR: {failure_reason}"
            terminal = True
            self._terminal = True

        return StepOutput(
            observation=self._last_observation,
            tool_name=tool_name,
            tool_arguments=dict(parsed.arguments),
            tool_result=tool_result,
            success=success,
            failure_reason=failure_reason,
            terminal=terminal,
            parsed_ok=parsed.parse_ok,
            gold_tool=state.get("gold_tool"),
            gold_arguments=dict(state.get("gold_arguments") or {}),
            tool_correct=bool(verdict.get("tool_correct")),
            argument_exact=bool(verdict.get("argument_exact")),
            argument_semantic=bool(verdict.get("argument_semantic")),
            decision_label=str(verdict.get("decision_label", "UNSCORABLE")),
            is_premature_downstream_action=bool(verdict.get("is_premature_downstream_action")),
            local_reward=1.0 if success else 0.0,
            extra={"step_index": state["step_index"]},
        )
