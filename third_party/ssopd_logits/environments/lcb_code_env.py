"""Single-step LiveCodeBench code-generation environment."""

from __future__ import annotations

import copy
from typing import Any

from environments.base_env import AgentEnvironment, StepOutput
from environments.lcb_adapter import LCB_TOOL
from rollout.lcb_sandbox import extract_python_code, format_env_feedback, run_lcb_tests
from rollout.verifier import verify_decision_fields


class LcbCodeEnv(AgentEnvironment):
    """One-shot code submission with sandbox verifier + rich failure feedback."""

    decision_after_step = True

    def __init__(self) -> None:
        self.task: dict[str, Any] | None = None
        self.step_index: int = 0
        self.history: list[dict[str, Any]] = []
        self._terminal: bool = False
        self._last_observation: str = ""
        self._last_eval: dict[str, Any] | None = None

    def reset(self, task: dict[str, Any]) -> str:
        self.task = copy.deepcopy(task)
        self.step_index = 0
        self.history = []
        self._terminal = False
        self._last_eval = None
        prompt = str(task.get("instruction") or "")
        self._last_observation = prompt
        return prompt

    def current_state(self) -> dict[str, Any]:
        assert self.task is not None
        plan = self.task["gold_plan"]
        gold = plan[0] if plan else {}
        return {
            "task_id": self.task["task_id"],
            "step_index": self.step_index,
            "completed_tools": [h.get("tool_name") for h in self.history],
            "observation": self._last_observation,
            "gold_tool": gold.get("tool_name", LCB_TOOL),
            "gold_arguments": dict(gold.get("arguments") or {}),
            "argument_options": {},
            "subgoal": gold.get("subgoal"),
            "valid_tools": list(self.task.get("valid_tools") or [LCB_TOOL]),
            "n_steps": 1,
            "history": copy.deepcopy(self.history),
            "source": "lcb",
        }

    def parse_action(self, action: str) -> dict[str, Any]:
        code = extract_python_code(action)
        return {
            "tool_name": LCB_TOOL if code else None,
            "arguments": {"code": code or ""},
            "parse_ok": code is not None,
        }

    def verify_decision(self, state: dict[str, Any], action: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(action, str):
            action = self.parse_action(action)
        return verify_decision_fields(state, action, task=self.task)

    def verify_trajectory(self, trajectory: dict[str, Any]) -> bool:
        return bool(trajectory.get("episode_success", False))

    def reward(self, trajectory: dict[str, Any]) -> float:
        return 1.0 if self.verify_trajectory(trajectory) else 0.0

    def valid_actions(self, state: dict[str, Any] | None = None) -> list[str]:
        if self._terminal:
            return []
        return list(self.task.get("valid_tools") or [LCB_TOOL]) if self.task else [LCB_TOOL]

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

        action_dict = self.parse_action(action)
        code = str(action_dict.get("arguments", {}).get("code") or "")
        test_cases = (self.task.get("lcb") or {}).get("test_cases") or {}
        eval_out = run_lcb_tests(test_cases, code)
        self._last_eval = eval_out
        feedback = format_env_feedback(eval_out)
        parse_ok = bool(action_dict.get("parse_ok"))
        all_pass = bool(eval_out.get("all_pass"))

        if not parse_ok:
            label = "PARSE_FAILURE"
            reason = "no_python_fence"
            success = False
        elif all_pass:
            label = "POSITIVE"
            reason = None
            success = True
        else:
            label = "NEGATIVE"
            reason = "tests_failed"
            success = False

        self.history.append(
            {
                "tool_name": LCB_TOOL if parse_ok else None,
                "arguments": {"code": code},
                "eval": eval_out,
            }
        )
        self.step_index += 1
        self._terminal = True
        self._last_observation = feedback

        return StepOutput(
            observation=feedback,
            tool_name=LCB_TOOL if parse_ok else None,
            tool_arguments={"code": code},
            tool_result=feedback,
            success=success,
            failure_reason=reason,
            terminal=True,
            parsed_ok=parse_ok,
            gold_tool=LCB_TOOL,
            tool_correct=all_pass and parse_ok,
            decision_label=label,
            local_reward=1.0 if success else 0.0,
            extra={"lcb_eval": eval_out},
        )
