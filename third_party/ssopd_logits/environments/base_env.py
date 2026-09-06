"""Unified agent environment interface for Logit-SSOPD."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepOutput:
    observation: str
    tool_name: str | None
    tool_arguments: dict[str, Any]
    tool_result: str
    success: bool
    failure_reason: str | None
    terminal: bool
    parsed_ok: bool = False
    gold_tool: str | None = None
    gold_arguments: dict[str, Any] = field(default_factory=dict)
    tool_correct: bool = False
    argument_exact: bool = False
    argument_semantic: bool = False
    decision_label: str = "UNSCORABLE"
    is_premature_downstream_action: bool = False
    local_reward: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class AgentEnvironment(ABC):
    @abstractmethod
    def reset(self, task: dict[str, Any]) -> str:
        """Start an episode. Returns the first observation / prompt."""

    @abstractmethod
    def step(self, action: str) -> StepOutput:
        """Apply a raw model action string and return the step result."""

    @abstractmethod
    def verify_decision(self, state: dict[str, Any], action: dict[str, Any] | str) -> dict[str, Any]:
        """Deterministic decision-level verification for a single step."""

    @abstractmethod
    def verify_trajectory(self, trajectory: dict[str, Any]) -> bool:
        """Deterministic episode success from a stored trajectory."""

    @abstractmethod
    def reward(self, trajectory: dict[str, Any]) -> float:
        """Episode reward. Default protocol: 1.0 if verify_trajectory else 0.0."""

    @abstractmethod
    def valid_actions(self, state: dict[str, Any] | None = None) -> list[str]:
        """Legal tool names for the current (or provided) state."""
