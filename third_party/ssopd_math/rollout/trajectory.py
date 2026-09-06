"""Trajectory record schema helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrajectoryRecord:
    problem_id: str
    trajectory_id: str
    problem: str
    prompt_text: str
    completion_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    full_token_ids: list[int]
    parsed_answer: str | None
    gold_answer: str
    final_reward: float
    verification_status: str
    token_count: int
    termination_reason: str
    random_seed: int
    model_name: str
    dataset_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(d.pop("extra", {}))
        return d


def trajectory_id(problem_id: str, group_index: int) -> str:
    return f"{problem_id}::k{group_index}"
