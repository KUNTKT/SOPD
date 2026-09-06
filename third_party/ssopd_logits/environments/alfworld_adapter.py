"""ALFWorld (text) adapter: multi-step, sparse terminal reward, admissible actions.

The admissible-command list ALFWorld returns at every step *is* the valid action
set `A(s)` that this repo's teachers have always been constrained to. That is the
only reason ALFWorld is the main carrier.

Deliberate omissions: the environment's own score/intermediate_reward and the
handcoded/planner expert plan are never surfaced as a training signal. The expert
plan is available only when `expert_plan=True`, is stored under
`StepOutput.extra["oracle_expert_next"]`, and is used exclusively by audits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from environments.base_env import AgentEnvironment, StepOutput

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

SPLIT_DIRS = {
    "train": "train",
    "valid_train": "valid_train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}

# `help` is a meta command that never advances the task; keeping it in A(s) only
# adds probability mass that no policy should ever want.
DROPPED_COMMANDS = frozenset({"help"})

ACTION_RE = re.compile(r"(?:^|\n)\s*(?:ACTION|Action|action)\s*[:：]\s*(.+)")

ALFWORLD_SYSTEM_PROMPT = (
    "You are an agent acting in a text household environment. At each step you "
    "see the task, the recent history, the current observation, and the full "
    "list of admissible actions. Choose exactly one action, copied verbatim from "
    "the admissible list. Reply with a single line of the form "
    "'ACTION: <action>' and nothing else."
)

# Two compact demonstrations. Base Qwen2.5-1.5B-Instruct emits commentary and
# invented commands without them; the point is format adherence, not strategy,
# so no demonstration is drawn from the evaluated task distribution's solutions.
ALFWORLD_FEWSHOT_PROMPT = ALFWORLD_SYSTEM_PROMPT + (
    "\n\nExamples of the required reply format.\n\n"
    "Admissible actions include 'go to shelf 2'. You reply:\n"
    "ACTION: go to shelf 2\n\n"
    "Admissible actions include 'put mug 1 in/on sinkbasin 1'. You reply:\n"
    "ACTION: put mug 1 in/on sinkbasin 1\n\n"
    "Never explain, never number the reply, never invent an action that is not "
    "listed."
)


def alfworld_data_root(root: str | os.PathLike[str] | None = None) -> Path:
    if root is not None:
        return Path(root)
    env_root = os.environ.get("ALFWORLD_DATA")
    if env_root:
        return Path(env_root)
    return Path.home() / ".cache" / "alfworld"


def _task_id(game_file: Path, data_root: Path) -> str:
    try:
        rel = game_file.parent.relative_to(data_root)
    except ValueError:
        rel = game_file.parent
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    return f"alfw_{digest}"


def _goal_text(traj_data: dict[str, Any]) -> str:
    anns = (traj_data.get("turk_annotations") or {}).get("anns") or []
    for ann in anns:
        desc = str(ann.get("task_desc") or "").strip()
        if desc:
            return desc
    return str(traj_data.get("task_type") or "")


def load_alfworld_tasks(
    *,
    split: str = "train",
    task_types: Iterable[int] = (1, 2, 3, 4, 5, 6),
    root: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Collect solvable ALFWorld text games as repo-standard task dicts."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_DIRS)}")
    data_root = alfworld_data_root(root)
    split_root = data_root / "json_2.1.1" / SPLIT_DIRS[split]
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"ALFWorld split not found: {split_root}. Run `alfworld-download` with "
            "ALFWORLD_DATA set."
        )

    allowed = {TASK_TYPES[t] for t in task_types if t in TASK_TYPES}
    if not allowed:
        raise ValueError("no valid task_types given")

    tasks: list[dict[str, Any]] = []
    for traj_path in sorted(split_root.rglob("traj_data.json")):
        parent = traj_path.parent
        parts = set(parent.parts)
        if "movable" in str(parent) or "Sliced" in str(parent):
            continue
        game_file = parent / "game.tw-pddl"
        if not game_file.is_file():
            continue
        try:
            traj_data = json.loads(traj_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if traj_data.get("task_type") not in allowed:
            continue
        try:
            gamedata = json.loads(game_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not gamedata.get("solvable"):
            continue
        low_actions = ((traj_data.get("plan") or {}).get("low_actions") or [])
        tasks.append(
            {
                "task_id": _task_id(game_file, data_root),
                "source": "alfworld",
                "split": split,
                "game_file": str(game_file),
                "task_type": traj_data["task_type"],
                "goal": _goal_text(traj_data),
                "scene": str(sorted(parts & {parent.name})[0]) if parts else parent.name,
                # Reference only. There is no per-step gold action in this env.
                "reference_plan_length": len(low_actions),
                "n_steps": None,
                "gold_plan": [],
                "valid_tools": [],
            }
        )
    # The directory walk groups tasks by type, so any `[:n]` slice downstream —
    # a probe, a training window, an eval pool — would be one task type only.
    # Ordering by a hash of the task id makes every prefix a stratified sample
    # while staying reproducible and independent of the walk order.
    tasks.sort(key=lambda t: hashlib.sha1(str(t["task_id"]).encode()).hexdigest())
    return tasks[:limit] if limit is not None else tasks


def split_alfworld_tasks(
    tasks: list[dict[str, Any]],
    *,
    select_frac: float = 0.5,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic hash split. Independent of task order and of `limit`."""
    select: list[dict[str, Any]] = []
    confirm: list[dict[str, Any]] = []
    for task in tasks:
        raw = f"{seed}:{task['task_id']}".encode()
        bucket = int(hashlib.sha1(raw).hexdigest()[:8], 16) / 0xFFFFFFFF
        (select if bucket < select_frac else confirm).append(task)
    return select, confirm


def partition_alfworld_tasks(
    tasks: list[dict[str, Any]],
    *,
    fractions: dict[str, float],
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic hash partition into named pools.

    Used to keep the cold-start SFT pool disjoint from the audit pools. Order- and
    `limit`-independent, so growing the task list never moves a task between
    pools and silently contaminates a confirm split.
    """
    names = list(fractions)
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    edges: list[tuple[str, float]] = []
    acc = 0.0
    for name in names:
        acc += fractions[name] / total
        edges.append((name, acc))

    out: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for task in tasks:
        raw = f"{seed}:{task['task_id']}".encode()
        bucket = int(hashlib.sha1(raw).hexdigest()[:8], 16) / 0xFFFFFFFF
        for name, edge in edges:
            if bucket < edge:
                out[name].append(task)
                break
        else:
            out[names[-1]].append(task)
    return out


def tasks_fingerprint(tasks: list[dict[str, Any]]) -> str:
    ids = sorted(str(t["task_id"]) for t in tasks)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def normalize_command(text: str) -> str:
    cleaned = str(text or "").strip().strip("`\"'*")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip(".!").strip().lower()


def parse_alfworld_action(text: str, admissible: list[str]) -> dict[str, Any]:
    """Extract one command from free-form model output and match it to `A(s)`.

    Returns `command` (what to send to the env) and `matched` (whether it is in
    `A(s)`). An unmatched command is still sent: ALFWorld answers "Nothing
    happens." and the episode continues, which is the correct multi-step
    behaviour. Terminating on a parse error would turn every format slip into an
    episode failure and re-create the EXP06b confound.
    """
    raw = str(text or "")
    hits = ACTION_RE.findall(raw)
    if hits:
        candidate = hits[-1]
    else:
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        candidate = lines[-1] if lines else ""
    candidate = candidate.split("\n")[0]
    norm = normalize_command(candidate)

    by_norm = {normalize_command(c): c for c in admissible}
    if norm in by_norm:
        return {"command": by_norm[norm], "matched": True, "raw": candidate.strip()}

    # A single containment hit is unambiguous; several is not, and guessing there
    # would silently fabricate an action the policy did not choose.
    contained = [orig for n, orig in by_norm.items() if n and (n in norm or norm in n)]
    if len(contained) == 1:
        return {"command": contained[0], "matched": True, "raw": candidate.strip()}

    return {"command": candidate.strip() or "look", "matched": False, "raw": candidate.strip()}


def render_alfworld_observation(
    *,
    goal: str,
    feedback: str,
    admissible: list[str],
    history: list[dict[str, Any]],
    step_index: int,
    max_steps: int,
    history_window: int = 12,
) -> str:
    """Full decision context for one step. This string is the policy prompt."""
    lines = [f"Task: {goal}" if goal else "Task: complete the household goal."]
    if history:
        lines.append("")
        lines.append("History:")
        for h in history[-history_window:]:
            lines.append(f"  > {h['command']}")
            lines.append(f"    {h['feedback']}")
    lines.append("")
    lines.append(f"Observation: {feedback}")
    lines.append("")
    lines.append("Admissible actions:")
    for cmd in admissible:
        lines.append(f"  - {cmd}")
    lines.append("")
    lines.append(f"Step {step_index + 1} of {max_steps}.")
    lines.append("Choose exactly one admissible action. Reply with a single line:")
    lines.append("ACTION: <action>")
    return "\n".join(lines)


class AlfWorldEnv(AgentEnvironment):
    """Single-episode ALFWorld text environment behind the repo's env interface."""

    def __init__(
        self,
        *,
        max_steps: int = 40,
        expert_plan: bool = False,
        history_window: int = 12,
        max_admissible: int | None = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.expert_plan = bool(expert_plan)
        self.history_window = int(history_window)
        self.max_admissible = max_admissible

        self.task: dict[str, Any] | None = None
        self._env: Any = None
        self._infos: dict[str, Any] = {}
        self._feedback: str = ""
        self._admissible: list[str] = []
        self._history: list[dict[str, Any]] = []
        self.step_index: int = 0
        self._terminal: bool = False
        self._won: bool = False
        self._last_observation: str = ""

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task: dict[str, Any]) -> str:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        self.task = dict(task)
        extras = ["gamefile"]
        wrappers: list[Any] = [AlfredDemangler(shuffle=False)]
        if self.expert_plan:
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredExpert,
                AlfredExpertType,
            )

            wrappers.append(AlfredExpert(expert_type=AlfredExpertType.PLANNER))
            extras.append("expert_plan")

        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=extras
        )
        env_id = textworld.gym.register_game(
            str(task["game_file"]),
            request_infos,
            max_episode_steps=self.max_steps + 1,
            wrappers=wrappers,
        )
        self._env = textworld.gym.make(env_id)
        feedback, infos = self._env.reset()

        self._infos = infos
        self._feedback = str(feedback)
        self._admissible = self._clean_admissible(infos.get("admissible_commands"))
        self._history = []
        self.step_index = 0
        self._terminal = False
        self._won = False
        self._last_observation = self._render()
        return self._last_observation

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    # -- state -------------------------------------------------------------

    def _clean_admissible(self, raw: Any) -> list[str]:
        if raw is None:
            return []
        cmds = list(raw[0]) if raw and isinstance(raw[0], (list, tuple)) else list(raw)
        out: list[str] = []
        seen: set[str] = set()
        for c in cmds:
            c = str(c).strip()
            n = normalize_command(c)
            if not n or n in DROPPED_COMMANDS or n in seen:
                continue
            seen.add(n)
            out.append(c)
        if self.max_admissible is not None and len(out) > self.max_admissible:
            out = out[: self.max_admissible]
        return out

    def _render(self) -> str:
        assert self.task is not None
        return render_alfworld_observation(
            goal=str(self.task.get("goal") or ""),
            feedback=self._feedback,
            admissible=self._admissible,
            history=self._history,
            step_index=self.step_index,
            max_steps=self.max_steps,
            history_window=self.history_window,
        )

    def current_state(self) -> dict[str, Any]:
        assert self.task is not None
        return {
            "task_id": self.task["task_id"],
            "step_index": self.step_index,
            "completed_tools": [h["command"] for h in self._history],
            "observation": self._last_observation,
            "feedback": self._feedback,
            "goal": self.task.get("goal"),
            # No per-step gold action exists in this environment. Sparse terminal
            # reward is the point of the experiment, not an oversight.
            "gold_tool": None,
            "gold_arguments": {},
            "argument_options": {},
            "subgoal": None,
            "valid_tools": list(self._admissible),
            "n_steps": None,
            "premature_downstream_tool": None,
            "history": [dict(h) for h in self._history],
            "source": "alfworld",
            "task_type": self.task.get("task_type"),
        }

    def valid_actions(self, state: dict[str, Any] | None = None) -> list[str]:
        if state is not None and "valid_tools" in state:
            return list(state["valid_tools"])
        return list(self._admissible)

    # -- verification ------------------------------------------------------

    def verify_decision(
        self, state: dict[str, Any], action: dict[str, Any] | str
    ) -> dict[str, Any]:
        admissible = list(state.get("valid_tools") or [])
        if isinstance(action, str):
            parsed = parse_alfworld_action(action, admissible)
        else:
            cmd = str(action.get("tool_name") or "")
            parsed = {
                "command": cmd,
                "matched": normalize_command(cmd)
                in {normalize_command(c) for c in admissible},
                "raw": cmd,
            }
        return {
            "parse_ok": bool(parsed["matched"]),
            "predicted_tool": parsed["command"],
            "predicted_arguments": {},
            "gold_tool": None,
            "gold_arguments": {},
            "tool_correct": False,
            "argument_exact": False,
            "argument_semantic": False,
            # No process label is available, and inventing one from the episode
            # outcome is exactly the outcome-confounding this repo is avoiding.
            "decision_label": "UNSCORABLE",
            "failure_reason": None if parsed["matched"] else "not_admissible",
            "is_premature_downstream_action": False,
        }

    def verify_trajectory(self, trajectory: dict[str, Any]) -> bool:
        return bool(trajectory.get("episode_success", False))

    def reward(self, trajectory: dict[str, Any]) -> float:
        return 1.0 if self.verify_trajectory(trajectory) else 0.0

    # -- transition --------------------------------------------------------

    def step(self, action: str) -> StepOutput:
        if self._env is None or self.task is None:
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
        parsed = parse_alfworld_action(action, self._admissible)
        oracle_expert = list(self._infos.get("extra.expert_plan") or []) or None

        feedback, _score, done, infos = self._env.step(parsed["command"])

        self._infos = infos
        self._feedback = str(feedback)
        self._admissible = self._clean_admissible(infos.get("admissible_commands"))
        self._history.append(
            {"command": parsed["command"], "feedback": self._feedback}
        )
        self.step_index += 1
        self._won = bool(infos.get("won"))
        self._terminal = bool(done) or self.step_index >= self.max_steps
        self._last_observation = self._render()

        failure_reason: str | None = None
        if not parsed["matched"]:
            failure_reason = "not_admissible"
        elif self._terminal and not self._won:
            failure_reason = "episode_failed"

        return StepOutput(
            observation=self._last_observation,
            tool_name=parsed["command"],
            tool_arguments={},
            tool_result=self._feedback,
            success=self._won,
            failure_reason=failure_reason,
            terminal=self._terminal,
            parsed_ok=bool(parsed["matched"]),
            gold_tool=None,
            gold_arguments={},
            tool_correct=False,
            argument_exact=False,
            argument_semantic=False,
            decision_label="UNSCORABLE",
            is_premature_downstream_action=False,
            # Sparse by construction: credit only at the verified episode end.
            local_reward=1.0 if (self._terminal and self._won) else 0.0,
            extra={
                "step_index": state["step_index"],
                "admissible": list(state["valid_tools"]),
                "next_admissible": list(self._admissible),
                "raw_action": parsed["raw"],
                "oracle_expert_next": oracle_expert,
            },
        )
