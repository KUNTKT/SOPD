"""Deterministic synthetic Tool-Selection task generator."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Literal

from environments.tool_registry import TOOL_NAMES, tool_descriptions_block

DATASET_VERSION = "tool_select_logit_v1"

SplitName = Literal["train", "validation", "test", "exp00", "confirm_d"]

# Template banks are split-disjoint by prefix.
_TEMPLATE_BANKS: dict[str, list[str]] = {
    "train": [f"ta_{i:03d}" for i in range(1, 41)],
    "validation": [f"tb_{i:03d}" for i in range(1, 21)],
    "test": [f"tc_{i:03d}" for i in range(1, 31)],
    "exp00": [f"te_{i:03d}" for i in range(1, 21)],
    "confirm_d": [f"td_{i:03d}" for i in range(1, 31)],
}

_CITIES = ("Paris", "Tokyo", "London", "Berlin", "Seattle", "Sydney", "Cairo", "Mumbai")
_LANGS = ("French", "Spanish", "German", "Japanese", "Chinese")
_FILES = ("/data/notes.txt", "/tmp/report.md", "/home/user/todo.txt", "/var/log/app.log")
_EMAILS = ("ada@example.com", "linus@kernel.org", "grace@navy.mil", "alan@bletchley.uk")
_EVENTS = ("dentist", "standup", "flight", "review", "workshop")
_DATES = ("2026-09-01", "2026-10-12", "2026-11-05", "2026-12-20")
_FACTS = (
    ("capital of France", "Paris"),
    ("inventor of the telephone", "Alexander Graham Bell"),
    ("chemical symbol for gold", "Au"),
    ("largest planet in the solar system", "Jupiter"),
    ("author of 1984", "George Orwell"),
    ("speed of light in vacuum, rounded Mm/s", "300"),
    ("first element on the periodic table", "Hydrogen"),
    ("currency of Japan", "Yen"),
)
_CONV = (
    (5, "miles", "km", "8.047"),
    (10, "km", "miles", "6.214"),
    (32, "celsius", "fahrenheit", "89.6"),
    (100, "fahrenheit", "celsius", "37.778"),
    (2, "kg", "lb", "4.409"),
    (16, "oz", "g", "453.592"),
)
_MATH = (
    ("17 + 25", "42"),
    ("9 * 8", "72"),
    ("144 / 12", "12"),
    ("100 - 37", "63"),
    ("3 ** 4", "81"),
    ("15 % 4", "3"),
)


def _stable_task_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def _pick_template(rng: random.Random, split: str) -> str:
    bank = _TEMPLATE_BANKS[split]
    return rng.choice(bank)


def _apply_split_template(task: dict[str, Any], split: str, template_id: str) -> None:
    """Apply split-specific surface forms without changing the gold plan."""
    if split != "confirm_d":
        return
    original = task["instruction"].rstrip(".")
    variant = int(template_id.rsplit("_", 1)[-1]) % 4
    wrappers = (
        "Execute the required tool workflow for this request: {text}.",
        "Determine the next executable operation and complete: {text}.",
        "Use only the listed capabilities, respecting dependencies: {text}.",
        "Handle this request step by step with valid tool calls: {text}.",
    )
    task["instruction"] = wrappers[variant].format(text=original)


def _single_step_task(rng: random.Random, task_id: str, template_id: str, seed: int) -> dict[str, Any]:
    tool = rng.choice(list(TOOL_NAMES))
    if tool == "calculator":
        expr, ans = rng.choice(_MATH)
        instruction = f"Compute the value of {expr}."
        args = {"expression": expr}
        obs_result = ans
    elif tool == "search":
        q, ans = rng.choice(_FACTS)
        instruction = f"What is the {q}?"
        args = {"query": q}
        obs_result = ans
    elif tool == "weather":
        loc = rng.choice(_CITIES)
        instruction = f"What is the current weather in {loc}?"
        args = {"location": loc}
        obs_result = f"Weather in {loc}: clear, 22C"
    elif tool == "translator":
        text = rng.choice(("hello", "good morning", "thank you", "goodbye"))
        lang = rng.choice(_LANGS)
        instruction = f'Translate "{text}" into {lang}.'
        args = {"text": text, "target_lang": lang}
        obs_result = f"[{lang}] {text}"
    elif tool == "calendar":
        date = rng.choice(_DATES)
        event = rng.choice(_EVENTS)
        instruction = f"Add a calendar event '{event}' on {date}."
        args = {"date": date, "event": event}
        obs_result = f"Scheduled {event} on {date}"
    elif tool == "file_read":
        path = rng.choice(_FILES)
        instruction = f"Read the file at {path}."
        args = {"path": path}
        obs_result = f"Contents of {path}: [mock data]"
    elif tool == "email_send":
        to = rng.choice(_EMAILS)
        subject = rng.choice(("Status update", "Meeting notes", "Reminder"))
        instruction = f"Send an email to {to} with subject '{subject}'."
        args = {"to": to, "subject": subject}
        obs_result = f"Email sent to {to}"
    else:  # unit_convert
        value, fu, tu, ans = rng.choice(_CONV)
        instruction = f"Convert {value} {fu} to {tu}."
        args = {"value": value, "from_unit": fu, "to_unit": tu}
        obs_result = ans

    return {
        "task_id": task_id,
        "instruction": instruction,
        "task_type": "single_step",
        "gold_plan": [{"step": 0, "tool_name": tool, "arguments": args, "expected_result": obs_result}],
        "valid_tools": list(TOOL_NAMES),
        "template_id": template_id,
        "difficulty": "easy",
        "seed": seed,
        "dataset_version": DATASET_VERSION,
        "n_steps": 1,
        "has_information_dependency": False,
        "premature_downstream_tool": None,
    }


def _two_step_dependent(rng: random.Random, task_id: str, template_id: str, seed: int) -> dict[str, Any]:
    """Step0 produces info needed by step1; premature = calling step1 tool first."""
    kind = rng.choice(("search_then_email", "file_then_calc", "weather_then_translate"))
    if kind == "search_then_email":
        q, ans = rng.choice(_FACTS)
        to = rng.choice(_EMAILS)
        instruction = (
            f"Look up the {q}, then email the answer to {to} "
            f"with subject 'Fact'."
        )
        plan = [
            {
                "step": 0,
                "tool_name": "search",
                "arguments": {"query": q},
                "expected_result": ans,
                "subgoal": "retrieve_fact",
            },
            {
                "step": 1,
                "tool_name": "email_send",
                "arguments": {"to": to, "subject": "Fact"},
                "expected_result": f"Email sent to {to}",
                "depends_on_step": 0,
                "subgoal": "send_fact",
                "uses_prior_result": True,
            },
        ]
        premature = "email_send"
    elif kind == "file_then_calc":
        path = rng.choice(_FILES)
        expr, ans = rng.choice(_MATH)
        instruction = (
            f"Read {path} for the expression, then compute it with the calculator. "
            f"(The expression stored in the file is {expr}.)"
        )
        plan = [
            {
                "step": 0,
                "tool_name": "file_read",
                "arguments": {"path": path},
                "expected_result": f"Contents of {path}: {expr}",
                "subgoal": "read_expression",
            },
            {
                "step": 1,
                "tool_name": "calculator",
                "arguments": {"expression": expr},
                "expected_result": ans,
                "depends_on_step": 0,
                "subgoal": "evaluate_expression",
                "uses_prior_result": True,
            },
        ]
        premature = "calculator"
    else:
        loc = rng.choice(_CITIES)
        lang = rng.choice(_LANGS)
        instruction = (
            f"Get the weather for {loc}, then translate the weather summary into {lang}."
        )
        weather_obs = f"Weather in {loc}: clear, 22C"
        plan = [
            {
                "step": 0,
                "tool_name": "weather",
                "arguments": {"location": loc},
                "expected_result": weather_obs,
                "subgoal": "fetch_weather",
            },
            {
                "step": 1,
                "tool_name": "translator",
                "arguments": {"text": weather_obs, "target_lang": lang},
                "expected_result": f"[{lang}] {weather_obs}",
                "depends_on_step": 0,
                "subgoal": "translate_weather",
                "uses_prior_result": True,
            },
        ]
        premature = "translator"

    return {
        "task_id": task_id,
        "instruction": instruction,
        "task_type": "multi_step",
        "gold_plan": plan,
        "valid_tools": list(TOOL_NAMES),
        "template_id": template_id,
        "difficulty": "medium",
        "seed": seed,
        "dataset_version": DATASET_VERSION,
        "n_steps": 2,
        "has_information_dependency": True,
        "premature_downstream_tool": premature,
    }


def _three_step_chain(rng: random.Random, task_id: str, template_id: str, seed: int) -> dict[str, Any]:
    q, ans = rng.choice(_FACTS)
    loc = rng.choice(_CITIES)
    to = rng.choice(_EMAILS)
    instruction = (
        f"First search for the {q}. Then check weather in {loc}. "
        f"Finally email both findings to {to}."
    )
    plan = [
        {
            "step": 0,
            "tool_name": "search",
            "arguments": {"query": q},
            "expected_result": ans,
            "subgoal": "retrieve_fact",
        },
        {
            "step": 1,
            "tool_name": "weather",
            "arguments": {"location": loc},
            "expected_result": f"Weather in {loc}: clear, 22C",
            "depends_on_step": 0,
            "subgoal": "fetch_weather",
        },
        {
            "step": 2,
            "tool_name": "email_send",
            "arguments": {"to": to, "subject": "Findings"},
            "expected_result": f"Email sent to {to}",
            "depends_on_step": 1,
            "subgoal": "send_summary",
            "uses_prior_result": True,
        },
    ]
    return {
        "task_id": task_id,
        "instruction": instruction,
        "task_type": "multi_step",
        "gold_plan": plan,
        "valid_tools": list(TOOL_NAMES),
        "template_id": template_id,
        "difficulty": "hard",
        "seed": seed,
        "dataset_version": DATASET_VERSION,
        "n_steps": 3,
        "has_information_dependency": True,
        "premature_downstream_tool": "email_send",
    }


def generate_task(
    *,
    index: int,
    seed: int,
    split: SplitName = "exp00",
    id_prefix: str = "tb",
    single_step_frac: float = 0.70,
    two_step_frac: float = 0.20,
    three_step_frac: float = 0.10,
) -> dict[str, Any]:
    """Generate one deterministic task. Fractions must sum to ~1."""
    rng = random.Random((seed << 16) ^ index ^ hashlib.md5(split.encode()).digest()[0])
    task_id = _stable_task_id(id_prefix, index)
    template_id = _pick_template(rng, split)
    u = rng.random()
    if u < single_step_frac:
        task = _single_step_task(rng, task_id, template_id, seed)
    elif u < single_step_frac + two_step_frac:
        task = _two_step_dependent(rng, task_id, template_id, seed)
    else:
        task = _three_step_chain(rng, task_id, template_id, seed)
    _apply_split_template(task, split, template_id)
    task["split"] = split
    return task


def generate_tasks(
    n_tasks: int,
    *,
    seed: int,
    split: SplitName = "exp00",
    id_prefix: str = "tb",
    single_step_frac: float = 0.70,
    two_step_frac: float = 0.20,
    three_step_frac: float = 0.10,
) -> list[dict[str, Any]]:
    return [
        generate_task(
            index=i,
            seed=seed,
            split=split,
            id_prefix=id_prefix,
            single_step_frac=single_step_frac,
            two_step_frac=two_step_frac,
            three_step_frac=three_step_frac,
        )
        for i in range(n_tasks)
    ]


def tasks_fingerprint(tasks: list[dict[str, Any]]) -> str:
    blob = json.dumps(tasks, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_agent_prompt(task: dict[str, Any], observation: str | None = None) -> str:
    if task.get("source") == "bfcl":
        from environments.bfcl_adapter import format_bfcl_tools_block

        tools = format_bfcl_tools_block(task)
        parts = [
            "You are a tool-using agent. Select exactly one available tool.",
            "After any brief reasoning, you MUST emit a tool call in this XML format:",
            "<tool>TOOL_NAME</tool>",
            '<args>{"key": "value"}</args>',
            "Do not give a final numeric/text answer yourself; call a tool instead.",
            "Args must be a single JSON object.",
            "",
            "Available tools:",
            tools,
            "",
            f"Task: {task['instruction']}",
        ]
    else:
        tools = tool_descriptions_block(task.get("valid_tools"))
        parts = [
            "You are a tool-using agent. Call exactly one tool when needed.",
            "Return only a valid tool call in this XML format:",
            "<tool>TOOL_NAME</tool>",
            '<args>{"key": "value"}</args>',
            "",
            "Available tools:",
            tools,
            "",
            f"Task: {task['instruction']}",
        ]
    if observation:
        parts.extend(["", f"Current observation:\n{observation}"])
    return "\n".join(parts)


def clone_task(task: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(task)
