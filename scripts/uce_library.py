#!/usr/bin/env python3
"""Instance-level UCE-lite workflow library (archive; no type-majority)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent_ssopd_build_workflows import ACTION_PREFIX, INSTANCE_RE

GOAL_TASK_RE = re.compile(r"(?:^|\n)\s*Task:\s*(.+?)(?:\n|$)", re.I)
GOAL_YOUR_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.I)
TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")
STOP = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "or",
    "in",
    "on",
    "of",
    "with",
    "for",
    "from",
    "put",
    "find",
    "your",
    "task",
    "is",
    "two",
    "them",
}


def strip_action(raw: str) -> str:
    return ACTION_PREFIX.sub("", str(raw or "").strip()).strip()


def abstract_action(raw: str) -> str:
    """Strip ACTION: and instance numbers; keep object/receptacle types."""
    text = strip_action(raw).lower()
    text = INSTANCE_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def abstract_goal(raw: str) -> str:
    text = str(raw or "").lower()
    text = INSTANCE_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_goal_from_prompt(prompt_text: str) -> str:
    parts: list[str] = []
    m_task = GOAL_TASK_RE.search(prompt_text or "")
    if m_task:
        parts.append(m_task.group(1).strip())
    m_your = GOAL_YOUR_RE.search(prompt_text or "")
    if m_your:
        parts.append(m_your.group(1).strip())
    return " / ".join(p for p in parts if p)


def tokenize(text: str) -> frozenset[str]:
    toks = TOKEN_RE.findall(abstract_goal(text))
    return frozenset(t for t in toks if t not in STOP and len(t) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def format_workflow(lines: list[str]) -> str:
    body = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    return (
        "Suggested workflow from a similar solved instance "
        "(adapt names to the current admissible list):\n"
        f"{body}"
    )


def action_lines(actions: list[str], *, max_steps: int = 12) -> list[str]:
    lines: list[str] = []
    for raw in actions:
        line = abstract_action(raw)
        if not line:
            continue
        if lines and lines[-1] == line:
            continue
        if line in {"look", "inventory"}:
            continue
        lines.append(line)
        if len(lines) >= max_steps:
            break
    return lines


def make_entry(
    *,
    entry_id: str,
    task_type: str,
    goal: str,
    actions: list[str],
    source_task_id: str,
    usage: int = 1,
) -> dict[str, Any]:
    goal_abs = abstract_goal(goal)
    lines = action_lines(actions)
    return {
        "id": entry_id,
        "task_type": str(task_type or "unknown"),
        "goal": goal_abs,
        "goal_tokens": sorted(tokenize(goal_abs)),
        "lines": lines,
        "text": format_workflow(lines) if lines else "",
        "source_task_id": source_task_id,
        "usage": int(usage),
    }


class UceLibrary:
    def __init__(self, entries: list[dict[str, Any]] | None = None):
        self.entries: list[dict[str, Any]] = list(entries or [])

    @classmethod
    def load(cls, path: Path) -> "UceLibrary":
        payload = json.loads(path.read_text())
        return cls(list(payload.get("entries") or []))

    def save(self, path: Path, **meta: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_entries": len(self.entries),
            "entries": self.entries,
            **meta,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def retrieve(self, task: dict[str, Any]) -> tuple[str, str | None, float]:
        tt = str(task.get("task_type") or "")
        qtoks = tokenize(str(task.get("goal") or ""))
        pool = [e for e in self.entries if e.get("task_type") == tt and e.get("text")]
        if not pool:
            pool = [e for e in self.entries if e.get("text")]
        best: dict[str, Any] | None = None
        best_key = (-1.0, -10**9)
        for e in pool:
            score = jaccard(qtoks, frozenset(e.get("goal_tokens") or []))
            key = (score, int(e.get("usage") or 0))
            if key > best_key:
                best_key = key
                best = e
        if best is None:
            return "", None, 0.0
        return str(best["text"]), str(best["id"]), float(best_key[0])

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def bump(self, entry_id: str | None, delta: int) -> None:
        if not entry_id:
            return
        for e in self.entries:
            if e.get("id") == entry_id:
                e["usage"] = int(e.get("usage") or 0) + int(delta)
                return

    def prune(self) -> int:
        before = len(self.entries)
        self.entries = [e for e in self.entries if int(e.get("usage") or 0) > 0 and e.get("text")]
        return before - len(self.entries)
