"""Tool registry for the synthetic Tool-Selection environment."""

from __future__ import annotations

from typing import Any

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "calculator": {
        "args": ["expression"],
        "desc": "Evaluate a numeric arithmetic expression. Use for pure math.",
        "aliases": ["calc", "math"],
    },
    "search": {
        "args": ["query"],
        "desc": "Look up a factual question (capitals, inventors, definitions).",
        "aliases": ["web_search", "lookup"],
    },
    "weather": {
        "args": ["location"],
        "desc": "Get current weather for a location. Use for temperature/forecast.",
        "aliases": ["forecast"],
    },
    "translator": {
        "args": ["text", "target_lang"],
        "desc": "Translate text into a target language.",
        "aliases": ["translate"],
    },
    "calendar": {
        "args": ["date", "event"],
        "desc": "Create or look up a calendar event on a date.",
        "aliases": ["schedule"],
    },
    "file_read": {
        "args": ["path"],
        "desc": "Read the contents of a local file path.",
        "aliases": ["read_file", "cat"],
    },
    "email_send": {
        "args": ["to", "subject"],
        "desc": "Send an email to a recipient with a subject.",
        "aliases": ["send_email", "mail"],
    },
    "unit_convert": {
        "args": ["value", "from_unit", "to_unit"],
        "desc": "Convert a numeric value from one unit to another.",
        "aliases": ["convert", "units"],
    },
}

TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SPECS.keys())


def all_tool_names() -> list[str]:
    return list(TOOL_NAMES)


def tool_argument_keys(tool_name: str) -> list[str]:
    if tool_name not in TOOL_SPECS:
        raise KeyError(f"unknown tool: {tool_name}")
    return list(TOOL_SPECS[tool_name]["args"])


def tool_aliases(tool_name: str) -> list[str]:
    """Return registered aliases; unknown/open-vocab tools have none."""
    if tool_name not in TOOL_SPECS:
        return []
    return list(TOOL_SPECS[tool_name].get("aliases", []))


def resolve_tool_name(
    name: str | None,
    valid_tools: list[str] | None = None,
    *,
    keep_unknown: bool = False,
) -> str | None:
    """Map alias or exact name to canonical tool name.

    If ``valid_tools`` is provided (e.g. BFCL per-task docs), match within that
    list first (case-sensitive, then case-insensitive). When ``keep_unknown`` is
    True, unrecognized identifier-like names are returned as-is so they can be
    scored as NEGATIVE wrong-tool rather than PARSE_FAILURE.
    """
    if name is None:
        return None
    raw = name.strip()
    if not raw:
        return None

    if valid_tools:
        for t in valid_tools:
            if t == raw:
                return t
        lower = raw.lower()
        for t in valid_tools:
            if t.lower() == lower:
                return t

    key = raw.lower()
    if key in TOOL_SPECS:
        return key
    for canonical, spec in TOOL_SPECS.items():
        aliases = [a.lower() for a in spec.get("aliases", [])]
        if key in aliases:
            return canonical

    if keep_unknown:
        return raw
    return None


def tool_descriptions_block(valid_tools: list[str] | None = None) -> str:
    names = valid_tools if valid_tools is not None else list(TOOL_NAMES)
    lines = []
    for n in names:
        if n not in TOOL_SPECS:
            continue
        args = ", ".join(TOOL_SPECS[n]["args"])
        lines.append(f"- {n}({args}): {TOOL_SPECS[n]['desc']}")
    return "\n".join(lines)
