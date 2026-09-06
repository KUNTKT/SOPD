"""Tool-call parsers: native DeepSeek markers, XML, and JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from environments.tool_registry import resolve_tool_name

# Tool / function identifiers may include dots (BFCL: triangle_properties.get).
_TOOL_NAME_CORE = r"[A-Za-z_][A-Za-z0-9_.]*"
_TOOL_NAME = rf"({_TOOL_NAME_CORE})"

# DeepSeek native (fullwidth special tokens)
_NATIVE_FW = re.compile(
    rf"<｜tool▁call▁begin｜>\s*(?:function)?\s*<｜tool▁sep｜>\s*{_TOOL_NAME}"
    r"(?:.*?```(?:json)?\s*(\{.*?\})\s*```)?",
    re.IGNORECASE | re.DOTALL,
)
# ASCII / underscore variants sometimes appear in text
_NATIVE_ASC = re.compile(
    rf"<\|tool[▁_]?call[▁_]?begin\|>\s*(?:function)?\s*<\|tool[▁_]?sep\|>\s*{_TOOL_NAME}"
    r"(?:.*?```(?:json)?\s*(\{.*?\})\s*```)?",
    re.IGNORECASE | re.DOTALL,
)
_XML_TOOL = re.compile(rf"<tool>\s*{_TOOL_NAME}\s*</tool>", re.IGNORECASE)
_XML_ARGS = re.compile(r"<args>\s*(.*?)\s*</args>", re.IGNORECASE | re.DOTALL)
_JSON_CALL = re.compile(
    rf'\{{\s*"name"\s*:\s*"({_TOOL_NAME_CORE})"\s*,\s*"arguments"\s*:\s*(\{{.*?\}})\s*\}}',
    re.DOTALL,
)
_LINE_TOOL = re.compile(
    rf"(?:tool|function)\s*(?:name)?\s*[:=]\s*['\"]?{_TOOL_NAME}",
    re.IGNORECASE,
)


@dataclass
class ParsedToolCall:
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_ok: bool = False
    format: str | None = None
    raw: str = ""


def _parse_args_blob(blob: str | None) -> dict[str, Any]:
    if not blob:
        return {}
    text = blob.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Tolerate non-JSON XML args like: "side1": "3"\n"side2": "4"
    kv_lines = re.findall(
        r'["\']?([A-Za-z_][A-Za-z0-9_]*)["\']?\s*:\s*["\']?([^"\',\n]+)["\']?',
        text,
    )
    if kv_lines:
        out: dict[str, Any] = {}
        for k, v in kv_lines:
            vv = v.strip().rstrip(",")
            if re.fullmatch(r"-?\d+", vv):
                out[k] = int(vv)
            elif re.fullmatch(r"-?\d+\.\d+", vv):
                out[k] = float(vv)
            elif vv.lower() in ("true", "false"):
                out[k] = vv.lower() == "true"
            else:
                out[k] = vv
        if out:
            return out
    # key=value pairs
    out = {}
    for part in re.split(r"[,\n]", text):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


def _canonicalize_parsed_name(raw_name: str) -> str | None:
    """Resolve synthetic aliases; keep BFCL-style open names."""
    resolved = resolve_tool_name(raw_name)
    if resolved is not None:
        return resolved
    return resolve_tool_name(raw_name, keep_unknown=True)


def parse_tool_call(text: str) -> ParsedToolCall:
    raw = text or ""
    for fmt, pattern in (("native_fw", _NATIVE_FW), ("native_asc", _NATIVE_ASC)):
        m = pattern.search(raw)
        if m:
            name = _canonicalize_parsed_name(m.group(1))
            args = _parse_args_blob(m.group(2) if m.lastindex and m.lastindex >= 2 else None)
            return ParsedToolCall(
                tool_name=name,
                arguments=args,
                parse_ok=name is not None,
                format=fmt,
                raw=raw,
            )

    m_xml = _XML_TOOL.search(raw)
    if m_xml:
        name = _canonicalize_parsed_name(m_xml.group(1))
        args_m = _XML_ARGS.search(raw)
        args = _parse_args_blob(args_m.group(1) if args_m else None)
        return ParsedToolCall(
            tool_name=name,
            arguments=args,
            parse_ok=name is not None,
            format="xml",
            raw=raw,
        )

    m_json = _JSON_CALL.search(raw)
    if m_json:
        name = _canonicalize_parsed_name(m_json.group(1))
        args = _parse_args_blob(m_json.group(2))
        return ParsedToolCall(
            tool_name=name,
            arguments=args,
            parse_ok=name is not None,
            format="json",
            raw=raw,
        )

    m_line = _LINE_TOOL.search(raw)
    if m_line:
        name = _canonicalize_parsed_name(m_line.group(1))
        return ParsedToolCall(
            tool_name=name,
            arguments={},
            parse_ok=name is not None,
            format="line",
            raw=raw,
        )

    return ParsedToolCall(parse_ok=False, format=None, raw=raw)


def format_native_tool_call(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    """DeepSeek-style native tool call string (with optional arguments)."""
    prefix = f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{tool_name}"
    if arguments is None:
        return prefix
    args_json = json.dumps(arguments, ensure_ascii=False)
    return (
        f"{prefix}\n```json\n{args_json}\n```"
        f"<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )


def format_xml_tool_call(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    if not arguments:
        return f"<tool>{tool_name}</tool>"
    return f"<tool>{tool_name}</tool><args>{json.dumps(arguments, ensure_ascii=False)}</args>"
