"""Map a generated action to residual-stream token positions for EXP03.

Main site: last reasoning token before the tool-call marker (not the tool name).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ssopd_agent.models.agent_model import DEFAULT_SYSTEM_PROMPT

XML_TOOL_RE = re.compile(r"<tool>\s*([A-Za-z0-9_]+)\s*</tool>", re.IGNORECASE)
NATIVE_BEGIN = "<｜tool▁call▁begin｜>"


@dataclass
class Site:
    representation_type: str
    token_position: int
    token_id: int
    token_text: str
    representation_fallback: bool


@dataclass
class PositionPack:
    prompt_length: int
    context_length: int
    input_ids: list[int]
    sites: dict[str, Site]
    marker_found: bool
    marker_char_start: int | None


def candidate_layers(num_hidden_layers: int) -> dict[str, Any]:
    """Hook transformer blocks only. Embedding is not a layer index."""
    l = int(num_hidden_layers)
    idx0 = [int(0.25 * l), int(0.5 * l), int(0.75 * l), int(0.9 * l)]
    idx0 = [min(max(i, 0), l - 1) for i in idx0]
    return {
        "num_hidden_layers": l,
        "embedding_counted": False,
        "index_base": "0-based HuggingFace model.model.layers[k]",
        "hook_indices_0based": idx0,
        "hook_indices_1based": [i + 1 for i in idx0],
        "fractions": [0.25, 0.5, 0.75, 0.9],
        "note": (
            f"L={l}; floor(fL) = {idx0}. layers[0] is the first transformer block, "
            "not the embedding matrix. 1-based paper ids are k+1 "
            f"= { [i+1 for i in idx0] }."
        ),
    }


def build_prompt_text(tokenizer, observation: str, system_prompt: str | None = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _encode_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _char_to_token(offsets: list[tuple[int, int]], char_pos: int) -> int | None:
    for i, (a, b) in enumerate(offsets):
        if a <= char_pos < b or (a == b == char_pos):
            return i
        if char_pos < a:
            return max(i - 1, 0)
    if offsets:
        return len(offsets) - 1
    return None


def _completion_offsets(tokenizer, completion: str) -> tuple[list[int], list[tuple[int, int]], str]:
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    return ids, offsets, completion


def _find_marker(completion: str) -> tuple[int | None, int | None, str | None]:
    """Return (char_start_marker, char_start_tool_name, predicted_from_span)."""
    m = XML_TOOL_RE.search(completion)
    if m:
        return m.start(), m.start(1), m.group(1).lower()
    idx = completion.find("<tool>")
    if idx >= 0:
        return idx, None, None
    nidx = completion.find(NATIVE_BEGIN)
    if nidx >= 0:
        return nidx, None, None
    return None, None, None


def locate_sites(tokenizer, observation: str, action: str, predicted_tool: str | None) -> PositionPack:
    prompt = build_prompt_text(tokenizer, observation)
    prompt_ids = _encode_ids(tokenizer, prompt)
    comp_ids, offsets, completion = _completion_offsets(tokenizer, action)
    full_ids = prompt_ids + comp_ids
    prompt_len = len(prompt_ids)
    ctx = len(full_ids)

    def site_at(local_comp_idx: int, rtype: str, fallback: bool) -> Site:
        if local_comp_idx < 0:
            pos = prompt_len + local_comp_idx  # -1 -> last prompt token
            pos = max(0, pos)
            tid = full_ids[pos]
            return Site(rtype, pos, int(tid), tokenizer.decode([tid]), True)
        pos = prompt_len + local_comp_idx
        pos = min(max(pos, 0), ctx - 1)
        tid = full_ids[pos]
        return Site(rtype, pos, int(tid), tokenizer.decode([tid]), fallback)

    marker_char, name_char, _span_name = _find_marker(completion)
    marker_found = marker_char is not None
    fallback = False

    if marker_char is not None:
        marker_local = _char_to_token(offsets, marker_char)
        if marker_local is None:
            marker_local = 0
            fallback = True
        if marker_local <= 0:
            # First completion token is the marker: use last prompt token.
            pre_reason = site_at(-1, "pre_call_reasoning", True)
            fallback = True
        else:
            pre_reason = site_at(marker_local - 1, "pre_call_reasoning", False)
        call_marker = site_at(marker_local, "call_marker", fallback)
        if name_char is not None:
            name_local = _char_to_token(offsets, name_char)
            if name_local is None or name_local <= 0:
                pre_name = site_at(marker_local, "pre_tool_name", True)
            else:
                pre_name = site_at(name_local - 1, "pre_tool_name", False)
        else:
            pre_name = site_at(marker_local, "pre_tool_name", True)
    else:
        fallback = True
        # No explicit marker: last completion token before predicted name, else last token.
        name_local = None
        if predicted_tool:
            m = re.search(re.escape(predicted_tool), completion, flags=re.IGNORECASE)
            if m:
                name_local = _char_to_token(offsets, m.start())
        if name_local is not None and name_local > 0:
            pre_reason = site_at(name_local - 1, "pre_call_reasoning", True)
            call_marker = site_at(name_local - 1, "call_marker", True)
            pre_name = site_at(name_local - 1, "pre_tool_name", True)
        elif comp_ids:
            pre_reason = site_at(len(comp_ids) - 1, "pre_call_reasoning", True)
            call_marker = pre_reason
            call_marker = Site("call_marker", pre_reason.token_position, pre_reason.token_id, pre_reason.token_text, True)
            pre_name = Site("pre_tool_name", pre_reason.token_position, pre_reason.token_id, pre_reason.token_text, True)
        else:
            pre_reason = site_at(-1, "pre_call_reasoning", True)
            call_marker = Site("call_marker", pre_reason.token_position, pre_reason.token_id, pre_reason.token_text, True)
            pre_name = Site("pre_tool_name", pre_reason.token_position, pre_reason.token_id, pre_reason.token_text, True)

    # Main site must not be the predicted tool-name token itself.
    pred = (predicted_tool or "").strip().lower()
    if pred and pre_reason.token_text.strip().lower() == pred and pre_reason.token_position > 0:
        shifted = site_at(
            pre_reason.token_position - prompt_len - 1, "pre_call_reasoning", True
        )
        pre_reason = shifted

    return PositionPack(
        prompt_length=prompt_len,
        context_length=ctx,
        input_ids=full_ids,
        sites={
            "pre_call_reasoning": pre_reason,
            "call_marker": call_marker,
            "pre_tool_name": pre_name,
        },
        marker_found=marker_found,
        marker_char_start=marker_char,
    )


def pack_to_meta(pack: PositionPack) -> dict[str, Any]:
    return {
        "prompt_length": pack.prompt_length,
        "context_length": pack.context_length,
        "marker_found": pack.marker_found,
        "marker_char_start": pack.marker_char_start,
        "sites": {k: asdict(v) for k, v in pack.sites.items()},
        "n_tokens": len(pack.input_ids),
    }
