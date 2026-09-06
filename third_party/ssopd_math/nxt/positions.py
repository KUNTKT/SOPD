"""Token position sites for MATH rollouts (reasoning_end before \\boxed)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step."
)

BOXED_MARKER_RE = re.compile(r"\\boxed")


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
    l = int(num_hidden_layers)
    idx0 = [int(0.25 * l), int(0.5 * l), int(0.75 * l), int(0.9 * l)]
    idx0 = [min(max(i, 0), l - 1) for i in idx0]
    return {
        "num_hidden_layers": l,
        "hook_indices_0based": idx0,
        "fractions": [0.25, 0.5, 0.75, 0.9],
    }


def build_prompt_text(
    tokenizer: Any,
    prompt_user: str,
    system_prompt: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    return f"{sys_p}\n\nUser:\n{prompt_user}\n\nAssistant:\n"


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _char_to_token(offsets: list[tuple[int, int]], char_pos: int) -> int | None:
    for i, (a, b) in enumerate(offsets):
        if a <= char_pos < b or (a == b == char_pos):
            return i
        if char_pos < a:
            return max(i - 1, 0)
    if offsets:
        return len(offsets) - 1
    return None


def locate_sites(
    tokenizer: Any,
    prompt_user: str,
    completion_text: str,
) -> PositionPack:
    """Main site: last reasoning token immediately before \\boxed{...}."""
    prompt = build_prompt_text(tokenizer, prompt_user)
    prompt_ids = _encode_ids(tokenizer, prompt)
    comp_enc = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    comp_ids = list(comp_enc["input_ids"])
    offsets = [(int(a), int(b)) for a, b in comp_enc["offset_mapping"]]
    full_ids = prompt_ids + comp_ids
    prompt_len = len(prompt_ids)
    ctx = len(full_ids)

    match = BOXED_MARKER_RE.search(completion_text)
    marker_found = match is not None
    marker_char = match.start() if match is not None else None
    fallback = False

    if marker_found and marker_char is not None:
        local = _char_to_token(offsets, marker_char)
        if local is None or local <= 0:
            token_pos = max(prompt_len - 1, 0)
            fallback = True
        else:
            token_pos = prompt_len + local - 1
    elif comp_ids:
        token_pos = prompt_len + len(comp_ids) - 1
        fallback = True
    else:
        token_pos = max(prompt_len - 1, 0)
        fallback = True

    token_pos = min(max(int(token_pos), 0), max(ctx - 1, 0))
    tid = int(full_ids[token_pos]) if full_ids else 0
    site = Site(
        "reasoning_end",
        token_pos,
        tid,
        tokenizer.decode([tid]) if full_ids else "",
        fallback,
    )
    return PositionPack(
        prompt_length=prompt_len,
        context_length=ctx,
        input_ids=full_ids,
        sites={"reasoning_end": site},
        marker_found=marker_found,
        marker_char_start=marker_char,
    )
