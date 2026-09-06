"""Hidden-state extraction via forward hooks."""

from __future__ import annotations

from typing import Any

import torch

from ssopd_math.nxt.extract import _unwrap_model, decoder_layers


def attach_block_output_hooks(
    model: Any,
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    """Register forward hooks on decoder blocks; return cache dict and handles."""
    cache: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    layers = decoder_layers(model)

    def _make(idx: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[idx] = h

        return hook

    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"layer index {idx} out of range (L={len(layers)})")
        handles.append(layers[idx].register_forward_hook(_make(idx)))
    return cache, handles


def remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        h.remove()


def extract_hidden_at_position(
    hidden: torch.Tensor,
    position: int,
    batch_index: int = 0,
) -> torch.Tensor:
    """Return hidden vector at (batch_index, position) from [B,T,D] or [T,D]."""
    if hidden.dim() == 2:
        if position < 0 or position >= hidden.shape[0]:
            raise IndexError(f"position {position} out of range for shape {tuple(hidden.shape)}")
        return hidden[position].detach()
    if hidden.dim() == 3:
        if batch_index < 0 or batch_index >= hidden.shape[0]:
            raise IndexError(f"batch_index {batch_index} out of range")
        if position < 0 or position >= hidden.shape[1]:
            raise IndexError(f"position {position} out of range for shape {tuple(hidden.shape)}")
        return hidden[batch_index, position].detach()
    raise ValueError(f"expected 2D or 3D hidden, got shape {tuple(hidden.shape)}")
