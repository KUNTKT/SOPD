"""Hidden-state extraction via forward hooks."""

from __future__ import annotations

from typing import Any

import torch


def _unwrap_model(model: Any) -> Any:
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base
    for attr in ("base_model", "transformer", "gpt_neox"):
        cand = getattr(model, attr, None)
        if cand is None:
            continue
        if hasattr(cand, "layers"):
            return cand
        inner = getattr(cand, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    return model


def decoder_layers(model: Any) -> list[Any]:
    base = _unwrap_model(model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("could not find decoder layers on model")
    return list(layers)


def attach_block_output_hooks(
    model: Any,
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor], list[Any]]:
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
