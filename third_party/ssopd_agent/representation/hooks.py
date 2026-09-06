"""Residual-stream hooks. Embedding is not indexed as a transformer layer."""

from __future__ import annotations

from typing import Any

import torch


def decoder_layers(model) -> list[Any]:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("could not find model.model.layers")
    return list(layers)


def attach_block_output_hooks(model, layer_indices: list[int]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    """Capture post-block residual stream: output of model.model.layers[k]."""
    cache: dict[int, torch.Tensor] = {}
    handles = []
    layers = decoder_layers(model)

    def _make(idx: int):
        def _hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            cache[idx] = hidden.detach()

        return _hook

    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"layer {idx} out of range 0..{len(layers)-1}")
        handles.append(layers[idx].register_forward_hook(_make(idx)))
    return cache, handles


def remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        h.remove()
