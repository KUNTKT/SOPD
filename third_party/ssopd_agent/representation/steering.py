"""Test-time residual-stream steering for EXP06.

Protocol injection (unit vector):
    h_tilde = h + alpha * v_hat

Relative injection (SEED-compatible, optional ablation):
    h_tilde = h + alpha * ||h|| / ||v|| * v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ssopd_agent.representation.hooks import decoder_layers

InjectMode = Literal["unit_additive", "relative"]


@dataclass
class SteeringSpec:
    layer: int
    alpha: float
    vector: torch.Tensor  # [hidden], preferably unit-norm for unit_additive
    mode: InjectMode = "unit_additive"
    site: str = "post_mlp_residual"
    # If set, inject only at these per-example token indices [B]; -1 = skip.
    token_indices: torch.Tensor | None = None
    injection_counts: list | None = None  # mutable [n_writes] filled by the hook


def steer_at_indices(
    hidden: torch.Tensor,
    vector: torch.Tensor,
    alpha: float,
    token_indices: torch.Tensor,
    *,
    mode: InjectMode = "unit_additive",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    """Add αv only at token_indices[b] for each batch row. Returns (steered, n_writes)."""
    if hidden.dim() != 3:
        raise ValueError(f"expected hidden [B,T,H], got {tuple(hidden.shape)}")
    idx = token_indices.to(device=hidden.device).long()
    if idx.dim() != 1 or idx.shape[0] != hidden.shape[0]:
        raise ValueError(f"token_indices shape {tuple(idx.shape)} != batch {hidden.shape[0]}")
    if abs(float(alpha)) < 1e-12:
        return hidden, 0
    valid = (idx >= 0) & (idx < hidden.shape[1])
    n_writes = int(valid.sum().item())
    if n_writes == 0:
        return hidden, 0
    steered = hidden.clone()
    v = vector.to(device=hidden.device, dtype=hidden.dtype)
    if v.dim() != 1 or v.shape[0] != hidden.shape[-1]:
        raise ValueError(f"vector {tuple(v.shape)} incompatible with hidden {tuple(hidden.shape)}")
    if mode == "unit_additive":
        v_hat = v.float() / v.float().norm().clamp_min(eps)
        delta = (float(alpha) * v_hat).to(dtype=hidden.dtype)
        b = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        steered[b, idx[b]] = steered[b, idx[b]] + delta
        return steered, n_writes
    if mode == "relative":
        b = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        h_sel = steered[b, idx[b]]
        h_norm = h_sel.float().norm(dim=-1, keepdim=True)
        v_norm = v.float().norm().clamp_min(eps)
        scale = (float(alpha) * h_norm / v_norm).to(dtype=hidden.dtype)
        steered[b, idx[b]] = h_sel + scale * v
        return steered, n_writes
    raise ValueError(f"unknown inject mode: {mode}")


def steer_activation(
    activation: torch.Tensor,
    vector: torch.Tensor,
    alpha: float,
    *,
    mode: InjectMode = "unit_additive",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply steering to a residual stream tensor [..., H]."""
    if abs(float(alpha)) < 1e-12:
        return activation
    v = vector.to(device=activation.device, dtype=activation.dtype)
    while v.dim() < activation.dim():
        v = v.unsqueeze(0)
    if mode == "unit_additive":
        v_norm = v.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        v_hat = (v.float() / v_norm).to(dtype=activation.dtype)
        return activation + float(alpha) * v_hat
    if mode == "relative":
        h_norm = activation.float().norm(dim=-1, keepdim=True)
        v_norm = v.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = (float(alpha) * h_norm / v_norm).to(dtype=activation.dtype)
        return activation + scale * v
    raise ValueError(f"unknown inject mode: {mode}")


def _unwrap_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rewrap_output(hidden, rest):
    if rest is None:
        return hidden
    return (hidden, *rest)


def attach_steering_hook(model, spec: SteeringSpec):
    """Inject into decoder block output (post-MLP residual). Returns removable handle."""
    layers = decoder_layers(model)
    if spec.layer < 0 or spec.layer >= len(layers):
        raise IndexError(f"layer {spec.layer} out of range 0..{len(layers)-1}")
    if spec.site != "post_mlp_residual":
        raise ValueError(f"unsupported site {spec.site}")
    layer = layers[spec.layer]
    vector = spec.vector.detach()
    alpha = float(spec.alpha)
    mode = spec.mode

    def hook(_mod, _inp, output):
        hidden, rest = _unwrap_output(output)
        if spec.token_indices is not None:
            steered, n_writes = steer_at_indices(
                hidden, vector, alpha, spec.token_indices, mode=mode
            )
            if spec.injection_counts is not None:
                spec.injection_counts.append(n_writes)
        else:
            steered = steer_activation(hidden, vector, alpha, mode=mode)
            if spec.injection_counts is not None:
                spec.injection_counts.append(int(hidden.shape[0] * hidden.shape[1]))
        return _rewrap_output(steered, rest)

    return layer.register_forward_hook(hook)


def remove_hook(handle: Any | None) -> None:
    if handle is not None:
        handle.remove()
