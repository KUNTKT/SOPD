"""Residual-stream steering hooks for SSOPD math models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SteeringSpec:
    layer: int
    alpha: float
    vector: torch.Tensor
    token_indices: list[int] | None = None
    mode: str = "unit_additive"
    response_mask: torch.Tensor | None = None
    injection_counts: list[int] = field(default_factory=list)
    h_norms: list[float] = field(default_factory=list)
    rho_values: list[float] = field(default_factory=list)


def normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = vector.detach().float().reshape(-1)
    return v / (v.norm() + eps)


def alpha_from_rho(rho: float, h_norm: float, v_norm: float, eps: float = 1e-8) -> float:
    return float(rho) * float(h_norm) / (float(v_norm) + eps)


def _unwrap_output(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rewrap_output(hidden: torch.Tensor, rest: Any) -> Any:
    if rest is None:
        return hidden
    return (hidden, *rest)


def steer_at_indices(
    hidden: torch.Tensor,
    vector: torch.Tensor,
    alpha: float,
    token_indices: list[int] | torch.Tensor,
    mode: str = "unit_additive",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int, list[float], list[float]]:
    mask = torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.bool, device=hidden.device)
    idx = torch.as_tensor(token_indices, device=hidden.device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < hidden.shape[1])]
    if idx.numel() == 0:
        return hidden, 0, [], []
    mask[:, idx] = True
    return steer_response_mask(hidden, vector, alpha, mask, mode=mode, eps=eps)


def steer_response_mask(
    hidden: torch.Tensor,
    vector: torch.Tensor,
    alpha: float,
    response_mask: torch.Tensor,
    mode: str = "unit_additive",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int, list[float], list[float]]:
    """Apply steering to masked positions of residual stream activations.

    hidden: [B, T, H]
    response_mask: [B, T] bool
    """
    mask = response_mask.to(device=hidden.device, dtype=torch.bool)
    if mask.shape[:2] != hidden.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} != hidden {tuple(hidden.shape[:2])}")
    n_writes = int(mask.sum().item())
    if n_writes == 0 or abs(float(alpha)) < 1e-12:
        return hidden, 0, [], []

    steered = hidden.clone()
    selected = steered[mask]  # [N, H]
    h_norms = selected.norm(dim=-1).detach().float().tolist()
    rhos: list[float] = []
    raw = vector.detach().float().reshape(-1)

    if mode in ("unit_additive", "unit"):
        v = normalize_vector(raw, eps=eps).to(device=hidden.device, dtype=hidden.dtype)
        delta = float(alpha) * v
        steered[mask] = selected + delta
        v_norm = float(v.norm().item()) + eps
        rhos = [float(abs(alpha)) * v_norm / (hn + eps) for hn in h_norms]
    elif mode in ("contrastive_cap", "cap", "paper"):
        # Paper-style: inject α * c where c is the contrastive (unnormalized) vector.
        # α=1 adds a full-scale mean POS−NEG residual difference.
        c = raw.to(device=hidden.device, dtype=hidden.dtype)
        c_norm = float(c.norm().item()) + eps
        steered[mask] = selected + float(alpha) * c
        rhos = [float(abs(alpha)) * c_norm / (hn + eps) for hn in h_norms]
    elif mode in ("rho", "relative"):
        v = normalize_vector(raw, eps=eps).to(device=hidden.device, dtype=hidden.dtype)
        # alpha interpreted as rho: scale = rho * ||h|| / ||v||
        scales = (float(alpha) * selected.norm(dim=-1, keepdim=True) / (v.norm() + eps)).to(
            dtype=selected.dtype
        )
        steered[mask] = selected + scales * v
        rhos = [float(alpha)] * n_writes
    else:
        raise ValueError(f"unknown steering mode: {mode}")

    return steered, n_writes, h_norms, rhos


def attach_steering_hook(model: Any, spec: SteeringSpec):
    from ssopd_math.nxt.extract import decoder_layers

    layers = decoder_layers(model)
    layer = int(spec.layer)

    def hook(_mod, _inp, output):
        hidden, rest = _unwrap_output(output)
        if spec.response_mask is not None:
            mask = spec.response_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(hidden.shape[0], -1)
            # Align length if cached decode passes T==1
            if mask.shape[1] != hidden.shape[1]:
                if hidden.shape[1] == 1:
                    mask = torch.ones(
                        hidden.shape[0], 1, dtype=torch.bool, device=hidden.device
                    )
                else:
                    mask = mask[:, : hidden.shape[1]].to(device=hidden.device)
        else:
            mask = torch.ones(
                hidden.shape[0], hidden.shape[1], dtype=torch.bool, device=hidden.device
            )
        steered, n_writes, h_norms, rhos = steer_response_mask(
            hidden,
            spec.vector,
            spec.alpha,
            mask,
            mode=spec.mode,
        )
        spec.injection_counts.append(n_writes)
        spec.h_norms.extend(h_norms)
        spec.rho_values.extend(rhos)
        return _rewrap_output(steered, rest)

    return layers[layer].register_forward_hook(hook)


def remove_hook(handle) -> None:
    try:
        handle.remove()
    except Exception:
        pass
