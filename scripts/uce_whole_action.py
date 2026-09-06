#!/usr/bin/env python3
"""Whole-action scores and JS for UCE counterfactual diagnostics (H5 offline)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from uce_counterfactual_logit import solve_beta  # noqa: E402


WORKFLOW_HEAD = "Suggested workflow"
ACTION_PREFIXES = ("ACTION:", "Action:", "action:")


def apply_workflow(obs: str, workflow_text: str | None) -> str:
    if not workflow_text:
        return str(obs or "")
    return f"{workflow_text}\n\n{obs}"


def strip_workflow(obs: str) -> str:
    text = str(obs or "")
    if text.startswith(WORKFLOW_HEAD):
        idx = text.find("\n\n")
        if idx >= 0:
            return text[idx + 2 :]
    return text


def extract_command(raw: str) -> str:
    text = str(raw or "").strip()
    for prefix in ACTION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text.split("\n")[0].strip()


def match_admissible(command: str, actions: list[str]) -> str | None:
    if command in actions:
        return command
    folded = {a.casefold(): a for a in actions}
    return folded.get(command.casefold())


def scores_to_logq(scores: torch.Tensor, tau_a: float = 1.0) -> torch.Tensor:
    t = max(float(tau_a), 1e-8)
    return F.log_softmax(scores.float() / t, dim=-1)


def js_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = log_p.float().exp()
    q = log_q.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-12).log()
    js = 0.5 * (p * (log_p.float() - log_m)).sum() + 0.5 * (q * (log_q.float() - log_m)).sum()
    return float(js)


def steered_logq(
    s0: torch.Tensor,
    s_plus: torch.Tensor,
    beta: float,
    *,
    tau_a: float = 1.0,
) -> torch.Tensor:
    log_q0 = scores_to_logq(s0, tau_a)
    log_qp = scores_to_logq(s_plus, tau_a)
    g = log_qp + float(beta) * (log_qp - log_q0)
    return F.log_softmax(g, dim=-1)


def action_kl_at_beta(
    s0: torch.Tensor,
    s_plus: torch.Tensor,
    beta: float,
    *,
    tau_a: float = 1.0,
) -> float:
    from uce_counterfactual_logit import kl_logp

    log_qp = scores_to_logq(s_plus, tau_a)
    log_qt = steered_logq(s0, s_plus, beta, tau_a=tau_a)
    return float(kl_logp(log_qt, log_qp))


def action_reachability(
    s0: torch.Tensor,
    s_plus: torch.Tensor,
    targets: list[float],
    *,
    beta_max: float = 4.0,
    tau_a: float = 1.0,
    bisection_steps: int = 12,
) -> dict[str, dict[str, float | bool]]:
    out: dict[str, dict[str, float | bool]] = {}
    for delta in targets:
        beta, ach, kmax, unreach = solve_beta(
            s0,
            s_plus,
            float(delta),
            beta_max=float(beta_max),
            bisection_steps=int(bisection_steps),
            kl_temperature=float(tau_a),
        )
        out[f"{float(delta):.2f}"] = {
            "beta": float(beta),
            "achieved_kl": float(ach),
            "k_max": float(kmax),
            "unreachable": bool(unreach),
        }
    return out


def command_content_mask(tokenizer: Any, completion_ids: list[int]) -> list[bool]:
    """Tokens after ACTION / colon. Qwen fuses the space into ' go', so the
    H4 next_token_is_protocol('ACTION:') test would drop the verb (and make
    one-token commands like 'look' all-NaN). Colon-split matches the plan:
    format = ACTION + colon; everything after is command content.
    """
    mask: list[bool] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    seen_colon = False
    for tid in completion_ids:
        if eos_id is not None and int(tid) == int(eos_id):
            mask.append(False)
            continue
        piece = tokenizer.decode([int(tid)], skip_special_tokens=True)
        if not seen_colon:
            if ":" in piece:
                seen_colon = True
            mask.append(False)
            continue
        mask.append(True)
    if seen_colon and not any(mask):
        # Fallback: never return an empty command span.
        for i in range(len(mask)):
            tid = completion_ids[i]
            if eos_id is not None and int(tid) == int(eos_id):
                continue
            mask[i] = True
            break
    return mask


def first_command_index(mask: list[bool]) -> int | None:
    for i, flag in enumerate(mask):
        if flag:
            return i
    return None


def mean_masked(values: list[float], mask: list[bool]) -> float:
    picked = [v for v, m in zip(values, mask) if m]
    if not picked:
        return float("nan")
    return float(sum(picked) / len(picked))


def rank_of(scores: list[float], index: int) -> int:
    """1 = best (highest score). Ties: earlier index wins."""
    better = sum(1 for i, s in enumerate(scores) if (s > scores[index]) or (s == scores[index] and i < index))
    return int(better + 1)


def summarize_pair(
    actions: list[str],
    s0: list[float],
    s_plus: list[float],
    *,
    executed: str | None,
    first_cmd_token_0: int | None,
    first_cmd_token_plus: int | None,
    tau_a: float = 1.0,
    beta_max: float = 4.0,
    kl_targets: tuple[float, ...] = (0.02, 0.05),
) -> dict[str, Any]:
    t0 = torch.tensor(s0, dtype=torch.float32)
    tp = torch.tensor(s_plus, dtype=torch.float32)
    log_q0 = scores_to_logq(t0, tau_a)
    log_qp = scores_to_logq(tp, tau_a)
    i0 = int(t0.argmax().item())
    ip = int(tp.argmax().item())
    top1_diff = actions[i0] != actions[ip]
    first0 = actions[i0].split()[0] if actions[i0].split() else ""
    firstp = actions[ip].split()[0] if actions[ip].split() else ""
    token_top1_diff = (
        first_cmd_token_0 is not None
        and first_cmd_token_plus is not None
        and int(first_cmd_token_0) != int(first_cmd_token_plus)
    )
    exec_idx = match_admissible(executed or "", actions)
    exec_i = actions.index(exec_idx) if exec_idx is not None else None
    rec: dict[str, Any] = {
        "n_actions": len(actions),
        "top1_base": actions[i0],
        "top1_uce": actions[ip],
        "whole_action_top1_diff": bool(top1_diff),
        "first_word_top1_diff": first0 != firstp,
        "first_cmd_token_top1_diff": bool(token_top1_diff),
        "token_same_action_diff": (not token_top1_diff) and bool(top1_diff),
        "js": js_divergence(log_qp, log_q0),
        "executed": executed,
        "executed_in_list": exec_i is not None,
        "rank_base": rank_of(s0, exec_i) if exec_i is not None else None,
        "rank_uce": rank_of(s_plus, exec_i) if exec_i is not None else None,
        "reach": action_reachability(t0, tp, list(kl_targets), beta_max=beta_max, tau_a=tau_a),
    }
    if rec["rank_base"] is not None and rec["rank_uce"] is not None:
        rec["rank_change"] = int(rec["rank_base"]) - int(rec["rank_uce"])
    else:
        rec["rank_change"] = None
    return rec


@torch.inference_mode()
def score_actions(
    agent: Any,
    observation: str,
    actions: list[str],
    *,
    batch_size: int = 8,
) -> tuple[list[float], int | None]:
    """Length-normalized command-content logp for each action; first-cmd vocab top-1."""
    if not actions:
        return [], None
    tokenizer = agent.tokenizer
    device = agent._resolve_device()
    model = agent.model
    model.eval()
    prompt = agent._prompt_text(observation)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    plen = len(prompt_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    scores: list[float] = [float("nan")] * len(actions)
    first_top1: int | None = None

    proto_ids = tokenizer.encode("ACTION:", add_special_tokens=False)
    proto_full = torch.tensor([prompt_ids + proto_ids], device=device, dtype=torch.long)
    proto_out = model(input_ids=proto_full, use_cache=False)
    first_top1 = int(proto_out.logits[0, -1].float().argmax().item())
    del proto_out

    for start in range(0, len(actions), int(batch_size)):
        chunk = actions[start : start + int(batch_size)]
        comps = [tokenizer.encode(f"ACTION: {a}", add_special_tokens=False) for a in chunk]
        max_c = max((len(c) for c in comps), default=0)
        if max_c == 0:
            continue
        rows = []
        masks = []
        for cids in comps:
            pad_n = max_c - len(cids)
            rows.append(prompt_ids + cids + [int(pad_id)] * pad_n)
            masks.append([1] * (plen + len(cids)) + [0] * pad_n)
        input_ids = torch.tensor(rows, device=device, dtype=torch.long)
        attn = torch.tensor(masks, device=device, dtype=torch.long)
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for i, cids in enumerate(comps):
            cmask = command_content_mask(tokenizer, cids)
            vals: list[float] = []
            for k, tid in enumerate(cids):
                # token k of completion is predicted by logits at plen+k-1
                pos = plen + k - 1
                if pos < 0:
                    continue
                vals.append(float(logp[i, pos, int(tid)].item()))
            scores[start + i] = mean_masked(vals, cmask)
        del logits, logp, input_ids, attn
    return scores, first_top1
