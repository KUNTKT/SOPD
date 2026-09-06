#!/usr/bin/env python3
"""UCE counterfactual action-logit steering (log-prob ratio, dual KV)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


KL_EPS = 1e-4
KMAX_ZERO = 1e-5
WORKFLOW_HEAD = "Suggested workflow"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Any) -> str:
    data = path.read_bytes() if hasattr(path, "read_bytes") else open(path, "rb").read()
    return hashlib.sha256(data).hexdigest()


def canonical_workflow(text: str) -> str:
    lines = [ln.rstrip() for ln in str(text or "").strip().splitlines()]
    return "\n".join(lines).strip()


def extract_workflow_block(text: str) -> str:
    """Pull the UCE workflow blob prepended to obs / buried in a chat prompt."""
    raw = str(text or "")
    idx = raw.find(WORKFLOW_HEAD)
    if idx < 0:
        return ""
    rest = raw[idx:]
    cut = rest.find("\n\n")
    if cut >= 0:
        rest = rest[:cut]
    return canonical_workflow(rest)


def workflow_sha256(text: str) -> str:
    return sha256_text(canonical_workflow(text))


def is_protocol_text(decoded: str) -> bool:
    """True while completion is still ACTION: format, not command content."""
    t = str(decoded or "").lstrip("\n\r")
    if t == "":
        return True
    marker = "ACTION:"
    if marker.startswith(t):
        return True
    if t.startswith(marker):
        rest = t[len(marker) :]
        if rest == "":
            return True
        return rest.strip() == ""
    return False


def next_token_is_protocol(decoded: str) -> bool:
    """Whether the upcoming token is still format (ACTION:/space), not command."""
    t = str(decoded or "").lstrip("\n\r")
    marker = "ACTION:"
    if t == "" or marker.startswith(t):
        return True
    if not t.startswith(marker):
        return False
    rest = t[len(marker) :]
    if rest == "":
        return True
    if rest.strip() == "" and not any(ch.isspace() for ch in rest):
        return True
    if rest.strip() == "" and any(ch.isspace() for ch in rest):
        return False
    return False


def log_softmax_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-8)
    return F.log_softmax(logits.float() / t, dim=-1)


def kl_logp(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = log_p.exp()
    return (p * (log_p - log_q)).sum()


def entropy_logp(log_p: torch.Tensor) -> float:
    p = log_p.exp()
    return float((-(p * log_p)).sum())


def guided_logp(
    z0: torch.Tensor,
    zp: torch.Tensor,
    beta: float,
    *,
    kl_temperature: float = 1.0,
) -> torch.Tensor:
    l0 = log_softmax_temp(z0, kl_temperature)
    lp = log_softmax_temp(zp, kl_temperature)
    g = lp + float(beta) * (lp - l0)
    return F.log_softmax(g, dim=-1)


def kl_at_beta(
    z0: torch.Tensor,
    zp: torch.Tensor,
    beta: float,
    *,
    kl_temperature: float = 1.0,
) -> float:
    lp = log_softmax_temp(zp, kl_temperature)
    log_pt = guided_logp(z0, zp, beta, kl_temperature=kl_temperature)
    return float(kl_logp(log_pt, lp))


def solve_beta(
    z0: torch.Tensor,
    zp: torch.Tensor,
    target_kl: float,
    *,
    beta_max: float = 4.0,
    bisection_steps: int = 12,
    kl_temperature: float = 1.0,
    eps: float = KL_EPS,
) -> tuple[float, float, float, bool]:
    """Return beta, achieved_kl, k_max, unreachable."""
    target = float(target_kl)
    k_max = kl_at_beta(z0, zp, beta_max, kl_temperature=kl_temperature)
    if k_max < KMAX_ZERO:
        return 0.0, 0.0, k_max, True
    if target <= 0.0:
        return 0.0, 0.0, k_max, False
    if k_max < target - eps:
        ach = kl_at_beta(z0, zp, beta_max, kl_temperature=kl_temperature)
        return float(beta_max), ach, k_max, True
    lo, hi = 0.0, float(beta_max)
    for _ in range(int(bisection_steps)):
        mid = 0.5 * (lo + hi)
        if kl_at_beta(z0, zp, mid, kl_temperature=kl_temperature) < target:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    ach = kl_at_beta(z0, zp, beta, kl_temperature=kl_temperature)
    return float(beta), float(ach), float(k_max), False


def sample_from_logp(
    log_p: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    logits = log_p.float()
    t = max(float(temperature), 1e-8)
    scores = logits / t
    if float(top_p) < 1.0 - 1e-12:
        sorted_logits, sorted_idx = torch.sort(scores, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > float(top_p)
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(sorted_logits.dtype).min)
        scores = torch.zeros_like(scores).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(scores, dim=-1)
    # Must sample on the same device as `generator`, otherwise RNG streams diverge.
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return idx.view(1)


def make_step_generator(seed: int, *, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


class KvBranch:
    """One prompt's KV cache; steps with prepare_inputs_for_generation."""

    def __init__(self, model: Any, input_ids: torch.Tensor):
        self.model = model
        self.device = input_ids.device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        self.input_ids = input_ids
        self.attention_mask = torch.ones_like(input_ids)
        out = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            use_cache=True,
        )
        self.past = out.past_key_values
        self.logits = out.logits[0, -1].float()
        self.seq_len = int(self.input_ids.shape[1])

    def step(self, token_id: int | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(token_id):
            tok = torch.tensor([[int(token_id)]], device=self.device, dtype=self.input_ids.dtype)
        else:
            tok = token_id.view(1, 1).to(self.device)
        ones = torch.ones(1, 1, device=self.device, dtype=self.attention_mask.dtype)
        self.attention_mask = torch.cat([self.attention_mask, ones], dim=1)
        cache_position = torch.tensor([self.seq_len], device=self.device, dtype=torch.long)
        # Keep the exact same step signature as HF generate's cached decoding.
        # Using prepare_inputs_for_generation() can subtly diverge for some
        # model implementations; direct forward is closer for parity.
        out = self.model(
            input_ids=tok,
            past_key_values=self.past,
            attention_mask=self.attention_mask,
            use_cache=True,
            cache_position=cache_position,
        )
        self.past = out.past_key_values
        self.logits = out.logits[0, -1].float()
        self.input_ids = torch.cat([self.input_ids, tok], dim=1)
        self.seq_len += 1
        return self.logits


@dataclass
class TokenDiag:
    is_protocol: bool
    beta: float
    kl_vs_uce: float
    kl_vs_base: float
    k_max: float
    unreachable: bool
    top1_flip_uce: bool
    top1_flip_base: bool
    uce_entropy: float
    teacher_entropy: float
    token_id: int


@dataclass
class CfGenerateResult:
    text: str
    token_ids: list[int]
    token_diags: list[TokenDiag] = field(default_factory=list)
    prompt_text_base: str = ""
    prompt_text_uce: str = ""


def _encode_prompt(agent: Any, text: str) -> torch.Tensor:
    device = agent._resolve_device()
    encoded = agent.tokenizer(text, return_tensors="pt")
    return encoded["input_ids"].to(device)


@torch.inference_mode()
def generate_counterfactual_action(
    agent: Any,
    base_obs: str,
    uce_obs: str,
    *,
    target_kl: float,
    step_seed: int,
    max_new_tokens: int | None = None,
    beta_max: float = 4.0,
    bisection_steps: int = 12,
    kl_temperature: float = 1.0,
    sample_temperature: float = 1.0,
    top_p: float = 1.0,
) -> CfGenerateResult:
    agent.model.eval()
    tokenizer = agent.tokenizer
    eos_id = tokenizer.eos_token_id
    n_new = int(max_new_tokens if max_new_tokens is not None else agent.max_new_tokens)
    prompt0 = agent._prompt_text(base_obs)
    promptp = agent._prompt_text(uce_obs)
    ids0 = _encode_prompt(agent, prompt0)
    idsp = _encode_prompt(agent, promptp)
    branch0 = KvBranch(agent.model, ids0)
    branchp = KvBranch(agent.model, idsp)
    device = agent._resolve_device()
    gen = make_step_generator(step_seed, device=device)
    out_ids: list[int] = []
    diags: list[TokenDiag] = []

    for _ in range(n_new):
        z0 = branch0.logits
        zp = branchp.logits
        decoded = tokenizer.decode(out_ids, skip_special_tokens=True)
        protocol = next_token_is_protocol(decoded)
        lp = log_softmax_temp(zp, kl_temperature)
        l0 = log_softmax_temp(z0, kl_temperature)
        k_max = kl_at_beta(z0, zp, beta_max, kl_temperature=kl_temperature)
        if protocol or float(target_kl) <= 0.0:
            beta, ach, unreachable = 0.0, 0.0, False
            log_pt = lp
        else:
            beta, ach, k_max, unreachable = solve_beta(
                z0,
                zp,
                float(target_kl),
                beta_max=beta_max,
                bisection_steps=bisection_steps,
                kl_temperature=kl_temperature,
            )
            if k_max < KMAX_ZERO:
                log_pt = lp
                beta, ach = 0.0, 0.0
            else:
                log_pt = guided_logp(z0, zp, beta, kl_temperature=kl_temperature)
        tok = sample_from_logp(
            log_pt,
            generator=gen,
            temperature=sample_temperature,
            top_p=top_p,
        )
        tid = int(tok.item())
        is_proto = bool(protocol) or tid == eos_id
        top_t = int(log_pt.argmax().item())
        top_u = int(lp.argmax().item())
        top_b = int(l0.argmax().item())
        kl_base = float(kl_logp(log_pt, l0))
        diags.append(
            TokenDiag(
                is_protocol=is_proto,
                beta=float(beta) if not is_proto else 0.0,
                kl_vs_uce=float(ach) if not is_proto else 0.0,
                kl_vs_base=kl_base if not is_proto else 0.0,
                k_max=float(k_max),
                unreachable=bool(unreachable) if not is_proto else False,
                top1_flip_uce=top_t != top_u and not is_proto,
                top1_flip_base=top_t != top_b and not is_proto,
                uce_entropy=entropy_logp(lp),
                teacher_entropy=entropy_logp(log_pt),
                token_id=tid,
            )
        )
        out_ids.append(tid)
        if tid == eos_id:
            break
        branch0.step(tid)
        branchp.step(tid)

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return CfGenerateResult(
        text=text,
        token_ids=out_ids,
        token_diags=diags,
        prompt_text_base=prompt0,
        prompt_text_uce=promptp,
    )


def aggregate_token_diags(diags: list[TokenDiag]) -> dict[str, float]:
    if not diags:
        return {
            "mean_beta": 0.0,
            "max_beta": 0.0,
            "mean_achieved_kl_vs_uce": 0.0,
            "first_action_kl_vs_uce": 0.0,
            "first_action_kl_vs_base": 0.0,
            "protocol_token_kl": 0.0,
            "command_token_kl": 0.0,
            "command_top1_flip_rate_vs_uce": 0.0,
            "command_top1_flip_rate_vs_base": 0.0,
            "action_top1_flip_rate_vs_uce": 0.0,
            "action_top1_flip_rate_vs_base": 0.0,
            "target_unreachable_rate": 0.0,
            "mean_uce_entropy": 0.0,
            "mean_teacher_entropy": 0.0,
            "n_protocol": 0.0,
            "n_command": 0.0,
        }
    proto = [d for d in diags if d.is_protocol]
    cmd = [d for d in diags if not d.is_protocol]
    first = cmd[0] if cmd else None

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    cmd_kl = mean([d.kl_vs_uce for d in cmd])
    proto_kl = mean([d.kl_vs_uce for d in proto])
    flip_u = mean([1.0 if d.top1_flip_uce else 0.0 for d in cmd])
    flip_b = mean([1.0 if d.top1_flip_base else 0.0 for d in cmd])
    return {
        "mean_beta": mean([d.beta for d in cmd]),
        "max_beta": max((d.beta for d in cmd), default=0.0),
        "mean_achieved_kl_vs_uce": cmd_kl,
        "first_action_kl_vs_uce": float(first.kl_vs_uce) if first else 0.0,
        "first_action_kl_vs_base": float(first.kl_vs_base) if first else 0.0,
        "protocol_token_kl": proto_kl,
        "command_token_kl": cmd_kl,
        "command_top1_flip_rate_vs_uce": flip_u,
        "command_top1_flip_rate_vs_base": flip_b,
        "action_top1_flip_rate_vs_uce": flip_u,
        "action_top1_flip_rate_vs_base": flip_b,
        "target_unreachable_rate": mean([1.0 if d.unreachable else 0.0 for d in cmd]),
        "mean_uce_entropy": mean([d.uce_entropy for d in diags]),
        "mean_teacher_entropy": mean([d.teacher_entropy for d in diags]),
        "n_protocol": float(len(proto)),
        "n_command": float(len(cmd)),
    }


def strict_paired_ids(recs_a: list[dict], recs_b: list[dict]) -> list[str]:
    ids_a = [str(r["task_id"]) for r in recs_a]
    ids_b = [str(r["task_id"]) for r in recs_b]
    if ids_a != ids_b:
        raise SystemExit(
            f"task_id mismatch: n_a={len(ids_a)} n_b={len(ids_b)} "
            f"first_diff={next((i for i,(x,y) in enumerate(zip(ids_a,ids_b)) if x!=y), 'len')}"
        )
    return ids_a


def paired_bootstrap_strict(
    recs_ctrl: list[dict],
    recs_h4: list[dict],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    import random

    strict_paired_ids(recs_ctrl, recs_h4)
    n = len(recs_ctrl)
    deltas = []
    wins = losses = ties = 0
    for a, b in zip(recs_ctrl, recs_h4):
        sa = 1.0 if a.get("episode_success") else 0.0
        sb = 1.0 if b.get("episode_success") else 0.0
        deltas.append(sb - sa)
        if sb > sa:
            wins += 1
        elif sb < sa:
            losses += 1
        else:
            ties += 1
    mean = sum(deltas) / n if n else 0.0
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    alpha = 1.0 - float(ci)
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1.0 - alpha / 2) * n_boot)
    hi_i = min(max(hi_i, 0), n_boot - 1)
    return {
        "mean": mean,
        "ci_lo": boots[lo_i],
        "ci_hi": boots[hi_i],
        "ci": ci,
        "n_tasks": n,
        "n_boot": n_boot,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "delta_pp": mean * 100.0,
    }
