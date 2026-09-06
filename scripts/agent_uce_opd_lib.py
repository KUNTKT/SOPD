#!/usr/bin/env python3
"""Shared splits, hashes, command masks, and reverse-KL for UCE-OPD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alfworld_common import load_eval_tasks, load_jsonl, load_task_pools, load_yaml_cfg
from uce_library import UceLibrary
from uce_whole_action import apply_workflow, command_content_mask, extract_command, strip_workflow


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def load_opd_cfg(path: str | Path | None = None) -> dict[str, Any]:
    default = Path(__file__).resolve().parents[1] / "configs/experiment_uce_opd.yaml"
    return load_yaml_cfg(Path(path) if path else default)


def evolve_ids_from_jsonl(path: Path) -> set[str]:
    return {str(r["task_id"]) for r in load_jsonl(path) if r.get("task_id")}


def _compute_task_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    evolve_n = int(cfg["env"].get("evolve_n", 80))
    limit = int(cfg.get("distill", {}).get("select_limit", 120))
    pools = load_task_pools(
        data_root=str(cfg["env"]["data_root"]),
        limit=1200,
        split="train",
        partition_seed=int(cfg.get("random_seed", 1010)),
    )
    audit = list(pools["pools"]["audit_select"])
    confirm = list(pools["pools"]["audit_confirm"])
    sft = list(pools["pools"]["sft"])
    skip = {str(t["task_id"]) for t in audit[:evolve_n]}
    evo_jsonl = Path(cfg["paths"]["evolve_jsonl"])
    if evo_jsonl.exists():
        skip |= evolve_ids_from_jsonl(evo_jsonl)
    evolve = [t for t in audit if str(t["task_id"]) in skip]
    # Keep evolve list in original audit order, first 80 if jsonl matches.
    evolve_ordered = [t for t in audit[:evolve_n]]
    rest = [t for t in audit if str(t["task_id"]) not in skip]
    distill = rest[:limit]
    eval_tasks, eval_meta = load_eval_tasks(cfg)
    payload = {
        "pools_meta": pools["meta"],
        "audit_select": audit,
        "audit_confirm": confirm,
        "sft": sft,
        "evolve": evolve_ordered,
        "evolve_ids": [str(t["task_id"]) for t in evolve_ordered],
        "distill": distill,
        "distill_ids": [str(t["task_id"]) for t in distill],
        "eval": eval_tasks,
        "eval_ids": [str(t["task_id"]) for t in eval_tasks],
        "eval_meta": eval_meta,
        "n_evolve_extra": len(evolve) - len(evolve_ordered),
    }
    cache = Path(cfg["paths"]["reports_dir"]) / "task_splits_cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, default=str))
    return payload


def task_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    cache = Path(cfg["paths"]["reports_dir"]) / "task_splits_cache.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        if len(raw.get("distill_ids") or []) == int(cfg.get("distill", {}).get("select_limit", 120)):
            return raw
    return _compute_task_splits(cfg)


def ids_of(tasks: list[dict]) -> set[str]:
    return {str(t["task_id"]) for t in tasks}


def completion_ids(tokenizer: Any, action_text: str) -> list[int]:
    text = str(action_text or "").strip()
    if not text.upper().startswith("ACTION"):
        cmd = extract_command(text)
        text = f"ACTION: {cmd}" if cmd else "ACTION: look"
    ids = tokenizer.encode(text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None and (not ids or int(ids[-1]) != int(eos)):
        ids = list(ids) + [int(eos)]
    return ids


def supervised_mask(tokenizer: Any, completion_ids_: list[int]) -> list[bool]:
    mask = command_content_mask(tokenizer, completion_ids_)
    eos = tokenizer.eos_token_id
    if eos is not None:
        for i, tid in enumerate(completion_ids_):
            if int(tid) == int(eos):
                mask[i] = True
    return mask


def reverse_kl_token(log_ps: torch.Tensor, log_pt: torch.Tensor) -> torch.Tensor:
    """KL(p_S || p_T) from log-softmax vectors."""
    return (log_ps.exp() * (log_ps - log_pt)).sum(dim=-1)


def mean_step_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: list[int],
    mask: list[bool],
    *,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """student_logits / teacher_logits: [L, V] aligned to completion tokens (predict token k)."""
    ts = max(float(student_temperature), 1e-8)
    tt = max(float(teacher_temperature), 1e-8)
    log_ps = F.log_softmax(student_logits.float() / ts, dim=-1)
    log_pt = F.log_softmax(teacher_logits.float() / tt, dim=-1).detach()
    kls = reverse_kl_token(log_ps, log_pt)
    weights = []
    vals = []
    for i, keep in enumerate(mask):
        if keep and i < kls.shape[0]:
            vals.append(kls[i])
            weights.append(1.0)
    if not vals:
        return student_logits.sum() * 0.0
    return torch.stack(vals).mean()


def retrieve_readonly(lib: UceLibrary, task: dict) -> tuple[str, str | None, float]:
    before = json.dumps([e.get("usage") for e in lib.entries], sort_keys=True)
    text, eid, score = lib.retrieve(task)
    after = json.dumps([e.get("usage") for e in lib.entries], sort_keys=True)
    if before != after:
        raise RuntimeError("UCE retrieve mutated usage")
    return text, eid, score


def completion_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    comp_ids: list[int],
    *,
    device: torch.device,
    max_prompt_tokens: int = 1536,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    keep = max(8, int(max_prompt_tokens) - len(comp_ids))
    if prompt_ids.shape[1] > keep:
        prompt_ids = prompt_ids[:, -keep:]
    comp = torch.tensor([comp_ids], device=device, dtype=prompt_ids.dtype)
    ids = torch.cat([prompt_ids, comp], dim=1)
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0]
    plen = int(prompt_ids.shape[1])
    sliced = logits[plen - 1 : plen - 1 + len(comp_ids)]
    del out
    return sliced


def set_adapter_enabled(model: Any, enabled: bool) -> None:
    if hasattr(model, "disable_adapter") and hasattr(model, "enable_adapter_layers"):
        if enabled:
            model.enable_adapter_layers()
        else:
            model.disable_adapter_layers()
        return
    if hasattr(model, "disable_adapters"):
        # newer peft context; fall back to train/eval flags
        pass
    if hasattr(model, "set_adapter"):
        pass


def build_theta0_agent(cfg: dict[str, Any]):
    """Merged coldstart only — frozen teacher / shared θ0."""
    from copy import deepcopy

    from peft import PeftModel

    from steerable_alfworld_agent import build_agent

    cfg2 = deepcopy(cfg)
    cfg2.setdefault("model", {})["adapter_path"] = None
    agent = build_agent(cfg2)
    cold = cfg["paths"].get("coldstart_adapter") or cfg["model"].get("adapter_path")
    if cold:
        agent.model = PeftModel.from_pretrained(agent.model, str(cold), is_trainable=False)
        agent.model = agent.model.merge_and_unload()
    agent.model.eval()
    for p in agent.model.parameters():
        p.requires_grad_(False)
    return agent


def build_student_agent(
    cfg: dict[str, Any],
    *,
    distill_path: Path | None = None,
    trainable: bool = False,
):
    """Base + merged coldstart + distill LoRA (new or loaded)."""
    from copy import deepcopy

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    from steerable_alfworld_agent import build_agent

    cfg2 = deepcopy(cfg)
    cfg2.setdefault("model", {})["adapter_path"] = None
    agent = build_agent(cfg2)
    cold = cfg["paths"].get("coldstart_adapter") or cfg["model"].get("adapter_path")
    if cold:
        agent.model = PeftModel.from_pretrained(agent.model, str(cold), is_trainable=False)
        agent.model = agent.model.merge_and_unload()
    dcfg = cfg.get("distill", {})
    if distill_path is not None and Path(distill_path).exists() and (Path(distill_path) / "adapter_config.json").exists():
        agent.model = PeftModel.from_pretrained(
            agent.model, str(distill_path), is_trainable=trainable
        )
    else:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(dcfg.get("lora_r", 16)),
            lora_alpha=int(dcfg.get("lora_alpha", 32)),
            lora_dropout=float(dcfg.get("lora_dropout", 0.05)),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        agent.model = get_peft_model(agent.model, lora)
    if trainable:
        for n, p in agent.model.named_parameters():
            p.requires_grad_(("lora_" in n))
        agent.model.train()
        if hasattr(agent.model, "enable_adapter_layers"):
            agent.model.enable_adapter_layers()
    else:
        agent.model.eval()
    return agent


class AdapterSwitch:
    """Enable distill LoRA for student; disable for frozen teacher (merged θ0)."""

    def __init__(self, model: Any):
        self.model = model

    def student(self) -> None:
        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()

    def teacher(self) -> None:
        if hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()
