#!/usr/bin/env python3
"""Shared helpers for BFCL tool-propensity steering (archive)."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import SSOPD_ROOT  # noqa: E402

for p in (str(SSOPD_ROOT), str(SSOPD_ROOT / "ssopd_logits")):
    if p not in sys.path:
        sys.path.insert(0, p)

from environments.bfcl_adapter import (  # noqa: E402
    bfcl_row_to_task,
    format_bfcl_tools_block,
    load_bfcl_category,
)
from rollout.parser import parse_tool_call  # noqa: E402
from rollout.verifier import verify_decision_fields  # noqa: E402

BFCL_SYSTEM = (
    "You are a tool-using assistant. If one of the listed tools can answer "
    "the user, return only a native tool-call in this exact form:\n"
    "<tool_call>\n"
    '{"name": "tool_name", "arguments": {}}\n'
    "</tool_call>\n"
    "If none of the tools are relevant, reply with a short natural-language "
    "answer and do not call a tool. Do not invent tools. Do not reason aloud."
)

QWEN_TOOL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
TOOL_PREFIXES = ("<tool_call", "<｜tool▁calls▁begin｜")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_irrelevance_tasks(root: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(root / "BFCL_v3_irrelevance.json")
    tasks: list[dict[str, Any]] = []
    for row in rows:
        task = bfcl_row_to_task(row, None, category="multiple", split="select")
        task["bfcl_category"] = "irrelevance"
        task["should_call"] = False
        task["gold_plan"] = []
        tasks.append(task)
    return tasks


def load_multiple_tasks(root: Path) -> list[dict[str, Any]]:
    tasks = load_bfcl_category("multiple", root=root)
    for t in tasks:
        t["should_call"] = True
        t["bfcl_category"] = "multiple"
    return tasks


def split_category(
    tasks: list[dict[str, Any]],
    *,
    seed: int,
    n_select: int,
    n_eval: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(tasks)))
    rng.shuffle(idxs)
    select_i = idxs[:n_select]
    eval_i = idxs[n_select : n_select + n_eval]
    select = []
    holdout = []
    for i in select_i:
        t = dict(tasks[i])
        t["split"] = "select"
        select.append(t)
    for i in eval_i:
        t = dict(tasks[i])
        t["split"] = "confirm"
        holdout.append(t)
    return select, holdout


def build_splits(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    root = Path(cfg["bfcl"]["root"])
    seed = int(cfg["bfcl"]["split_seed"])
    n_sel = int(cfg["bfcl"]["n_select_per_cat"])
    n_eval = int(cfg["bfcl"]["n_eval_per_cat"])
    mult_s, mult_e = split_category(
        load_multiple_tasks(root), seed=seed, n_select=n_sel, n_eval=n_eval
    )
    irr_s, irr_e = split_category(
        load_irrelevance_tasks(root), seed=seed + 1, n_select=n_sel, n_eval=n_eval
    )
    select = mult_s + irr_s
    holdout = mult_e + irr_e
    select.sort(key=lambda t: str(t["task_id"]))
    holdout.sort(key=lambda t: str(t["task_id"]))
    return {"select": select, "eval": holdout}


def user_prompt(task: dict[str, Any]) -> str:
    q = str(task.get("instruction") or "").strip()
    tools = format_bfcl_tools_block(task)
    return (
        f"User request:\n{q}\n\nAvailable tools:\n{tools}\n\n"
        "If a listed tool is relevant, emit exactly one <tool_call> JSON block. "
        "Otherwise answer in plain text and do not call a tool."
    )


def parse_any_tool_call(text: str) -> Any:
    parsed = parse_tool_call(text)
    if parsed.parse_ok:
        return parsed
    m = QWEN_TOOL_RE.search(text or "")
    if not m:
        return parsed
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return parsed
    if isinstance(obj, dict) and obj.get("name"):
        parsed.tool_name = str(obj["name"])
        args = obj.get("arguments") or {}
        parsed.arguments = args if isinstance(args, dict) else {}
        parsed.parse_ok = True
        parsed.format = "qwen_tool_call"
    return parsed


def score_completion(task: dict[str, Any], text: str) -> dict[str, Any]:
    parsed = parse_any_tool_call(text)
    called = bool(parsed.parse_ok and parsed.tool_name)
    should = bool(task.get("should_call"))
    tool_correct = False
    parse_ok = bool(parsed.parse_ok)
    if should:
        plan0 = (task.get("gold_plan") or [{}])[0]
        state = {
            "valid_tools": task.get("valid_tools") or [],
            "gold_tool": plan0.get("tool_name"),
            "gold_arguments": plan0.get("arguments") or {},
            "argument_options": plan0.get("argument_options"),
            "step_index": 0,
        }
        verd = verify_decision_fields(
            state,
            {
                "parse_ok": parsed.parse_ok,
                "tool_name": parsed.tool_name,
                "arguments": parsed.arguments,
            },
            task=task,
        )
        tool_correct = bool(verd.get("tool_correct"))
        parse_ok = bool(verd.get("parse_ok"))
    return {
        "task_id": task["task_id"],
        "should_call": should,
        "called": called,
        "parse_ok": parse_ok,
        "tool_correct": tool_correct,
        "pred_tool": parsed.tool_name,
        "bfcl_category": task.get("bfcl_category"),
        "text": text,
    }


def f1_should_call(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(1 for r in rows if r["should_call"] and r["called"])
    fp = sum(1 for r in rows if (not r["should_call"]) and r["called"])
    fn = sum(1 for r in rows if r["should_call"] and (not r["called"]))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rel = [r for r in rows if r["should_call"]]
    irr = [r for r in rows if not r["should_call"]]
    f1 = f1_should_call(rows)
    return {
        "n": n,
        "parse_ok_relevance": (
            sum(1 for r in rel if r["parse_ok"]) / len(rel) if rel else 0.0
        ),
        "call_rate": sum(1 for r in rows if r["called"]) / n if n else 0.0,
        "call_rate_relevance": (
            sum(1 for r in rel if r["called"]) / len(rel) if rel else 0.0
        ),
        "call_rate_irrelevance": (
            sum(1 for r in irr if r["called"]) / len(irr) if irr else 0.0
        ),
        "tool_correct_relevance": (
            sum(1 for r in rel if r["tool_correct"]) / len(rel) if rel else 0.0
        ),
        **f1,
    }


def paired_f1_bootstrap(
    base_rows: list[dict[str, Any]],
    steered_rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    gb = {r["task_id"]: r for r in base_rows}
    gs = {r["task_id"]: r for r in steered_rows}
    common = sorted(set(gb) & set(gs))
    if not common:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_tasks": 0}
    rng = random.Random(seed)

    def f1_of(ids: list[str], table: dict[str, dict]) -> float:
        return f1_should_call([table[i] for i in ids])["f1"]

    mean = f1_of(common, gs) - f1_of(common, gb)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [common[rng.randrange(len(common))] for _ in range(len(common))]
        boots.append(f1_of(sample, gs) - f1_of(sample, gb))
    boots.sort()
    return {
        "mean": mean,
        "ci_lo": boots[int(0.025 * n_boot)],
        "ci_hi": boots[int(0.975 * n_boot)],
        "n_tasks": float(len(common)),
    }


def prefix_logp(agent: Any, prompt: str, prefix: str) -> float:
    import torch

    device = agent._resolve_device()
    enc_p = agent.tokenizer(prompt, return_tensors="pt")
    enc_f = agent.tokenizer(prompt + prefix, return_tensors="pt")
    plen = int(enc_p["input_ids"].shape[1])
    ids = enc_f["input_ids"].to(device)
    n_pref = int(ids.shape[1]) - plen
    if n_pref <= 0:
        return 0.0
    with torch.no_grad():
        logits = agent.model(ids, use_cache=False).logits
    pred = logits[0, plen - 1 : plen - 1 + n_pref].float()
    targets = ids[0, plen : plen + n_pref]
    logp = torch.log_softmax(pred, dim=-1)
    tok = logp.gather(1, targets.view(-1, 1)).squeeze(1)
    return float(tok.mean().item())


def max_tool_prefix_logp(agent: Any, prompt: str) -> float:
    return max(prefix_logp(agent, prompt, p) for p in TOOL_PREFIXES)


def fit_v_tool(hiddens: list[np.ndarray], called: list[bool]) -> np.ndarray | None:
    pos = [h for h, c in zip(hiddens, called) if c]
    neg = [h for h, c in zip(hiddens, called) if not c]
    if not pos or not neg:
        return None
    v = np.mean(np.stack(pos, axis=0), axis=0) - np.mean(np.stack(neg, axis=0), axis=0)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return (v / n).astype(np.float32)
