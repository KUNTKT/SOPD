"""Load BFCL JSONL and adapt into Logit-SSOPD task schema.

Protocol:
- Use non-live `multiple` (and optionally `simple`) for selection/confirm splits.
- Gold / tool docs come from BFCL; evidence & teacher logits still from own rollouts.
- Never tune on the confirm/held-out split.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Literal

DATASET_VERSION = "bfcl_v3_adapted_v1"

BfclCategory = Literal["simple", "multiple", "live_simple", "live_multiple"]
BfclSplit = Literal["select", "confirm"]

DEFAULT_BFCL_ROOT = Path(__file__).resolve().parents[1] / "data" / "bfcl" / "hf_raw"

_CATEGORY_FILES: dict[str, str] = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_text(question: Any) -> str:
    """BFCL question is nested list of chat turns; flatten to user text."""
    parts: list[str] = []
    if not isinstance(question, list):
        return str(question)
    for turn_group in question:
        if not isinstance(turn_group, list):
            continue
        for msg in turn_group:
            if isinstance(msg, dict) and msg.get("role") == "user":
                parts.append(str(msg.get("content", "")))
            elif isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
    return "\n".join(p for p in parts if p).strip()


def _first_acceptable_args(arg_options: dict[str, list[Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, opts in arg_options.items():
        if not opts:
            continue
        # Prefer first non-empty option
        chosen = None
        for o in opts:
            if o != "" and o is not None:
                chosen = o
                break
        if chosen is None:
            continue
        out[k] = chosen
    return out


def _parse_ground_truth(gt_entry: dict[str, Any]) -> tuple[str, dict[str, list[Any]]]:
    """One ground-truth object maps {tool_name: {arg: [options...]}}."""
    if len(gt_entry) != 1:
        # Prefer first key deterministically
        tool_name = sorted(gt_entry.keys())[0]
    else:
        tool_name = next(iter(gt_entry.keys()))
    raw_args = gt_entry[tool_name]
    options: dict[str, list[Any]] = {}
    if isinstance(raw_args, dict):
        for k, v in raw_args.items():
            if isinstance(v, list):
                options[k] = list(v)
            else:
                options[k] = [v]
    return str(tool_name), options


def bfcl_row_to_task(
    row: dict[str, Any],
    answer: dict[str, Any] | None,
    *,
    category: str,
    split: BfclSplit,
) -> dict[str, Any]:
    funcs = list(row.get("function") or [])
    valid_tools = [str(f["name"]) for f in funcs if isinstance(f, dict) and "name" in f]
    instruction = _question_text(row.get("question"))
    bfcl_id = str(row.get("id", "unknown"))

    gold_tool = valid_tools[0] if valid_tools else None
    arg_options: dict[str, list[Any]] = {}
    gold_args: dict[str, Any] = {}
    if answer and answer.get("ground_truth"):
        gt_list = answer["ground_truth"]
        if isinstance(gt_list, list) and gt_list:
            first = gt_list[0]
            if isinstance(first, dict) and first:
                gold_tool, arg_options = _parse_ground_truth(first)
                gold_args = _first_acceptable_args(arg_options)

    plan = [
        {
            "step": 0,
            "tool_name": gold_tool,
            "arguments": gold_args,
            "argument_options": arg_options,
            "expected_result": "OK",
            "subgoal": "bfcl_single_call",
        }
    ]
    return {
        "task_id": f"bfcl_{bfcl_id}",
        "bfcl_id": bfcl_id,
        "instruction": instruction,
        "task_type": "single_step",
        "gold_plan": plan,
        "valid_tools": valid_tools,
        "function_docs": funcs,
        "template_id": f"bfcl_{category}",
        "difficulty": "bfcl",
        "seed": 0,
        "dataset_version": DATASET_VERSION,
        "n_steps": 1,
        "has_information_dependency": False,
        "premature_downstream_tool": None,
        "source": "bfcl",
        "bfcl_category": category,
        "split": split,
    }


def load_bfcl_category(
    category: BfclCategory,
    *,
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    root_p = Path(root) if root is not None else DEFAULT_BFCL_ROOT
    fname = _CATEGORY_FILES[category]
    rows = _load_jsonl(root_p / fname)
    ans_path = root_p / "possible_answer" / fname
    answers = {a["id"]: a for a in _load_jsonl(ans_path)} if ans_path.exists() else {}
    # placeholder split; caller assigns select/confirm
    return [
        bfcl_row_to_task(r, answers.get(r.get("id")), category=category, split="select")
        for r in rows
    ]


def split_bfcl_tasks(
    tasks: list[dict[str, Any]],
    *,
    select_frac: float = 0.5,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic held-out split. Confirm is frozen for one-shot use only."""
    rng = random.Random(seed)
    idxs = list(range(len(tasks)))
    rng.shuffle(idxs)
    n_select = int(round(len(tasks) * select_frac))
    select_i = set(idxs[:n_select])
    select: list[dict[str, Any]] = []
    confirm: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        t2 = dict(t)
        if i in select_i:
            t2["split"] = "select"
            select.append(t2)
        else:
            t2["split"] = "confirm"
            confirm.append(t2)
    return select, confirm


def load_bfcl_splits(
    category: BfclCategory = "multiple",
    *,
    root: Path | str | None = None,
    select_frac: float = 0.5,
    seed: int = 0,
    max_select: int | None = None,
    max_confirm: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    all_tasks = load_bfcl_category(category, root=root)
    select, confirm = split_bfcl_tasks(all_tasks, select_frac=select_frac, seed=seed)
    if max_select is not None:
        select = select[: int(max_select)]
    if max_confirm is not None:
        confirm = confirm[: int(max_confirm)]
    return {"select": select, "confirm": confirm, "all": all_tasks}


def tasks_fingerprint(tasks: list[dict[str, Any]]) -> str:
    blob = json.dumps(tasks, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def format_bfcl_tools_block(task: dict[str, Any]) -> str:
    docs = task.get("function_docs") or []
    if not docs:
        names = task.get("valid_tools") or []
        return "\n".join(f"- {n}" for n in names)
    lines: list[str] = []
    for fn in docs:
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters") or {}
        lines.append(f"- {name}: {desc}")
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        for pk, pv in props.items():
            req = "required" if pk in required else "optional"
            pdesc = pv.get("description", "") if isinstance(pv, dict) else ""
            ptype = pv.get("type", "") if isinstance(pv, dict) else ""
            lines.append(f"    - {pk} ({ptype}, {req}): {pdesc}")
    return "\n".join(lines)
