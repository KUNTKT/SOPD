"""Join EXP02 labels with EXP01 rollouts and locate EXP03 token sites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssopd_agent.labeling.tool_selection import NEGATIVE, PARSE_FAILURE, POSITIVE
from ssopd_agent.representation.positions import locate_sites, pack_to_meta


SCORABLE = {POSITIVE, NEGATIVE}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def join_decisions(
    rollouts: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traj = {r["trajectory_id"]: r for r in rollouts}
    out = []
    for lab in labels:
        rec = traj[lab["trajectory_id"]]
        t = int(lab["step_id"])
        steps = (rec.get("task") or {}).get("steps") or []
        plan = [str(s["gold_tool"]) for s in steps] or list(rec.get("gold_tools") or [])
        template_id = f"{rec.get('kind', 'single')}:{'>'.join(plan)}"
        task_family = str((rec.get("task") or {}).get("gold_tool") or rec.get("gold_tools", ["unknown"])[0])
        action = rec["actions"][t]
        obs = rec["observations"][t] if t < len(rec["observations"]) else rec["observations"][-1]
        out.append(
            {
                "task_id": lab["task_id"],
                "template_id": template_id,
                "task_family": task_family,
                "trajectory_id": lab["trajectory_id"],
                "step_id": t,
                "tool_name_gold": lab["gold_tool"],
                "tool_name_predicted": lab.get("selected_tool"),
                "tool_selection_label": lab["tool_selection_label"],
                "trajectory_length": rec.get("trajectory_length"),
                "kind": rec.get("kind"),
                "observation": obs,
                "action": action,
            }
        )
    return out


def attach_positions(tokenizer, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    located = []
    for row in rows:
        pack = locate_sites(
            tokenizer,
            row["observation"],
            row["action"],
            predicted_tool=row.get("tool_name_predicted"),
        )
        meta = pack_to_meta(pack)
        item = dict(row)
        item.pop("observation", None)
        item.pop("action", None)
        item["prompt_length"] = meta["prompt_length"]
        item["context_length"] = meta["context_length"]
        item["marker_found"] = meta["marker_found"]
        item["sites"] = meta["sites"]
        item["input_ids"] = pack.input_ids
        located.append(item)
    return located


def main_site_is_tool_name(row: dict[str, Any]) -> bool:
    pred = (row.get("tool_name_predicted") or "").strip().lower()
    if not pred:
        return False
    text = str(row["sites"]["pre_call_reasoning"]["token_text"]).strip().lower()
    return text == pred


def position_fingerprint(rows: list[dict[str, Any]]) -> str:
    import hashlib

    h = hashlib.sha256()
    for row in rows:
        key = (
            row["trajectory_id"],
            row["step_id"],
            row["sites"]["pre_call_reasoning"]["token_position"],
            row["sites"]["pre_call_reasoning"]["token_id"],
            row["sites"]["call_marker"]["token_position"],
            row["sites"]["pre_tool_name"]["token_position"],
            int(row["sites"]["pre_call_reasoning"]["representation_fallback"]),
        )
        h.update(repr(key).encode())
    return h.hexdigest()[:16]
