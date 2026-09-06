#!/usr/bin/env python3
"""Mine per-task_type short workflows from select-pool success trajectories."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import dump_json, load_jsonl, load_yaml_cfg  # noqa: E402

VERB_RE = re.compile(
    r"^(go to|take|put|move|open|close|heat|cool|clean|use|examine|look|inventory)\b",
    re.I,
)
ACTION_PREFIX = re.compile(r"^\s*(?:ACTION|Action|action)\s*[:：]\s*", re.I)
INSTANCE_RE = re.compile(r"\b([a-z][a-z0-9]*)\s+\d+\b", re.I)

# Readable fallbacks if a type has too few successes.
CANONICAL = {
    "pick_and_place_simple": [
        "go to the receptacle that may hold the target",
        "take the target object",
        "go to the destination receptacle",
        "put the object in/on the destination",
    ],
    "look_at_obj_in_light": [
        "find and take the target object",
        "go to a lamp",
        "use the lamp to look at the object",
    ],
    "pick_clean_then_place_in_recep": [
        "take the target object",
        "go to a sinkbasin",
        "clean the object with the sinkbasin",
        "go to the destination receptacle",
        "put the cleaned object in/on the destination",
    ],
    "pick_heat_then_place_in_recep": [
        "take the target object",
        "go to a microwave",
        "heat the object with the microwave",
        "go to the destination receptacle",
        "put the heated object in/on the destination",
    ],
    "pick_cool_then_place_in_recep": [
        "take the target object",
        "go to a fridge",
        "cool the object with the fridge",
        "go to the destination receptacle",
        "put the cooled object in/on the destination",
    ],
    "pick_two_obj_and_place": [
        "take the first target object",
        "put it in/on the destination",
        "take the second target object",
        "put it in/on the same destination",
    ],
}

VERB_TO_LINE = {
    "go to": "go to the next relevant receptacle",
    "take": "take the target object from the current receptacle",
    "put": "put the object in/on the destination",
    "move": "move the object to the destination",
    "open": "open the receptacle if it is closed",
    "close": "close the receptacle if needed",
    "heat": "heat the object with a microwave",
    "cool": "cool the object with a fridge",
    "clean": "clean the object with a sinkbasin",
    "use": "use the lamp or appliance required by the task",
    "examine": "examine the object only if needed",
    "look": "look around if the object is not visible",
    "inventory": "check inventory if you already hold the object",
}


def strip_action(raw: str) -> str:
    text = ACTION_PREFIX.sub("", str(raw or "").strip()).strip()
    return text


def abstract_action(raw: str) -> str:
    text = strip_action(raw).lower()
    text = INSTANCE_RE.sub(r"<\1>", text)
    return re.sub(r"\s+", " ", text).strip()


def verb_of(abstracted: str) -> str:
    m = VERB_RE.match(abstracted)
    return m.group(1).lower() if m else ""


def skeleton_from_actions(actions: list[str]) -> list[str]:
    verbs: list[str] = []
    for a in actions:
        v = verb_of(abstract_action(a))
        if not v:
            continue
        if verbs and verbs[-1] == v and v in {"go to", "look", "examine", "inventory"}:
            continue
        verbs.append(v)
    # Drop leading look/inventory noise and collapse long wander.
    while verbs and verbs[0] in {"look", "inventory", "examine"}:
        verbs.pop(0)
    # Keep first 6 distinctive steps.
    out: list[str] = []
    for v in verbs:
        if len(out) >= 6:
            break
        out.append(v)
    return out


def majority_skeleton(skeletons: list[list[str]]) -> list[str]:
    if not skeletons:
        return []
    # Position-wise mode over the most common length (clipped 4–6).
    lengths = Counter(min(6, max(3, len(s))) for s in skeletons)
    target_len = lengths.most_common(1)[0][0]
    padded = []
    for s in skeletons:
        row = list(s[:target_len])
        if len(row) < target_len:
            row += [""] * (target_len - len(row))
        padded.append(row)
    verbs: list[str] = []
    for i in range(target_len):
        col = [row[i] for row in padded if row[i]]
        if not col:
            continue
        verbs.append(Counter(col).most_common(1)[0][0])
    return verbs[:6]


def lines_from_verbs(task_type: str, verbs: list[str]) -> list[str]:
    if not verbs:
        return list(CANONICAL.get(task_type, CANONICAL["pick_and_place_simple"]))
    lines = [VERB_TO_LINE.get(v, v) for v in verbs]
    # Prefer canonical if mined skeleton is too generic (only go/take).
    distinctive = {v for v in verbs if v in {"heat", "cool", "clean", "use", "put", "move"}}
    if task_type in CANONICAL and not distinctive and task_type != "pick_and_place_simple":
        return list(CANONICAL[task_type])
    if len(lines) < 3:
        return list(CANONICAL.get(task_type, lines))
    return lines[:6]


def format_workflow(lines: list[str]) -> str:
    body = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    return (
        "Suggested workflow (adapt names to the current admissible list):\n"
        f"{body}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_hybrid.yaml"),
    )
    args = ap.parse_args()
    cfg = load_yaml_cfg(Path(args.config))
    records = load_jsonl(Path(cfg["paths"]["rollouts_select"]))
    by_type: dict[str, list[list[str]]] = defaultdict(list)
    n_success = 0
    for rec in records:
        if not rec.get("episode_success"):
            continue
        n_success += 1
        tt = str(rec.get("task_type") or "unknown")
        by_type[tt].append(skeleton_from_actions(rec.get("actions") or []))

    workflows: dict[str, dict] = {}
    for tt, skels in sorted(by_type.items()):
        verbs = majority_skeleton(skels)
        lines = lines_from_verbs(tt, verbs)
        workflows[tt] = {
            "n_success": len(skels),
            "majority_verbs": verbs,
            "lines": lines,
            "text": format_workflow(lines),
        }

    # Ensure all six ALFWorld types exist.
    for tt, canon in CANONICAL.items():
        if tt not in workflows:
            workflows[tt] = {
                "n_success": 0,
                "majority_verbs": [],
                "lines": list(canon),
                "text": format_workflow(canon),
            }

    out = Path(cfg["paths"]["workflows"])
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        out,
        {
            "n_success_used": n_success,
            "source": str(cfg["paths"]["rollouts_select"]),
            "workflows": workflows,
        },
    )
    print(f"wrote {out} types={list(workflows)} n_success={n_success}")
    for tt, payload in workflows.items():
        print(f"  {tt}: n={payload['n_success']} verbs={payload['majority_verbs']}")
        print(payload["text"])
        print()


if __name__ == "__main__":
    main()
