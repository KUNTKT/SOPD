#!/usr/bin/env python3
"""TAME-OPD Gate T0: seeds, prefixes, manifest, audit, bootstrap, gate math."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")
STOP_TOKS = {
    "the", "a", "an", "to", "and", "or", "in", "on", "of", "with",
    "for", "from", "put", "find", "your", "task", "is", "two", "them",
}


def tokenize(text: str) -> frozenset[str]:
    toks = TOKEN_RE.findall(re.sub(r"\b([a-z][a-z0-9]*)\s+\d+\b", r"\1", str(text or "").lower()))
    return frozenset(t for t in toks if t not in STOP_TOKS and len(t) > 1)


def format_workflow(lines: list[str]) -> str:
    body = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    return (
        "Suggested workflow from a similar solved instance "
        "(adapt names to the current admissible list):\n"
        f"{body}"
    )


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def load_yaml_cfg(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text())

WORKFLOW_HEAD = "Suggested workflow"


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evolve_ids_from_jsonl(path: Path) -> set[str]:
    return {str(r["task_id"]) for r in load_jsonl(path) if r.get("task_id")}

FORBIDDEN_SPLITS = ("valid_unseen",)
CANDIDATE_ORDER = ("M0", "M1", "M2", "M3", "M4")
LIBRARY_SEEDS = {"M1": 11, "M2": 12, "M3": 13, "M4": 14}
RUN_SEEDS = (2020, 2021, 2022)
MOD_31 = 2**31
INSTANCE_RE = re.compile(r"\b([a-z][a-z0-9]*)[ \t]+\d+\b", re.I)
TASK_ID_RE = re.compile(r"\balfw_[0-9a-f]{6,}\b", re.I)
ROOM_NUM_RE = re.compile(
    r"\b(?:room|bedroom|bathroom|kitchen|livingroom|living)\s+\d+\b", re.I
)
STEP_LINE_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|-+\s*)?(.+?)\s*$")
OBJ_RECEP_RE = re.compile(
    r"\b([a-z][a-z0-9]*)\s+(?:in|on|from|to|with)\s+([a-z][a-z0-9]*)\b", re.I
)

EDITOR_SYSTEM = (
    "Rewrite one ALFWorld workflow. Keep the same task type and the same "
    "procedural meaning. Be executable and concise. You may drop redundancy, "
    "state preconditions, and add failure-correction rules that follow from "
    "the given traces. Do not invent a six-type skeleton. Do not add generic "
    "agent advice. Do not copy instance numbers, task IDs, room numbers, or "
    "specific object-container answers from the traces. Reply with a numbered "
    "list of action lines only."
)


def load_tame_cfg(path: str | Path | None = None) -> dict[str, Any]:
    default = Path(__file__).resolve().parents[1] / "configs/experiment_tame_opd.yaml"
    return load_yaml_cfg(Path(path) if path else default)


def refuse_forbidden_split(split: str | None) -> None:
    name = str(split or "")
    if name in FORBIDDEN_SPLITS or name.startswith("valid_unseen"):
        raise SystemExit(f"forbidden split {name!r}: valid_unseen must not be opened")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def editor_prompt_hash() -> str:
    return sha256_text(EDITOR_SYSTEM)


def entry_seed(library_seed: int, entry_id: str) -> int:
    digest = hashlib.sha256(str(entry_id).encode("utf-8")).hexdigest()
    return (int(library_seed) + int(digest[:16], 16)) % MOD_31


def sub_seed(run_seed: int, role: str) -> int:
    raw = f"{int(run_seed)}:{role}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % MOD_31


def with_rollout_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    from copy import deepcopy

    out = deepcopy(cfg)
    out.setdefault("rollout", {})["seed"] = int(seed)
    return out


def teacher_prefix(prefix_base: str, workflow_text: str | None) -> str:
    """B-uce equivalent: apply workflow to this step's full render only."""
    return apply_workflow(strip_workflow(prefix_base), workflow_text)


def fake_whole_transcript_prepend(
    first_prefix_base: str, current_prefix_base: str, workflow_text: str
) -> str:
    packed = f"{strip_workflow(first_prefix_base)}\n---\n{strip_workflow(current_prefix_base)}"
    return apply_workflow(packed, workflow_text)


def annotate_episode(rec: dict[str, Any], workflow_text: str, *, workflow_id: str | None = None, score: float = 0.0) -> dict[str, Any]:
    rec = dict(rec)
    rec["workflow_text"] = workflow_text
    rec["workflow_id"] = workflow_id
    rec["retrieval_score"] = score
    dps = []
    for dp in rec.get("decision_points") or []:
        item = dict(dp)
        base = strip_workflow(str(item.get("prefix_text") or item.get("prefix_base") or ""))
        item["prefix_base"] = base
        item["prefix_uce"] = teacher_prefix(base, workflow_text)
        item["workflow_text"] = workflow_text
        dps.append(item)
    rec["decision_points"] = dps
    return rec


def retrieve_once(lib: UceLibrary, task: dict[str, Any]) -> tuple[str, str | None, float]:
    before = json.dumps([e.get("usage") for e in lib.entries], sort_keys=True)
    text, eid, score = lib.retrieve(task)
    after = json.dumps([e.get("usage") for e in lib.entries], sort_keys=True)
    if before != after:
        raise RuntimeError("UCE retrieve mutated usage")
    return text, eid, score


def leading_verb(command: str) -> str:
    tok = str(command or "").strip().split()
    return tok[0].lower() if tok else ""


def verbs_from_admissible(commands: list[str]) -> set[str]:
    return {leading_verb(c) for c in commands if leading_verb(c)}


def core_verbs(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        v = leading_verb(line)
        if v:
            out.append(v)
    return out


def verb_retention(original_lines: list[str], rewrite_lines: list[str]) -> float:
    src = core_verbs(original_lines)
    if not src:
        return 1.0
    dst = set(core_verbs(rewrite_lines))
    return sum(1 for v in src if v in dst) / len(src)


def goal_token_coverage(goal_tokens: list[str], rewrite_text: str) -> float:
    toks = list(goal_tokens or [])
    if not toks:
        return 1.0
    hay = set(tokenize(rewrite_text))
    return sum(1 for t in toks if t in hay) / len(toks)


def parse_rewrite_lines(raw: str) -> list[str]:
    numbered: list[str] = []
    loose: list[str] = []
    for row in str(raw or "").splitlines():
        row = row.strip()
        if not row or row.lower().startswith("suggested workflow"):
            continue
        nm = re.match(r"^\s*\d+[\.\)]\s+(.+)$", row)
        body = nm.group(1) if nm else row
        body = re.sub(r"^(ACTION|Action|action)\s*[:：]\s*", "", body)
        body = re.sub(r"\s+", " ", body).strip().lower()
        if not body or body in {"look", "inventory", "thanks", "thank you", "ok", "okay"}:
            continue
        if nm:
            if not numbered or numbered[-1] != body:
                numbered.append(body)
        else:
            if not loose or loose[-1] != body:
                loose.append(body)
        if len(numbered) >= 12:
            break
    return numbered[:12] if numbered else loose[:12]


def novel_leak_hits(
    rewrite_text: str,
    original_text: str,
    *,
    extra_goals: list[str] | None = None,
    extra_pairs: list[str] | None = None,
) -> list[str]:
    """Leaks are patterns in rewrite that are not already in the original workflow."""
    rw = str(rewrite_text or "")
    orig = str(original_text or "")
    hits: list[str] = []

    def novel(match: str) -> bool:
        return match.casefold() not in orig.casefold()

    for m in INSTANCE_RE.finditer(rw):
        if novel(m.group(0)):
            hits.append(f"instance:{m.group(0)}")
    for m in TASK_ID_RE.finditer(rw):
        if novel(m.group(0)):
            hits.append(f"task_id:{m.group(0)}")
    for m in ROOM_NUM_RE.finditer(rw):
        if novel(m.group(0)):
            hits.append(f"room:{m.group(0)}")
    for goal in extra_goals or []:
        g = str(goal or "").strip()
        if g and g.casefold() in rw.casefold() and g.casefold() not in orig.casefold():
            hits.append(f"goal:{g[:80]}")
    for pair in extra_pairs or []:
        p = str(pair or "").strip()
        if p and p.casefold() in rw.casefold() and p.casefold() not in orig.casefold():
            hits.append(f"pair:{p}")
    return sorted(set(hits))


def object_container_pairs(text: str) -> list[str]:
    return [f"{a.lower()}|{b.lower()}" for a, b in OBJ_RECEP_RE.findall(text or "")]


def length_flags(original: str, rewrite: str) -> list[str]:
    o = len(original or "")
    r = len(rewrite or "")
    flags = []
    if r > 2000:
        flags.append("chars_gt_2000")
    if o > 0 and r > 3 * o:
        flags.append("chars_gt_3x")
    return flags


def copy_m0_entry(src: dict[str, Any]) -> dict[str, Any]:
    e = dict(src)
    e["parent_id"] = src.get("id")
    e["parent_hash"] = sha256_text(str(src.get("text") or ""))
    e["gen_seed"] = None
    e["fallback"] = False
    e["reverted"] = False
    return e


def build_m0_library(evolved: dict[str, Any]) -> dict[str, Any]:
    entries = [copy_m0_entry(e) for e in evolved.get("entries") or []]
    return {
        "n_entries": len(entries),
        "candidate_id": "M0",
        "library_seed": None,
        "source": "uce_library_evolved",
        "entries": entries,
    }


def revert_entry_to_m0(entry: dict[str, Any], m0_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    src = m0_by_id.get(str(entry.get("id")))
    if src is None:
        out = dict(entry)
        out["fallback"] = True
        out["reverted"] = True
        return out
    out = copy_m0_entry(src)
    out["gen_seed"] = entry.get("gen_seed")
    out["fallback"] = True
    out["reverted"] = True
    out["revert_reason"] = entry.get("audit_reasons") or ["audit_fail"]
    return out


def audit_entry(
    entry: dict[str, Any],
    m0_entry: dict[str, Any],
    *,
    whitelist: set[str],
    extra_goals: list[str] | None = None,
    extra_pairs: list[str] | None = None,
) -> dict[str, Any]:
    text = str(entry.get("text") or "")
    orig = str(m0_entry.get("text") or "")
    lines = list(entry.get("lines") or [])
    orig_lines = list(m0_entry.get("lines") or [])
    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_text")
    if str(entry.get("task_type")) != str(m0_entry.get("task_type")):
        reasons.append("task_type_changed")
    leaks = novel_leak_hits(text, orig, extra_goals=extra_goals, extra_pairs=extra_pairs)
    if leaks:
        reasons.append("leak")
    illegal = [v for v in core_verbs(lines) if whitelist and v not in whitelist]
    if illegal:
        reasons.append("illegal_verb")
    flags = length_flags(orig, text)
    reasons.extend(flags)
    return {
        "id": entry.get("id"),
        "ok": not reasons,
        "reasons": reasons,
        "leaks": leaks,
        "illegal_verbs": illegal,
        "verb_retention": verb_retention(orig_lines, lines),
        "goal_token_coverage": goal_token_coverage(m0_entry.get("goal_tokens") or [], text),
        "n_chars": len(text),
        "n_chars_orig": len(orig),
        "length_ratio": (len(text) / len(orig)) if orig else None,
        "edit_distance": _edit_distance(orig, text),
        "entry_hash": sha256_text(text),
    }


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) > 4000 or len(b) > 4000:
        return abs(len(a) - len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def audit_library(
    lib: dict[str, Any],
    m0: dict[str, Any],
    *,
    whitelist: set[str],
    extra_goals: list[str] | None = None,
    extra_pairs: list[str] | None = None,
    fallback_rate_max: float = 0.05,
) -> dict[str, Any]:
    m0_by = {str(e["id"]): e for e in m0.get("entries") or []}
    rows = []
    n_fallback = 0
    n_fail = 0
    for e in lib.get("entries") or []:
        m0e = m0_by.get(str(e.get("id")), {})
        row = audit_entry(e, m0e, whitelist=whitelist, extra_goals=extra_goals, extra_pairs=extra_pairs)
        row["fallback"] = bool(e.get("fallback") or e.get("reverted"))
        if row["fallback"]:
            n_fallback += 1
        if not row["ok"] and not row["fallback"]:
            n_fail += 1
        rows.append(row)
    n = max(len(rows), 1)
    fallback_rate = n_fallback / n
    return {
        "candidate_id": lib.get("candidate_id"),
        "n_entries": len(rows),
        "n_fallback": n_fallback,
        "n_unresolved_fail": n_fail,
        "fallback_rate": fallback_rate,
        "valid": fallback_rate <= fallback_rate_max and n_fail == 0,
        "mean_verb_retention": sum(r["verb_retention"] for r in rows) / n,
        "mean_goal_token_coverage": sum(r["goal_token_coverage"] for r in rows) / n,
        "entries": rows,
    }


def apply_deterministic_fallback(
    lib: dict[str, Any],
    m0: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    m0_by = {str(e["id"]): e for e in m0.get("entries") or []}
    fail_ids = {str(r["id"]) for r in audit.get("entries") or [] if not r.get("ok")}
    out_entries = []
    for e in lib.get("entries") or []:
        if str(e.get("id")) in fail_ids:
            bad = dict(e)
            bad["audit_reasons"] = next(
                (r.get("reasons") for r in audit["entries"] if str(r["id"]) == str(e.get("id"))),
                ["audit_fail"],
            )
            out_entries.append(revert_entry_to_m0(bad, m0_by))
        else:
            out_entries.append(dict(e))
    out = dict(lib)
    out["entries"] = out_entries
    out["n_entries"] = len(out_entries)
    return out


def sample_audit_ids(entries: list[dict[str, Any]], n: int, seed: int) -> list[str]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_type[str(e.get("task_type") or "unknown")].append(str(e["id"]))
    rng = random.Random(int(seed))
    picked: list[str] = []
    types = sorted(by_type)
    while len(picked) < n and any(by_type[t] for t in types):
        for t in types:
            if len(picked) >= n:
                break
            bucket = by_type[t]
            if not bucket:
                continue
            i = rng.randrange(len(bucket))
            picked.append(bucket.pop(i))
    return picked


def plan_probe_budget(
    n_supervised: list[int],
    *,
    max_steps: int = 64,
    max_tokens: int = 4096,
) -> list[int]:
    """Return keep_n per incoming step (0 means unused). Exact stop at max_tokens."""
    keep: list[int] = []
    used_tokens = 0
    used_steps = 0
    for k in n_supervised:
        k = int(k)
        if used_steps >= max_steps or used_tokens >= max_tokens:
            keep.append(0)
            continue
        if used_tokens + k <= max_tokens:
            keep.append(k)
            used_tokens += k
            used_steps += 1
        else:
            rest = max_tokens - used_tokens
            keep.append(rest)
            used_tokens += rest
            used_steps += 1
    return keep


def truncate_mask(mask: list[bool], keep_n: int) -> list[bool]:
    out = [False] * len(mask)
    seen = 0
    for i, bit in enumerate(mask):
        if bit and seen < keep_n:
            out[i] = True
            seen += 1
    return out


def argmax_with_tiebreak(scores: dict[str, float], eligible: list[str]) -> str | None:
    if not eligible:
        return None
    def key(name: str) -> tuple[float, int]:
        return (float(scores[name]), -CANDIDATE_ORDER.index(name) if name in CANDIDATE_ORDER else -99)

    # Smaller index wins ties: sort by score desc, then index asc.
    ranked = sorted(eligible, key=lambda n: (-float(scores[n]), CANDIDATE_ORDER.index(n) if n in CANDIDATE_ORDER else 99))
    return ranked[0]


def task_mean_success(records_by_seed: dict[int, list[dict[str, Any]]]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for recs in records_by_seed.values():
        seen: dict[str, list[int]] = defaultdict(list)
        for rec in recs:
            seen[str(rec["task_id"])].append(1 if rec.get("episode_success") else 0)
        for tid, ys in seen.items():
            acc[tid].append(sum(ys) / len(ys))
    return {tid: sum(vs) / len(vs) for tid, vs in acc.items()}


def task_mean_paired_bootstrap(
    means_a: dict[str, float],
    means_b: dict[str, float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    common = sorted(set(means_a) & set(means_b))
    if not common:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_tasks": 0}
    deltas = [means_b[t] - means_a[t] for t in common]
    mean = sum(deltas) / len(deltas)
    rng = random.Random(int(seed))
    boots = []
    n = len(deltas)
    for _ in range(int(n_boot)):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    return {
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_tasks": len(common),
        "n_win": wins,
        "n_loss": losses,
        "n_tie": ties,
    }


def per_seed_success(recs: list[dict[str, Any]]) -> float:
    if not recs:
        return 0.0
    return sum(1 for r in recs if r.get("episode_success")) / len(recs)


def evaluate_gates(
    *,
    teach_pick: str | None,
    teacher_pick: str | None,
    eligible: list[str],
    delta_m0: float,
    ci_lo_m0: float,
    delta_teacher: float,
    ci_lo_teacher: float,
    n_pos_seeds_vs_m0: int,
    teach_vs_init: float,
    teacher_j_teach: float,
    teacher_j_m0: float,
    min_delta_m0_pp: float = 3.0,
    teacher_slack_pp: float = 2.0,
    min_positive_seeds: int = 2,
) -> dict[str, Any]:
    only_m0 = eligible == ["M0"] or (eligible and set(eligible) == {"M0"})
    same_pick = teach_pick is not None and teach_pick == teacher_pick
    identifiable = (not only_m0) and (teach_pick is not None) and (teacher_pick is not None) and (not same_pick)

    t0a_reasons = []
    t0a = True
    if teach_pick is None or only_m0:
        t0a = False
        t0a_reasons.append("not_identifiable")
    else:
        if delta_m0 * 100 < min_delta_m0_pp:
            t0a = False
            t0a_reasons.append("delta_m0_lt_3pp")
        if ci_lo_m0 <= 0:
            t0a = False
            t0a_reasons.append("ci_lo_m0_not_positive")
        if n_pos_seeds_vs_m0 < min_positive_seeds:
            t0a = False
            t0a_reasons.append("lt_2_of_3_seeds_positive")
        if teach_vs_init <= 0:
            t0a = False
            t0a_reasons.append("teach_vs_init_not_positive")
        if teacher_j_teach < teacher_j_m0 - teacher_slack_pp / 100.0:
            t0a = False
            t0a_reasons.append("teacher_below_m0_slack")

    t0b_status = "pass"
    t0b_reasons = []
    if only_m0 or teach_pick is None or teacher_pick is None or same_pick:
        t0b_status = "not_identifiable"
        t0b_reasons.append("not_identifiable")
    else:
        if delta_teacher <= 0:
            t0b_status = "fail"
            t0b_reasons.append("delta_teacher_not_positive")
        if ci_lo_teacher <= 0:
            t0b_status = "fail"
            t0b_reasons.append("ci_lo_teacher_not_positive")

    method_pass = bool(t0a) and t0b_status == "pass"
    return {
        "teach_pick": teach_pick,
        "teacher_pick": teacher_pick,
        "eligible": eligible,
        "identifiable": identifiable,
        "T0_A": {"pass": t0a, "reasons": t0a_reasons},
        "T0_B": {"status": t0b_status, "pass": t0b_status == "pass", "reasons": t0b_reasons},
        "method_gate_pass": method_pass,
        "claim": _claim_text(t0a, t0b_status),
    }


def _claim_text(t0a: bool, t0b_status: str) -> str:
    if t0a and t0b_status == "pass":
        return (
            "T0 PASS: brief no-workflow student utility is a usable memory selection "
            "target on this frozen population. Not a complete scalable TAME method."
        )
    if t0a and t0b_status == "fail":
        return "Some rewrites distill better than original UCE. Cannot claim teaching-utility selection beats teacher-utility selection."
    if t0b_status == "not_identifiable":
        return "T0-B not identifiable. Method gate FAIL. Not a disproof of TAME."
    return "T0-A FAIL. Stop. Do not expand the method."


def stratified_take(
    tasks: list[dict[str, Any]],
    n: int,
    *,
    seed: int,
    taken: set[str],
) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        tid = str(t["task_id"])
        if tid in taken:
            continue
        by_type[str(t.get("task_type") or "unknown")].append(t)
    for tt in by_type:
        by_type[tt].sort(key=lambda x: str(x["task_id"]))
    total = sum(len(v) for v in by_type.values())
    if total < n:
        raise SystemExit(f"not enough tasks for stratified sample: have {total} need {n}")
    types = sorted(by_type)
    alloc = {tt: int(n * len(by_type[tt]) / total) for tt in types}
    while sum(alloc.values()) < n:
        tt = max(types, key=lambda k: len(by_type[k]) - alloc[k])
        if alloc[tt] < len(by_type[tt]):
            alloc[tt] += 1
        else:
            break
    while sum(alloc.values()) > n:
        tt = max(types, key=lambda k: alloc[k])
        if alloc[tt] > 0:
            alloc[tt] -= 1
    rng = random.Random(int(seed))
    picked: list[dict[str, Any]] = []
    for tt in types:
        bucket = list(by_type[tt])
        rng.shuffle(bucket)
        picked.extend(bucket[: alloc[tt]])
    picked.sort(key=lambda t: str(t["task_id"]))
    return picked


def slim_task(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": t.get("task_id"),
        "task_type": t.get("task_type"),
        "goal": t.get("goal"),
        "split": t.get("split"),
        "game_file": t.get("game_file"),
        "source": t.get("source"),
    }


def build_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    refuse_forbidden_split(str(cfg.get("env", {}).get("eval_split") or ""))
    evolve_path = Path(cfg["paths"]["evolve_jsonl"])
    evolve_ids = sorted(evolve_ids_from_jsonl(evolve_path))
    if len(evolve_ids) != int(cfg["env"].get("evolve_n", 80)):
        raise SystemExit(f"expected 80 evolve ids, found {len(evolve_ids)}")
    from alfworld_common import load_task_pools

    archive = Path(cfg["paths"]["archive"])
    opd_cache = archive / "reports/uce_opd/task_splits_cache.json"
    if opd_cache.exists():
        cached = json.loads(opd_cache.read_text())
        pools = {
            "pools": {
                "sft": cached["sft"],
                "audit_select": cached["audit_select"],
                "audit_confirm": cached["audit_confirm"],
            },
            "meta": cached.get("pools_meta") or {},
        }
        if "eval" in cached or "eval_ids" in cached:
            # Train-pool reuse only; drop any holdout fields.
            pass
    else:
        pools = load_task_pools(
            data_root=str(cfg["env"]["data_root"]),
            limit=int(cfg["env"].get("train_limit", 1200)),
            split="train",
            partition_seed=int(cfg.get("random_seed", 1010)),
        )
    universe = []
    for name in ("sft", "audit_select", "audit_confirm"):
        universe.extend(pools["pools"][name])
    by_id = {str(t["task_id"]): t for t in universe}
    evolve_tasks = [slim_task(by_id[i]) for i in evolve_ids if i in by_id]
    if len(evolve_tasks) != len(evolve_ids):
        missing = [i for i in evolve_ids if i not in by_id]
        raise SystemExit(f"evolve ids missing from train pool: {missing[:8]}")
    taken = set(evolve_ids)
    rest = [t for t in universe if str(t["task_id"]) not in taken]
    rest.sort(key=lambda t: str(t["task_id"]))
    ms = int(cfg.get("manifest_seed", 3030))
    inner = stratified_take(rest, int(cfg["env"]["inner_train_n"]), seed=ms, taken=taken)
    taken |= {str(t["task_id"]) for t in inner}
    meta = stratified_take(rest, int(cfg["env"]["meta_select_n"]), seed=ms + 1, taken=taken)
    taken |= {str(t["task_id"]) for t in meta}
    confirm = stratified_take(rest, int(cfg["env"]["probe_confirm_n"]), seed=ms + 2, taken=taken)
    groups = {
        "excluded_uce_evolution": [slim_task(t) for t in evolve_tasks],
        "inner_train": [slim_task(t) for t in inner],
        "meta_select": [slim_task(t) for t in meta],
        "probe_confirm": [slim_task(t) for t in confirm],
    }
    ids = {k: [str(t["task_id"]) for t in v] for k, v in groups.items()}
    _assert_disjoint(ids)
    type_counts = {
        k: _type_counts(v) for k, v in groups.items()
    }
    id_blob = json.dumps(ids, sort_keys=True)
    payload = {
        "manifest_seed": ms,
        "partition_seed": int(cfg.get("random_seed", 1010)),
        "forbidden_splits": list(cfg.get("forbidden_splits") or FORBIDDEN_SPLITS),
        "evolve_jsonl": str(evolve_path),
        "evolve_jsonl_hash": sha256_file(evolve_path),
        "pools_meta": pools["meta"],
        "ids": ids,
        "tasks": groups,
        "type_counts": type_counts,
        "n": {k: len(v) for k, v in ids.items()},
        "manifest_hash": sha256_text(id_blob),
    }
    if "valid_unseen" in json.dumps(payload):
        # IDs must never be recorded; the forbidden name is allowed only as the split label.
        if payload.get("ids", {}).get("valid_unseen") or "valid_unseen" in payload.get("tasks", {}):
            raise SystemExit("manifest must not record valid_unseen task IDs")
    return payload


def _type_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for t in tasks:
        c[str(t.get("task_type") or "unknown")] += 1
    return dict(sorted(c.items()))


def _assert_disjoint(ids: dict[str, list[str]]) -> None:
    names = list(ids)
    for i, a in enumerate(names):
        sa = set(ids[a])
        if len(sa) != len(ids[a]):
            raise SystemExit(f"duplicate ids in {a}")
        for b in names[i + 1 :]:
            inter = sa & set(ids[b])
            if inter:
                raise SystemExit(f"split overlap {a} ∩ {b}: {sorted(inter)[:6]}")


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if raw.get("ids", {}).get("valid_unseen") or "valid_unseen" in (raw.get("tasks") or {}):
        raise SystemExit("manifest contains valid_unseen tasks")
    return raw


def tasks_of(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    refuse_forbidden_split(split)
    return list(manifest["tasks"][split])


def save_library(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    (path.parent / "library.sha256").write_text(digest + "\n")
    return digest


def load_library(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_uce_library(payload: dict[str, Any]):
    from uce_library import UceLibrary

    return UceLibrary(list(payload.get("entries") or []))


def editor_user_prompt(
    *,
    original_text: str,
    task_type: str,
    success_notes: list[str],
    fail_notes: list[str],
    diagnostics: str,
) -> str:
    succ = "\n".join(f"- {s}" for s in success_notes[:2]) or "- none"
    fail = "\n".join(f"- {s}" for s in fail_notes[:2]) or "- none"
    return (
        f"Task type: {task_type}\n\n"
        f"Original workflow:\n{original_text}\n\n"
        f"Successful traces (abstracted):\n{succ}\n\n"
        f"Failed traces (abstracted):\n{fail}\n\n"
        f"Diagnostics:\n{diagnostics}\n"
    )


def build_editor_messages(user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EDITOR_SYSTEM},
        {"role": "user", "content": user},
    ]
