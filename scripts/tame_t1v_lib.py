#!/usr/bin/env python3
"""T1-V: delexicalizer, JSON patch schema, canonical renderer, Gate V math."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INSTANCE_RE = re.compile(r"\b([a-z][a-z0-9]*)[ \t]+\d+\b", re.I)
TASK_ID_RE = re.compile(r"\balfw_[0-9a-f]{6,}\b", re.I)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
ALLOWED_TOP = ("delete_rules", "rewrite_rules", "add_rules")
LIBRARY_SEEDS = (11, 12, 13, 14)
USAGE_BUCKETS = ("0", "1-2", ">=3")
MAX_STR = 200
EMPTY_PATCH = {"delete_rules": [], "rewrite_rules": [], "add_rules": []}

EDITOR_SYSTEM = (
    "You edit one ALFWorld workflow by emitting a single JSON object and nothing else. "
    "Do not write the final workflow. Do not write prose before or after the JSON. "
    "Do not use environment instance numbers (such as apple 1). "
    "You may delete redundant rules, rewrite a rule, or add a generic failure-correction rule. "
    "Keep the same task procedure. "
    "Schema: {\"delete_rules\": [int], \"rewrite_rules\": [{\"rule_id\": int, \"new_text\": str}], "
    "\"add_rules\": [{\"condition\": str, \"instruction\": str}]}. "
    "rule_id is 1-based. Empty arrays are allowed."
)


def load_cfg(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    default = Path(__file__).resolve().parents[1] / "configs/experiment_tame_t1v.yaml"
    return yaml.safe_load(Path(path or default).read_text())


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def editor_prompt_hash() -> str:
    return sha256_text(EDITOR_SYSTEM)


def entry_seed(library_seed: int, entry_id: str) -> int:
    digest = hashlib.sha256(str(entry_id).encode("utf-8")).hexdigest()
    return (int(library_seed) + int(digest[:16], 16)) % (2**31)


def usage_bucket(usage: Any) -> str:
    u = int(usage or 0)
    if u <= 0:
        return "0"
    if u <= 2:
        return "1-2"
    return ">=3"


def delexicalize(text: str) -> str:
    return INSTANCE_RE.sub(r"\1", str(text or ""))


def novel_instance_hits(text: str, original: str) -> list[str]:
    hits = []
    orig = original or ""
    for m in INSTANCE_RE.finditer(text or ""):
        if m.group(0).casefold() not in orig.casefold():
            hits.append(m.group(0))
    for m in TASK_ID_RE.finditer(text or ""):
        if m.group(0).casefold() not in orig.casefold():
            hits.append(m.group(0))
    return hits


def classify_error(feedback: str, verdict: dict | None) -> str | None:
    fb = delexicalize(feedback or "").lower()
    ver = verdict or {}
    if ver.get("failure_reason") == "not_admissible" or "nothing happens" in fb:
        return "ACTION_REJECTED"
    if "closed" in fb:
        return "CONTAINER_CLOSED"
    if any(k in fb for k in ("not here", "nothing there", "can't find", "cannot find", "not found")):
        return "OBJECT_NOT_FOUND"
    return None


def aggregate_error_stats(records: list[dict]) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n_dp": 0, "n_invalid": 0, "n_repeat": 0, "types": Counter()}
    )
    for rec in records:
        tt = str(rec.get("task_type") or "unknown")
        prev = None
        for dp in rec.get("decision_points") or []:
            by[tt]["n_dp"] += 1
            ver = dp.get("verdict") if isinstance(dp.get("verdict"), dict) else {}
            if ver.get("failure_reason") == "not_admissible" or dp.get("failure_reason") == "not_admissible":
                by[tt]["n_invalid"] += 1
            act = str(dp.get("raw_action_text") or dp.get("action_text") or "")
            if prev and act == prev:
                by[tt]["n_repeat"] += 1
            prev = act
            fb = ""
            state = dp.get("state") or {}
            fb = str(state.get("feedback") or dp.get("feedback") or "")
            et = classify_error(fb, ver)
            if et:
                by[tt]["types"][et] += 1
    out = {}
    for tt, s in by.items():
        n = max(int(s["n_dp"]), 1)
        out[tt] = {
            "invalid_action_rate": s["n_invalid"] / n,
            "repeat_action_rate": s["n_repeat"] / n,
            "error_type_counts": dict(s["types"]),
            "n_dp": s["n_dp"],
        }
    return out


def stats_for_prompt(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "invalid_action_rate=NA; repeat_action_rate=NA; error_types=[]"
    types = ",".join(sorted(stats.get("error_type_counts") or {})) or "none"
    return (
        f"invalid_action_rate={stats['invalid_action_rate']:.3f}; "
        f"repeat_action_rate={stats['repeat_action_rate']:.3f}; "
        f"error_types=[{types}]"
    )


def numbered_lines(lines: list[str]) -> str:
    return "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))


def editor_user_prompt(lines: list[str], type_stats: dict | None) -> str:
    return (
        "Original workflow rules:\n"
        f"{numbered_lines(lines)}\n\n"
        "Abstract error statistics for this task type (no observations, no goals):\n"
        f"{stats_for_prompt(type_stats)}\n\n"
        "Emit one JSON patch object only."
    )


def canonical_lines(lines: list[str]) -> list[str]:
    out = []
    for ln in lines or []:
        s = re.sub(r"\s+", " ", str(ln).strip().lower())
        if s:
            out.append(s)
    return out


def canonical_hash(lines: list[str]) -> str:
    return sha256_text("\n".join(canonical_lines(lines)))


def format_workflow(lines: list[str]) -> str:
    body = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    return (
        "Suggested workflow from a similar solved instance "
        "(adapt names to the current admissible list):\n"
        f"{body}"
    )


def extract_json_object(raw: str) -> tuple[dict | None, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, "empty_raw"
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, "not_object"
    except json.JSONDecodeError:
        pass
    m = JSON_OBJ_RE.search(str(raw or ""))
    if not m:
        return None, "no_json"
    before = str(raw or "")[: m.start()].strip()
    after = str(raw or "")[m.end() :].strip()
    if before or after:
        return None, "prose_around_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "json_decode"
    if not isinstance(obj, dict):
        return None, "not_object"
    return obj, None


def validate_patch(obj: dict, n_rules: int, *, max_str: int = MAX_STR) -> str | None:
    if set(obj.keys()) - set(ALLOWED_TOP):
        return "additional_properties"
    if set(ALLOWED_TOP) - set(obj.keys()):
        return "missing_keys"
    deletes = obj.get("delete_rules")
    rewrites = obj.get("rewrite_rules")
    adds = obj.get("add_rules")
    if not isinstance(deletes, list) or not isinstance(rewrites, list) or not isinstance(adds, list):
        return "bad_list_types"
    if any(not isinstance(x, int) or isinstance(x, bool) for x in deletes):
        return "delete_not_int"
    if len(deletes) != len(set(deletes)):
        return "delete_not_unique"
    rids = []
    for rw in rewrites:
        if not isinstance(rw, dict) or set(rw.keys()) != {"rule_id", "new_text"}:
            return "rewrite_schema"
        if not isinstance(rw["rule_id"], int) or isinstance(rw["rule_id"], bool):
            return "rewrite_id_type"
        nt = rw.get("new_text")
        if not isinstance(nt, str) or not nt.strip() or len(nt) > max_str:
            return "rewrite_text"
        rids.append(int(rw["rule_id"]))
    if len(rids) != len(set(rids)):
        return "rewrite_id_not_unique"
    if set(deletes) & set(rids):
        return "delete_rewrite_overlap"
    for ad in adds:
        if not isinstance(ad, dict) or set(ad.keys()) != {"condition", "instruction"}:
            return "add_schema"
        for k in ("condition", "instruction"):
            v = ad.get(k)
            if not isinstance(v, str) or not v.strip() or len(v) > max_str:
                return "add_text"
    all_ids = list(deletes) + rids
    if any(i < 1 or i > n_rules for i in all_ids):
        return "rule_id_oob"
    if n_rules - len(set(deletes)) < 1:
        return "deleted_all"
    return None


def render_patch(lines: list[str], patch: dict) -> list[str]:
    keep = []
    deletes = set(int(x) for x in patch.get("delete_rules") or [])
    rewrites = {int(r["rule_id"]): str(r["new_text"]).strip() for r in patch.get("rewrite_rules") or []}
    for i, ln in enumerate(lines, 1):
        if i in deletes:
            continue
        keep.append(rewrites.get(i, ln))
    for ad in patch.get("add_rules") or []:
        keep.append(f"if {ad['condition'].strip()}: {ad['instruction'].strip()}")
    return keep


def process_generation(
    raw: str,
    original_lines: list[str],
    original_text: str,
    *,
    max_str: int = MAX_STR,
) -> dict[str, Any]:
    n = len(original_lines)
    orig_hash = canonical_hash(original_lines)
    raw_leaks = novel_instance_hits(raw, original_text)
    rec = {
        "raw_generation": raw,
        "parsed_patch": None,
        "rendered_lines": list(original_lines),
        "canonical_hash": orig_hash,
        "fallback": False,
        "fallback_reason": None,
        "schema_ok": False,
        "raw_leak": bool(raw_leaks),
        "residual_leak": False,
        "noop": False,
        "real_modify": False,
        "accepted": False,
    }
    obj, err = extract_json_object(raw)
    verr = validate_patch(obj, n, max_str=max_str) if obj is not None and err is None else err
    if obj is not None and err is None and verr is None:
        rec["schema_ok"] = True
        rec["parsed_patch"] = obj
    if raw_leaks:
        rec["fallback"] = True
        rec["fallback_reason"] = "raw_leak"
        return rec
    if err:
        rec["fallback"] = True
        rec["fallback_reason"] = err
        return rec
    if verr:
        rec["fallback"] = True
        rec["fallback_reason"] = verr
        return rec
    rendered = render_patch(original_lines, obj)
    rec["rendered_lines"] = rendered
    rec["canonical_hash"] = canonical_hash(rendered)
    residual = novel_instance_hits("\n".join(rendered), original_text)
    rec["residual_leak"] = bool(residual)
    if residual:
        rec["fallback"] = True
        rec["fallback_reason"] = "residual_leak"
        rec["rendered_lines"] = list(original_lines)
        rec["canonical_hash"] = orig_hash
        rec["accepted"] = False
        return rec
    rec["accepted"] = True
    rec["noop"] = rec["canonical_hash"] == orig_hash
    rec["real_modify"] = rec["accepted"] and not rec["noop"]
    return rec


def sample_60(entries: list[dict], n: int, seed: int) -> dict[str, Any]:
    usable = [e for e in entries if (e.get("lines") or []) and str(e.get("text") or "").strip()]
    layers: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in usable:
        layers[(str(e.get("task_type") or "unknown"), usage_bucket(e.get("usage")))].append(e)
    keys = sorted(layers)
    for k in keys:
        layers[k].sort(key=lambda x: str(x["id"]))
    counts = {k: len(layers[k]) for k in keys}
    total = sum(counts.values())
    if total < n:
        raise SystemExit(f"not enough M0 entries: {total} < {n}")
    alloc = {k: int(n * counts[k] / total) for k in keys}
    while sum(alloc.values()) < n:
        for k in keys:
            if alloc[k] < counts[k]:
                alloc[k] += 1
                if sum(alloc.values()) >= n:
                    break
    while sum(alloc.values()) > n:
        for k in reversed(keys):
            if alloc[k] > 0:
                alloc[k] -= 1
                break
    planned = dict(alloc)
    picked: list[dict] = []
    layer_actual: dict[str, dict[str, int]] = {}
    rng = random.Random(int(seed))
    leftover_need = 0
    for k in keys:
        bucket = list(layers[k])
        rng.shuffle(bucket)
        take = min(alloc[k], len(bucket))
        leftover_need += alloc[k] - take
        layers[k] = bucket[take:]
        picked.extend(bucket[:take])
        layer_actual[f"{k[0]}|{k[1]}"] = {"planned": planned[k], "actual": take, "available": counts[k]}
    if leftover_need:
        for k in keys:
            if leftover_need <= 0:
                break
            extra = layers[k][:leftover_need]
            if not extra:
                continue
            picked.extend(extra)
            layers[k] = layers[k][len(extra) :]
            layer_actual[f"{k[0]}|{k[1]}"]["actual"] += len(extra)
            leftover_need -= len(extra)
    picked.sort(key=lambda x: str(x["id"]))
    slim = [
        {
            "id": e["id"],
            "type": e.get("task_type"),
            "task_type": e.get("task_type"),
            "usage": e.get("usage"),
            "bucket": usage_bucket(e.get("usage")),
            "usage_bucket": usage_bucket(e.get("usage")),
            "parent_hash": e.get("parent_hash") or sha256_text(str(e.get("text") or "")),
            "n_lines": len(e.get("lines") or []),
        }
        for e in picked[:n]
    ]
    return {
        "sample_seed": seed,
        "n": len(slim),
        "layer_actual": layer_actual,
        "entries": slim,
        "ids": [s["id"] for s in slim],
    }


def per_seed_metrics(rows: list[dict]) -> dict[str, Any]:
    n = max(len(rows), 1)
    n_schema = sum(1 for r in rows if r.get("schema_ok"))
    n_fb = sum(1 for r in rows if r.get("fallback"))
    n_mod = sum(1 for r in rows if r.get("real_modify"))
    accepted = [r for r in rows if r.get("accepted")]
    n_resid = sum(1 for r in accepted if r.get("residual_leak"))
    return {
        "n": len(rows),
        "schema_rate": n_schema / n,
        "fallback_rate": n_fb / n,
        "modify_rate": n_mod / n,
        "raw_leak_rate": sum(1 for r in rows if r.get("raw_leak")) / n,
        "n_accepted": len(accepted),
        "residual_leak_rate": (n_resid / len(accepted)) if accepted else 0.0,
        "residual_leak_count": n_resid,
    }


def seed_passes(m: dict[str, Any], cfg: dict | None = None) -> tuple[bool, list[str]]:
    g = (cfg or {}).get("gate") or {}
    reasons = []
    if m["schema_rate"] < float(g.get("schema_min", 0.95)):
        reasons.append("schema")
    if m["fallback_rate"] > float(g.get("fallback_max", 0.05)):
        reasons.append("fallback")
    if m["modify_rate"] < float(g.get("modify_min", 0.80)):
        reasons.append("modify")
    if m["residual_leak_count"] > 0 or m["residual_leak_rate"] > float(g.get("residual_leak_max", 0.0)):
        reasons.append("residual_leak")
    return (not reasons), reasons


def diversity_ok(by_entry: dict[str, dict[int, dict]], *, min_entries: int = 45, min_distinct: int = 3) -> dict[str, Any]:
    n_ok = 0
    detail = {}
    for eid, seeds in by_entry.items():
        hashes = []
        for s in LIBRARY_SEEDS:
            r = seeds.get(s) or {}
            if r.get("real_modify") and not r.get("fallback"):
                hashes.append(r.get("canonical_hash"))
        n_dist = len(set(hashes))
        ok = n_dist >= min_distinct
        detail[eid] = {"n_real_distinct": n_dist, "ok": ok}
        if ok:
            n_ok += 1
    return {"n_ok": n_ok, "n_entries": len(by_entry), "pass": n_ok >= min_entries, "detail": detail}


def disagreement_matrix(by_entry: dict[str, dict[int, dict]]) -> dict[str, float]:
    mat = {}
    eids = list(by_entry)
    n = max(len(eids), 1)
    for a in LIBRARY_SEEDS:
        for b in LIBRARY_SEEDS:
            if a >= b:
                continue
            d = 0
            for eid in eids:
                ha = (by_entry[eid].get(a) or {}).get("canonical_hash")
                hb = (by_entry[eid].get(b) or {}).get("canonical_hash")
                if ha != hb:
                    d += 1
            mat[f"{a}vs{b}"] = d / n
    return mat


def evaluate_gate_v(
    per_seed: dict[int, dict[str, Any]],
    div: dict[str, Any],
    structured_rate: float | None,
    *,
    human_ok: bool | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    g = (cfg or {}).get("gate") or {}
    seed_ok = {}
    all_seed = True
    for s in LIBRARY_SEEDS:
        ok, reasons = seed_passes(per_seed[s], cfg)
        seed_ok[str(s)] = {"pass": ok, "reasons": reasons, **per_seed[s]}
        all_seed = all_seed and ok
    reasons = []
    if not all_seed:
        reasons.append("per_seed")
    if all_seed and not div.get("pass"):
        reasons.append("diversity")
    struct_min = float(g.get("structured_validity_min", 0.90))
    if structured_rate is not None and structured_rate < struct_min:
        reasons.append("structured_validity")
    if human_ok is False:
        reasons.append("human_semantic")
    passed = not reasons
    claim = (
        "Gate V PASS: four seed-level library individuals are structurally valid, "
        "have zero residual leak, and show real cross-seed diversity. "
        "Not semantic preservation unless a signed human audit exists. "
        "Not a complete TAME method."
        if passed
        else "Gate V FAIL. Stop ALFWorld TAME on Qwen3-1.7B. Do not retune leak thresholds."
    )
    if passed and human_ok is not True:
        claim += " structured_validity only; do not write semantic preservation."
    return {
        "pass": passed,
        "reasons": reasons,
        "per_seed": seed_ok,
        "diversity": {k: v for k, v in div.items() if k != "detail"},
        "structured_validity_rate": structured_rate,
        "human_semantic": human_ok,
        "claim": claim,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def dump_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def stratified_audit_ids(sample_entries: list[dict], seeds: list[int], n: int, rng_seed: int) -> list[dict]:
    cells: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for e in sample_entries:
        for s in seeds:
            cells[(int(s), str(e.get("task_type") or "unknown"))].append({"id": e["id"], "seed": int(s), "task_type": e.get("task_type")})
    keys = sorted(cells)
    rng = random.Random(int(rng_seed))
    picked = []
    while len(picked) < n and any(cells[k] for k in keys):
        for k in keys:
            if len(picked) >= n:
                break
            bucket = cells[k]
            if not bucket:
                continue
            i = rng.randrange(len(bucket))
            picked.append(bucket.pop(i))
    return picked
