#!/usr/bin/env python3
"""Generate frozen TAME population M0-M4, audit, fallback, freeze."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_ssopd_teacher_eval import run_npm_episodes  # noqa: E402
from agent_uce_opd_lib import build_theta0_agent  # noqa: E402
from alfworld_common import dump_json, ensure_alfworld_env, load_jsonl  # noqa: E402
from tame_opd_lib import (  # noqa: E402
    CANDIDATE_ORDER,
    LIBRARY_SEEDS,
    apply_deterministic_fallback,
    as_uce_library,
    audit_library,
    build_editor_messages,
    build_m0_library,
    editor_prompt_hash,
    editor_user_prompt,
    entry_seed,
    format_workflow,
    leading_verb,
    load_manifest,
    load_tame_cfg,
    object_container_pairs,
    parse_rewrite_lines,
    sample_audit_ids,
    save_library,
    sha256_text,
    tasks_of,
    verbs_from_admissible,
    with_rollout_seed,
)
from uce_eval import make_workflow_fn  # noqa: E402
from uce_library import UceLibrary  # noqa: E402


def _collect(cfg, agent, tasks, out_path: Path, workflow_fn, split: str, resume: bool):
    return run_npm_episodes(
        agent=agent,
        tasks=tasks,
        cfg=cfg,
        memory=None,
        alpha=0.0,
        top_k=8,
        use_gate=False,
        gate_threshold=0.45,
        static_vector=None,
        out_path=out_path,
        resume=resume,
        dataset_split=split,
        workflow_fn=workflow_fn,
    )


def _trace_note(rec: dict) -> str:
    acts = [str(a) for a in (rec.get("actions") or [])][:8]
    body = "; ".join(acts)
    flag = "success" if rec.get("episode_success") else "fail"
    return f"{rec.get('task_type')} {flag}: {body}"[:400]


def _diagnostics(recs: list[dict]) -> str:
    n_inv = 0
    n_rep = 0
    n_dp = 0
    for rec in recs:
        prev = None
        for dp in rec.get("decision_points") or []:
            n_dp += 1
            if dp.get("failure_reason") == "not_admissible" or (
                isinstance(dp.get("verdict"), dict) and dp["verdict"].get("failure_reason") == "not_admissible"
            ):
                n_inv += 1
            act = str(dp.get("raw_action_text") or dp.get("action_text") or "")
            if prev and act == prev:
                n_rep += 1
            prev = act
    return f"invalid_or_inadmissible={n_inv}/{n_dp}; repeated_actions={n_rep}"


def _extra_pairs(recs: list[dict]) -> list[str]:
    pairs: set[str] = set()
    for rec in recs:
        for dp in rec.get("decision_points") or []:
            obs = str(dp.get("prefix_text") or "")
            pairs.update(object_container_pairs(obs))
    return sorted(pairs)


def collect_inner(cfg, manifest, reports: Path, resume: bool) -> tuple[list[dict], list[dict]]:
    tasks = tasks_of(manifest, "inner_train")
    agent = build_theta0_agent(cfg)
    cfg_r = with_rollout_seed(cfg, 2020)
    m0_path = Path(cfg["paths"]["uce_library_evolved"])
    lib = UceLibrary.load(m0_path)
    nowf = _collect(cfg_r, agent, tasks, reports / "inner_train_nowf.jsonl", None, "inner_train", resume)
    w_m0 = _collect(
        cfg_r,
        agent,
        tasks,
        reports / "inner_train_m0.jsonl",
        make_workflow_fn(lib),
        "inner_train",
        resume,
    )
    agent.close()
    return nowf, w_m0


def notes_by_type(recs: list[dict]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"success": [], "fail": []})
    for rec in recs:
        tt = str(rec.get("task_type") or "unknown")
        key = "success" if rec.get("episode_success") else "fail"
        out[tt][key].append(_trace_note(rec))
    return out


def build_whitelist(recs: list[dict], reports: Path) -> set[str]:
    verbs: set[str] = set()
    for rec in recs:
        for dp in rec.get("decision_points") or []:
            state = dp.get("state") or {}
            verbs |= verbs_from_admissible(list(state.get("valid_tools") or []))
            act = str(dp.get("predicted_tool") or dp.get("raw_action_text") or "")
            v = leading_verb(act)
            if v:
                verbs.add(v)
    payload = {"verbs": sorted(verbs), "source": "inner_train_admissible_plus_parser_leading_token"}
    dump_json(reports / "action_verb_whitelist.json", payload)
    (reports / "action_verb_whitelist.sha256").write_text(sha256_text(json.dumps(payload, sort_keys=True)) + "\n")
    return verbs


def generate_candidate(
    *,
    cid: str,
    library_seed: int,
    m0: dict,
    notes: dict[str, dict[str, list[str]]],
    diagnostics: str,
    agent,
    tok,
    device,
    cfg,
    out_dir: Path,
    resume: bool,
) -> dict:
    ed = cfg["editor"]
    temp = float(ed["temperature"])
    max_new = int(ed["max_new_tokens"])
    partial = out_dir / "partial.jsonl"
    done: dict[str, dict] = {}
    if resume and partial.exists():
        for rec in load_jsonl(partial):
            done[str(rec["id"])] = rec
    entries = []
    t0 = time.time()
    for i, src in enumerate(m0["entries"], 1):
        eid = str(src["id"])
        if eid in done:
            entries.append(done[eid])
            continue
        seed = entry_seed(library_seed, eid)
        tt = str(src.get("task_type") or "unknown")
        user = editor_user_prompt(
            original_text=str(src.get("text") or ""),
            task_type=tt,
            success_notes=notes.get(tt, {}).get("success") or [],
            fail_notes=notes.get(tt, {}).get("fail") or [],
            diagnostics=diagnostics,
        )
        messages = build_editor_messages(user)
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        torch.manual_seed(seed)
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = agent.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=max(temp, 1e-5),
                top_p=1.0,
            )
        new_ids = gen[0, ids["input_ids"].shape[1] :]
        raw = tok.decode(new_ids, skip_special_tokens=True)
        lines = parse_rewrite_lines(raw)
        fallback = not bool(lines)
        if fallback:
            text = str(src.get("text") or "")
            lines = list(src.get("lines") or [])
        else:
            text = format_workflow(lines)
        rec = {
            "id": src["id"],
            "task_type": src.get("task_type"),
            "goal": src.get("goal"),
            "goal_tokens": src.get("goal_tokens"),
            "lines": lines,
            "text": text,
            "source_task_id": src.get("source_task_id"),
            "usage": src.get("usage"),
            "parent_id": src.get("id"),
            "parent_hash": sha256_text(str(src.get("text") or "")),
            "gen_seed": seed,
            "library_seed": library_seed,
            "fallback": fallback,
            "reverted": False,
            "raw_rewrite": raw[:2000],
        }
        entries.append(rec)
        with partial.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0:
            print(f"  {cid} {i}/{len(m0['entries'])} fallback={fallback} wall={time.time()-t0:.0f}s", flush=True)
    return {
        "n_entries": len(entries),
        "candidate_id": cid,
        "library_seed": library_seed,
        "editor_prompt_hash": editor_prompt_hash(),
        "entries": entries,
    }


def structured_sample_audit(lib: dict, m0: dict, audit: dict, reports: Path, n: int, seed: int) -> dict:
    ids = sample_audit_ids(lib.get("entries") or [], n, seed)
    m0_by = {str(e["id"]): e for e in m0.get("entries") or []}
    a_by = {str(r["id"]): r for r in audit.get("entries") or []}
    e_by = {str(e["id"]): e for e in lib.get("entries") or []}
    lines = [
        f"# structured_sample_audit {lib.get('candidate_id')}",
        "",
        "This is a structured sample audit, not a human audit.",
        "",
    ]
    flagged = []
    for eid in ids:
        e = e_by[eid]
        a = a_by.get(eid, {})
        m0e = m0_by.get(eid, {})
        leak = bool(a.get("leaks"))
        illegal = bool(a.get("illegal_verbs"))
        if leak or illegal:
            flagged.append(eid)
        lines.append(f"## {eid}")
        lines.append(f"- task_type: {e.get('task_type')}")
        lines.append(f"- verb_retention: {a.get('verb_retention')}")
        lines.append(f"- goal_token_coverage: {a.get('goal_token_coverage')}")
        lines.append(f"- leaks: {a.get('leaks')}")
        lines.append(f"- illegal_verbs: {a.get('illegal_verbs')}")
        lines.append("- original:")
        lines.append("```")
        lines.append(str(m0e.get("text") or ""))
        lines.append("```")
        lines.append("- rewrite:")
        lines.append("```")
        lines.append(str(e.get("text") or ""))
        lines.append("```")
        lines.append("")
    out = reports / f"structured_sample_audit_{lib.get('candidate_id')}.md"
    out.write_text("\n".join(lines))
    return {"path": str(out), "n": len(ids), "flagged_ids": flagged, "human_signoff": False}


def freeze_population(cfg, reports: Path, candidates: dict[str, dict], audits: dict[str, dict]) -> dict:
    hashes = {}
    valid = {}
    for cid in CANDIDATE_ORDER:
        if cid not in candidates:
            continue
        digest = save_library(Path(cfg["paths"]["candidates_dir"]) / cid / "library.json", candidates[cid])
        hashes[cid] = digest
        valid[cid] = bool((audits.get(cid) or {}).get("valid", cid == "M0"))
    payload = {
        "frozen": True,
        "editor_prompt_hash": editor_prompt_hash(),
        "library_hashes": hashes,
        "valid": valid,
        "note": "Libraries frozen. No further repair or regeneration.",
    }
    dump_json(reports / "population_freeze.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()
    cfg = load_tame_cfg(args.config)
    ensure_alfworld_env(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    cand_root = Path(cfg["paths"]["candidates_dir"])
    cand_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(reports / "task_manifest.json")

    evolved = json.loads(Path(cfg["paths"]["uce_library_evolved"]).read_text())
    m0 = build_m0_library(evolved)
    save_library(cand_root / "M0" / "library.json", m0)

    nowf, w_m0 = collect_inner(cfg, manifest, reports, args.resume)
    notes = notes_by_type(nowf + w_m0)
    diagnostics = _diagnostics(nowf + w_m0)
    whitelist = build_whitelist(nowf + w_m0, reports)
    extras_goals = [str(t.get("goal") or "") for t in tasks_of(manifest, "inner_train")]
    extras_pairs = _extra_pairs(nowf + w_m0)
    fb_max = float(cfg["gate"]["fallback_rate_max"])

    if not args.skip_generate:
        agent = build_theta0_agent(cfg)
        tok = agent.tokenizer
        device = agent._resolve_device()
        for cid, seed in LIBRARY_SEEDS.items():
            print(f"generate {cid} library_seed={seed}", flush=True)
            out_dir = cand_root / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = generate_candidate(
                cid=cid,
                library_seed=int(seed),
                m0=m0,
                notes=notes,
                diagnostics=diagnostics,
                agent=agent,
                tok=tok,
                device=device,
                cfg=cfg,
                out_dir=out_dir,
                resume=args.resume,
            )
            save_library(out_dir / "library.raw.json", payload)
        agent.close()

    candidates = {"M0": m0}
    audits = {}
    samples = {}
    for cid in CANDIDATE_ORDER:
        if cid == "M0":
            candidates["M0"] = m0
            audits["M0"] = audit_library(m0, m0, whitelist=whitelist, fallback_rate_max=fb_max)
            audits["M0"]["valid"] = True
            continue
        raw_path = cand_root / cid / "library.raw.json"
        if not raw_path.exists():
            raise SystemExit(f"missing generated library {raw_path}")
        raw = json.loads(raw_path.read_text())
        auto1 = audit_library(
            raw,
            m0,
            whitelist=whitelist,
            extra_goals=extras_goals,
            extra_pairs=extras_pairs,
            fallback_rate_max=fb_max,
        )
        sample = structured_sample_audit(
            raw, m0, auto1, reports, int(cfg["gate"]["sample_audit_n"]), seed=int(LIBRARY_SEEDS[cid])
        )
        samples[cid] = sample
        fail_audit = dict(auto1)
        extra_fail = set(sample.get("flagged_ids") or [])
        for row in fail_audit.get("entries") or []:
            if str(row["id"]) in extra_fail:
                row["ok"] = False
                row.setdefault("reasons", []).append("structured_sample_flag")
        reverted = apply_deterministic_fallback(raw, m0, fail_audit)
        auto2 = audit_library(
            reverted,
            m0,
            whitelist=whitelist,
            extra_goals=extras_goals,
            extra_pairs=extras_pairs,
            fallback_rate_max=fb_max,
        )
        candidates[cid] = reverted
        audits[cid] = auto2
        dump_json(reports / f"audit_{cid}_pass1.json", {k: v for k, v in auto1.items() if k != "entries"})
        dump_json(reports / f"audit_{cid}.json", {k: v for k, v in auto2.items() if k != "entries"})
        dump_json(reports / f"audit_{cid}_full.json", auto2)

    freeze = freeze_population(cfg, reports, candidates, audits)
    dump_json(reports / "candidate_audit.json", {cid: {k: v for k, v in a.items() if k != "entries"} for cid, a in audits.items()})
    dump_json(reports / "structured_sample_audit.json", samples)
    print(json.dumps({"freeze": freeze["library_hashes"], "valid": freeze["valid"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
