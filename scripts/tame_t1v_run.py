#!/usr/bin/env python3
"""T1-V: sample 60, generate 240 JSON patches, Gate V, then stop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tame_t1v_lib import (  # noqa: E402
    EDITOR_SYSTEM,
    LIBRARY_SEEDS,
    aggregate_error_stats,
    canonical_hash,
    disagreement_matrix,
    diversity_ok,
    dump_json,
    editor_prompt_hash,
    editor_user_prompt,
    entry_seed,
    evaluate_gate_v,
    format_workflow,
    load_cfg,
    load_jsonl,
    per_seed_metrics,
    process_generation,
    sample_60,
    seed_passes,
    stratified_audit_ids,
)


def generate_one(agent, tok, device, user: str, seed: int, max_new: int, temp: float) -> str:
    import torch

    messages = [{"role": "system", "content": EDITOR_SYSTEM}, {"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    torch.manual_seed(int(seed))
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = agent.model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=max(temp, 1e-5), top_p=1.0)
    new = out[0, ids["input_ids"].shape[1] :]
    return tok.decode(new, skip_special_tokens=True)


def structured_valid(row: dict, orig_lines: list[str]) -> bool:
    if row.get("fallback") or not row.get("accepted"):
        return False
    if row.get("residual_leak") or row.get("raw_leak"):
        return False
    if not row.get("real_modify"):
        return False
    rendered = row.get("rendered_lines") or []
    if not rendered:
        return False
    orig_verbs = {ln.split()[0] for ln in orig_lines if ln.split()}
    new_verbs = {ln.split()[0] for ln in rendered if ln.split()}
    if orig_verbs and not (orig_verbs & new_verbs):
        return False
    return True


def append_all_results(archive: Path, gate: dict) -> None:
    path = archive / "ALL_RESULTS.md"
    marker = "## TAME T1-V"
    block = (
        f"\n{marker}\n\n"
        "更新：2026-09-06。60-entry JSON-patch mutation gate。未重跑 OPD。\n\n"
        f"- Gate V: `{'PASS' if gate.get('pass') else 'FAIL'}` reasons=`{gate.get('reasons')}`\n"
        f"- 主张：{gate.get('claim')}\n"
        "- 产物：`reports/tame_t1v/` · `notes/tame_t1v_plan.md`\n"
        "- 停止。不自动扩全库，不调泄漏阈值。\n"
    )
    text = path.read_text() if path.exists() else ""
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n" + block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    reports = Path(cfg["paths"]["reports_dir"])
    data = Path(cfg["paths"]["data_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    m0 = json.loads(Path(cfg["paths"]["m0_library"]).read_text())
    sample = sample_60(m0["entries"], int(cfg["sample_n"]), int(cfg["sample_seed"]))
    dump_json(reports / "sample60.json", sample)
    by_id = {str(e["id"]): e for e in m0["entries"]}

    recs = load_jsonl(Path(cfg["paths"]["inner_nowf"])) + load_jsonl(Path(cfg["paths"]["inner_m0"]))
    stats = aggregate_error_stats(recs)
    dump_json(reports / "error_stats_by_type.json", stats)

    gens_path = data / "generations.jsonl"
    done = {}
    if args.resume and gens_path.exists():
        for rec in load_jsonl(gens_path):
            done[(rec["id"], int(rec["library_seed"]))] = rec

    need = [(eid, s) for eid in sample["ids"] for s in LIBRARY_SEEDS if (eid, s) not in done]
    if need:
        from agent_uce_opd_lib import build_theta0_agent

        print(f"generate {len(need)} remaining of {len(sample['ids'])*4}", flush=True)
        agent = build_theta0_agent(cfg)
        tok = agent.tokenizer
        device = agent._resolve_device()
        ed = cfg["editor"]
        t0 = time.time()
        for i, (eid, seed) in enumerate(need, 1):
            src = by_id[eid]
            user = editor_user_prompt(list(src["lines"]), stats.get(str(src.get("task_type"))))
            raw = generate_one(agent, tok, device, user, entry_seed(seed, eid), int(ed["max_new_tokens"]), float(ed["temperature"]))
            rec = process_generation(raw, list(src["lines"]), str(src.get("text") or ""), max_str=int(ed["max_string_chars"]))
            rec.update({"id": eid, "library_seed": seed, "task_type": src.get("task_type")})
            rec["rendered_text"] = format_workflow(rec["rendered_lines"])
            rec["original_hash"] = canonical_hash(src["lines"])
            with gens_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(eid, seed)] = rec
            if i % 20 == 0 or i == len(need):
                print(f"  {i}/{len(need)} fallback={rec.get('fallback')} modify={rec.get('real_modify')} wall={time.time()-t0:.0f}s", flush=True)
        agent.close()

    rows = [done[(eid, s)] for eid in sample["ids"] for s in LIBRARY_SEEDS]
    by_entry = {eid: {s: done[(eid, s)] for s in LIBRARY_SEEDS} for eid in sample["ids"]}
    per_seed = {s: per_seed_metrics([done[(eid, s)] for eid in sample["ids"]]) for s in LIBRARY_SEEDS}
    overall = per_seed_metrics(rows)
    div = diversity_ok(
        by_entry,
        min_entries=int(cfg["gate"]["diverse_min_entries"]),
        min_distinct=int(cfg["gate"]["min_distinct_real"]),
    )
    mat = disagreement_matrix(by_entry)

    audit_items = stratified_audit_ids(sample["entries"], list(LIBRARY_SEEDS), int(cfg["audit_n"]), 5050)
    audit_rows = []
    n_ok = 0
    for it in audit_items:
        rec = done[(it["id"], it["seed"])]
        src = by_id[it["id"]]
        ok = structured_valid(rec, list(src["lines"]))
        n_ok += int(ok)
        audit_rows.append({**it, "ok": ok, "fallback": rec.get("fallback"), "real_modify": rec.get("real_modify")})
    structured_rate = n_ok / max(len(audit_rows), 1)
    signoff = reports / "human_signoff.json"
    human_ok = None
    if signoff.exists():
        hs = json.loads(signoff.read_text())
        human_ok = int(hs.get("n_preserved") or 0) >= int(cfg["gate"]["human_semantic_min"])

    gate = evaluate_gate_v(per_seed, div, structured_rate, human_ok=human_ok, cfg=cfg)
    gate["overall_aux"] = overall
    gate["disagreement_matrix"] = mat
    gate["editor_prompt_hash"] = editor_prompt_hash()
    gate["n_generations"] = len(rows)
    dump_json(reports / "V_gate.json", {k: v for k, v in gate.items()})
    dump_json(reports / "per_seed_metrics.json", {str(k): v for k, v in per_seed.items()})
    dump_json(reports / "diversity.json", {k: v for k, v in div.items() if k != "detail"})
    dump_json(reports / "structured_audit.json", {"n": len(audit_rows), "n_ok": n_ok, "rate": structured_rate, "human_signoff": signoff.exists(), "items": audit_rows})

    seed_lines = ["| seed | schema | fallback | modify | residual | raw_leak | pass |", "|---|---:|---:|---:|---:|---:|---|"]
    for s in LIBRARY_SEEDS:
        m = per_seed[s]
        ok, reasons = seed_passes(m, cfg)
        seed_lines.append(
            f"| {s} | {m['schema_rate']:.3f} | {m['fallback_rate']:.3f} | {m['modify_rate']:.3f} | "
            f"{m['residual_leak_count']} | {m['raw_leak_rate']:.3f} | {'yes' if ok else 'no:'+','.join(reasons)} |"
        )
    lines = [
        "# T1-V Gate summary",
        "",
        "This gate tests whether four seed-level library individuals can be formed.",
        "It does not claim semantic preservation unless a signed human audit exists.",
        "It is not a complete TAME method.",
        "",
        f"- pass: `{gate['pass']}`",
        f"- reasons: `{gate['reasons']}`",
        f"- structured_validity_rate: `{structured_rate:.3f}` (not semantic preservation)",
        f"- diversity n_ok: `{div['n_ok']}/{div['n_entries']}` (need ≥{cfg['gate']['diverse_min_entries']})",
        f"- disagreement matrix: `{mat}`",
        f"- overall aux n=240: schema={overall['schema_rate']:.3f} fallback={overall['fallback_rate']:.3f} "
        f"modify={overall['modify_rate']:.3f} raw_leak={overall['raw_leak_rate']:.3f}",
        "",
        "## Per-seed",
        "",
        *seed_lines,
        "",
        gate["claim"],
        "",
        "Stopped after Gate V. No full-library rewrite. No OPD. Do not retune leak thresholds.",
    ]
    (reports / "V_summary.md").write_text("\n".join(lines) + "\n")
    append_all_results(Path(cfg["paths"]["archive"]), gate)
    print(json.dumps({"pass": gate["pass"], "reasons": gate["reasons"], "per_seed": {s: gate["per_seed"][s]["pass"] for s in gate["per_seed"]}, "diversity": div["n_ok"], "structured_validity_rate": structured_rate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
