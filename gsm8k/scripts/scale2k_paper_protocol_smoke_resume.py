#!/usr/bin/env python3
"""Paper-protocol smoke with per-batch checkpoint + --resume (tmp overlay)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
if str(SSOPD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSOPD_ROOT))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.nxt.bootstrap import task_cluster_bootstrap
from ssopd_math.verifier.reward import POSITIVE, verify_completion


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_per_problem(records: list[dict], problem_ids: set[str]) -> list[dict]:
    out, seen = [], set()
    for r in records:
        pid = str(r["problem_id"])
        if pid not in problem_ids or pid in seen:
            continue
        seen.add(pid)
        out.append(r)
    return out


def resolve_direction_key(ss: dict, layer: int) -> tuple[str, str]:
    explicit = ss.get("direction_key")
    variant = ss.get("direction_variant", "task_balanced_paired")
    vec_kind = str(ss.get("direction_vector", "v_cap")).lower()
    mode = str(ss.get("inject_mode", "contrastive_cap"))
    if explicit:
        key = str(explicit)
    else:
        suffix = "v_cap" if vec_kind in ("v_cap", "cap") else "v_hat"
        key = f"layer_{layer}_{variant}_{suffix}"
        if suffix == "v_hat" and mode in ("contrastive_cap", "cap", "paper"):
            mode = "unit_additive"
    if vec_kind in ("v_cap", "cap") or key.endswith("_v_cap"):
        mode = "contrastive_cap"
    return key, mode


def _aggregate_per(per: dict[str, dict]) -> tuple[int, int, list[int]]:
    correct = sum(v["correct"] for v in per.values())
    parse_ok = sum(v["parse_ok"] for v in per.values())
    lengths = [v["length"] for v in per.values()]
    return correct, parse_ok, lengths


def audit_batch(
    model: HFMathModel,
    records: list[dict],
    *,
    layer: int | None,
    direction: torch.Tensor | None,
    alpha: float,
    seed: int,
    batch_size: int,
    steer_mode: str,
    per_problem: dict[str, dict] | None = None,
    on_batch_done: Callable[[dict[str, dict], int, int], None] | None = None,
) -> dict:
    per: dict[str, dict] = dict(per_problem or {})
    n = len(records)
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = [
            r for r in records[start : start + batch_size] if str(r["problem_id"]) not in per
        ]
        if chunk:
            prompts = [r.get("prompt_user", r["problem"]) for r in chunk]
            golds = [r["gold_answer"] for r in chunk]
            gens = model.generate_batch(
                prompts,
                golds,
                seed=seed + start,
                steer_layer=layer if abs(alpha) > 1e-12 else None,
                steer_direction=direction if abs(alpha) > 1e-12 else None,
                steer_alpha=float(alpha),
                steer_mode=steer_mode,
            )
            for rec, gen, gold in zip(chunk, gens, golds):
                ver = verify_completion(gen.text, gold)
                per[str(rec["problem_id"])] = {
                    "correct": int(ver["verification_status"] == POSITIVE),
                    "parse_ok": int(bool(ver.get("parse_ok", False))),
                    "length": len(gen.completion_token_ids),
                }
        done = min(start + batch_size, n)
        if on_batch_done is not None:
            on_batch_done(per, done, n)
        mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        print(f"audit rollout {done}/{n} alpha={alpha} peak_alloc_gb={mem:.1f}", flush=True)
    correct, parse_ok, lengths = _aggregate_per(per)
    return {
        "n": n,
        "accuracy": correct / max(n, 1),
        "parse_rate": parse_ok / max(n, 1),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "per_problem": per,
        "wall_s": time.time() - t0,
    }


def split_metrics(
    clean_per: dict[str, dict],
    steered_per: dict[str, dict],
    problem_splits: dict[str, str],
    split_name: str,
) -> dict:
    ids = [pid for pid, s in problem_splits.items() if s == split_name]
    ids = [pid for pid in ids if pid in clean_per and pid in steered_per]
    if not ids:
        return {"n": 0, "clean_acc": 0.0, "steered_acc": 0.0, "gain_pp": 0.0}
    clean_acc = sum(clean_per[i]["correct"] for i in ids) / len(ids)
    steered_acc = sum(steered_per[i]["correct"] for i in ids) / len(ids)
    return {
        "n": len(ids),
        "clean_acc": clean_acc,
        "steered_acc": steered_acc,
        "gain_pp": (steered_acc - clean_acc) * 100.0,
    }


def both_finished_metrics(
    clean_per: dict[str, dict],
    steered_per: dict[str, dict],
    mt: int,
) -> dict:
    ids = [pid for pid in clean_per if pid in steered_per]
    both = [
        pid
        for pid in ids
        if clean_per[pid]["length"] < mt and steered_per[pid]["length"] < mt
    ]
    if not both:
        return {
            "n": 0,
            "clean_acc": 0.0,
            "steered_acc": 0.0,
            "gain_pp": 0.0,
            "clean_hit_frac": 0.0,
            "steered_hit_frac": 0.0,
            "steered_only": 0,
            "steered_only_from_clean_hit": 0,
            "clean_only": 0,
        }
    clean_acc = sum(clean_per[i]["correct"] for i in both) / len(both)
    steered_acc = sum(steered_per[i]["correct"] for i in both) / len(both)
    clean_hit = sum(1 for i in ids if clean_per[i]["length"] >= mt) / max(len(ids), 1)
    steered_hit = sum(1 for i in ids if steered_per[i]["length"] >= mt) / max(len(ids), 1)
    steered_only = [
        i for i in ids if not clean_per[i]["correct"] and steered_per[i]["correct"]
    ]
    from_hit = sum(1 for i in steered_only if clean_per[i]["length"] >= mt)
    clean_only = sum(1 for i in ids if clean_per[i]["correct"] and not steered_per[i]["correct"])
    return {
        "n": len(both),
        "clean_acc": clean_acc,
        "steered_acc": steered_acc,
        "gain_pp": (steered_acc - clean_acc) * 100.0,
        "clean_hit_frac": clean_hit,
        "steered_hit_frac": steered_hit,
        "steered_only": len(steered_only),
        "steered_only_from_clean_hit": from_hit,
        "clean_only": clean_only,
    }


def write_truncation_gate_report(
    md_path: Path,
    *,
    cfg: dict,
    key: str,
    vnorm: float,
    batch_size: int,
    split_name: str,
    clean: dict,
    curve: list[dict],
    best_overall_alpha: float,
    best_both_alpha: float | None,
    best_both_gain: float,
) -> None:
    mt = int(cfg["model"].get("max_new_tokens", 2048))
    lines = [
        f"# Truncation-gate α sweep ({split_name}, mt={mt})",
        "",
        f"- direction `{key}`, ‖v‖={vnorm:.4f}",
        f"- n={clean['n']}, batch={batch_size}, clean acc={clean['accuracy']*100:.2f}%",
        f"- Gate: **both-finished gain ≥ +2 pp**（非 overall gain）",
        "",
        "| alpha | overall_gain | both_n | both_gain | hit_clean | hit_steered | steered_only (from hit) |",
        "|------:|-------------:|-------:|----------:|----------:|------------:|------------------------:|",
    ]
    for row in curve:
        if row["alpha"] == 0.0:
            continue
        bf = row.get("both_finished", {})
        lines.append(
            f"| {row['alpha']} | {row['gain_pp']:.2f} | {bf.get('n', 0)} | "
            f"{bf.get('gain_pp', 0):.2f} | {bf.get('clean_hit_frac', 0)*100:.1f}% | "
            f"{bf.get('steered_hit_frac', 0)*100:.1f}% | "
            f"{bf.get('steered_only', 0)} ({bf.get('steered_only_from_clean_hit', 0)}) |"
        )
    both_pass = best_both_gain >= 2.0
    lines.extend(
        [
            "",
            f"- Best **overall** α={best_overall_alpha}",
            f"- Best **both-finished** α={best_both_alpha}, gain={best_both_gain:.2f} pp",
            f"- Truncation-gate (>=2pp both-finished): **{both_pass}**",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))


def dirscale_label_from_cfg(cfg: dict) -> str:
    directions = str(cfg["paths"].get("directions", ""))
    m = re.search(r"/N(\d+)/", directions)
    return f"N{m.group(1)}" if m else "DirScale"


def write_all2k_report(
    md_path: Path,
    *,
    cfg: dict,
    label: str,
    confirm_gain_ref: float | None,
    key: str,
    steer_mode: str,
    vnorm: float,
    batch_size: int,
    clean: dict,
    steered: dict,
    alpha: float,
    problem_splits: dict[str, str],
    seed: int,
) -> None:
    clean_per = clean["per_problem"]
    steered_per = steered["per_problem"]
    gain_pp = (steered["accuracy"] - clean["accuracy"]) * 100.0
    paired = {
        pid: float(steered_per[pid]["correct"] - clean_per[pid]["correct"])
        for pid in clean_per
        if pid in steered_per
    }
    boot = task_cluster_bootstrap(paired, seed=seed)
    mt = int(cfg["model"].get("max_new_tokens", 2048))
    rows = []
    for sp in ("vector_fit", "selection", "confirm"):
        m = split_metrics(clean_per, steered_per, problem_splits, sp)
        rows.append((sp, m))
    both_lt = [
        pid
        for pid in clean_per
        if pid in steered_per
        and clean_per[pid]["length"] < mt
        and steered_per[pid]["length"] < mt
    ]
    if both_lt:
        c_acc = sum(clean_per[i]["correct"] for i in both_lt) / len(both_lt)
        s_acc = sum(steered_per[i]["correct"] for i in both_lt) / len(both_lt)
        both_gain = (s_acc - c_acc) * 100.0
    else:
        both_gain = 0.0
        c_acc = s_acc = 0.0
    lines = [
        f"# {label} DirScale · 全 2000 题 L14 注入 @ α=2 (`max_new_tokens=2048`)",
        "",
        f"- 方向：{label} `vector_fit` paired，`{key}`，‖v‖={vnorm:.4f}",
        f"- 协议：L14 `contrastive_cap`，α=**{alpha}**，n=**{clean['n']}**，batch={batch_size}",
        f"- Config: `{cfg['experiment_id']}`",
        "",
        "## Headline（全体 2000 题）",
        "",
        "| setting | clean acc | steered@α=2 | gain_pp |",
        "|---------|----------:|------------:|--------:|",
        f"| all2000 | {clean['accuracy']*100:.2f}% | {steered['accuracy']*100:.2f}% | **{gain_pp:.2f}** |",
        "",
        f"- Paired bootstrap 95% CI on gain: **[{boot['ci_low']*100:.2f}, {boot['ci_high']*100:.2f}] pp**",
        f"- Gate (>=2pp): **{gain_pp >= 2.0}**",
        "",
        "## 按 split 分层",
        "",
        "| split | n | clean acc | steered acc | gain_pp |",
        "|-------|--:|----------:|------------:|--------:|",
    ]
    for sp, m in rows:
        lines.append(
            f"| {sp} | {m['n']} | {m['clean_acc']*100:.2f}% | {m['steered_acc']*100:.2f}% | {m['gain_pp']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 未截断子集",
            "",
            f"| subset | n | clean acc | steered acc | gain_pp |",
            f"|--------|--:|----------:|------------:|--------:|",
            f"| both length < {mt} | {len(both_lt)} | {c_acc*100:.1f}% | {s_acc*100:.1f}% | {both_gain:.2f} |",
            "",
            "## 与 confirm400 对照",
            "",
            f"- {label} confirm400 @α=2 mt2048：**{confirm_gain_ref:+.2f} pp**（同方向，holdout 400）"
            if confirm_gain_ref is not None
            else f"- {label} confirm400 @α=2 mt2048：见 `reports/ssopd03_dirscale_{label}/results_confirm.json`",
            f"- 全 2000 题（含 vector_fit 1200）：**{gain_pp:.2f} pp**",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="selection", choices=["selection", "confirm", "all"])
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--output-dir", default=None, help="Override reports output directory.")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["paths"]["root"])
    reports_dir = Path(args.output_dir) if args.output_dir else root / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "scale2k_paper_protocol.log"
    ckpt_path = reports_dir / f"checkpoint_{args.split}.json"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    def save_ckpt(clean_per: dict, steered_map: dict[str, dict], phase: str) -> None:
        ckpt_path.write_text(
            json.dumps(
                {
                    "experiment_id": cfg["experiment_id"],
                    "split": args.split,
                    "phase": phase,
                    "clean_per_problem": clean_per,
                    "steered_per_problem": steered_map,
                },
                indent=2,
            )
        )

    ss = cfg["ssopd03"]
    layer = int(ss["layer_index"])
    alphas = [float(a) for a in ss["alpha_grid"]]
    if args.alpha is not None:
        alphas = [float(args.alpha)]
    batch_size = int(ss.get("audit_batch_size") or cfg.get("audit_batch_size") or 48)
    key, steer_mode = resolve_direction_key(ss, layer)

    splits = json.loads((root / cfg["paths"]["splits"]).read_text())
    split_name = args.split
    if split_name == "all":
        ids = sorted(splits["problem_splits"].keys())
    else:
        ids = sorted(pid for pid, s in splits["problem_splits"].items() if s == split_name)
    if args.max_problems is not None:
        ids = ids[: int(args.max_problems)]
    records = one_per_problem(load_jsonl(root / cfg["paths"]["rollouts"]), set(ids))
    records = sorted(records, key=lambda r: str(r["problem_id"]))

    ckpt = json.loads(ckpt_path.read_text()) if args.resume and ckpt_path.exists() else None
    clean_per: dict[str, dict] = (ckpt or {}).get("clean_per_problem", {})
    steered_ckpt: dict[str, dict] = (ckpt or {}).get("steered_per_problem", {})
    if ckpt:
        log(f"resume: loaded checkpoint clean={len(clean_per)} steered_keys={list(steered_ckpt)}")

    z = np.load(root / cfg["paths"]["directions"])
    direction = torch.tensor(z[key], dtype=torch.float32, device=cfg["model"].get("device", "cuda"))
    vnorm = float(direction.norm().item())
    mc = cfg["model"]
    log(
        f"PAPER SMOKE start split={split_name} layer={layer} key={key} "
        f"mode={steer_mode} ||v||={vnorm:.4f} n={len(records)} alphas={alphas} batch={batch_size} "
        f"resume={args.resume} out={reports_dir}"
    )

    model = HFMathModel(
        mc["path"],
        device=mc.get("device", "cuda"),
        torch_dtype=mc.get("torch_dtype", "bfloat16"),
        local_files_only=True,
        use_cache=True,
        max_new_tokens=int(mc.get("max_new_tokens", 1024)),
        temperature=float(mc.get("temperature", 1.0)),
        top_p=float(mc.get("top_p", 0.95)),
        do_sample=bool(mc.get("do_sample", True)),
    )
    seed = int(cfg.get("random_seed", 42))
    t0 = time.time()
    steered_by_alpha: dict[str, dict] = {}

    steered_map: dict[str, dict] = dict(steered_ckpt)

    def on_clean_batch(per: dict[str, dict], done: int, total: int) -> None:
        save_ckpt(per, steered_map, "clean")

    if len(clean_per) >= len(records):
        log(f"resume: clean complete ({len(clean_per)}/{len(records)}), skipping generation")
        correct, parse_ok, lengths = _aggregate_per(clean_per)
        clean = {
            "n": len(records),
            "accuracy": correct / max(len(records), 1),
            "parse_rate": parse_ok / max(len(records), 1),
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "per_problem": clean_per,
            "wall_s": 0.0,
        }
    else:
        log("clean alpha=0")
        clean = audit_batch(
            model,
            records,
            layer=None,
            direction=None,
            alpha=0.0,
            seed=seed,
            batch_size=batch_size,
            steer_mode=steer_mode,
            per_problem=clean_per,
            on_batch_done=on_clean_batch,
        )
        clean_per = clean["per_problem"]

    curve = [
        {
            "alpha": 0.0,
            "accuracy": clean["accuracy"],
            "gain_pp": 0.0,
            **{k: clean[k] for k in ("n", "parse_rate", "mean_length", "wall_s")},
        }
    ]
    best_alpha, best_acc = 0.0, clean["accuracy"]

    for a in alphas:
        akey = str(a)
        existing = steered_map.get(akey, {})
        log(f"steered alpha={a} (have {len(existing)}/{len(records)})")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def on_steered_batch(per: dict[str, dict], done: int, total: int, _a=akey) -> None:
            steered_map[_a] = per
            save_ckpt(clean_per, steered_map, f"steered_{_a}")

        if len(existing) >= len(records):
            log(f"resume: steered alpha={a} complete, skipping")
            correct, parse_ok, lengths = _aggregate_per(existing)
            steered = {
                "n": len(records),
                "accuracy": correct / max(len(records), 1),
                "parse_rate": parse_ok / max(len(records), 1),
                "mean_length": float(np.mean(lengths)) if lengths else 0.0,
                "per_problem": existing,
                "wall_s": 0.0,
            }
        else:
            steered = audit_batch(
                model,
                records,
                layer=layer,
                direction=direction,
                alpha=a,
                seed=seed + 10_000,
                batch_size=batch_size,
                steer_mode=steer_mode,
                per_problem=existing,
                on_batch_done=on_steered_batch,
            )
            steered_map[akey] = steered["per_problem"]

        gain_pp = (steered["accuracy"] - clean["accuracy"]) * 100.0
        mt = int(mc.get("max_new_tokens", 2048))
        bf = both_finished_metrics(clean["per_problem"], steered["per_problem"], mt)
        curve.append(
            {
                "alpha": a,
                "accuracy": steered["accuracy"],
                "gain_pp": gain_pp,
                "n": steered["n"],
                "parse_rate": steered["parse_rate"],
                "mean_length": steered["mean_length"],
                "wall_s": steered["wall_s"],
                "both_finished": bf,
            }
        )
        steered_by_alpha[akey] = steered
        log(
            f"{split_name} sweep alpha={a} acc={steered['accuracy']:.4f} "
            f"gain_pp={gain_pp:.2f} both_finished_n={bf['n']} both_gain_pp={bf['gain_pp']:.2f}"
        )
        if steered["accuracy"] > best_acc:
            best_acc = steered["accuracy"]
            best_alpha = a

    best_gain = (best_acc - clean["accuracy"]) * 100.0
    best_both_alpha = None
    best_both_gain = float("-inf")
    for row in curve:
        if row["alpha"] == 0.0:
            continue
        g = float(row.get("both_finished", {}).get("gain_pp", float("-inf")))
        if g > best_both_gain:
            best_both_gain = g
            best_both_alpha = row["alpha"]
    if best_both_alpha is None:
        best_both_gain = 0.0

    results = {
        "experiment_id": cfg["experiment_id"],
        "protocol": "paper_2504.19483_adapted_truncation_gate",
        "split": split_name,
        "layer": layer,
        "direction_key": key,
        "steer_mode": steer_mode,
        "direction_norm": vnorm,
        "max_new_tokens": int(mc.get("max_new_tokens", 1024)),
        "selection_n": len(records),
        "audit_batch_size": batch_size,
        "clean": {k: clean[k] for k in ("n", "accuracy", "parse_rate", "mean_length", "wall_s")},
        "selection_curve": curve,
        "best_alpha": best_alpha,
        "best_gain_pp": best_gain,
        "best_both_finished_alpha": best_both_alpha,
        "best_both_finished_gain_pp": best_both_gain,
        "gate_pass_ge_2pp": bool(best_gain >= 2.0),
        "truncation_gate_pass_ge_2pp": bool(best_both_gain >= 2.0),
        "wall_clock_s": time.time() - t0,
        "per_problem": {
            "clean": clean["per_problem"],
            "steered": {a: steered_by_alpha[a]["per_problem"] for a in steered_by_alpha},
        },
    }
    tag = "results.json" if split_name == "selection" else f"results_{split_name}.json"
    out = reports_dir / tag
    out.write_text(json.dumps(results, indent=2))

    if split_name == "all" and alphas:
        label = dirscale_label_from_cfg(cfg)
        confirm_gain_ref = None
        confirm_path = root / f"reports/ssopd03_dirscale_{label}/results_confirm.json"
        if confirm_path.exists():
            confirm_gain_ref = float(
                json.loads(confirm_path.read_text()).get("best_gain_pp", 0.0)
            )
        md_name = Path(cfg["paths"].get("report_md", f"reports/ssopd03_dirscale_{label}_all2k_report.md")).name
        md = reports_dir / md_name
        write_all2k_report(
            md,
            cfg=cfg,
            label=label,
            confirm_gain_ref=confirm_gain_ref,
            key=key,
            steer_mode=steer_mode,
            vnorm=vnorm,
            batch_size=batch_size,
            clean=clean,
            steered=steered_by_alpha[str(alphas[0])],
            alpha=alphas[0],
            problem_splits=splits["problem_splits"],
            seed=seed,
        )
        log(f"wrote report {md}")

    if split_name == "confirm" and alphas:
        md_name = Path(
            cfg["paths"].get("report_md", "reports/truncation_gate_alpha_sweep_report.md")
        ).name
        md = reports_dir / md_name
        write_truncation_gate_report(
            md,
            cfg=cfg,
            key=key,
            vnorm=vnorm,
            batch_size=batch_size,
            split_name=split_name,
            clean=clean,
            curve=curve,
            best_overall_alpha=best_alpha,
            best_both_alpha=best_both_alpha,
            best_both_gain=best_both_gain,
        )
        log(f"wrote truncation-gate report {md}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        log(f"removed checkpoint {ckpt_path}")

    log(f"PAPER SMOKE done best_alpha={best_alpha} gain_pp={best_gain:.2f} wrote {out}")
    model.close()


if __name__ == "__main__":
    main()
