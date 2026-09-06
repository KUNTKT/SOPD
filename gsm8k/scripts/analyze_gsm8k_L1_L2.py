#!/usr/bin/env python3
"""L1 length bins + L2 both-finished for GSM8K confirm α sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BINS = [
    ("[0,512)", 0, 512),
    ("[512,1024)", 512, 1024),
    ("[1024,1536)", 1024, 1536),
    ("[1536,2048)", 1536, 2048),
    ("[2048,inf)", 2048, 10**9),
]


def acc(per: dict, ids: list[str]) -> float:
    if not ids:
        return 0.0
    return sum(per[i]["correct"] for i in ids) / len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_confirm.json from smoke resume")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--mt", type=int, default=2048)
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())
    clean = res["per_problem"]["clean"]
    steered_map = res["per_problem"]["steered"]
    mt = int(args.mt)

    report = {
        "mt": mt,
        "n": len(clean),
        "clean_acc": sum(v["correct"] for v in clean.values()) / max(len(clean), 1),
        "clean_mean_len": sum(v["length"] for v in clean.values()) / max(len(clean), 1),
        "clean_hit_frac": sum(1 for v in clean.values() if v["length"] >= mt) / max(len(clean), 1),
        "alphas": {},
        "branch": None,
    }

    md = [
        f"# GSM8K L1/L2 confirm (mt={mt})",
        "",
        f"- n={report['n']}",
        f"- clean acc={report['clean_acc']*100:.2f}% mean_len={report['clean_mean_len']:.1f} hit_max={report['clean_hit_frac']*100:.1f}%",
        "",
    ]

    best_both = float("-inf")
    best_both_alpha = None
    any_finished_pos = False

    for akey, steered in sorted(steered_map.items(), key=lambda x: float(x[0])):
        ids = [pid for pid in clean if pid in steered]
        both = [
            pid
            for pid in ids
            if clean[pid]["length"] < mt and steered[pid]["length"] < mt
        ]
        overall_gain = (acc(steered, ids) - acc(clean, ids)) * 100.0
        both_gain = (acc(steered, both) - acc(clean, both)) * 100.0 if both else 0.0
        hit_clean = sum(1 for i in ids if clean[i]["length"] >= mt) / max(len(ids), 1)
        hit_steered = sum(1 for i in ids if steered[i]["length"] >= mt) / max(len(ids), 1)

        bins = []
        for name, lo, hi in BINS:
            b_ids = [i for i in ids if lo <= clean[i]["length"] < hi]
            if not b_ids:
                continue
            g = (acc(steered, b_ids) - acc(clean, b_ids)) * 100.0
            bins.append(
                {
                    "bin": name,
                    "n": len(b_ids),
                    "clean_acc": acc(clean, b_ids),
                    "steered_acc": acc(steered, b_ids),
                    "gain_pp": g,
                }
            )
            if hi <= mt and g > 0:
                any_finished_pos = True

        row = {
            "alpha": float(akey),
            "overall_acc": acc(steered, ids),
            "overall_gain_pp": overall_gain,
            "mean_len": sum(steered[i]["length"] for i in ids) / max(len(ids), 1),
            "hit_frac": hit_steered,
            "both_finished_n": len(both),
            "both_finished_gain_pp": both_gain,
            "clean_hit_frac": hit_clean,
            "bins_by_base_length": bins,
        }
        report["alphas"][akey] = row
        if both_gain > best_both:
            best_both = both_gain
            best_both_alpha = float(akey)

        md += [
            f"## α={akey}",
            "",
            f"- overall {row['overall_acc']*100:.2f}% (gain {overall_gain:+.2f} pp), mean_len={row['mean_len']:.1f}, hit={hit_steered*100:.1f}%",
            f"- both-finished n={len(both)} gain={both_gain:+.2f} pp",
            "",
            "| bin (base len) | n | clean | steered | gain |",
            "|---|---:|---:|---:|---:|",
        ]
        for b in bins:
            md.append(
                f"| {b['bin']} | {b['n']} | {b['clean_acc']*100:.1f}% | "
                f"{b['steered_acc']*100:.1f}% | {b['gain_pp']:+.1f} |"
            )
        md.append("")

    # Branch decision (plan table)
    if best_both_alpha is None or (best_both <= 0 and not any_finished_pos):
        branch = (
            "truncation_cross_dataset: both-finished ≤0 and gains concentrated in hit-max; "
            "defer distill phase-2"
        )
        verdict = "截断结论跨数据集成立 → 蒸馏二期低优先"
    elif best_both >= 2.0 or any_finished_pos:
        branch = (
            "open_distill_phase2: both-finished ≥+2pp or finished bins steered>base"
        )
        verdict = "值得开二期：steered vs vanilla LoRA + 长度分箱"
    else:
        branch = "borderline: inspect α/layer; small positive both-finished but <2pp"
        verdict = "边界结果：可缩小 selection 网格或再扫 α，暂不默认蒸馏"

    report["best_both_finished_alpha"] = best_both_alpha
    report["best_both_finished_gain_pp"] = best_both if best_both_alpha is not None else None
    report["branch"] = branch
    report["verdict_zh"] = verdict

    md += [
        "## 分支判定（相对 GSM8K MVP 计划）",
        "",
        f"- best both-finished α={best_both_alpha}, gain={best_both:+.2f} pp",
        f"- **{verdict}**",
        f"- branch: `{branch}`",
        "",
    ]

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    Path(args.out_md).write_text("\n".join(md))
    print(json.dumps({"best_both_alpha": best_both_alpha, "best_both_gain": best_both, "branch": branch}, indent=2))


if __name__ == "__main__":
    main()
