# Steering, Truncation, and Distillation on MATH (Draft)

**Status:** internal draft — **ICLR “efficiency ≠ shortening” claim REJECTED by L1–L4**  
**Model:** Qwen3-1.7B · MATH-lighteval · splits 1200/400/400 · mt=2048  
**Artifacts:** `/home/yiyangba/ssopd_paper_archive/`

> **Claim policy (updated 2026-08-29 evening).** Accuracy gains on the **2048 hit-max bin count as real gains**. Primary distill gate is **confirm400 overall**. Next line: Qwen3-8B vanilla teacher → 1.7B LoRA (vs existing 1.7B steered/vanilla teachers).

See [`ICLR_GATE_DECISION.md`](ICLR_GATE_DECISION.md) for the full gate table.

---

## Headline numbers (confirm400)

| condition | gain_pp | mean_len | both-finished vs base |
|-----------|--------:|---------:|----------------------:|
| steered LoRA seed=42 | +12.50 | 1492 | −3.0 |
| vanilla LoRA 743 | +7.75 | 1727 | 0.0 |
| steered L4 lenmatch teachers | +13.75 | 1551 | −3.0 |
| vanilla L4 lenmatch teachers | +3.50 | 1720 | +0.6 |

Base-length bins: finished bins s−b ≤ 0; **[2048,∞) s−b ≈ +24 pp**.

---

## Background (still valid)

- Protocol: L7 unit FAIL → L14 `α·v_cap` overall PASS under short decode.  
- Truncation ladder: +24.75 → +8.5 → −2.0; trunc-gate α sweep FAIL.  
- DirScale: usable ≥200, saturated ≥400.  
- **GSM8K MVP（换集）**：confirm200 clean 83%；α=0.5 overall +4.0、both-finished −1.17；α≥1.5 崩。hit-max 箱 +35.7 pp。截断主导 **跨数据集成立**。见 [`../notes/truncation_methodology.md`](../notes/truncation_methodology.md)。

---

## Distillation phenomenology

Steered teachers beat vanilla on **overall** holdout (+4.75 pp at matched seed; +10 pp on L4 lenmatch teacher sets), but the **signature matches truncation rescue**, not length-controlled efficiency.

---

## Venue

Workshop / technical report on truncation evaluation + distill phenomenology.  
**Do not** submit to ICLR under the “efficiency not shortening” main claim without a new, successful length-controlled experiment line.
