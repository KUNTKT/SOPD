# Truncation-gate α sweep — 完成（2026-08-28）

Confirm400, full2k L14 `v_cap`, mt=2048.

| α | overall | both-finished | trunc-gate |
|--:|--------:|--------------:|:----------:|
| 0.5 | +5.00 | **+1.96** | FAIL |
| 1.0 | +8.50 | 0.00 | FAIL |
| 1.5 | **+11.00** | −2.40 | FAIL |
| 2.0 | +8.50 | −5.36 | FAIL |
| 2.5 | −7.50 | −23.35 | FAIL |

- Best overall: α=1.5 (+11.00)
- Best both-finished: α=0.5 (+1.96)，差 0.04 pp 过 +2 门
- **truncation_gate_pass_ge_2pp = False**

产物：`/tmp/ssopd03_full2k_L14_truncation_gate_alpha/`
