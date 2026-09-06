# Truncation-gate 实验计划（2026-08-28）

## 基线（已有 α=2 all2k）

| | overall | both-finished | steered_only 来自 clean 撞上限 |
|---|--------:|--------------:|-------------------------------:|
| N200 | +10.15 | −2.62 (n=800) | 272/275 |
| N800 | +11.50 | −4.20 (n=809) | 307/311 |

结论：α=2 的整体增益几乎全是截断救援；写完题上净伤害。

## 本次实验

- Split: **confirm400**
- Direction: full2k L14 `v_cap`（‖v‖≈24.50）
- α grid: **0.5, 1.0, 1.5, 2.0, 2.5**
- mt=2048, batch=48
- 主门控：**both-finished gain ≥ +2 pp**（不是 overall）

## 输出

`/tmp/ssopd03_full2k_L14_truncation_gate_alpha/`
