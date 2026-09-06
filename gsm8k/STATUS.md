# GSM8K MVP 状态

更新：2026-08-29 16:45 PDT

## 一期（已完成）

| 步骤 | 状态 |
|------|------|
| D0 smoke | ✅ |
| D1 1000×8 | ✅ PASS，sr=85.7% |
| D2 splits | ✅ 600/200/200 |
| D3 L14 fit | ✅ paired=**98**/600（462 全对） |
| D4 α∈{0.5,1.5,2.0} | ✅ 仅 0.5 可用（+4 / both −1.17） |
| D5 L1/L2 | ✅ 截断跨集成立；蒸馏二期 **不开** |

## 续跑（进行中）

计划第三分支：强 α 崩 + paired 偏少 → 补 **弱 α**（不重跑 clean）。

- 覆盖诊断：`reports/gsm8k_paired_coverage.md`
- 方法论：`../notes/truncation_methodology.md`
- Confirm 弱 α∈{0.15, 0.25, 0.35}，复用 D4 clean
