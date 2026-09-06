# Truncation 方法论（MATH + GSM8K）

更新：2026-08-29  
协议：Qwen3-1.7B · L14 `task_balanced_paired` `v_cap` · `contrastive_cap` · mt=2048

## 主张

Test-time steering（以及随后的 steered-teacher 蒸馏）的 **overall 增益可以真实存在**，但默认应视为 **截断缓解**，除非 **both-finished**（双方 length &lt; mt）和 **按 base 长度分箱的 finished 箱** 也给出正增益。

主门控：**both-finished gain ≥ +2 pp**，不用 overall 单独过关。

## 跨数据集证据

| | MATH confirm400 | GSM8K confirm200 |
|--|----------------:|-----------------:|
| clean acc | ~50% | 83% |
| clean hit-max | ~50%+ | 14% |
| 最佳 overall | +11.0 @α=1.5 | +4.0 @α=0.5 |
| 最佳 both-finished | +1.96 @α=0.5（未过门） | −1.17 @α=0.5 |
| 增益是否集中 hit-max | 是（~+22 pp） | 是（+35.7 pp，n=28） |
| 强 α | 仍可有正 overall | α≥1.5 **崩坏** |

GSM8K 轨迹更短、撞上限更少，**截断签名仍然在**：唯一可用的弱 α 上，overall +4 几乎全部来自 base hit-max 箱（28 题里 2→12 对）。

## 拟合注意（GSM8K）

`task_balanced_paired` 需要同题正负轨迹。GSM8K 正确率高（rollouts ~86%）→ vector_fit 600 题里可配对远少于 600（实测 **98**），方向 ‖v‖ 也更大（~57 vs MATH ~23）。因此 **同一绝对 α 在 GSM8K 上更猛**，不能直接抄 MATH 的 α=1.5/2.0。

## 不写进论文的句子

- 「steering / 蒸馏提升了未截断路径上的推理效率」
- 「换到更短的 GSM8K 后截断伪效应消失」

## 可写

- 评测必须同时报 overall、hit-max 分箱、both-finished
- DirScale：N=200 已可用，N=800 仅再 +1.3 pp overall，both-finished 仍负
- 蒸馏 overall steered>vanilla，但机制与注入同构

产物：`ALL_RESULTS.md` · `gsm8k/reports/gsm8k_L1_L2_verdict.md` · `reports/iclr_L1_L2_verdict.md`
