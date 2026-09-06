# Agent-SSOPD 收口主张（2026-09-03）

评测协议：Qwen3-1.7B + coldstart LoRA，ALFWorld valid_unseen 100，**同一 custom runner**。不与 R0 批量 runner 的 39%/47%、不与 8B 无 LoRA 的 15% 混比。不烧 confirm358。过门才蒸馏；本轮蒸馏未开。

## 一句话

在本设定下，**站住的是冻结模型对自身成功轨迹的实例级 workflow 检索（UCE）**；hidden / token / whole-action 三层 steering **都没有过门**。UCE 提供的是不可通过局部线性外推继续增强的上下文条件能力。不能把「超我教师 → 无外挂学生」写成主结果。

## 可写进正文的主张

**成立（Gate B PASS）**

同源 agent 把历史成功执行抽象成去实例号的 procedural memory，按 task type + goal Jaccard 检索并 prepend。valid_unseen 上：

- H0（无 workflow、无注入）**29%**
- B-uce（进化后冻结库）**40%**，Δ=**+11 pp**，paired bootstrap CI **[+1, +22] pp**

8B 无 LoRA、同一冻结库：15%→26%，仍是 **+11 pp**（基线不同，只作跨模型重复，不与 29%/40% 混比）。

这是 **privileged 文本条件**，不是隐层向量，也不是 logit 外推。

**不成立（不得写成主创新）**

| 方法 | 对照 | 结果 | 门 |
|------|------|------|----|
| 静态 CAA / 扫 inject_style | confirm 或 R0 批量 runner | 点估计有时正，CI 含 0 或未过预注册 | 停扫 |
| NPM-lite（hidden 余弦 Δh） | custom H0 29% | +0 | A1 FAIL |
| 6 类固定模板 ± NPM | H0 29% | 最优 +0 | Gate H FAIL |
| UCE + 旧 NPM-lite | B-uce 40% | 最优 +0 | H3 FAIL |
| 忠实 NPM（Jaccard+PCA+每 token+三层）1.7B | H0 / B-uce | +2 / −5 | Gate N FAIL |
| 同上 8B | 8B H0 / 8B B-uce | +0 / −1 | Gate N FAIL |
| H4 UCE 反事实 logit steering | **H4-control 38%**（手写 decoder，δ=0） | 最优 −1 pp | Gate H4 FAIL |
| H5 完整动作分布（离线） | H4-control 前缀，非性能门 | top-1 22% 过；median JS 0.0033 不过 | 信号门 FAIL |
| BFCL 工具开关 | α=0 F1 | ΔF1=+0.008 | Gate A FAIL |

H4 正式账：HF generate 与手写 decoder 未逐 token 对齐（入账时 137/344），故主对照是 H4-control 而非 B-uce 40%。workflow 与 B-uce 逐题 hash 一致，差异不是检错库。δ=0.05：success 37%，command KL 0.020（目标 0.05，unreachable 61%），相对 control −1 pp，97.5% CI 含 0。

入账之后的 decoder RNG 修补只把 parity 提到 340/344，**没有重跑 100 题，不改 38/33/37**。

**H4 正式关闭（不挽救）。** 结果说明的不是「steering 强度不够」，而是：UCE 的收益不能通过逐 token 放大 \(p_{\mathrm{UCE}}/p_0\) 稳定增强；该方向主要改变后续词法和格式，没有对第一项关键动作形成有效控制。δ=0.02 已达有效扰动但 −5 pp；δ=0.05 有 61% token 达不到目标 KL；第一 command token KL=0.001、top-1 flip=2.9%；invalid 上升；两强度无正向趋势。**禁止**增大 \(\beta_{\max}\)、改 token KL 或扩样本去补扫。

## 禁止的表述

- 不得把 UCE 的 +11 pp 写成 NPM / H4 / H5 steering 的收益。
- 不得把 MATH 上的注入/蒸馏结果直接外推到 ALFWorld agent。
- 不得把 custom runner 29% 与 `collect_episodes` 39%/47% 并表。
- 不得声称已蒸馏出「无 UCE、无 steering」的超我学生：Gate A1/N/H3/H4 与 H5 信号门均 FAIL，蒸馏未跑。
- 若另开 UCE-only 蒸馏，只能写「privileged workflow 可模仿」，不能写 logit/hidden steering 被蒸进去。

## 机制一句

文本 workflow 改变的是 **决策时可见的计划**。当前状态下 \(q_+\) 与 \(q_0\) 几乎重合（median JS=0.0033）；即使完整动作 top-1 有 22% 不同，也只是薄边上的翻转，不是可放大的局部策略差。UCE 的 +11 pp 更可能来自早期随机分叉、整段上下文或非线性交互，而不是稳定的 hidden / token / action direction。

## 锁定

**ALFWorld steering 实验全部结束。** 不再扫 `inject_style`、NPM、H4、H5 评测。不蒸 steering 教师。

新主线：**Self-Amortized Procedural Memory**（固定 UCE → OPD 内化；过门才交替进化）。预注册：[`notes/uce_opd_plan.md`](/home/yiyangba/ssopd_paper_archive/notes/uce_opd_plan.md)。创新若成立，来自 UCE 内化，不是 steering。

<!-- UCE_OPD_D0 -->
**UCE-OPD Gate D0：FAIL。** 无记忆评测 valid_unseen 100：UCE-OPD **34%**，H0 29%（+5 pp，CI [−3, +13] 含 0），Vanilla-LoRA **50%**（−16 pp），No-PI 28%，UCE-SFT 37%。未过门；不调参、不进 D1、不宣称 memory 可内化。记为 UCE 与 parameterization 的机制负结果。
