# Agent-SSOPD：SSOPD.pdf 的多轮 Agent 版

> 目标：把 PDF 的「探索→提向量→超我教师→OPD」迁到 ALFWorld；相对 NPM 多蒸馏闭环。  
> 载体：Qwen3-1.7B + coldstart LoRA；评测 valid_unseen 100；不烧 confirm358。

## 文献映射

| PDF 设定 | Agent 适配 | 来源 |
|----------|------------|------|
| Domain `E[H+]−E[H−]` | 检索合成 `v(q)`（inter+intra 记忆） | [NPM](https://arxiv.org/abs/2606.29824) |
| 盲目注入 | `decision_point` + ASA-lite error probe 门控 | [ASA](https://arxiv.org/abs/2602.04935) |
| Online reverse-KL | 先 offline SFT（A1 过门后） | MATH Q4 / PDF 工程折中 |

相关：RepE `2310.01405`、EAST、CAST、KV Cache Steering。

## 流水线

```
A0 memory (inter/intra Δh @ L14)
  → A1 Super-Self Teacher (retrieve v(q), optional gate)
  → [Gate A1: +5pp & CI_lo>0] → A2 distill LoRA
```

## A0 记忆库

- 输入：select rollouts 1432
- **inter**=185（prompt 首次分歧步）、**intra**=2699（admissible vs not）
- Probe（需 steer）：train_acc=0.956，n=4000
- 产物：`data/ssopd05_agent_ssopd/npm_memory.npz`

## A1 教师评测（valid_unseen）

**协议注**：NPM 条件用自定义逐步 runner；初期复用 R0 `collect_episodes` 基线会混淆。**公平对照**用同一 runner 重跑 base/static：

| 条件 | success | Δ vs custom base |
|------|---------|------------------|
| base_custom (α=0) | **29.0%** | — |
| static_custom α=-1.5 | 28.0% | −1.0 pp |
| npm 最优 (α=+1.0) | 29.0% | **0.0 pp** |
| npm_gated（各 α） | 29.0% | 0.0 pp（steered_steps≈0） |

（参考：R0 批量 runner 上 static α=-1.5 曾达 47%、base 39%——**runner 不同，不可混比**。）

- **Gate A1：FAIL**（Δ=0 pp，CI [-3,+3] pp）
- 解读：本设定下检索式 `v(q)` **未超过**同协议 coldstart；门控几乎不开（admissible 已高）。相对 NPM 论文：更小模型、无 textual workflow hybrid、episode 级检索（非逐步）。

## A2 蒸馏

**跳过**（预注册：A1 不过门禁止蒸馏）。标记：`reports/agent_ssopd_alfworld/a2_results.json`。

## 结论分支

对应计划分支 **3：A1 FAIL** → 小模型 + 纯隐式记忆不够；下一跳优先：

1. Hybrid：短 textual workflow + NPM steering（NPM 最强设定）
2. 更大 backbone（Qwen3-4B/8B）
3. WebShop（决策更密）

**不做**：再扫 ALFWorld inject_style / late-divergent。

## Hybrid（短 workflow + NPM）

从 select 569 条成功轨迹抽 6 类 task-type 模板（去实例号），prepend 到 user obs。同协议 custom runner：

| 条件 | success | Δ vs H0 |
|------|---------|---------|
| H0 base | 29.0% | — |
| H1 workflow-only | 26.0% | −3.0 pp |
| H2 workflow + NPM α=+1.0 | 29.0% | **0.0 pp** |

- H2−H1 = +3 pp，CI [-1, +8] 含 0（文本略伤、steering 只把成功率拉回基线）
- **Gate H：FAIL**（最优 Δ=0，CI [-10, +10]）
- 分支：1.7B ALFWorld 上纯文本 workflow 与 hybrid 都不够；停这条记忆线，下一跳更大 backbone 或 WebShop。不蒸馏。

产物：`data/ssopd05_agent_ssopd/workflows.json` · `reports/agent_ssopd_hybrid/results.json`

## UCE-lite（实例级 workflow 库，不注入）

H1 用 6 类固定骨架 FAIL。本轮按 UCE（2606.02304）改成 **每条成功轨迹一条**、去实例号、同 type + goal Jaccard 检索；进化只在 `audit_select` 80，eval 冻结。同 custom runner，对照 H0 **29%**。

| 条件 | success | Δ vs H0 | bootstrap CI |
|------|---------|---------|--------------|
| H0 base | 29.0% | — | — |
| B-ret（冻结检索） | 36.0% | +7.0 pp | [−4, +19] pp |
| B-uce（进化后再冻结） | **40.0%** | **+11.0 pp** | **[+1, +22] pp** |

- 进化：80 局，成功 33 / 失败 47，入库 +33、usage≤0 剪 41，库 569→561。
- **Gate B：PASS**（B-uce +11 pp，CI_lo=0.01>0）。B-ret 单独 CI 含 0。
- 无 NPM、无注入。本轮 todos 不含蒸馏，**未开 SFT**（过门后允许另开）。
- 产物：`data/ssopd05_agent_ssopd/uce_library.json` · `uce_library_evolved.json` · `reports/uce_alfworld/results.json`

## H3：UCE + 旧 NPM-lite

冻结进化 UCE 库，叠 A1 的 hidden-cosine `episode_v`（L14 一层、`decision_point`）。同 custom runner。

| 条件 | success | Δ vs B-uce | bootstrap CI |
|------|---------|------------|--------------|
| B-uce | 40.0% | — | — |
| H3 α=+1.0 | 35.0% | **−5.0 pp** | — |
| H3 α=−1.5（最优） | 40.0% | **+0.0 pp** | [−7, +7] pp |

- **Gate H3：FAIL**（需 vs B-uce +5 pp 且 CI_lo>0）。
- 教师预注册为 UCE-only（`teacher_alpha=0`）；**未蒸馏**。失败原因按当时协议记为「隐式检索合成不对」，不是「有了 UCE 就不需要 NPM」。
- 产物：`reports/uce_npm_hybrid/results.json`

## NPM 忠实协议（N0–N2）

H3 失败后按原文（2606.29824）对齐三处，**不再扫** `inject_style`：

1. 检索：任务 goal 文本 Jaccard（与 UCE 同一套 tokenize），不用当前 hidden 余弦。
2. 合成：召回 top-K=8 任务的 `(h+,h−)` 中心化后取第一主成分 → 每层一条 `v_l(q)`。
3. 注入：Qwen3-1.7B 对齐相对深度挂 **L13–15**；`prefill_decode`（prefill 末 + 每个 decode token）；α∈`{0.5,1.0,1.5}`，第一步 KL(P‖P_α)≤0.2 取最大合法值。

**N0**：select 1432 → inter=185、intra=815、n_tasks=307、layers={13,14,15}；wall≈227s。  
产物：`data/ssopd05_agent_ssopd/npm_faithful.npz`

同 runner，对照仍是 H0 29% / B-uce 40%。评测 mean_α=**1.5**（第一步 KL≈0，离散集全过）。

| 条件 | success | Δ | bootstrap CI |
|------|---------|---|--------------|
| H0 | 29.0% | — | — |
| N1 忠实 NPM（无 UCE） | **31.0%** | vs H0 **+2.0 pp** | [−3, +8] pp |
| B-uce | 40.0% | vs H0 +11.0 pp | — |
| N2 忠实 NPM + 冻结 UCE | **35.0%** | vs B-uce **−5.0 pp** | [−11, +1] pp |

- **Gate N：FAIL**（N1 未到 +5 且 CI 含 0；N2 为负）。
- 解读：对齐检索 / PCA / 每 token / 三层后，1.7B 上隐式 `v(q)` 相对 H0 仅点估计 +2，叠 UCE 仍伤 B-uce。KL≈0 说明第一步分布几乎不动。
- **不蒸馏**。记「1.7B 原文协议仍无加性」；**下一跳才是 4B**，本轮不找 4B。
- 产物：`scripts/npm_faithful_*.py` · `configs/experiment_npm_faithful.yaml` · `reports/npm_faithful/results.json`

## NPM 忠实协议 · Qwen3-8B

本机无 Qwen3-4B，改用原文也评过的 **Qwen3-8B**。无 ALFWorld LoRA（1.7B adapter 不能挂）。层对齐原文：36 层 → **17–19**。pair 仍来自 select 1432，hidden 在 8B 上重提。H0 / B-uce **在 8B 上重跑**，不复用 1.7B 的 29%/40%。UCE 库冻结 `uce_library_evolved.json`。

**N0**：inter=185、intra=815、n_tasks=307、layers={17,18,19}；wall≈376s。  
产物：`data/ssopd05_agent_ssopd/npm_faithful_8b.npz`

| 条件 | success | Δ | bootstrap CI |
|------|---------|---|--------------|
| H0（8B few-shot） | 15.0% | — | — |
| N1 忠实 NPM | **15.0%** | vs H0 **+0.0 pp** | [−4, +4] pp |
| B-uce | 26.0% | vs H0 +11.0 pp | — |
| N2 忠实 NPM+UCE | **25.0%** | vs B-uce **−1.0 pp** | [−5, +4] pp |

- 评测 mean_α=**1.5**（第一步 KL≈0）。
- **Gate N（8B）：FAIL**（N1 +0；N2 −1；CI 均含 0）。
- 解读：换 8B 后隐式协议仍无加性；UCE 文本在 8B 上仍单独 +11 pp。H0=15% 低于 1.7B+coldstart，缺任务 SFT。
- **不蒸馏**。不在本轮再扫 inject_style。
- 产物：`configs/experiment_npm_faithful_8b.yaml` · `reports/npm_faithful_8b/results.json`

## H4：UCE Counterfactual Action-Logit Steering（预注册，教师评测）

命题：冻结 1.7B+coldstart 在同一 \(s_t\) 上跑无 UCE / 有 UCE 两支，用 log-prob ratio 放大 UCE 相对无记忆的策略差，KL 校准后只从教师分布采一个 token。不扫 hidden / α / inject_style。蒸馏仅当 Gate H4 PASS。

**公式**（\(T_{\mathrm{KL}}=1\)）

\[
\ell^0=\log\mathrm{softmax}(z^0),\quad
\ell^+=\log\mathrm{softmax}(z^+),\quad
r=\ell^+-\ell^0,\quad
g_\beta=\ell^++\beta r,\quad
p_T=\mathrm{softmax}(g_\beta)
\]

即 \(p_T\propto p_{\mathrm{UCE}}^{1+\beta}p_0^{-\beta}\)。\(\beta=0\) 严格退化为 UCE。零方向看 \(K_{\max}=\mathrm{KL}(p_T(\beta_{\max})\|p_{\mathrm{UCE}})\)，不用 \(\|z^+-z^0\|\)。`ACTION:` / 冒号 / 空格 / EOS 用 \(p_{\mathrm{UCE}}\)；从第一个 command-content token 起校准。局部 `torch.Generator`，逐步 seed 与 B-uce 相同，不每 token 重置全局 RNG。

**条件**：`delta002`（δ=0.02）、`delta005`（δ=0.05）；`beta_max=4`，二分 12 步。UCE 库冻结、retrieve 只读。逐题 workflow SHA256 必须与 B-uce `prompt_text` 一致，否则停止比较。

**Decoder 混杂**：先 10 局 `δ=0` 随机 vs HF `generate` 逐 token。能复现则主对照旧 B-uce 40%；否则主对照 **H4-control**（同一手写 decoder，`target_kl=0`），旧 40% 只作旁证。

**Gate H4**（两 δ 取最优 + Bonferroni）：最佳条件相对 **主对照** Δsuccess ≥ +5 pp，且该条件 **97.5% CI_lo>0**（n_boot=10000）。95% CI 只报告。UCE 的 29→40 不算 H4 增益。H0 29% 辅助。FAIL 不调参、不蒸馏。

**Gate D**（本轮不执行）：H4-distill vs H0 ≥ +2 且 CI_lo>0；≥ Vanilla-LoRA；**> UCE-distill** 才能把增量归于 logit steering。

产物（拟）：`scripts/uce_counterfactual_logit*.py` · `configs/experiment_uce_counterfactual_logit.yaml` · `reports/uce_counterfactual_logit/`

## H4 教师评测结果

Decoder 随机等价 **未过**（10 局 137/344 step 逐 token 匹配）→ 主对照 **H4-control**（同一手写 decoder，δ=0），不是 HF 的 B-uce。Workflow hash 100/100 与 B-uce `prompt_text` 一致；retrieve 只读。

| 条件 | success | vs H4-control | 95% CI | Bonferroni 97.5% CI |
|------|---------|---------------|--------|---------------------|
| H0 | 29% | — | — | — |
| B-uce（HF generate，旁证） | 40% | — | — | — |
| H4-control δ=0 | **38%** | — | — | — |
| δ=0.02 | **33%** | **−5 pp** | [−14, +3] | [−15, +5] |
| δ=0.05（最优） | **37%** | **−1 pp** | [−10, +8] | [−11, +10] |

δ=0.05 诊断：command_token_kl=0.020（目标 0.05；unreachable 61%）；first_action_kl_vs_uce=0.001；command top-1 flip vs UCE=2.9%；invalid/ep 3.68→5.2；admissible 0.880→0.841。β mean≈1.40。

- **Gate H4：FAIL**（最优 −1 pp，CI_lo<0）。fail_kind=`invalid_or_entropy_worse`（invalid 上升且成功率未升）。也符合「改了部分 command token 但没提高成功率」；KL 经常打不到目标。
- **不蒸馏、不调参。** UCE 文本 29→40 仍只算 UCE，不算 logit steering。
- 产物：`reports/uce_counterfactual_logit/H4_summary.json`
- 入账后 CUDA generator 修补使 parity 到 340/344，**未重跑 100 题**，数字仍以本表为准。

**正式关闭 H4。** 不是「强度不够」：UCE 收益不能通过逐 token 放大 \(p_{\mathrm{UCE}}/p_0\) 稳定增强；该方向主要改变后续词法和格式，没有对第一项关键动作形成有效控制。δ=0.02 已扰动但 −5 pp；δ=0.05 unreachable 61%；第一 command KL=0.001、flip 2.9%；invalid 上升；两强度无正向趋势。**禁止**增大 \(\beta_{\max}\)、改 token KL、扩样本。

## H5 离线诊断（预注册，2026-09-03）

不是性能 Gate。数据：冻结 [`H4_control.jsonl`](/home/yiyangba/ssopd_paper_archive/reports/uce_counterfactual_logit/H4_control.jsonl) 的 `prefix_text` + `state.valid_tools`。不再 rollout。

同一 chat template 下对每个 \(a\in\mathcal A(s)\) teacher-force `ACTION: {a}`。\(S_c\) 只对 command-content token 长度归一化。\(\tau_A=1.0\)。\(q_c=\mathrm{softmax}(S_c/\tau_A)\)。\(\beta_{\max}=4\) 只用来问有多少状态能达到 \(\delta_A\in\{0.02,0.05\}\)，不采样。

**信号门（二者都要）：** whole-action top-1 不同 ≥ 10%；\(\mathrm{median}\,D_{\mathrm{JS}}(q_+,q_0)>0.01\)。

FAIL → 结束全部 ALFWorld steering。PASS → 停下请示，不自动开 H5 评测。

若日后评测：主对照是同一 scoring 的 \(q_+\)（H5-control），不是 B-uce / H4-control。

## H5 离线诊断结果（信号门 FAIL）

数据：H4-control 100 局、**3067** 个决策点。\(\tau_A=1.0\)，command-content = 冒号之后（Qwen 把空格并进动词 token）。

| 指标 | 值 | 门 |
|------|----|----|
| whole-action top-1 不同 | **22.0%** | ≥10% PASS |
| median \(D_{\mathrm{JS}}(q_+,q_0)\) | **0.0033** | >0.01 **FAIL** |
| mean JS | 0.0055 | — |
| 第一 command token top-1 不同 | 11.6% | — |
| token 不变但完整动作变 | 16.3% | — |
| 已执行动作 rank change 中位数 | 0 | — |
| \(\beta_{\max}=4\) 可达 δ=0.02 / 0.05 | 99.8% / 93.6% | 只报告 |

早期步 0–2：token top-1 仅 2.0%，完整动作 top-1 28%——H4 确实漏掉共享动词前缀后的动作差。但分布几乎重合（median JS≪0.01），top-1 翻转是近乎相同的 \(q\) 上的薄边，不是可稳定放大的局部策略差。

- **信号门：FAIL**（须两项都过）。不跑 H5-control / steer episode。
- **正式结束全部 ALFWorld steering。** 三级否证齐：hidden（NPM）/ token（H4）/ whole-action（H5 离线）。
- 产物：`reports/uce_whole_action/H5_offline_summary.json`

## 收口（论文主张，2026-09-03）

完整条文：[`notes/agent_ssopd_claim.md`](/home/yiyangba/ssopd_paper_archive/notes/agent_ssopd_claim.md)

- **可写**：实例级 UCE procedural memory，custom runner 上 29%→40%（+11 pp，CI_lo>0）。
- **不可写**：NPM / H4 / H5 steering 作为教师增益；也没有无外挂蒸馏学生。
- **停**：全部 ALFWorld steering。H5 评测不开。
- **新主线**：固定 UCE 的 on-policy 内化，见 [`notes/uce_opd_plan.md`](/home/yiyangba/ssopd_paper_archive/notes/uce_opd_plan.md)。

## 工具开关 steering（BFCL，Qwen3-1.7B 无 LoRA）

新 `split_seed=2026`：select/held-out 各 40 multiple + 40 irrelevance。不复用旧 exp_bfcl confirm。不与 R1-1.5B 旧账混比。

- A0：relevance `parse_ok=1.0`（未触 kill switch）；called=64 / not=16；`v_tool` 单位化。
- Held-out α=0：F1=0.784，call_rel=1.00，call_irr=0.55，tool_correct=0.95。
- 最优 α=−1.5：F1=0.792（**ΔF1=+0.008**），call_irr 0.55→0.525；tool_correct 不变；Δlogp=0（decision_point 下 prefix 分数未动）。
- **Gate A：FAIL**（需 ΔF1≥0.10 且 CI_lo>0；实得 CI [0.00, 0.026]）。不蒸馏。
- 解读：1.7B 在本提示下 relevance 已饱和（必调用），开关几乎只剩 irrelevance 少调 1 例；不是 2608.25198 那种 0→90% 调用率。
- 产物：`data/bfcl_propensity/` · `reports/bfcl_propensity/results.json`

## 产物

- `scripts/agent_ssopd_*.py`
- `scripts/uce_*.py` · `scripts/bfcl_propensity*.py` · `scripts/npm_faithful_*.py`
- `configs/experiment_agent_ssopd_alfworld.yaml`
- `configs/experiment_uce_alfworld.yaml` · `configs/experiment_bfcl_propensity.yaml`
- `configs/experiment_uce_counterfactual_logit.yaml`
- `configs/experiment_npm_faithful.yaml` · `configs/experiment_npm_faithful_8b.yaml`
- `reports/agent_ssopd_alfworld/`
- `reports/uce_alfworld/` · `reports/uce_npm_hybrid/` · `reports/bfcl_propensity/`
- `reports/uce_counterfactual_logit/`
- `reports/npm_faithful/` · `reports/npm_faithful_8b/`
- `notes/agent_ssopd_claim.md`
- `configs/experiment_uce_whole_action.yaml`
- `reports/uce_whole_action/`
