# ALFWorld Agent Activation Steering 注入优化研究

> 对照 ssopd05 confirm 结果（decision_point α=−1.5，+1.68 pp）与文献，在 **valid_unseen** holdout 上消融「在哪注入 / 何时注入」。

## ssopd05 差距（动机）

| 维度 | ssopd05 主配置 | 文献常见做法 |
|------|----------------|--------------|
| 注入 token | prompt **物理末 token** only | **信息聚合 token**（assistant 末、决策边界） |
| decode 阶段 | `decision_point` **跳过** T=1 | CAA / EAST：**prompt 后所有 token** 持续加 v |
| 向量标签 | episode success vs fail | step 决策态（admissible、entropy） |
| 评测 | audit_confirm（已烧） | 本研究用 **valid_unseen** |

**假说**：ALFWorld 每步只生成一行 `ACTION:`（短 decode），当前协议只 steer「读 prompt」不 steer「写 action」，导致增益偏小。

## 文献表（与多轮 agent 直接相关）

| 工作 | 链接 | 提取 | 注入时机 / 位点 |
|------|------|------|----------------|
| 2504.19483 RepEng | [arxiv](https://arxiv.org/abs/2504.19483) | contrastive activations @ 推理末 | residual + α·c，中层 |
| CAA / ActAdd | [ACL'24](https://arxiv.org/html/2312.06681v3) | mean-diff on pairs | **prompt 后所有 token** |
| EAST | [arxiv](https://arxiv.org/html/2406.00244v2) | agent log，决策前 hidden，entropy 加权 | **每个生成 token** |
| FASB | [NeurIPS'25](https://papers.neurips.cc/paper_files/paper/2025/file/0c6c92a0c5237761168eafd4549f1584-Paper-Conference.pdf) | 分类器检测偏离 | **按需**注入 + backtrack |
| CAST | [arxiv](https://arxiv.org/pdf/2409.05907) | condition vector | **上下文匹配才**加 steer |
| KV Cache Steering | [arxiv](https://arxiv.org/pdf/2507.08799) | 聚合 token 的 K/V | prefill 后 **一次**，decode 经 attention 传播 |
| EasySteer | [arxiv](https://arxiv.org/abs/2509.25175) | 多种 | token/层/多向量（工程向） |
| exp06c（本仓库） | `ssopd_agent` | paired rollouts @ t* | frozen-prefix **单 token** |

## R0 预注册消融（valid_unseen）

| ID | `inject_style` | 说明 |
|----|----------------|------|
| R0a | `decision_point` | baseline：prefill 末 token，decode 不注入 |
| R0b | `prefill_decode` | prefill 末 + **每步 decode** 全注入 |
| R0c | `decode_only` | 仅 decode 注入 |
| R0d | `generation_wide` | prefill+decode 全序列 |
| R0e | `action_boundary` | prompt 内末个 `ACTION:` 标记 token + decode 注入 |

- α grid：`{-1.5, -1.0, 0, 1.0, 1.5}`
- Gate（相对 R0a）：Δsuccess ≥ **+3 pp**，bootstrap CI_lo > 0，放大比 > 1

## R1（仅当 R0 有赢家或作诊断）

- **R1a**：step-level admissible paired 重拟 v
- **R1b**：CAST-lite（上一步 inadmissible 才注入）

## 结果

完整 JSON：`reports/ssopd05_alfworld_unseen_ablation/results.json`

### R0（100 valid_unseen tasks，α∈{-1.5,-1.0,0,1.0,1.5}）

| ID | inject_style | α=0 | best | Δ vs α=0 |
|----|--------------|-----|------|----------|
| R0a | decision_point | 39.0% | **47.0%** @ α=-1.5 | **+8.0 pp** |
| R0b | prefill_decode | 39.0% | 45.0% @ α=-1.5 | +6.0 pp |
| R0c | decode_only | 39.0% | 41.0% @ α=-1.5 | +2.0 pp |
| R0d | generation_wide | 39.0% | 41.0% @ α=-1.5 | +2.0 pp |
| R0e | action_boundary | 39.0% | 37.0% @ α=1.5 | -2.0 pp |

- **预注册 Gate：FAIL**（R0b vs R0a α=0：Δ=+6 pp，但 bootstrap CI [-6, +18] pp 含 0；admissible 略降，放大比<0）
- **解读**：
  1. `decode_only` / `generation_wide` 仅 +2 pp → decode 注入有贡献但非主因
  2. **R0a（仅 prefill 末 token）反而最高 +8 pp** → 与「必须 steer action token」假说部分矛盾；可能 v 已编码在 context-read 表征
  3. `action_boundary` 无增益 → 聚合位点重定位未帮助
  4. 与 confirm +1.68 pp 分开：unseen holdout 上点估计更大但统计不显著

### R1

| 条件 | success | 备注 |
|------|---------|------|
| R1a step-admissible v（action_boundary 提取） | 43.0% | ‖v‖=2.49，cos(MATH)=+0.055 |
| R1b CAST-lite（inadmissible 才注入） | 29.0% | 低于 always-on 45.0% |
| winner always-on（R0b α=-1.5） | 45.0% | 对照 |

- CAST-lite 与 step-level 重拟均未改善；按需注入在此任务上伤害性能。

## B3 late-divergent 重拟 v

协议：对 paired success/fail，找首次动作分歧 `t_act`；因 `decision_points[t_act].prompt_text` 仍相同（前缀共享），实际提取位点为 **首次 prompt 分歧步** `t_prompt`（通常 `t_act+1`）。

| 条件 | success | 备注 |
|------|---------|------|
| α=0 | 39.0% | 共享基线（复用 R0a） |
| `v_episode` @ α=-1.5 | **47.0%** | episode mean-pool（现默认） |
| `v_late` @ α=-1.5 | 40.0% | ‖v‖=2.12，cos(episode)=0.133 |
| `v_late` @ α=-1.0 | 33.0% | 更差 |

- Gate vs `v_episode`@-1.5：**FAIL**（Δ=−7.0 pp，CI [-15, +1]）
- 分支：**Gate FAIL 且更差** → late step 噪声更大；保留 episode mean-pool；停止 ALFWorld 注入协议优化
- 产物：`reports/ssopd05_alfworld_b3_late/results.json`
