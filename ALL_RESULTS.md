# SSOPD 实验结果总汇

更新：2026-09-04  
模型：Qwen3-1.7B · 协议默认 `max_new_tokens=2048` · 方向：L14 `task_balanced_paired` `v_cap` · 注入：`contrastive_cap`  
归档根目录：`/home/yiyangba/ssopd_paper_archive/`

---

## 0. 总结论（先读这个）

| 主张 | 判定 |
|------|------|
| Test-time 注入在 MATH 上 overall 有增益 | thinking-on：**+5~11 pp**；no-think 重拟后 **α=−1.5 → +3.00** |
| 关 thinking + **旧 thinking-on v** | 正 α 全负；负 α 最佳仅 **+0.50**（未过 +2） |
| 关 thinking + **同分布重拟 v**（对齐 2504.19483） | **PASS**：α=−1.5 overall **+3.00**，both-finished **+4.12** |
| **2048 hit-max 箱上的准确率提升** | thinking-on 仍算有效果；no-think 成功点 **不是** hit-max（both 也过门） |
| Steered-teacher 蒸馏 overall > vanilla | thinking-on：**成立**（约 +4~10 pp）；**no-think：学生相对 base 未过 +2**（steered +0.00，vanilla −1.25） |
| Qwen3-8B vanilla 教师 → 1.7B LoRA | **FAIL −7.5 pp**（教师 38% / 74% 撞 2048；从未关 thinking） |
| ALFWorld agent：UCE 文本 workflow | **PASS** custom runner 29%→**40%**（+11 pp，CI [+1,+22]） |
| ALFWorld agent：NPM / 忠实 NPM / H4 logit / H5 动作分布 | **FAIL**（steering 三层否证；未蒸馏） |
| ALFWorld agent：UCE-OPD 内化 | **FAIL** 34%（ΔH0 +5，CI 含 0；vs Vanilla −16 pp）。不进 D1 |

主门槛：**confirm400 overall**；仍报长度分箱；hit-max 增益在 thinking-on 协议下计入成功。  
Agent 收口：[`notes/agent_ssopd_claim.md`](notes/agent_ssopd_claim.md)。  
Steering 冻结。UCE-OPD Gate D0 **FAIL**，不进 D1：[`notes/uce_opd_plan.md`](notes/uce_opd_plan.md)。

---

## 1. MATH — Test-time 注入

### 1.1 DirScale：拟合方向用多少题（N-ablation）

**设计**：用 N 道题的 paired 轨迹拟合 L14 `v_cap`，再在 **全 2000 题** 上同一协议注入（α=2，mt=2048）。N = 拟合规模，不是评测规模。

| N（拟合） | ‖v‖ | all2000 clean→steered | all2000 gain | bootstrap 95% CI | confirm400 gain | both-finished |
|----------:|----:|----------------------:|-------------:|:-----------------|----------------:|--------------:|
| **200** | 22.08 | 49.30%→59.45% | **+10.15** | [8.45, 11.90] | **+10.25** | **−2.62** (n=800) |
| **800** | 23.87 | 49.30%→60.80% | **+11.50** | [9.60, 13.30] | +8.25 | **−4.20** (n=809) |

N200 按 split 分层（all2000 上）：

| split | n | clean | steered | gain |
|-------|--:|------:|--------:|-----:|
| vector_fit | 1200 | 49.08% | 59.58% | +10.50 |
| selection | 400 | 48.50% | 57.50% | +9.00 |
| confirm | 400 | 50.75% | 61.00% | +10.25 |

**结论**

1. N200 已可用；N800 全量 overall 仅再高约 **+1.3 pp**（接近饱和）。  
2. confirm holdout 上 N200 甚至略高于 N800（+10.25 vs +8.25）——多数据不保证 holdout 更大。  
3. 两者 both-finished 皆负 → **加大 N 不改变「截断主导」**。  
4. 草稿口径：usable ≥200，大致饱和 ≥400。

产物：`reports/ssopd03_dirscale_N{200,800}_all2k/` · `notes/dirscale_all2k_comparison.md`

### 1.2 Truncation-gate α 扫（confirm400，full2k L14）

| α | overall | both-finished | trunc-gate（≥+2 both） |
|--:|--------:|--------------:|:----------------------:|
| 0.5 | +5.00 | **+1.96** | FAIL（差 0.04） |
| 1.0 | +8.50 | 0.00 | FAIL |
| 1.5 | **+11.00** | −2.40 | FAIL |
| 2.0 | +8.50 | −5.36 | FAIL |
| 2.5 | −7.50 | −23.35 | FAIL |

- Best overall：α=1.5  
- Best both-finished：α=0.5  
- **truncation_gate_pass = False**

产物：`reports/ssopd03_full2k_L14_truncation_gate_alpha/` · `notes/truncation_gate_result.md`

### 1.3 关 thinking 重跑注入（confirm400，2026-08-31）

同一 L14 `v_cap`、同一 α 网格；`enable_thinking=False`（prompt 预填空的 `<think></think>`）。**方向未重拟合**（仍是 thinking-on 轨迹上的 v）。clean + steered 都在 no-think 下评。

| α | overall | both-finished | mean_len |
|--:|--------:|--------------:|---------:|
| 0.0 clean | **68.25%** | — | 757 |
| 0.5 | **−1.50** | **−2.51** | 695 |
| 1.0 | −5.00 | −6.63 | 635 |
| 1.5 | −12.00 | −13.65 | 585 |
| 2.0 | −26.50 | −30.00 | 469 |
| 2.5 | −44.75 | −48.57 | 435 |

对照 thinking-on trunc-gate（同 confirm400）：

| α | thinking-on overall | no-think overall |
|--:|--------------------:|-----------------:|
| 0.5 | +5.00 | −1.50 |
| 1.0 | +8.50 | −5.00 |
| 1.5 | **+11.00** | **−12.00** |
| 2.0 | +8.50 | −26.50 |
| 2.5 | −7.50 | −44.75 |

- no-think clean hit-max **9.0%**（thinking-on 约一半撞 2048）
- Best = **不注入**；α 越大越差
- 产物：`reports/ssopd03_math_L14_nothink_confirm/` · `notes/nothink_inject_result.md`

### 1.4 关 thinking 必须成功（对齐 2504.19483，2026-08-31）

论文：控制向量须从「正确解题时的典型状态」提取；GSM8K 成功点是 **负 α**。

**P0** 旧 v + 负 α（未重拟）：最佳 α=−0.5 **+0.50** / both +1.46，未过 +2。

**P1** no-think vf1200 × K=4：4800 轨迹，success 66.9%，**paired=331**。新 L14 `v_cap` ‖v‖=12.75，与 thinking-on v 余弦 0.678。

**P2** 新方向 confirm400（复用 clean 68.25%）：

| α | overall | both-finished | mean_len |
|--:|--------:|--------------:|---------:|
| −2.5 | −1.00 | −0.30 | 883 |
| −2.0 | −2.00 | −2.10 | 867 |
| **−1.5** | **+3.00** | **+4.12** | 825 |
| −1.0 | +0.50 | +0.29 | 808 |
| −0.5 | +1.25 | +0.57 | 781 |
| +0.5 | −0.50 | −0.57 | 735 |
| +1.0 | −2.00 | −3.12 | 705 |

- 门槛 overall ≥ +2：**PASS**（α=−1.5）
- both-finished 也 ≥ +2：**PASS**
- P3（PCA / last-token / prefill）未做

产物：`reports/ssopd03_math_L14_nothink_refit/` · `reports/ssopd03_math_L14_nothink_negalpha/` · `notes/nothink_inject_success.md` · `data/ssopd01_qwen3_1_7b_nothink_vf1200/` · `data/ssopd02_qwen3_1_7b_nothink/`

---

## 2. MATH — 蒸馏（Q4 offline LoRA SFT）

### 2.1 做法摘要

1. **Steered 教师**：vector_fit 1200 题，推理时注入 L14 `v_cap` α=1.5 → 正确率 61.9% → **743** 条 correct  
2. **Vanilla 教师**：同底座、**无注入**，正确轨迹作对照（主实验 743；另有 matched-633 / L4 长度配对 367）  
3. **学生**：底座 + LoRA（r=16，q/k/v/o），只对 completion 做 CE；训练时 **无** steering  
4. **评测**：confirm400 holdout，无 steering

脚本：`scripts/generate_steered_teacher_traj.py` · `scripts/q4_offline_lora_sft.py`

### 2.2 Overall（confirm400）

| 条件 | acc | vs base | mean_len | both-finished vs base |
|------|----:|--------:|---------:|----------------------:|
| base | 51.75% | — | 1716 | — |
| steered LoRA（原 post seed） | 63.25% | **+11.50** | 1517 | −2.35 |
| steered LoRA seed=42（L3） | — | **+12.50** | 1492 | −2.99 |
| vanilla LoRA 743 | 59.50% | +7.75 | 1727 | 0.00 |
| vanilla matched-633（L5） | — | +5.25 | 1756 | −0.68 |
| steered L4 lenmatch n=367 | — | **+13.75** | 1551 | −3.03 |
| vanilla L4 lenmatch n=367 | — | +3.50 | 1720 | +0.64 |
| **Qwen3-8B 教师 LoRA** | **44.25%** | **−7.50** | 2105 | — |

- steered−vanilla（同 seed42）≈ **+4.75 pp** overall  
- L4 长度配对后 steered−vanilla ≈ **+10.25 pp** overall（但仍赢在截断箱）

### 2.4 Qwen3-8B vanilla 教师（2026-08-29）

- 教师：无注入，MATH vf1200，mt=2048，A100 batch=24，checkpoint 续跑  
- 教师自身：acc **38.0%**（456/1200），hit_max **74.4%**，均长 1917  
- 学生 1.7B LoRA（456 正确轨迹，2 epoch，r=16）：confirm400 **44.25%** vs base 51.75%，**−7.50 pp**（CI [−11.3, −3.8]）  
- 门槛（overall ≥ vanilla +7.75）：**FAIL**；弱于 1.7B steered/vanilla 教师  
- 产物：`reports/ssopd04_teacher_qwen3_8b_vf1200/` · `reports/ssopd04_qwen3_8b_teacher_lora_confirm/`

### 2.5 No-think 蒸馏（2026-08-31）

同协议关 thinking；教师 α=−1.5 新 v；学生 confirm400 无注入；base 复用 68.25%。

| | 教师 acc（vf1200） | 学生 acc | vs base | both | mean_len |
|--|-----------------:|---------:|--------:|-----:|---------:|
| base | — | 68.25% | — | — | 757 |
| steered LoRA（822 正确） | 68.50% | **68.25%** | **+0.00** | −0.29 | 1100 |
| vanilla LoRA（795 正确，k=0） | 66.25% | 67.00% | **−1.25** | −1.75 | 1035 |

- 门槛 overall ≥ +2：**FAIL**
- steered−vanilla 学生 **+1.25 pp**（thinking-on 约 +4.75）
- 产物：`reports/ssopd04_nothink_*_lora_confirm/` · `notes/nothink_distill_result.md`

---

## 3. GSM8K — 最小复现包（换集检验截断）

工作目录：`gsm8k/`

| 步骤 | 结果 |
|------|------|
| D1 rollouts | 1000×8=8000，PASS，成功率 **85.7%** |
| D2 splits | 600 / 200 / 200 |
| D3 fit L14 | n_hidden=4796，**paired=98**/600（462 全对），‖v‖≈57.1 |
| D4 confirm200 α∈{0.5,1.5,2.0} | 见下表 |
| D5 L1/L2 | 截断跨集成立；蒸馏二期 **低优先** |

### 3.1 Confirm α 扫（clean acc=83.0%，hit_max=14%）

| α | overall | both-finished | 备注 |
|--:|--------:|--------------:|------|
| 0.5 | **+4.00** | **−1.17** | 唯一可用；hit-max 箱约 **+35.7** |
| 1.5 | −76.00 | −88.96 | 崩 |
| 2.0 | −83.00 | −95.20 | 崩 |

### 3.2 α=0.5 按 base 长度分箱

| bin | n | clean | steered | gain |
|-----|--:|------:|--------:|-----:|
| [0,512) | 36 | 97.2% | 100.0% | +2.8 |
| [512,1024) | 88 | 96.6% | 93.2% | −3.4 |
| [1024,1536) | 32 | 96.9% | 93.8% | −3.1 |
| [1536,2048) | 16 | 81.2% | 87.5% | +6.2 |
| **[2048,∞)** | **28** | 7.1% | 42.9% | **+35.7** |

分支：`truncation_cross_dataset` → 蒸馏二期低优先。  
报告：`gsm8k/reports/gsm8k_L1_L2_verdict.md`

---

## 4. MATH vs GSM8K（注入对照）

| | MATH confirm | GSM8K confirm |
|--|:------------:|:-------------:|
| clean acc | ~50% | ~83% |
| clean hit-max | 高（~50%+） | 低（14%） |
| 最佳 overall α | 1.5（+11） | 0.5（+4） |
| 最佳 both-finished | +1.96（仍 FAIL 门） | −1.17 |
| 强 α（≥1.5） | 仍可有正 overall | **崩坏** |
| 增益是否集中 hit-max | **是** | **是** |

---

## 5. 投稿 / 下一步（已定）

1. **不要**把「效率且非缩短」当 ICLR 主 claim；**不**为该 claim 开 E2 第二模型。  
2. 可写：**truncation 评测方法论** + DirScale overall 现象 + 蒸馏 overall 但机制同构。  
3. 档位：**workshop / 技术报告**。  
4. GSM8K：**不默认**开 steered vs vanilla LoRA 二期（除非另开「弱 α / 更多 paired」诊断）。

---

## 6. 关键产物索引

| 路径 | 内容 |
|------|------|
| `reports/ssopd03_dirscale_N200_all2k/` | MATH N200 注入 |
| `reports/ssopd03_dirscale_N800_all2k/` | MATH N800 注入 |
| `reports/ssopd03_full2k_L14_truncation_gate_alpha/` | MATH α 扫（thinking-on） |
| `reports/ssopd03_math_L14_nothink_confirm/` | MATH α 扫（thinking OFF，旧 v） |
| `reports/ssopd03_math_L14_nothink_negalpha/` | 旧 v + 负 α |
| `reports/ssopd03_math_L14_nothink_refit/` | no-think 重拟 v α 扫（PASS +3.00） |
| `data/ssopd02_qwen3_1_7b_nothink/` | no-think L14 v_cap |
| `notes/nothink_inject_result.md` | 关 thinking + 旧 v 失败 |
| `notes/nothink_inject_success.md` | 关 thinking 重拟成功 |
| `reports/ssopd04_teacher_nothink_L14_am1_5/` | no-think steered 教师 |
| `reports/ssopd04_nothink_steered_lora_confirm/` | no-think steered LoRA（+0.00） |
| `reports/ssopd04_nothink_vanilla_lora_confirm/` | no-think vanilla LoRA（−1.25） |
| `notes/nothink_distill_result.md` | no-think 蒸馏结论 |
| `reports/ssopd04_teacher_vf1200_L14_a1_5/` | steered 教师轨迹 |
| `reports/ssopd04_offline_lora_sft_confirm/` | steered LoRA |
| `reports/ssopd04_vanilla_lora_sft_confirm/` | vanilla LoRA 743 |
| `reports/ssopd04_steered_lora_reseed42/` | L3 |
| `reports/ssopd04_*_L4_lenmatch/` | L4 |
| `reports/ssopd04_vanilla_lora_matched633/` | L5 |
| `reports/iclr_L1_L2_verdict.md` | MATH 长度门控 |
| `paper/ICLR_GATE_DECISION.md` | ICLR 停 claim |
| `gsm8k/reports/ssopd03_*_confirm_alpha/` | GSM8K confirm |
| `gsm8k/reports/gsm8k_L1_L2_verdict.md` | GSM8K 分支 |
| `scripts/` | resume smoke / teacher / LoRA / GSM8K 管线 |

## ALFWorld 多轮注入（ssopd05）

- Coldstart LoRA 后 select success **39.73%**（P1），paired=187。
- L14 v_cap ‖v‖=13.35，cos(MATH-nothink)=-0.026。
- Confirm：style=`decision_point` α=-1.5，Δepisode **+1.68 pp**，Δadmissible **-0.49 pp**，amplification=-3.45271502168482。
- Phase A（80 task）探 α 见 +6.25 pp，但 **全量 confirm358** 仅 +1.68 pp，bootstrap CI [-0.034, 0.073] 含 0。不支持干净的 superlinear 放大；负 α 方向与 MATH no-think 一致。
- 产物：`reports/ssopd05_alfworld_confirm/` · `notes/alfworld_steering_result.md` · `data/ssopd05_alfworld/directions.npz`

## valid_unseen 注入位点消融（R0/R1）

- 评测池：valid_unseen，n=100 tasks
- R0 赢家：**R0b** (`prefill_decode`) α=-1.5
- vs R0a α=0：Δsuccess **+6.00 pp**
- Gate（+3pp, CI_lo>0, amp>1）：**FAIL**
  - Δsuccess=+6.00 pp，bootstrap CI [-6.00, 18.00] pp
  - 放大比=-25.868598538845472

| ID | inject_style | α=0 success | best success | Δ vs α=0 |
|----|--------------|-------------|--------------|----------|
| R0a | decision_point | 39.0% | 47.0% @ α=-1.5 | +8.00 pp |
| R0b | prefill_decode | 39.0% | 45.0% @ α=-1.5 | +6.00 pp |
| R0c | decode_only | 39.0% | 41.0% @ α=-1.5 | +2.00 pp |
| R0d | generation_wide | 39.0% | 41.0% @ α=-1.5 | +2.00 pp |
| R0e | action_boundary | 39.0% | 37.0% @ α=1.5 | -2.00 pp |

### R1

- R1a step-admissible v：success 43.0%
- R1b CAST-lite：success 29.0%
- 产物：`reports/ssopd05_alfworld_unseen_ablation/` · `notes/alfworld_steering_injection_research.md`

## B3 late-divergent 重拟 v（valid_unseen）

- 提取：success/fail 首次 **prompt 分歧步**（≈ t_act+1；动作分歧步 prompt 相同 → v≡0），n_pairs=185，‖v‖=2.12，cos(episode_v)=0.133
- 注入：`decision_point`；α∈{-1.5,-1.0,0}
- α=0：**39.0%**；`v_episode`@-1.5：**47.0%**；`v_late`@-1.5：**40.0%**；`v_late`@-1.0：**33.0%**
- Gate vs `v_episode`@-1.5：**FAIL**（Δ=−7.0 pp，bootstrap CI [-15, +1] pp）
- 结论：late-divergent 提取 **更差**；保留 episode mean-pool 为默认；**停止 ALFWorld 注入优化**
- 产物：`reports/ssopd05_alfworld_b3_late/` · `data/ssopd05_alfworld/directions_b3_late_divergent.npz`

## Agent-SSOPD（NPM-lite 教师 + 蒸馏门）

- 动机：SSOPD.pdf 闭环迁到 agent；用检索 `v(q)` 替代静态 CAA；过门再蒸馏。
- A0：inter=185 + intra=2699，probe acc=0.956 → `data/ssopd05_agent_ssopd/npm_memory.npz`
- A1（同协议 custom runner）：base **29%**；NPM 最优 **29%（+0 pp）**；static −1 pp；**Gate FAIL**
- A2：按预注册 **跳过蒸馏**
- Hybrid（同 runner）：H0 29%；H1 workflow-only **26%（−3 pp）**；H2 workflow+NPM **29%（+0 pp）**；**Gate FAIL**
- 结论（当时）：1.7B ALFWorld 上纯隐式与 6 类固定模板 hybrid 均未过门。

## UCE-lite（实例级 workflow，ALFWorld）

- 同 custom runner vs H0 **29%**。无注入、无 NPM。
- B-ret：36%（+7 pp），CI [−4, +19] pp 含 0
- B-uce：audit_select 80 进化后 **40%（+11 pp）**，CI **[+1, +22] pp**
- **Gate B：PASS**。本轮未蒸馏。
- 产物：`reports/uce_alfworld/` · `data/ssopd05_agent_ssopd/uce_library.json`

## H3：UCE + 旧 NPM-lite（custom runner）

- 冻结 `uce_library_evolved.json` + A0/A1 的 hidden-cosine `episode_v`；`decision_point`；L14 一层。对照仍是 H0 29% / B-uce 40%。
- α=+1.0：**35%**（vs B-uce **−5 pp**）
- α=−1.5：**40%**（vs B-uce **+0 pp**），bootstrap CI [−7, +7] pp
- **Gate H3：FAIL**。教师预注册为 UCE-only；**未蒸馏**。
- 产物：`reports/uce_npm_hybrid/results.json`

## NPM 忠实协议（任务检索 + PCA + 每 token + 三层）

- 对照文献 2606.29824：检索改 goal Jaccard（与 UCE 同一套 tokenize），合成改召回 pair 中心化第一主成分，注入改 L13–15 + `prefill_decode`，α 在 `{0.5,1.0,1.5}` 里用第一步 KL≤0.2 取最大合法值。不扫 inject_style。
- N0：select 1432 → inter=185 / intra=815 / n_tasks=307；`data/ssopd05_agent_ssopd/npm_faithful.npz`
- 评测时 mean_α=1.5（KL≈0，离散集全过，始终取最大）

| 条件 | success | Δ | bootstrap CI |
|------|---------|---|--------------|
| H0 | 29.0% | — | — |
| N1 忠实 NPM | **31.0%** | vs H0 **+2.0 pp** | [−3, +8] pp |
| B-uce | 40.0% | vs H0 +11.0 pp | — |
| N2 忠实 NPM+UCE | **35.0%** | vs B-uce **−5.0 pp** | [−11, +1] pp |

- **Gate N：FAIL**（需 N1 vs H0 或 N2 vs B-uce：+5 pp 且 CI_lo>0）。不蒸馏。
- 记：「1.7B 原文协议仍无加性」；下一跳才是 4B，本轮不找 4B。
- 产物：`reports/npm_faithful/results.json` · `configs/experiment_npm_faithful.yaml`

## NPM 忠实协议 · Qwen3-8B（无 coldstart LoRA）

- 本机无 Qwen3-4B；改用原文同款 **Qwen3-8B**（36 层，挂 **17–19**）。1.7B LoRA / H0 29% / B-uce 40% **不可混比**。同 custom runner，valid_unseen 100。pair 库用同一批 select 轨迹在 8B 上重提 hidden。
- N0：inter=185 / intra=815 / n_tasks=307；`data/ssopd05_agent_ssopd/npm_faithful_8b.npz`
- 评测 mean_α=1.5（KL≈0）

| 条件 | success | Δ | bootstrap CI |
|------|---------|---|--------------|
| H0（8B few-shot） | 15.0% | — | — |
| N1 忠实 NPM | **15.0%** | vs H0 **+0.0 pp** | [−4, +4] pp |
| B-uce（冻结 1.7B 进化库） | 26.0% | vs H0 +11.0 pp | — |
| N2 忠实 NPM+UCE | **25.0%** | vs B-uce **−1.0 pp** | [−5, +4] pp |

- **Gate N（8B）：FAIL**。不蒸馏。
- 解读：放大到 8B 后隐式 `v(q)` 仍无加性；UCE 文本检索单独仍 +11 pp。8B 无 ALFWorld SFT，H0 低于 1.7B+LoRA。
- 产物：`reports/npm_faithful_8b/results.json` · `configs/experiment_npm_faithful_8b.yaml`

## H4：UCE Counterfactual Action-Logit Steering（1.7B custom runner）

- 公式：log-prob ratio \(p_T\propto p_{\mathrm{UCE}}^{1+\beta}p_0^{-\beta}\)；command-content token 才 KL 校准；局部 Generator。Decoder 与 HF generate **不等价**（137/344），主对照 H4-control 38%，旧 B-uce 40% 旁证。workflow hash 100/100 对齐。

| 条件 | success | vs H4-control | 95% CI | 97.5% Bonferroni |
|------|---------|---------------|--------|------------------|
| H4-control δ=0 | 38% | — | — | — |
| δ=0.02 | 33% | −5 pp | [−14, +3] | [−15, +5] |
| δ=0.05 | 37% | −1 pp | [−10, +8] | [−11, +10] |

- 最优 δ=0.05：command KL=0.020（目标 0.05，unreachable 61%）；first-command KL=0.001；flip vs UCE 2.9%；invalid/ep 3.68→5.2。
- **Gate H4：FAIL。正式关闭，不挽救。** 不是强度不够：逐 token 放大 \(p_{\mathrm{UCE}}/p_0\) 主要改词法/格式，未控制第一项关键动作。禁止增大 \(\beta_{\max}\)、改 token KL、扩样本。
- 不蒸馏。UCE 29→40 不算本方法增益。数字 38/33/37 不改。
- 产物：`reports/uce_counterfactual_logit/H4_summary.json`
- Agent 线收口：成立的是 UCE 文本记忆；hidden / token / whole-action steering 均未过门。见 `notes/agent_ssopd_claim.md`。

## H5：UCE Counterfactual Whole-Action（离线诊断，非性能 Gate）

- 数据：H4-control 100 局 3067 步；同一 1.7B+LoRA teacher-force admissible 列表。不跑新 episode。
- whole-action top-1 不同 **22.0%**（≥10% PASS）；median JS **0.0033**（须 >0.01，**FAIL**）。
- 第一 command token top-1 不同 11.6%；token 不变但动作变 16.3%；早期步 token 差仅 2%、动作差 28%。
- 已执行动作 rank change 中位数 0。\(\beta_{\max}=4\) 下 93.6% 状态可达 action KL 0.05——能推 KL，但推的是近乎重合的分布。
- **信号门 FAIL。不评测、不蒸馏。正式结束全部 ALFWorld steering。**
- 产物：`reports/uce_whole_action/H5_offline_summary.json`

## BFCL 工具开关 steering（Qwen3-1.7B）

- 新 split seed=2026，40+40 multiple / irrelevance；无 ALFWorld LoRA。不与旧 R1-1.5B BFCL 账混比。
- A0 parse_ok(relevance)=1.0，未杀实验。`v_tool` 可拟合（64 called / 16 not）。
- 最优 α=−1.5：F1 0.784→0.792（**ΔF1=+0.008**），CI [0.00, 0.026]；call_irr 0.55→0.525；tool_correct=0.95 不变。
- **Gate A：FAIL**（需 +0.10）。不蒸馏。
- 产物：`reports/bfcl_propensity/` · `data/bfcl_propensity/`

## UCE-OPD（Self-Amortized Procedural Memory）

更新：2026-09-04。评测：valid_unseen 100，无记忆、HF generate、只加载蒸馏 LoRA。n_boot=10000。

| 条件 | success | vs H0 |
|------|---------|-------|
| H0（复用） | 29.0% | — |
| Vanilla-LoRA | **50.0%** | +21.0 pp |
| No-PI OPD | 28.0% | −1.0 pp |
| UCE-SFT（不得叫 OPD） | 37.0% | +8.0 pp |
| UCE-OPD | 34.0% | +5.0 pp |
| B-uce（复用，只作 \(R_{\mathrm{int}}\)） | 40.0% | +11.0 pp |

- vs H0：+5.0 pp，CI **[−3.0, +13.0] pp**（含 0）
- vs Vanilla：**−16.0 pp**，CI [−27, −5]
- vs No-PI：+6.0 pp，CI [−3, +15]
- vs UCE-SFT：−3.0 pp（只作文）
- \(R_{\mathrm{int}}=0.45\)
- **Gate D0：FAIL。** 不调 KL/lr/epoch/rank，不进 D1，不宣称 memory 可内化。
- 产物：`reports/uce_opd/summary.json` · `paired_bootstrap.json` · `eval_*.jsonl`

## TAME-OPD Gate T0

更新：2026-09-05。train-only manifest；valid_unseen 未打开。

- teach_pick: `None` · teacher_pick: `None`
- T0-A: `{'pass': False, 'reasons': ['not_identifiable']}`
- T0-B: `{'status': 'not_identifiable', 'pass': False, 'reasons': ['not_identifiable']}`
- method_gate_pass: `False`
- 主张：T0-B not identifiable. Method gate FAIL. Not a disproof of TAME.
- 产物：`reports/tame_opd/` · `notes/tame_opd_plan.md`
- 停止。不自动多轮训练。不把结果写成完整 TAME。

## TAME T1-V

更新：2026-09-05。60-entry JSON-patch mutation gate。未重跑 OPD。

- Gate V: `FAIL` reasons=`['per_seed', 'structured_validity']`
- 主张：Gate V FAIL. Stop ALFWorld TAME on Qwen3-1.7B. Do not retune leak thresholds.
- 产物：`reports/tame_t1v/` · `notes/tame_t1v_plan.md`
- 停止。不自动扩全库，不调泄漏阈值。
