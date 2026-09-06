# Self-Amortized Procedural Memory（预注册）

Steering 线冻结。本线只验证：固定 UCE 教师的 action-level on-policy distillation 能否在无 UCE 的学生上留下增益。过门才做记忆—参数交替。

协议：Qwen3-1.7B + 合并后的 coldstart \(\theta_0\)；蒸馏 120 = `audit_select` 去掉进化 80 后的前 120；评测 valid_unseen 100；不烧 confirm358。不扫 KL / LoRA / lr / T。

## 假设

同一 \(\theta_0\) 在 UCE procedural memory 下是更强教师；在学生实际访问的状态上做 reverse-KL，学生不使用 UCE 也能保留部分教师增益。

## P0 来源

见 `reports/uce_opd/provenance.json`。P0：**isolation_ok**；措辞 **self-generated procedural memory**（同模型轨迹 + 规则抽象，非 LLM 改写）。种子库 runner 是 `collect_episodes`，不可与 custom 29%/40% 混比。蒸馏 120 中有 **84** 题的自身历史成功已在种子库（不重编冻结库）。retrieve 只读通过。

## D0 教师 / 学生

- \(\theta_0\)：Qwen3-1.7B + coldstart merge。教师冻结，每局检索一次 evolved 库，逐步 prepend，无 steering。
- 学生：\(\theta_0\) + 新 distill LoRA。Rollout 无 UCE、HF `generate`、H0 prompt。每 epoch 重采。成功和失败都进 OPD。

## Loss

只监督 command-content + 结束 EOS/换行。先对 token 均，再对 step / episode 均。full-vocab reverse-KL。`lr=1e-4`，`epochs=2`，`lora_r=16`，`lora_alpha=32`，`T=1`，`grad_clip=1`。

## 对照

H0（复用 29%）· Vanilla-LoRA（\(\theta_0\) 无 UCE 成功 SFT）· No-PI OPD（教师无 UCE）· UCE-SFT（UCE 成功轨迹 SFT，不得叫 OPD）· UCE-OPD · B-uce（复用 40%，只作 \(R_{\mathrm{int}}\)）。

## Gate D0（同时）

\(\Delta_{\mathrm{H0}}\ge 2\) pp 且 paired \(CI_{\mathrm{lo}}>0\)（n_boot=10000）；\(\ge\) Vanilla-LoRA；\(>\) No-PI OPD。vs UCE-SFT 只作文。FAIL 不调参、不进 D1、不宣称 memory 可内化。

## D1（仅 D0 PASS）

\(M_1\)：无记忆 \(\theta_1\) 在同一 80 进化题上 rollout，规则建库。教师仍 \(\theta_0\)，只换 \(M\)。对照：Fixed-memory \(M_0\)、Re-evolved \(M_1\)、Continued Vanilla。Gate：\(S_{\theta_2,M_1}-S_{\theta_1}\ge 2\) pp 且 \(CI_{\mathrm{lo}}>0\)，且 \(>S_{\theta_2,M_0}\)。

## D0 结果

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
- vs No-PI：+6.0 pp，CI [−3, +15]（点估计过，CI 含 0）
- vs UCE-SFT：−3.0 pp（只作文）
- \(R_{\mathrm{int}}=0.45\)
- **Gate D0：FAIL**（三条同时：CI_lo>0 未过；未 ≥ Vanilla；点估计虽 > No-PI 但门已破）。不调 KL/lr/epoch/rank，**不进 D1**，不宣称 memory 可内化。记为 UCE 与 parameterization 的机制负结果：成功轨迹 SFT 远强于 on-policy reverse-KL。
- 产物：`reports/uce_opd/summary.json` · `paired_bootstrap.json` · `eval_*.jsonl`
