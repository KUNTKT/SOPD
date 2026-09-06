# ALFWorld 多轮注入放大实验

协议：Qwen3-1.7B + few-shot ALFWorld，sft 池 planner 轨迹 coldstart LoRA，
`audit_select` 拟合 L14 `v_cap`，`audit_confirm` 一次性 α 扫。无蒸馏。

## P0 Base vs coldstart

- 裸跑（20 task × K=2）：success **2.50%**，
admissible 0.917 → GATE FAIL
- Qwen3 coldstart LoRA 后：success **35.00%**，
admissible 0.899 → GATE PASS
- adapter: `data/ssopd05_alfworld_coldstart/coldstart_lora`

## P1 Fit-pool rollouts

- pool=`audit_select`，n_tasks=358，K=4
- success **39.73%**，admissible 0.896
- paired / mixed groups: **187**

## P2 Direction

- n_paired_used=187
- ‖v_cap‖=13.349332809448242
- cosine vs MATH no-think v: -0.025903892279993245
- 提取：每条轨迹 4 个决策点 prompt 末 token mean-pool（step-0 同构会导致 v=0）

## P3 Confirm

- selected inject_style=`decision_point` α=-1.5
- Phase A（80 task）best gain: +6.25 pp（α=-1.5，**非 holdout 全量**）

### Phase B（full audit_confirm n=358）

| | success | admissible | mean_steps |
|--|--------:|-----------:|-----------:|
| α=0 | 41.90% | 0.912 | 29.0 |
| steered | 43.58% | 0.907 | 28.5 |
| Δ | +1.68 pp | -0.49 pp | — |

- amplification_ratio = Δepisode / Δadmissible = **-3.45271502168482**
- task-level bootstrap mean=0.01675977653631285 CI=[-0.0335195530726257, 0.07262569832402235] n=358
- confirm 显著性（CI_lo>0）: **False**

## Verdict

Phase A（80 task）探 α 见 +6.25 pp，但 **全量 confirm358** 仅 +1.68 pp，bootstrap CI [-0.034, 0.073] 含 0。不支持干净的 superlinear 放大；负 α 方向与 MATH no-think 一致。

## MATH 对照

| 设定 | proxy | primary | 放大比 |
|------|-------|---------|--------|
| MATH no-think confirm400 | — | +3.00 pp | 1×（单轮） |
| ALFWorld confirm | -0.49 pp admissible | +1.68 pp episode | -3.45271502168482 |
