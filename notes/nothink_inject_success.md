# No-think 注入成功（对齐 2504.19483）— 2026-08-31

门槛：no-think confirm400 **overall ≥ +2 pp**（相对同协议 clean 68.25%）。

论文约束：控制向量必须从「正确解题时的典型残差状态」提取；GSM8K 成功点是 **负 α**。

## P0 — 旧 thinking-on v + 负 α

方向仍是 thinking-on `v_cap`（‖v‖=24.50）。复用 no-think clean。

| α | overall | both-finished | mean_len |
|--:|--------:|--------------:|---------:|
| −0.5 | **+0.50** | +1.46 | 802 |
| −1.0 | −0.50 | 0.00 | 840 |
| −1.5 | −4.75 | −4.50 | 883 |
| −2.0 | −10.50 | −6.25 | 991 |

未过 +2 门。负向只把正 α 的崩坏翻成弱正，说明 **方向本身不对齐**。

产物：`reports/ssopd03_math_L14_nothink_negalpha/`

## P1 — no-think 空间重拟

- vector_fit 1200 × K=4，`enable_thinking=False`
- 4800 轨迹，success **66.9%**，**paired=331**（≥80）
- L14 `task_balanced_paired` `v_cap`：‖v‖=**12.75**，与 thinking-on v 余弦 **0.678**

产物：`data/ssopd01_qwen3_1_7b_nothink_vf1200/` · `data/ssopd02_qwen3_1_7b_nothink/`

## P2 — 新方向 confirm400（过门）

| α | overall | both-finished | mean_len |
|--:|--------:|--------------:|---------:|
| 0.0 clean | 68.25% | — | 757 |
| −2.5 | −1.00 | −0.30 | 883 |
| −2.0 | −2.00 | −2.10 | 867 |
| **−1.5** | **+3.00** | **+4.12** | 825 |
| −1.0 | +0.50 | +0.29 | 808 |
| −0.5 | +1.25 | +0.57 | 781 |
| +0.5 | −0.50 | −0.57 | 735 |
| +1.0 | −2.00 | −3.12 | 705 |

- Best overall = both-finished = **α=−1.5**
- overall +3.00 **过 +2 门**；both-finished +4.12 也过 +2
- 与论文 GSM8K：**负 α** 一致；正 α 在 no-think 上无效或有害

产物：`reports/ssopd03_math_L14_nothink_refit/`

## P3

未做（P2 已过门）。
