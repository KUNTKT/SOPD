# MATH L14 注入 · 关 thinking（confirm400）— 完成（2026-08-31）

同一方向、同一 split、同一 α 网格，只把 Qwen3 `apply_chat_template(..., enable_thinking=False)`（prompt 预填空 `<think></think>`）。**clean 与 steered 都在 no-think 下重评。**

- 模型：Qwen3-1.7B，mt=2048，batch=48，L14 `v_cap` `contrastive_cap`
- 方向仍来自 thinking-on 拟合的 `layer_14_task_balanced_paired_v_cap`（未重拟合）
- clean：**68.25%**，均长 757，hit-max **9.0%**（thinking-on confirm 约 50% / 均长~1700+）

| α | overall | both-finished | mean_len | hit_steered |
|--:|--------:|--------------:|---------:|------------:|
| 0.0 | 68.25%（基线） | — | 757 | 9.0% |
| 0.5 | **−1.50** | **−2.51** | 695 | 5.5% |
| 1.0 | −5.00 | −6.63 | 635 | 1.2% |
| 1.5 | −12.00 | −13.65 | 585 | 2.0% |
| 2.0 | −26.50 | −30.00 | 469 | 1.0% |
| 2.5 | −44.75 | −48.57 | 435 | 5.2% |

- Best overall / both-finished：都是 **不注入**（α=0）；注入单调变差
- 对照 thinking-on trunc-gate：α=1.5 曾是 **+11.00** overall，此处 **−12.00**

产物：`reports/ssopd03_math_L14_nothink_confirm/`
