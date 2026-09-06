# No-think 蒸馏（steered vs vanilla）— 2026-08-31

协议：confirm400、关 thinking、学生评测无注入。Base = 已有 no-think clean **68.25%**。LoRA r=16，2 epoch，lr=2e-4。

## 教师

| | n | acc | 正确条 | 均长 |
|--|--:|----:|------:|-----:|
| vanilla k=0（现成 rollouts） | 1200 | 66.25% | **795** | 744 |
| steered L14 α=−1.5 | 1200 | **68.50%** | **822** | 842 |

教师自身 +2.25 pp（接近注入 confirm 的 +3.00）。

## 学生 confirm400（无注入）

| 条件 | acc | vs base | both-finished | mean_len | hit_max |
|------|----:|--------:|--------------:|---------:|--------:|
| no-think base | 68.25% | — | — | 757 | 9.0% |
| steered LoRA | **68.25%** | **+0.00** | −0.29 (n=339) | 1100 | 12.5% |
| vanilla LoRA | 67.00% | **−1.25** | −1.75 (n=343) | 1035 | 10.0% |

- 门槛 steered − base ≥ +2：**FAIL**（CI [−3.75, +4.00]）
- steered 学生 − vanilla 学生 = **+1.25 pp**（thinking-on 时约 +4.75）
- 两套学生均长都变长（~750 → ~1100），不像 thinking-on 蒸馏那样缩短

产物：
- `reports/ssopd04_teacher_nothink_L14_am1_5/`
- `reports/ssopd04_teacher_nothink_vanilla_k0/`
- `reports/ssopd04_nothink_steered_lora_confirm/`
- `reports/ssopd04_nothink_vanilla_lora_confirm/`
