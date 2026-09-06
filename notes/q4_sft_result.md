# Q4 offline LoRA SFT — PASS（2026-08-28）

Confirm400 holdout, mt=2048.

| | acc |
|--|----:|
| base | 51.75% |
| post (LoRA on 743 teacher correct) | **63.25%** |
| **gain** | **+11.50 pp** |
| bootstrap 95% CI | [7.75, 15.75] |

- Teachers: vector_fit1200 @ L14 α=1.5 steered, correct-only
- Train 669 / val 74; 2 epochs; lora_r=16; ~3.5 min train
- Gate ≥2pp: **PASS**
- 产物：`/tmp/ssopd04_offline_lora_sft_confirm/`
