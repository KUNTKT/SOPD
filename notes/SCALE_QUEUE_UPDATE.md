# SCALE_QUEUE 更新草案（2026-08-29）

当前 GPU：空闲（归档时）  
约定：paper-protocol 默认 `max_new_tokens=2048`；主 claim 不以 overall gain 写「干净推理提升」。

| 任务 | 状态 |
|------|------|
| N200_all2k | ✅ +10.15 pp（both-finished −2.62） |
| N800_all2k | ✅ +11.50 pp（both-finished −4.20） |
| TruncGate α 扫 | ✅ FAIL（best both +1.96 @α=0.5） |
| Q4 teacher traj | ✅ 1200 @α=1.5，acc 61.9%，743 correct |
| Q4 steered LoRA | ✅ PASS +11.50 pp confirm400 |
| Q4 vanilla LoRA 对照 | ✅ +7.75 pp（vs steered +11.50；差值 +3.75） |

产物归档：`/home/yiyangba/ssopd_paper_archive/`（待拷入项目 reports）
