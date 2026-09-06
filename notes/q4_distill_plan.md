# Q4 蒸馏路线（2026-08-28）

## Step 1（进行中）：教师轨迹生成

- Split: **vector_fit 1200**
- Direction: full2k L14 `v_cap`（‖v‖≈24.50）
- α=**1.5**（trunc-gate confirm 上 overall 最佳 +11pp；both-finished 仍负，但教师正确率最高）
- mt=2048, batch=48, 每题 1 条 steered 轨迹
- 输出：`/tmp/ssopd04_teacher_vf1200_L14_a1_5/`

## Step 2（生成完成后）

1. 过滤 `correct=1` 作正样本教师
2. 新建 scale2k SSOPD04 配置（或 offline SFT）+ **confirm400 独立 holdout eval**
3. 注意：现有 `ssopd04_distill.py` 是 pyc stub，且为 online reverse-KL；离线轨迹更适合先做 **positive SFT / imitation** 探针

## 为何不是 α=2

trunc-gate：α=1.5 overall 优于 α=2；α=0.5 both-finished 最好但正确率更低。教师优先正确率 → α=1.5。
