# Agent Handoff 更新草案 — 2026-08-29

## 已完成（相对旧 HANDOFF）

1. N200/N800 all2k 注入 ✅
2. Truncation-gate α∈{0.5…2.5} confirm400 ✅ → **FAIL**
3. Q4 steered teacher + offline LoRA ✅ → **PASS +11.5 pp**
4. 带 checkpoint 的 smoke / teacher / SFT 脚本在 archive `scripts/`

## 环境注意

- 用户 `yiyangba` **不能写** `/scratch/ktang115/SSOPD/`
- 权威副本：`/home/yiyangba/ssopd_paper_archive/`
- Python：`/scratch/ktang115/envs/bootstrap_alignment/bin/python`
- PYTHONPATH：`/scratch/ktang115/SSOPD`

## 论文状态

可开写诚实技术稿；主 claim 禁止「干净推理提升」。蒸馏强结论依赖 vanilla 对照结果。
