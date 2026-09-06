# N200 all2k 任务完成（2026-08-28 06:10 PDT）

## 状态：✅ 完成

结果目录：`/tmp/ssopd03_dirscale_N200_all2k/`

| 文件 | 说明 |
|------|------|
| `results_all.json` | 全 2000 题评测结果 |
| `ssopd03_dirscale_N200_all2k_report.md` | 分层报告 |
| `scale2k_paper_protocol_dirscale_N200_all2k.log` | 运行日志 |

## Headline

| 指标 | 值 |
|------|-----|
| clean acc | **49.30%** |
| steered @ α=2 | **59.45%** |
| gain_pp | **+10.15** |
| bootstrap 95% CI | **[8.45, 11.90] pp** |
| Gate (>=2pp) | **PASS** |
| wall time | ~4.2 h |

## 按 split

| split | n | gain_pp |
|-------|--:|--------:|
| vector_fit | 1200 | +10.50 |
| selection | 400 | +9.00 |
| confirm | 400 | +10.25 |

## 未截断子集

both length < 2048 (n=800): **−2.62 pp**（与 full2k 截断伪效应一致）

## 复制到项目（需 ktang115）

```bash
cp -a /tmp/ssopd03_dirscale_N200_all2k/results_all.json \
  /scratch/ktang115/SSOPD/ssopd_math/reports/ssopd03_dirscale_N200_all2k/
cp -a /tmp/ssopd03_dirscale_N200_all2k/ssopd03_dirscale_N200_all2k_report.md \
  /scratch/ktang115/SSOPD/ssopd_math/reports/
cp -a /tmp/scale2k_paper_protocol_smoke_resume.py \
  /scratch/ktang115/SSOPD/ssopd_math/scripts/scale2k_paper_protocol_smoke.py
# 更新 SCALE_QUEUE.md N200_all2k → ✅
```
