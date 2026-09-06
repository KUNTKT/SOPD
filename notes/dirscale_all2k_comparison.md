# DirScale all2k 对比（N200 vs N800）

更新时间：2026-08-28 10:30 PDT

| N | confirm400 gain | all2k gain | bootstrap CI | both-finished gain |
|---|----------------:|-----------:|:-------------|-------------------:|
| 200 | +10.25 pp | **+10.15 pp** | [8.45, 11.90] | −2.62 pp |
| 800 | +8.25 pp | **+11.50 pp** | [9.60, 13.30] | −4.20 pp |

## 结论

1. **N800 全量 +11.50 pp > N200 +10.15 pp**，但 confirm400 上 N200 更高（+10.25 vs +8.25）→ 全量含 vector_fit 1200 时 N800 方向更强。
2. 两者 confirm split gain 接近（N200 +10.25，N800 +9.50）。
3. 未截断子集均为负，截断伪效应结论不变。

## 产物

- N200: `/tmp/ssopd03_dirscale_N200_all2k/`
- N800: `/tmp/ssopd03_dirscale_N800_all2k/`
