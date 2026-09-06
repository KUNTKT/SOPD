# ICLR P0 门槛判定（2026-08-29）

## 主 claim

「Steered-teacher 蒸馏提升效率，且**不是**因为缩短生成。」

## 实验结果汇总

| 条件 | overall gain | post mean_len | both-finished gain vs base |
|------|-------------:|--------------:|---------------------------:|
| steered LoRA (orig seed+1000) | +11.50 | 1517 | −2.35 |
| steered LoRA **seed=42 (L3)** | **+12.50** | 1492 | −2.99 |
| vanilla 743 | +7.75 | 1727 | 0.00 |
| vanilla matched-633 (L5) | +5.25 | 1756 | −0.68 |
| steered L4 lenmatch n=367 | **+13.75** | 1551 | −3.03 |
| vanilla L4 lenmatch n=367 | +3.50 | 1720 | +0.64 |

L3 后 steered−vanilla（同 seed42）≈ **+12.5 − 7.75 = +4.75 pp**（overall 仍显著）。  
L4 同长度教师对：steered−vanilla ≈ **+10.25 pp** overall。

## L1/L2（决定性）

按 **BASE length** 分箱：finished 箱内 steered **不优于** base；**几乎全部 overall 增益来自 base hit-max 箱 [2048,∞)**。  
三模型 both-finished：steered **低于** base 与 vanilla。

L3/L4 复检后同构：hit-max 箱 s−b ≈ +24–26 pp；finished 箱 s−b ≤ 0。

## 计划门槛对照

| 标准 | 结果 |
|------|------|
| L1 同箱内 steered 仍显著优于 base 且 ≥ vanilla | **FAIL** |
| L2 both-finished 正增益 | **FAIL** |
| L4 长度匹配后仍支持「非缩短」 | **FAIL**（overall 仍赢，但赢在截断箱；finished 仍负） |
| L3 steered≈vanilla | 否（overall 仍差 +4.75）但无关主 claim |

## 最终决定

1. **停止「效率且非缩短」作为 ICLR 主 claim。**  
2. **不启动 E2 第二模型**（为该主 claim 服务）。  
3. 可保留的弱结论：steered 教师在 **overall** 上优于 vanilla（+4–10 pp），但机制与 test-time steering 类似，**主要是截断/长度行为迁移**，不能写成干净效率提升。  
4. 投稿档位：回到 **workshop / 技术报告**（truncation 方法论 + 蒸馏 overall 现象）；若坚持 ICLR，须改 claim（例如纯评测伪效应负结果长文），另开计划。

## 产物

- `iclr_L1_L2_verdict.md` / `iclr_L1_L2_length_analysis.json`
- `ssopd04_steered_lora_reseed42/`
- `ssopd04_vanilla_lora_matched633/`
- `ssopd04_steered_lora_L4_lenmatch/` + `ssopd04_vanilla_lora_L4_lenmatch/`
- `L4_length_match_note.md`
