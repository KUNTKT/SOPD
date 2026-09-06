# T1-V 去实例化结构变异可行性门（预注册）

T0 冻结。本阶段只证明：Qwen3-1.7B 能否形成四个可用 library individuals。不重跑 OPD，不生成 2244 条全文改写。

## 判定链

四个 seed 分别有效 → accepted residual leak = 0 → 跨 seed 至少 3 个实质不同 canonical hash（45/60）→ structured validity（无真人签字不得称 semantic preservation）→ Gate V。

240 总体统计只作辅报。

## 编辑器契约

看不到 observation、轨迹、goal。只看原 workflow lines 与按 task_type 聚合的去实例化错误统计。只输出 JSON patch。确定性 renderer 生成最终 workflow。

## 停止

FAIL：停止 Qwen3-1.7B 上的 ALFWorld TAME，不调泄漏阈值。PASS：先汇报，不自动扩全库。
