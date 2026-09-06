# DVPD Gate S0（预注册）

UCE 只提出分叉点动作。可写回价值由无 workflow 的 coldstart 策略、在剩余预算下的反事实续跑决定。本阶段不训练。

## 范围

结论只能写在「已发生 base/UCE 完整动作分叉」的任务状态上，不能推广到全部任务。

## 协议

- 排除集 = 既有 train-side ID 真实并集（evolve / distill / T0 四组），写出各来源计数。不打开 valid_unseen。
- 60 题 × pair seed {0,1}，`sample_seed=6060`。
- 分叉只比较 `predicted_tool`。
- 续跑：干净 raw-obs replay；`H_remain = H_max - (t*+1)`；即时终止复制 K=8；rejected force 仍 `step` 并进 ΔQ。
- 主统计：task-macro mean；bootstrap 重采样 \(T_D\)。
- 主门大效应：`ΔQ ≥ 0.25` 的状态数 ≥ 30（带符号）。

## 停止

FAIL 终止本方法线。PASS 只汇报，不自动训练。
