# SSOPD 实验交接说明

给下一位接手的人：先读完这一页，再动代码或写论文。  
仓库：https://github.com/KUNTKT/SOPD  
权威数字：[`ALL_RESULTS.md`](ALL_RESULTS.md)  
ALFWorld 主张：[`notes/agent_ssopd_claim.md`](notes/agent_ssopd_claim.md)

本仓库是内部归档的 **公开切片**（约 20MB）：脚本、配置、笔记、UCE 库、方向向量、评测 summary。  
**不含** 模型权重、coldstart LoRA、多 GB 的 rollout jsonl。

---

## 1. 一分钟结论

| 线 | 判定 | 你能写什么 |
|----|------|------------|
| MATH thinking-on 注入 | overall 能涨；截断门 FAIL | 可写 overall；**不能**写成「效率且非缩短」 |
| MATH no-think 重拟 v，α=−1.5 | **PASS** +3.00 / both +4.12 | 关 thinking 必须重拟方向，负 α |
| GSM8K 注入 | α=0.5 +4；强 α 崩 | 增益仍堆在 hit-max 箱 |
| ALFWorld hidden / NPM / H4 / H5 | **全部 FAIL，已冻结** | 不得再扫，不得写成主创新 |
| ALFWorld UCE 文本 workflow | **唯一站住** 29%→**40%**（+11，CI [+1,+22]） | 可写 privileged 文本记忆，**不是** steering |
| UCE-OPD 内化 | **FAIL** 34%；Vanilla-LoRA 50% | 不进 D1；不宣称 memory 可内化 |

**论文能站住的 agent 结果只有一条：** 同模型成功轨迹 → 规则抽象成实例级 workflow → 检索 prepend。  
hidden / token / 整段动作外推都过不了门。不能写「超我教师 → 无外挂学生」。

---

## 2. 接手第一天读什么（按顺序）

1. 本 README（协议陷阱在第 4 节）
2. [`ALL_RESULTS.md`](ALL_RESULTS.md) — 全线数字
3. [`notes/agent_ssopd_claim.md`](notes/agent_ssopd_claim.md) — 能写 / 不能写
4. [`notes/agent_ssopd_plan.md`](notes/agent_ssopd_plan.md) — ALFWorld 每条实验怎么做的
5. [`notes/uce_opd_plan.md`](notes/uce_opd_plan.md) — 内化线预注册和 D0 失败

MATH 细节：`notes/nothink_inject_success.md`、`notes/truncation_gate_result.md`  
ALFWorld 早期注入：`notes/alfworld_steering_result.md`、`notes/alfworld_steering_injection_research.md`

---

## 3. 环境

```bash
git clone https://github.com/KUNTKT/SOPD.git
cd SOPD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install alfworld          # 要跑环境才需要
```

```bash
export SSOPD_MODEL=/path/to/Qwen3-1.7B
export ALFWORLD_DATA=/path/to/alfworld          # 里面应有 json_2.1.1
export PYTHONPATH="$PWD/scripts:$PWD/third_party:$PWD/third_party/ssopd_logits:$PYTHONPATH"
export PYTHONUNBUFFERED=1
```

完整变量见 [`env.example`](env.example)。  
YAML 里还留着原集群绝对路径；`scripts/alfworld_common.py` 会在加载时改写成上述环境变量和本仓库路径。

原集群（若你还有账号，完整数据在这里，**不要提交**）：

| 东西 | 路径 |
|------|------|
| 完整归档 | `/home/yiyangba/ssopd_paper_archive/` |
| 本公开切片 | `/home/yiyangba/ssopd-release/` |
| Qwen3-1.7B | `/scratch/ktang115/models/Qwen3-1.7B` |
| ALFWorld 数据 | `/scratch/ktang115/cache/alfworld` |
| coldstart LoRA | `/home/yiyangba/ssopd_paper_archive/data/ssopd05_alfworld_coldstart/coldstart_lora` |
| 内部 SSOPD 源码 | `/scratch/ktang115/SSOPD/`（`yiyangba` 无写权限） |
| Python | `/scratch/ktang115/envs/bootstrap_alignment/bin/python` |

不烧 **confirm358**。Train 分区 seed=**1010**：sft 484 / audit_select 358 / audit_confirm 358。

---

## 4. 协议陷阱（写错数字比写错代码更糟）

**两条 runner 不能并表。** 同一 1.7B + coldstart LoRA、同一 valid_unseen：

- 批量 `collect_episodes`（早期 ssopd05 / R0）：α=0 大约 **39%**
- 逐步 **custom runner**（NPM 以后、UCE、OPD 评测）：H0 = **29%**

正式 agent 账只用 custom runner 的 **29% / 40%**。  
8B 无 LoRA 是另一本账：H0 **15%** / B-uce **26%**（仍是 +11 pp，只作跨模型重复）。

其它硬规则：

- 不要把 UCE 的 +11 pp 算进 NPM / H4 / H5
- 不要把 MATH 注入/蒸馏直接外推到 ALFWorld
- **过门才蒸馏**。A1 / H / H3 / N / H4 全 FAIL，steering 教师从未蒸过
- UCE-SFT 叫 SFT，不叫 OPD
- H4 主对照是手写 decoder 的 **H4-control 38%**，不是 B-uce 40%（HF generate 对不齐）。数字 **38 / 33 / 37 不改**
- ALFWorld steering **已冻结**：不扫 `inject_style`、NPM、H4、H5，不加大 β

---

## 5. 仓库里有什么 / 没有什么

```
scripts/          实验入口
configs/          YAML
notes/            预注册和收口
paper/            草稿 / 场地判断
gsm8k/            GSM8K 脚本 + 小报告
tests/            CPU 单测（不需要 GPU / 环境）
third_party/      从内部 SSOPD 抽出的 Python（rollout / env / math / hook）
artifacts/uce/    冻结 UCE 库（种子库 + 进化后 561 条）
artifacts/directions/   L14 v_cap（MATH no-think + ALFWorld）
artifacts/reports/      各实验 summary JSON（不是全量 jsonl）
```

刻意没放：`*.safetensors`、select 轨迹、confirm/unseen 全量 jsonl、NPM `*.npz`。  
R0 全量 `results.json`（约 470MB）也没放，数字见笔记。

UCE 库是 **规则抽象**（去 `ACTION:`、去实例号、去 look/inventory、最长 12 步），不是 LLM 改写。  
种子库来自 `collect_episodes` 在 audit_select 上的成功轨迹；进化 80 题用的是 custom runner。措辞用 **self-generated procedural memory**，并写明「非 LLM 改写」。

---

## 6. 本地先跑通（不需要 ALFWorld）

```bash
PYTHONPATH=scripts:third_party:third_party/ssopd_logits python tests/test_agent_uce_opd.py
PYTHONPATH=scripts:third_party:third_party/ssopd_logits python tests/test_uce_whole_action.py
```

应打印 `ALL_MATH_OK`。

---

## 7. 复现站住的 UCE（需要 GPU + 环境 + LoRA）

1. 准备 Qwen3-1.7B 和 ALFWorld `json_2.1.1`
2. 若没有 coldstart LoRA：用 sft 池（seed=1010 的 40%）跑  
   `scripts/alfworld_qwen3_coldstart_sft.py`
3. 把 `configs/experiment_uce_alfworld.yaml` 里的 `adapter_path` 指到该 LoRA
4. 评测：

```bash
python scripts/uce_eval.py --config configs/experiment_uce_alfworld.yaml --phase uce
```

协议：custom runner，valid_unseen 100，`enable_thinking=False`，HF `generate`，库用 `artifacts/uce/uce_library_evolved.json`，retrieve **只读**。

建库 / 进化（不要重编冻结库，否则没法对照已入账的 40%）：

- 建种子库：`scripts/uce_build_library.py`
- 进化 + 评测：`uce_eval.py --phase uce`（`evolve_n=80` = `audit_select[:80]`）

---

## 8. 各条线入口（都已跑完，默认不要重开）

| 线 | 配置 | 入口 | 状态 |
|----|------|------|------|
| MATH no-think 注入 | `configs/experiment_ssopd03_math_L14_nothink_*.yaml` | `scripts/fit_nothink_directions.py` 等 | 重拟 PASS |
| MATH 蒸馏 | `scripts/q4_offline_lora_sft.py` | thinking-on 成立；no-think FAIL | |
| GSM8K | `gsm8k/` | 截断跨集；蒸馏低优先 | |
| 静态 CAA / R0 / B3 | `configs/experiment_ssopd05_alfworld_*.yaml` | `alfworld_injection_*.py` | **冻结** |
| NPM-lite / hybrid | `configs/experiment_agent_ssopd_*.yaml` | `agent_ssopd_*.py` | FAIL，未蒸馏 |
| 忠实 NPM 1.7B / 8B | `configs/experiment_npm_faithful*.yaml` | `npm_faithful_*.py` | FAIL |
| UCE | `configs/experiment_uce_alfworld.yaml` | `uce_eval.py` | **PASS 40%** |
| H3 UCE+NPM | `configs/experiment_uce_npm_hybrid.yaml` | `uce_npm_hybrid_eval.py` | FAIL |
| H4 logit | `configs/experiment_uce_counterfactual_logit.yaml` | `uce_counterfactual_logit*.py` | **关闭，不挽救** |
| H5 整段动作 | `configs/experiment_uce_whole_action.yaml` | `uce_whole_action*.py` | 信号门 FAIL |
| UCE-OPD D0 | `configs/experiment_uce_opd.yaml` | `run_uce_opd_d0.sh` | FAIL，**不进 D1** |
| BFCL | `configs/experiment_bfcl_propensity.yaml` | `bfcl_*.py` | Gate A FAIL |

数字 JSON 在 `artifacts/reports/`。

---

## 9. 你还可以做什么 / 不要做什么

**可以**

- 用冻结 UCE 库复现 29→40（同一 custom runner）
- 写「文本 procedural memory」，并注明规则抽象、非 LLM 改写
- 另开完全新的预注册（换任务、换模型）——不要改已入账的旧门

**不要**

- 再扫 ALFWorld `inject_style` / α / NPM / H4 β / H5 评测
- 为了过 D0 去调 KL、lr、epoch、rank（预注册禁止）
- 把 UCE-OPD 34% 写成成功内化
- 重编进化库再和旧 40% 比
- 把本仓库没有的 LoRA / jsonl 提交到 git

UCE-OPD 的 D1 脚本在 `scripts/agent_uce_opd_d1.py`，**Gate D0 FAIL，不要跑**。

---

## 10. 许可

实验脚本与笔记：MIT（见 `LICENSE`）。  
`third_party/` 来自内部 SSOPD 源码切片，见 `NOTICE`。  
ALFWorld、Qwen、Hugging Face 权重各有条款。
