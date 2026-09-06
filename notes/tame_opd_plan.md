# TAME-OPD Gate T0（预注册）

Teaching-Aware Memory Evolution 的可教性验证。本阶段只回答：在一个预先冻结的 UCE 改写 population 上，按短暂 OPD 更新后的无 workflow 学生效用选库，是否优于原始 UCE，以及是否优于按教师即时成功率选库。

## 主张边界

- T0 PASS 只证明：短暂学生更新后的无 workflow 效用，可以作为记忆选择目标。
- 不证明完整 self-evolution，也不证明可扩展 TAME。每个候选仍要实际训练一次学生。
- M1–M4 是冻结 population 的四个 individual，不是四个随机训练超参。
- Probe 是 pre-update student states 上的单次刷新 OPD，不是持续 on-policy。
- `teach_pick == teacher_pick`：T0-B not identifiable，不能写成反证 TAME。
- T0-A PASS 且 T0-B FAIL：只能写“某些改写比原始 UCE 更适蒸馏”。
- 无真人签字时，30 条样本审查叫 `structured_sample_audit`，论文不得写 human audit。
- 不打开 `valid_unseen`。不复用 H0 29% / B-uce 40% 作本门数字。不使用 `collect_episodes` 39%/47%。
- 旧 H0–H5、Gate B、NPM、A1、UCE-OPD D0 账本不得改写。

## 冻结协议

- 模型：Qwen3-1.7B + coldstart LoRA。steering 仅为 teacher-only contextual（教师看到 \(M_j\)，学生看不到）。
- 80 个进化题来自 `reports/uce_alfworld/B_uce_evolve.jsonl`。
- Manifest seed=3030。Run seeds=2020/2021/2022。Library seeds M1–M4 = 11/12/13/14。
- Entry seed：`(library_seed + int(SHA256(entry_id)[:16], 16)) mod 2^31`。
- Run 子 seed：`int(SHA256(f"{s}:{role}")[:16], 16) mod 2^31`，role ∈ {init, rollout, shuffle, evaluation}。
- Probe：64 optimizer steps，4096 supervised tokens，最后一步精确截断。
- Teacher prefix：每步对完整 render 做 `apply_workflow`，与 B-uce 等价。
- 审计顺序：生成 → 自动审计 → structured_sample_audit → 确定性回退 → 再审计 → 冻结 hash。冻结后禁止修复。
- 回退率 > 5%：整库无效，不补生成。并列破同分 `M0<M1<M2<M3<M4`。约束后只剩 M0：not identifiable。

## Gate

- T0-A：teach-pick vs M0 ≥ +3 pp，\(CI_{lo}>0\)，至少 2/3 seed 为正，相对 init 为正，教师不低于 M0 超过 2 pp。
- T0-B：teach-pick vs teacher-pick 的 \(\Delta>0\) 且 \(CI_{lo}>0\)。相同则 not identifiable。
- 统计：先对每题三 seed 取均值，再 task-level paired bootstrap。不把 300 局当独立样本。
- Gate 后停止。不自动多轮训练。本阶段不实现 \(\widehat U_{\mathrm{teach}}\) surrogate。
