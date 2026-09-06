# SSOPD Agent Experiments

Research code for **activation steering** and **instance-level procedural memory (UCE)** on MATH, GSM8K, and ALFWorld.

This checkout is a **GitHub-ready slice** of an internal archive: scripts, configs, notes, small artifacts. It does **not** include model weights, LoRA adapters, or multi-GB rollout jsonl.

## What stood up

| Line | Result |
|------|--------|
| MATH thinking-on inject | overall +5–11 pp; truncation-gate (both-finished ≥+2) **FAIL** |
| MATH no-think, refit v, α=−1.5 | **PASS** +3.00 / both +4.12 |
| ALFWorld hidden / NPM / H4 / H5 | **FAIL** (steering frozen) |
| ALFWorld UCE text workflow | **PASS** 29% → **40%** (+11 pp, CI [+1, +22]) |
| UCE-OPD internalization | **FAIL** 34% vs Vanilla-LoRA 50% |

Ledger: [`ALL_RESULTS.md`](ALL_RESULTS.md). Agent claim: [`notes/agent_ssopd_claim.md`](notes/agent_ssopd_claim.md).

## Layout

```
scripts/          experiment entry points (ALFWorld / MATH / BFCL)
configs/          YAML (cluster paths are rewritten at load time)
notes/            pre-registrations and write-ups
paper/            draft / venue notes
gsm8k/            GSM8K MVP scripts + small reports
tests/            CPU unit tests
third_party/      vendored SSOPD helpers (rollout, env, math, steering hooks)
artifacts/
  uce/            frozen UCE libraries (~561 workflows)
  directions/     L14 v_cap npz (MATH no-think + ALFWorld)
  reports/        JSON summaries only (not full jsonl)
```

Omitted on purpose: `*.safetensors`, select-pool rollouts (~0.5GB), confirm/unseen jsonl (GBs), NPM `*.npz` memories.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ALFWorld (optional, for env rollouts)
pip install alfworld
export ALFWORLD_DATA=/path/to/alfworld
export SSOPD_MODEL=/path/to/Qwen3-1.7B
export PYTHONPATH="$PWD/scripts:$PWD/third_party:$PWD/third_party/ssopd_logits:$PYTHONPATH"
```

Copy [`env.example`](env.example) to `.env` if you like. YAML still contains original cluster absolute paths; `scripts/alfworld_common.py` remaps them to this repo and the env vars above.

Unit tests (no GPU, no ALFWorld data):

```bash
PYTHONPATH=scripts:third_party:third_party/ssopd_logits python tests/test_agent_uce_opd.py
PYTHONPATH=scripts:third_party:third_party/ssopd_logits python tests/test_uce_whole_action.py
```

## Reproduce the standing ALFWorld result (UCE)

Needs Qwen3-1.7B + the **coldstart LoRA** (not in this repo; train with `scripts/alfworld_qwen3_coldstart_sft.py` on the sft partition, seed=1010) and ALFWorld `json_2.1.1`.

```bash
python scripts/uce_eval.py --config configs/experiment_uce_alfworld.yaml --phase uce
```

Eval protocol: **custom stepwise runner**, `valid_unseen` 100, `enable_thinking=False`, HF `generate`. Do **not** mix with `collect_episodes` numbers (39%/47%).

## Protocol rules (please keep)

- Do not put custom-runner 29%/40% in the same table as R0 `collect_episodes` 39%/47%.
- Do not attribute the UCE +11 pp to NPM / H4 / H5 steering.
- Distill only after a pre-registered gate; A1 / H / H3 / N / H4 all failed, so no steering teacher was distilled.
- UCE-SFT is SFT, not OPD.

## License

MIT for the experiment scripts and notes in this repository. Vendored `third_party/` code follows the original SSOPD project license if you redistribute it separately. ALFWorld, Qwen, and Hugging Face weights have their own terms.
