#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/ktang115/SSOPD:/scratch/ktang115/SSOPD/ssopd_logits
export ALFWORLD_DATA=/scratch/ktang115/cache/alfworld
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
CFG=/home/yiyangba/ssopd_paper_archive/configs/experiment_uce_opd.yaml
REP=/home/yiyangba/ssopd_paper_archive/reports/uce_opd
AD=/home/yiyangba/ssopd_paper_archive/data/uce_opd

full_opd_done() {
  local tag="$1"
  $PY - "$REP/train_${tag}.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text())
eps = d.get("epochs") or []
ok = len(eps) >= 2 and all(int(e.get("n_episodes") or 0) >= 120 for e in eps)
raise SystemExit(0 if ok else 1)
PY
}

if [[ ! -f "$REP/collect_vanilla_theta0.jsonl" ]]; then
  $PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_collect.py --config "$CFG" --mode vanilla_theta0 --out "$REP/collect_vanilla_theta0.jsonl"
fi
if [[ ! -f "$REP/collect_uce_teacher.jsonl" ]]; then
  $PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_collect.py --config "$CFG" --mode uce_teacher --out "$REP/collect_uce_teacher.jsonl"
fi
if [[ ! -f "$AD/vanilla_sft/adapter_config.json" ]]; then
  $PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_train.py --config "$CFG" --mode vanilla_sft --source-jsonl "$REP/collect_vanilla_theta0.jsonl"
fi
if [[ ! -f "$AD/uce_sft/adapter_config.json" ]]; then
  $PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_train.py --config "$CFG" --mode uce_sft --source-jsonl "$REP/collect_uce_teacher.jsonl"
fi
if ! full_opd_done uce_opd; then
  $PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_train.py --config "$CFG" --mode uce_opd --no-resume
fi
if ! full_opd_done no_pi_opd; then
  $PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_train.py --config "$CFG" --mode no_pi_opd --no-resume
fi
$PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_eval.py --config "$CFG" --no-resume
$PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_ledger.py --stage d0
echo D0_PIPELINE_DONE
if $PY - "$REP/summary.json" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if d.get("gate", {}).get("gate_pass") else 1)
PY
then
  bash /home/yiyangba/ssopd_paper_archive/scripts/run_uce_opd_d1.sh
else
  echo "Gate D0 FAIL; stop. No D1, no hyperparameter search."
fi
