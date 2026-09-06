#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/ktang115/SSOPD:/scratch/ktang115/SSOPD/ssopd_logits
export ALFWORLD_DATA=/scratch/ktang115/cache/alfworld
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
CFG=/home/yiyangba/ssopd_paper_archive/configs/experiment_uce_opd.yaml
SUM=/home/yiyangba/ssopd_paper_archive/reports/uce_opd/summary.json
if [[ ! -f "$SUM" ]]; then
  echo "missing D0 summary; refuse D1"
  exit 1
fi
$PY - "$SUM" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if d.get("gate", {}).get("gate_pass") else 1)
PY
$PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_d1.py --config "$CFG"
$PY /home/yiyangba/ssopd_paper_archive/scripts/agent_uce_opd_ledger.py --stage d1
echo D1_PIPELINE_DONE
