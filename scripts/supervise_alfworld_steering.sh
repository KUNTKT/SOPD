#!/usr/bin/env bash
set -u
BASE=/home/yiyangba/ssopd_paper_archive
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
export PYTHONPATH=/scratch/ktang115/SSOPD:/scratch/ktang115/SSOPD/ssopd_logits
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0
export ALFWORLD_DATA=/scratch/ktang115/cache/alfworld

ADAPTER="${ADAPTER_PATH:-$BASE/data/ssopd05_alfworld_coldstart/coldstart_lora}"
CFG_SMOKE=$BASE/configs/experiment_ssopd05_alfworld_smoke.yaml
CFG=$BASE/configs/experiment_ssopd05_alfworld_confirm.yaml
LOG=$BASE/reports/ssopd05_alfworld_confirm/supervise.log
mkdir -p "$BASE/reports/ssopd05_alfworld_confirm" "$BASE/reports/ssopd05_alfworld_smoke" "$BASE/data/ssopd05_alfworld"

adapter_args() {
  if [ -n "$ADAPTER" ] && [ -d "$ADAPTER" ]; then
    echo "--adapter-path" "$ADAPTER"
  fi
}

run_stage() {
  local name=$1
  shift
  echo "[$(date -Is)] START $name" | tee -a "$LOG"
  if ! "$@"; then
    echo "[$(date -Is)] FAIL $name" | tee -a "$LOG"
    return 1
  fi
  echo "[$(date -Is)] OK $name" | tee -a "$LOG"
}

if [ ! -f "$BASE/reports/ssopd05_alfworld_smoke/smoke_results_coldstart.json" ]; then
  if [ ! -d "$ADAPTER" ]; then
    run_stage P0_coldstart $PY -u "$BASE/scripts/alfworld_qwen3_coldstart_sft.py" --config "$CFG_SMOKE"
  fi
  run_stage P0_smoke $PY -u "$BASE/scripts/alfworld_steering_smoke.py" \
    --config "$CFG_SMOKE" --backend vllm $(adapter_args) || true
  if [ -f "$BASE/data/ssopd05_alfworld_smoke/rollouts.jsonl" ]; then
    $PY -c "
import json, sys
sys.path.insert(0,'$BASE/scripts')
from alfworld_common import load_jsonl, trajectory_metrics, dump_json
from pathlib import Path
recs=load_jsonl(Path('$BASE/data/ssopd05_alfworld_smoke/rollouts.jsonl'))
m=trajectory_metrics(recs)
gate=m['success_rate']>=0.05 and m['admissible_action_rate']>=0.35
dump_json(Path('$BASE/reports/ssopd05_alfworld_smoke/smoke_results_coldstart.json'),
  {'metrics':m,'gate_pass':gate,'adapter_path':'$ADAPTER'})
print('gate_pass', gate)
"
  fi
fi

if [ ! -f "$BASE/reports/ssopd05_alfworld_confirm/p1_rollout_summary.json" ]; then
  run_stage P1_rollouts $PY -u "$BASE/scripts/alfworld_collect_rollouts.py" \
    --config "$CFG" $(adapter_args) --n-rollouts 4 --resume
fi

if [ ! -f "$BASE/data/ssopd05_alfworld/fit_meta.json" ]; then
  run_stage P2_fit $PY -u "$BASE/scripts/fit_alfworld_directions.py" --config "$CFG" --skip-smoke
fi

if [ ! -f "$BASE/reports/ssopd05_alfworld_confirm/results_confirm.json" ]; then
  run_stage P3_sweep $PY -u "$BASE/scripts/alfworld_injection_sweep.py" \
    --config "$CFG" $(adapter_args) --phase all
fi

echo "[$(date -Is)] ALL DONE" | tee -a "$LOG"
$PY -c "import json; r=json.load(open('$BASE/reports/ssopd05_alfworld_confirm/results_confirm.json')); print(json.dumps(r.get('analysis',{}), indent=2))"
