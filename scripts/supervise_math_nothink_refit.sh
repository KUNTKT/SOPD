#!/usr/bin/env bash
set -u
BASE=/home/yiyangba/ssopd_paper_archive
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
OUT=$BASE/reports/ssopd03_math_L14_nothink_refit
CFG=$BASE/configs/experiment_ssopd03_math_L14_nothink_refit.yaml
LOG=$OUT/supervise.log
export PYTHONPATH=/scratch/ktang115/SSOPD
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"

job_alive() {
  pgrep -af 'scale2k_paper_protocol_smoke_resume.py' | grep -q 'ssopd03_math_L14_nothink_refit'
}

start_job() {
  echo "[$(date -Is)] START/RESUME nothink-refit inject" | tee -a "$LOG"
  nohup env PYTHONPATH=/scratch/ktang115/SSOPD PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/yiyangba/hf_cache CUDA_VISIBLE_DEVICES=0 \
    $PY -u "$BASE/scripts/scale2k_paper_protocol_smoke_resume.py" \
    --config "$CFG" \
    --split confirm \
    --output-dir "$OUT" \
    --no-thinking \
    --resume \
    >> "$OUT/inject_refit.log" 2>&1 &
  echo "[$(date -Is)] pid=$!" | tee -a "$LOG"
}

while true; do
  if test -f "$OUT/results_confirm.json"; then
    echo "[$(date -Is)] DONE" | tee -a "$LOG"
    $PY -c "import json;r=json.load(open('$OUT/results_confirm.json'));print('best',r.get('best_alpha'),r.get('best_gain_pp'),'both',r.get('best_both_finished_gain_pp'))"
    exit 0
  fi
  if ! job_alive; then
    echo "[$(date -Is)] DEAD — resume" | tee -a "$LOG"
    tail -20 "$OUT/inject_refit.log" 2>/dev/null | tee -a "$LOG"
    start_job
  fi
  last=$(grep -E 'audit rollout|PAPER SMOKE|confirm sweep|prompt_check|steered alpha' "$OUT/inject_refit.log" 2>/dev/null | tail -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
  echo "[$(date -Is)] $last | gpu=$gpu" | tee -a "$LOG"
  sleep 180
done
