#!/usr/bin/env bash
set -u
BASE=/home/yiyangba/ssopd_paper_archive
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
OUT=$BASE/data/ssopd01_qwen3_1_7b_nothink_vf1200
LOG=$OUT/supervise.log
export PYTHONPATH=/scratch/ktang115/SSOPD
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"

job_alive() {
  pgrep -f 'generate_nothink_rollouts.py' >/dev/null
}

start_job() {
  echo "[$(date -Is)] START/RESUME nothink rollouts" | tee -a "$LOG"
  nohup env PYTHONPATH=/scratch/ktang115/SSOPD PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/yiyangba/hf_cache CUDA_VISIBLE_DEVICES=0 \
    $PY -u "$BASE/scripts/generate_nothink_rollouts.py" \
    --group-size 4 --batch-size 48 \
    >> "$OUT/rollouts.log" 2>&1 &
  echo "[$(date -Is)] pid=$!" | tee -a "$LOG"
}

while true; do
  if test -f "$OUT/summary.json"; then
    echo "[$(date -Is)] DONE" | tee -a "$LOG"
    cat "$OUT/summary.json" | tee -a "$LOG"
    exit 0
  fi
  if ! job_alive; then
    echo "[$(date -Is)] DEAD — resume" | tee -a "$LOG"
    tail -20 "$OUT/rollouts.log" 2>/dev/null | tee -a "$LOG"
    start_job
  fi
  last=$(grep -E 'rollout |DONE|prompt_check|nothink rollouts' "$OUT/rollouts.log" 2>/dev/null | tail -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
  echo "[$(date -Is)] $last | gpu=$gpu" | tee -a "$LOG"
  sleep 180
done
