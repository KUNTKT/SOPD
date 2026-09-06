#!/usr/bin/env bash
set -u
BASE=/home/yiyangba/ssopd_paper_archive
OUTS=$BASE/reports/ssopd04_nothink_sft_pipeline.log
STEER=$BASE/reports/ssopd04_nothink_steered_lora_confirm/results.json
VAN=$BASE/reports/ssopd04_nothink_vanilla_lora_confirm/results.json

job_alive() {
  pgrep -f 'q4_offline_lora_sft.py' >/dev/null || pgrep -f 'run_nothink_distill_sft_both.sh' >/dev/null
}

start_job() {
  echo "[$(date -Is)] START/RESUME both SFT" | tee -a "$OUTS"
  nohup bash "$BASE/scripts/run_nothink_distill_sft_both.sh" >> "$OUTS" 2>&1 &
  echo "[$(date -Is)] pid=$!" | tee -a "$OUTS"
}

while true; do
  if test -f "$STEER" && test -f "$VAN"; then
    echo "[$(date -Is)] BOTH RESULTS PRESENT" | tee -a "$OUTS"
    exit 0
  fi
  if ! job_alive; then
    echo "[$(date -Is)] DEAD — resume" | tee -a "$OUTS"
    start_job
  fi
  last=$(grep -E 'epoch=|eval |DONE|START|gain' \
    $BASE/reports/ssopd04_nothink_steered_lora_confirm/train_eval.log \
    $BASE/reports/ssopd04_nothink_vanilla_lora_confirm/train_eval.log \
    2>/dev/null | tail -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
  echo "[$(date -Is)] $last | gpu=$gpu" | tee -a "$OUTS"
  sleep 180
done
