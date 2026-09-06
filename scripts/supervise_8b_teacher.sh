#!/usr/bin/env bash
# Keep 8B teacher gen + 1.7B LoRA SFT alive; resume on crash.
set -u
BASE=/home/yiyangba/ssopd_paper_archive
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
GEN=$BASE/reports/ssopd04_teacher_qwen3_8b_vf1200
SFT=$BASE/reports/ssopd04_qwen3_8b_teacher_lora_confirm
LOG=$BASE/reports/ssopd04_8b_supervise.log
export PYTHONPATH=/scratch/ktang115/SSOPD
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0

start_gen() {
  echo "[$(date -Is)] START/RESUME gen" | tee -a "$LOG"
  nohup $PY -u "$BASE/scripts/generate_teacher_traj.py" \
    --model-path /scratch/ktang115/models/Qwen3-8B \
    --output-dir "$GEN" \
    --split vector_fit \
    --batch-size 24 \
    --max-new-tokens 2048 \
    --no-steer \
    --resume \
    >> "$GEN/teacher_gen.log" 2>&1 &
}

start_sft() {
  echo "[$(date -Is)] START SFT" | tee -a "$LOG"
  mkdir -p "$SFT"
  nohup $PY -u "$BASE/scripts/q4_offline_lora_sft.py" \
    --teacher-jsonl "$GEN/teacher_correct.jsonl" \
    --output-dir "$SFT" \
    --model-path /scratch/ktang115/models/Qwen3-1.7B \
    --rollouts /scratch/ktang115/SSOPD/ssopd_math/data/ssopd01_qwen3_1_7b_scale2k/rollouts.jsonl \
    --splits /scratch/ktang115/SSOPD/ssopd_math/data/ssopd02_qwen3_1_7b_scale2k/splits.json \
    --reuse-base-eval "$BASE/reports/ssopd04_offline_lora_sft_confirm/base_eval.json" \
    --post-eval-seed 42 \
    --epochs 2 --lr 2e-4 --lora-r 16 \
    --eval-batch-size 48 --max-new-tokens 2048 \
    >> "$SFT/train_eval.log" 2>&1 &
}

while true; do
  if test -f "$SFT/results.json"; then
    echo "[$(date -Is)] ALL DONE" | tee -a "$LOG"
    python3 -c "import json;d=json.load(open('$SFT/results.json'));print('teachers',d.get('n_teacher_correct'),'post',d.get('post_eval'))" | tee -a "$LOG"
    exit 0
  fi
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  if test -f "$GEN/summary.json"; then
    ncor=$(wc -l < "$GEN/teacher_correct.jsonl" 2>/dev/null || echo 0)
    if ! pgrep -f '[p]ython.*q4_offline_lora_sft.py' >/dev/null; then
      if test -s "$GEN/teacher_correct.jsonl"; then
        start_sft
      else
        echo "[$(date -Is)] summary but empty correct" | tee -a "$LOG"
        exit 1
      fi
    fi
    last=$(grep -E 'epoch=|POST|BASE|eval |gain' "$SFT/train_eval.log" 2>/dev/null | tail -1)
    echo "[$(date -Is)] SFT n_correct=$ncor | $last | gpu=$gpu" | tee -a "$LOG"
  else
    n=$(wc -l < "$GEN/teacher_trajectories.jsonl" 2>/dev/null || echo 0)
    last=$(grep '^\[.*\] teacher ' "$GEN/teacher_gen.log" 2>/dev/null | tail -1)
    if ! pgrep -f '[p]ython.*generate_teacher_traj.py' >/dev/null; then
      echo "[$(date -Is)] GEN_DEAD n=$n — resume" | tee -a "$LOG"
      tail -15 "$GEN/teacher_gen.log" | tee -a "$LOG"
      start_gen
    else
      echo "[$(date -Is)] GEN n=$n/1200 | $last | gpu=$gpu" | tee -a "$LOG"
    fi
  fi
  sleep 180
done
