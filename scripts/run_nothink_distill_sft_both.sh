#!/usr/bin/env bash
set -eu
BASE=/home/yiyangba/ssopd_paper_archive
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
export PYTHONPATH=/scratch/ktang115/SSOPD
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0

BASE_EVAL=$BASE/reports/ssopd04_nothink_base_eval.json
STEER_T=$BASE/reports/ssopd04_teacher_nothink_L14_am1_5/teacher_correct.jsonl
VAN_T=$BASE/reports/ssopd04_teacher_nothink_vanilla_k0/teacher_correct.jsonl
STEER_O=$BASE/reports/ssopd04_nothink_steered_lora_confirm
VAN_O=$BASE/reports/ssopd04_nothink_vanilla_lora_confirm
mkdir -p "$STEER_O" "$VAN_O"

run_sft() {
  local teacher=$1 out=$2 tag=$3
  if test -f "$out/results.json"; then
    echo "[$(date -Is)] skip $tag already has results.json"
    return 0
  fi
  echo "[$(date -Is)] START $tag"
  $PY -u "$BASE/scripts/q4_offline_lora_sft.py" \
    --teacher-jsonl "$teacher" \
    --output-dir "$out" \
    --reuse-base-eval "$BASE_EVAL" \
    --epochs 2 --lr 2e-4 --lora-r 16 \
    --eval-batch-size 48 --max-new-tokens 2048 \
    --no-thinking \
    >> "$out/train_eval.log" 2>&1
  echo "[$(date -Is)] DONE $tag"
}

run_sft "$STEER_T" "$STEER_O" steered
run_sft "$VAN_T" "$VAN_O" vanilla
echo "[$(date -Is)] BOTH SFT DONE"
