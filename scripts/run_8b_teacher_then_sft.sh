#!/usr/bin/env bash
# After 8B teacher gen finishes: filter already written; LoRA SFT 1.7B + confirm400.
set -euo pipefail
BASE=/home/yiyangba/ssopd_paper_archive
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
export PYTHONPATH=/scratch/ktang115/SSOPD
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0

GEN=$BASE/reports/ssopd04_teacher_qwen3_8b_vf1200
SFT=$BASE/reports/ssopd04_qwen3_8b_teacher_lora_confirm
SUM=$GEN/summary.json
LOG=$BASE/reports/ssopd04_8b_teacher_pipeline.log

{
  echo "[$(date -Is)] wait 8B teacher gen"
  while ! test -f "$SUM"; do
    if ! pgrep -f 'generate_teacher_traj.py' >/dev/null; then
      echo "[$(date -Is)] gen process dead without summary"
      tail -30 "$GEN/teacher_gen.log" || true
      exit 1
    fi
    n=$(wc -l < "$GEN/teacher_trajectories.jsonl" 2>/dev/null || echo 0)
    echo "[$(date -Is)] gen progress n=$n/1200"
    sleep 180
  done
  echo "[$(date -Is)] gen done"
  cat "$SUM"
  test -s "$GEN/teacher_correct.jsonl"

  echo "[$(date -Is)] LoRA SFT 1.7B"
  mkdir -p "$SFT"
  $PY -u "$BASE/scripts/q4_offline_lora_sft.py" \
    --teacher-jsonl "$GEN/teacher_correct.jsonl" \
    --output-dir "$SFT" \
    --model-path /scratch/ktang115/models/Qwen3-1.7B \
    --rollouts /scratch/ktang115/SSOPD/ssopd_math/data/ssopd01_qwen3_1_7b_scale2k/rollouts.jsonl \
    --splits /scratch/ktang115/SSOPD/ssopd_math/data/ssopd02_qwen3_1_7b_scale2k/splits.json \
    --reuse-base-eval "$BASE/reports/ssopd04_offline_lora_sft_confirm/base_eval.json" \
    --post-eval-seed 42 \
    --epochs 2 --lr 2e-4 --lora-r 16 \
    --eval-batch-size 48 --max-new-tokens 2048
  echo "[$(date -Is)] SFT DONE"
  python3 - <<'PY'
import json
from pathlib import Path
p=Path("/home/yiyangba/ssopd_paper_archive/reports/ssopd04_qwen3_8b_teacher_lora_confirm/results.json")
if p.exists():
    d=json.loads(p.read_text())
    print({k:d.get(k) for k in ("n_teacher_correct","base_eval") if k in d})
    pe=d.get("post_eval") or d.get("post")
    print("post", pe)
PY
} 2>&1 | tee -a "$LOG"
