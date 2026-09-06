#!/usr/bin/env bash
# After D1 rollouts finish: D2 splits → D3 fit → D4 confirm α → D5 L1/L2.
set -euo pipefail
BASE=/home/yiyangba/ssopd_paper_archive/gsm8k
PY=/scratch/ktang115/envs/bootstrap_alignment/bin/python
export PYTHONPATH=/scratch/ktang115/SSOPD
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/yiyangba/hf_cache
export CUDA_VISIBLE_DEVICES=0

ROLLOUTS=$BASE/data/ssopd01_qwen3_1_7b_gsm8k_1k/rollouts.jsonl
RESULTS=$BASE/reports/ssopd01_qwen3_1_7b_gsm8k_1k/results.json
LOG=$BASE/reports/d2_d5_pipeline.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

d1_ok() {
  $PY - <<'PY'
import json
from collections import Counter
from pathlib import Path
roll=Path("/home/yiyangba/ssopd_paper_archive/gsm8k/data/ssopd01_qwen3_1_7b_gsm8k_1k/rollouts.jsonl")
res=Path("/home/yiyangba/ssopd_paper_archive/gsm8k/reports/ssopd01_qwen3_1_7b_gsm8k_1k/results.json")
if not roll.exists() or not res.exists():
    raise SystemExit(1)
recs=[json.loads(l) for l in roll.read_text().splitlines() if l.strip()]
c=Counter(r["verification_status"] for r in recs)
d=json.loads(res.read_text())
ok = (
    len(recs) >= 8000
    and c.get("ERROR", 0) < 100
    and float(d.get("stats", {}).get("success_rate", 0)) >= 0.05
    and d.get("note") == "HF generate_batch seed= kwarg fix"
)
print(f"n={len(recs)} status={dict(c)} success={d.get('stats',{}).get('success_rate')} note={d.get('note')} ok={ok}")
raise SystemExit(0 if ok else 1)
PY
}

echo "[$(date -Is)] wait for D1 (strict)..."
while true; do
  if d1_ok; then
    echo "[$(date -Is)] D1 OK"
    break
  fi
  if ! pgrep -f 'run_gsm8k_rollouts.py' >/dev/null; then
    echo "[$(date -Is)] rollouts process dead; checking..."
    d1_ok || { echo "D1 failed validation"; exit 1; }
    break
  fi
  n=$(wc -l < "$ROLLOUTS" 2>/dev/null || echo 0)
  echo "[$(date -Is)] D1 progress n=$n/8000"
  sleep 300
done

echo "[$(date -Is)] D2 splits"
$PY -u "$BASE/scripts/build_gsm8k_splits.py" \
  --rollouts "$ROLLOUTS" \
  --out "$BASE/data/ssopd02_qwen3_1_7b_gsm8k_1k/splits.json" \
  --seed 42

echo "[$(date -Is)] D3 fit L14 v_cap"
$PY -u "$BASE/scripts/fit_gsm8k_direction.py" \
  --rollouts "$ROLLOUTS" \
  --splits "$BASE/data/ssopd02_qwen3_1_7b_gsm8k_1k/splits.json" \
  --out-dir "$BASE/data/ssopd02_qwen3_1_7b_gsm8k_1k" \
  --split-filter vector_fit

echo "[$(date -Is)] D4 confirm α grid"
OUT=$BASE/reports/ssopd03_qwen3_1_7b_gsm8k_confirm_alpha
mkdir -p "$OUT"
$PY -u "$BASE/scripts/scale2k_paper_protocol_smoke_resume.py" \
  --config "$BASE/configs/experiment_ssopd03_qwen3_1_7b_gsm8k_confirm_alpha.yaml" \
  --split confirm \
  --output-dir "$OUT" \
  --resume \
  | tee "$OUT/gsm8k_confirm_alpha.log"

echo "[$(date -Is)] D5 L1/L2 analysis"
$PY -u "$BASE/scripts/analyze_gsm8k_L1_L2.py" \
  --results "$OUT/results_confirm.json" \
  --out-json "$BASE/reports/gsm8k_L1_L2_length_analysis.json" \
  --out-md "$BASE/reports/gsm8k_L1_L2_verdict.md" \
  --mt 2048

echo "[$(date -Is)] PIPELINE DONE"
cat "$BASE/reports/gsm8k_L1_L2_verdict.md"
