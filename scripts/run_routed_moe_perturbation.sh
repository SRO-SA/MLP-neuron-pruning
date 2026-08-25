#!/usr/bin/env bash
# Frozen target-6, local same-input routed-MoE perturbation audit.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:?Set MANIFEST to the five-checkpoint fine-frontier manifest}"
TOKENIZER_AUDIT="${TOKENIZER_AUDIT:?Set TOKENIZER_AUDIT to the fine-frontier audit JSON}"
SCORE_BUNDLE="${SCORE_BUNDLE:?Set SCORE_BUNDLE to all_expert_certificate_scores.npz}"
SCORE_MANIFEST="${SCORE_MANIFEST:?Set SCORE_MANIFEST to all_expert_certificate_scores.json}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a new output directory}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-32}"
SKIP_DOCUMENTS="${SKIP_DOCUMENTS:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_REVISION="${MODEL_REVISION:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
DRY_RUN="${DRY_RUN:-0}"

LABELS=(
  rmsnorm_alloc__ellipsoid_rank__p95__target6
  certified_hybrid__downnorm_refinement_slack0p25__target6
  certified_hybrid__downnorm_refinement_slack2__target6
  rmsnorm_alloc__downnorm_rank__p95__target6
)
CAPTURE_DIR="$RUN_DIR/baseline_capture"
RESULTS_DIR="$RUN_DIR/plan_results"
TABLE_DIR="$RUN_DIR/paper_tables"

for path in "$MANIFEST" "$TOKENIZER_AUDIT" "$SCORE_BUNDLE" "$SCORE_MANIFEST"; do
  test -f "$path" || { echo "[routed-moe] ERROR missing input: $path"; exit 1; }
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[routed-moe] DRY RUN"
  echo "  manifest       : $MANIFEST"
  echo "  tokenizer audit: $TOKENIZER_AUDIT"
  echo "  capture        : $CAPTURE_DIR"
  echo "  documents      : $NUM_DOCUMENTS C4 validation after eligible offset $SKIP_DOCUMENTS"
  echo "  max sequence   : $MAX_SEQ_LEN"
  printf '  plans          : %s\n' "${LABELS[*]}"
  exit 0
fi

test ! -e "$RUN_DIR" || {
  echo "[routed-moe] ERROR output already exists: $RUN_DIR"
  exit 1
}
mkdir -p "$RUN_DIR"

GPU_COUNT="$("$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the routed-MoE audit")
print(torch.cuda.device_count())
PY
)"
echo "[routed-moe] CUDA OK devices=$GPU_COUNT"
TIMING_TSV="$RUN_DIR/stage_timing.tsv"
printf 'stage\tstart_epoch\tend_epoch\twall_seconds\tgpu_count\tgpu_hours\n' > "$TIMING_TSV"

STAGE_START="$(date +%s)"
"$PYTHON_BIN" scripts/capture_routed_moe_baseline.py \
  --checkpoint-manifest "$MANIFEST" \
  --tokenizer-audit "$TOKENIZER_AUDIT" \
  --output-dir "$CAPTURE_DIR" \
  --num-documents "$NUM_DOCUMENTS" \
  --skip-documents "$SKIP_DOCUMENTS" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --model-revision "$MODEL_REVISION" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  2>&1 | tee "$RUN_DIR/baseline_capture.log"
STAGE_END="$(date +%s)"
"$PYTHON_BIN" - "$TIMING_TSV" baseline_capture "$STAGE_START" "$STAGE_END" "$GPU_COUNT" <<'PY'
import sys
path, stage, start, end, gpus = sys.argv[1:]
seconds = int(end) - int(start)
with open(path, "a", encoding="utf-8") as handle:
    handle.write(f"{stage}\t{start}\t{end}\t{seconds}\t{gpus}\t{seconds * int(gpus) / 3600:.9f}\n")
PY

mkdir -p "$RESULTS_DIR"
for label in "${LABELS[@]}"; do
  echo "[routed-moe] START fresh physical-checkpoint process: $label"
  STAGE_START="$(date +%s)"
  "$PYTHON_BIN" scripts/evaluate_routed_moe_perturbation.py \
    --capture-dir "$CAPTURE_DIR" \
    --checkpoint-manifest "$MANIFEST" \
    --label "$label" \
    --score-bundle "$SCORE_BUNDLE" \
    --score-manifest "$SCORE_MANIFEST" \
    --output-dir "$RESULTS_DIR/$label" \
    --dtype "$DTYPE" \
    2>&1 | tee "$RUN_DIR/${label}.log"
  STAGE_END="$(date +%s)"
  "$PYTHON_BIN" - "$TIMING_TSV" "$label" "$STAGE_START" "$STAGE_END" "$GPU_COUNT" <<'PY'
import sys
path, stage, start, end, gpus = sys.argv[1:]
seconds = int(end) - int(start)
with open(path, "a", encoding="utf-8") as handle:
    handle.write(f"{stage}\t{start}\t{end}\t{seconds}\t{gpus}\t{seconds * int(gpus) / 3600:.9f}\n")
PY
  echo "[routed-moe] FINISH $label"
done

"$PYTHON_BIN" scripts/summarize_routed_moe_perturbation.py \
  --run-dir "$RESULTS_DIR" \
  --output-dir "$TABLE_DIR" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 42

echo "[routed-moe] COMPLETE $RUN_DIR"
