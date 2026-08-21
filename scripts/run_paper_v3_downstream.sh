#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:?set MANIFEST to checkpoint_specs.json}"
TOKENIZER_AUDIT="${TOKENIZER_AUDIT:?set TOKENIZER_AUDIT to tokenizer_audit.json}"
RUN_DIR="${RUN_DIR:?set a new RUN_DIR}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
TRUST_DATASET_CODE="${TRUST_DATASET_CODE:?set TRUST_DATASET_CODE=1 to authorize the pinned MathQA dataset script}"
ONLY_TARGETS="${ONLY_TARGETS:-}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
RUN_KIND="${RUN_KIND:-all}"
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY:?set the pinned lm-eval git commit or package version}"

case "${TRUST_DATASET_CODE}" in
  0|1) ;;
  *) echo "[downstream] ERROR: TRUST_DATASET_CODE must be 0 or 1"; exit 1 ;;
esac

if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi
if [ "${DRY_RUN}" != "1" ] && [ -e "${RUN_DIR}" ]; then
  echo "[downstream] ERROR: refusing to overwrite ${RUN_DIR}"; exit 1
fi

HARNESS_ID="${LM_EVAL_IDENTITY}"
echo "[downstream] lm-eval identity: ${HARNESS_ID}"

case "${RUN_KIND}" in
  all|primary_only|comparator_only|additional_only) ;;
  *) echo "[downstream] ERROR: RUN_KIND must be all, primary_only, comparator_only, or additional_only"; exit 1 ;;
esac

while IFS=$'\t' read -r label checkpoint target comparator additional; do
  case "${RUN_KIND}" in
    comparator_only) [ "${comparator}" = "True" ] || continue ;;
    additional_only) [ "${additional}" = "True" ] || continue ;;
    primary_only) [ "${comparator}" != "True" ] && [ "${additional}" != "True" ] || continue ;;
  esac
  if [ -n "${ONLY_TARGETS}" ]; then
    target_int="${target%%.*}"
    case ",${ONLY_TARGETS}," in
      *",${target_int},"*) ;;
      *) continue ;;
    esac
  fi
  args=(--checkpoint "${checkpoint}" --label "${label}"
        --tokenizer-audit "${TOKENIZER_AUDIT}"
        --output "${RUN_DIR}/${label}/lm_eval_results.json"
        --expected-harness-identity "${HARNESS_ID}"
        --batch-size "${BATCH_SIZE}" --dtype "${DTYPE}" --seed "${SEED}")
  if [ "${INCLUDE_OPTIONAL}" = "1" ]; then args+=(--include-optional); fi
  if [ "${TRUST_DATASET_CODE}" = "1" ]; then args+=(--trust-dataset-code); fi
  if [ -n "${SMOKE_LIMIT}" ]; then args+=(--limit "${SMOKE_LIMIT}"); fi
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[downstream] WOULD RUN ${label}: ${checkpoint}"
  else
    mkdir -p "${RUN_DIR}/${label}"
    python3 scripts/run_paper_v3_lm_eval.py "${args[@]}" 2>&1 \
      | tee "${RUN_DIR}/${label}/run.log"
  fi
done < <(python3 - "${MANIFEST}" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1])):
    print(
        f"{row['label']}\t{row['checkpoint_dir']}\t{row['target_pct']}\t"
        f"{row.get('downstream_comparator_only', False)}\t"
        f"{row.get('additional_operating_point', False)}"
    )
PY
)

if [ "${DRY_RUN}" = "1" ]; then exit 0; fi
if [ "${SKIP_SUMMARY}" = "1" ]; then
  echo "[downstream] SKIP_SUMMARY=1; raw smoke outputs retained in ${RUN_DIR}"
  exit 0
fi
python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "${MANIFEST}" --run-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}/paper_tables" --bootstrap-resamples 10000
