#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:?set MANIFEST to checkpoint_specs.json}"
RUN_DIR="${RUN_DIR:?set a new RUN_DIR}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY:?set the pinned lm-eval git commit or package version}"

if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi
if [ "${DRY_RUN}" != "1" ] && [ -e "${RUN_DIR}" ]; then
  echo "[downstream] ERROR: refusing to overwrite ${RUN_DIR}"; exit 1
fi

HARNESS_ID="${LM_EVAL_IDENTITY}"
echo "[downstream] lm-eval identity: ${HARNESS_ID}"

while IFS=$'\t' read -r label checkpoint; do
  args=(--checkpoint "${checkpoint}" --label "${label}"
        --output "${RUN_DIR}/${label}/lm_eval_results.json"
        --expected-harness-identity "${HARNESS_ID}"
        --batch-size "${BATCH_SIZE}" --dtype "${DTYPE}" --seed "${SEED}")
  if [ "${INCLUDE_OPTIONAL}" = "1" ]; then args+=(--include-optional); fi
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
    print(f"{row['label']}\t{row['checkpoint_dir']}")
PY
)

if [ "${DRY_RUN}" = "1" ]; then exit 0; fi
python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "${MANIFEST}" --run-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}/paper_tables" --bootstrap-resamples 10000
