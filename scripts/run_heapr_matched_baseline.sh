#!/usr/bin/env bash
# Compatibility-gated HEAPr run. This does not claim equal pruning semantics.
set -euo pipefail

HEAPR_COMMIT="${HEAPR_COMMIT:?pin the official HEAPr git commit}"
WORK_DIR="${WORK_DIR:?set a new external-work directory}"
RUN_DIR="${RUN_DIR:?set a new result directory}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
RATIO="${RATIO:-0.06207}"
CALI_NSAMPLES="${CALI_NSAMPLES:-128}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRUST_DATASET_CODE="${TRUST_DATASET_CODE:?set TRUST_DATASET_CODE=1 to authorize the pinned MathQA dataset script}"
EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT:-1}"
DRY_RUN="${DRY_RUN:-0}"
HEAPR_VENV="${HEAPR_VENV:-/workspace/venvs/heapr-paper-v3}"
INSTALL_HEAPR_DEPS="${INSTALL_HEAPR_DEPS:-0}"
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY:-}"
LM_EVAL_INSTALL_SPEC="${LM_EVAL_INSTALL_SPEC:-}"

if [ -e "${RUN_DIR}" ]; then echo "[heapr] ERROR: refusing to overwrite ${RUN_DIR}"; exit 1; fi
if [ "${DRY_RUN}" = "1" ]; then
  echo "[heapr] official_repo=https://github.com/LLIKKE/HEAPr.git commit=${HEAPR_COMMIT}"
  echo "[heapr] model=${MODEL} ratio=${RATIO} calibration=wiki samples=${CALI_NSAMPLES} seed=${SEED}"
  echo "[heapr] tasks=openbookqa arc_easy winogrande hellaswag arc_challenge piqa mathqa batch=${BATCH_SIZE}"
  echo "[heapr] export_physical_checkpoint=${EXPORT_CHECKPOINT}"
  echo "[heapr] NOTE: HEAPr atomic-expert pruning is not identical to shared packed-channel width pruning."
  exit 0
fi
if [ -e "${WORK_DIR}" ]; then echo "[heapr] ERROR: refusing existing WORK_DIR=${WORK_DIR}"; exit 1; fi
git clone https://github.com/LLIKKE/HEAPr.git "${WORK_DIR}"
git -C "${WORK_DIR}" checkout --detach "${HEAPR_COMMIT}"
ACTUAL_COMMIT="$(git -C "${WORK_DIR}" rev-parse HEAD)"
if [ "${ACTUAL_COMMIT}" != "${HEAPR_COMMIT}" ]; then
  echo "[heapr] ERROR: commit mismatch"; exit 1
fi
if [ "${INSTALL_HEAPR_DEPS}" = "1" ]; then
  if [ -e "${HEAPR_VENV}" ]; then
    echo "[heapr] ERROR: refusing existing HEAPR_VENV=${HEAPR_VENV}"; exit 1
  fi
  python3 -m venv "${HEAPR_VENV}"
  "${HEAPR_VENV}/bin/python" -m pip install -r "${WORK_DIR}/requirements.txt"
  if [ -n "${LM_EVAL_INSTALL_SPEC}" ]; then
    "${HEAPR_VENV}/bin/python" -m pip install --upgrade "${LM_EVAL_INSTALL_SPEC}"
  fi
fi
if [ ! -f "${HEAPR_VENV}/bin/activate" ]; then
  echo "[heapr] ERROR: HEAPr environment missing: ${HEAPR_VENV}"; exit 1
fi
source "${HEAPR_VENV}/bin/activate"
if [ -z "${LM_EVAL_IDENTITY}" ]; then
  echo "[heapr] ERROR: set LM_EVAL_IDENTITY to the same pinned harness identity as the primary evaluation"; exit 1
fi
ACTUAL_LM_EVAL="$(python3 - <<'PY'
from scripts.run_paper_v3_lm_eval import harness_identity
x = harness_identity(); print(x['git_revision'] or x['package_version'])
PY
)"
if [ "${ACTUAL_LM_EVAL}" != "${LM_EVAL_IDENTITY}" ]; then
  echo "[heapr] ERROR: lm-eval identity mismatch: ${ACTUAL_LM_EVAL}"; exit 1
fi
mkdir -p "${RUN_DIR}"
python3 scripts/patch_heapr_matched_reporting.py \
  --repo-dir "${WORK_DIR}" --patch-record "${RUN_DIR}/reporting_patch.json"
export HEAPR_MATCHED_BATCH_SIZE="${BATCH_SIZE}"
export HEAPR_MATCHED_SEED="${SEED}"
export HEAPR_MATCHED_RESULTS_JSON="$(realpath "${RUN_DIR}")/lm_eval_results.json"
export HEAPR_MATCHED_TRUST_DATASET_CODE="${TRUST_DATASET_CODE}"
export HEAPR_MATCHED_CHECKPOINT_DIR="$(realpath "${RUN_DIR}")/physical_checkpoint"
export HEAPR_MATCHED_EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT}"
python3 scripts/run_with_gpu_resource_monitor.py \
  --output "${RUN_DIR}/construction_cost.json" -- \
  python3 "${WORK_DIR}/main.py" --model_path "${MODEL}" \
  --compress_ratio "${RATIO}" --cali_data wiki \
  --cali_nsamples "${CALI_NSAMPLES}" --cali_batch_size 8 \
  --eval_batch_size "${BATCH_SIZE}" --seed "${SEED}" --zero_shot \
  --tasks openbookqa arc_easy winogrande hellaswag arc_challenge piqa mathqa \
  --log_dir "${RUN_DIR}/heapr_logs" 2>&1 | tee "${RUN_DIR}/run.log"
python3 scripts/record_heapr_protocol.py \
  --run-dir "${RUN_DIR}" --repo-dir "${WORK_DIR}" --commit "${HEAPR_COMMIT}" \
  --model "${MODEL}" --requested-ratio "${RATIO}" \
  --calibration-samples "${CALI_NSAMPLES}" --seed "${SEED}" \
  --lm-eval-identity "${LM_EVAL_IDENTITY}" \
  --lm-eval-results "${RUN_DIR}/lm_eval_results.json" \
  --reporting-patch "${RUN_DIR}/reporting_patch.json" \
  --checkpoint-dir "${RUN_DIR}/physical_checkpoint"
