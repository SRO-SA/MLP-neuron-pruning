#!/usr/bin/env bash
# Recompute compact bound-score diagnostics, then compare the two saved plans.
# No pruning and no PPL evaluation occur in this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/moe_selector_baselines/20260818_222729}"
RESULTS_BASE="${RESULTS_BASE:-results/moe_plan_diagnostics}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_BASE}/${RUN_ID}_target2_plan_comparison}"
SCORE_DIR="${OUTPUT_DIR}/score_comparison"
REPORT_DIR="${OUTPUT_DIR}/plan_comparison"
SOURCE_CONFIG="${SOURCE_RUN_DIR}/rmsnorm_ellipsoid_bound_target2/run_config.yaml"
TEMP_CONFIG="${OUTPUT_DIR}/score_comparison_config.yaml"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${SOURCE_CONFIG}" ]; then
    echo "[plan-compare] ERROR: source config not found: ${SOURCE_CONFIG}"
    exit 1
fi
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

echo "[plan-compare] Source run : ${SOURCE_RUN_DIR}"
echo "[plan-compare] Output dir : ${OUTPUT_DIR}"
echo "[plan-compare] Mode       : scores + plan comparison only; no pruning/PPL"
if [ "${DRY_RUN}" = "1" ]; then
    echo "[plan-compare] WOULD create an isolated score-comparison config."
    echo "[plan-compare] WOULD run --moe-score-comparison-only."
    echo "[plan-compare] WOULD compare both target-2 plan JSON files."
    exit 0
fi

mkdir -p "${SCORE_DIR}" "${REPORT_DIR}"
python3 - "${SOURCE_CONFIG}" "${SCORE_DIR}" "${TEMP_CONFIG}" <<'PY'
import sys, yaml
source, output_dir, destination = sys.argv[1:4]
with open(source) as handle:
    cfg = yaml.safe_load(handle)
cfg["output_dir"] = output_dir
cfg["moe_score_comparison_only"] = True
cfg["moe_selection_dry_run"] = False
cfg["moe_fixed_allocation_plan"] = None
cfg["moe_fixed_allocation_selector"] = None
cfg["load_pruning_plan"] = None
cfg["save_pruning_plan"] = False
with open(destination, "w") as handle:
    yaml.safe_dump(cfg, handle, default_flow_style=False, sort_keys=False)
print("[plan-compare] isolated config:", destination)
PY

python3 run_experiment.py \
    --config "${TEMP_CONFIG}" \
    --moe-score-comparison-only \
    2>&1 | tee "${OUTPUT_DIR}/score_comparison.log"

score_json="$(find "${SCORE_DIR}" -maxdepth 1 \
    -name 'moe_bound_comparison_*.json' -print -quit)"
if [ -z "${score_json}" ]; then
    echo "[plan-compare] ERROR: score-comparison JSON was not written."
    exit 1
fi
python3 scripts/compare_moe_pruning_plans.py \
    --run-dir "${SOURCE_RUN_DIR}" \
    --score-comparison-json "${score_json}" \
    --output-dir "${REPORT_DIR}"

echo "[plan-compare] OK: no pruning or PPL evaluation was run."
echo "[plan-compare] Report directory: ${REPORT_DIR}"
