#!/usr/bin/env bash
# Each matrix cell is one fresh Python process and one freshly loaded model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="${PROFILE:-replicate2}"
# Full 2/4/6/8 selector baseline run.  The earlier 20260818_190239 run is
# partial and does not contain the down_norm allocation plans.
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-results/moe_selector_baselines/20260818_203025}"
TARGET2_RMSNORM_RUN_DIR="${TARGET2_RMSNORM_RUN_DIR:-results/moe_selector_baselines/20260818_222729}"
RESULTS_BASE="${RESULTS_BASE:-results/moe_allocation_ranking}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_RUN_DIR="${ALLOW_EXISTING_RUN_DIR:-0}"
REFERENCE_HYBRID_DIR="${REFERENCE_HYBRID_DIR:-results/moe_plan_replay/20260818_233748_n512/original_allocation_ellipsoid_ranking}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

case "${PROFILE}" in
    replicate2)
        N_EVAL="${N_EVAL:-512}"
        EVAL_DATASETS="${EVAL_DATASETS:-wikitext2}"
        ;;
    target2_extended|target2_paper_freeze|target4|target4_rankings|target4_exact_budget|target4_aggregation_rmsnorm|target4_aggregation_downnorm|target6_rmsnorm_primary|target6_rmsnorm_downnorm_ranking_only|target6_downnorm_primary|target6_exact_budget|target6_aggregation_rmsnorm|target6_aggregation_limited|target8_rmsnorm_primary|pure_downnorm_rmsnorm_allocation_curve)
        N_EVAL="${N_EVAL:-1024}"
        EVAL_DATASETS="${EVAL_DATASETS:-wikitext2,c4}"
        ;;
    *)
        echo "[alloc-rank] ERROR: unsupported PROFILE=${PROFILE}"
        exit 1
        ;;
esac

RUN_DIR="${RUN_DIR:-${RESULTS_BASE}/${RUN_ID}_${PROFILE}_n${N_EVAL}}"
CONFIG_DIR="${RUN_DIR}/configs"
MANIFEST="${CONFIG_DIR}/matrix_manifest.json"

if [ "${DRY_RUN}" != "1" ] && [ -e "${RUN_DIR}" ] && \
   [ "${ALLOW_EXISTING_RUN_DIR}" != "1" ]; then
    echo "[alloc-rank] ERROR: refusing to overwrite existing RUN_DIR=${RUN_DIR}"
    echo "[alloc-rank] Use a new RUN_DIR (recommended), or set ALLOW_EXISTING_RUN_DIR=1 to resume intentionally."
    exit 1
fi

if [ ! -d "${BASELINE_RUN_DIR}" ]; then
    echo "[alloc-rank] ERROR: baseline run not found: ${BASELINE_RUN_DIR}"
    exit 1
fi
if [ ! -d "${TARGET2_RMSNORM_RUN_DIR}" ]; then
    echo "[alloc-rank] ERROR: target-2 RMSNorm run not found: ${TARGET2_RMSNORM_RUN_DIR}"
    exit 1
fi
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi
echo "[alloc-rank] ======================================================="
echo "[alloc-rank] Profile       : ${PROFILE}"
echo "[alloc-rank] Evaluation    : ${EVAL_DATASETS}, n_eval=${N_EVAL}, max_seq=512"
echo "[alloc-rank] Results       : ${RUN_DIR}"
echo "[alloc-rank] Process policy: one new Python process/model load per cell"
echo "[alloc-rank] ======================================================="

generate_args=(
    --profile "${PROFILE}"
    --baseline-run-dir "${BASELINE_RUN_DIR}"
    --target2-rmsnorm-run-dir "${TARGET2_RMSNORM_RUN_DIR}"
    --results-dir "${RUN_DIR}"
    --config-dir "${CONFIG_DIR}"
    --n-eval "${N_EVAL}"
    --eval-datasets "${EVAL_DATASETS}"
    --overwrite
)
if [ "${DRY_RUN}" = "1" ]; then
    python3 scripts/generate_moe_allocation_ranking_configs.py \
        "${generate_args[@]}" --dry-run
    echo "[alloc-rank] DRY_RUN complete; no model was loaded."
    exit 0
fi

python3 -c "import torch; assert torch.cuda.is_available(); print('[alloc-rank] torch', torch.__version__, 'CUDA OK')"
mkdir -p "${RUN_DIR}"
python3 scripts/generate_moe_allocation_ranking_configs.py "${generate_args[@]}"

while IFS=$'\t' read -r name config_path output_dir allocation ranking allocation_plan; do
    mkdir -p "${output_dir}"
    log="${RUN_DIR}/${name}.log"
    echo ""
    echo "[alloc-rank] START fresh process: ${name}"
    echo "[alloc-rank] allocation_source=${allocation} ranking_source=${ranking}"
    python3 run_experiment.py --config "${config_path}" --moe-target-pruning \
        2>&1 | tee "${log}"

    derived_plan="$(find "${output_dir}/pruning_plans" -maxdepth 1 \
        -name '*.json' -print -quit)"
    if [ -z "${derived_plan}" ]; then
        echo "[alloc-rank] ERROR: derived plan missing for ${name}"
        exit 1
    fi
    python3 scripts/validate_moe_allocation_ranking.py \
        --allocation-plan "${allocation_plan}" \
        --derived-plan "${derived_plan}" \
        --allocation-source "${allocation}" \
        --ranking-source "${ranking}" \
        --experiment-name "${name}"
    echo "[alloc-rank] FINISH process: ${name}"
done < <(
    python3 - "${MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    for row in json.load(handle):
        print("\t".join([
            row["experiment_name"], row["config_path"], row["output_dir"],
            row["allocation_source"], row["ranking_source"],
            row["allocation_plan"],
        ]))
PY
)

python3 scripts/summarize_moe_allocation_ranking.py \
    --run-dir "${RUN_DIR}" --manifest "${MANIFEST}"
if [ "${PROFILE}" = "target6_aggregation_limited" ]; then
    python3 scripts/summarize_moe_aggregation_frontier.py \
        --run-dir "${RUN_DIR}" \
        --output-dir "${RUN_DIR}/aggregation_frontier_tables"
fi
if [ "${PROFILE}" = "replicate2" ]; then
    if [ ! -d "${REFERENCE_HYBRID_DIR}" ]; then
        echo "[alloc-rank] ERROR: reference hybrid not found: ${REFERENCE_HYBRID_DIR}"
        exit 1
    fi
    python3 scripts/compare_moe_hybrid_replication.py \
        --reference-dir "${REFERENCE_HYBRID_DIR}" \
        --replication-dir "${RUN_DIR}/rmsnorm_alloc__ellipsoid_rank"
fi
echo "[alloc-rank] COMPLETE: ${RUN_DIR}"
