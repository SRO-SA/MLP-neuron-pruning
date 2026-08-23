#!/usr/bin/env bash
# Validate frozen target-6 plans, collect all-expert scores, and build frontier.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MILESTONE_ROOT="${MILESTONE_ROOT:?set a new milestone directory}"
T6_PRIMARY_RUN="${T6_PRIMARY_RUN:-results/moe_allocation_ranking/target6_rmsnorm_primary_n1024_v1}"
T6_DOWN_RUN="${T6_DOWN_RUN:-results/moe_allocation_ranking/target6_rmsnorm_downnorm_rank_n1024_20260821_221602}"
FROZEN_CHECKPOINT_ROOT="${FROZEN_CHECKPOINT_ROOT:-/paper_v3_checkpoints/20260820_060635}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
DRY_RUN="${DRY_RUN:-0}"

if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi
if [ "${DRY_RUN}" != "1" ] && [ -e "${MILESTONE_ROOT}" ]; then
  echo "[hybrid-prepare] ERROR: refusing to overwrite ${MILESTONE_ROOT}"
  exit 1
fi

find_one() {
  local pattern="$1"
  mapfile -t matches < <(compgen -G "${pattern}" || true)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "[hybrid-prepare] ERROR: expected one match for ${pattern}; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s' "${matches[0]}"
}

ELLIPSOID_PLAN="$(find_one "${T6_PRIMARY_RUN}/rmsnorm_alloc__ellipsoid_rank/pruning_plans/*.json")"
ACTIVATION_PLAN="$(find_one "${T6_PRIMARY_RUN}/rmsnorm_alloc__activation_rank/pruning_plans/*.json")"
RMSNORM_PLAN="$(find_one "${T6_PRIMARY_RUN}/rmsnorm_alloc__rmsnorm_rank/pruning_plans/*.json")"
DOWN_PLAN="$(find_one "${T6_DOWN_RUN}/rmsnorm_alloc__downnorm_rank/pruning_plans/*.json")"

ELLIPSOID_RESULT="$(find_one "${T6_PRIMARY_RUN}/rmsnorm_alloc__ellipsoid_rank/moe_target_pruning_*.json")"
ACTIVATION_RESULT="$(find_one "${T6_PRIMARY_RUN}/rmsnorm_alloc__activation_rank/moe_target_pruning_*.json")"
RMSNORM_RESULT="$(find_one "${T6_PRIMARY_RUN}/rmsnorm_alloc__rmsnorm_rank/moe_target_pruning_*.json")"
DOWN_RESULT="$(find_one "${T6_DOWN_RUN}/rmsnorm_alloc__downnorm_rank/moe_target_pruning_*.json")"

ELLIPSOID_CHECKPOINT="${FROZEN_CHECKPOINT_ROOT}/rmsnorm_alloc__ellipsoid_rank__p95__target6/checkpoint_verification.json"
ACTIVATION_CHECKPOINT="${FROZEN_CHECKPOINT_ROOT}/rmsnorm_alloc__activation_score_rank__p95__target6/checkpoint_verification.json"
RMSNORM_CHECKPOINT="${FROZEN_CHECKPOINT_ROOT}/rmsnorm_alloc__rmsnorm_bound_rank__p95__target6/checkpoint_verification.json"
DOWN_CHECKPOINT="${FROZEN_CHECKPOINT_ROOT}/rmsnorm_alloc__downnorm_rank__p95__target6/checkpoint_verification.json"
for path in "${ELLIPSOID_CHECKPOINT}" "${ACTIVATION_CHECKPOINT}" \
            "${RMSNORM_CHECKPOINT}" "${DOWN_CHECKPOINT}"; do
  [ -f "${path}" ] || { echo "[hybrid-prepare] ERROR: missing ${path}"; exit 1; }
done

VALIDATION="${MILESTONE_ROOT}/matched_plan_validation.json"
validation_args=(
  --plan "ellipsoid=${ELLIPSOID_PLAN}"
  --plan "down_norm=${DOWN_PLAN}"
  --plan "activation=${ACTIVATION_PLAN}"
  --plan "rmsnorm_bound=${RMSNORM_PLAN}"
  --result "ellipsoid=${ELLIPSOID_RESULT}"
  --result "down_norm=${DOWN_RESULT}"
  --result "activation=${ACTIVATION_RESULT}"
  --result "rmsnorm_bound=${RMSNORM_RESULT}"
  --checkpoint-verification "ellipsoid=${ELLIPSOID_CHECKPOINT}"
  --checkpoint-verification "down_norm=${DOWN_CHECKPOINT}"
  --checkpoint-verification "activation=${ACTIVATION_CHECKPOINT}"
  --checkpoint-verification "rmsnorm_bound=${RMSNORM_CHECKPOINT}"
  --expected-total 2288 --output "${VALIDATION}"
)
if [ "${DRY_RUN}" = "1" ]; then
  python3 scripts/validate_target6_matched_plans.py "${validation_args[@]}" --dry-run
  python3 scripts/collect_moe_certificate_scores.py \
    --model Qwen/Qwen3-30B-A3B --output-dir "${MILESTONE_ROOT}/scores" --dry-run
  echo "[hybrid-prepare] WOULD BUILD frontier in ${MILESTONE_ROOT}/certification_frontier"
  exit 0
fi

mkdir -p "${MILESTONE_ROOT}"
python3 scripts/validate_target6_matched_plans.py "${validation_args[@]}"
python3 -c "import torch; assert torch.cuda.is_available(); print('[hybrid-prepare] CUDA OK', torch.__version__)"
python3 scripts/collect_moe_certificate_scores.py \
  --model Qwen/Qwen3-30B-A3B --output-dir "${MILESTONE_ROOT}/scores"
python3 scripts/build_moe_certified_hybrid_frontier.py \
  --ellipsoid-plan "${ELLIPSOID_PLAN}" \
  --down-norm-plan "${DOWN_PLAN}" \
  --score-bundle "${MILESTONE_ROOT}/scores/all_expert_certificate_scores.npz" \
  --score-manifest "${MILESTONE_ROOT}/scores/all_expert_certificate_scores.json" \
  --matched-validation "${VALIDATION}" \
  --output-dir "${MILESTONE_ROOT}/certification_frontier" \
  --seed 42
echo "[hybrid-prepare] COMPLETE ${MILESTONE_ROOT}"
