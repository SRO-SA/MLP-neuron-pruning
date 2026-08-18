#!/usr/bin/env bash
# Run only the two target-2 fixed-layer-allocation diagnostic experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/moe_selector_baselines/20260818_222729}"
N_EVAL="${N_EVAL:-128}"
DRY_RUN="${DRY_RUN:-0}"
RESULTS_BASE="${RESULTS_BASE:-results/moe_plan_replay}"
CONFIG_DIR="${CONFIG_DIR:-configs/moe_plan_replay}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${RESULTS_BASE}/${RUN_ID}_n${N_EVAL}}"

if [ "${N_EVAL}" != "128" ] && [ "${N_EVAL}" != "512" ]; then
    echo "[replay] ERROR: N_EVAL must be 128 or 512."
    exit 1
fi
if [ ! -d "${SOURCE_RUN_DIR}" ]; then
    echo "[replay] ERROR: source run directory not found: ${SOURCE_RUN_DIR}"
    exit 1
fi
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

echo "[replay] ======================================================="
echo "[replay] Fixed-layer-allocation target-2 diagnostic"
echo "[replay] Source run : ${SOURCE_RUN_DIR}"
echo "[replay] N_EVAL    : ${N_EVAL}"
echo "[replay] Run dir   : ${RUN_DIR}"
echo "[replay] Experiments: C and D only"
echo "[replay] ======================================================="

_generate_args=(
    --source-run-dir "${SOURCE_RUN_DIR}"
    --results-dir "${RUN_DIR}"
    --config-dir "${CONFIG_DIR}"
    --n-eval "${N_EVAL}"
    --overwrite
)
if [ "${DRY_RUN}" = "1" ]; then
    python3 scripts/generate_moe_plan_replay_configs.py \
        "${_generate_args[@]}" --dry-run
    echo "[replay] DRY_RUN complete; no configs, pruning, or PPL were run."
    exit 0
fi

mkdir -p "${RUN_DIR}"
python3 scripts/generate_moe_plan_replay_configs.py "${_generate_args[@]}"

labels=(
    original_allocation_ellipsoid_ranking
    ellipsoid_allocation_original_ranking
)
source_experiments=(
    rmsnorm_bound_target2
    rmsnorm_ellipsoid_bound_target2
)
source_selectors=(
    rmsnorm_bound
    rmsnorm_ellipsoid_bound
)
alternate_selectors=(
    rmsnorm_ellipsoid_bound
    rmsnorm_bound
)
expected_totals=(832 768)

for i in 0 1; do
    label="${labels[$i]}"
    cfg="${CONFIG_DIR}/${label}_n${N_EVAL}.yaml"
    log="${RUN_DIR}/${label}.log"
    echo ""
    echo "[replay] --- ${label} ---"
    echo "[replay] fixed_allocation=${source_selectors[$i]} channel_selector=${alternate_selectors[$i]}"
    python3 run_experiment.py --config "${cfg}" --moe-target-pruning 2>&1 | tee "${log}"

    source_plan="$(find "${SOURCE_RUN_DIR}/${source_experiments[$i]}/pruning_plans" \
        -maxdepth 1 -name '*.json' -print -quit)"
    derived_plan="$(find "${RUN_DIR}/${label}/pruning_plans" \
        -maxdepth 1 -name '*fixedalloc*.json' -print -quit)"
    if [ -z "${source_plan}" ] || [ -z "${derived_plan}" ]; then
        echo "[replay] ERROR: source or derived plan is missing for ${label}."
        exit 1
    fi
    python3 scripts/validate_moe_plan_replay.py \
        --source "${source_plan}" \
        --derived "${derived_plan}" \
        --source-selector "${source_selectors[$i]}" \
        --alternate-selector "${alternate_selectors[$i]}" \
        --expected-total "${expected_totals[$i]}"

    result_csv="$(find "${RUN_DIR}/${label}" -maxdepth 1 \
        -name 'moe_target_pruning_*.csv' ! -name '*_per_layer.csv' -print -quit)"
    if [ -z "${result_csv}" ]; then
        echo "[replay] ERROR: no result CSV written for ${label}."
        exit 1
    fi
    python3 - "${result_csv}" "${source_selectors[$i]}" "${alternate_selectors[$i]}" <<'PY'
import csv, sys
path, source_selector, alternate_selector = sys.argv[1:4]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1:
    raise SystemExit(f"[replay] ERROR: expected one result row in {path}, found {len(rows)}")
row = rows[0]
if row.get("source_allocation_selector") != source_selector:
    raise SystemExit("[replay] ERROR: result source allocation selector mismatch")
if row.get("alternate_channel_selector") != alternate_selector:
    raise SystemExit("[replay] ERROR: result alternate channel selector mismatch")
if row.get("forward_check", "").lower() not in ("true", "1"):
    raise SystemExit("[replay] ERROR: forward/shape check did not pass")
print(
    "[replay] RESULT: " + row.get("experiment_label", "")
    + " actual_pct=" + row.get("actual_pct", "")
    + " layer_channels=" + row.get("selected_layer_channels", "")
    + " ppl_base=" + row.get("baseline_ppl", "")
    + " ppl_pruned=" + row.get("compressed_ppl", "")
    + " ppl_rel_inc_pct=" + row.get("relative_delta_pct", "")
)
PY
done

echo ""
echo "[replay] OK: both mixed experiments passed plan, allocation, and shape validation."
echo "[replay] Results: ${RUN_DIR}"
