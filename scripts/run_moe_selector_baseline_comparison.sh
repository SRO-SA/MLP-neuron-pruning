#!/usr/bin/env bash
# run_moe_selector_baseline_comparison.sh
#
# Selector baseline comparison: 4 selectors x 4 targets x 2 datasets = 32 runs.
# All configs use pure_delete pruning + moe_budget_mode=uniform so that
# actual_pct is identical across selectors (fair comparison).
#
# Selectors:
#   rmsnorm_bound    -- weight-only RMSNorm-bounded SwiGLU score  (proposed)
#   down_norm        -- L2 norm of each down_proj column           (simple baseline)
#   activation_score -- activation x down-column-norm             (needs calib data)
#   random           -- uniform random                             (random baseline)
#
# Targets:  2%, 4%, 6%, 8%
# Datasets: wikitext2, c4
# n_eval:   512
#
# Usage:
#   bash scripts/run_moe_selector_baseline_comparison.sh           # full 32-run
#   SMOKE=1   bash scripts/run_moe_selector_baseline_comparison.sh # 2 configs (rmsnorm_bound + random, target2 x wikitext2)
#   DRY_RUN=1 bash scripts/run_moe_selector_baseline_comparison.sh # list 32, do not run
#
# Env overrides:
#   SMOKE=1             Run 2 configs: rmsnorm_bound + random, target2 x wikitext2
#   DRY_RUN=1           Print planned runs, skip all execution
#   CONTINUE_ON_FAIL=1  Keep going past failures (default: stop)
#   CONFIG_DIR=...      Config dir (default: configs/moe_selector_baseline)
#   RESULTS_DIR=...     Results dir (default: results)
#   VENV=...            Virtualenv path (default: /workspace/venvs/qwen-pruning)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Config
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/moe_selector_baseline}"
RESULTS_DIR="${RESULTS_DIR:-results}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
EXPECTED_TOTAL=32   # 4 selectors x 4 targets x 2 datasets

# Sweep ID
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_DIR}/moe_selector_baseline_runs/${SWEEP_ID}"
RUN_CSV_DIR="${RUN_DIR}/csvs"
RUN_LOG_DIR="${RUN_DIR}/logs"
RUN_JSON_DIR="${RUN_DIR}/jsons"
MANIFEST="${RUN_DIR}/sweep_manifest.json"

echo "[sel] Sweep ID:  ${SWEEP_ID}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "[sel] DRY_RUN=1: will list planned runs without executing."
else
    echo "[sel] Run dir:   ${RUN_DIR}"
    mkdir -p "${RUN_CSV_DIR}" "${RUN_LOG_DIR}" "${RUN_JSON_DIR}"
fi

# Activate virtualenv
if [ -f "${VENV}/bin/activate" ]; then
    echo "[sel] Activating virtualenv: ${VENV}"
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "[sel] No virtualenv at ${VENV}, using system Python"
fi

# Environment check (skip for dry-run)
if [ "${DRY_RUN}" != "1" ] && [ -f "scripts/check_env.py" ]; then
    echo "[sel] Running environment check..."
    python3 scripts/check_env.py --strict || {
        echo "[sel] ERROR: environment check failed. Aborting."
        exit 1
    }
fi

# Generate configs
echo "[sel] Generating configs in ${CONFIG_DIR}/ ..."
if [ "${DRY_RUN}" = "1" ]; then
    python3 scripts/generate_moe_selector_baseline_configs.py \
        --out-dir "${CONFIG_DIR}" --dry-run
else
    python3 scripts/generate_moe_selector_baseline_configs.py \
        --out-dir "${CONFIG_DIR}"
fi

# Validate a YAML config (required keys for --moe-target-pruning mode)
validate_config() {
    local cfg_path="$1"
    python3 - "${cfg_path}" << 'PYEOF'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
REQUIRED = [
    "scaling_models", "target_pruning_percents", "scaling_methods",
    "eval_datasets", "reconstruction_eval_samples", "moe_selector",
    "moe_budget_mode",
]
missing = [k for k in REQUIRED if k not in cfg]
if missing:
    print(f"CONFIG VALIDATION FAILED: {path}")
    print(f"  Missing required keys: {missing}")
    sys.exit(1)
if cfg.get("moe_budget_mode") != "uniform":
    print(f"CONFIG VALIDATION FAILED: {path}")
    print(f"  moe_budget_mode must be 'uniform' for selector comparison, got: {cfg.get('moe_budget_mode')!r}")
    sys.exit(1)
PYEOF
}

# DRY_RUN: list planned runs and exit
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[sel] Planned runs (${EXPECTED_TOTAL} total):"
    n=0
    for selector in rmsnorm_bound down_norm activation_score random; do
        for target in 2 4 6 8; do
            for dataset in wikitext2 c4; do
                n=$(( n + 1 ))
                printf "  %2d. selector=%-18s  target=%s%%  dataset=%s\n" \
                    "${n}" "${selector}" "${target}" "${dataset}"
            done
        done
    done
    echo ""
    echo "[sel] Budget mode: uniform (same per-layer channel count for all selectors)"
    echo "[sel] DRY_RUN complete. To run: bash scripts/run_moe_selector_baseline_comparison.sh"
    exit 0
fi

# Collect all configs in explicit selector order
declare -a ALL_CONFIGS=()
for selector in rmsnorm_bound down_norm activation_score random; do
    for target in 2 4 6 8; do
        for dataset in wikitext2 c4; do
            cfg="${CONFIG_DIR}/qwen3_30b_a3b_${dataset}_n512_target${target}_sel_${selector}.yaml"
            if [ -f "${cfg}" ]; then
                ALL_CONFIGS+=("${cfg}")
            else
                echo "[sel] WARNING: expected config not found: ${cfg}"
            fi
        done
    done
done

TOTAL_FOUND=${#ALL_CONFIGS[@]}
echo "[sel] Found ${TOTAL_FOUND} config(s)."

if [ ${TOTAL_FOUND} -eq 0 ]; then
    echo "[sel] ERROR: No configs found in ${CONFIG_DIR}/. Aborting."
    exit 1
fi

# Assert full matrix for non-smoke runs
if [ "${SMOKE}" != "1" ] && [ ${TOTAL_FOUND} -ne ${EXPECTED_TOTAL} ]; then
    echo "[sel] ERROR: Expected ${EXPECTED_TOTAL} configs, found ${TOTAL_FOUND}."
    echo "[sel]   Regenerate: python3 scripts/generate_moe_selector_baseline_configs.py"
    exit 1
fi

# Validate all configs before any GPU work
echo "[sel] Validating all ${TOTAL_FOUND} configs..."
for cfg_path in "${ALL_CONFIGS[@]}"; do
    validate_config "${cfg_path}" || exit 1
done
echo "[sel] All configs valid (moe_budget_mode=uniform confirmed)."

# SMOKE mode: 2 configs — rmsnorm_bound + random, target2 x wikitext2
# (Chosen as the two extreme-opposite approaches for a fast sanity check)
if [ "${SMOKE}" = "1" ]; then
    echo "[sel] SMOKE=1: running 2 configs (rmsnorm_bound + random, target2 x wikitext2)."
    declare -a RUN_CONFIGS=()
    for selector in rmsnorm_bound random; do
        cfg="${CONFIG_DIR}/qwen3_30b_a3b_wikitext2_n512_target2_sel_${selector}.yaml"
        if [ -f "${cfg}" ]; then
            RUN_CONFIGS+=("${cfg}")
            echo "[sel]   + $(basename "${cfg}")"
        else
            echo "[sel]   WARNING: smoke config not found: ${cfg}"
        fi
    done
    echo "[sel] ${TOTAL_FOUND} configs generated; running ${#RUN_CONFIGS[@]} (smoke)."
else
    RUN_CONFIGS=("${ALL_CONFIGS[@]}")
fi

N_RUNS=${#RUN_CONFIGS[@]}
echo "[sel] Will run ${N_RUNS} config(s)."

# Write initial manifest
python3 - "${MANIFEST}" "${SWEEP_ID}" "${SMOKE}" "${N_RUNS}" << 'PYEOF'
import json, sys, datetime
manifest_path, sweep_id, smoke, n_runs = sys.argv[1:5]
manifest = {
    "sweep_id": sweep_id,
    "benchmark": "moe_selector_baseline_comparison",
    "smoke": smoke == "1",
    "budget_mode": "uniform",
    "n_planned": int(n_runs),
    "started_at": datetime.datetime.now().isoformat(),
    "runs": [],
    "csv_files": [],
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
PYEOF

# Helper: check log file for error indicators
log_has_error() {
    local logfile="$1"
    grep -qE '(\*\*\* ERROR|ERROR Failed|UnboundLocalError|KeyError|AttributeError|Traceback \(most recent)' \
        "${logfile}" 2>/dev/null
}

# Run each config
FAILED=0
SUCCEEDED=0
declare -a MANIFEST_CSV_FILES=()

for cfg_path in "${RUN_CONFIGS[@]}"; do
    cfg_name="$(basename "${cfg_path}" .yaml)"
    run_log="${RUN_LOG_DIR}/${cfg_name}.log"

    echo ""
    echo "================================================================"
    echo "[sel] Config: ${cfg_name}"
    echo "[sel] Log:    ${run_log}"
    echo "================================================================"

    BEFORE_MAIN_CSV=( $(ls "${RESULTS_DIR}"/moe_target_pruning_[0-9]*.csv 2>/dev/null | \
                         grep -v '_per_layer' || true) )
    BEFORE_MAIN_JSON=( $(ls "${RESULTS_DIR}"/moe_target_pruning_[0-9]*.json 2>/dev/null || true) )

    t_start=$(date +%s)

    set +e
    python3 run_experiment.py \
        --config "${cfg_path}" \
        --moe-target-pruning \
        2>&1 | tee "${run_log}"
    exit_code=${PIPESTATUS[0]}
    set -e

    t_end=$(date +%s)
    elapsed=$(( t_end - t_start ))

    declare -a NEW_MAIN_CSVS=()
    declare -a NEW_MAIN_JSONS=()
    for f in $(ls "${RESULTS_DIR}"/moe_target_pruning_[0-9]*.csv 2>/dev/null | \
               grep -v '_per_layer' || true); do
        if [[ ! " ${BEFORE_MAIN_CSV[*]:-} " =~ " ${f} " ]]; then
            NEW_MAIN_CSVS+=("$f")
        fi
    done
    for f in $(ls "${RESULTS_DIR}"/moe_target_pruning_[0-9]*.json 2>/dev/null || true); do
        if [[ ! " ${BEFORE_MAIN_JSON[*]:-} " =~ " ${f} " ]]; then
            NEW_MAIN_JSONS+=("$f")
        fi
    done

    # True success: exit 0 + no error in log + at least one main CSV created
    run_ok=1
    if [ ${exit_code} -ne 0 ]; then
        echo "[sel] FAIL: exit code ${exit_code}"
        run_ok=0
    fi
    if log_has_error "${run_log}"; then
        echo "[sel] FAIL: error indicators found in log"
        run_ok=0
    fi
    if [ ${#NEW_MAIN_CSVS[@]} -eq 0 ]; then
        echo "[sel] FAIL: no main CSV created (per-layer CSVs alone do not count)"
        run_ok=0
    fi

    if [ ${run_ok} -eq 1 ]; then
        echo "[sel] OK ${cfg_name} done in ${elapsed}s  (main CSVs: ${#NEW_MAIN_CSVS[@]})"
        SUCCEEDED=$(( SUCCEEDED + 1 ))

        for f in "${NEW_MAIN_CSVS[@]}"; do
            cp "${f}" "${RUN_CSV_DIR}/"
            MANIFEST_CSV_FILES+=("${RUN_CSV_DIR}/$(basename "${f}")")
            echo "[sel]   csv: $(basename "${f}")"
        done
        for f in "${NEW_MAIN_JSONS[@]}"; do
            cp "${f}" "${RUN_JSON_DIR}/"
            echo "[sel]   json: $(basename "${f}")"
        done
        for f in $(ls "${RESULTS_DIR}"/moe_target_pruning_*_per_layer.csv 2>/dev/null || true); do
            fname="$(basename "${f}")"
            if [ ! -f "${RUN_CSV_DIR}/${fname}" ]; then
                cp "${f}" "${RUN_CSV_DIR}/"
            fi
        done
    else
        echo "[sel] FAILED ${cfg_name} after ${elapsed}s"
        FAILED=$(( FAILED + 1 ))

        if [ "${CONTINUE_ON_FAIL}" != "1" ]; then
            echo "[sel] Aborting. Set CONTINUE_ON_FAIL=1 to continue past failures."
            break
        fi
    fi
done

# Update final manifest
export MANIFEST_CSV_LIST
MANIFEST_CSV_LIST="$(IFS='|'; echo "${MANIFEST_CSV_FILES[*]:-}")"

python3 - "${MANIFEST}" "${SWEEP_ID}" "${SUCCEEDED}" "${FAILED}" << PYEOF
import json, sys, os, datetime
manifest_path, sweep_id, succeeded, failed = sys.argv[1:5]
with open(manifest_path) as f:
    manifest = json.load(f)
manifest["succeeded"] = int(succeeded)
manifest["failed"] = int(failed)
manifest["finished_at"] = datetime.datetime.now().isoformat()
csv_list = os.environ.get("MANIFEST_CSV_LIST", "").split("|")
manifest["csv_files"] = [p for p in csv_list if p and p.endswith(".csv") and "_per_layer" not in p]
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[sel] Manifest updated: {manifest_path}")
print(f"[sel]   csv_files ({len(manifest['csv_files'])}): {manifest['csv_files']}")
PYEOF

echo ""
echo "================================================================"
echo "[sel] SWEEP COMPLETE  (id=${SWEEP_ID})"
echo "[sel]   Succeeded: ${SUCCEEDED} / ${N_RUNS}"
echo "[sel]   Failed:    ${FAILED} / ${N_RUNS}"
echo "[sel]   Run dir:   ${RUN_DIR}"
echo "[sel]   Manifest:  ${MANIFEST}"
echo "================================================================"

if [ "${SUCCEEDED}" -eq 0 ]; then
    echo "[sel] No successful runs; skipping summarizer."
elif [ -f "scripts/summarize_moe_residual_sweep.py" ]; then
    echo "[sel] Running summarizer..."
    python3 scripts/summarize_moe_residual_sweep.py \
        --run-dir "${RUN_DIR}" \
        --out-dir "${RUN_DIR}" || true
fi

# Post-run validation for smoke mode
if [ "${SMOKE}" = "1" ] && [ "${SUCCEEDED}" -gt 0 ]; then
    echo ""
    echo "[sel] Running smoke validation..."
    python3 - "${RUN_DIR}" << 'PYEOF'
import csv, glob, os, sys

run_dir = sys.argv[1]
csv_dir = os.path.join(run_dir, "csvs")
csv_files = [
    f for f in glob.glob(os.path.join(csv_dir, "moe_target_pruning_*.csv"))
    if "_per_layer" not in f
]

if not csv_files:
    print("[validate] ERROR: no main CSVs found in run dir")
    sys.exit(1)

rows = []
for f in csv_files:
    with open(f, newline="", encoding="utf-8") as fh:
        rows.extend(list(csv.DictReader(fh)))

print(f"[validate] Checking {len(rows)} rows from {len(csv_files)} CSV file(s)...")

errors = []

# 1. All rows must have selector
for i, row in enumerate(rows):
    if not row.get("selector"):
        errors.append(f"row {i}: missing 'selector' field")

# 2. All rows must have status=ok (or empty)
for i, row in enumerate(rows):
    st = str(row.get("status", "")).lower()
    if st not in ("ok", ""):
        errors.append(f"row {i} selector={row.get('selector','?')}: status={row.get('status')!r}")

# 3. All rows must have forward_check=True
for i, row in enumerate(rows):
    fc = str(row.get("forward_check", "")).lower()
    if fc not in ("true", "1", ""):
        errors.append(f"row {i} selector={row.get('selector','?')}: forward_check={row.get('forward_check')!r}")

# 4. Within each (target_pct, dataset) group, actual_pct must be the same
from collections import defaultdict
groups = defaultdict(list)
for row in rows:
    t = row.get("target_pct") or row.get("expert_target_pct") or "?"
    d = row.get("eval_dataset") or row.get("dataset") or "?"
    groups[(t, d)].append(row)

for (t, d), grp in groups.items():
    pcts = {row.get("actual_pct", "") for row in grp if row.get("actual_pct", "")}
    if len(pcts) > 1:
        # Allow small floating point differences (< 0.01 absolute)
        try:
            float_pcts = [float(p) for p in pcts]
            spread = max(float_pcts) - min(float_pcts)
            if spread > 0.01:
                selectors = [row.get("selector", "?") for row in grp]
                errors.append(
                    f"target={t}% dataset={d}: actual_pct differs across selectors "
                    f"{pcts} (spread={spread:.4f}%) — moe_budget_mode=uniform not working"
                )
        except ValueError:
            errors.append(f"target={t}% dataset={d}: could not compare actual_pct values: {pcts}")

if errors:
    print(f"[validate] SMOKE VALIDATION FAILED ({len(errors)} error(s)):")
    for e in errors:
        print(f"  ERROR: {e}")
    sys.exit(1)
else:
    print(f"[validate] All {len(rows)} rows passed smoke validation:")
    print(f"  - all rows have 'selector'")
    print(f"  - all rows have status=ok")
    print(f"  - all rows have forward_check=True (or empty)")
    print(f"  - actual_pct consistent within each (target_pct, dataset) group")
    print("[validate] SMOKE VALIDATION PASSED")
PYEOF
    val_exit=$?
    if [ ${val_exit} -ne 0 ]; then
        echo "[sel] ERROR: Smoke validation failed. See above."
        exit 1
    fi
fi

if [ "${FAILED}" -gt 0 ]; then
    echo "[sel] WARNING: ${FAILED} run(s) failed. Logs: ${RUN_LOG_DIR}/"
    exit 1
fi
exit 0
