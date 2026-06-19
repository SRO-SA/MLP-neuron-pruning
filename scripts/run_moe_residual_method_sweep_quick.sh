#!/usr/bin/env bash
# run_moe_residual_method_sweep_quick.sh
#
# Runs the MoE residual method comparison sweep.
# Generates 40 configs (4 targets × 10 methods), runs them in order.
# Pure-delete runs first per target (saves pruning plan; others load it).
#
# Usage:
#   bash scripts/run_moe_residual_method_sweep_quick.sh
#
# Env overrides:
#   SMOKE=1             Run 4 representative configs only
#   CONTINUE_ON_FAIL=1  Keep going past failures (default: stop)
#   CONFIG_DIR=...      Config dir (default: configs/moe_residual_sweep)
#   RESULTS_DIR=...     Results dir (default: results)
#   VENV=...            Virtualenv path (default: /workspace/venvs/qwen-pruning)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/moe_residual_sweep}"
RESULTS_DIR="${RESULTS_DIR:-results}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"

# ── Sweep ID (timestamp) ──────────────────────────────────────────────────────
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_DIR}/moe_residual_sweep_runs/${SWEEP_ID}"
RUN_CSV_DIR="${RUN_DIR}/csvs"
RUN_LOG_DIR="${RUN_DIR}/logs"
MANIFEST="${RUN_DIR}/sweep_manifest.json"

echo "[sweep] Sweep ID: ${SWEEP_ID}"
echo "[sweep] Run dir:  ${RUN_DIR}"
mkdir -p "${RUN_CSV_DIR}" "${RUN_LOG_DIR}"

# ── Activate virtualenv ────────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    echo "[sweep] Activating virtualenv: ${VENV}"
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "[sweep] No virtualenv at ${VENV}, using system Python"
fi

# ── Environment check ─────────────────────────────────────────────────────────
if [ -f "scripts/check_env.py" ]; then
    echo "[sweep] Running environment check..."
    python3 scripts/check_env.py --strict || {
        echo "[sweep] ERROR: environment check failed. Aborting."
        exit 1
    }
fi

# ── Generate configs ──────────────────────────────────────────────────────────
echo "[sweep] Generating configs in ${CONFIG_DIR}/ ..."
python3 scripts/generate_moe_residual_sweep_configs.py --out-dir "${CONFIG_DIR}"

# ── Validate a YAML config (required keys for --moe-target-pruning mode) ─────
validate_config() {
    local cfg_path="$1"
    python3 - "${cfg_path}" << 'PYEOF'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
REQUIRED = [
    "scaling_models",
    "target_pruning_percents",
    "scaling_methods",
    "eval_datasets",
    "reconstruction_eval_samples",
]
missing = [k for k in REQUIRED if k not in cfg]
if missing:
    print(f"CONFIG VALIDATION FAILED: {path}")
    print(f"  Missing required keys: {missing}")
    print(f"  Present keys: {sorted(cfg.keys())}")
    sys.exit(1)
PYEOF
}

# ── Collect configs in correct order ──────────────────────────────────────────
# Targets match generator: 2 4 6 8
# pure_delete runs first per target (saves the pruning plan for others to load).
declare -a ALL_CONFIGS=()
for target in 2 4 6 8; do
    pd="${CONFIG_DIR}/qwen3_30b_a3b_wikitext2_n64_target${target}_pure_delete.yaml"
    if [ -f "${pd}" ]; then
        ALL_CONFIGS+=("${pd}")
    fi
    for f in $(ls "${CONFIG_DIR}"/qwen3_30b_a3b_wikitext2_n64_target${target}_*.yaml 2>/dev/null | sort); do
        if [ "${f}" != "${pd}" ]; then
            ALL_CONFIGS+=("${f}")
        fi
    done
done

TOTAL_FOUND=${#ALL_CONFIGS[@]}
echo "[sweep] Found ${TOTAL_FOUND} config(s)."

if [ ${TOTAL_FOUND} -eq 0 ]; then
    echo "[sweep] ERROR: No configs found in ${CONFIG_DIR}/. Aborting."
    exit 1
fi

# Assert full 4×10 matrix for non-smoke runs
if [ "${SMOKE}" != "1" ] && [ ${TOTAL_FOUND} -ne 40 ]; then
    echo "[sweep] ERROR: Expected 40 configs (4 targets × 10 methods), found ${TOTAL_FOUND}."
    echo "[sweep]   Regenerate: python3 scripts/generate_moe_residual_sweep_configs.py"
    exit 1
fi

# ── Validate all configs before any GPU work ──────────────────────────────────
echo "[sweep] Validating all ${TOTAL_FOUND} configs..."
for cfg_path in "${ALL_CONFIGS[@]}"; do
    validate_config "${cfg_path}" || exit 1
done
echo "[sweep] All configs valid."

# ── SMOKE mode: select 4 representative configs from target2 ─────────────────
if [ "${SMOKE}" = "1" ]; then
    echo "[sweep] SMOKE=1: selecting 4 representative configs from target2."
    SMOKE_PATTERNS=(
        "target2_pure_delete"
        "target2_residual_ridge_lam1e-2"
        "target2_residual_ridge_oii_lam1e-2"
        "target2_residual_nearest_merge"
    )
    declare -a RUN_CONFIGS=()
    for pattern in "${SMOKE_PATTERNS[@]}"; do
        matched=""
        for f in "${ALL_CONFIGS[@]}"; do
            if [[ "${f}" == *"${pattern}"* ]]; then
                matched="${f}"
                break
            fi
        done
        if [ -n "${matched}" ]; then
            RUN_CONFIGS+=("${matched}")
            echo "[sweep]   + $(basename "${matched}")"
        else
            echo "[sweep]   WARNING: no config matched pattern '${pattern}'"
        fi
    done
    echo "[sweep] ${TOTAL_FOUND} configs generated; running ${#RUN_CONFIGS[@]} (smoke)."
else
    RUN_CONFIGS=("${ALL_CONFIGS[@]}")
fi

N_RUNS=${#RUN_CONFIGS[@]}
echo "[sweep] Will run ${N_RUNS} config(s)."

# ── Write initial manifest ────────────────────────────────────────────────────
python3 - "${MANIFEST}" "${SWEEP_ID}" "${SMOKE}" "${N_RUNS}" << 'PYEOF'
import json, sys, datetime
manifest_path, sweep_id, smoke, n_runs = sys.argv[1:5]
manifest = {
    "sweep_id": sweep_id,
    "smoke": smoke == "1",
    "n_planned": int(n_runs),
    "started_at": datetime.datetime.now().isoformat(),
    "runs": [],
    "csv_files": [],
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
PYEOF

# ── Helper: check log file for error indicators ───────────────────────────────
log_has_error() {
    local logfile="$1"
    grep -qE '(\*\*\* ERROR|ERROR Failed|UnboundLocalError|KeyError|AttributeError|Traceback \(most recent)' \
        "${logfile}" 2>/dev/null
}

# ── Run each config ───────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0
declare -a MANIFEST_CSV_FILES=()

for cfg_path in "${RUN_CONFIGS[@]}"; do
    cfg_name="$(basename "${cfg_path}" .yaml)"
    run_log="${RUN_LOG_DIR}/${cfg_name}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[sweep] Config: ${cfg_name}"
    echo "[sweep] Log:    ${run_log}"
    echo "════════════════════════════════════════════════════════════════"

    # Snapshot existing MAIN CSV/JSON files (exclude per_layer)
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

    # Find new MAIN CSV/JSON files (exclude per_layer)
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

    # Determine true success:
    #   1. exit code must be 0
    #   2. log must not contain error indicators
    #   3. at least one new MAIN CSV must have been created
    run_ok=1
    if [ ${exit_code} -ne 0 ]; then
        echo "[sweep] ✗ exit code ${exit_code}"
        run_ok=0
    fi
    if log_has_error "${run_log}"; then
        echo "[sweep] ✗ error indicators found in log"
        run_ok=0
    fi
    if [ ${#NEW_MAIN_CSVS[@]} -eq 0 ]; then
        echo "[sweep] ✗ no main CSV created (only per-layer CSVs do not count)"
        run_ok=0
    fi

    if [ ${run_ok} -eq 1 ]; then
        echo "[sweep] ✓ ${cfg_name} done in ${elapsed}s  (main CSVs: ${#NEW_MAIN_CSVS[@]})"
        SUCCEEDED=$(( SUCCEEDED + 1 ))

        # Copy MAIN CSVs and JSONs to run dir
        for f in "${NEW_MAIN_CSVS[@]}"; do
            cp "${f}" "${RUN_CSV_DIR}/"
            MANIFEST_CSV_FILES+=("${RUN_CSV_DIR}/$(basename "${f}")")
            echo "[sweep]   copied: $(basename "${f}")"
        done
        for f in "${NEW_MAIN_JSONS[@]}"; do
            cp "${f}" "${RUN_CSV_DIR}/"
        done
        # Also copy per_layer CSVs separately (for reference, not for summarizer)
        for f in $(ls "${RESULTS_DIR}"/moe_target_pruning_*_per_layer.csv 2>/dev/null || true); do
            fname="$(basename "${f}")"
            if [ ! -f "${RUN_CSV_DIR}/${fname}" ]; then
                cp "${f}" "${RUN_CSV_DIR}/"
            fi
        done
    else
        echo "[sweep] ✗ ${cfg_name} FAILED after ${elapsed}s"
        FAILED=$(( FAILED + 1 ))

        if [ "${CONTINUE_ON_FAIL}" != "1" ]; then
            echo "[sweep] Aborting. Set CONTINUE_ON_FAIL=1 to continue past failures."
            break
        fi
    fi
done

# ── Update final manifest with list of main CSVs ──────────────────────────────
python3 - "${MANIFEST}" "${SWEEP_ID}" "${SUCCEEDED}" "${FAILED}" << PYEOF
import json, sys, datetime
manifest_path, sweep_id, succeeded, failed = sys.argv[1:5]
with open(manifest_path) as f:
    manifest = json.load(f)
manifest["succeeded"] = int(succeeded)
manifest["failed"] = int(failed)
manifest["finished_at"] = datetime.datetime.now().isoformat()
# csv_files: only MAIN CSVs (no per_layer), listed in env var MANIFEST_CSV_LIST
import os
csv_list = os.environ.get("MANIFEST_CSV_LIST", "").split("|")
manifest["csv_files"] = [p for p in csv_list if p.endswith(".csv") and "_per_layer" not in p]
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[sweep] Manifest: {manifest_path}")
PYEOF

# Export for the python subprocess above
export MANIFEST_CSV_LIST
MANIFEST_CSV_LIST="$(IFS='|'; echo "${MANIFEST_CSV_FILES[*]:-}")"
# Re-run manifest update now that env var is set
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
print(f"[sweep] Manifest updated: {manifest_path}")
print(f"[sweep]   csv_files ({len(manifest['csv_files'])}): {manifest['csv_files']}")
PYEOF

# ── Final summary ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[sweep] SWEEP COMPLETE  (id=${SWEEP_ID})"
echo "[sweep]   Succeeded: ${SUCCEEDED} / ${N_RUNS}"
echo "[sweep]   Failed:    ${FAILED} / ${N_RUNS}"
echo "[sweep]   Run dir:   ${RUN_DIR}"
echo "[sweep]   Manifest:  ${MANIFEST}"
echo "════════════════════════════════════════════════════════════════"

if [ "${SUCCEEDED}" -eq 0 ]; then
    echo "[sweep] No successful runs; skipping summarizer."
elif [ -f "scripts/summarize_moe_residual_sweep.py" ]; then
    echo "[sweep] Running summarizer..."
    python3 scripts/summarize_moe_residual_sweep.py \
        --manifest "${MANIFEST}" \
        --out-dir "${RUN_DIR}" || true
fi

if [ "${FAILED}" -gt 0 ]; then
    echo "[sweep] WARNING: ${FAILED} run(s) failed. Logs: ${RUN_LOG_DIR}/"
    exit 1
fi
exit 0
