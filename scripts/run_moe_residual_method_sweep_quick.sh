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
# Uses Python inline so we get clear error messages before launching GPU jobs.
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
# Targets match what the generator uses: 2 4 6 8
# pure_delete must run first per target (it saves the pruning plan).
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
    echo "[sweep] ERROR: No config files found in ${CONFIG_DIR}/. Aborting."
    exit 1
fi

# Assert full matrix for non-smoke runs
if [ "${SMOKE}" != "1" ] && [ ${TOTAL_FOUND} -ne 40 ]; then
    echo "[sweep] ERROR: Expected 40 configs (4 targets × 10 methods), found ${TOTAL_FOUND}."
    echo "[sweep]   CONFIG_DIR=${CONFIG_DIR}"
    echo "[sweep]   Regenerate with: python3 scripts/generate_moe_residual_sweep_configs.py"
    exit 1
fi

# ── Validate all configs before starting any GPU work ─────────────────────────
echo "[sweep] Validating all ${TOTAL_FOUND} configs..."
for cfg_path in "${ALL_CONFIGS[@]}"; do
    validate_config "${cfg_path}" || exit 1
done
echo "[sweep] All configs valid."

# ── SMOKE mode: select 4 representative configs ───────────────────────────────
# Always from target2: pure_delete + ridge_lam1e-2 + ridge_oii_lam1e-2 + nearest_merge
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
    echo "[sweep] ${#ALL_CONFIGS[@]} configs generated; running ${#RUN_CONFIGS[@]} (smoke)."
else
    RUN_CONFIGS=("${ALL_CONFIGS[@]}")
fi

N_RUNS=${#RUN_CONFIGS[@]}
echo "[sweep] Will run ${N_RUNS} config(s)."

# ── Write initial manifest ─────────────────────────────────────────────────────
python3 - "${MANIFEST}" "${SWEEP_ID}" "${SMOKE}" "${N_RUNS}" << 'PYEOF'
import json, sys, datetime
manifest_path, sweep_id, smoke, n_runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
manifest = {
    "sweep_id": sweep_id,
    "smoke": smoke == "1",
    "n_planned": int(n_runs),
    "started_at": datetime.datetime.now().isoformat(),
    "runs": []
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
PYEOF

# ── Helper: find new CSV/JSON files written since a snapshot ─────────────────
snapshot_results() {
    # Print list of moe_target_pruning_*.csv and *.json in RESULTS_DIR
    find "${RESULTS_DIR}" -maxdepth 1 \
        \( -name "moe_target_pruning_*.csv" -o -name "moe_target_pruning_*.json" \) \
        -newer /proc/self/exe 2>/dev/null || true
}

# ── Run each config ────────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0
declare -a MANIFEST_RUNS=()

for cfg_path in "${RUN_CONFIGS[@]}"; do
    cfg_name="$(basename "${cfg_path}" .yaml)"
    run_log="${RUN_LOG_DIR}/${cfg_name}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[sweep] Config: ${cfg_name}"
    echo "[sweep] Log:    ${run_log}"
    echo "════════════════════════════════════════════════════════════════"

    # Snapshot existing files before the run
    BEFORE_CSV=( $(ls "${RESULTS_DIR}"/moe_target_pruning_*.csv 2>/dev/null || true) )
    BEFORE_JSON=( $(ls "${RESULTS_DIR}"/moe_target_pruning_*.json 2>/dev/null || true) )

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

    # Find new CSV/JSON files written by this run
    declare -a NEW_CSVS=()
    declare -a NEW_JSONS=()
    for f in $(ls "${RESULTS_DIR}"/moe_target_pruning_*.csv 2>/dev/null || true); do
        if [[ ! " ${BEFORE_CSV[*]} " =~ " ${f} " ]]; then
            NEW_CSVS+=("$f")
        fi
    done
    for f in $(ls "${RESULTS_DIR}"/moe_target_pruning_*.json 2>/dev/null || true); do
        if [[ ! " ${BEFORE_JSON[*]} " =~ " ${f} " ]]; then
            NEW_JSONS+=("$f")
        fi
    done

    if [ ${exit_code} -eq 0 ]; then
        echo "[sweep] ✓ ${cfg_name} done in ${elapsed}s"
        SUCCEEDED=$(( SUCCEEDED + 1 ))

        # Copy outputs to run dir
        for f in "${NEW_CSVS[@]:-}"; do
            [ -z "${f}" ] && continue
            cp "${f}" "${RUN_CSV_DIR}/"
            echo "[sweep]   copied: $(basename "${f}")"
        done
        for f in "${NEW_JSONS[@]:-}"; do
            [ -z "${f}" ] && continue
            cp "${f}" "${RUN_CSV_DIR}/"
        done

        MANIFEST_RUNS+=("{\"config\":\"${cfg_name}\",\"status\":\"ok\",\"elapsed\":${elapsed}}")
    else
        echo "[sweep] ✗ ${cfg_name} FAILED (exit ${exit_code}) after ${elapsed}s"
        FAILED=$(( FAILED + 1 ))
        MANIFEST_RUNS+=("{\"config\":\"${cfg_name}\",\"status\":\"failed\",\"elapsed\":${elapsed}}")

        if [ "${CONTINUE_ON_FAIL}" != "1" ]; then
            echo "[sweep] Aborting. Set CONTINUE_ON_FAIL=1 to continue past failures."
            break
        fi
    fi
done

# ── Update final manifest ──────────────────────────────────────────────────────
python3 - "${MANIFEST}" "${SWEEP_ID}" "${SUCCEEDED}" "${FAILED}" "${RUN_CSV_DIR}" << 'PYEOF'
import json, sys, glob, datetime
manifest_path, sweep_id, succeeded, failed, csv_dir = sys.argv[1:6]
with open(manifest_path) as f:
    manifest = json.load(f)
manifest["succeeded"] = int(succeeded)
manifest["failed"] = int(failed)
manifest["finished_at"] = datetime.datetime.now().isoformat()
# List all CSVs in the run dir (exclude per_layer and summary)
manifest["csv_files"] = sorted(
    f for f in glob.glob(f"{csv_dir}/moe_target_pruning_*.csv")
)
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[sweep] Manifest updated: {manifest_path}")
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
    echo "[sweep] WARNING: ${FAILED} run(s) failed. Check ${RUN_LOG_DIR}/ for details."
    exit 1
fi
exit 0
