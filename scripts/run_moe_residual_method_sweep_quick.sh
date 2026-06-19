#!/usr/bin/env bash
# run_moe_residual_method_sweep_quick.sh
#
# Runs the MoE residual method comparison sweep.
# Generates 40 configs (4 targets × 10 methods), then runs them in order.
# Pure-delete runs first per target (they save the pruning plan that other methods load).
#
# Usage:
#   bash scripts/run_moe_residual_method_sweep_quick.sh
#
# Environment overrides:
#   SMOKE=1             Run only 4 representative configs (quick sanity check)
#   CONTINUE_ON_FAIL=1  Keep going even if a run fails (default: stop on failure)
#   CONFIG_DIR=...      Override config directory (default: configs/moe_residual_sweep)
#   RESULTS_DIR=...     Override results directory (default: results)
#   LOG_DIR=...         Override log directory (default: results/logs)
#   N_EVAL=...          Override n_eval (default: from config yaml)
#   VENV=...            Override virtualenv path (default: /workspace/venvs/qwen-pruning)

set -euo pipefail

# ── Repo root ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ─────────────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/moe_residual_sweep}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"

# ── Activate virtualenv if it exists ───────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    echo "[sweep] Activating virtualenv: ${VENV}"
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "[sweep] No virtualenv found at ${VENV}, using system Python"
fi

# ── Check environment ──────────────────────────────────────────────────────────
if [ -f "scripts/check_env.py" ]; then
    echo "[sweep] Running environment check..."
    python3 scripts/check_env.py --strict || {
        echo "[sweep] ERROR: environment check failed. Aborting."
        exit 1
    }
fi

# ── Generate configs ───────────────────────────────────────────────────────────
echo "[sweep] Generating configs in ${CONFIG_DIR}/ ..."
python3 scripts/generate_moe_residual_sweep_configs.py --out-dir "${CONFIG_DIR}"
echo "[sweep] Configs ready."

# ── Collect configs in correct order ──────────────────────────────────────────
# Order matters: pure_delete must run before other methods at same target
# (it saves the pruning plan that other methods load).
# File naming convention: ..._target{T}_pure_delete.yaml comes before others for same T.
declare -a ALL_CONFIGS=()
for target in 2 4 8 16; do
    pd="${CONFIG_DIR}/qwen3_30b_a3b_wikitext2_n64_target${target}_pure_delete.yaml"
    if [ -f "${pd}" ]; then
        ALL_CONFIGS+=("${pd}")
    fi
    # Then all non-pure-delete configs for this target
    for f in $(ls "${CONFIG_DIR}"/qwen3_30b_a3b_wikitext2_n64_target${target}_*.yaml 2>/dev/null | sort); do
        if [ "${f}" != "${pd}" ]; then
            ALL_CONFIGS+=("${f}")
        fi
    done
done

if [ ${#ALL_CONFIGS[@]} -eq 0 ]; then
    echo "[sweep] ERROR: No config files found in ${CONFIG_DIR}/. Aborting."
    exit 1
fi
echo "[sweep] Found ${#ALL_CONFIGS[@]} config(s) total."

# ── SMOKE mode: select 4 representative configs ───────────────────────────────
if [ "${SMOKE}" = "1" ]; then
    echo "[sweep] SMOKE=1: selecting 4 representative configs."
    SMOKE_METHODS=(
        "target2_pure_delete"
        "target2_residual_ridge_lam1e-2"
        "target2_residual_ridge_oii_lam1e-2"
        "target2_residual_nearest_merge"
    )
    declare -a RUN_CONFIGS=()
    for pattern in "${SMOKE_METHODS[@]}"; do
        matched=""
        for f in "${ALL_CONFIGS[@]}"; do
            if [[ "${f}" == *"${pattern}"* ]]; then
                matched="${f}"
                break
            fi
        done
        if [ -n "${matched}" ]; then
            RUN_CONFIGS+=("${matched}")
            echo "[sweep]   + ${matched}"
        else
            echo "[sweep]   WARNING: no config matched pattern '${pattern}'"
        fi
    done
else
    RUN_CONFIGS=("${ALL_CONFIGS[@]}")
fi

echo "[sweep] Will run ${#RUN_CONFIGS[@]} config(s)."

# ── Log directory ──────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
SWEEP_LOG="${LOG_DIR}/moe_residual_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[sweep] Sweep log: ${SWEEP_LOG}"

# ── Run each config ────────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0
SKIPPED=0

for cfg_path in "${RUN_CONFIGS[@]}"; do
    cfg_name="$(basename "${cfg_path}" .yaml)"
    run_log="${LOG_DIR}/${cfg_name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[sweep] Running: ${cfg_name}"
    echo "[sweep] Config:  ${cfg_path}"
    echo "[sweep] Log:     ${run_log}"
    echo "════════════════════════════════════════════════════════════════"

    # Record start time
    t_start=$(date +%s)

    # Run the experiment, tee to log and stdout
    set +e
    python3 run_experiment.py --config "${cfg_path}" 2>&1 | tee "${run_log}"
    exit_code=${PIPESTATUS[0]}
    set -e

    t_end=$(date +%s)
    elapsed=$(( t_end - t_start ))

    if [ ${exit_code} -eq 0 ]; then
        echo "[sweep] ✓ ${cfg_name} completed in ${elapsed}s"
        SUCCEEDED=$(( SUCCEEDED + 1 ))
    else
        echo "[sweep] ✗ ${cfg_name} FAILED (exit ${exit_code}) after ${elapsed}s"
        FAILED=$(( FAILED + 1 ))

        # Log the failure to sweep log
        echo "FAILED: ${cfg_name} (exit ${exit_code})" >> "${SWEEP_LOG}"

        if [ "${CONTINUE_ON_FAIL}" != "1" ]; then
            echo "[sweep] Aborting sweep. Set CONTINUE_ON_FAIL=1 to continue past failures."
            break
        else
            echo "[sweep] CONTINUE_ON_FAIL=1: continuing despite failure."
        fi
    fi

    # Append per-run summary line to sweep log
    echo "$(date '+%Y-%m-%d %H:%M:%S') exit=${exit_code} elapsed=${elapsed}s config=${cfg_name}" >> "${SWEEP_LOG}"

    # Run summarizer after each successful run (if it exists)
    if [ ${exit_code} -eq 0 ] && [ -f "scripts/summarize_moe_residual_sweep.py" ]; then
        python3 scripts/summarize_moe_residual_sweep.py \
            --results-dir "${RESULTS_DIR}" \
            --quiet 2>/dev/null || true
    fi
done

# ── Final summary ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[sweep] SWEEP COMPLETE"
echo "[sweep]   Succeeded: ${SUCCEEDED}"
echo "[sweep]   Failed:    ${FAILED}"
echo "[sweep]   Total:     $(( SUCCEEDED + FAILED ))"
echo "════════════════════════════════════════════════════════════════"

# Final summarizer run
if [ -f "scripts/summarize_moe_residual_sweep.py" ]; then
    echo "[sweep] Running final summarizer..."
    python3 scripts/summarize_moe_residual_sweep.py \
        --results-dir "${RESULTS_DIR}" \
        --out-dir "${RESULTS_DIR}" || true
fi

if [ ${FAILED} -gt 0 ]; then
    echo "[sweep] WARNING: ${FAILED} run(s) failed. Check ${LOG_DIR}/ for details."
    exit 1
fi

exit 0
