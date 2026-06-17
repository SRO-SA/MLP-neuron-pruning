#!/usr/bin/env bash
# =============================================================================
# run_moe_full48_benchmark.sh
#
# Runs all 24 Full-48 benchmark configs in order:
#   Group A: WikiText2, n_eval=64
#   Group B: WikiText2, n_eval=512
#   Group C: C4, n_eval=64
#   Group D: C4, n_eval=512
#
# Usage:
#   bash scripts/run_moe_full48_benchmark.sh
#
# Environment variables:
#   CONTINUE_ON_FAIL=1  -- continue running remaining configs even if one fails
#
# Logs are written to logs/moe_full48/<config_basename>.log
# =============================================================================

set -euo pipefail

CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-0}

CONFIGS_DIR="configs"
LOGS_DIR="logs/moe_full48"
SUMMARIZE_CMD="python scripts/summarize_moe_results.py --glob \"results/moe_*.csv\" --no-residual"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${LOGS_DIR}"

echo "============================================================"
echo "  Full-48 MoE Benchmark — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Logs: ${LOGS_DIR}/"
echo "  CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Ordered list of 24 configs
# ---------------------------------------------------------------------------
CONFIGS=(
  # --- Group A: WikiText2, n_eval=64 ---
  "moe_full48_packed_p95_1pct_pure_delete_wikitext2_n64.yaml"
  "moe_full48_packed_p95_2pct_pure_delete_wikitext2_n64.yaml"
  "moe_full48_packed_p95_4pct_pure_delete_wikitext2_n64.yaml"
  "moe_full48_packed_p95_1pct_residual_full_moe_wikitext2_n64.yaml"
  "moe_full48_packed_p95_2pct_residual_full_moe_wikitext2_n64.yaml"
  "moe_full48_packed_p95_4pct_residual_full_moe_wikitext2_n64.yaml"

  # --- Group B: WikiText2, n_eval=512 ---
  "moe_full48_packed_p95_1pct_pure_delete_wikitext2_n512.yaml"
  "moe_full48_packed_p95_2pct_pure_delete_wikitext2_n512.yaml"
  "moe_full48_packed_p95_4pct_pure_delete_wikitext2_n512.yaml"
  "moe_full48_packed_p95_1pct_residual_full_moe_wikitext2_n512.yaml"
  "moe_full48_packed_p95_2pct_residual_full_moe_wikitext2_n512.yaml"
  "moe_full48_packed_p95_4pct_residual_full_moe_wikitext2_n512.yaml"

  # --- Group C: C4, n_eval=64 ---
  "moe_full48_packed_p95_1pct_pure_delete_c4_n64.yaml"
  "moe_full48_packed_p95_2pct_pure_delete_c4_n64.yaml"
  "moe_full48_packed_p95_4pct_pure_delete_c4_n64.yaml"
  "moe_full48_packed_p95_1pct_residual_full_moe_c4_n64.yaml"
  "moe_full48_packed_p95_2pct_residual_full_moe_c4_n64.yaml"
  "moe_full48_packed_p95_4pct_residual_full_moe_c4_n64.yaml"

  # --- Group D: C4, n_eval=512 ---
  "moe_full48_packed_p95_1pct_pure_delete_c4_n512.yaml"
  "moe_full48_packed_p95_2pct_pure_delete_c4_n512.yaml"
  "moe_full48_packed_p95_4pct_pure_delete_c4_n512.yaml"
  "moe_full48_packed_p95_1pct_residual_full_moe_c4_n512.yaml"
  "moe_full48_packed_p95_2pct_residual_full_moe_c4_n512.yaml"
  "moe_full48_packed_p95_4pct_residual_full_moe_c4_n512.yaml"
)

TOTAL=${#CONFIGS[@]}
FAILED=0
PASSED=0

# ---------------------------------------------------------------------------
# Run each config
# ---------------------------------------------------------------------------
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  config_path="${CONFIGS_DIR}/${cfg}"
  basename="${cfg%.yaml}"
  log_file="${LOGS_DIR}/${basename}.log"
  run_num=$((i + 1))

  echo "------------------------------------------------------------"
  echo "  [${run_num}/${TOTAL}] ${cfg}"
  echo "  Log: ${log_file}"
  echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  set +e
  python run_experiment.py --config "${config_path}" --moe-target-pruning 2>&1 | tee "${log_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ ${exit_code} -ne 0 ]]; then
    FAILED=$((FAILED + 1))
    echo ""
    echo "  ERROR: ${cfg} failed with exit code ${exit_code}."
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      echo "  Stopping. Set CONTINUE_ON_FAIL=1 to continue past failures."
      echo ""
      echo "============================================================"
      echo "  Benchmark aborted after ${run_num}/${TOTAL} configs."
      echo "  Passed: ${PASSED}  Failed: ${FAILED}"
      echo "============================================================"
      exit 1
    else
      echo "  CONTINUE_ON_FAIL=1 — continuing to next config."
    fi
  else
    PASSED=$((PASSED + 1))
    echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
  fi

  echo ""
  echo "  [Interim summary after run ${run_num}/${TOTAL}]"
  eval "${SUMMARIZE_CMD} 2>/dev/null || true"
  echo ""
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Full-48 Benchmark Complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Passed: ${PASSED}/${TOTAL}  Failed: ${FAILED}/${TOTAL}"
echo "============================================================"
echo ""
echo "Final summary:"
eval "${SUMMARIZE_CMD} 2>/dev/null || true"

if [[ ${FAILED} -gt 0 ]]; then
  exit 1
fi
