#!/usr/bin/env bash
# run_moe_selector_baseline_ppl.sh
#
# PPL-only selector attribution benchmark.
# 4 selectors × 4 targets × wikitext2 = 16 runs.
# All use: pure_delete method, moe_budget_mode=uniform (identical per-layer
# channel budgets → actual_pct is selector-independent; only WHICH channels
# differ).
#
# Selectors compared:
#   rmsnorm_bound    — weight-only RMSNorm-bounded SwiGLU score (proposed)
#   down_norm        — L2 norm of each down_proj column (simple baseline)
#   activation_score — SiLU(gate)*up activation × down-norm (calib-based)
#   random           — uniform random (random baseline)
#
# Usage:
#   bash scripts/run_moe_selector_baseline_ppl.sh          # full 16-run
#   SMOKE=1 bash scripts/run_moe_selector_baseline_ppl.sh  # 2 runs (rmsnorm+random, target2)
#   DRY_RUN=1 bash scripts/run_moe_selector_baseline_ppl.sh
#
# Env overrides:
#   SMOKE=1                      2 runs: rmsnorm_bound+random, target2 only
#   DRY_RUN=1                    list planned runs, skip execution
#   ONLY_SELECTORS=a,b,c         comma-separated selector filter
#   ONLY_TARGETS=2,4,6,8         comma-separated target pct filter (integers)
#   N_EVAL=512                   calibration/eval samples (must match config filename)
#   CONTINUE_ON_FAIL=1           keep going past failures (default: abort on first)
#   RESULTS_BASE=...             base dir (default: results/moe_selector_baselines)
#   CONFIG_DIR=...               config dir (default: configs/moe_selector_baseline)
#   PLAN_DIR=...                 plan save dir (default: results/pruning_plans)
#   VENV=...                     virtualenv path (default: /workspace/venvs/qwen-pruning)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Env config ────────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_SELECTORS="${ONLY_SELECTORS:-rmsnorm_bound,down_norm,activation_score,random}"
ONLY_TARGETS="${ONLY_TARGETS:-2,4,6,8}"
N_EVAL="${N_EVAL:-512}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
RESULTS_BASE="${RESULTS_BASE:-results/moe_selector_baselines}"
CONFIG_DIR="${CONFIG_DIR:-configs/moe_selector_baseline}"
PLAN_DIR="${PLAN_DIR:-results/pruning_plans}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
DATASET="wikitext2"
MODEL="Qwen/Qwen3-30B-A3B"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_BASE}/${RUN_ID}"
SUMMARY_CSV="${RUN_DIR}/selector_baseline_summary.csv"
ATTRIBUTION_CSV="${RUN_DIR}/selector_attribution_summary.csv"

# ── Parse selector/target lists ───────────────────────────────────────────────
IFS=',' read -ra _SEL_LIST <<< "${ONLY_SELECTORS}"
IFS=',' read -ra _TGT_LIST <<< "${ONLY_TARGETS}"

# SMOKE: restrict to 2 representative configs
if [ "${SMOKE}" = "1" ]; then
    _SEL_LIST=(rmsnorm_bound random)
    _TGT_LIST=(2)
    echo "[ppl] SMOKE=1: running 2 configs (rmsnorm_bound + random, target2)."
fi

echo "[ppl] ======================================================="
echo "[ppl] Selector attribution PPL benchmark"
echo "[ppl] Run ID   : ${RUN_ID}"
echo "[ppl] Selectors: ${_SEL_LIST[*]}"
echo "[ppl] Targets  : ${_TGT_LIST[*]}%"
echo "[ppl] Dataset  : ${DATASET}"
echo "[ppl] N_EVAL   : ${N_EVAL}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "[ppl] Mode     : DRY_RUN (no execution)"
else
    echo "[ppl] Run dir  : ${RUN_DIR}"
fi
echo "[ppl] ======================================================="

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

# ── DRY_RUN: list planned runs and exit ───────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
    n=0
    echo "[ppl] Planned runs:"
    for sel in "${_SEL_LIST[@]}"; do
        for tgt in "${_TGT_LIST[@]}"; do
            n=$(( n + 1 ))
            cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
            if [ -f "${cfg}" ]; then
                status="FOUND"
            else
                status="MISSING -- run: python3 scripts/generate_moe_selector_baseline_configs.py"
            fi
            printf "  %2d. %-20s  target=%2s%%  [%s]\n" \
                "${n}" "${sel}" "${tgt}" "${status}"
        done
    done
    echo ""
    echo "[ppl] ${n} run(s) planned."
    echo "[ppl] Results → ${RESULTS_BASE}/<run_id>/"
    echo "[ppl] Summary → selector_baseline_summary.csv"
    echo "[ppl]         → selector_attribution_summary.csv"
    exit 0
fi

mkdir -p "${RUN_DIR}"

# ── Pre-flight: validate all configs exist ────────────────────────────────────
_MISSING=0
for sel in "${_SEL_LIST[@]}"; do
    for tgt in "${_TGT_LIST[@]}"; do
        cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
        if [ ! -f "${cfg}" ]; then
            echo "[ppl] ERROR: config missing: ${cfg}"
            _MISSING=$(( _MISSING + 1 ))
        fi
    done
done
if [ "${_MISSING}" -gt 0 ]; then
    echo "[ppl] ERROR: ${_MISSING} config(s) missing."
    echo "[ppl]   Generate: python3 scripts/generate_moe_selector_baseline_configs.py"
    exit 1
fi
echo "[ppl] Pre-flight OK: all configs found."

# ── Validate config correctness (moe_budget_mode=uniform required) ────────────
_validate_cfg() {
    python3 - "$1" << 'PYVAL'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
bad = []
if cfg.get("moe_budget_mode") != "uniform":
    bad.append("moe_budget_mode must be 'uniform', got: " + repr(cfg.get("moe_budget_mode")))
if cfg.get("scaling_methods") != ["pure_delete"]:
    bad.append("scaling_methods must be ['pure_delete'], got: " + repr(cfg.get("scaling_methods")))
if bad:
    for b in bad: print("[ppl] CONFIG ERROR:", b)
    sys.exit(1)
PYVAL
}

for sel in "${_SEL_LIST[@]}"; do
    for tgt in "${_TGT_LIST[@]}"; do
        cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
        _validate_cfg "${cfg}" || exit 1
    done
done
echo "[ppl] Config validation OK (moe_budget_mode=uniform confirmed for all)."

# ── Run loop ──────────────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0

for sel in "${_SEL_LIST[@]}"; do
    for tgt in "${_TGT_LIST[@]}"; do
        label="${sel}_target${tgt}"
        run_out="${RUN_DIR}/${label}"
        log="${RUN_DIR}/${label}.log"
        base_cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
        tmp_cfg="${run_out}/run_config.yaml"

        echo ""
        echo "[ppl] --- selector=${sel}  target=${tgt}%  ---"
        mkdir -p "${run_out}"

        # Create temp config with output_dir overridden to run_out,
        # and plan_dir pointed at the shared plan dir (selector-specific filenames).
        python3 - "${base_cfg}" "${run_out}" "${PLAN_DIR}" "${tmp_cfg}" << 'PYCFG'
import sys, yaml, os
src, out_dir, plan_dir, dst = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = out_dir
# Keep save_pruning_plan=true so selector-specific plan is saved.
# Plan filenames already embed the selector name (from make_pruning_plan_path).
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print("[ppl]   Temp config: " + dst)
PYCFG

        set +e
        python3 run_experiment.py \
            --config "${tmp_cfg}" \
            --moe-target-pruning \
            2>&1 | tee "${log}"
        _exit=${PIPESTATUS[0]}
        set -e

        if [ "${_exit}" -ne 0 ]; then
            echo "[ppl] ERROR: run failed (exit ${_exit}): ${label}"
            FAILED=$(( FAILED + 1 ))
            if [ "${CONTINUE_ON_FAIL}" = "1" ]; then
                continue
            fi
            exit 1
        fi

        # Sanity: verify at least one CSV was written
        _n_csvs=$(find "${run_out}" -name "moe_target_pruning_*.csv" 2>/dev/null | wc -l)
        if [ "${_n_csvs}" -eq 0 ]; then
            echo "[ppl] ERROR: run exited 0 but no moe_target_pruning_*.csv in ${run_out}"
            FAILED=$(( FAILED + 1 ))
            [ "${CONTINUE_ON_FAIL}" = "1" ] && continue || exit 1
        fi

        echo "[ppl] OK: ${label} (${_n_csvs} CSV(s) written)"
        SUCCEEDED=$(( SUCCEEDED + 1 ))
    done
done

echo ""
echo "[ppl] ======================================================="
echo "[ppl] Runs: ${SUCCEEDED} succeeded, ${FAILED} failed."
echo "[ppl] ======================================================="

if [ "${FAILED}" -gt 0 ]; then
    echo "[ppl] ERROR: ${FAILED} run(s) failed. Aborting summary."
    exit 1
fi

# ── Build summary + attribution CSVs ─────────────────────────────────────────
EXPECTED_ROWS=$(( ${#_SEL_LIST[@]} * ${#_TGT_LIST[@]} ))
# For SMOKE or partial runs, don't require the full 16
if [ "${SMOKE}" = "1" ]; then
    MIN_ROWS="${EXPECTED_ROWS}"
else
    MIN_ROWS=16
fi

echo "[ppl] Building summary CSVs ..."
python3 scripts/summarize_moe_selector_ppl.py \
    --run-dir      "${RUN_DIR}" \
    --model        "${MODEL}" \
    --dataset      "${DATASET}" \
    --selectors    "${ONLY_SELECTORS}" \
    --targets      "${ONLY_TARGETS}" \
    --orig-moe-dim 768 \
    --moe-align    16 \
    --min-rows     "${MIN_ROWS}" \
    2>&1

echo ""
echo "[ppl] Summary CSV   : ${SUMMARY_CSV}"
echo "[ppl] Attribution   : ${ATTRIBUTION_CSV}"
echo "[ppl] Run complete."
