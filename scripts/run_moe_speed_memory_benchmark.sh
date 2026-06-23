#!/usr/bin/env bash
# run_moe_speed_memory_benchmark.sh
#
# Timing and GPU-memory benchmark for Qwen3-30B-A3B before and after pruning.
#
# 7 settings:
#   1. Baseline (no pruning)
#   2. rmsnorm_bound, wikitext2, 2%
#   3. rmsnorm_bound, wikitext2, 4%
#   4. rmsnorm_bound, wikitext2, 8%
#   5. rmsnorm_bound, c4,        2%
#   6. rmsnorm_bound, c4,        4%
#   7. rmsnorm_bound, c4,        8%
#
# Pruning plans are loaded from results/pruning_plans/.  They must have been
# produced by a prior run of run_moe_residual_selected_full_benchmark.sh
# (which saves pure_delete plans with save_pruning_plan: true).
#
# Usage:
#   bash scripts/run_moe_speed_memory_benchmark.sh           # full benchmark
#   DRY_RUN=1 bash scripts/run_moe_speed_memory_benchmark.sh # list settings, no GPU
#   SMOKE=1   bash scripts/run_moe_speed_memory_benchmark.sh # baseline + 1 pruned
#
# Env overrides:
#   DRY_RUN=1         List settings only, no model loading
#   SMOKE=1           2 settings only (baseline + 2%/wikitext2)
#   RESULTS_DIR=...   Results directory (default: results)
#   MODEL=...         HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)
#   DTYPE=...         bfloat16 | float16 | float32 (default: bfloat16)
#   VENV=...          Virtualenv path (default: /workspace/venvs/qwen-pruning)
#   N_WARMUP=...      Warm-up iterations (default: 2)
#   N_BENCH=...       Measured iterations (default: 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
DTYPE="${DTYPE:-bfloat16}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
N_WARMUP="${N_WARMUP:-2}"
N_BENCH="${N_BENCH:-5}"

MODEL_SLUG="$(echo "${MODEL}" | tr '/' '_' | tr '-' '_')"
PLAN_DIR="${RESULTS_DIR}/pruning_plans"

# ── Sweep ID ──────────────────────────────────────────────────────────────────
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_DIR}/speed_memory_runs/${SWEEP_ID}"
OUT_CSV="${OUT_DIR}/speed_memory_results.csv"
MANIFEST="${OUT_DIR}/plans_manifest.json"

echo "[speed] Speed/memory benchmark ID: ${SWEEP_ID}"
echo "[speed] Model:  ${MODEL}"
echo "[speed] Dtype:  ${DTYPE}"
echo "[speed] Output: ${OUT_CSV}"

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    echo "[speed] Activating virtualenv: ${VENV}"
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "[speed] No virtualenv at ${VENV}, using system Python"
fi

# ── Build list of 7 settings ──────────────────────────────────────────────────
# Selector used when generating the plans from run_moe_residual_selected_full_benchmark.sh:
SELECTOR="rmsnorm_bound"
AGG_MODE="p95"
ALIGN="16"
N_EVAL="512"
CALIB_N="512"

# All targets to benchmark (must match existing plan files)
TARGETS="2 4 8"
DATASETS="wikitext2 c4"

# ── DRY_RUN: list settings and exit ──────────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[speed] Planned settings (7 total):"
    echo "   1. baseline_no_pruning                   (no plan)"
    n=1
    for dataset in ${DATASETS}; do
        for target in ${TARGETS}; do
            n=$(( n + 1 ))
            plan_file="${PLAN_DIR}/${MODEL_SLUG}_${dataset}_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target}.0pct_align${ALIGN}.json"
            label="${SELECTOR}_${dataset}_target${target}pct"
            exists=""
            [ -f "${plan_file}" ] && exists="[plan exists]" || exists="[PLAN MISSING - run full benchmark first]"
            printf "  %2d. %-45s  %s\n" "${n}" "${label}" "${exists}"
        done
    done
    echo ""
    echo "[speed] DRY_RUN complete. Plans must exist in ${PLAN_DIR}/"
    echo "[speed] To generate plans: bash scripts/run_moe_residual_selected_full_benchmark.sh"
    echo "[speed] To run:            bash scripts/run_moe_speed_memory_benchmark.sh"
    exit 0
fi

mkdir -p "${OUT_DIR}"

# ── Check that at least one plan exists ──────────────────────────────────────
found_plans=0
for dataset in ${DATASETS}; do
    for target in ${TARGETS}; do
        plan_file="${PLAN_DIR}/${MODEL_SLUG}_${dataset}_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target}.0pct_align${ALIGN}.json"
        [ -f "${plan_file}" ] && found_plans=$(( found_plans + 1 ))
    done
done

if [ ${found_plans} -eq 0 ]; then
    echo "[speed] ERROR: No pruning plans found in ${PLAN_DIR}/"
    echo "[speed]   Run: bash scripts/run_moe_residual_selected_full_benchmark.sh"
    echo "[speed]   (Plans are saved when save_pruning_plan: true in config)"
    exit 1
fi

echo "[speed] Found ${found_plans} plan file(s) in ${PLAN_DIR}/"

# ── Build plans manifest JSON ─────────────────────────────────────────────────
python3 - "${MANIFEST}" "${MODEL_SLUG}" "${PLAN_DIR}" \
    "${SELECTOR}" "${AGG_MODE}" "${ALIGN}" "${N_EVAL}" "${CALIB_N}" \
    "${SMOKE}" "${TARGETS}" "${DATASETS}" << 'PYEOF'
import json, os, sys

manifest_path = sys.argv[1]
model_slug    = sys.argv[2]
plan_dir      = sys.argv[3]
selector      = sys.argv[4]
agg_mode      = sys.argv[5]
align         = sys.argv[6]
n_eval        = sys.argv[7]
calib_n       = sys.argv[8]
smoke         = sys.argv[9] == "1"
targets       = sys.argv[10].split()
datasets      = sys.argv[11].split()

settings = []
for dataset in datasets:
    for target in targets:
        fname = (
            f"{model_slug}_{dataset}_n{n_eval}_calib{calib_n}"
            f"_{selector}_{agg_mode}_{target}.0pct_align{align}.json"
        )
        fpath = os.path.join(plan_dir, fname)
        label = f"{selector}_{dataset}_target{target}pct"
        if os.path.isfile(fpath):
            settings.append({"label": label, "plan": fpath})
            if smoke:
                # SMOKE: only the first match (2% wikitext2)
                break
    if smoke and settings:
        break

manifest = {"settings": settings}
with open(manifest_path, "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"[speed] Plans manifest: {manifest_path}  ({len(settings)} pruned settings)")
PYEOF

# ── Environment check ─────────────────────────────────────────────────────────
if [ -f "scripts/check_env.py" ]; then
    echo "[speed] Running environment check..."
    python3 scripts/check_env.py --strict || {
        echo "[speed] ERROR: environment check failed. Aborting."
        exit 1
    }
fi

# ── SMOKE mode: 2 settings only (baseline + first pruned plan) ────────────────
EXTRA_ARGS=""
if [ "${SMOKE}" = "1" ]; then
    echo "[speed] SMOKE=1: running baseline + first pruned plan only."
fi

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo "[speed] Starting benchmark..."
python3 scripts/benchmark_moe_speed_memory.py \
    --model    "${MODEL}" \
    --plans    "${MANIFEST}" \
    --out      "${OUT_CSV}" \
    --dtype    "${DTYPE}" \
    --n-warmup "${N_WARMUP}" \
    --n-bench  "${N_BENCH}" \
    ${EXTRA_ARGS}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[speed] BENCHMARK COMPLETE  (id=${SWEEP_ID})"
echo "[speed]   Results: ${OUT_CSV}"
echo "[speed]   Plans:   ${MANIFEST}"
echo "════════════════════════════════════════════════════════════════"
exit 0
