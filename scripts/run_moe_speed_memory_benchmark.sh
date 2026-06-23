#!/usr/bin/env bash
# run_moe_speed_memory_benchmark.sh
#
# Speed/memory benchmark for Qwen3-30B-A3B before and after MoE pruning.
#
# KEY DESIGN: each setting runs in a SEPARATE Python subprocess.
# This ensures GPU memory stats are fully isolated — no carryover between
# settings, no doubled peak-memory from two models loaded simultaneously.
#
# SMOKE mode (2 settings):
#   1. baseline_no_pruning
#   2. residual_nearest_channel_merge_moe at 4%, wikitext2
#      label: residual_nearest_channel_merge_moe__rmsnorm_bound__wikitext2__target4pct__actual4.2pct
#
# Full mode: baseline + 6 pruned settings
#   rmsnorm_bound × {wikitext2, c4} × {4%, 8%} (residual_nearest)
#   + pure_delete × wikitext2 × 8% (for direct comparison)
#
# Usage:
#   SMOKE=1   bash scripts/run_moe_speed_memory_benchmark.sh   # 2 settings
#   DRY_RUN=1 bash scripts/run_moe_speed_memory_benchmark.sh   # list, no GPU
#   bash scripts/run_moe_speed_memory_benchmark.sh              # full benchmark
#
# Env overrides:
#   DRY_RUN=1         List settings, do not run
#   SMOKE=1           2 settings only (baseline + residual_nearest 4%/wikitext2)
#   MODEL=...         HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)
#   DTYPE=...         bfloat16|float16|float32 (default: bfloat16)
#   RESULTS_DIR=...   Results dir (default: results)
#   VENV=...          Virtualenv (default: /workspace/venvs/qwen-pruning)
#   N_WARMUP=...      Warm-up iterations (default: 2)
#   N_BENCH=...       Measured iterations (default: 5)
#   BATCH_SIZE=...    Batch size (default: 1)
#   SELECTOR=...      Selector used for plan files (default: rmsnorm_bound)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
AUTO_GENERATE_PLAN="${AUTO_GENERATE_PLAN:-0}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
DTYPE="${DTYPE:-bfloat16}"
RESULTS_DIR="${RESULTS_DIR:-results}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
N_WARMUP="${N_WARMUP:-2}"
N_BENCH="${N_BENCH:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="32"
SELECTOR="${SELECTOR:-rmsnorm_bound}"
AGG_MODE="p95"
ALIGN="16"
N_EVAL="512"
CALIB_N="512"
D_FF="768"   # Qwen3-30B-A3B MoE intermediate size

MODEL_SLUG="$(echo "${MODEL}" | tr '/' '_' | tr '-' '_')"
# Config prefix = model name without org prefix, lowercased, dashes→underscores
# e.g. "Qwen/Qwen3-30B-A3B" → "qwen3_30b_a3b"  (matches generate_moe_selector_baseline_configs.py)
CONFIG_PREFIX="$(echo "${MODEL}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
PLAN_DIR="${RESULTS_DIR}/pruning_plans"

# ── Sweep ID ──────────────────────────────────────────────────────────────────
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_DIR}/speed_memory_runs/${SWEEP_ID}"
JSON_DIR="${OUT_DIR}/jsons"
LOG_DIR="${OUT_DIR}/logs"
OUT_CSV="${OUT_DIR}/speed_memory_results.csv"

echo "[speed] Speed/memory benchmark ID: ${SWEEP_ID}"
echo "[speed] Model:  ${MODEL}"
echo "[speed] Dtype:  ${DTYPE}"
echo "[speed] Memory isolation: one Python subprocess per setting."
echo "[speed] Out:    ${OUT_CSV}"

# ── Virtualenv ────────────────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    echo "[speed] Activating virtualenv: ${VENV}"
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "[speed] No virtualenv at ${VENV}, using system Python"
fi

# ── Helper: compute actual_pct for uniform budget ─────────────────────────────
# Uses round() matching moe_budget_mode=uniform in moe_pruning.py:
#   n = round(d_ff * target / 100.0 / align) * align
#   actual_pct = 100.0 * n / d_ff
_actual_pct() {
    local t="$1"
    python3 -c "
d_ff=${D_FF}; align=${ALIGN}; t=${t}
n = int(round(d_ff * t / 100.0 / align)) * align
print(f'{100.0 * n / d_ff:.1f}')
" 2>/dev/null || echo "0.0"
}

# ── Build settings array ──────────────────────────────────────────────────────
# Format: "label|plan_path_or_NONE|method|selector|dataset|target_pct|actual_pct"
declare -a SETTINGS=()
SETTINGS+=("baseline_no_pruning|NONE|baseline|none|none|0.0|0.0")

_add_setting() {
    local method="$1" dataset="$2" target_pct="$3"
    local actual_pct
    actual_pct="$(_actual_pct "${target_pct}")"
    local plan_file="${MODEL_SLUG}_${dataset}_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target_pct}.0pct_align${ALIGN}.json"
    local plan_path="${PLAN_DIR}/${plan_file}"
    local label="${method}__${SELECTOR}__${dataset}__target${target_pct}pct__actual${actual_pct}pct"
    SETTINGS+=("${label}|${plan_path}|${method}|${SELECTOR}|${dataset}|${target_pct}|${actual_pct}")
}

if [ "${SMOKE}" = "1" ]; then
    _add_setting "residual_nearest_channel_merge_moe" "wikitext2" "4"
    echo "[speed] SMOKE=1: 2 settings (baseline + residual_nearest_channel_merge_moe 4% wikitext2)"
else
    # Full: baseline + 6 pruned settings
    for dataset in wikitext2 c4; do
        for target in 4 8; do
            _add_setting "residual_nearest_channel_merge_moe" "${dataset}" "${target}"
        done
    done
    _add_setting "pure_delete" "wikitext2" "8"
fi

# ── DRY_RUN: list and exit ────────────────────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[speed] Planned settings (${#SETTINGS[@]} total):"
    n=0
    for setting in "${SETTINGS[@]}"; do
        n=$(( n + 1 ))
        IFS='|' read -r label plan_path method selector dataset target_pct actual_pct <<< "${setting}"
        exists_str=""
        if [ "${plan_path}" = "NONE" ]; then
            exists_str="(baseline)"
        elif [ -f "${plan_path}" ]; then
            exists_str="[plan exists]"
        else
            exists_str="[PLAN MISSING — run full benchmark first]"
        fi
        printf "  %2d. %-68s  %s\n" "${n}" "${label}" "${exists_str}"
    done
    echo ""
    echo "[speed] Plans are generated by: bash scripts/run_moe_residual_selected_full_benchmark.sh"
    echo "[speed] Each setting will run in a SEPARATE Python process (memory isolation)."
    if [ "${SMOKE}" = "1" ]; then
        echo "[speed] DRY_RUN complete. To run: SMOKE=1 bash scripts/run_moe_speed_memory_benchmark.sh"
    else
        echo "[speed] DRY_RUN complete. To run: bash scripts/run_moe_speed_memory_benchmark.sh"
    fi
    exit 0
fi

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "${JSON_DIR}" "${LOG_DIR}"
echo "[speed] Output dir: ${OUT_DIR}"
echo "[speed] Running ${#SETTINGS[@]} settings ..."

# ── Run each setting in a separate Python subprocess ─────────────────────────
SUCCEEDED=0
FAILED=0
SKIPPED=0

for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label plan_path method selector dataset target_pct actual_pct <<< "${setting}"
    out_json="${JSON_DIR}/${label}.json"
    log_file="${LOG_DIR}/${label}.log"

    echo ""
    echo "────────────────────────────────────────────────────────────────────────"
    echo "[speed] Setting: ${label}"
    echo "[speed]   method=${method}  selector=${selector}  dataset=${dataset}"
    echo "[speed]   target=${target_pct}%  actual=${actual_pct}%"
    if [ "${plan_path}" = "NONE" ]; then
        echo "[speed]   plan: (none — baseline)"
    else
        echo "[speed]   plan: ${plan_path}"
    fi

    # Handle missing plans (except baseline)
    if [ "${plan_path}" != "NONE" ] && [ ! -f "${plan_path}" ]; then
        if [ "${AUTO_GENERATE_PLAN}" = "1" ]; then
            # Generate ONLY the specific plan needed — not the full 24-run benchmark.
            # Derive the matching selector-baseline config for this (selector, dataset, target).
            local_target_int="${target_pct%.*}"   # strip .0 → e.g. "4"
            # Config names use CONFIG_PREFIX (lowercase, no org), not MODEL_SLUG
            local_cfg="configs/moe_selector_baseline/${CONFIG_PREFIX}_${dataset}_n${N_EVAL}_target${local_target_int}_sel_${selector}.yaml"
            echo "[speed] AUTO_GENERATE_PLAN=1: plan missing — will generate it now."
            echo "[speed]   Config : ${local_cfg}"
            echo "[speed]   Plan   : ${plan_path}"

            # Ensure selector-baseline configs exist
            if [ ! -f "${local_cfg}" ]; then
                echo "[speed]   Generating selector-baseline configs first ..."
                python3 scripts/generate_moe_selector_baseline_configs.py || {
                    echo "[speed] ERROR: failed to generate selector-baseline configs."
                    FAILED=$(( FAILED + 1 ))
                    continue
                }
            fi

            if [ ! -f "${local_cfg}" ]; then
                echo "[speed] ERROR: config still missing after generation: ${local_cfg}"
                echo "[speed]   Selector '${selector}' may not match any generated config."
                FAILED=$(( FAILED + 1 ))
                continue
            fi

            echo "[speed]   Running: python3 run_experiment.py --config ${local_cfg} --moe-target-pruning"
            set +e
            python3 run_experiment.py \
                --config "${local_cfg}" \
                --moe-target-pruning \
                2>&1 | tee "${LOG_DIR}/${label}_plan_gen.log"
            gen_exit="${PIPESTATUS[0]}"
            set -e

            if [ "${gen_exit}" -ne 0 ]; then
                echo "[speed] ERROR: plan generation failed (exit ${gen_exit}). Aborting this setting."
                FAILED=$(( FAILED + 1 ))
                continue
            fi

            if [ ! -f "${plan_path}" ]; then
                echo "[speed] ERROR: plan still missing after generation: ${plan_path}"
                echo "[speed]   Check that save_pruning_plan: true is set in ${local_cfg}"
                FAILED=$(( FAILED + 1 ))
                continue
            fi

            echo "[speed] Plan generated: ${plan_path}"
        else
            # No AUTO_GENERATE_PLAN — fail clearly rather than silently skip.
            echo "[speed] ERROR: required plan not found."
            echo "[speed]   Missing: ${plan_path}"
            echo "[speed]"
            echo "[speed]   To generate it, run one of:"
            echo "[speed]     AUTO_GENERATE_PLAN=1 SMOKE=1 bash scripts/run_moe_speed_memory_benchmark.sh"
            echo "[speed]     bash scripts/run_moe_residual_selected_full_benchmark.sh"
            echo "[speed]"
            echo "[speed]   Or to run just this setting's plan:"
            local_target_int="${target_pct%.*}"
            echo "[speed]     python3 scripts/generate_moe_selector_baseline_configs.py"
            echo "[speed]     python3 run_experiment.py --config configs/moe_selector_baseline/${CONFIG_PREFIX}_${dataset}_n${N_EVAL}_target${local_target_int}_sel_${selector}.yaml --moe-target-pruning"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
    fi

    # Build Python args as an array (handles paths with spaces safely)
    py_args=(
        --model          "${MODEL}"
        --label          "${label}"
        --method         "${method}"
        --selector       "${selector}"
        --dataset        "${dataset}"
        --target-pct     "${target_pct}"
        --actual-pct     "${actual_pct}"
        --out-json       "${out_json}"
        --dtype          "${DTYPE}"
        --n-warmup       "${N_WARMUP}"
        --n-bench        "${N_BENCH}"
        --batch-size     "${BATCH_SIZE}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
    )
    if [ "${plan_path}" != "NONE" ]; then
        py_args+=(--plan "${plan_path}")
    fi

    echo "[speed] Launching subprocess: python3 scripts/benchmark_moe_speed_memory.py ..."
    set +e
    # Each subprocess is a fresh Python process: isolated CUDA context + memory stats
    python3 scripts/benchmark_moe_speed_memory.py "${py_args[@]}" \
        2>&1 | tee "${log_file}"
    py_exit="${PIPESTATUS[0]}"
    set -e

    if [ "${py_exit}" -ne 0 ]; then
        echo "[speed] ✗ FAILED: ${label} (exit ${py_exit})"
        FAILED=$(( FAILED + 1 ))
    elif [ -f "${out_json}" ]; then
        echo "[speed] ✓ OK: ${label}"
        SUCCEEDED=$(( SUCCEEDED + 1 ))
    else
        echo "[speed] ✗ FAILED: ${label} (no JSON written)"
        FAILED=$(( FAILED + 1 ))
    fi
done

# ── Aggregate JSON → CSV ──────────────────────────────────────────────────────
echo ""
echo "[speed] Aggregating results → ${OUT_CSV}"

python3 - "${JSON_DIR}" "${OUT_CSV}" << 'PYEOF'
import csv, json, os, sys, glob

json_dir = sys.argv[1]
out_csv  = sys.argv[2]

CSV_FIELDS = [
    "label", "method", "selector", "dataset", "target_pct", "actual_pct",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "active_expert_flop_reduction_pct",
    "prompt_len", "generated_tokens", "batch_size",
    "prefill_latency_ms_mean", "decode_latency_ms_mean",
    "end_to_end_latency_ms_mean", "tokens_per_sec_mean",
    "peak_allocated_mib_total", "peak_reserved_mib_total",
    "peak_allocated_mib_gpu0", "peak_allocated_mib_gpu1",
    "memory_after_load_mib_total", "memory_after_pruning_mib_total",
    "memory_after_benchmark_mib_total",
    "n_layers_pruned", "params_before", "params_after",
    "load_sec", "n_warmup", "n_bench",
    "model_name", "plan_path", "status",
]

rows = []
for jf in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
    try:
        with open(jf) as fh:
            rows.append(json.load(fh))
    except Exception as e:
        print(f"[speed] WARNING: cannot parse {jf}: {e}")

if not rows:
    print("[speed] WARNING: no JSON results found.")
    sys.exit(0)

os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
with open(out_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"[speed] CSV written: {out_csv}  ({len(rows)} rows)")

W = 68
print("\n[speed] Summary")
hdr = f"  {'Label':<{W}}  {'Param%':>7}  {'Prefill ms':>10}  {'Tok/s':>7}  {'PeakMiB':>8}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for r in rows:
    lbl = r.get("label", "?")[:W]
    st  = r.get("status", "?")
    if st != "ok":
        print(f"  {lbl:<{W}}  {st}")
        continue
    pct = r.get("total_model_param_reduction_pct", 0.0) or 0.0
    pre = r.get("prefill_latency_ms_mean", float("nan"))
    tps = r.get("tokens_per_sec_mean", float("nan"))
    mem = r.get("peak_allocated_mib_total", float("nan"))
    print(f"  {lbl:<{W}}  {pct:>7.2f}  {pre:>10.1f}  {tps:>7.1f}  {mem:>8.0f}")
print()
PYEOF

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "[speed] BENCHMARK COMPLETE  (id=${SWEEP_ID})"
echo "[speed]   Succeeded : ${SUCCEEDED}"
echo "[speed]   Skipped   : ${SKIPPED}  (missing plans)"
echo "[speed]   Failed    : ${FAILED}"
echo "[speed]   CSV       : ${OUT_CSV}"
echo "[speed]   Logs      : ${LOG_DIR}/"
echo "[speed]   Memory    : each setting ran in a separate Python process."
echo "══════════════════════════════════════════════════════════════════════════════"

if [ "${FAILED}" -gt 0 ] || [ "${SKIPPED}" -gt 0 ]; then
    exit 1
fi
exit 0
