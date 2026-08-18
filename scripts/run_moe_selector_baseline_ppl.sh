#!/usr/bin/env bash
# run_moe_selector_baseline_ppl.sh
#
# PPL-only selector attribution benchmark.
# Default: 4 selectors x 4 targets x wikitext2 = 16 runs.
# ONLY_SELECTORS/ONLY_TARGETS may define smaller controlled verification runs.
# All use: pure_delete method, moe_budget_mode=global (greedy global layer-channel selection;
# channel budgets -> actual_pct is selector-independent; only WHICH channels
# differ).
#
# Selectors compared:
#   rmsnorm_bound    -- weight-only RMSNorm-bounded SwiGLU score (proposed)
#   down_norm        -- L2 norm of each down_proj column (simple baseline)
#   activation_score -- SiLU(gate)*up activation x down-norm (calib-based)
#   random           -- uniform random (random baseline)
#
# Usage:
#   bash scripts/run_moe_selector_baseline_ppl.sh          # full 16-run
#   SMOKE=1 bash scripts/run_moe_selector_baseline_ppl.sh  # 2 runs (rmsnorm+random, target2)
#   DRY_RUN=1 bash scripts/run_moe_selector_baseline_ppl.sh
#   SUMMARIZE_ONLY=1 RUN_DIR=results/moe_selector_baselines/20260707_194241 \
#       bash scripts/run_moe_selector_baseline_ppl.sh
#
# Env overrides:
#   SMOKE=1                      2 runs: rmsnorm_bound+random, target2 only
#   DRY_RUN=1                    list planned runs, skip execution
#   SUMMARIZE_ONLY=1             skip runs; rebuild CSVs from existing raw files
#   RUN_DIR=...                  explicit run dir (required with SUMMARIZE_ONLY=1)
#   ONLY_SELECTORS=a,b,c         comma-separated selector filter
#   ONLY_TARGETS=2,4,6,8         comma-separated target pct filter (integers)
#   N_EVAL=512                   calibration/eval samples (must match config filename)
#   AUTO_GENERATE_CONFIGS=1      auto-generate missing configs before running (default: 1)
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
SUMMARIZE_ONLY="${SUMMARIZE_ONLY:-0}"
ONLY_SELECTORS="${ONLY_SELECTORS:-rmsnorm_bound,down_norm,activation_score,random}"
ONLY_TARGETS="${ONLY_TARGETS:-2,4,6,8}"
N_EVAL="${N_EVAL:-512}"
AUTO_GENERATE_CONFIGS="${AUTO_GENERATE_CONFIGS:-1}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
RESULTS_BASE="${RESULTS_BASE:-results/moe_selector_baselines}"
CONFIG_DIR="${CONFIG_DIR:-configs/moe_selector_baseline}"
PLAN_DIR="${PLAN_DIR:-results/pruning_plans}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
DATASET="wikitext2"
MODEL="Qwen/Qwen3-30B-A3B"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
# Allow RUN_DIR override for SUMMARIZE_ONLY mode; otherwise auto-generate
RUN_DIR="${RUN_DIR:-${RESULTS_BASE}/${RUN_ID}}"
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
if [ "${SUMMARIZE_ONLY}" = "1" ]; then
    echo "[ppl] Mode     : SUMMARIZE_ONLY"
    echo "[ppl] Run dir  : ${RUN_DIR}"
elif [ "${DRY_RUN}" = "1" ]; then
    echo "[ppl] Mode     : DRY_RUN (no execution)"
else
    echo "[ppl] Run dir  : ${RUN_DIR}"
fi
echo "[ppl] AutoGenCfg: ${AUTO_GENERATE_CONFIGS}"
echo "[ppl] ======================================================="

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

# ── SUMMARIZE_ONLY: rebuild CSVs from existing raw files; skip all runs ───────
if [ "${SUMMARIZE_ONLY}" = "1" ]; then
    if [ ! -d "${RUN_DIR}" ]; then
        echo "[ppl] ERROR: RUN_DIR=${RUN_DIR} does not exist."
        echo "[ppl]   Set RUN_DIR=results/moe_selector_baselines/<run_id>"
        exit 1
    fi
    EXPECTED_ROWS=$(( ${#_SEL_LIST[@]} * ${#_TGT_LIST[@]} ))
    echo "[ppl] Rebuilding summaries from ${RUN_DIR}"
    echo "[ppl] Expected rows: ${EXPECTED_ROWS}"
    python3 scripts/summarize_moe_selector_ppl.py \
        --run-dir      "${RUN_DIR}" \
        --model        "${MODEL}" \
        --dataset      "${DATASET}" \
        --selectors    "${ONLY_SELECTORS}" \
        --targets      "${ONLY_TARGETS}" \
        --orig-moe-dim 768 \
        --moe-align    16 \
        --min-rows     "${EXPECTED_ROWS}" \
        2>&1
    _rc=$?
    echo ""
    echo "[ppl] Summary CSV   : ${SUMMARY_CSV}"
    echo "[ppl] Attribution   : ${ATTRIBUTION_CSV}"
    exit ${_rc}
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
            elif [ "${AUTO_GENERATE_CONFIGS}" = "1" ]; then
                status="MISSING (will auto-generate)"
            else
                status="MISSING -- set AUTO_GENERATE_CONFIGS=1 or run: python3 scripts/generate_moe_selector_baseline_configs.py --n-eval ${N_EVAL} --dataset ${DATASET} --selectors ${ONLY_SELECTORS} --targets ${ONLY_TARGETS}"
            fi
            printf "  %2d. %-20s  target=%2s%%  [%s]\n" \
                "${n}" "${sel}" "${tgt}" "${status}"
        done
    done
    echo ""
    echo "[ppl] ${n} run(s) planned."
    echo "[ppl] Results -> ${RESULTS_BASE}/<run_id>/"
    echo "[ppl] Summary -> selector_baseline_summary.csv"
    echo "[ppl]         -> selector_attribution_summary.csv"
    exit 0
fi

mkdir -p "${RUN_DIR}"

# ── Auto-generate missing configs ─────────────────────────────────────────────
_NEED_SELECTORS="${ONLY_SELECTORS}"
_NEED_TARGETS="${ONLY_TARGETS}"
if [ "${SMOKE}" = "1" ]; then
    _NEED_SELECTORS="rmsnorm_bound,random"
    _NEED_TARGETS="2"
fi

_any_missing=0
for sel in "${_SEL_LIST[@]}"; do
    for tgt in "${_TGT_LIST[@]}"; do
        cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
        if [ ! -f "${cfg}" ]; then
            _any_missing=1
            break 2
        fi
    done
done

if [ "${_any_missing}" = "1" ]; then
    if [ "${AUTO_GENERATE_CONFIGS}" = "1" ]; then
        echo "[ppl] Missing configs detected -- auto-generating (N_EVAL=${N_EVAL}) ..."
        python3 scripts/generate_moe_selector_baseline_configs.py \
            --model      "${MODEL}" \
            --dataset    "${DATASET}" \
            --selectors  "${_NEED_SELECTORS}" \
            --targets    "${_NEED_TARGETS}" \
            --n-eval     "${N_EVAL}" \
            --config-dir "${CONFIG_DIR}" \
            --overwrite
        echo "[ppl] Config generation complete."
    else
        echo "[ppl] ERROR: missing configs and AUTO_GENERATE_CONFIGS is not set."
        echo "[ppl]   To fix, run:"
        echo "[ppl]     python3 scripts/generate_moe_selector_baseline_configs.py \\"
        echo "[ppl]         --n-eval ${N_EVAL} --dataset ${DATASET} \\"
        echo "[ppl]         --selectors ${_NEED_SELECTORS} \\"
        echo "[ppl]         --targets ${_NEED_TARGETS}"
        echo "[ppl]   Or re-run with AUTO_GENERATE_CONFIGS=1."
        exit 1
    fi
fi

# ── Pre-flight: validate all configs exist ────────────────────────────────────
_MISSING=0
for sel in "${_SEL_LIST[@]}"; do
    for tgt in "${_TGT_LIST[@]}"; do
        cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
        if [ ! -f "${cfg}" ]; then
            echo "[ppl] ERROR: config still missing after generation: ${cfg}"
            _MISSING=$(( _MISSING + 1 ))
        fi
    done
done
if [ "${_MISSING}" -gt 0 ]; then
    echo "[ppl] ERROR: ${_MISSING} config(s) still missing (generation may have failed)."
    exit 1
fi
echo "[ppl] Pre-flight OK: all configs found."

# ── Validate config correctness (moe_budget_mode=global required) ────────────
_validate_cfg() {
    python3 - "$1" "$2" << 'PYVAL'
import sys, yaml
path = sys.argv[1]
expected_selector = sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f)
bad = []
if cfg.get("moe_budget_mode") not in ("global", None):
    bad.append("moe_budget_mode must be 'global' (or None/missing), got: " + repr(cfg.get("moe_budget_mode")))
if cfg.get("scaling_methods") != ["pure_delete"]:
    bad.append("scaling_methods must be ['pure_delete'], got: " + repr(cfg.get("scaling_methods")))
if cfg.get("moe_selector") != expected_selector:
    bad.append("moe_selector must match requested selector " + repr(expected_selector) + ", got: " + repr(cfg.get("moe_selector")))
if cfg.get("moe_pruning_mode") != "packed_same_channel":
    bad.append("moe_pruning_mode must be 'packed_same_channel', got: " + repr(cfg.get("moe_pruning_mode")))
if cfg.get("moe_same_channel_aggregation") != "p95":
    bad.append("moe_same_channel_aggregation must be 'p95', got: " + repr(cfg.get("moe_same_channel_aggregation")))
if int(cfg.get("moe_channel_alignment", -1)) != 16:
    bad.append("moe_channel_alignment must be 16, got: " + repr(cfg.get("moe_channel_alignment")))
if float(cfg.get("max_expert_frac", -1)) != 0.2:
    bad.append("max_expert_frac must be 0.2, got: " + repr(cfg.get("max_expert_frac")))
if bad:
    for b in bad: print("[ppl] CONFIG ERROR:", b)
    sys.exit(1)
PYVAL
}

for sel in "${_SEL_LIST[@]}"; do
    for tgt in "${_TGT_LIST[@]}"; do
        cfg="${CONFIG_DIR}/qwen3_30b_a3b_${DATASET}_n${N_EVAL}_target${tgt}_sel_${sel}.yaml"
        _validate_cfg "${cfg}" "${sel}" || exit 1
    done
done
echo "[ppl] Config validation OK (moe_budget_mode=global confirmed for all)."

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

        # Create temp config with output_dir overridden to run_out
        python3 - "${base_cfg}" "${run_out}" "${PLAN_DIR}" "${tmp_cfg}" << 'PYCFG'
import sys, yaml, os
src, out_dir, plan_dir, dst = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = out_dir
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
        _n_csvs=$(find "${run_out}" -name "moe_target_pruning_*.csv" \
                  ! -name "*_per_layer.csv" 2>/dev/null | wc -l)
        if [ "${_n_csvs}" -eq 0 ]; then
            echo "[ppl] ERROR: run exited 0 but no moe_target_pruning_*.csv in ${run_out}"
            FAILED=$(( FAILED + 1 ))
            [ "${CONTINUE_ON_FAIL}" = "1" ] && continue || exit 1
        fi

        echo "[ppl] OK: ${label} (${_n_csvs} CSV(s) written)"

        # ── Per-run removal distribution diagnostic ───────────────────────────
        _per_layer_csv=$(find "${run_out}" -name "*_per_layer.csv" | head -1)
        if [ -n "${_per_layer_csv}" ]; then
            echo "[ppl] --- Removal distribution for ${label} ---"
            python3 - "${_per_layer_csv}" << 'PYDIAG'
import sys, csv, collections
path = sys.argv[1]
removed = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        # Try explicit column first, then derive from old-new
        for col in ("removed", "channels_removed", "num_removed"):
            if col in row and str(row[col]).strip():
                try:
                    removed.append(int(float(row[col])))
                    break
                except (ValueError, TypeError):
                    pass
        else:
            old_i = row.get("old_intermediate", row.get("original_intermediate", ""))
            new_i = row.get("new_intermediate", row.get("new_i", ""))
            try:
                removed.append(int(float(old_i)) - int(float(new_i)))
            except (ValueError, TypeError):
                pass
if not removed:
    print("[ppl]   (no removal data in per-layer CSV)")
else:
    counts = collections.Counter(removed)
    print("[ppl]   removed per layer value counts:")
    for v, c in sorted(counts.items()):
        print(f"[ppl]     {v:4d}   {c} layers")
    total    = sum(v * c for v, c in counts.items())
    n_zero   = counts.get(0, 0)
    n_nz     = sum(c for v, c in counts.items() if v > 0)
    print(f"[ppl]   total removed layer-channels : {total}")
    print(f"[ppl]   layers with zero pruning     : {n_zero}")
    print(f"[ppl]   layers with nonzero pruning  : {n_nz}")
    print(f"[ppl]   max removed in one layer     : {max(removed)}")
PYDIAG
        fi

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
MIN_ROWS="${EXPECTED_ROWS}"

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
