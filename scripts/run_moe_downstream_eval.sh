#!/usr/bin/env bash
# run_moe_downstream_eval.sh
#
# Run lm-evaluation-harness (lm_eval) on baseline and pruned MoE models.
#
# Evaluates on: arc_easy, arc_challenge, hellaswag, winogrande, mmlu (opt.)
#
# Usage:
#   bash scripts/run_moe_downstream_eval.sh
#   DRY_RUN=1         bash scripts/run_moe_downstream_eval.sh
#   SMOKE=1           bash scripts/run_moe_downstream_eval.sh   # limit 50
#   CHECK_DEPS=1      bash scripts/run_moe_downstream_eval.sh
#   INSTALL_LM_EVAL=1 CHECK_DEPS=1 bash scripts/run_moe_downstream_eval.sh
#   INSTALL_LM_EVAL=1 SMOKE=1 bash scripts/run_moe_downstream_eval.sh
#
# Exit codes:
#   0   success (or CHECK_DEPS passed)
#   1   evaluation failure (plan missing / lm_eval error / no results)
#   2   dependency missing (lm_eval not installed)
#
# Env overrides:
#   DRY_RUN=1           List settings only, no GPU
#   SMOKE=1             baseline+2%/wikitext2 arc_easy with --limit 50
#   CHECK_DEPS=1        Check Python + lm_eval + CUDA; exit 0 if OK, 1 if not
#   INSTALL_LM_EVAL=1   Auto-install lm-evaluation-harness before running
#   RESULTS_DIR=...     Results directory (default: results)
#   MODEL=...           HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)
#   DTYPE=...           bfloat16 | float16 | float32 (default: bfloat16)
#   VENV=...            Virtualenv path (default: /workspace/venvs/qwen-pruning)
#   NUM_FEWSHOT=...     Few-shot examples (default: 0)
#   SKIP_MMLU=1         Skip MMLU (default: 1)
#   BATCH_SIZE=...      lm_eval batch size (default: 4)
#   SMOKE_LIMIT=...     --limit N used in SMOKE mode (default: 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CHECK_DEPS="${CHECK_DEPS:-0}"
INSTALL_LM_EVAL="${INSTALL_LM_EVAL:-0}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
DTYPE="${DTYPE:-bfloat16}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
SKIP_MMLU="${SKIP_MMLU:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SMOKE_LIMIT="${SMOKE_LIMIT:-50}"
CHECK_LOGITS_DIFF="${CHECK_LOGITS_DIFF:-0}"
DOWNSTREAM_METHOD="${DOWNSTREAM_METHOD:-unknown}"
INCLUDE_RESIDUAL="${INCLUDE_RESIDUAL:-0}"
RESIDUAL_METHOD="${RESIDUAL_METHOD:-residual_nearest_channel_merge_moe}"
MODEL_MOE_DIM="${MODEL_MOE_DIM:-768}"
MOE_ALIGN="${MOE_ALIGN:-16}"
SUMMARIZE_ONLY="${SUMMARIZE_ONLY:-0}"
RUN_DIR="${RUN_DIR:-}"
ONLY_METHODS="${ONLY_METHODS:-}"
ONLY_TARGETS="${ONLY_TARGETS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
AUTO_GENERATE_PLAN="${AUTO_GENERATE_PLAN:-0}"
CONFIG_PREFIX="$(echo "${MODEL}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-0}"

MODEL_SLUG="$(echo "${MODEL}" | tr '/' '_' | tr '-' '_')"
PLAN_DIR="${RESULTS_DIR}/pruning_plans"

SELECTOR="rmsnorm_bound"
AGG_MODE="p95"
ALIGN="16"
N_EVAL="512"
CALIB_N="512"

# Tasks (comma-separated for lm_eval)
if [ "${SMOKE}" = "1" ]; then
    TASKS="arc_easy"
elif [ "${SKIP_MMLU}" = "1" ]; then
    TASKS="arc_easy,arc_challenge,hellaswag,winogrande"
else
    TASKS="arc_easy,arc_challenge,hellaswag,winogrande,mmlu"
fi

# lm_eval --limit flag (only in SMOKE mode)
LIMIT_ARGS=""
if [ "${SMOKE}" = "1" ]; then
    LIMIT_ARGS="--limit ${SMOKE_LIMIT}"
fi

# ── Sweep ID / Run directory ───────────────────────────────────────────────────
if [ -n "${RUN_DIR}" ] && [ "${SUMMARIZE_ONLY}" != "1" ]; then
    OUT_DIR="${RUN_DIR}"
    SWEEP_ID="$(basename "${RUN_DIR}")"
    echo "[eval] RUN_DIR set: reusing existing run directory ${OUT_DIR}"
else
    SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
    OUT_DIR="${RESULTS_DIR}/downstream_eval_runs/${SWEEP_ID}"
fi
SUMMARY_CSV="${OUT_DIR}/downstream_summary.csv"
CKPT_BASE_DIR="${OUT_DIR}/pruned_checkpoints"

echo "[eval] Downstream eval ID: ${SWEEP_ID}"
echo "[eval] Model:  ${MODEL}"
echo "[eval] Tasks:  ${TASKS}"
if [ "${SMOKE}" = "1" ]; then
    echo "[eval] SMOKE=1: using --limit ${SMOKE_LIMIT}"
    echo "[eval] SMOKE=1 uses --limit ${SMOKE_LIMIT} for plumbing verification only; do not interpret these as real downstream metrics."
fi

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    echo "[eval] Activating virtualenv: ${VENV}"
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "[eval] WARNING: No virtualenv at ${VENV}, using system Python"
fi

# Print Python/pip diagnostic paths (reflects activated venv)
echo "[eval] Python/pip paths:"
echo "[eval]   which python   : $(which python  2>/dev/null || echo 'NOT FOUND')"
echo "[eval]   python -V      : $(python -V 2>&1 || echo 'NOT FOUND')"
echo "[eval]   sys.executable : $(python -c 'import sys; print(sys.executable)' 2>/dev/null || echo 'NOT FOUND')"
echo "[eval]   python -m pip  : $(python -m pip -V 2>&1 || echo 'NOT FOUND')"

# ── Helper: check lm_eval by Python import (not just CLI presence) ────────────
_check_lm_eval_import() {
    python - << 'PY'
import importlib.util, sys
spec = importlib.util.find_spec("lm_eval")
if spec is None:
    sys.exit(1)
print("lm_eval found:", spec.origin)
PY
}

# ── Helper: print debug info when lm_eval is missing ─────────────────────────
_debug_lm_eval_missing() {
    echo "[eval] -------------------------------------------------------------------"
    echo "[eval] lm-evaluation-harness is NOT installed in the active Python."
    echo "[eval]"
    echo "[eval] pip show results:"
    python -m pip show lm-eval               2>/dev/null || true
    python -m pip show lm_eval               2>/dev/null || true
    python -m pip show lm-evaluation-harness 2>/dev/null || true
    echo "[eval]"
    echo "[eval] To install manually:"
    echo "[eval]   source ${VENV}/bin/activate"
    echo "[eval]   python -m pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git"
    echo "[eval]"
    echo "[eval] Or re-run with auto-install:"
    echo "[eval]   INSTALL_LM_EVAL=1 SMOKE=1 bash scripts/run_moe_downstream_eval.sh"
    echo "[eval] -------------------------------------------------------------------"
}

# ── Optional auto-install ─────────────────────────────────────────────────────
if [ "${INSTALL_LM_EVAL}" = "1" ]; then
    echo "[eval] INSTALL_LM_EVAL=1: installing lm-evaluation-harness ..."
    python -m pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
    echo "[eval] Install complete. Re-running import check ..."
fi

# ── CHECK_DEPS mode ───────────────────────────────────────────────────────────
if [ "${CHECK_DEPS}" = "1" ]; then
    echo "[eval] CHECK_DEPS=1: checking all dependencies ..."
    _deps_ok=1

    echo ""
    echo "[eval] --- lm_eval import ---"
    if _check_lm_eval_import; then
        echo "[eval]   lm_eval: OK"
    else
        echo "[eval]   lm_eval: MISSING"
        _debug_lm_eval_missing
        _deps_ok=0
    fi

    echo ""
    echo "[eval] --- torch + CUDA ---"
    set +e
    python - << 'PY'
import sys
try:
    import torch
    avail = torch.cuda.is_available()
    ndev  = torch.cuda.device_count()
    print(f"[eval]   torch         : {torch.__version__}")
    print(f"[eval]   cuda_available: {avail}")
    print(f"[eval]   device_count  : {ndev}")
    sys.exit(0 if avail else 2)
except ImportError as e:
    print(f"[eval]   torch: NOT INSTALLED ({e})")
    sys.exit(1)
PY
    _cuda_exit=$?
    set -e
    if [ "${_cuda_exit}" -ne 0 ]; then
        if [ "${_cuda_exit}" -eq 2 ]; then
            echo "[eval]   CUDA not available (CPU-only environment)"
        else
            echo "[eval]   torch: MISSING"
        fi
        _deps_ok=0
    fi

    echo ""
    if [ "${_deps_ok}" = "1" ]; then
        echo "[eval] CHECK_DEPS: all dependencies OK."
        exit 0
    else
        echo "[eval] CHECK_DEPS: one or more dependencies missing or unavailable."
        exit 1
    fi
fi

# ── Build settings array ──────────────────────────────────────────────────────
# Format: "label|plan_path|method|selector|dataset|target_pct"
# actual_pct is computed at result-parse time from the plan JSON itself.
declare -a SETTINGS=()
SETTINGS+=("baseline_no_pruning|NONE|baseline|none|none|0.0")

# Helper: compute actual pruned pct from a plan JSON (returns "NA" if plan missing/unreadable)
_plan_actual_pct() {
    local pf="$1"
    [ -f "${pf}" ] || { echo "NA"; return; }
    python - "${pf}" << 'PYSMALL'
import json, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    t = sum(l.get("old_intermediate", 0) for l in d.get("layers", []))
    p = sum(len(l.get("prune_idx", [])) for l in d.get("layers", []))
    print(f"{100.*p/max(t,1):.1f}" if t else "NA")
except Exception: print("NA")
PYSMALL
}

# Helper: read pruning method name for the label.
# Priority: DOWNSTREAM_METHOD env (if not "unknown") > plan JSON "method" field > "pure_delete".
_plan_method() {
    local pf="$1"
    if [ "${DOWNSTREAM_METHOD}" != "unknown" ] && [ -n "${DOWNSTREAM_METHOD}" ]; then
        echo "${DOWNSTREAM_METHOD}"
        return
    fi
    [ -f "${pf}" ] || { echo "pure_delete"; return; }
    python - "${pf}" << 'PYMETHOD'
import json, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    m = (d.get("method") or d.get("pruning_method") or "").strip()
    print(m if m else "pure_delete")
except Exception: print("pure_delete")
PYMETHOD
}

# ── Helper: check if a setting already has lm_eval JSON output ───────────────
_setting_has_output() {
    local _lbl="$1"
    local _lm_out="${OUT_DIR}/${_lbl}_lm_eval"
    if [ -d "${_lm_out}" ] && [ "$(find "${_lm_out}" -name "*.json" 2>/dev/null | wc -l)" -gt "0" ]; then
        return 0
    fi
    return 1
}

# ── Helper: apply ONLY_METHODS / ONLY_TARGETS filter ─────────────────────────
# Returns 0 (pass) or 1 (filtered out).
# Baseline (method=baseline) always passes.
_setting_passes_filter() {
    local _fm="$1"
    local _ft="$2"   # target_pct string, e.g. "2.0" or "6.0"

    [ "${_fm}" = "baseline" ] && return 0

    if [ -n "${ONLY_METHODS}" ]; then
        local _found=0
        local _om
        IFS=',' read -ra _oms <<< "${ONLY_METHODS}"
        for _om in "${_oms[@]}"; do
            [ "${_om}" = "${_fm}" ] && _found=1 && break
        done
        [ "${_found}" = "0" ] && return 1
    fi

    if [ -n "${ONLY_TARGETS}" ]; then
        local _tint="${_ft%%.*}"   # "2.0" -> "2"
        local _found=0
        local _ot
        IFS=',' read -ra _ots <<< "${ONLY_TARGETS}"
        for _ot in "${_ots[@]}"; do
            [ "${_ot}" = "${_tint}" ] && _found=1 && break
        done
        [ "${_found}" = "0" ] && return 1
    fi

    return 0
}

if [ "${SMOKE}" = "1" ]; then
    plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_2.0pct_align${ALIGN}.json"
    _apct="$(_plan_actual_pct "${plan}")"
    _meth="$(_plan_method "${plan}")"
    label="${_meth}__${SELECTOR}__wikitext2__target2pct__actual${_apct}pct"
    SETTINGS+=("${label}|${plan}|${_meth}|${SELECTOR}|wikitext2|2.0")
else
    for target in 2 4 6 8; do
        plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target}.0pct_align${ALIGN}.json"
        _apct="$(_plan_actual_pct "${plan}")"
        _meth="$(_plan_method "${plan}")"
        label="${_meth}__${SELECTOR}__wikitext2__target${target}pct__actual${_apct}pct"
        SETTINGS+=("${label}|${plan}|${_meth}|${SELECTOR}|wikitext2|${target}.0")
    done
fi

# ── Optional residual settings (INCLUDE_RESIDUAL=1) ─────────────────────────
if [ "${INCLUDE_RESIDUAL}" = "1" ] && [ "${SMOKE}" != "1" ]; then
    echo "[eval] INCLUDE_RESIDUAL=1: adding residual settings (2%, 4%, 6%, 8%)"
    for target in 2 4 6 8; do
        r_plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target}.0pct_align${ALIGN}.json"
        _rapct="$(_plan_actual_pct "${r_plan}")"
        r_label="${RESIDUAL_METHOD}__${SELECTOR}__wikitext2__target${target}pct__actual${_rapct}pct"
        SETTINGS+=("${r_label}|${r_plan}|${RESIDUAL_METHOD}|${SELECTOR}|wikitext2|${target}.0")
    done
else
    if [ "${SMOKE}" != "1" ]; then
        echo "[eval] INCLUDE_RESIDUAL=0: residual settings skipped (set INCLUDE_RESIDUAL=1 to enable)"
    fi
fi

# ── DRY_RUN: list settings and exit ──────────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[eval] Planned settings (${#SETTINGS[@]} total):"
    [ -n "${RUN_DIR}" ] && [ -d "${OUT_DIR}" ] && echo "[eval] RUN_DIR: reusing ${OUT_DIR}"
    [ -n "${ONLY_METHODS}" ] && echo "[eval] ONLY_METHODS filter : ${ONLY_METHODS}"
    [ -n "${ONLY_TARGETS}" ] && echo "[eval] ONLY_TARGETS filter : ${ONLY_TARGETS}"
    [ "${SKIP_EXISTING}" = "1" ] && echo "[eval] SKIP_EXISTING=1: completed settings will be skipped"
    n=0
    for setting in "${SETTINGS[@]}"; do
        n=$(( n + 1 ))
        IFS='|' read -r label plan method selector dataset target_pct <<< "${setting}"
        if ! _setting_passes_filter "${method}" "${target_pct}"; then
            printf "  %2d. %-65s  [filtered]\n" "${n}" "${label}"
            continue
        fi
        _dr_skip=""
        if [ "${SKIP_EXISTING}" = "1" ] && _setting_has_output "${label}"; then
            _dr_skip="[SKIP: output exists]"
        fi
        _dr_plan=""
        if [ "${plan}" = "NONE" ]; then
            _dr_plan="(baseline)"
        elif [ -f "${plan}" ]; then
            _dr_plan="[plan ok]"
        elif [ "${AUTO_GENERATE_PLAN}" = "1" ]; then
            _dr_plan="[plan missing → auto-generate]"
        else
            _dr_plan="[PLAN MISSING]"
        fi
        if [ -n "${_dr_skip}" ]; then
            printf "  %2d. %-65s  %s\n" "${n}" "${label}" "${_dr_skip}"
        else
            printf "  %2d. %-65s  %-35s → RUN\n" "${n}" "${label}" "${_dr_plan}"
        fi
    done
    echo ""
    echo "[eval] Tasks: ${TASKS}"
    if [ "${SMOKE}" = "1" ]; then
        echo "[eval] Limit: --limit ${SMOKE_LIMIT}"
    fi
    echo ""
    echo "[eval] DRY_RUN complete."
    echo "[eval] Missing plans: bash scripts/run_moe_residual_selected_full_benchmark.sh"
    exit 0
fi

# ── SUMMARIZE_ONLY mode: rebuild summaries from existing lm_eval outputs ────────
if [ "${SUMMARIZE_ONLY}" = "1" ]; then
    if [ -z "${RUN_DIR}" ]; then
        echo "[eval] ERROR: SUMMARIZE_ONLY=1 requires RUN_DIR=<run_dir_path>"
        exit 1
    fi
    if [ ! -d "${RUN_DIR}" ]; then
        echo "[eval] ERROR: RUN_DIR not found: ${RUN_DIR}"
        exit 1
    fi
    echo "[eval] SUMMARIZE_ONLY=1: rebuilding summaries from ${RUN_DIR} ..."
    python3 scripts/downstream_eval_summarize.py \
        --run-dir    "${RUN_DIR}" \
        --summarize-only \
        --plan-dir   "${PLAN_DIR}" \
        --orig-moe-dim "${MODEL_MOE_DIM}" \
        --moe-align  "${MOE_ALIGN}" \
        --model      "${MODEL}"
    exit $?
fi

# ── Require lm_eval -- exit 2 if missing ─────────────────────────────────────
echo "[eval] Checking lm_eval import ..."
if ! _check_lm_eval_import; then
    _debug_lm_eval_missing
    echo "[eval] ERROR: lm_eval not installed. Cannot run evaluation."
    echo "[eval] Exiting with code 2 (dependency missing)."
    exit 2
fi

# ── Determine lm_eval launch command ─────────────────────────────────────────
# Prefer 'python -m lm_eval' (guaranteed to use the active venv Python).
LM_EVAL_CMD=""
set +e
python -m lm_eval --help >/dev/null 2>&1
_mlm_exit=$?
set -e

if [ "${_mlm_exit}" -eq 0 ]; then
    LM_EVAL_CMD="python -m lm_eval"
    echo "[eval] lm_eval command: python -m lm_eval  (module, active venv)"
elif command -v lm_eval >/dev/null 2>&1; then
    LM_EVAL_CMD="$(command -v lm_eval)"
    echo "[eval] lm_eval command: ${LM_EVAL_CMD}  (CLI entry point)"
else
    echo "[eval] ERROR: lm_eval import passed but module and CLI both unreachable."
    exit 2
fi

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "${OUT_DIR}"
echo "[eval] Output dir: ${OUT_DIR}"

# ── Write CSV header (skip if already exists — RUN_DIR append mode) ──────────
python - "${SUMMARY_CSV}" << 'PYEOF'
import csv, os, sys
out_path = sys.argv[1]
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
if os.path.isfile(out_path):
    print("[eval] CSV exists (append mode): " + out_path)
    sys.exit(0)
fields = [
    "setting_label", "method", "selector", "dataset",
    "target_pct", "actual_pct", "moe_dim",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "pruning_plan_path", "model_path", "is_pruned",
    "requested_method", "actual_method", "residual_applied", "residual_fallback_used",
    "task", "metric", "value", "stderr",
    "num_fewshot", "limit", "batch_size", "status",
]
with open(out_path, "w", newline="") as fh:
    csv.DictWriter(fh, fieldnames=fields).writeheader()
print("[eval] CSV header written: " + out_path)
PYEOF

# ── Run each setting ──────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0

for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label plan method selector dataset target_pct <<< "${setting}"
    out_json="${OUT_DIR}/${label}.json"
    log_file="${OUT_DIR}/${label}.log"

    # ── Filter by ONLY_METHODS / ONLY_TARGETS ────────────────────────────────
    if ! _setting_passes_filter "${method}" "${target_pct}"; then
        echo "[eval] Skipping (filtered): ${label}"
        continue
    fi

    # ── SKIP_EXISTING: skip settings that already have output ─────────────────
    if [ "${SKIP_EXISTING}" = "1" ] && _setting_has_output "${label}"; then
        echo "[eval] Skipping (output exists): ${label}"
        SUCCEEDED=$(( SUCCEEDED + 1 ))
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[eval] Setting : ${label}"
    echo "[eval]   method=${method}  selector=${selector}  dataset=${dataset}  target=${target_pct}%"
    echo "================================================================"

    if [ "${plan}" = "NONE" ]; then
        eval_model="${MODEL}"
        echo "[eval] Using baseline model (no pruning)."
    else
        if [ ! -f "${plan}" ]; then
            if [ "${AUTO_GENERATE_PLAN}" = "1" ]; then
                _local_target_int="${target_pct%%.*}"
                _local_cfg="configs/moe_selector_baseline/${CONFIG_PREFIX}_wikitext2_n${N_EVAL}_target${_local_target_int}_sel_${SELECTOR}.yaml"
                echo "[eval] AUTO_GENERATE_PLAN=1: plan missing, generating ..."
                echo "[eval]   Config: ${_local_cfg}"
                echo "[eval]   Plan  : ${plan}"
                if [ ! -f "${_local_cfg}" ]; then
                    echo "[eval]   Config not found; running generate_moe_selector_baseline_configs.py ..."
                    python3 scripts/generate_moe_selector_baseline_configs.py || {
                        echo "[eval] ERROR: failed to generate selector-baseline configs."
                        FAILED=$(( FAILED + 1 ))
                        continue
                    }
                fi
                if [ ! -f "${_local_cfg}" ]; then
                    echo "[eval] ERROR: config still missing: ${_local_cfg}"
                    FAILED=$(( FAILED + 1 ))
                    continue
                fi
                set +e
                python3 run_experiment.py                     --config "${_local_cfg}"                     --moe-target-pruning                     2>&1 | tee "${OUT_DIR}/${label}_plan_gen.log"
                _gen_exit="${PIPESTATUS[0]}"
                set -e
                if [ "${_gen_exit}" -ne 0 ] || [ ! -f "${plan}" ]; then
                    echo "[eval] ERROR: plan generation failed for ${label}"
                    FAILED=$(( FAILED + 1 ))
                    continue
                fi
                echo "[eval] Plan generated: ${plan}"
            else
                echo "[eval] ERROR: plan not found: ${plan}"
                echo "[eval]   Generate plans: bash scripts/run_moe_residual_selected_full_benchmark.sh"
                echo "[eval]   Or retry with: AUTO_GENERATE_PLAN=1"
                FAILED=$(( FAILED + 1 ))
                continue
            fi
        fi
        SETTING_CKPT_DIR="${CKPT_BASE_DIR}/${label}"
        echo "[eval] Applying pruning plan (method=${method}): ${plan}"
        mkdir -p "${CKPT_BASE_DIR}"
        set +e
        python3 scripts/apply_moe_plan_save_checkpoint.py \
            --model    "${MODEL}" \
            --plan     "${plan}" \
            --method   "${method}" \
            --ckpt-dir "${SETTING_CKPT_DIR}" \
            --dtype    "${DTYPE}" \
            --label    "${label}" \
            --calib-n  64 \
            2>&1 | tee "${OUT_DIR}/${label}_apply.log"
        _apply_exit="${PIPESTATUS[0]}"
        set -e
        if [ "${_apply_exit}" -ne 0 ]; then
            echo "[eval] ERROR: checkpoint creation failed for ${label} (exit ${_apply_exit})"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        eval_model="${SETTING_CKPT_DIR}"
        echo "[eval] Pruned model saved to: ${eval_model}"

        # ── Pruned model sanity checks (read pruning_metadata.json) ──────────
        echo "[eval] --- Pruned model sanity checks ---"
        python - "${SETTING_CKPT_DIR}" "${plan}" << 'PYSANITY'
import json, glob, os, sys
save_dir  = sys.argv[1]
plan_path = sys.argv[2]

pm_path  = os.path.join(save_dir, "pruning_metadata.json")
cfg_path = os.path.join(save_dir, "config.json")
orig_moe = saved_moe = hidden_size = "?"
requested_method = actual_method = "?"
residual_applied = residual_fallback = weight_hash = "?"

if os.path.isfile(pm_path):
    try:
        with open(pm_path) as f:
            pm = json.load(f)
        saved_moe         = pm.get("saved_moe_intermediate_size", "?")
        orig_moe          = pm.get("original_moe_intermediate_size", "?")
        requested_method  = pm.get("requested_method", "?")
        actual_method     = pm.get("actual_method", "?")
        residual_applied  = str(pm.get("residual_applied", "?"))
        residual_fallback = str(pm.get("residual_fallback_used", "?"))
        weight_hash       = pm.get("weight_hash", "?")
    except Exception as e:
        print("[eval]   WARNING: could not read pruning_metadata.json: " + str(e))

if os.path.isfile(cfg_path):
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        if saved_moe == "?":
            saved_moe = cfg.get("moe_intermediate_size", "?")
        hidden_size = cfg.get("hidden_size", "?")
    except Exception:
        pass

if orig_moe == "?" and os.path.isfile(plan_path):
    try:
        with open(plan_path) as f:
            plan_data = json.load(f)
        for lc in plan_data.get("layers", []):
            old = lc.get("old_intermediate")
            if old:
                orig_moe = old
                break
    except Exception:
        pass

shard_files = (
    glob.glob(os.path.join(save_dir, "model*.safetensors")) +
    glob.glob(os.path.join(save_dir, "pytorch_model*.bin"))
)
n_shards   = len(shard_files)
gate_shape = "[" + str(saved_moe) + ", " + str(hidden_size) + "]"
up_shape   = "[" + str(saved_moe) + ", " + str(hidden_size) + "]"
down_shape = "[" + str(hidden_size) + ", " + str(saved_moe) + "]"

print("[eval]   pruned model path              : " + save_dir)
print("[eval]   original moe_intermediate_size : " + str(orig_moe))
print("[eval]   saved moe_intermediate_size    : " + str(saved_moe))
print("[eval]   sample gate_proj.weight shape  : " + gate_shape)
print("[eval]   sample up_proj.weight shape    : " + up_shape)
print("[eval]   sample down_proj.weight shape  : " + down_shape)
print("[eval]   checkpoint shards              : " + str(n_shards))
print("[eval]   requested_method               : " + str(requested_method))
print("[eval]   actual_method                  : " + str(actual_method))
print("[eval]   residual_applied               : " + str(residual_applied))
print("[eval]   residual_fallback_used         : " + str(residual_fallback))
print("[eval]   weight_hash                    : " + str(weight_hash))
if str(residual_fallback) == "True":
    print("[eval]   WARNING: residual not applied -- results are equivalent to pure_delete")
PYSANITY
    fi

    # ── Optional logits-difference sanity check ───────────────────────────────
    if [ "${plan}" != "NONE" ] && [ "${CHECK_LOGITS_DIFF}" = "1" ]; then
        echo "[eval] CHECK_LOGITS_DIFF=1: comparing baseline vs pruned logits ..."
        set +e
        python - "${MODEL}" "${eval_model}" "${DTYPE}" << 'PYLOGITS'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_name, pruned_dir, dtype_str = sys.argv[1:4]
# Use float32 for clean numeric diff comparison regardless of model dtype
dtype = torch.float32
PROMPT = "Question: What is the capital of France?\nAnswer:"

print(f"[logits_diff] Prompt: {repr(PROMPT)}")
tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
inputs = tok(PROMPT, return_tensors="pt")

print("[logits_diff] Loading baseline model (CPU, float32) ...")
base_m = AutoModelForCausalLM.from_pretrained(
    base_name, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
)
base_m.eval()
with torch.no_grad():
    base_logits = base_m(**inputs).logits[0, -1, :].clone()
del base_m

print(f"[logits_diff] Loading pruned model from {pruned_dir} (CPU, float32) ...")
pruned_m = AutoModelForCausalLM.from_pretrained(
    pruned_dir, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
)
pruned_m.eval()
with torch.no_grad():
    pruned_logits = pruned_m(**inputs).logits[0, -1, :].clone()
del pruned_m

n = min(base_logits.shape[0], pruned_logits.shape[0])
diff = (base_logits[:n] - pruned_logits[:n]).abs()
mean_diff     = diff.mean().item()
max_diff      = diff.max().item()
exactly_equal = bool((base_logits[:n] == pruned_logits[:n]).all())

print(f"[logits_diff] mean_abs_diff : {mean_diff:.6f}")
print(f"[logits_diff] max_abs_diff  : {max_diff:.6f}")
print(f"[logits_diff] exactly_equal : {exactly_equal}")

if exactly_equal:
    sys.exit(
        "ERROR: baseline and pruned logits are EXACTLY EQUAL -- "
        "the pruned model is likely not being applied correctly."
    )
print("[logits_diff] OK: logits differ between baseline and pruned model.")
PYLOGITS
        _logits_exit="${PIPESTATUS[0]}"
        set -e
        if [ "${_logits_exit}" -ne 0 ]; then
            echo "[eval] ERROR: logits-diff check failed for ${label}"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        echo "[eval] Logits-diff check: PASSED"
    fi

    lm_out="${OUT_DIR}/${label}_lm_eval"
    mkdir -p "${lm_out}"

    echo "[eval] lm_eval command : ${LM_EVAL_CMD}"
    echo "[eval] eval model path : ${eval_model}"
    echo "[eval] tasks           : ${TASKS}"
    if [ "${SMOKE}" = "1" ]; then
        echo "[eval] limit           : ${SMOKE_LIMIT}"
    fi

    set +e
    # shellcheck disable=SC2086
    ${LM_EVAL_CMD} \
        --model hf \
        --model_args "pretrained=${eval_model},dtype=${DTYPE},trust_remote_code=True" \
        --tasks "${TASKS}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${lm_out}" \
        ${LIMIT_ARGS} \
        2>&1 | tee "${log_file}"
    lm_exit=${PIPESTATUS[0]}
    set -e

    if [ ${lm_exit} -ne 0 ]; then
        echo "[eval] ERROR: lm_eval exited ${lm_exit} for ${label}"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # Verify at least one result JSON was written (search recursively)
    lm_result_count=$(find "${lm_out}" -name "*.json" 2>/dev/null | wc -l)
    if [ "${lm_result_count}" -eq 0 ]; then
        echo "[eval] ERROR: lm_eval exited 0 but no JSON results in ${lm_out}"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # Parse lm_eval JSON results and append to summary CSV
    _limit_now="none"
    [ "${SMOKE}" = "1" ] && _limit_now="${SMOKE_LIMIT}"
    python - "${lm_out}" "${label}" "${plan}" \
        "${MODEL}" "${NUM_FEWSHOT}" "${SUMMARY_CSV}" \
        "${method}" "${selector}" "${dataset}" "${target_pct}" \
        "${eval_model}" "${_limit_now}" "${BATCH_SIZE}" \
        "${MODEL_MOE_DIM}" "${MOE_ALIGN}" << 'PYEOF'
import csv, json, os, sys, glob

(results_dir, label, plan, model_name,
 num_fewshot, summary_csv,
 method, selector, dataset, target_pct,
 eval_model_dir, limit_val, batch_size_val,
 orig_moe_dim_str, moe_align_str) = sys.argv[1:16]

orig_moe_dim = int(orig_moe_dim_str)
moe_align    = int(moe_align_str)
is_pruned    = (plan != "NONE")

# Compute actual_pct from plan JSON
actual_pct = 0.0
if is_pruned and os.path.isfile(plan):
    try:
        with open(plan) as pf:
            plan_data = json.load(pf)
        total  = sum(lc.get("old_intermediate", 0) for lc in plan_data.get("layers", []))
        pruned = sum(len(lc.get("prune_idx", []))   for lc in plan_data.get("layers", []))
        if total > 0:
            actual_pct = round(100.0 * pruned / total, 3)
    except Exception as e:
        print("[eval] WARNING: cannot compute actual_pct: " + str(e))

# Read pruning_metadata.json for actual_method and residual status
requested_method_val   = method
actual_method_val      = method
residual_applied_val   = False
residual_fallback_val  = False
if is_pruned:
    pm_path = os.path.join(eval_model_dir, "pruning_metadata.json")
    if os.path.isfile(pm_path):
        try:
            with open(pm_path) as pmf:
                pm = json.load(pmf)
            requested_method_val  = pm.get("requested_method", method)
            actual_method_val     = pm.get("actual_method", method)
            residual_applied_val  = pm.get("residual_applied", False)
            residual_fallback_val = pm.get("residual_fallback_used", False)
        except Exception as e:
            print("[eval] WARNING: could not read pruning_metadata.json: " + str(e))
    else:
        # Old checkpoint without pruning_metadata.json; assume pure_delete
        actual_method_val = "pure_delete"

# Infer moe_dim: prefer saved config.json, else compute from actual_pct
moe_dim = orig_moe_dim
if is_pruned:
    cfg_path = os.path.join(eval_model_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as cf:
                cfg = json.load(cf)
            moe_dim = cfg.get("moe_intermediate_size", orig_moe_dim)
        except Exception:
            pass
    if moe_dim == orig_moe_dim and actual_pct > 0:
        pruned_n = round(orig_moe_dim * actual_pct / 100.0 / moe_align) * moe_align
        moe_dim  = max(0, orig_moe_dim - pruned_n)

expert_pct = actual_pct if is_pruned else 0.0

# Find lm_eval JSON output recursively
result_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
result_files = list(set(result_files))

rows = []
for rf in sorted(result_files):
    try:
        with open(rf) as fh:
            data = json.load(fh)
        results = data.get("results", {})
        if not results:
            continue
        for task, metrics in results.items():
            for metric_key, val in metrics.items():
                if "_stderr," in metric_key:
                    continue
                if not (metric_key.endswith(",none") or metric_key == "acc"):
                    continue
                metric     = metric_key.replace(",none", "")
                stderr_key = metric + "_stderr,none"
                stderr     = metrics.get(stderr_key, "")
                row_status = "ok"
                if residual_fallback_val and method not in ("baseline", "pure_delete"):
                    row_status = "residual_not_applied"
                rows.append({
                    "setting_label":                  label,
                    "method":                         actual_method_val,
                    "selector":                       selector,
                    "dataset":                        dataset,
                    "target_pct":                     target_pct,
                    "actual_pct":                     actual_pct,
                    "moe_dim":                        moe_dim,
                    "expert_param_reduction_pct":     expert_pct,
                    "total_model_param_reduction_pct": "NA",
                    "pruning_plan_path":              plan if is_pruned else "NONE",
                    "model_path":                     eval_model_dir,
                    "is_pruned":                      str(is_pruned),
                    "requested_method":               requested_method_val,
                    "actual_method":                  actual_method_val,
                    "residual_applied":               str(residual_applied_val),
                    "residual_fallback_used":         str(residual_fallback_val),
                    "task":                           task,
                    "metric":                         metric,
                    "value":                          val,
                    "stderr":                         stderr,
                    "num_fewshot":                    num_fewshot,
                    "limit":                          limit_val,
                    "batch_size":                     batch_size_val,
                    "status":                         row_status,
                })
    except Exception as e:
        print("[eval] WARNING: could not parse {}: {}".format(rf, e))

if rows:
    fields = [
        "setting_label", "method", "selector", "dataset",
        "target_pct", "actual_pct", "moe_dim",
        "expert_param_reduction_pct", "total_model_param_reduction_pct",
        "pruning_plan_path", "model_path", "is_pruned",
        "requested_method", "actual_method", "residual_applied", "residual_fallback_used",
        "task", "metric", "value", "stderr",
        "num_fewshot", "limit", "batch_size", "status",
    ]
    with open(summary_csv, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore").writerows(rows)
    print("[eval] Appended {} rows to {}".format(len(rows), summary_csv))
    for r in rows:
        line = "[eval]   {}  {}={:.4f}".format(r["task"], r["metric"], float(r["value"]))
        if isinstance(r.get("stderr"), float):
            line += "  stderr={:.4f}".format(r["stderr"])
        print(line)
else:
    print("[eval] WARNING: no results parsed from " + results_dir)
    print("[eval]   Files searched: " + str(result_files))
    sys.exit(1)
PYEOF

    parse_exit="${PIPESTATUS[0]}"
    if [ "${parse_exit:-0}" -ne 0 ]; then
        echo "[eval] ERROR: result parsing failed for ${label}"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    echo "[eval] OK: ${label}"
    SUCCEEDED=$(( SUCCEEDED + 1 ))

    # Checkpoints kept by default for hash comparison and potential re-use.
    # Set CLEANUP_CHECKPOINTS=1 to delete after eval.
    if [ "${CLEANUP_CHECKPOINTS}" = "1" ] && [ "${plan}" != "NONE" ] && [ -d "${SETTING_CKPT_DIR:-}" ]; then
        rm -rf "${SETTING_CKPT_DIR}"
        echo "[eval] Cleaned up checkpoint: ${SETTING_CKPT_DIR}"
    fi
done

# ── Build comparison summary ─────────────────────────────────────────────────
if [ "${SUCCEEDED}" -gt 0 ]; then
    echo "[eval] Building comparison summary ..."
    python3 scripts/downstream_eval_summarize.py \
        --run-dir    "${OUT_DIR}" \
        --plan-dir   "${PLAN_DIR}" \
        --orig-moe-dim "${MODEL_MOE_DIM}" \
        --moe-align  "${MOE_ALIGN}" \
        --model      "${MODEL}" \
        2>&1 || echo "[eval] WARNING: comparison summary step failed (non-fatal)"
fi

# ── Checkpoint hash comparison diagnostic ────────────────────────────────────
if [ -d "${CKPT_BASE_DIR:-}" ] && [ "$(ls -A "${CKPT_BASE_DIR}" 2>/dev/null)" ]; then
    echo ""
    echo "[eval] --- Checkpoint hash comparison diagnostic ---"
    python - "${CKPT_BASE_DIR}" << 'PYHASH'
import json, os, sys

ckpt_base = sys.argv[1]
by_target = {}   # {target_pct_str: {requested_method: info_dict}}

for entry in sorted(os.scandir(ckpt_base), key=lambda e: e.name):
    if not entry.is_dir():
        continue
    pm_path = os.path.join(entry.path, "pruning_metadata.json")
    if not os.path.isfile(pm_path):
        continue
    try:
        with open(pm_path) as f:
            pm = json.load(f)
    except Exception:
        continue
    tgt = str(pm.get("target_pct", pm.get("actual_pct", "?")))
    req = pm.get("requested_method", "?")
    if tgt not in by_target:
        by_target[tgt] = {}
    by_target[tgt][req] = {
        "actual_method":         pm.get("actual_method", "?"),
        "residual_applied":      pm.get("residual_applied", "?"),
        "residual_fallback_used": pm.get("residual_fallback_used", "?"),
        "weight_hash":           pm.get("weight_hash", "?"),
        "ckpt_dir":              entry.path,
    }

if not by_target:
    print("[eval] No pruning_metadata.json found in " + ckpt_base)
    sys.exit(0)

print("[eval] {:>6}  {:<38}  {:<16}  {:>16}  {:<16}  {}".format(
    "Target", "requested_method", "actual_method", "weight_hash", "residual_applied", "fallback"))
print("[eval] " + "-" * 110)

issues = []
for tgt in sorted(by_target.keys()):
    methods = by_target[tgt]
    pd_hash = methods.get("pure_delete", {}).get("weight_hash")
    for req, info in sorted(methods.items()):
        h = info["weight_hash"]
        note = ""
        if req != "pure_delete" and pd_hash and h == pd_hash:
            if not info.get("residual_fallback_used"):
                note = " *** SAME HASH AS PURE_DELETE -- residual not applied ***"
                issues.append("target={} method={}: identical hash to pure_delete but fallback=False".format(tgt, req))
            else:
                note = " (fallback to pure_delete, expected same hash)"
        print("[eval] {:>6}  {:<38}  {:<16}  {:>16}  {:<16}  {}{}".format(
            tgt, req[:38], str(info["actual_method"])[:16], str(h)[:16],
            str(info["residual_applied"]), str(info["residual_fallback_used"]), note))

print("[eval]")
if issues:
    for iss in issues:
        print("[eval] ERROR: " + iss)
    sys.exit(1)
else:
    print("[eval] Hash check: OK")
PYHASH
    _hash_exit=$?
    if [ "${_hash_exit}" -ne 0 ]; then
        echo "[eval] WARNING: hash comparison detected issue (see above)."
    fi
fi

# -- Final summary -----------------------------------------------------------
echo ""
echo "================================================================"
echo "[eval] EVAL COMPLETE  (id=${SWEEP_ID})"
echo "[eval]   Succeeded: ${SUCCEEDED} / ${#SETTINGS[@]}"
echo "[eval]   Failed:    ${FAILED} / ${#SETTINGS[@]}"
echo "[eval]   Summary:   ${SUMMARY_CSV}"
echo "[eval]   Full logs: ${OUT_DIR}/"
echo "================================================================"

if [ "${SMOKE}" = "1" ]; then
    echo ""
    echo "[eval] NOTE: identical accuracy on ${SMOKE_LIMIT}-example smoke is possible;"
    echo "[eval]   use full (no-limit) eval for real downstream metrics."
fi

if [ "${FAILED}" -gt 0 ]; then
    echo "[eval] ERROR: ${FAILED} setting(s) failed."
    exit 1
fi
if [ "${SUCCEEDED}" -eq 0 ]; then
    echo "[eval] ERROR: no settings completed successfully."
    exit 1
fi
exit 0
