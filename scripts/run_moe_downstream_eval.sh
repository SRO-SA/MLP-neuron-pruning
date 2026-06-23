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
#   SMOKE=1           bash scripts/run_moe_downstream_eval.sh
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
#   SMOKE=1             baseline + 2%/wikitext2 on arc_easy only
#   CHECK_DEPS=1        Check Python + lm_eval + CUDA; exit 0 if OK, 1 if not
#   INSTALL_LM_EVAL=1   Auto-install lm-evaluation-harness before checking/running
#   RESULTS_DIR=...     Results directory (default: results)
#   MODEL=...           HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)
#   DTYPE=...           bfloat16 | float16 | float32 (default: bfloat16)
#   VENV=...            Virtualenv path (default: /workspace/venvs/qwen-pruning)
#   NUM_FEWSHOT=...     Few-shot examples (default: 0)
#   SKIP_MMLU=1         Skip MMLU (default: 1)
#   BATCH_SIZE=...      lm_eval batch size (default: 4)

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

# ── Sweep ID ──────────────────────────────────────────────────────────────────
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_DIR}/downstream_eval_runs/${SWEEP_ID}"
SUMMARY_CSV="${OUT_DIR}/downstream_summary.csv"
PRUNED_MODEL_DIR="${OUT_DIR}/pruned_model_tmp"

echo "[eval] Downstream eval ID: ${SWEEP_ID}"
echo "[eval] Model:  ${MODEL}"
echo "[eval] Tasks:  ${TASKS}"

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
# Format: "label|plan_path"   (use "NONE" for baseline)
declare -a SETTINGS=()
SETTINGS+=("baseline_no_pruning|NONE")

if [ "${SMOKE}" = "1" ]; then
    plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_2.0pct_align${ALIGN}.json"
    SETTINGS+=("${SELECTOR}_wikitext2_target2pct|${plan}")
else
    for target in 2 4 8; do
        plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target}.0pct_align${ALIGN}.json"
        SETTINGS+=("${SELECTOR}_wikitext2_target${target}pct|${plan}")
    done
fi

# ── DRY_RUN: list settings and exit ──────────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[eval] Planned settings (${#SETTINGS[@]} total):"
    n=0
    for setting in "${SETTINGS[@]}"; do
        n=$(( n + 1 ))
        label="${setting%%|*}"
        plan="${setting##*|}"
        exists=""
        if [ "${plan}" = "NONE" ]; then
            exists="(baseline)"
        elif [ -f "${plan}" ]; then
            exists="[plan exists]"
        else
            exists="[PLAN MISSING -- run full benchmark first]"
        fi
        printf "  %2d. %-45s  %s\n" "${n}" "${label}" "${exists}"
    done
    echo ""
    echo "[eval] Tasks: ${TASKS}"
    echo ""
    echo "[eval] DRY_RUN complete."
    echo "[eval] To run: bash scripts/run_moe_downstream_eval.sh"
    echo "[eval] Missing plans: bash scripts/run_moe_residual_selected_full_benchmark.sh"
    exit 0
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
# Fall back to the CLI entry point only if module invocation fails.
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

# ── Helper: apply pruning plan and save pruned model to disk ─────────────────
apply_and_save_pruned_model() {
    local plan_path="$1"
    local save_dir="$2"
    python - "${MODEL}" "${plan_path}" "${save_dir}" "${DTYPE}" << 'PYEOF'
import sys, json, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name, plan_path, save_dir, dtype_str = sys.argv[1:5]
dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
dtype = dtype_map.get(dtype_str, torch.bfloat16)

print(f"[apply_plan] Loading model {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
)
model.eval()

print(f"[apply_plan] Loading plan {plan_path} ...")
with open(plan_path) as fh:
    plan = json.load(fh)

layer_list = None
for attr in ("model", "transformer"):
    sub = getattr(model, attr, None)
    if sub is not None:
        for la in ("layers", "h", "blocks"):
            ll = getattr(sub, la, None)
            if ll is not None:
                layer_list = ll
                break
    if layer_list is not None:
        break

if layer_list is None:
    sys.exit("ERROR: cannot find layer list in model")

for lcfg in plan.get("layers", []):
    li        = lcfg["layer_idx"]
    prune_idx = lcfg["prune_idx"]
    old_d_ff  = lcfg["old_intermediate"]
    if not prune_idx:
        continue
    keep = torch.ones(old_d_ff, dtype=torch.bool)
    keep[prune_idx] = False
    layer = layer_list[li]
    mlp   = getattr(layer, "mlp", None)
    if mlp is None:
        continue
    experts = list(mlp.experts) if hasattr(mlp, "experts") else [mlp]
    with torch.no_grad():
        for expert in experts:
            gate = getattr(expert, "gate_proj", None)
            up   = getattr(expert, "up_proj",   None)
            down = getattr(expert, "down_proj",  None)
            if gate is None or up is None or down is None:
                continue
            gate.weight = nn.Parameter(gate.weight[keep, :].contiguous())
            up.weight   = nn.Parameter(up.weight[keep, :].contiguous())
            down.weight = nn.Parameter(down.weight[:, keep].contiguous())

print(f"[apply_plan] Saving pruned model to {save_dir} ...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("[apply_plan] Done.")
PYEOF
}

# ── Write CSV header ──────────────────────────────────────────────────────────
python - "${SUMMARY_CSV}" << 'PYEOF'
import csv, os, sys
out_path = sys.argv[1]
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
fields = ["label", "plan_path", "task", "metric", "value",
          "num_fewshot", "model_name", "status"]
with open(out_path, "w", newline="") as fh:
    csv.DictWriter(fh, fieldnames=fields).writeheader()
print(f"[eval] CSV header written: {out_path}")
PYEOF

# ── Run each setting ──────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0

for setting in "${SETTINGS[@]}"; do
    label="${setting%%|*}"
    plan="${setting##*|}"

    echo ""
    echo "================================================================"
    echo "[eval] Setting: ${label}"
    echo "================================================================"

    if [ "${plan}" = "NONE" ]; then
        eval_model="${MODEL}"
        echo "[eval] Using baseline model (no pruning)."
    else
        if [ ! -f "${plan}" ]; then
            echo "[eval] ERROR: plan not found: ${plan}"
            echo "[eval]   Generate plans: bash scripts/run_moe_residual_selected_full_benchmark.sh"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        echo "[eval] Applying pruning plan: ${plan}"
        rm -rf "${PRUNED_MODEL_DIR}"
        if ! apply_and_save_pruned_model "${plan}" "${PRUNED_MODEL_DIR}"; then
            echo "[eval] ERROR: plan application failed for ${label}"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        eval_model="${PRUNED_MODEL_DIR}"
    fi

    lm_out="${OUT_DIR}/${label}"
    mkdir -p "${lm_out}"

    echo "[eval] lm_eval command: ${LM_EVAL_CMD}"
    echo "[eval] Tasks: ${TASKS}"
    set +e
    ${LM_EVAL_CMD} \
        --model hf \
        --model_args "pretrained=${eval_model},dtype=${DTYPE},trust_remote_code=True" \
        --tasks "${TASKS}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${lm_out}" \
        2>&1 | tee "${lm_out}/lm_eval.log"
    lm_exit=${PIPESTATUS[0]}
    set -e

    if [ ${lm_exit} -ne 0 ]; then
        echo "[eval] ERROR: lm_eval exited ${lm_exit} for ${label}"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # Verify at least one result file was written
    lm_result_count=$(find "${lm_out}" -name "*.json" | wc -l)
    if [ "${lm_result_count}" -eq 0 ]; then
        echo "[eval] ERROR: lm_eval exited 0 but no JSON results in ${lm_out}"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # Parse lm_eval JSON results and append to summary CSV
    python - "${lm_out}" "${label}" "${plan}" \
        "${MODEL}" "${NUM_FEWSHOT}" "${SUMMARY_CSV}" << 'PYEOF'
import csv, json, os, sys, glob

results_dir, label, plan, model_name, num_fewshot, summary_csv = sys.argv[1:7]
rows = []

result_files = (glob.glob(os.path.join(results_dir, "*.json")) +
                glob.glob(os.path.join(results_dir, "results*.json")))
for rf in result_files:
    try:
        with open(rf) as fh:
            data = json.load(fh)
        for task, metrics in data.get("results", {}).items():
            for metric_key, val in metrics.items():
                if metric_key.endswith(",none") or metric_key == "acc":
                    rows.append({
                        "label":       label,
                        "plan_path":   plan,
                        "task":        task,
                        "metric":      metric_key.replace(",none", ""),
                        "value":       val,
                        "num_fewshot": num_fewshot,
                        "model_name":  model_name,
                        "status":      "ok",
                    })
    except Exception as e:
        print(f"[eval] WARNING: could not parse {rf}: {e}")

if rows:
    fields = ["label","plan_path","task","metric","value","num_fewshot","model_name","status"]
    with open(summary_csv, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writerows(rows)
    print(f"[eval] Appended {len(rows)} rows to {summary_csv}")
else:
    print(f"[eval] WARNING: no results parsed from {results_dir}")
PYEOF

    echo "[eval] OK: ${label}"
    SUCCEEDED=$(( SUCCEEDED + 1 ))

    if [ "${plan}" != "NONE" ] && [ -d "${PRUNED_MODEL_DIR}" ]; then
        rm -rf "${PRUNED_MODEL_DIR}"
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[eval] EVAL COMPLETE  (id=${SWEEP_ID})"
echo "[eval]   Succeeded: ${SUCCEEDED} / ${#SETTINGS[@]}"
echo "[eval]   Failed:    ${FAILED} / ${#SETTINGS[@]}"
echo "[eval]   Summary:   ${SUMMARY_CSV}"
echo "[eval]   Full logs: ${OUT_DIR}/"
echo "================================================================"

if [ "${FAILED}" -gt 0 ]; then
    echo "[eval] ERROR: ${FAILED} setting(s) failed."
    exit 1
fi
if [ "${SUCCEEDED}" -eq 0 ]; then
    echo "[eval] ERROR: no settings completed successfully."
    exit 1
fi
exit 0
