#!/usr/bin/env bash
# run_moe_downstream_eval.sh
#
# Run lm-evaluation-harness (lm_eval) on baseline and pruned models.
#
# Evaluates on:
#   - arc_easy        (ARC Easy — multiple-choice)
#   - arc_challenge   (ARC Challenge)
#   - hellaswag       (sentence completion)
#   - winogrande      (pronoun resolution)
#   - mmlu            (Massive Multitask Language Understanding — optional, slow)
#
# For each pruning setting, the script:
#   1. Loads the original model weights
#   2. Applies the pruning plan JSON (physically removes channels)
#   3. Saves a temporary pruned model to disk
#   4. Runs lm_eval on the temporary model
#   5. Collects JSON results into results/downstream_eval_runs/<id>/
#
# Graceful dependency handling:
#   - If lm_eval is not installed, prints clear instructions and exits 0
#     (does not abort an outer pipeline).
#   - If a specific task fails, logs the error and continues.
#
# Settings:
#   1. Baseline (no pruning)
#   2. rmsnorm_bound, wikitext2, 2%
#   3. rmsnorm_bound, wikitext2, 4%
#   4. rmsnorm_bound, wikitext2, 8%
#
# Usage:
#   bash scripts/run_moe_downstream_eval.sh
#   DRY_RUN=1 bash scripts/run_moe_downstream_eval.sh   # list settings, no GPU
#   SMOKE=1   bash scripts/run_moe_downstream_eval.sh   # baseline + 2% on arc_easy only
#
# Env overrides:
#   DRY_RUN=1         List settings only, no GPU
#   SMOKE=1           1 setting (baseline) + 1 task (arc_easy)
#   RESULTS_DIR=...   Results directory (default: results)
#   MODEL=...         HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)
#   DTYPE=...         bfloat16 | float16 | float32 (default: bfloat16)
#   VENV=...          Virtualenv path (default: /workspace/venvs/qwen-pruning)
#   NUM_FEWSHOT=...   Few-shot examples (default: 0)
#   SKIP_MMLU=1       Skip MMLU (very slow; default: 1)
#   BATCH_SIZE=...    lm_eval batch size (default: 4)

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
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
SKIP_MMLU="${SKIP_MMLU:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

MODEL_SLUG="$(echo "${MODEL}" | tr '/' '_' | tr '-' '_')"
PLAN_DIR="${RESULTS_DIR}/pruning_plans"

# Tasks to evaluate (space-separated)
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
    echo "[eval] No virtualenv at ${VENV}, using system Python"
fi

# ── Check lm_eval is installed — graceful exit if missing ────────────────────
check_lm_eval() {
    python3 -c "import lm_eval" 2>/dev/null && return 0
    echo ""
    echo "[eval] ─────────────────────────────────────────────────────────────"
    echo "[eval] lm-evaluation-harness is NOT installed."
    echo "[eval]"
    echo "[eval] To install:"
    echo "[eval]   pip install lm-eval --break-system-packages"
    echo "[eval]   # or: pip install lm-eval[vllm] --break-system-packages"
    echo "[eval]"
    echo "[eval] GitHub: https://github.com/EleutherAI/lm-evaluation-harness"
    echo "[eval] ─────────────────────────────────────────────────────────────"
    echo ""
    return 1
}

# ── Settings list ─────────────────────────────────────────────────────────────
# Format: "label|plan_path"   (use "NONE" for baseline)
SELECTOR="rmsnorm_bound"
AGG_MODE="p95"
ALIGN="16"
N_EVAL="512"
CALIB_N="512"

declare -a SETTINGS=()
SETTINGS+=("baseline_no_pruning|NONE")

if [ "${SMOKE}" = "1" ]; then
    # Smoke: only baseline + 2%/wikitext2
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
            exists="[PLAN MISSING — run full benchmark first]"
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

# ── Require lm_eval ──────────────────────────────────────────────────────────
if ! check_lm_eval; then
    echo "[eval] Skipping evaluation (lm_eval missing). Exiting with code 0."
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "[eval] Output dir: ${OUT_DIR}"

# ── Helper: apply pruning plan and save pruned model ─────────────────────────
apply_and_save_pruned_model() {
    local plan_path="$1"
    local save_dir="$2"
    python3 - "${MODEL}" "${plan_path}" "${save_dir}" "${DTYPE}" << 'PYEOF'
import sys, json, torch
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

import torch.nn as nn

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
print(f"[apply_plan] Done.")
PYEOF
}

# ── Write CSV header ──────────────────────────────────────────────────────────
python3 - "${SUMMARY_CSV}" << 'PYEOF'
import csv, sys
out_path = sys.argv[1]
fields = [
    "label", "plan_path", "task", "metric", "value",
    "num_fewshot", "model_name", "status",
]
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
    echo "════════════════════════════════════════════════════════════════"
    echo "[eval] Setting: ${label}"
    echo "════════════════════════════════════════════════════════════════"

    # Determine model path to pass to lm_eval
    if [ "${plan}" = "NONE" ]; then
        eval_model="${MODEL}"
        echo "[eval] Using baseline model (no pruning)."
    else
        if [ ! -f "${plan}" ]; then
            echo "[eval] WARNING: plan not found: ${plan}"
            echo "[eval]   Skipping this setting."
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        echo "[eval] Applying pruning plan: ${plan}"
        rm -rf "${PRUNED_MODEL_DIR}"
        apply_and_save_pruned_model "${plan}" "${PRUNED_MODEL_DIR}" || {
            echo "[eval] ERROR: failed to apply plan for ${label}"
            FAILED=$(( FAILED + 1 ))
            continue
        }
        eval_model="${PRUNED_MODEL_DIR}"
    fi

    # lm_eval output for this setting
    lm_out="${OUT_DIR}/${label}"
    mkdir -p "${lm_out}"

    echo "[eval] Running lm_eval on tasks: ${TASKS}"
    set +e
    python3 -m lm_eval \
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
        echo "[eval] ✗ lm_eval failed for ${label} (exit ${lm_exit})"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # Parse lm_eval JSON results and append to summary CSV
    python3 - "${lm_out}" "${label}" "${plan}" \
        "${MODEL}" "${NUM_FEWSHOT}" "${SUMMARY_CSV}" << 'PYEOF'
import csv, json, os, sys, glob

results_dir, label, plan, model_name, num_fewshot, summary_csv = sys.argv[1:7]
rows = []

# lm_eval writes results/<timestamp>/*.json or results.json
result_files = glob.glob(os.path.join(results_dir, "*.json")) + \
               glob.glob(os.path.join(results_dir, "results*.json"))
for rf in result_files:
    try:
        with open(rf) as fh:
            data = json.load(fh)
        # lm_eval v0.4+ format: {"results": {"task": {"metric,none": value, ...}}}
        results = data.get("results", {})
        for task, metrics in results.items():
            for metric_key, val in metrics.items():
                if metric_key.endswith(",none") or metric_key == "acc":
                    metric = metric_key.replace(",none", "")
                    rows.append({
                        "label":       label,
                        "plan_path":   plan,
                        "task":        task,
                        "metric":      metric,
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
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writerows(rows)
    print(f"[eval] Appended {len(rows)} rows to {summary_csv}")
else:
    print(f"[eval] WARNING: no results parsed from {results_dir}")
PYEOF

    echo "[eval] ✓ ${label} complete."
    SUCCEEDED=$(( SUCCEEDED + 1 ))

    # Clean up temporary pruned model to save disk space
    if [ "${plan}" != "NONE" ] && [ -d "${PRUNED_MODEL_DIR}" ]; then
        rm -rf "${PRUNED_MODEL_DIR}"
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[eval] EVAL COMPLETE  (id=${SWEEP_ID})"
echo "[eval]   Succeeded: ${SUCCEEDED} / ${#SETTINGS[@]}"
echo "[eval]   Failed:    ${FAILED} / ${#SETTINGS[@]}"
echo "[eval]   Summary:   ${SUMMARY_CSV}"
echo "[eval]   Full logs: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════"

if [ "${FAILED}" -gt 0 ]; then
    echo "[eval] WARNING: ${FAILED} setting(s) failed."
    exit 1
fi
exit 0
