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

# ── Sweep ID ──────────────────────────────────────────────────────────────────
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_DIR}/downstream_eval_runs/${SWEEP_ID}"
SUMMARY_CSV="${OUT_DIR}/downstream_summary.csv"
PRUNED_MODEL_DIR="${OUT_DIR}/pruned_model_tmp"

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

if [ "${SMOKE}" = "1" ]; then
    plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_2.0pct_align${ALIGN}.json"
    _apct="$(_plan_actual_pct "${plan}")"
    _meth="$(_plan_method "${plan}")"
    label="${_meth}__${SELECTOR}__wikitext2__target2pct__actual${_apct}pct"
    SETTINGS+=("${label}|${plan}|${_meth}|${SELECTOR}|wikitext2|2.0")
else
    for target in 2 4 8; do
        plan="${PLAN_DIR}/${MODEL_SLUG}_wikitext2_n${N_EVAL}_calib${CALIB_N}_${SELECTOR}_${AGG_MODE}_${target}.0pct_align${ALIGN}.json"
        _apct="$(_plan_actual_pct "${plan}")"
        _meth="$(_plan_method "${plan}")"
        label="${_meth}__${SELECTOR}__wikitext2__target${target}pct__actual${_apct}pct"
        SETTINGS+=("${label}|${plan}|${_meth}|${SELECTOR}|wikitext2|${target}.0")
    done
fi

# ── DRY_RUN: list settings and exit ──────────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[eval] Planned settings (${#SETTINGS[@]} total):"
    n=0
    for setting in "${SETTINGS[@]}"; do
        n=$(( n + 1 ))
        IFS='|' read -r label plan method selector dataset target_pct <<< "${setting}"
        exists=""
        if [ "${plan}" = "NONE" ]; then
            exists="(baseline)"
        elif [ -f "${plan}" ]; then
            exists="[plan exists]"
        else
            exists="[PLAN MISSING -- run full benchmark first]"
        fi
        printf "  %2d. %-45s  method=%-8s  %s\n" "${n}" "${label}" "${method}" "${exists}"
    done
    echo ""
    echo "[eval] Tasks: ${TASKS}"
    if [ "${SMOKE}" = "1" ]; then
        echo "[eval] Limit: --limit ${SMOKE_LIMIT}"
    fi
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

# ── Helper: apply pruning plan, update config, save pruned model ──────────────
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

# Find the transformer layer list
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

# Apply pruning
n_pruned = 0
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
    n_pruned += 1

print(f"[apply_plan] Pruned {n_pruned} layers.")

# --- Detect new MoE intermediate size from expert weights ---
all_sizes = set()
for layer in layer_list:
    mlp     = getattr(layer, "mlp", None)
    if mlp is None:
        continue
    experts = getattr(mlp, "experts", None)
    if experts is None:
        continue
    for expert in experts:
        down = getattr(expert, "down_proj", None)
        if down is not None:
            all_sizes.add(down.weight.shape[1])

new_size = None
if not all_sizes:
    print("[apply_plan] WARNING: no MoE experts found, skipping config update")
elif len(all_sizes) > 1:
    sys.exit(
        f"ERROR: pruned MoE has non-uniform expert sizes: {sorted(all_sizes)}. "
        "Vanilla HF reload requires a uniform moe_intermediate_size. "
        "Use moe_budget_mode=uniform or evaluate in-memory."
    )
else:
    orig_moe = getattr(model.config, "moe_intermediate_size", None)
    new_size  = list(all_sizes)[0]
    print(f"[apply_plan] original moe_intermediate_size={orig_moe}")
    print(f"[apply_plan] new uniform moe_intermediate_size={new_size}")
    model.config.moe_intermediate_size = new_size

    # Print sample expert weight shapes for verification (all three projections)
    for layer in layer_list:
        mlp     = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp else None
        if experts:
            exp0 = experts[0]
            gate = getattr(exp0, "gate_proj", None)
            up   = getattr(exp0, "up_proj",   None)
            down = getattr(exp0, "down_proj",  None)
            if gate is not None:
                print(f"[apply_plan] sample gate_proj shape ={list(gate.weight.shape)}")
            if up is not None:
                print(f"[apply_plan] sample up_proj shape   ={list(up.weight.shape)}")
            if down is not None:
                print(f"[apply_plan] sample down_proj shape ={list(down.weight.shape)}")
            break
    print("[apply_plan] forward_check=True")

# Save
print(f"[apply_plan] Saving to {save_dir} ...")
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
print("[apply_plan] Save complete.")

# Config sanity check: read back config.json and verify the stored size
import pathlib, glob as _glob
cfg_path = pathlib.Path(save_dir) / "config.json"
if cfg_path.exists():
    import json as _json
    with open(cfg_path) as _f:
        _cfg = _json.load(_f)
    saved_moe = _cfg.get("moe_intermediate_size", "NOT_FOUND")
    print(f"[apply_plan] saved moe_intermediate_size={saved_moe}")
    if new_size is not None and saved_moe != new_size:
        sys.exit(f"ERROR: config sanity failed: saved={saved_moe}, expected={new_size}")
    print("[apply_plan] Config sanity OK")
else:
    print("[apply_plan] WARNING: config.json not found after save")

# Count checkpoint shards
shard_files = (
    _glob.glob(str(pathlib.Path(save_dir) / "model*.safetensors")) +
    _glob.glob(str(pathlib.Path(save_dir) / "pytorch_model*.bin"))
)
print(f"[apply_plan] checkpoint shards: {len(shard_files)}")

# AutoConfig reload verification
from transformers import AutoConfig as _AutoCfg
print("[apply_plan] Verifying with AutoConfig.from_pretrained ...")
loaded_cfg = _AutoCfg.from_pretrained(save_dir, trust_remote_code=True)
loaded_moe = getattr(loaded_cfg, "moe_intermediate_size", None)
print(f"[apply_plan] AutoConfig moe_intermediate_size={loaded_moe}")
if new_size is not None and loaded_moe != new_size:
    sys.exit(f"ERROR: AutoConfig mismatch: loaded={loaded_moe}, expected={new_size}")
print("[apply_plan] AutoConfig reload: OK")

print("[apply_plan] Done.")
PYEOF
}

# ── Write CSV header ──────────────────────────────────────────────────────────
python - "${SUMMARY_CSV}" << 'PYEOF'
import csv, os, sys
out_path = sys.argv[1]
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
fields = ["setting_label", "method", "selector", "dataset", "target_pct", "actual_pct",
          "task", "metric", "value", "stderr", "num_fewshot", "model_name", "status"]
with open(out_path, "w", newline="") as fh:
    csv.DictWriter(fh, fieldnames=fields).writeheader()
print(f"[eval] CSV header written: {out_path}")
PYEOF

# ── Run each setting ──────────────────────────────────────────────────────────
FAILED=0
SUCCEEDED=0

for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label plan method selector dataset target_pct <<< "${setting}"
    out_json="${OUT_DIR}/${label}.json"
    log_file="${OUT_DIR}/${label}.log"

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
            echo "[eval] ERROR: plan not found: ${plan}"
            echo "[eval]   Generate plans: bash scripts/run_moe_residual_selected_full_benchmark.sh"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        echo "[eval] Applying pruning plan: ${plan}"
        rm -rf "${PRUNED_MODEL_DIR}"
        if ! apply_and_save_pruned_model "${plan}" "${PRUNED_MODEL_DIR}"; then
            echo "[eval] ERROR: plan application/save failed for ${label}"
            FAILED=$(( FAILED + 1 ))
            continue
        fi
        eval_model="${PRUNED_MODEL_DIR}"
        echo "[eval] Pruned model saved to: ${eval_model}"

        # ── Pruned model sanity checks (printed for every pruned setting) ─────
        echo "[eval] --- Pruned model sanity checks ---"
        python - "${PRUNED_MODEL_DIR}" "${plan}" << 'PYSANITY'
import json, glob, os, sys
save_dir  = sys.argv[1]
plan_path = sys.argv[2]

# Read config.json for moe_intermediate_size and hidden_size
cfg_path = os.path.join(save_dir, "config.json")
orig_moe = saved_moe = hidden_size = "?"
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    saved_moe   = cfg.get("moe_intermediate_size", "?")
    hidden_size = cfg.get("hidden_size", "?")

# Recover original moe_intermediate_size from plan (old_intermediate = pre-prune d_ff)
if os.path.isfile(plan_path):
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

# Count checkpoint shards
shard_files = (
    glob.glob(os.path.join(save_dir, "model*.safetensors")) +
    glob.glob(os.path.join(save_dir, "pytorch_model*.bin"))
)
n_shards = len(shard_files)

# Derived weight shapes: gate/up are [d_ff_new, d_model], down is [d_model, d_ff_new]
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

    # Parse lm_eval JSON results (recursive search) and append to summary CSV
    python - "${lm_out}" "${label}" "${plan}" \
        "${MODEL}" "${NUM_FEWSHOT}" "${SUMMARY_CSV}" \
        "${method}" "${selector}" "${dataset}" "${target_pct}" << 'PYEOF'
import csv, json, os, sys, glob

(results_dir, label, plan, model_name,
 num_fewshot, summary_csv,
 method, selector, dataset, target_pct) = sys.argv[1:11]

# Compute actual_pct from plan JSON (reliable, plan-side ground truth)
actual_pct = 0.0
if plan != "NONE" and os.path.isfile(plan):
    try:
        with open(plan) as pf:
            plan_data = json.load(pf)
        total  = sum(lcfg.get("old_intermediate", 0) for lcfg in plan_data.get("layers", []))
        pruned = sum(len(lcfg.get("prune_idx", []))   for lcfg in plan_data.get("layers", []))
        if total > 0:
            actual_pct = round(100.0 * pruned / total, 3)
    except Exception as e:
        print(f"[eval] WARNING: cannot compute actual_pct from plan: {e}")

# Find lm_eval JSON output recursively (location varies by lm_eval version)
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
                # Skip stderr entries -- we look them up separately
                if "_stderr," in metric_key:
                    continue
                if not (metric_key.endswith(",none") or metric_key == "acc"):
                    continue
                metric = metric_key.replace(",none", "")
                # Corresponding stderr key
                stderr_key = metric + "_stderr,none"
                stderr = metrics.get(stderr_key, "")
                rows.append({
                    "setting_label": label,
                    "method":        method,
                    "selector":      selector,
                    "dataset":       dataset,
                    "target_pct":    target_pct,
                    "actual_pct":    actual_pct,
                    "task":          task,
                    "metric":        metric,
                    "value":         val,
                    "stderr":        stderr,
                    "num_fewshot":   num_fewshot,
                    "model_name":    model_name,
                    "status":        "ok",
                })
    except Exception as e:
        print("[eval] WARNING: could not parse {}: {}".format(rf, e))

if rows:
    fields = ["setting_label", "method", "selector", "dataset", "target_pct", "actual_pct",
              "task", "metric", "value", "stderr", "num_fewshot", "model_name", "status"]
    with open(summary_csv, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writerows(rows)
    print("[eval] Appended {} rows to {}".format(len(rows), summary_csv))
    for r in rows:
        task_v   = r["task"]
        metric_v = r["metric"]
        value_v  = r["value"]
        stderr_v = r.get("stderr")
        line = "[eval]   {}  {}={:.4f}".format(task_v, metric_v, value_v)
        if isinstance(stderr_v, float):
            line += "  stderr={:.4f}".format(stderr_v)
        print(line)
else:
    result_files_str = str(result_files)
    print("[eval] WARNING: no results parsed from {}".format(results_dir))
    print("[eval]   Files searched: {}".format(result_files_str))
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

    # Clean up temporary pruned model
    if [ "${plan}" != "NONE" ] && [ -d "${PRUNED_MODEL_DIR}" ]; then
        rm -rf "${PRUNED_MODEL_DIR}"
    fi
done

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
