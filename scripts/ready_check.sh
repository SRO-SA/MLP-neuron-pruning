#!/usr/bin/env bash
# ready_check.sh — end-to-end readiness check for qwen_swiglu_pruning
#
# Runs all four gates in order:
#   1. Environment verification  (check_env.py)
#   2. Model + dataset download  (prepare_models.py)
#   3. Dense smoke test          (Qwen2.5-0.5B, n_eval=8)
#   4. MoE smoke test            (Qwen3-30B-A3B, smoke layers, 1%, n_eval=8)
#
# Prints "READY TO RUN PAPER BENCHMARKS" only if all four pass.
#
# Usage:
#   bash scripts/ready_check.sh
#
# Optional env vars:
#   SKIP_DOWNLOAD=1   — skip step 2 (models already cached)
#   SKIP_DENSE=1      — skip step 3
#   SKIP_MOE=1        — skip step 4
#   HF_TOKEN=...      — passed to prepare_models.py for gated models
#   HF_CACHE=...      — cache directory (default: /workspace/hf_cache)
# --------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
SKIP_DENSE=${SKIP_DENSE:-0}
SKIP_MOE=${SKIP_MOE:-0}
HF_CACHE=${HF_CACHE:-/workspace/hf_cache}
HF_TOKEN_ARG=""
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ARG="--token ${HF_TOKEN}"
fi

PASS_MARKER="  ✓"
FAIL_MARKER="  ✗"
STEP_PASS=0
STEP_FAIL=0

step_pass() { echo "${PASS_MARKER}  $1"; (( STEP_PASS++ )) || true; }
step_fail() { echo "${FAIL_MARKER}  $1" >&2; (( STEP_FAIL++ )) || true; }

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"

cd "${REPO_ROOT}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ready_check.sh — qwen_swiglu_pruning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Environment check ────────────────────────────────────────────────
echo "── Step 1/4: Environment check ──────────────────────────────"
if python scripts/check_env.py; then
    step_pass "check_env.py passed"
else
    step_fail "check_env.py failed — fix environment before continuing"
    echo ""
    echo "  Hint: run one of:"
    echo "    USE_EXISTING_TORCH=1 bash setup_env.sh"
    echo "    INSTALL_TORCH_CU128=1 bash setup_env.sh"
    exit 1
fi

echo ""

# ── Step 2: Download models and datasets ─────────────────────────────────────
echo "── Step 2/4: Prepare models and datasets ────────────────────"
if [[ $SKIP_DOWNLOAD -eq 1 ]]; then
    echo "  (skipped — SKIP_DOWNLOAD=1)"
    step_pass "download skipped"
else
    if python scripts/prepare_models.py \
        --models Qwen/Qwen2.5-0.5B Qwen/Qwen3-30B-A3B \
        --datasets wikitext2 c4 \
        --cache-dir "${HF_CACHE}" \
        --skip-existing \
        ${HF_TOKEN_ARG}; then
        step_pass "models and datasets ready"
    else
        step_fail "prepare_models.py failed"
        echo "  Hint: check network access and HF_TOKEN if models are gated"
        exit 1
    fi
fi

echo ""

# ── Step 3: Dense smoke test (Qwen2.5-0.5B) ──────────────────────────────────
echo "── Step 3/4: Dense smoke test (Qwen2.5-0.5B, n_eval=8) ─────"
if [[ $SKIP_DENSE -eq 1 ]]; then
    echo "  (skipped — SKIP_DENSE=1)"
    step_pass "dense smoke skipped"
else
    DENSE_LOG="results/smoke_dense_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p results
    echo "  Running: python run_experiment.py --config configs/smoke_dense_0.5b.yaml"
    echo "  Log: ${DENSE_LOG}"

    if python run_experiment.py \
        --config configs/smoke_dense_0.5b.yaml \
        2>&1 | tee "${DENSE_LOG}" | tail -5; then
        step_pass "dense smoke passed"
    else
        step_fail "dense smoke FAILED — see ${DENSE_LOG}"
        exit 1
    fi
fi

echo ""

# ── Step 4: MoE smoke test (Qwen3-30B-A3B) ───────────────────────────────────
echo "── Step 4/4: MoE smoke test (Qwen3-30B-A3B, 1%, n_eval=8) ──"
if [[ $SKIP_MOE -eq 1 ]]; then
    echo "  (skipped — SKIP_MOE=1)"
    step_pass "MoE smoke skipped"
else
    MOE_LOG="results/smoke_moe_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p results
    echo "  Running: python run_experiment.py --config configs/smoke_moe_1pct.yaml --moe-target-pruning"
    echo "  Log: ${MOE_LOG}"

    if python run_experiment.py \
        --config configs/smoke_moe_1pct.yaml \
        --moe-target-pruning \
        2>&1 | tee "${MOE_LOG}" | tail -5; then
        step_pass "MoE smoke passed"
    else
        step_fail "MoE smoke FAILED — see ${MOE_LOG}"
        exit 1
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Steps passed: ${STEP_PASS}  |  Steps failed: ${STEP_FAIL}"
echo ""
if [[ $STEP_FAIL -eq 0 ]]; then
    echo "  READY TO RUN PAPER BENCHMARKS"
    echo ""
    echo "  Next steps:"
    echo "    bash scripts/run_moe_full48_benchmark.sh"
    echo "    python scripts/summarize_moe_results.py --glob 'results/moe_*.csv'"
else
    echo "  NOT READY — fix the failed steps above"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ $STEP_FAIL -eq 0 ]]
