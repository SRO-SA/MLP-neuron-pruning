#!/usr/bin/env bash
# setup_env.sh — reproducible environment setup for qwen_swiglu_pruning
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#   Option A: keep whatever torch the container already has
#     USE_EXISTING_TORCH=1 bash setup_env.sh
#
#   Option B: install stable CUDA 12.8 torch family
#     INSTALL_TORCH_CU128=1 bash setup_env.sh
#
#   Option C: install nightly CUDA 12.8 torch family
#     (use only when stable torch doesn't support the GPU, e.g. Blackwell sm_120)
#     INSTALL_TORCH_NIGHTLY_CU128=1 bash setup_env.sh
#
#   Override cache directory (default: /workspace/hf_cache):
#     HF_CACHE=/my/cache bash setup_env.sh
#
# ── What this does ───────────────────────────────────────────────────────────
#   1. Print system information
#   2. Set HuggingFace cache paths
#   3. Install Python requirements (requirements.txt — no torch)
#   4. Handle torch/torchvision/torchaudio based on the option chosen
#   5. Set PYTHONPATH
#   6. Run scripts/check_env.py
# ----------------------------------------------------------------------------

set -euo pipefail

# ── Environment flags (set via env vars, not positional args) ─────────────────
USE_EXISTING_TORCH=${USE_EXISTING_TORCH:-0}
INSTALL_TORCH_CU128=${INSTALL_TORCH_CU128:-0}
INSTALL_TORCH_NIGHTLY_CU128=${INSTALL_TORCH_NIGHTLY_CU128:-0}
HF_CACHE=${HF_CACHE:-/workspace/hf_cache}

# Validate: exactly one torch mode must be set
TORCH_MODE_COUNT=$(( USE_EXISTING_TORCH + INSTALL_TORCH_CU128 + INSTALL_TORCH_NIGHTLY_CU128 ))
if [[ $TORCH_MODE_COUNT -eq 0 ]]; then
    echo ""
    echo "ERROR: No torch mode selected. Set one of:"
    echo "  USE_EXISTING_TORCH=1            — keep existing torch (e.g. container default)"
    echo "  INSTALL_TORCH_CU128=1           — install stable cu128 torch family"
    echo "  INSTALL_TORCH_NIGHTLY_CU128=1   — install nightly cu128 (for Blackwell / new GPUs)"
    echo ""
    exit 1
fi
if [[ $TORCH_MODE_COUNT -gt 1 ]]; then
    echo "ERROR: More than one torch mode is set. Set exactly one."
    exit 1
fi

PIP="python -m pip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. System information ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  qwen_swiglu_pruning — environment setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "── System info ──────────────────────────────────────────────"
echo "  Python  : $(python --version 2>&1)"
echo "  pip     : $(python -m pip --version 2>&1 | head -1)"
echo "  Host    : $(hostname)"
echo "  Date    : $(date)"
echo ""

echo "── GPU / CUDA ────────────────────────────────────────────────"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=, read -r idx name mem drv; do
        echo "  GPU ${idx}: ${name}  (${mem} MiB, driver ${drv})"
      done || true
    echo "  CUDA (driver): $(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -1 || echo 'unknown')"
else
    echo "  nvidia-smi not found — CPU-only or driver not installed"
fi
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo ""

echo "── Disk space ───────────────────────────────────────────────"
df -h / 2>/dev/null | tail -1 | awk '{print "  /: " $4 " free of " $2}' || true
if [[ -d /workspace ]]; then
    df -h /workspace 2>/dev/null | tail -1 | awk '{print "  /workspace: " $4 " free of " $2}' || true
fi
echo ""

# ── 2. Cache paths ────────────────────────────────────────────────────────────
echo "── HuggingFace cache paths ──────────────────────────────────"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
echo "  HF_HOME             = ${HF_HOME}"
echo "  TRANSFORMERS_CACHE  = ${TRANSFORMERS_CACHE}"
echo "  HF_DATASETS_CACHE   = ${HF_DATASETS_CACHE}"
echo ""

# ── 3. Install Python requirements (no torch) ─────────────────────────────────
echo "── [1/3] Installing Python requirements ─────────────────────"
$PIP install --upgrade pip --quiet
$PIP install -r "${SCRIPT_DIR}/requirements.txt"
echo ""

# ── 4. Handle torch / torchvision / torchaudio ────────────────────────────────
#
# IMPORTANT: torch, torchvision, and torchaudio must all come from the same
# wheel index and the same build.  Mixing sources causes ABI mismatches that
# manifest as "undefined symbol" or "cannot open shared object file" errors
# when Transformers loads a model.
# ---------------------------------------------------------------------------

if [[ $USE_EXISTING_TORCH -eq 1 ]]; then
    echo "── [2/3] Torch mode: USE_EXISTING_TORCH ─────────────────────"
    echo "  Verifying existing torch, torchvision, torchaudio imports ..."
    python - <<'PYEOF'
import sys

errors = []

# torch
try:
    import torch
    print(f"  torch       {torch.__version__}")
except ImportError as e:
    errors.append(f"torch import failed: {e}")

# torchvision
try:
    import torchvision
    print(f"  torchvision {torchvision.__version__}")
except ImportError as e:
    print(f"  torchvision NOT FOUND ({e}) — this may be acceptable if unused")

# torchaudio — must be verified because Transformers can indirectly import it
try:
    import torchaudio
    print(f"  torchaudio  {torchaudio.__version__}")
except ImportError as e:
    print(f"  WARNING: torchaudio import failed: {e}")
    print("  If Transformers fails to load a model, torchaudio ABI mismatch")
    print("  may be the cause. Re-run with INSTALL_TORCH_CU128=1 to fix.")

if errors:
    for err in errors:
        print(f"  ERROR: {err}", file=sys.stderr)
    sys.exit(1)
PYEOF

elif [[ $INSTALL_TORCH_CU128 -eq 1 ]]; then
    echo "── [2/3] Torch mode: INSTALL_TORCH_CU128 (stable) ──────────"
    echo "  Installing torch + torchvision + torchaudio from cu128 stable index ..."
    echo "  (all three packages from the same index prevents ABI mismatches)"
    $PIP install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

elif [[ $INSTALL_TORCH_NIGHTLY_CU128 -eq 1 ]]; then
    echo "── [2/3] Torch mode: INSTALL_TORCH_NIGHTLY_CU128 ───────────"
    echo "  NOTE: nightly = pre-release. Use only if stable torch does not"
    echo "  support your GPU (e.g. Blackwell sm_120 requires nightly)."
    echo ""
    echo "  Installing torch + torchvision + torchaudio from cu128 NIGHTLY index ..."
    $PIP install \
        --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
fi

echo ""

# ── 5. PYTHONPATH ─────────────────────────────────────────────────────────────
echo "── [3/3] Setting PYTHONPATH ──────────────────────────────────"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
echo "  PYTHONPATH = ${PYTHONPATH}"
echo ""

# ── 6. Environment check ──────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running environment check ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${SCRIPT_DIR}/scripts/check_env.py"
