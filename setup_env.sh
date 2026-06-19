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
#   Option D: fix a mismatched torchvision / torchaudio against an existing torch
#     FIX_TORCH_FAMILY=1 bash setup_env.sh
#     Detects the installed torch version, computes matching torchvision +
#     torchaudio versions, and reinstalls them (--no-deps so torch is unchanged).
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
FIX_TORCH_FAMILY=${FIX_TORCH_FAMILY:-0}
HF_CACHE=${HF_CACHE:-/workspace/hf_cache}

# Validate: exactly one torch mode must be set
TORCH_MODE_COUNT=$(( USE_EXISTING_TORCH + INSTALL_TORCH_CU128 + INSTALL_TORCH_NIGHTLY_CU128 + FIX_TORCH_FAMILY ))
if [[ $TORCH_MODE_COUNT -eq 0 ]]; then
    echo ""
    echo "ERROR: No torch mode selected. Set one of:"
    echo "  USE_EXISTING_TORCH=1            — keep existing torch, just verify imports"
    echo "  INSTALL_TORCH_CU128=1           — install stable cu128 torch family"
    echo "  INSTALL_TORCH_NIGHTLY_CU128=1   — install nightly cu128 (for Blackwell / new GPUs)"
    echo "  FIX_TORCH_FAMILY=1              — fix mismatched torchvision/torchaudio against existing torch"
    echo ""
    echo "  PyTorch version compatibility table:"
    echo "    torch 2.5.x  →  torchvision 0.20.x  →  torchaudio 2.5.x"
    echo "    torch 2.6.x  →  torchvision 0.21.x  →  torchaudio 2.6.x"
    echo "    torch 2.7.x  →  torchvision 0.22.x  →  torchaudio 2.7.x"
    echo "    torch 2.8.x  →  torchvision 0.23.x  →  torchaudio 2.8.x"
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

echo "── PyTorch version compatibility table ─────────────────────"
echo "    torch 2.5.x  →  torchvision 0.20.x  →  torchaudio 2.5.x"
echo "    torch 2.6.x  →  torchvision 0.21.x  →  torchaudio 2.6.x"
echo "    torch 2.7.x  →  torchvision 0.22.x  →  torchaudio 2.7.x"
echo "    torch 2.8.x  →  torchvision 0.23.x  →  torchaudio 2.8.x"
echo "    (torchaudio always matches torch; torchvision minor = torch minor + 15)"
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
# wheel index and the same build.  Mixing sources (e.g. torchvision from PyPI
# while torch came from the PyTorch cu128 index) causes ABI mismatches that
# manifest as:
#   - RuntimeError: operator torchvision::nms does not exist
#   - OSError: libtorchaudio.so: undefined symbol ...
# ---------------------------------------------------------------------------

if [[ $USE_EXISTING_TORCH -eq 1 ]]; then
    echo "── [2/3] Torch mode: USE_EXISTING_TORCH ─────────────────────"
    echo "  Verifying existing torch, torchvision, torchaudio imports ..."
    python - <<'PYEOF'
import sys

errors = []

try:
    import torch
    print(f"  torch       {torch.__version__}")
except ImportError as e:
    errors.append(f"torch import failed: {e}")

try:
    import torchvision
    print(f"  torchvision {torchvision.__version__}")
    # Probe native ops
    import torchvision.ops as tvops
    tvops.nms(
        torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32),
        torch.tensor([0.9], dtype=torch.float32),
        iou_threshold=0.5,
    )
    print("  torchvision native ops OK")
except RuntimeError as e:
    if "does not exist" in str(e) or "nms" in str(e).lower():
        print(f"  WARNING: torchvision ABI mismatch: {e}", file=sys.stderr)
        print("  Run: FIX_TORCH_FAMILY=1 bash setup_env.sh", file=sys.stderr)
    else:
        print(f"  WARNING: torchvision error: {e}", file=sys.stderr)
except ImportError as e:
    print(f"  torchvision NOT FOUND ({e})")

try:
    import torchaudio
    print(f"  torchaudio  {torchaudio.__version__}")
except (OSError, ImportError) as e:
    msg = str(e)
    if "undefined symbol" in msg or "cannot open shared object" in msg:
        print(f"  WARNING: torchaudio ABI mismatch: {e}", file=sys.stderr)
        print("  Run: FIX_TORCH_FAMILY=1 bash setup_env.sh", file=sys.stderr)
    else:
        print(f"  torchaudio NOT FOUND ({e})")

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

elif [[ $FIX_TORCH_FAMILY -eq 1 ]]; then
    echo "── [2/3] Torch mode: FIX_TORCH_FAMILY ──────────────────────"
    echo "  Detecting installed torch version and computing matching wheels ..."
    echo ""

    # Compute matching torchvision + torchaudio versions from installed torch
    TORCH_INFO=$(python3 - <<'PYEOF'
import sys, re

try:
    import torch
    ver = torch.__version__
except ImportError:
    print("ERROR: torch is not installed. Cannot fix family.")
    sys.exit(1)

# Parse: "2.7.1+cu128"  or  "2.7.1"  or  "2.8.0.dev20250501+cu128"
m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\.dev\d+)?(?:\+(\w+))?", ver)
if not m:
    print(f"ERROR: cannot parse torch version: {ver}")
    sys.exit(1)

major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
cuda_tag = m.group(4) or "cpu"
torch_base = f"{major}.{minor}.{patch}"

# torchvision: torch 2.x → torchvision 0.(minor+15).patch
if major == 2:
    tv_ver = f"0.{minor + 15}.{patch}"
else:
    # Fallback: cannot auto-compute; user must set manually
    print(f"ERROR: cannot auto-compute torchvision version for torch {major}.x")
    sys.exit(1)

ta_ver = torch_base

print(f"{torch_base} {cuda_tag} {tv_ver} {ta_ver}")
PYEOF
    )

    if [[ "$TORCH_INFO" == ERROR* ]]; then
        echo "  $TORCH_INFO"
        exit 1
    fi

    TORCH_BASE=$(echo "$TORCH_INFO" | awk '{print $1}')
    CUDA_TAG=$(echo   "$TORCH_INFO" | awk '{print $2}')
    TV_VER=$(echo     "$TORCH_INFO" | awk '{print $3}')
    TA_VER=$(echo     "$TORCH_INFO" | awk '{print $4}')
    INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

    echo "  Installed torch : ${TORCH_BASE}+${CUDA_TAG}"
    echo "  Target torchvision : ${TV_VER}"
    echo "  Target torchaudio  : ${TA_VER}"
    echo "  Index URL          : ${INDEX_URL}"
    echo ""
    echo "  Using --force-reinstall --no-deps so torch itself is not changed."
    echo ""

    # Try with explicit +CUDA suffix first (e.g. 0.22.1+cu128)
    echo "  Attempt 1: install with +${CUDA_TAG} suffix ..."
    if $PIP install --force-reinstall --no-deps \
        "torchvision==${TV_VER}+${CUDA_TAG}" \
        "torchaudio==${TA_VER}+${CUDA_TAG}" \
        --index-url "${INDEX_URL}" 2>/dev/null; then
        echo "  ✓ Installed with +${CUDA_TAG} suffix"
    else
        echo "  Exact +${CUDA_TAG} wheel not found."
        echo "  Attempt 2: install without suffix (let index serve correct build) ..."
        if $PIP install --force-reinstall --no-deps \
            "torchvision==${TV_VER}" \
            "torchaudio==${TA_VER}" \
            --index-url "${INDEX_URL}"; then
            echo "  ✓ Installed without explicit CUDA suffix"
        else
            echo ""
            echo "  ERROR: Could not find a matching wheel for:"
            echo "    torchvision==${TV_VER}  torchaudio==${TA_VER}"
            echo "  on index: ${INDEX_URL}"
            echo ""
            echo "  Options:"
            echo "    1. Try INSTALL_TORCH_CU128=1 to reinstall the full torch family at once"
            echo "    2. Check available wheels manually:"
            echo "       pip index versions torchvision --index-url ${INDEX_URL}"
            exit 1
        fi
    fi
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
