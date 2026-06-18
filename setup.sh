#!/usr/bin/env bash
# setup.sh — one-shot environment setup for qwen_swiglu_pruning
#
# Usage:
#   bash setup.sh          # auto-detect CUDA, install everything
#   bash setup.sh --cpu    # force CPU-only PyTorch (for testing / no GPU)
#
# What this does:
#   1. Detect CUDA version from nvidia-smi (or nvcc)
#   2. Install PyTorch + torchvision from the correct wheel index
#   3. pip install -r requirements.txt  (skips torch/torchvision, already handled)
#   4. Quick sanity check: python -c "import torch; print(torch.__version__)"
# ---------------------------------------------------------------------------

set -euo pipefail

FORCE_CPU=0
for arg in "$@"; do
    [[ "$arg" == "--cpu" ]] && FORCE_CPU=1
done

PIP="python -m pip"

# ── 1. Detect CUDA version ──────────────────────────────────────────────────
detect_cuda_tag() {
    if [[ $FORCE_CPU -eq 1 ]]; then
        echo "cpu"
        return
    fi

    # Try nvidia-smi first (most reliable on driver-only machines)
    if command -v nvidia-smi &>/dev/null; then
        CUDA_STR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || true)
        if [[ -n "$CUDA_STR" ]]; then
            MAJOR=$(echo "$CUDA_STR" | cut -d. -f1)
            MINOR=$(echo "$CUDA_STR" | cut -d. -f2)
            if   [[ $MAJOR -ge 12 && $MINOR -ge 4 ]]; then echo "cu124"
            elif [[ $MAJOR -ge 12 && $MINOR -ge 1 ]]; then echo "cu121"
            elif [[ $MAJOR -ge 11 && $MINOR -ge 8 ]]; then echo "cu118"
            else echo "cu118"   # safe minimum
            fi
            return
        fi
    fi

    # Try nvcc as fallback
    if command -v nvcc &>/dev/null; then
        CUDA_STR=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1 || true)
        if [[ -n "$CUDA_STR" ]]; then
            MAJOR=$(echo "$CUDA_STR" | cut -d. -f1)
            MINOR=$(echo "$CUDA_STR" | cut -d. -f2)
            if   [[ $MAJOR -ge 12 && $MINOR -ge 4 ]]; then echo "cu124"
            elif [[ $MAJOR -ge 12 && $MINOR -ge 1 ]]; then echo "cu121"
            elif [[ $MAJOR -ge 11 && $MINOR -ge 8 ]]; then echo "cu118"
            else echo "cu118"
            fi
            return
        fi
    fi

    echo "cpu"
}

CUDA_TAG=$(detect_cuda_tag)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  qwen_swiglu_pruning setup"
echo "  Detected build tag: ${CUDA_TAG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 2. Install PyTorch ───────────────────────────────────────────────────────
if [[ "$CUDA_TAG" == "cpu" ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cpu"
    echo "[1/3] Installing PyTorch (CPU-only) ..."
else
    INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
    echo "[1/3] Installing PyTorch + CUDA (${CUDA_TAG}) ..."
fi

$PIP install --upgrade pip --quiet
$PIP install torch torchvision --index-url "$INDEX_URL"

# ── 3. Install remaining requirements ───────────────────────────────────────
echo ""
echo "[2/3] Installing other requirements ..."
# torch/torchvision already handled above; pass them so pip doesn't downgrade
$PIP install -r requirements.txt \
    --ignore-installed torch torchvision \
    --extra-index-url "$INDEX_URL"

# ── 4. Sanity check ─────────────────────────────────────────────────────────
echo ""
echo "[3/3] Sanity checks ..."
python - <<'EOF'
import sys
import importlib

ok = True
checks = [
    ("torch",           "torch"),
    ("transformers",    "transformers"),
    ("datasets",        "datasets"),
    ("accelerate",      "accelerate"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors",     "safetensors"),
    ("sentencepiece",   "sentencepiece"),
    ("numpy",           "numpy"),
    ("pandas",          "pandas"),
    ("tqdm",            "tqdm"),
    ("yaml",            "pyyaml"),
]

for mod, pkg in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓  {mod:<22} {ver}")
    except ImportError:
        print(f"  ✗  {mod:<22} NOT FOUND  (pip install {pkg})", file=sys.stderr)
        ok = False

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"\n  torch.cuda.is_available() = {cuda_ok}")
    if cuda_ok:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA runtime: {torch.version.cuda}")
    else:
        print("  (CPU-only build or no GPU detected)")
except Exception as e:
    print(f"  torch check failed: {e}", file=sys.stderr)
    ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete."
echo "  Run: python run_experiment.py --config configs/moe_full48_packed_p95_1pct_pure_delete_wikitext2_n64.yaml --moe-target-pruning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
