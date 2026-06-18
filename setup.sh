#!/usr/bin/env bash
# setup.sh — one-shot environment setup for qwen_swiglu_pruning
#
# Usage:
#   bash setup.sh            # auto-detect CUDA, install everything
#   bash setup.sh --cpu      # force CPU-only PyTorch (for testing / no GPU)
#   bash setup.sh --reinstall-torch  # force reinstall torch even if already present
#
# What this does:
#   1. Check if a working CUDA torch is already installed; if so, skip reinstall
#      (avoids downgrading cloud/Docker images that ship custom torch builds)
#   2. If torch is missing or broken, install from the correct wheel index
#   3. pip install -r requirements.txt for everything else
#   4. Quick sanity check (torchaudio intentionally excluded — not used by this project)
# ---------------------------------------------------------------------------

set -euo pipefail

FORCE_CPU=0
REINSTALL_TORCH=0
for arg in "$@"; do
    [[ "$arg" == "--cpu"             ]] && FORCE_CPU=1
    [[ "$arg" == "--reinstall-torch" ]] && REINSTALL_TORCH=1
done

PIP="python -m pip"

# ── 1. Detect CUDA version ──────────────────────────────────────────────────
detect_cuda_tag() {
    if [[ $FORCE_CPU -eq 1 ]]; then echo "cpu"; return; fi

    if command -v nvidia-smi &>/dev/null; then
        CUDA_STR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || true)
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

$PIP install --upgrade pip --quiet

# ── 2. Install PyTorch (skip if already working) ────────────────────────────
# Many cloud / Docker images ship custom torch builds (e.g. +cu130, +cu126)
# that are NOT available on download.pytorch.org/whl.  Force-reinstalling
# from the public index would downgrade and break them.
# Only install if torch is absent, broken, or --reinstall-torch was passed.

TORCH_OK=0
if [[ $REINSTALL_TORCH -eq 0 ]]; then
    if python -c "import torch; torch.tensor([1.0])" 2>/dev/null; then
        TORCH_VER=$(python -c "import torch; print(torch.__version__)")
        echo "[1/3] torch ${TORCH_VER} already installed — skipping reinstall."
        echo "      (pass --reinstall-torch to force reinstall from public index)"
        TORCH_OK=1
    fi
fi

if [[ $TORCH_OK -eq 0 ]]; then
    if [[ "$CUDA_TAG" == "cpu" ]]; then
        INDEX_URL="https://download.pytorch.org/whl/cpu"
        echo "[1/3] Installing PyTorch (CPU-only) ..."
    else
        INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
        echo "[1/3] Installing PyTorch + CUDA (${CUDA_TAG}) ..."
    fi
    # NOTE: torchaudio is intentionally excluded — it is not used by this project
    # and its .so files frequently cause ABI mismatch errors on pre-configured images.
    $PIP install torch torchvision --index-url "$INDEX_URL"
fi

# ── 3. Install remaining requirements ───────────────────────────────────────
echo ""
echo "[2/3] Installing other requirements ..."
$PIP install \
    "transformers>=4.43.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "huggingface_hub>=0.23.0" \
    "safetensors>=0.4.3" \
    "tokenizers>=0.19.0" \
    "sentencepiece>=0.1.99" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0"

# ── 4. Sanity check ─────────────────────────────────────────────────────────
echo ""
echo "[3/3] Sanity checks ..."
python - <<'EOF'
import sys
import importlib

ok = True
# torchaudio is intentionally excluded: this project does not use audio
# and torchaudio .so files cause ABI errors on many Docker/cloud images.
checks = [
    ("torch",           "torch"),
    ("torchvision",     "torchvision"),
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
    except ImportError as e:
        print(f"  ✗  {mod:<22} NOT FOUND  ({e})", file=sys.stderr)
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
