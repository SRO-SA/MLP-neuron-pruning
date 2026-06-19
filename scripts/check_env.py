#!/usr/bin/env python3
"""
check_env.py — comprehensive environment verification for qwen_swiglu_pruning.

Checks:
  A. Python package imports
  B. Torch / CUDA availability and device info
  C. Torch-family version compatibility (torch / torchvision / torchaudio)
  D. Transformers Qwen3 MoE import and config fetch
  E. Repo syntax check (py_compile)

Exits 0 if all critical checks pass, 1 otherwise.
Ends output with "ENV CHECK PASSED" on success.
"""

import importlib
import os
import py_compile
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS = "  ✓"
WARN = "  ⚠"
FAIL = "  ✗"

overall_ok = True
warnings_issued = []

FIX_CMD = "FIX_TORCH_FAMILY=1 bash setup_env.sh"


def _hdr(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}")


def _ok(msg: str) -> None:
    print(f"{PASS}  {msg}")


def _warn(msg: str) -> None:
    print(f"{WARN}  {msg}")
    warnings_issued.append(msg)


def _fail(msg: str) -> None:
    global overall_ok
    print(f"{FAIL}  {msg}", file=sys.stderr)
    overall_ok = False


def _is_abi_mismatch_torchvision(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "does not exist" in s
        or "torchvision::nms" in s
        or "no such operator" in s
        or "undefined symbol" in s
        or "cannot open shared object" in s
    )


def _is_abi_mismatch_torchaudio(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "undefined symbol" in s
        or "cannot open shared object" in s
        or "shared library" in s
        or "libtorchaudio" in s
        or "_torchaudio" in s
    )


# ─────────────────────────────────────────────────────────────────────────────
# A. Python package imports
# ─────────────────────────────────────────────────────────────────────────────
_hdr("A. Python imports")

imported: dict = {}

# ── torch (required) ─────────────────────────────────────────────────────────
try:
    import torch as _torch
    _ok(f"{'torch':<22} {_torch.__version__}")
    imported["torch"] = _torch
except ImportError as e:
    _fail(f"{'torch':<22} NOT FOUND — {e}")
    imported["torch"] = None

# ── torchvision (must be absent OR working; broken = hard fail) ───────────────
# Rule: if torchvision is installed, it must match torch exactly.
# Diagnosis path 1: RuntimeError at import ("operator ... does not exist")
# Diagnosis path 2: RuntimeError on functional probe (nms call)
# Diagnosis path 3: OSError / ImportError — not installed at all (soft warn)
try:
    import torchvision as _tv
    _tv_ver = _tv.__version__

    # Functional probe: actually exercise a native C++ op.
    # "operator torchvision::nms does not exist" fires here when the compiled
    # torchvision.so was built against a different torch version.
    if imported.get("torch") is not None:
        try:
            import torchvision.ops as _tvops
            _t = imported["torch"]
            _tvops.nms(
                _t.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=_t.float32),
                _t.tensor([0.9], dtype=_t.float32),
                iou_threshold=0.5,
            )
        except (RuntimeError, Exception) as _probe_err:
            if _is_abi_mismatch_torchvision(_probe_err):
                raise RuntimeError(str(_probe_err))   # re-raise for unified handler

    _ok(f"{'torchvision':<22} {_tv_ver}")
    imported["torchvision"] = _tv

except (RuntimeError, Exception) as e:
    if _is_abi_mismatch_torchvision(e):
        _fail(
            f"{'torchvision':<22} ABI MISMATCH\n"
            f"         torchvision is installed but its compiled ops do not match torch.\n"
            f"         Error : {e}\n"
            f"         Fix   : {FIX_CMD}\n"
            f"         This detects your torch version and reinstalls the matching\n"
            f"         torchvision wheel from the same PyTorch CUDA index."
        )
    elif isinstance(e, (ImportError, ModuleNotFoundError)):
        _warn(f"{'torchvision':<22} NOT INSTALLED (optional — warn only)")
    else:
        _fail(f"{'torchvision':<22} import error: {e}")
    imported["torchvision"] = None

# ── torchaudio (must be absent OR working; broken = hard fail) ────────────────
# Same rule as torchvision.  Shared-library errors raise OSError, not ImportError.
try:
    import torchaudio as _ta
    _ok(f"{'torchaudio':<22} {_ta.__version__}")
    imported["torchaudio"] = _ta

except (OSError, RuntimeError) as e:
    if _is_abi_mismatch_torchaudio(e):
        _fail(
            f"{'torchaudio':<22} ABI MISMATCH\n"
            f"         torchaudio is installed but does not match torch.\n"
            f"         Error : {e}\n"
            f"         Fix   : {FIX_CMD}\n"
            f"         This detects your torch version and reinstalls the matching\n"
            f"         torchaudio wheel from the same PyTorch CUDA index."
        )
    else:
        _fail(f"{'torchaudio':<22} failed to load: {e}")
    imported["torchaudio"] = None

except ImportError:
    _warn(
        f"{'torchaudio':<22} NOT INSTALLED (optional — warn only)\n"
        f"         This project does not use torchaudio directly, but recent\n"
        f"         Transformers may import it indirectly. If model loading fails\n"
        f"         with a .so symbol error, run: {FIX_CMD}"
    )
    imported["torchaudio"] = None

# ── Remaining packages ────────────────────────────────────────────────────────
OTHER_IMPORTS = [
    ("transformers",    True),
    ("datasets",        True),
    ("accelerate",      True),
    ("safetensors",     True),
    ("pandas",          True),
    ("yaml",            True),
    ("scipy",           True),
    ("numpy",           True),
    ("tqdm",            True),
    ("psutil",          True),
    ("sklearn",         True),
    ("matplotlib",      True),
]

for mod, required in OTHER_IMPORTS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        _ok(f"{mod:<22} {ver}")
        imported[mod] = m
    except ImportError as e:
        if required:
            _fail(f"{mod:<22} NOT FOUND — {e}")
        else:
            _warn(f"{mod:<22} NOT FOUND (optional) — {e}")
        imported[mod] = None

# ─────────────────────────────────────────────────────────────────────────────
# B. Torch / CUDA
# ─────────────────────────────────────────────────────────────────────────────
_hdr("B. Torch / CUDA")

torch = imported.get("torch")
if torch is None:
    _fail("torch not available — skipping all CUDA checks")
else:
    _ok(f"torch version       {torch.__version__}")
    _ok(f"torch.version.cuda  {torch.version.cuda or 'N/A'}")

    cuda_ok = torch.cuda.is_available()
    _ok(f"cuda.is_available() {cuda_ok}")

    n_dev = torch.cuda.device_count()
    _ok(f"cuda.device_count() {n_dev}")

    if cuda_ok and n_dev > 0:
        for i in range(n_dev):
            name = torch.cuda.get_device_name(i)
            cap  = torch.cuda.get_device_capability(i)
            sm   = f"sm_{cap[0]}{cap[1]}"
            note = ""
            if cap[0] >= 12:
                note = (
                    "  ← Blackwell / sm_120+; stable torch may not support this GPU — "
                    "use INSTALL_TORCH_NIGHTLY_CU128=1 if needed"
                )
            _ok(f"GPU {i}: {name}  ({sm}){note}")
    else:
        _warn("No CUDA GPUs detected — running on CPU")

# ─────────────────────────────────────────────────────────────────────────────
# C. Torch-family version compatibility
# ─────────────────────────────────────────────────────────────────────────────
_hdr("C. Torch-family compatibility")

# Version map (torch 2.x series):
#   torch 2.5.x  →  torchvision 0.20.x  →  torchaudio 2.5.x
#   torch 2.6.x  →  torchvision 0.21.x  →  torchaudio 2.6.x
#   torch 2.7.x  →  torchvision 0.22.x  →  torchaudio 2.7.x
#   torch 2.8.x  →  torchvision 0.23.x  →  torchaudio 2.8.x
# Pattern: torchvision minor = torch minor + 15  (for torch 2.x)
#          torchaudio version = torch version

if torch is not None:
    import re
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__)
    torch_major = int(m.group(1)) if m else None
    torch_minor = int(m.group(2)) if m else None
    torch_patch = int(m.group(3)) if m else None

    # Expected versions
    if torch_major == 2 and torch_minor is not None:
        exp_tv_minor = torch_minor + 15
        exp_tv = f"0.{exp_tv_minor}.{torch_patch}"
        exp_ta = f"{torch_major}.{torch_minor}.{torch_patch}"
    else:
        exp_tv = exp_ta = None

    _ok(f"torch       {torch.__version__}")

    tv = imported.get("torchvision")
    if tv is not None:
        _exp_tv_str = exp_tv if exp_tv else "?"
        _ok(f"torchvision {tv.__version__}  (expected {_exp_tv_str})")
        if exp_tv and not tv.__version__.startswith(exp_tv.rsplit(".", 1)[0]):
            _warn(
                f"torchvision version ({tv.__version__}) does not match expected "
                f"({exp_tv}) for torch {torch.__version__}.\n"
                f"         Fix: {FIX_CMD}"
            )
    else:
        if overall_ok:  # only warn if no hard fail already logged
            _warn("torchvision not installed or broken (see Section A)")

    ta = imported.get("torchaudio")
    if ta is not None:
        _exp_ta_str = exp_ta if exp_ta else "?"
        _ok(f"torchaudio  {ta.__version__}  (expected {_exp_ta_str})")