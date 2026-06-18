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


# ─────────────────────────────────────────────────────────────────────────────
# A. Python package imports
# ─────────────────────────────────────────────────────────────────────────────
_hdr("A. Python imports")

REQUIRED_IMPORTS = [
    ("torch",           True),
    ("torchvision",     True),
    ("torchaudio",      False),   # soft: see Part C for ABI details
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

imported = {}
for mod, required in REQUIRED_IMPORTS:
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
                note = "  ← Blackwell / sm_120+; stable torch may not support this GPU — use nightly if needed"
            _ok(f"GPU {i}: {name}  ({sm}){note}")
    else:
        _warn("No CUDA GPUs detected — running on CPU")

# ─────────────────────────────────────────────────────────────────────────────
# C. Torch-family compatibility
# ─────────────────────────────────────────────────────────────────────────────
_hdr("C. Torch-family compatibility")

if torch is not None:
    tv  = imported.get("torchvision")
    ta  = imported.get("torchaudio")

    torch_ver = torch.__version__
    tv_ver    = getattr(tv, "__version__", None) if tv else None
    ta_ver    = getattr(ta, "__version__", None) if ta else None

    _ok(f"torch       {torch_ver}")

    if tv_ver:
        _ok(f"torchvision {tv_ver}")
        # Rough version prefix check (major.minor should match torch)
        torch_mm = ".".join(torch_ver.split(".")[:2])
        tv_mm    = ".".join(tv_ver.split(".")[:2])
        if torch_mm != tv_mm:
            _warn(
                f"torch ({torch_mm}) and torchvision ({tv_mm}) major.minor differ — "
                "install from the same PyTorch wheel index"
            )
    else:
        _warn("torchvision not installed")

    if ta_ver:
        _ok(f"torchaudio  {ta_ver}")
        # Check that torchaudio can actually load its native extension
        try:
            torch.ops.load_library  # noqa
            _ok("torchaudio native ext loaded OK")
        except Exception as e:
            _warn(f"torchaudio native ext may have issues: {e}")
    else:
        _warn(
            "torchaudio not installed.\n"
            "     This project does not use torchaudio directly, but recent\n"
            "     Transformers may import it indirectly. If model loading fails\n"
            "     with a .so symbol error, install torchaudio from the same\n"
            "     PyTorch wheel index as torch:\n"
            "       INSTALL_TORCH_CU128=1 bash setup_env.sh"
        )

# ─────────────────────────────────────────────────────────────────────────────
# D. Transformers Qwen3 MoE
# ─────────────────────────────────────────────────────────────────────────────
_hdr("D. Transformers Qwen3 MoE")

transformers = imported.get("transformers")
if transformers is None:
    _fail("transformers not available — skipping Qwen3 checks")
else:
    _ok(f"transformers {transformers.__version__}")

    # Check that Qwen3MoeForCausalLM class is registered
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        qwen3_moe_present = any("qwen3_moe" in str(k).lower() for k in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
        if qwen3_moe_present:
            _ok("Qwen3MoeForCausalLM is registered in AutoModel mapping")
        else:
            _warn(
                "Qwen3MoeForCausalLM not found in AutoModel mapping.\n"
                "     This requires transformers >= 4.51.0.\n"
                f"     Installed: {transformers.__version__}"
            )
    except Exception as e:
        _warn(f"Could not check Qwen3 registration: {e}")

    # Fetch config from Hub (requires internet or cached model)
    print(f"  Fetching Qwen/Qwen3-30B-A3B config from Hub (needs internet or cache) ...")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
        _ok(f"AutoConfig loaded — model_type: {cfg.model_type}")
        if hasattr(cfg, "num_experts"):
            _ok(f"  num_experts:        {cfg.num_experts}")
        if hasattr(cfg, "num_experts_per_tok"):
            _ok(f"  num_experts_per_tok:{cfg.num_experts_per_tok}")
        if hasattr(cfg, "intermediate_size"):
            _ok(f"  intermediate_size:  {cfg.intermediate_size}")
        if hasattr(cfg, "num_hidden_layers"):
            _ok(f"  num_hidden_layers:  {cfg.num_hidden_layers}")
    except Exception as e:
        _warn(
            f"AutoConfig.from_pretrained('Qwen/Qwen3-30B-A3B') failed: {e}\n"
            "     This is expected if there is no internet access and the model\n"
            "     is not yet cached.  Run scripts/prepare_models.py to download."
        )

# ─────────────────────────────────────────────────────────────────────────────
# E. Repo syntax check
# ─────────────────────────────────────────────────────────────────────────────
_hdr("E. Repo syntax check")

FILES_TO_CHECK = [
    "run_experiment.py",
    "src/moe_pruning.py",
    "src/model_utils.py",
    "src/scoring.py",
    "src/pruning.py",
    "src/evaluation.py",
    "scripts/summarize_moe_results.py",
    "scripts/check_env.py",
    "scripts/prepare_models.py",
]

for rel in FILES_TO_CHECK:
    fpath = os.path.join(REPO_ROOT, rel)
    if not os.path.exists(fpath):
        _warn(f"{rel} — file not found (skipped)")
        continue
    try:
        py_compile.compile(fpath, doraise=True)
        _ok(f"{rel}")
    except py_compile.PyCompileError as e:
        _fail(f"{rel} — SYNTAX ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("━" * 60)
if warnings_issued:
    print(f"  {len(warnings_issued)} warning(s):")
    for w in warnings_issued:
        first_line = w.split("\n")[0]
        print(f"    ⚠  {first_line}")
    print()

if overall_ok:
    print("  ENV CHECK PASSED")
    print("━" * 60)
    sys.exit(0)
else:
    print("  ENV CHECK FAILED — fix the errors above before running experiments")
    print("━" * 60)
    sys.exit(1)
