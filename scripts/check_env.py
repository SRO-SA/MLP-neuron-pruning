#!/usr/bin/env python3
"""
check_env.py — comprehensive environment verification for qwen_swiglu_pruning.

Sections
--------
  A. Python version
  B. Python package imports
  C. Torch / CUDA availability + device info
  D. Torch-family version compatibility
  E. Transformers — Qwen3 MoE config fetch
  F. Repo syntax check (py_compile)
  G. [--strict] Exact version verification against env_expected.yaml
  H. [--strict] Model layout check (requires GPU + model weights)

Modes
-----
  Default:             sections A-F, soft version warnings only
  --strict:            additionally runs G + H; fails on any version mismatch
  --skip-model-layout: skip section H even in --strict mode

Exit codes
----------
  0  all critical checks passed
  1  one or more critical checks failed
"""

import argparse
import importlib
import os
import py_compile
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="qwen_swiglu_pruning environment check")
parser.add_argument("--strict", action="store_true",
                    help="Fail on any version mismatch against env_expected.yaml")
parser.add_argument("--skip-model-layout", action="store_true",
                    help="Skip section H (model layout check). Avoids downloading weights.")
args = parser.parse_args()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS = "  ✓"
WARN = "  ⚠"
FAIL = "  ✗"

overall_ok = True
warnings_issued: list = []

FIX_CMD = "RESET_ENV=1 bash setup_repro_env.sh"


def _hdr(title: str) -> None:
    print(f"\n── {title} {chr(0x2500) * max(0, 60 - len(title))}")


def _ok(msg: str) -> None:
    print(f"{PASS}  {msg}")


def _warn(msg: str) -> None:
    print(f"{WARN}  {msg}")
    warnings_issued.append(msg)


def _fail(msg: str, strict_only: bool = False) -> None:
    global overall_ok
    if strict_only and not args.strict:
        _warn(msg)
        return
    print(f"{FAIL}  {msg}", file=sys.stderr)
    overall_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# A. Python version
# ─────────────────────────────────────────────────────────────────────────────
_hdr("A. Python version")
py_ver = sys.version_info
py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
_ok(f"Python {py_str}")

if py_ver.major != 3 or py_ver.minor != 10:
    _fail(
        f"Python {py_str} detected. Paper environment used Python 3.10.x. "
        "Other versions may work but are untested.",
        strict_only=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# B. Python package imports
# ─────────────────────────────────────────────────────────────────────────────
_hdr("B. Python package imports")

REQUIRED_PKGS = [
    ("torch",            True),
    ("torchvision",      True),
    ("torchaudio",       True),
    ("transformers",     True),
    ("datasets",         True),
    ("accelerate",       True),
    ("tokenizers",       True),
    ("safetensors",      True),
    ("numpy",            True),
    ("scipy",            True),
    ("pandas",           True),
    ("yaml",             True),   # pyyaml
    ("huggingface_hub",  False),
    ("sentencepiece",    False),
    ("tqdm",             False),
    ("psutil",           False),
    ("matplotlib",       False),
    ("sklearn",          False),
]

imported: dict = {}
for pkg, required in REQUIRED_PKGS:
    try:
        m = importlib.import_module(pkg)
        imported[pkg] = m
        ver = getattr(m, "__version__", "?")
        _ok(f"{pkg:<22} {ver}")
    except Exception as exc:
        imported[pkg] = None
        if required:
            _fail(f"{pkg:<22} MISSING — {exc}")
        else:
            _warn(f"{pkg:<22} not installed (optional) — {exc}")

torch = imported.get("torch")

# ─────────────────────────────────────────────────────────────────────────────
# C. Torch / CUDA availability
# ─────────────────────────────────────────────────────────────────────────────
_hdr("C. Torch / CUDA")

if torch is None:
    _fail("torch not available — skipping all CUDA checks")
else:
    _ok(f"torch version       {torch.__version__}")
    _ok(f"torch.version.cuda  {torch.version.cuda or 'N/A'}")

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        _ok("cuda.is_available() True")
    else:
        _fail("cuda.is_available() False — GPU required for MoE pruning experiments")

    n_dev = torch.cuda.device_count()
    if n_dev > 0:
        _ok(f"cuda.device_count() {n_dev}")
        for i in range(n_dev):
            name = torch.cuda.get_device_name(i)
            cap  = torch.cuda.get_device_capability(i)
            sm   = f"sm_{cap[0]}{cap[1]}"
            note = "  ← Blackwell / sm_120+" if cap[0] >= 12 else ""
            _ok(f"  GPU {i}: {name}  ({sm}){note}")
    else:
        _fail("GPU count == 0 — no GPUs detected")

# ─────────────────────────────────────────────────────────────────────────────
# D. Torch-family version compatibility
# ─────────────────────────────────────────────────────────────────────────────
_hdr("D. Torch-family compatibility")

if torch is not None:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__)
    torch_major = int(m.group(1)) if m else None
    torch_minor = int(m.group(2)) if m else None
    torch_patch = int(m.group(3)) if m else None

    if torch_major == 2 and torch_minor is not None:
        exp_tv = f"0.{torch_minor + 15}.{torch_patch}"
        exp_ta = f"{torch_major}.{torch_minor}.{torch_patch}"
    else:
        exp_tv = exp_ta = None

    tv = imported.get("torchvision")
    if tv is not None:
        _ok(f"torchvision {tv.__version__}  (expected prefix {exp_tv or '?'})")
        if exp_tv and not tv.__version__.startswith(exp_tv.rsplit(".", 1)[0]):
            _fail(
                f"torchvision {tv.__version__} does not match torch {torch.__version__}. "
                f"Fix: {FIX_CMD}",
                strict_only=True,
            )
    else:
        _warn("torchvision missing or broken")

    ta = imported.get("torchaudio")
    if ta is not None:
        _ok(f"torchaudio  {ta.__version__}  (expected prefix {exp_ta or '?'})")
        if exp_ta and not ta.__version__.startswith(exp_ta.rsplit(".", 1)[0]):
            _fail(
                f"torchaudio {ta.__version__} does not match torch {torch.__version__}. "
                f"Fix: {FIX_CMD}",
                strict_only=True,
            )
    else:
        _warn("torchaudio missing or broken")

# ─────────────────────────────────────────────────────────────────────────────
# E. Transformers — Qwen3 MoE
# ─────────────────────────────────────────────────────────────────────────────
_hdr("E. Transformers / Qwen3 MoE")

transformers = imported.get("transformers")
if transformers is None:
    _fail("transformers not available — skipping Qwen3 checks")
else:
    _ok(f"transformers {transformers.__version__}")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "?")
        _ok(f"AutoConfig loaded  model_type={model_type!r}")
        if "qwen3" not in model_type.lower():
            _fail(f"Expected Qwen3 MoE model_type, got {model_type!r}", strict_only=True)
        else:
            n_exp = getattr(cfg, "num_experts", "?")
            d_ff  = getattr(cfg, "moe_intermediate_size", "?")
            d_mod = getattr(cfg, "hidden_size", "?")
            _ok(f"  num_experts={n_exp}  moe_intermediate_size={d_ff}  hidden_size={d_mod}")
    except Exception as exc:
        _warn(f"AutoConfig.from_pretrained failed: {exc} (model weights may not be downloaded)")

# ─────────────────────────────────────────────────────────────────────────────
# F. Repo syntax check
# ─────────────────────────────────────────────────────────────────────────────
_hdr("F. Repo syntax (py_compile)")

SYNTAX_CHECK_FILES = [
    "run_experiment.py",
    "src/moe_pruning.py",
    "scripts/summarize_moe_results.py",
]

for rel_path in SYNTAX_CHECK_FILES:
    full_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(full_path):
        _warn(f"{rel_path}  NOT FOUND")
        continue
    try:
        py_compile.compile(full_path, doraise=True)
        _ok(f"{rel_path}  OK")
    except py_compile.PyCompileError as exc:
        _fail(f"{rel_path}  SYNTAX ERROR — {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# G. [--strict] Exact version verification against env_expected.yaml
# ─────────────────────────────────────────────────────────────────────────────
if args.strict:
    _hdr("G. Strict version check (env_expected.yaml)")

    expected_yaml = os.path.join(REPO_ROOT, "env_expected.yaml")
    if not os.path.exists(expected_yaml):
        _fail(f"env_expected.yaml not found at {expected_yaml}")
    else:
        import yaml as _yaml  # type: ignore
        with open(expected_yaml) as fh:
            exp = _yaml.safe_load(fh)

        def _check_ver(label: str, actual: str, expected: str) -> None:
            """Compare version strings, ignoring local suffixes like +cu128."""
            actual_base   = actual.split("+")[0].strip()
            expected_base = expected.split("+")[0].strip()
            if actual_base == expected_base:
                _ok(f"{label:<30} {actual}  ✓ matches {expected}")
            else:
                _fail(
                    f"{label:<30} MISMATCH — got {actual!r}, expected {expected!r}. "
                    f"Fix: {FIX_CMD}"
                )

        # Python
        py_mm  = f"{py_ver.major}.{py_ver.minor}"
        exp_py = str(exp.get("python_major_minor", ""))
        if py_mm == exp_py:
            _ok(f"{'python (major.minor)':<30} {py_mm}  ✓")
        else:
            _fail(f"{'python (major.minor)':<30} MISMATCH — got {py_mm!r}, expected {exp_py!r}")

        # PyTorch stack
        if torch is not None:
            _check_ver("torch",      torch.__version__,       str(exp.get("torch", "")))
            _check_ver("torch_cuda", str(torch.version.cuda or ""), str(exp.get("torch_cuda", "")))
            tv = imported.get("torchvision")
            if tv:
                _check_ver("torchvision", tv.__version__, str(exp.get("torchvision", "")))
            ta = imported.get("torchaudio")
            if ta:
                _check_ver("torchaudio",  ta.__version__, str(exp.get("torchaudio", "")))

        # All other pinned packages
        PKG_MAP = {
            "transformers": "transformers",
            "datasets":     "datasets",
            "accelerate":   "accelerate",
            "tokenizers":   "tokenizers",
            "safetensors":  "safetensors",
            "numpy":        "numpy",
            "scipy":        "scipy",
            "pandas":       "pandas",
            "yaml":         "pyyaml",
        }
        for import_name, yaml_key in PKG_MAP.items():
            mod = imported.get(import_name)
            if mod is None:
                _fail(f"{yaml_key:<30} NOT INSTALLED")
                continue
            actual_ver   = getattr(mod, "__version__", "?")
            expected_ver = str(exp.get(yaml_key, ""))
            if expected_ver:
                _check_ver(yaml_key, actual_ver, expected_ver)

# ─────────────────────────────────────────────────────────────────────────────
# H. [--strict, unless --skip-model-layout] Model layout check
# ─────────────────────────────────────────────────────────────────────────────
if args.strict and not args.skip_model_layout:
    _hdr("H. Model layout check (Qwen/Qwen3-30B-A3B)")

    import yaml as _yaml  # type: ignore
    expected_yaml = os.path.join(REPO_ROOT, "env_expected.yaml")
    with open(expected_yaml) as fh:
        exp_layout = _yaml.safe_load(fh)

    exp_el   = exp_layout.get("expected_expert_layout", "unpacked")
    exp_gate = exp_layout.get("expected_expert_gate_shape", [768, 2048])
    exp_up   = exp_layout.get("expected_expert_up_shape",  [768, 2048])
    exp_down = exp_layout.get("expected_expert_down_shape", [2048, 768])

    if torch is None or not torch.cuda.is_available():
        _warn("No GPU — skipping model layout check (cannot load 30B model on CPU)")
    else:
        try:
            import contextlib
            import torch.nn as nn
            from transformers import AutoConfig, AutoModelForCausalLM

            _ok("Loading Qwen/Qwen3-30B-A3B (this may take a few minutes first time) ...")
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-30B-A3B",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            model.eval()

            # Detect expert layout from first MoE layer
            actual_layout = "unknown"
            gate_shape = up_shape = down_shape = None

            for _name, module in model.named_modules():
                if not hasattr(module, "experts"):
                    continue
                experts = module.experts
                if isinstance(experts, nn.ModuleList) and len(experts) > 0:
                    exp0 = experts[0]
                    if (hasattr(exp0, "gate_proj") and hasattr(exp0, "up_proj")
                            and hasattr(exp0, "down_proj")):
                        actual_layout = "unpacked"
                        gate_shape = list(exp0.gate_proj.weight.shape)
                        up_shape   = list(exp0.up_proj.weight.shape)
                        down_shape = list(exp0.down_proj.weight.shape)
                        break
                elif (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
                    actual_layout = "packed"
                    _, two_inter, hidden = experts.gate_up_proj.shape
                    intermediate = two_inter // 2
                    _, d_model, d_ff = experts.down_proj.shape
                    gate_shape = [intermediate, hidden]
                    up_shape   = [intermediate, hidden]
                    down_shape = [d_model, d_ff]
                    break

            del model
            torch.cuda.empty_cache()

            _ok(f"expert_layout      : {actual_layout}")
            _ok(f"expert[0].gate_proj: {gate_shape}")
            _ok(f"expert[0].up_proj  : {up_shape}")
            _ok(f"expert[0].down_proj: {down_shape}")

            layout_ok = (actual_layout == exp_el)
            shape_ok  = (gate_shape == exp_gate and up_shape == exp_up and down_shape == exp_down)

            if layout_ok and shape_ok:
                _ok("Model layout matches env_expected.yaml — locked environment confirmed")
            else:
                msgs = []
                if not layout_ok:
                    msgs.append(f"layout: got {actual_layout!r}, expected {exp_el!r}")
                if not shape_ok:
                    msgs.append(
                        f"shapes: got gate={gate_shape} up={up_shape} down={down_shape}, "
                        f"expected gate={exp_gate} up={exp_up} down={exp_down}"
                    )
                _fail(
                    "Model layout differs from the locked paper environment: "
                    + "; ".join(msgs)
                )

        except Exception as exc:
            _fail(f"Model layout check failed: {exc}")

elif args.strict and args.skip_model_layout:
    _hdr("H. Model layout check")
    _warn("Skipped (--skip-model-layout)")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 64)
if warnings_issued:
    print(f"  {len(warnings_issued)} warning(s):")
    for w in warnings_issued:
        first_line = w.splitlines()[0]
        print(f"    ⚠  {first_line}")
print()
if overall_ok:
    mode_tag = " [strict]" if args.strict else ""
    print(f"  ENV CHECK PASSED{mode_tag}")
    sys.exit(0)
else:
    print(f"  ENV CHECK FAILED — fix the errors above, then re-run:")
    print(f"    {FIX_CMD}")
    sys.exit(1)
