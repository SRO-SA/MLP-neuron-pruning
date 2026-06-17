"""
model_utils.py
==============
Utilities for loading the Qwen2.5 model / tokenizer and inspecting the
MLP weight matrices inside each transformer layer.

Qwen2.5 architecture recap
---------------------------
Each decoder layer contains:
  - self_attn           : grouped-query attention
  - mlp                 : SwiGLU block
      gate_proj  : Linear(d_model → d_ff)   weight shape [d_ff, d_model]
      up_proj    : Linear(d_model → d_ff)   weight shape [d_ff, d_model]
      down_proj  : Linear(d_ff   → d_model) weight shape [d_model, d_ff]
  - input_layernorm          : RMSNorm before the attention sub-layer
  - post_attention_layernorm : RMSNorm before the MLP sub-layer  ← we use this

MLP forward pass (layer-wise):
  g  = r @ gate_proj.weight.T          shape [seq, d_ff]
  u  = r @ up_proj.weight.T            shape [seq, d_ff]
  a  = SiLU(g) * u                     shape [seq, d_ff]   (SwiGLU activation)
  m  = a @ down_proj.weight.T          shape [seq, d_model]

Neuron-wise contribution (single token vector r ∈ ℝ^d_model):
  m(r) = Σ_i  SiLU(r · w_gate_i) * (r · w_up_i) * w_down_i

where
  w_gate_i  = gate_proj.weight[i, :]   (row i, shape [d_model])
  w_up_i    = up_proj.weight[i,   :]   (row i, shape [d_model])
  w_down_i  = down_proj.weight[:, i]   (col i, shape [d_model])

Pruning neuron i therefore removes:
  • row    i from gate_proj.weight
  • row    i from up_proj.weight
  • column i from down_proj.weight
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    fallback_name: Optional[str] = None,
    device: str = "auto",
    dtype_str: str = "float32",
    device_map: Optional[str] = None,
    max_memory: Optional[dict] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load a causal LM and tokenizer from HuggingFace Hub.

    Returns (model, tokenizer, resolved_model_name).
    Falls back to *fallback_name* if the primary load raises an exception.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float32)

    # Resolve device
    if device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    logger.info("Using device: %s", device_str)

    names_to_try = [model_name]
    if fallback_name and fallback_name != model_name:
        names_to_try.append(fallback_name)

    last_exc: Optional[Exception] = None
    for name in names_to_try:
        try:
            logger.info("Loading tokenizer: %s", name)
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

            logger.info("Loading model: %s  (dtype=%s, device=%s, device_map=%s)",
                        name, dtype_str, device_str, device_map)
            _load_kw = dict(torch_dtype=torch_dtype, trust_remote_code=True)
            if device_map is not None:
                _load_kw["device_map"] = device_map
                if max_memory is not None:
                    _load_kw["max_memory"] = max_memory
            model = AutoModelForCausalLM.from_pretrained(name, **_load_kw)
            if device_map is None:
                model = model.to(device_str)
            model.eval()
            if device_map is not None and hasattr(model, "hf_device_map"):
                _dm = model.hf_device_map
                _dv: dict = {}
                for _ln, _dv_id in _dm.items():
                    _dv[str(_dv_id)] = _dv.get(str(_dv_id), 0) + 1
                print(f"  hf_device_map summary: {dict(sorted(_dv.items()))}")

            # Ensure pad token exists
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            logger.info("Model loaded successfully: %s", name)
            try:
                print_model_info(model)
            except Exception as pmi_exc:
                logger.warning(
                    "print_model_info failed (non-fatal): %s", pmi_exc
                )
            return model, tokenizer, name

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", name, exc)
            last_exc = exc

    raise RuntimeError(
        f"Could not load any of {names_to_try}. Last error: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Architecture inspection
# ---------------------------------------------------------------------------

def get_transformer_layers(model: AutoModelForCausalLM):
    """Return the list of transformer decoder layers."""
    # Qwen2.5 / LLaMA-style: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback
    raise AttributeError(
        "Cannot locate transformer layers. "
        "Expected model.model.layers (Qwen2/LLaMA style)."
    )


def get_mlp_module(layer) -> object:
    """Return the MLP sub-module from a decoder layer."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    raise AttributeError(f"Layer {layer} has no 'mlp' attribute.")


def get_rmsnorm_before_mlp(layer) -> torch.nn.Module:
    """Return the RMSNorm applied before the MLP sub-layer."""
    # Qwen2.5 / LLaMA: post_attention_layernorm
    if hasattr(layer, "post_attention_layernorm"):
        return layer.post_attention_layernorm
    raise AttributeError(
        "Cannot find RMSNorm before MLP. "
        "Expected layer.post_attention_layernorm."
    )


def get_mlp_weights(layer) -> Dict[str, torch.Tensor]:
    """
    Extract the three MLP weight tensors for *layer*.

    Returns a dict with keys:
        'gate'  : gate_proj.weight  shape [d_ff, d_model]
        'up'    : up_proj.weight    shape [d_ff, d_model]
        'down'  : down_proj.weight  shape [d_model, d_ff]
        'd_model': int
        'd_ff'  : int

    Shape assertions are included to catch layout surprises early.
    """
    mlp = get_mlp_module(layer)

    assert hasattr(mlp, "gate_proj"), "MLP missing gate_proj"
    assert hasattr(mlp, "up_proj"),   "MLP missing up_proj"
    assert hasattr(mlp, "down_proj"), "MLP missing down_proj"

    w_gate = mlp.gate_proj.weight  # [d_ff, d_model]
    w_up   = mlp.up_proj.weight    # [d_ff, d_model]
    w_down = mlp.down_proj.weight  # [d_model, d_ff]

    d_ff_gate, d_model_gate = w_gate.shape
    d_ff_up,   d_model_up   = w_up.shape
    d_model_down, d_ff_down = w_down.shape

    # Shape consistency checks
    assert d_ff_gate == d_ff_up, (
        f"gate_proj d_ff={d_ff_gate} != up_proj d_ff={d_ff_up}"
    )
    assert d_ff_gate == d_ff_down, (
        f"gate_proj d_ff={d_ff_gate} != down_proj d_ff={d_ff_down}"
    )
    assert d_model_gate == d_model_up == d_model_down, (
        f"d_model mismatch: gate={d_model_gate}, up={d_model_up}, down={d_model_down}"
    )

    return {
        "gate":    w_gate,
        "up":      w_up,
        "down":    w_down,
        "d_model": d_model_gate,
        "d_ff":    d_ff_gate,
    }


def print_model_info(model: AutoModelForCausalLM) -> None:
    """Print a summary of model architecture and MLP shapes per layer.

    Handles both dense (Qwen2.5-style) and MoE (Qwen3MoE-style) first layers
    without raising an AssertionError when gate_proj is absent.
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"  Model class   : {type(model).__name__}")

    cfg = model.config
    print(f"  Hidden size   : {getattr(cfg, 'hidden_size', '?')}")
    print(f"  Intermediate  : {getattr(cfg, 'intermediate_size', '?')}")
    print(f"  MoE inter.    : {getattr(cfg, 'moe_intermediate_size', '?')}")
    print(f"  Num layers    : {getattr(cfg, 'num_hidden_layers', '?')}")
    print(f"  Num heads     : {getattr(cfg, 'num_attention_heads', '?')}")
    print(f"  Num experts   : {getattr(cfg, 'num_experts', '?')}")
    print(f"  Experts/tok   : {getattr(cfg, 'num_experts_per_tok', getattr(cfg, 'top_k', '?'))}")
    print(f"  Vocab size    : {getattr(cfg, 'vocab_size', '?')}")

    try:
        layers  = get_transformer_layers(model)
        first   = layers[0]
        mlp     = getattr(first, "mlp", None)

        print(f"\n  First-layer MLP ({type(mlp).__name__ if mlp else 'None'}):")

        if mlp is not None and hasattr(mlp, "gate_proj"):
            # ── Dense MLP (Qwen2.5 / LLaMA style) ──────────────────────────
            w = get_mlp_weights(first)
            print(f"    gate_proj.weight : {list(w['gate'].shape)}")
            print(f"    up_proj.weight   : {list(w['up'].shape)}")
            print(f"    down_proj.weight : {list(w['down'].shape)}")
            print(f"    d_model          : {w['d_model']}")
            print(f"    d_ff             : {w['d_ff']}")
            try:
                rmsnorm = get_rmsnorm_before_mlp(first)
                print(f"    RMSNorm before MLP : {type(rmsnorm).__name__} "
                      f"(gamma shape {list(rmsnorm.weight.shape)})")
            except Exception as rms_exc:
                print(f"    RMSNorm before MLP : could not inspect ({rms_exc})")

        elif mlp is not None and hasattr(mlp, "experts"):
            # ── MoE MLP (Qwen3MoE style) ────────────────────────────────────
            experts = list(mlp.experts)
            router  = (getattr(mlp, "gate",          None) or
                       getattr(mlp, "router",         None) or
                       getattr(mlp, "expert_router",  None))
            shared  = getattr(mlp, "shared_expert", None)

            print(f"    MLP type       : {type(mlp).__name__} (MoE)")
            if router is not None:
                r_shape = (list(router.weight.shape)
                           if hasattr(router, "weight") else "no .weight")
                print(f"    Router         : {type(router).__name__}  shape={r_shape}")
            print(f"    Num experts    : {len(experts)}")
            if experts:
                e0 = experts[0]
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    m = getattr(e0, proj, None)
                    if m is not None and hasattr(m, "weight"):
                        print(f"    expert[0].{proj}.weight : {list(m.weight.shape)}")
            if shared is not None:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    m = getattr(shared, proj, None)
                    if m is not None and hasattr(m, "weight"):
                        print(f"    shared.{proj}.weight    : {list(m.weight.shape)}")

        else:
            print(f"    Unknown MLP type — shape inspection skipped")

    except Exception as exc:
        logger.warning("print_model_info: inspection failed: %s", exc)
        print(f"  (shape inspection skipped: {exc})")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: AutoModelForCausalLM) -> Dict[str, int]:
    """Count total and MLP-only parameters (non-embedding, non-LM-head)."""
    total = sum(p.numel() for p in model.parameters())

    mlp_total = 0
    for layer in get_transformer_layers(model):
        mlp = get_mlp_module(layer)
        for p in mlp.parameters():
            mlp_total += p.numel()

    return {"total": total, "mlp": mlp_total}


# ---------------------------------------------------------------------------
# Deep-copy helper (used before each pruning run)
# ---------------------------------------------------------------------------

def clone_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Return a deep copy of the model on the same device."""
    return copy.deepcopy(model)
