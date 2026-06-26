#!/usr/bin/env python3
"""
apply_moe_plan_save_checkpoint.py
==================================
Stage 1 helper: apply a pruning plan to the original model and save a
physically-pruned HF checkpoint that can be reloaded with AutoModelForCausalLM.

For residual methods (e.g. residual_nearest_channel_merge_moe), calibration
activations are collected via forward pre-hooks on each expert before physical
pruning.  The residual reconstruction updates down_proj.weight in-place *before*
channels are deleted.

If moe_residual_methods cannot be imported or calibration activations are
insufficient, the script falls back to pure_delete and records
residual_fallback_used=True in pruning_metadata.json.

Writes per checkpoint directory:
  <ckpt_dir>/                  -- HF checkpoint (safetensors + config.json + tokenizer)
  <ckpt_dir>/.bench_meta.json  -- metadata for benchmark_moe_speed_memory.py
  <ckpt_dir>/pruning_metadata.json  -- extended metadata including residual status

Usage:
    python scripts/apply_moe_plan_save_checkpoint.py \\
        --model   Qwen/Qwen3-30B-A3B \\
        --plan    results/pruning_plans/<plan>.json \\
        --method  residual_nearest_channel_merge_moe \\
        --ckpt-dir results/downstream_eval_runs/<id>/pruned_checkpoints/<label> \\
        --dtype   bfloat16 \\
        --calib-n 64
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("ERROR: torch not installed.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except ImportError:
    sys.exit("ERROR: transformers not installed.")


# ---------------------------------------------------------------------------
# Residual methods import (optional; falls back gracefully)
# ---------------------------------------------------------------------------

_RESIDUAL_DISPATCH = None
RESIDUAL_METHODS: Set[str] = set()

try:
    _src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from moe_residual_methods import (          # type: ignore[import]
        apply_residual_dispatch_unpacked,
        RESIDUAL_METHODS as _RM,
    )
    _RESIDUAL_DISPATCH = apply_residual_dispatch_unpacked
    RESIDUAL_METHODS   = _RM
    print("[ckpt] moe_residual_methods imported OK")
except ImportError as _ie:
    print(f"[ckpt] WARNING: moe_residual_methods not available ({_ie}). "
          "Residual methods will fall back to pure_delete.")


# ---------------------------------------------------------------------------
# Fallback calibration corpus
# ---------------------------------------------------------------------------

_FALLBACK_CORPUS: List[str] = [
    "The attention mechanism in transformers allows each token to attend to all other tokens in the sequence.",
    "Language models learn representations by predicting the probability of the next token given all previous context.",
    "Mixture-of-experts models route each token to a subset of specialized expert feed-forward networks.",
    "Structured pruning removes entire neurons or channels from neural networks to reduce memory and compute.",
    "The SwiGLU activation function computes the element-wise product of a gate and a value projection.",
    "Gradient-free pruning methods rank neurons by weight norms or activation statistics collected offline.",
    "In transformer models, each feed-forward block consists of two linear projections and an activation function.",
    "Expert routing in sparse MoE models is controlled by a learned gating network applied to token embeddings.",
    "The key insight of residual reconstruction is that pruned channels can be approximated by kept channels.",
    "Calibration-based pruning methods collect statistics on real data to identify less important neurons.",
    "Low-rank approximations of weight matrices provide a compact representation with controllable accuracy loss.",
    "The hidden state dimension in Qwen3-30B-A3B is 2048, with MoE intermediate size of 768 per expert.",
    "Nearest-channel-merge reconstruction finds the kept channel most similar to each pruned channel.",
    "Weight norms provide a simple but effective proxy for the importance of individual neurons.",
    "The output of an MoE layer is the weighted sum of the outputs of all selected experts.",
    "Perplexity measures the cross-entropy loss exponentiated, providing a measure of language model quality.",
    "Fine-tuning after pruning can recover accuracy lost during the pruning process.",
    "Quantization reduces the bit-width of model weights and activations to decrease memory footprint.",
    "The Llama and Qwen model families both use the transformer decoder-only architecture.",
    "Structured sparsity removes entire rows or columns, enabling efficient dense matrix operations.",
    "Knowledge distillation trains a small student model to mimic the outputs of a large teacher model.",
    "Post-training quantization applies quantization to a pre-trained model without additional training.",
    "The feedforward network in a transformer applies two linear transformations with a nonlinearity between them.",
    "In MoE models, the number of active parameters per forward pass is much smaller than the total parameter count.",
    "Channel pruning selects a subset of feature map channels to retain based on an importance criterion.",
    "The down-projection matrix in an MoE expert maps from the intermediate dimension back to the model dimension.",
    "Calibration data should represent the target distribution to ensure accurate importance estimates.",
    "The gating network in a MoE transformer produces routing probabilities for each token and expert pair.",
    "Memory bandwidth is often the bottleneck for LLM inference on modern GPU hardware.",
    "Tensor parallelism distributes model layers across multiple GPUs to enable larger model sizes.",
    "The tokenizer converts raw text into a sequence of integer token IDs for input to the model.",
    "Layer normalization stabilizes training by normalizing activations to have zero mean and unit variance.",
    "Rotary position embeddings encode relative position information directly into the attention computation.",
    "The vocabulary size of modern language models is typically between 32,000 and 150,000 tokens.",
    "Flash attention computes the attention matrix in a memory-efficient, tiled fashion on GPU hardware.",
    "Sparse attention patterns restrict each token to attend only to a subset of positions in the sequence.",
    "The residual stream accumulates information from each transformer layer as it flows through the network.",
    "Multi-query attention uses a single key and value head shared across all query heads.",
    "Temperature scaling at inference time controls the sharpness of the next-token probability distribution.",
    "Nucleus sampling restricts the sampling distribution to the smallest set of tokens whose probability sums to p.",
]


def _load_calib_texts(n_texts: int = 64) -> List[str]:
    """Load calibration texts from WikiText-2 or fall back to built-in corpus."""
    try:
        from datasets import load_dataset  # type: ignore[import]
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          trust_remote_code=True)
        texts = [
            row["text"].strip() for row in ds
            if len(row["text"].strip()) > 80
        ][:n_texts]
        if len(texts) >= max(n_texts // 2, 8):
            print(f"[ckpt] Calibration texts: {len(texts)} from WikiText-2")
            return texts
    except Exception as exc:
        print(f"[ckpt] WikiText-2 unavailable ({exc}); using built-in corpus")
    texts = (_FALLBACK_CORPUS * ((n_texts // len(_FALLBACK_CORPUS)) + 2))[:n_texts]
    print(f"[ckpt] Calibration texts: {len(texts)} from fallback corpus")
    return texts


# ---------------------------------------------------------------------------
# Core helpers (mirror moe_pruning.py logic -- no import to avoid side-effects)
# ---------------------------------------------------------------------------

def _find_layer_list(model: Any) -> Optional[Any]:
    for attr in ("model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        for la in ("layers", "h", "blocks"):
            ll = getattr(sub, la, None)
            if ll is not None:
                return ll
    return None


def apply_pruning_plan(model: Any, plan: Dict) -> int:
    """
    Physical structured pruning: remove rows from gate_proj/up_proj and
    columns from down_proj for each expert.

    For MoE layer i, neuron k corresponds to:
      gate_proj.weight[k, :]   (row k)  — shape [d_ff, d_model]
      up_proj.weight[k, :]     (row k)  — shape [d_ff, d_model]
      down_proj.weight[:, k]   (column k) — shape [d_model, d_ff]

    Returns number of layers modified.
    """
    layer_list = _find_layer_list(model)
    if layer_list is None:
        raise RuntimeError("Cannot find transformer layer list in model.")

    n_modified = 0
    for lcfg in plan.get("layers", []):
        li        = lcfg["layer_idx"]
        prune_idx = lcfg.get("prune_idx", [])
        old_d_ff  = lcfg.get("old_intermediate", 0)
        if not prune_idx or old_d_ff == 0:
            continue

        keep = torch.ones(old_d_ff, dtype=torch.bool)
        for idx in prune_idx:
            keep[idx] = False

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
                if gate.bias is not None:
                    gate.bias = nn.Parameter(gate.bias[keep].contiguous())
                up.weight = nn.Parameter(up.weight[keep, :].contiguous())
                if up.bias is not None:
                    up.bias = nn.Parameter(up.bias[keep].contiguous())
                down.weight = nn.Parameter(down.weight[:, keep].contiguous())
        n_modified += 1

    return n_modified


def detect_uniform_moe_dim(model: Any) -> Optional[int]:
    """Detect the new (pruned) MoE intermediate size from down_proj.weight.shape[1]."""
    layer_list = _find_layer_list(model)
    if layer_list is None:
        return None

    sizes: Set[int] = set()
    for layer in layer_list:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = list(mlp.experts) if hasattr(mlp, "experts") else [mlp]
        for expert in experts:
            down = getattr(expert, "down_proj", None)
            if down is not None:
                sizes.add(down.weight.shape[1])

    if not sizes:
        return None
    if len(sizes) > 1:
        raise RuntimeError(
            f"Non-uniform expert sizes after pruning: {sorted(sizes)}. "
            "HF AutoModel requires a single moe_intermediate_size. "
            "Ensure the pruning plan uses moe_budget_mode=uniform."
        )
    return list(sizes)[0]


def sample_expert_shapes(model: Any) -> Dict[str, Any]:
    """Return gate_proj/up_proj/down_proj weight shapes from the first expert found."""
    layer_list = _find_layer_list(model)
    result: Dict[str, Any] = {}
    if layer_list is None:
        return result
    for layer in layer_list:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = list(mlp.experts) if hasattr(mlp, "experts") else [mlp]
        if not experts:
            continue
        exp0 = experts[0]
        for proj, key in [("gate_proj", "sample_gate_proj_shape"),
                           ("up_proj",   "sample_up_proj_shape"),
                           ("down_proj", "sample_down_proj_shape")]:
            m = getattr(exp0, proj, None)
            if m is not None:
                result[key] = list(m.weight.shape)
        break
    return result


# ---------------------------------------------------------------------------
# Calibration activation collection
# ---------------------------------------------------------------------------

def collect_expert_activations_by_layer(
    model:        Any,
    tokenizer:    Any,
    calib_texts:  List[str],
    max_seq_len:  int = 512,
    plan_layers:  Optional[Set[int]] = None,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Run forward passes on calib_texts and collect per-expert input activations
    via forward pre-hooks.

    plan_layers: if given, only register hooks on those layer indices to save memory.

    Returns {layer_idx: {expert_idx: Tensor[N_routed, d_model]}}.
    """
    layer_list = _find_layer_list(model)
    if layer_list is None:
        print("[ckpt] WARNING: cannot find layer list; skipping calibration")
        return {}

    # Find a device for inputs (works for both single-GPU and multi-GPU device_map)
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    raw_acts: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
    hooks: List[Any] = []

    for li, layer in enumerate(layer_list):
        if plan_layers is not None and li not in plan_layers:
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = list(mlp.experts) if hasattr(mlp, "experts") else [mlp]
        for ei, expert in enumerate(experts):
            def _make_hook(li_: int = li, ei_: int = ei):
                def _hook(mod: Any, args: tuple) -> None:
                    if args and isinstance(args[0], torch.Tensor):
                        # args[0]: [n_routed_tokens, d_model] on expert's device
                        raw_acts[(li_, ei_)].append(
                            args[0].detach().float().cpu()
                        )
                return _hook
            h = expert.register_forward_pre_hook(_make_hook())
            hooks.append(h)

    n_hooks = len(hooks)
    print(f"[ckpt] Registered {n_hooks} forward pre-hooks on expert modules")

    try:
        model.eval()
        with torch.no_grad():
            for ti, text in enumerate(calib_texts):
                if not text.strip():
                    continue
                try:
                    enc = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_seq_len,
                    )
                    enc = {k: v.to(input_device) for k, v in enc.items()}
                    model(**enc)
                except Exception as exc:
                    print(f"[ckpt] WARNING: calibration pass {ti} failed: {exc}")
                    continue
        print(f"[ckpt] Calibration complete: {len(calib_texts)} texts processed")
    finally:
        for h in hooks:
            h.remove()
        print(f"[ckpt] Removed {len(hooks)} hooks")

    # Concatenate per-(layer, expert) tensors
    result: Dict[int, Dict[int, torch.Tensor]] = {}
    for (li, ei), tensors in raw_acts.items():
        if not tensors:
            continue
        if li not in result:
            result[li] = {}
        result[li][ei] = torch.cat(tensors, dim=0)

    total_tokens = sum(
        t.shape[0] for ldict in result.values() for t in ldict.values()
    )
    print(f"[ckpt] Calibration activations: {len(result)} layers, "
          f"{total_tokens} total routed tokens")
    return result


# ---------------------------------------------------------------------------
# Weight hash (changes when residual updates down_proj before physical prune)
# ---------------------------------------------------------------------------

def _compute_weight_hash(model: Any) -> str:
    """
    MD5 of the first 1024 bytes of the first two down_proj weight tensors.
    This hash changes when residual reconstruction updates down_proj in-place
    before physical pruning — use it to verify residual was actually applied.
    """
    layer_list = _find_layer_list(model)
    if layer_list is None:
        return "nolayers"
    m = hashlib.md5()
    n_found = 0
    for layer in layer_list:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = list(mlp.experts) if hasattr(mlp, "experts") else [mlp]
        for expert in experts:
            down = getattr(expert, "down_proj", None)
            if down is None or not hasattr(down, "weight"):
                continue
            data = down.weight.detach().float().cpu().numpy().tobytes()[:1024]
            m.update(data)
            n_found += 1
            if n_found >= 2:
                break
        if n_found >= 2:
            break
    return m.hexdigest()[:16] if n_found > 0 else "nohash"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply a MoE pruning plan to a model and save a HF checkpoint."
    )
    ap.add_argument("--model",         required=True,
                    help="HuggingFace model name or path")
    ap.add_argument("--plan",          required=True,
                    help="Path to pruning plan JSON")
    ap.add_argument("--ckpt-dir",      required=True,
                    help="Directory to save the pruned checkpoint")
    ap.add_argument("--method",        default="pure_delete",
                    help="Pruning method label (e.g. pure_delete, residual_nearest_channel_merge_moe)")
    ap.add_argument("--dtype",         default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--label",         default="",
                    help="Human-readable label (used only for logging)")
    ap.add_argument("--calib-n",       type=int, default=64,
                    help="Number of calibration texts for residual methods")
    ap.add_argument("--calib-seq-len", type=int, default=512,
                    help="Max sequence length during calibration")
    args = ap.parse_args()

    method    = args.method
    ckpt_dir  = args.ckpt_dir
    label_str = args.label or method

    # ── Fast-path: skip if already saved ─────────────────────────────────────
    bench_meta_path = os.path.join(ckpt_dir, ".bench_meta.json")
    pm_path         = os.path.join(ckpt_dir, "pruning_metadata.json")
    if os.path.isfile(bench_meta_path) and os.path.isfile(pm_path):
        try:
            with open(pm_path) as f:
                pm = json.load(f)
            if pm.get("save_complete"):
                print(f"[ckpt] Fast-path: checkpoint already complete at {ckpt_dir}")
                return
        except Exception:
            pass
        print(f"[ckpt] save_complete not confirmed; re-applying.")

    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[ckpt] ===================================================")
    print(f"[ckpt] label   : {label_str}")
    print(f"[ckpt] method  : {method}")
    print(f"[ckpt] plan    : {args.plan}")
    print(f"[ckpt] ckpt_dir: {ckpt_dir}")
    print(f"[ckpt] dtype   : {args.dtype}")
    print(f"[ckpt] ===================================================")

    # ── Load plan ─────────────────────────────────────────────────────────────
    if not os.path.isfile(args.plan):
        print(f"[ckpt] ERROR: plan not found: {args.plan}")
        sys.exit(1)
    with open(args.plan) as f:
        plan_data = json.load(f)

    plan_layers_set: Set[int] = {
        lc["layer_idx"]
        for lc in plan_data.get("layers", [])
        if lc.get("prune_idx")
    }
    print(f"[ckpt] Plan: {len(plan_data.get('layers', []))} layer configs, "
          f"{len(plan_layers_set)} layers with actual pruning")

    orig_moe_dim_from_plan: Optional[int] = None
    for lc in plan_data.get("layers", []):
        v = lc.get("old_intermediate")
        if v:
            orig_moe_dim_from_plan = int(v)
            break

    # ── Load model ────────────────────────────────────────────────────────────
    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"[ckpt] Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print("[ckpt] Model loaded.")

    # ── Residual reconstruction (if requested) ────────────────────────────────
    residual_applied     = False
    residual_fallback    = False
    actual_method        = method
    residual_n_layers    = 0
    residual_stats_summary: Dict[str, Any] = {}

    if method in RESIDUAL_METHODS:
        if _RESIDUAL_DISPATCH is None:
            print(f"[ckpt] WARNING: moe_residual_methods not importable. "
                  f"Falling back to pure_delete for method={method}")
            residual_fallback = True
            actual_method     = "pure_delete"
        else:
            print(f"[ckpt] Residual method: {method}")
            print(f"[ckpt] Collecting calibration activations (n={args.calib_n}) ...")
            calib_texts = _load_calib_texts(n_texts=args.calib_n)
            try:
                layer_acts = collect_expert_activations_by_layer(
                    model, tokenizer, calib_texts,
                    max_seq_len=args.calib_seq_len,
                    plan_layers=plan_layers_set,
                )
                if not layer_acts:
                    print("[ckpt] WARNING: no activations collected; falling back to pure_delete")
                    residual_fallback = True
                    actual_method     = "pure_delete"
                else:
                    print(f"[ckpt] Applying residual method to {len(layer_acts)} layers ...")
                    layer_list = _find_layer_list(model)
                    if layer_list is None:
                        print("[ckpt] WARNING: cannot find layer list; residual skipped")
                        residual_fallback = True
                        actual_method     = "pure_delete"
                    else:
                        for lc in plan_data.get("layers", []):
                            li        = lc["layer_idx"]
                            prune_idx = lc.get("prune_idx", [])
                            if not prune_idx or li not in layer_acts:
                                continue
                            layer   = layer_list[li]
                            mlp     = getattr(layer, "mlp", None)
                            if mlp is None:
                                continue
                            experts        = list(mlp.experts) if hasattr(mlp, "experts") else [mlp]
                            acts_by_expert = layer_acts[li]
                            for ei, expert in enumerate(experts):
                                if ei not in acts_by_expert:
                                    continue
                                acts = acts_by_expert[ei]
                                try:
                                    stats = _RESIDUAL_DISPATCH(
                                        method, expert, prune_idx, acts,
                                    )
                                    residual_n_layers += 1
                                    if stats:
                                        residual_stats_summary[f"L{li}_E{ei}"] = stats
                                except Exception as exc:
                                    print(f"[ckpt] WARNING: residual L{li} E{ei} failed: {exc}")
                        residual_applied = True
                        actual_method    = method
                        print(f"[ckpt] Residual applied to {residual_n_layers} (layer, expert) pairs")
            except Exception as exc:
                print(f"[ckpt] WARNING: residual calibration failed: {exc}; falling back to pure_delete")
                residual_fallback = True
                actual_method     = "pure_delete"

    # Hash after residual but before physical pruning
    hash_post_residual = _compute_weight_hash(model)
    print(f"[ckpt] weight_hash (post-residual, pre-prune): {hash_post_residual}")

    # ── Physical pruning ──────────────────────────────────────────────────────
    print("[ckpt] Applying physical pruning ...")
    n_mod = apply_pruning_plan(model, plan_data)
    print(f"[ckpt] Physical pruning complete: {n_mod} layers modified")

    # Hash after physical pruning — this is the final saved hash
    hash_final = _compute_weight_hash(model)
    print(f"[ckpt] weight_hash (post-prune, final): {hash_final}")

    # ── Detect new uniform MoE dim + update config ────────────────────────────
    try:
        new_moe_size: Optional[int] = detect_uniform_moe_dim(model)
    except RuntimeError as exc:
        print(f"[ckpt] ERROR: {exc}")
        sys.exit(1)

    if new_moe_size is not None:
        old_moe = getattr(model.config, "moe_intermediate_size", None)
        print(f"[ckpt] moe_intermediate_size: {old_moe} → {new_moe_size}")
        model.config.moe_intermediate_size = new_moe_size
        shapes = sample_expert_shapes(model)
        for k, v in shapes.items():
            print(f"[ckpt] {k}: {v}")
    else:
        print("[ckpt] WARNING: no expert down_proj found; skipping config update")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    print(f"[ckpt] Saving checkpoint to {ckpt_dir} ...")
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    print("[ckpt] Save complete.")

    # Count shards
    shard_files = (
        glob.glob(os.path.join(ckpt_dir, "model*.safetensors")) +
        glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin"))
    )
    print(f"[ckpt] Checkpoint shards: {len(shard_files)}")

    # AutoConfig reload verification
    loaded_cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
    loaded_moe = getattr(loaded_cfg, "moe_intermediate_size", None)
    if new_moe_size is not None and loaded_moe != new_moe_size:
        print(f"[ckpt] ERROR: AutoConfig mismatch: loaded={loaded_moe}, expected={new_moe_size}")
        sys.exit(1)
    print(f"[ckpt] AutoConfig verify OK: moe_intermediate_size={loaded_moe}")

    # ── Write metadata ────────────────────────────────────────────────────────
    # .bench_meta.json (backward compat for benchmark_moe_speed_memory.py)
    bench_meta = {
        "model":                        args.model,
        "plan":                         args.plan,
        "method":                       actual_method,
        "label":                        label_str,
        "dtype":                        args.dtype,
        "saved_moe_intermediate_size":  new_moe_size,
        "original_moe_intermediate_size": orig_moe_dim_from_plan,
    }
    with open(bench_meta_path, "w") as f:
        json.dump(bench_meta, f, indent=2)

    # pruning_metadata.json (new — includes residual status and hash for verification)
    pruning_metadata: Dict[str, Any] = {
        "requested_method":                     method,
        "actual_method":                        actual_method,
        "residual_applied":                     residual_applied,
        "residual_fallback_used":               residual_fallback,
        "weight_hash":                          hash_final,
        "weight_hash_post_residual_pre_prune":  hash_post_residual,
        "residual_n_layers_applied":            residual_n_layers,
        "residual_stats_summary":               residual_stats_summary,
        "calib_n":                              args.calib_n if method in RESIDUAL_METHODS else 0,
        "saved_moe_intermediate_size":          new_moe_size,
        "original_moe_intermediate_size":       orig_moe_dim_from_plan,
        "target_pct":                           plan_data.get("target_pct"),
        "save_complete":                        True,
    }
    with open(pm_path, "w") as f:
        json.dump(pruning_metadata, f, indent=2)

    print(f"[ckpt] pruning_metadata.json written: {pm_path}")
    print(f"[ckpt] requested_method       = {method}")
    print(f"[ckpt] actual_method          = {actual_method}")
    print(f"[ckpt] residual_applied       = {residual_applied}")
    print(f"[ckpt] residual_fallback_used = {residual_fallback}")
    print(f"[ckpt] weight_hash            = {hash_final}")
    print(f"[ckpt] Done.")


if __name__ == "__main__":
    main()
