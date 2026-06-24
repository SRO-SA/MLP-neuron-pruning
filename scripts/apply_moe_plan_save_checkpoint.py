#!/usr/bin/env python3
"""
apply_moe_plan_save_checkpoint.py
==================================
Stage 1 helper: apply a pruning plan to the original model and save a
physically-pruned HF checkpoint that can be reloaded with AutoModelForCausalLM.

Writes:
  <ckpt_dir>/                  -- HF checkpoint (safetensors + config.json + tokenizer)
  <ckpt_dir>/.bench_meta.json  -- metadata consumed by benchmark_moe_speed_memory.py

If <ckpt_dir>/.bench_meta.json already has save_complete=true, exits immediately
without reloading the model.

Usage:
    python scripts/apply_moe_plan_save_checkpoint.py \\
        --model   Qwen/Qwen3-30B-A3B \\
        --plan    results/pruning_plans/<plan>.json \\
        --method  pure_delete \\
        --ckpt-dir results/speed_memory_runs/<id>/pruned_checkpoints/<label> \\
        --dtype   bfloat16

Expected output (Qwen3-30B-A3B, 4% target):
  original moe_intermediate_size : 768
  saved moe_intermediate_size    : 736
  sample gate_proj shape         : [736, 2048]
  sample up_proj shape           : [736, 2048]
  sample down_proj shape         : [2048, 736]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Set

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
# Helpers (mirror moe_pruning.py logic -- no import to avoid side effects)
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
      gate_proj.weight[k, :]   (row k)
      up_proj.weight[k, :]     (row k)
      down_proj.weight[:, k]   (column k)

    Removing neuron k means:
      gate_proj.weight shape: [d_ff, d_model] -> keep rows where keep[k]=True
      up_proj.weight shape:   [d_ff, d_model] -> keep rows
      down_proj.weight shape: [d_model, d_ff] -> keep columns

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
    """
    Detect the new (pruned) MoE intermediate size by reading down_proj.weight.shape[1].
    Fails loudly if expert sizes are non-uniform (HF AutoModel cannot represent that).
    """
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
                # down_proj.weight: [d_model, d_ff_new]
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
        gate = getattr(exp0, "gate_proj", None)
        up   = getattr(exp0, "up_proj",   None)
        down = getattr(exp0, "down_proj",  None)
        if gate is not None:
            result["sample_gate_proj_shape"] = list(gate.weight.shape)
        if up is not None:
            result["sample_up_proj_shape"] = list(up.weight.shape)
        if down is not None:
            result["sample_down_proj_shape"] = list(down.weight.shape)
        break
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",     required=True,
                    help="HF model ID or local path (original, unpruned)")
    ap.add_argument("--plan",      required=True,
                    help="Pruning plan JSON path")
    ap.add_argument("--method",    default="pure_delete",
                    help="Method name (stored in metadata only)")
    ap.add_argument("--ckpt-dir",  required=True,
                    help="Output directory for the saved pruned checkpoint")
    ap.add_argument("--dtype",     default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--label",     default="",
                    help="Setting label (stored in metadata only)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ckpt_dir  = args.ckpt_dir
    meta_path = os.path.join(ckpt_dir, ".bench_meta.json")

    # Fast-path: skip if checkpoint already complete
    if os.path.isdir(ckpt_dir) and os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("save_complete"):
                saved_dim = meta.get("saved_moe_intermediate_size")
                print(f"[ckpt] Checkpoint already complete: {ckpt_dir}")
                print(f"[ckpt]   saved_moe_intermediate_size = {saved_dim}")
                return
        except Exception:
            pass  # corrupt meta -- recreate

    print("[ckpt] apply_moe_plan_save_checkpoint.py")
    print(f"[ckpt]   model    : {args.model}")
    print(f"[ckpt]   plan     : {args.plan}")
    print(f"[ckpt]   method   : {args.method}")
    print(f"[ckpt]   ckpt_dir : {ckpt_dir}")
    print(f"[ckpt]   dtype    : {args.dtype}")

    if not os.path.isfile(args.plan):
        sys.exit(f"ERROR: plan not found: {args.plan}")

    with open(args.plan) as f:
        plan_data = json.load(f)

    n_plan_layers = len(plan_data.get("layers", []))
    total  = sum(lc.get("old_intermediate", 0) for lc in plan_data.get("layers", []))
    pruned = sum(len(lc.get("prune_idx", []))  for lc in plan_data.get("layers", []))
    actual_pct = round(100.0 * pruned / max(total, 1), 3) if total else 0.0
    print(f"[ckpt]   plan     : {n_plan_layers} layers, {pruned}/{total} channels ({actual_pct:.3f}%)")

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # ── Load original model ──────────────────────────────────────────────────
    print("[ckpt] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[ckpt] Loading original model (dtype={args.dtype}, device_map=auto) ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    load_sec = time.perf_counter() - t0
    print(f"[ckpt] Loaded in {load_sec:.1f}s")

    original_moe_dim = getattr(model.config, "moe_intermediate_size", None)
    print(f"[ckpt]   original moe_intermediate_size : {original_moe_dim}")

    # ── Apply pruning ────────────────────────────────────────────────────────
    print("[ckpt] Applying pruning plan ...")
    n_modified = apply_pruning_plan(model, plan_data)
    print(f"[ckpt]   {n_modified} layers modified.")

    # Detect new uniform MoE dim
    try:
        new_moe_dim = detect_uniform_moe_dim(model)
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")

    if new_moe_dim is None:
        print("[ckpt] WARNING: no MoE experts found -- moe_intermediate_size not updated")
        new_moe_dim = original_moe_dim
    else:
        model.config.moe_intermediate_size = new_moe_dim
        print(f"[ckpt]   saved moe_intermediate_size   : {new_moe_dim}")

    # Print sample expert shapes for verification
    shapes = sample_expert_shapes(model)
    for key in ("sample_gate_proj_shape", "sample_up_proj_shape", "sample_down_proj_shape"):
        if key in shapes:
            short_key = key.replace("sample_", "").replace("_shape", "")
            print(f"[ckpt]   {short_key} shape : {shapes[key]}")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[ckpt] Saving to {ckpt_dir} ...")
    t1 = time.perf_counter()
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    save_sec = time.perf_counter() - t1
    print(f"[ckpt] Saved in {save_sec:.1f}s")

    # Count shards and total size
    shard_files = (
        glob.glob(os.path.join(ckpt_dir, "model*.safetensors")) +
        glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin"))
    )
    n_shards = len(shard_files)
    ckpt_size_gib = (
        sum(os.path.getsize(f) for f in shard_files) / (1024 ** 3)
        if shard_files else 0.0
    )
    print(f"[ckpt]   num checkpoint shards : {n_shards}  ({ckpt_size_gib:.2f} GiB)")

    # ── Verify saved config.json ─────────────────────────────────────────────
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        saved_cfg = json.load(f)
    saved_dim_in_file = saved_cfg.get("moe_intermediate_size", "NOT_FOUND")
    print(f"[ckpt]   config.json moe_intermediate_size : {saved_dim_in_file}")
    if new_moe_dim is not None and saved_dim_in_file != new_moe_dim:
        sys.exit(f"ERROR: config sanity failed: saved={saved_dim_in_file}, expected={new_moe_dim}")

    # AutoConfig reload verification
    print("[ckpt] AutoConfig.from_pretrained verification ...")
    loaded_cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
    loaded_dim = getattr(loaded_cfg, "moe_intermediate_size", None)
    print(f"[ckpt]   AutoConfig moe_intermediate_size : {loaded_dim}")
    if new_moe_dim is not None and loaded_dim != new_moe_dim:
        sys.exit(f"ERROR: AutoConfig mismatch: loaded={loaded_dim}, expected={new_moe_dim}")
    print("[ckpt] AutoConfig reload: OK")

    # ── Write metadata ───────────────────────────────────────────────────────
    meta: Dict[str, Any] = {
        "label":                           args.label,
        "base_model_name":                 args.model,
        "plan_path":                       args.plan,
        "method":                          args.method,
        "actual_pct":                      actual_pct,
        "original_moe_intermediate_size":  original_moe_dim,
        "saved_moe_intermediate_size":     new_moe_dim,
        "n_shards":                        n_shards,
        "checkpoint_size_gib":             round(ckpt_size_gib, 3),
        "save_complete":                   True,
    }
    meta.update(shapes)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ckpt] Metadata: {meta_path}")
    print(f"[ckpt] Checkpoint complete: {ckpt_dir}")


if __name__ == "__main__":
    main()
