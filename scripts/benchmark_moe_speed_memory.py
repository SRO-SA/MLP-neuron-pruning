#!/usr/bin/env python3
"""
benchmark_moe_speed_memory.py
==============================
Standalone timing and memory benchmark for MoE models before and after
structured MLP pruning.

This script does NOT call run_experiment.py.  It loads the model directly,
applies a pruning plan produced by a prior moe_pruning run, and measures:

  - GPU memory (peak allocated, reserved) at load time and after pruning
  - Prefill latency   (first-token time)  over N_WARMUP + N_BENCH iterations
  - Decode latency    (per-token time)
  - Throughput        (tokens / second)

Seven settings are measured:
  1. Baseline (no pruning)
  2–7. Pruning plans: rmsnorm_bound × {wikitext2,c4} × {2%,4%,8%}

Usage:
    python scripts/benchmark_moe_speed_memory.py \
        --model Qwen/Qwen3-30B-A3B \
        --plan  results/pruning_plans/<plan>.json \
        --out   results/speed_memory_results.csv

    # Run all settings (baseline + 6 plans listed in a manifest JSON):
    python scripts/benchmark_moe_speed_memory.py \
        --model  Qwen/Qwen3-30B-A3B \
        --plans  results/speed_memory_plans.json \
        --out    results/speed_memory_results.csv

The script is designed to be called from run_moe_speed_memory_benchmark.sh,
which generates the plan manifest and invokes this script once.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Guard: fail early with a clear message if torch is not installed ──────────
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch is not installed. Run: pip install torch --break-system-packages")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("ERROR: transformers not installed. Run: pip install transformers")

# ── Constants ─────────────────────────────────────────────────────────────────
N_WARMUP     = 2     # warm-up iterations (not included in timing)
N_BENCH      = 5     # measured iterations
PROMPT       = "The quick brown fox jumps over the lazy dog. " * 8
MAX_NEW_TOK  = 32    # tokens to generate per iteration
SEQ_LEN      = 256   # input sequence length for prefill benchmark


# ── Memory helpers ────────────────────────────────────────────────────────────

def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_mib": float("nan"), "peak_reserved_mib": float("nan")}
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {"peak_allocated_mib": alloc, "peak_reserved_mib": reserved}


def current_memory_mib(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"current_allocated_mib": float("nan"), "current_reserved_mib": float("nan")}
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return {"current_allocated_mib": alloc, "current_reserved_mib": reserved}


def clear_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Physical pruning (replicates moe_pruning logic without importing it) ─────

def _apply_pruning_plan(model: Any, plan: Dict, device: torch.device) -> None:
    """
    Apply a pruning plan JSON to the model in-place.

    Plan format (from moe_residual_methods.build_pruning_plan):
      {
        "model_id": "...",
        "layers": [
          {
            "layer_idx": <int>,
            "prune_idx": [<int>, ...],
            "old_intermediate": <int>,
            "new_intermediate": <int>,
          },
          ...
        ]
      }

    For each layer the same channel indices are removed from all experts:
      gate_proj.weight  — remove rows   prune_idx  (shape [d_ff, d_model])
      up_proj.weight    — remove rows   prune_idx
      down_proj.weight  — remove cols   prune_idx  (shape [d_model, d_ff])
    """
    import torch.nn as nn

    layers_cfg = plan.get("layers", [])
    if not layers_cfg:
        print("[bench_py] WARNING: pruning plan has no layers — no pruning applied.")
        return

    # Try to locate the transformer layer list
    # Qwen3-30B-A3B: model.model.layers
    layer_list = None
    for attr in ("model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None:
            for la in ("layers", "h", "blocks"):
                ll = getattr(sub, la, None)
                if ll is not None:
                    layer_list = ll
                    break
        if layer_list is not None:
            break

    if layer_list is None:
        raise RuntimeError("Cannot find transformer layer list in model.")

    keep_mask_cache: Dict[int, torch.Tensor] = {}

    for lcfg in layers_cfg:
        li        = lcfg["layer_idx"]
        prune_idx = lcfg["prune_idx"]
        old_d_ff  = lcfg["old_intermediate"]

        if not prune_idx:
            continue  # nothing to prune at this layer

        # Build keep mask (same for all experts in this layer)
        if li not in keep_mask_cache:
            keep = torch.ones(old_d_ff, dtype=torch.bool)
            keep[prune_idx] = False
            keep_mask_cache[li] = keep
        keep = keep_mask_cache[li]

        layer = layer_list[li]

        # Locate MLP (may be layer.mlp or nested as layer.block.mlp, etc.)
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            print(f"[bench_py] WARNING: layer {li} has no .mlp attribute — skipping.")
            continue

        # Detect layout: single MLP with Linear layers vs. MoE with experts list
        if hasattr(mlp, "experts"):
            experts = list(mlp.experts)
        else:
            experts = [mlp]

        with torch.no_grad():
            for expert in experts:
                gate = getattr(expert, "gate_proj", None)
                up   = getattr(expert, "up_proj",   None)
                down = getattr(expert, "down_proj",  None)

                if gate is None or up is None or down is None:
                    continue  # not a SwiGLU MLP — skip silently

                # gate_proj: [d_ff, d_model] → remove rows at prune_idx
                gate.weight = nn.Parameter(gate.weight[keep, :].contiguous())
                if gate.bias is not None:
                    gate.bias = nn.Parameter(gate.bias[keep].contiguous())

                # up_proj: [d_ff, d_model] → remove rows at prune_idx
                up.weight = nn.Parameter(up.weight[keep, :].contiguous())
                if up.bias is not None:
                    up.bias = nn.Parameter(up.bias[keep].contiguous())

                # down_proj: [d_model, d_ff] → remove cols at prune_idx
                down.weight = nn.Parameter(down.weight[:, keep].contiguous())
                if down.bias is not None:
                    pass  # down bias is [d_model] — unchanged

    print(f"[bench_py] Pruning plan applied: {len(layers_cfg)} layer(s) processed.")


# ── Timing helpers ────────────────────────────────────────────────────────────

def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str = PROMPT,
    max_new_tokens: int = MAX_NEW_TOK,
    n_warmup: int = N_WARMUP,
    n_bench: int = N_BENCH,
) -> Dict[str, float]:
    """
    Measure prefill + decode latency.

    Returns a dict with:
      prefill_ms_mean, prefill_ms_std,
      decode_ms_per_token_mean, decode_ms_per_token_std,
      tokens_per_second_mean
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    prefill_times: List[float] = []
    decode_times:  List[float] = []

    with torch.no_grad():
        for i in range(n_warmup + n_bench):
            _cuda_sync(device)
            t0 = time.perf_counter()

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            _cuda_sync(device)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0

            n_new = out.shape[1] - input_len
            if n_new <= 0:
                n_new = max_new_tokens

            # Rough split: first token ≈ prefill, remaining ≈ decode
            # For a more accurate split we'd need to hook the model;
            # this is a reasonable approximation for benchmarking.
            decode_ms_per_tok = total_ms / max(n_new, 1)
            prefill_ms = total_ms - decode_ms_per_tok * (n_new - 1)

            if i >= n_warmup:
                prefill_times.append(prefill_ms)
                decode_times.append(decode_ms_per_tok)

    import statistics

    def _safe_stdev(lst: list) -> float:
        return statistics.stdev(lst) if len(lst) > 1 else 0.0

    p_mean = statistics.mean(prefill_times)
    d_mean = statistics.mean(decode_times)
    return {
        "prefill_ms_mean":             p_mean,
        "prefill_ms_std":              _safe_stdev(prefill_times),
        "decode_ms_per_token_mean":    d_mean,
        "decode_ms_per_token_std":     _safe_stdev(decode_times),
        "tokens_per_second_mean":      1000.0 / d_mean if d_mean > 0 else float("nan"),
        "n_warmup":                    n_warmup,
        "n_bench":                     n_bench,
        "input_len":                   input_len,
        "max_new_tokens":              max_new_tokens,
    }


# ── Parameter counting ────────────────────────────────────────────────────────

def count_parameters(model: Any) -> Dict[str, int]:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


# ── Single-setting benchmark ──────────────────────────────────────────────────

def run_one_setting(
    model_name:  str,
    plan_path:   Optional[str],
    label:       str,
    device:      torch.device,
    dtype:       torch.dtype,
    n_warmup:    int = N_WARMUP,
    n_bench:     int = N_BENCH,
) -> Dict[str, Any]:
    """Load model, optionally prune, benchmark, return result dict."""
    print(f"\n[bench_py] ── Setting: {label} {'(baseline)' if plan_path is None else ''} ──")

    clear_cache(device)
    reset_peak_memory(device)

    print(f"[bench_py] Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[bench_py] Loading model (dtype={dtype}) ...")
    t_load_0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _cuda_sync(device)
    t_load_1 = time.perf_counter()
    load_sec = t_load_1 - t_load_0

    mem_after_load = {**current_memory_mib(device), **peak_memory_mib(device)}
    params_before  = count_parameters(model)

    plan_applied = False
    plan_n_layers_pruned = 0

    if plan_path is not None:
        print(f"[bench_py] Loading pruning plan: {plan_path}")
        with open(plan_path) as fh:
            plan = json.load(fh)
        reset_peak_memory(device)
        _apply_pruning_plan(model, plan, device)
        plan_applied = True
        plan_n_layers_pruned = sum(
            1 for l in plan.get("layers", []) if l.get("prune_idx")
        )
        _cuda_sync(device)

    mem_after_prune = {**current_memory_mib(device), **peak_memory_mib(device)}
    params_after    = count_parameters(model)

    print(f"[bench_py] Benchmarking generation ({n_warmup} warmup + {n_bench} measured) ...")
    timing = benchmark_generation(
        model, tokenizer, device,
        n_warmup=n_warmup,
        n_bench=n_bench,
    )

    mem_peak = peak_memory_mib(device)

    result = {
        "label":                    label,
        "model_name":               model_name,
        "plan_path":                plan_path or "",
        "plan_applied":             plan_applied,
        "n_layers_pruned":          plan_n_layers_pruned,
        "load_sec":                 round(load_sec, 2),
        "params_before":            params_before["total_params"],
        "params_after":             params_after["total_params"],
        "param_reduction_pct":      round(
            100.0 * (params_before["total_params"] - params_after["total_params"])
            / max(params_before["total_params"], 1), 3
        ),
        # Memory (MiB)
        "mem_load_alloc_mib":       round(mem_after_load.get("current_allocated_mib", float("nan")), 1),
        "mem_load_peak_alloc_mib":  round(mem_after_load.get("peak_allocated_mib",    float("nan")), 1),
        "mem_prune_alloc_mib":      round(mem_after_prune.get("current_allocated_mib", float("nan")), 1),
        "mem_peak_alloc_mib":       round(mem_peak.get("peak_allocated_mib", float("nan")), 1),
        "mem_peak_reserved_mib":    round(mem_peak.get("peak_reserved_mib",  float("nan")), 1),
        # Timing
        "prefill_ms_mean":          round(timing["prefill_ms_mean"], 2),
        "prefill_ms_std":           round(timing["prefill_ms_std"],  2),
        "decode_ms_per_tok_mean":   round(timing["decode_ms_per_token_mean"], 2),
        "decode_ms_per_tok_std":    round(timing["decode_ms_per_token_std"],  2),
        "tokens_per_sec_mean":      round(timing["tokens_per_second_mean"], 2),
        "input_len":                timing["input_len"],
        "max_new_tokens":           timing["max_new_tokens"],
        "n_warmup":                 timing["n_warmup"],
        "n_bench":                  timing["n_bench"],
    }

    # Clean up to free GPU memory for the next setting
    del model
    clear_cache(device)

    return result


# ── CSV output ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "label", "model_name", "plan_path", "plan_applied", "n_layers_pruned",
    "load_sec", "params_before", "params_after", "param_reduction_pct",
    "mem_load_alloc_mib", "mem_load_peak_alloc_mib",
    "mem_prune_alloc_mib", "mem_peak_alloc_mib", "mem_peak_reserved_mib",
    "prefill_ms_mean", "prefill_ms_std",
    "decode_ms_per_tok_mean", "decode_ms_per_tok_std",
    "tokens_per_sec_mean",
    "input_len", "max_new_tokens", "n_warmup", "n_bench",
]


def write_csv(results: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[bench_py] Results written to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",  default="Qwen/Qwen3-30B-A3B",
                    help="HuggingFace model ID (default: Qwen/Qwen3-30B-A3B)")
    ap.add_argument("--plan",   default=None,
                    help="Single pruning plan JSON (run baseline + this plan)")
    ap.add_argument("--plans",  default=None,
                    help="JSON manifest listing multiple plan paths + labels")
    ap.add_argument("--out",    required=True,
                    help="Output CSV path")
    ap.add_argument("--dtype",  default="bfloat16",
                    choices=["float32", "float16", "bfloat16"],
                    help="Model dtype (default: bfloat16)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (default: cuda if available, else cpu)")
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP,
                    help=f"Warm-up iterations (default: {N_WARMUP})")
    ap.add_argument("--n-bench",  type=int, default=N_BENCH,
                    help=f"Measured iterations (default: {N_BENCH})")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip the unpruned baseline measurement")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned settings, do not run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype  = {"float32": torch.float32,
               "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]

    # Build list of (label, plan_path) settings
    settings: List[tuple[str, Optional[str]]] = []

    if not args.no_baseline:
        settings.append(("baseline_no_pruning", None))

    if args.plans is not None:
        # Manifest JSON: {"settings": [{"label": "...", "plan": "..."}, ...]}
        with open(args.plans) as fh:
            manifest = json.load(fh)
        for s in manifest.get("settings", []):
            settings.append((s["label"], s.get("plan")))
    elif args.plan is not None:
        label = Path(args.plan).stem
        settings.append((label, args.plan))

    if not settings:
        sys.exit("ERROR: no settings to benchmark. Provide --plan, --plans, or remove --no-baseline.")

    print(f"[bench_py] Benchmark: {len(settings)} setting(s)  device={device}  dtype={dtype}")
    for i, (label, plan) in enumerate(settings, 1):
        plan_str = plan or "(none — baseline)"
        print(f"  {i:2d}. {label:<40s}  plan={plan_str}")

    if args.dry_run:
        print("\n[bench_py] DRY_RUN: exiting without running benchmarks.")
        return

    results = []
    for label, plan_path in settings:
        try:
            r = run_one_setting(
                model_name=args.model,
                plan_path=plan_path,
                label=label,
                device=device,
                dtype=dtype,
                n_warmup=args.n_warmup,
                n_bench=args.n_bench,
            )
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"[bench_py] ERROR in setting '{label}': {exc}")
            traceback.print_exc()
            results.append({
                "label":      label,
                "model_name": args.model,
                "plan_path":  plan_path or "",
                "error":      str(exc),
            })

    write_csv(results, args.out)

    # Print summary table
    print("\n[bench_py] ── Summary ────────────────────────────────────────────────")
    hdr = f"  {'Label':<40s}  {'Param↓%':>8s}  {'Prefill(ms)':>12s}  {'Tok/s':>8s}  {'PeakMiB':>9s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<40s}  ERROR: {r['error']}")
            continue
        print(
            f"  {r['label']:<40s}"
            f"  {r.get('param_reduction_pct', 0.0):>8.2f}"
            f"  {r.get('prefill_ms_mean', float('nan')):>12.1f}"
            f"  {r.get('tokens_per_sec_mean', float('nan')):>8.1f}"
            f"  {r.get('mem_peak_alloc_mib', float('nan')):>9.0f}"
        )
    print()


if __name__ == "__main__":
    main()
