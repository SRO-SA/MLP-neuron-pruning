#!/usr/bin/env python3
"""
benchmark_moe_speed_memory.py
==============================
Single-setting subprocess for MoE speed/memory benchmarking.

Called once per setting by run_moe_speed_memory_benchmark.sh.
Each invocation loads exactly one model, optionally applies one pruning plan,
measures latency and GPU memory, writes ONE JSON result file, then exits.

Running one process per setting ensures GPU memory stats are fully isolated —
no carryover from prior settings, no double-counting.

Usage:
    python scripts/benchmark_moe_speed_memory.py \
        --model  Qwen/Qwen3-30B-A3B \
        --plan   results/pruning_plans/my_plan.json \
        --label  "residual_nearest__rmsnorm_bound__wikitext2__target4pct__actual4.2pct" \
        --method residual_nearest_channel_merge_moe \
        --selector  rmsnorm_bound \
        --dataset   wikitext2 \
        --target-pct 4.0 \
        --actual-pct 4.167 \
        --out-json  results/speed_memory_runs/<id>/jsons/<label>.json

    # Baseline (no pruning plan):
        omit --plan, or pass --plan NONE

    # Dry-run (print settings, do not load model):
        --dry-run
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── Guard: require torch ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("ERROR: torch not installed. Run: pip install torch --break-system-packages")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("ERROR: transformers not installed. Run: pip install transformers")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
DEFAULT_MAX_NEW_TOKENS = 32


# ── Multi-GPU memory helpers ──────────────────────────────────────────────────

def _n_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def reset_all_peak_memory() -> None:
    """Reset peak memory stats on every visible GPU."""
    for i in range(_n_gpus()):
        torch.cuda.reset_peak_memory_stats(i)


def current_total_allocated_mib() -> float:
    """Sum of currently allocated memory across all GPUs (MiB)."""
    n = _n_gpus()
    if n == 0:
        return float("nan")
    return sum(torch.cuda.memory_allocated(i) for i in range(n)) / (1024 ** 2)


def peak_memory_snapshot() -> Dict[str, float]:
    """
    Return peak allocated + reserved per GPU and total.
    Keys: peak_allocated_mib_gpu{i}, peak_reserved_mib_gpu{i},
          peak_allocated_mib_total, peak_reserved_mib_total
    Always includes gpu0 and gpu1 keys (nan if GPU not present).
    """
    n = _n_gpus()
    result: Dict[str, float] = {
        "peak_allocated_mib_gpu0": float("nan"),
        "peak_allocated_mib_gpu1": float("nan"),
        "peak_reserved_mib_gpu0":  float("nan"),
        "peak_reserved_mib_gpu1":  float("nan"),
        "peak_allocated_mib_total": float("nan"),
        "peak_reserved_mib_total":  float("nan"),
    }
    if n == 0:
        return result
    tot_alloc = 0.0
    tot_res   = 0.0
    for i in range(n):
        alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        res   = torch.cuda.max_memory_reserved(i)  / (1024 ** 2)
        result[f"peak_allocated_mib_gpu{i}"] = round(alloc, 1)
        result[f"peak_reserved_mib_gpu{i}"]  = round(res,   1)
        tot_alloc += alloc
        tot_res   += res
    result["peak_allocated_mib_total"] = round(tot_alloc, 1)
    result["peak_reserved_mib_total"]  = round(tot_res,   1)
    return result


def cuda_sync_all() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gc_clear() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Pruning plan helpers ──────────────────────────────────────────────────────

def _find_layer_list(model: Any) -> Any:
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
    columns from down_proj for each expert.  Mirrors moe_pruning.py logic.
    Returns count of layers that had non-empty prune_idx.
    """
    layer_list = _find_layer_list(model)
    if layer_list is None:
        raise RuntimeError("Cannot find transformer layer list in model.")

    n_pruned = 0
    for lcfg in plan.get("layers", []):
        li        = lcfg["layer_idx"]
        prune_idx = lcfg.get("prune_idx", [])
        old_d_ff  = lcfg.get("old_intermediate", 0)
        if not prune_idx or old_d_ff == 0:
            continue

        keep = torch.ones(old_d_ff, dtype=torch.bool)
        keep[prune_idx] = False

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
                # gate_proj: [d_ff, d_model] → keep rows
                gate.weight = nn.Parameter(gate.weight[keep, :].contiguous())
                if gate.bias is not None:
                    gate.bias = nn.Parameter(gate.bias[keep].contiguous())
                # up_proj: [d_ff, d_model] → keep rows
                up.weight   = nn.Parameter(up.weight[keep, :].contiguous())
                if up.bias is not None:
                    up.bias = nn.Parameter(up.bias[keep].contiguous())
                # down_proj: [d_model, d_ff] → keep columns
                down.weight = nn.Parameter(down.weight[:, keep].contiguous())
        n_pruned += 1

    return n_pruned


def compute_actual_pct_from_plan(plan: Dict) -> float:
    """Compute actual channel-reduction % from plan JSON layers list."""
    total   = sum(lcfg.get("old_intermediate", 0) for lcfg in plan.get("layers", []))
    pruned  = sum(len(lcfg.get("prune_idx", [])) for lcfg in plan.get("layers", []))
    if total == 0:
        return 0.0
    return round(100.0 * pruned / total, 3)


# ── Timing benchmark ──────────────────────────────────────────────────────────

def benchmark_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    n_warmup: int,
    n_bench: int,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Measure prefill, decode, and end-to-end latency.
    Returns dict with all timing and shape fields.
    """
    import statistics

    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    prompts = [prompt] * batch_size
    inputs  = tokenizer(prompts, return_tensors="pt", padding=True).to(dev)
    prompt_len = inputs["input_ids"].shape[1]

    e2e_ms_list:    List[float] = []
    decode_ms_list: List[float] = []

    with torch.no_grad():
        for i in range(n_warmup + n_bench):
            sync()
            t0  = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            sync()
            t1     = time.perf_counter()
            e2e_ms = (t1 - t0) * 1000.0
            n_new  = max(out.shape[1] - prompt_len, 1)

            if i >= n_warmup:
                e2e_ms_list.append(e2e_ms)
                decode_ms_list.append(e2e_ms / n_new)  # ms per token

    def _m(lst: list) -> float:
        return statistics.mean(lst) if lst else float("nan")

    e2e_mean    = _m(e2e_ms_list)
    decode_mean = _m(decode_ms_list)
    # prefill ≈ total - (decode_per_tok × (n_new-1))
    prefill_mean = e2e_mean - decode_mean * (max_new_tokens - 1)

    return {
        "prefill_latency_ms_mean":    round(prefill_mean, 2),
        "decode_latency_ms_mean":     round(decode_mean,  4),
        "end_to_end_latency_ms_mean": round(e2e_mean,     2),
        "tokens_per_sec_mean":        round(1000.0 * batch_size / max(decode_mean, 1e-9), 2),
        "prompt_len":                 prompt_len,
        "generated_tokens":           int(out.shape[1] - prompt_len),
        "batch_size":                 batch_size,
        "n_warmup":                   n_warmup,
        "n_bench":                    n_bench,
    }


def count_params(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",           required=True)
    ap.add_argument("--plan",            default=None,
                    help="Pruning plan JSON path, 'NONE', or omit for baseline")
    ap.add_argument("--label",           required=True,
                    help="Result label (also used as JSON filename stem)")
    ap.add_argument("--method",          default="baseline")
    ap.add_argument("--selector",        default="none")
    ap.add_argument("--dataset",         default="none")
    ap.add_argument("--target-pct",      type=float, default=0.0)
    ap.add_argument("--actual-pct",      type=float, default=0.0,
                    help="Estimated actual %; will be overridden from plan JSON")
    ap.add_argument("--out-json",        required=True)
    ap.add_argument("--dtype",           default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-warmup",        type=int, default=2)
    ap.add_argument("--n-bench",         type=int, default=5)
    ap.add_argument("--batch-size",      type=int, default=1)
    ap.add_argument("--max-new-tokens",  type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--dry-run",         action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    plan_path: Optional[str] = args.plan
    if plan_path in (None, "", "NONE", "none"):
        plan_path = None

    print(f"[bench] ── benchmark_moe_speed_memory.py ──────────────────────────────")
    print(f"[bench] label       : {args.label}")
    print(f"[bench] model       : {args.model}")
    print(f"[bench] plan        : {plan_path or '(none — baseline)'}")
    print(f"[bench] method      : {args.method}  selector: {args.selector}")
    print(f"[bench] dataset     : {args.dataset}  target: {args.target_pct}%  actual(est): {args.actual_pct}%")
    print(f"[bench] dtype       : {args.dtype}  warmup/bench: {args.n_warmup}/{args.n_bench}")
    print(f"[bench] out_json    : {args.out_json}")

    if args.dry_run:
        result = {
            "label": args.label, "method": args.method, "selector": args.selector,
            "dataset": args.dataset, "target_pct": args.target_pct,
            "actual_pct": args.actual_pct, "status": "dry_run",
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(result, fh, indent=2)
        print("[bench] DRY_RUN: wrote stub JSON, exiting.")
        return

    if plan_path is not None and not os.path.isfile(plan_path):
        print(f"[bench] ERROR: plan not found: {plan_path}")
        sys.exit(1)

    # Load plan early to get actual_pct
    plan_data: Optional[Dict] = None
    actual_pct = args.actual_pct
    if plan_path is not None:
        with open(plan_path) as fh:
            plan_data = json.load(fh)
        actual_pct = compute_actual_pct_from_plan(plan_data)
        print(f"[bench] actual_pct from plan: {actual_pct:.3f}%")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Fresh process = fresh CUDA context; reset peak stats to be explicit
    reset_all_peak_memory()

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n[bench] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[bench] Loading model (dtype={args.dtype}, device_map=auto) ...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    cuda_sync_all()
    load_sec = time.perf_counter() - t0_load

    params_before          = count_params(model)
    mem_after_load_mib     = current_total_allocated_mib()
    print(f"[bench] Loaded in {load_sec:.1f}s  params={params_before:,}  "
          f"allocated={mem_after_load_mib:.0f} MiB")

    # ── Apply pruning plan ───────────────────────────────────────────────────
    n_layers_pruned = 0
    if plan_data is not None:
        print(f"[bench] Applying pruning plan ...")
        reset_all_peak_memory()
        n_layers_pruned = apply_pruning_plan(model, plan_data)
        cuda_sync_all()
        print(f"[bench] {n_layers_pruned} layers pruned.")

    params_after            = count_params(model)
    mem_after_pruning_mib   = current_total_allocated_mib()
    total_param_reduction   = round(100.0 * (params_before - params_after) / max(params_before, 1), 3)
    # For MoE packed pruning, all removed params are in expert MLPs
    expert_param_reduction  = total_param_reduction
    flop_reduction          = total_param_reduction   # FLOPs ∝ d_ff for MLP

    print(f"[bench] param reduction: {total_param_reduction:.3f}%  "
          f"({params_before:,} → {params_after:,})")

    # ── Benchmark ────────────────────────────────────────────────────────────
    reset_all_peak_memory()
    print(f"\n[bench] Benchmarking ({args.n_warmup} warmup + {args.n_bench} measured) ...")
    timing = benchmark_generation(
        model, tokenizer, DEFAULT_PROMPT,
        max_new_tokens=args.max_new_tokens,
        n_warmup=args.n_warmup,
        n_bench=args.n_bench,
        batch_size=args.batch_size,
    )
    cuda_sync_all()

    mem_after_bench_mib = current_total_allocated_mib()
    peak_mem            = peak_memory_snapshot()

    print(f"[bench] prefill={timing['prefill_latency_ms_mean']:.1f}ms  "
          f"decode={timing['decode_latency_ms_mean']:.2f}ms/tok  "
          f"tok/s={timing['tokens_per_sec_mean']:.1f}  "
          f"peak_alloc={peak_mem['peak_allocated_mib_total']:.0f} MiB")

    # ── Assemble result ───────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "label":                           args.label,
        "method":                          args.method,
        "selector":                        args.selector,
        "dataset":                         args.dataset,
        "target_pct":                      args.target_pct,
        "actual_pct":                      actual_pct,
        "expert_param_reduction_pct":      expert_param_reduction,
        "total_model_param_reduction_pct": total_param_reduction,
        "active_expert_flop_reduction_pct": flop_reduction,
        "prompt_len":                      timing["prompt_len"],
        "generated_tokens":                timing["generated_tokens"],
        "batch_size":                      timing["batch_size"],
        "prefill_latency_ms_mean":         timing["prefill_latency_ms_mean"],
        "decode_latency_ms_mean":          timing["decode_latency_ms_mean"],
        "end_to_end_latency_ms_mean":      timing["end_to_end_latency_ms_mean"],
        "tokens_per_sec_mean":             timing["tokens_per_sec_mean"],
        "peak_allocated_mib_total":        peak_mem["peak_allocated_mib_total"],
        "peak_reserved_mib_total":         peak_mem["peak_reserved_mib_total"],
        "peak_allocated_mib_gpu0":         peak_mem["peak_allocated_mib_gpu0"],
        "peak_allocated_mib_gpu1":         peak_mem["peak_allocated_mib_gpu1"],
        "memory_after_load_mib_total":     round(mem_after_load_mib,    1),
        "memory_after_pruning_mib_total":  round(mem_after_pruning_mib, 1),
        "memory_after_benchmark_mib_total": round(mem_after_bench_mib,  1),
        "n_layers_pruned":                 n_layers_pruned,
        "params_before":                   params_before,
        "params_after":                    params_after,
        "load_sec":                        round(load_sec, 2),
        "n_warmup":                        timing["n_warmup"],
        "n_bench":                         timing["n_bench"],
        "model_name":                      args.model,
        "plan_path":                       plan_path or "",
        "status":                          "ok",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n[bench] Result written: {args.out_json}")


if __name__ == "__main__":
    main()
