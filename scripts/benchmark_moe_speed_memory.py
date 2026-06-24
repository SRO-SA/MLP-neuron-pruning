#!/usr/bin/env python3
"""
benchmark_moe_speed_memory.py
==============================
Stage 2 of the two-stage speed/memory pipeline.

Default behavior (saved-checkpoint mode):
  --model points to a saved pruned checkpoint (from apply_moe_plan_save_checkpoint.py).
  No pruning is applied inside this process.
  Peak memory reflects the pruned model only.

Opt-in legacy (--apply-plan-inside-benchmark):
  Loads the original model and applies the pruning plan in-process.
  Peak memory includes the full original model load -- NOT a clean benchmark.
  Only use for quick debugging; not for published results.

Memory reporting:
  memory_after_load_*    : allocated/reserved after model.from_pretrained returns
                           (before benchmark; peak reset here)
  peak_inference_*       : peak allocated/reserved during generate() calls only
  These are separated by a reset_all_peak_memory() call between load and benchmark.

Usage:
  # Baseline:
  python scripts/benchmark_moe_speed_memory.py \\
      --model Qwen/Qwen3-30B-A3B --base-model Qwen/Qwen3-30B-A3B \\
      --label baseline_no_pruning --method baseline ...

  # Pruned (two-stage default):
  python scripts/benchmark_moe_speed_memory.py \\
      --model results/speed_memory_runs/<id>/pruned_checkpoints/<label> \\
      --base-model Qwen/Qwen3-30B-A3B \\
      --plan results/pruning_plans/<plan>.json \\
      --label <label> --method pure_delete ...
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("ERROR: torch not installed. Run: pip install torch --break-system-packages")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("ERROR: transformers not installed. Run: pip install transformers")

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
DEFAULT_MAX_NEW_TOKENS = 32


# ---------------------------------------------------------------------------
# Multi-GPU memory helpers
# ---------------------------------------------------------------------------

def _n_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def reset_all_peak_memory() -> None:
    """Reset peak memory stats on every visible GPU using device-context API."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                    torch.cuda.reset_accumulated_memory_stats()
        except RuntimeError as e:
            print(f"[bench] WARNING: CUDA memory reset failed cuda:{i}: {e}")


def current_allocated_and_reserved_mib() -> tuple:
    """Return (total_allocated_mib, total_reserved_mib) across all GPUs."""
    n = _n_gpus()
    if n == 0:
        return float("nan"), float("nan")
    tot_alloc = 0.0
    tot_res   = 0.0
    for i in range(n):
        try:
            with torch.cuda.device(i):
                tot_alloc += torch.cuda.memory_allocated() / (1024 ** 2)
                tot_res   += torch.cuda.memory_reserved()  / (1024 ** 2)
        except RuntimeError:
            pass
    return tot_alloc, tot_res


def peak_memory_snapshot() -> Dict[str, float]:
    """
    Return peak allocated + reserved per GPU and totals.
    Always includes gpu0 and gpu1 keys (nan if not present).
    """
    n = _n_gpus()
    result: Dict[str, float] = {
        "peak_allocated_mib_gpu0":  float("nan"),
        "peak_allocated_mib_gpu1":  float("nan"),
        "peak_reserved_mib_gpu0":   float("nan"),
        "peak_reserved_mib_gpu1":   float("nan"),
        "peak_allocated_mib_total": float("nan"),
        "peak_reserved_mib_total":  float("nan"),
    }
    if n == 0:
        return result
    tot_alloc = 0.0
    tot_res   = 0.0
    for i in range(n):
        try:
            with torch.cuda.device(i):
                alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                res   = torch.cuda.max_memory_reserved()  / (1024 ** 2)
            result[f"peak_allocated_mib_gpu{i}"] = round(alloc, 1)
            result[f"peak_reserved_mib_gpu{i}"]  = round(res,   1)
            tot_alloc += alloc
            tot_res   += res
        except RuntimeError as e:
            print(f"[bench] WARNING: peak memory read failed cuda:{i}: {e}")
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


# ---------------------------------------------------------------------------
# Pruning helpers (used only in --apply-plan-inside-benchmark mode)
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
    """Physical structured pruning (in-process). Returns count of modified layers."""
    layer_list = _find_layer_list(model)
    if layer_list is None:
        raise RuntimeError("Cannot find transformer layer list.")
    n = 0
    for lcfg in plan.get("layers", []):
        li        = lcfg["layer_idx"]
        prune_idx = lcfg.get("prune_idx", [])
        old_d_ff  = lcfg.get("old_intermediate", 0)
        if not prune_idx or old_d_ff == 0:
            continue
        keep = torch.ones(old_d_ff, dtype=torch.bool)
        for idx in prune_idx:
            keep[idx] = False
        layer   = layer_list[li]
        mlp     = getattr(layer, "mlp", None)
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
        n += 1
    return n


def compute_actual_pct_from_plan(plan: Dict) -> float:
    total  = sum(lc.get("old_intermediate", 0) for lc in plan.get("layers", []))
    pruned = sum(len(lc.get("prune_idx", []))  for lc in plan.get("layers", []))
    return round(100.0 * pruned / max(total, 1), 3) if total else 0.0


def sample_expert_shapes_from_model(model: Any) -> Dict[str, Any]:
    """Get gate_proj/up_proj/down_proj shapes from the first expert found in the model."""
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
            w = getattr(exp0, proj, None)
            if w is not None:
                result[key] = list(w.weight.shape)
        break
    return result


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def benchmark_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    n_warmup: int,
    n_bench: int,
    batch_size: int = 1,
) -> Dict[str, Any]:
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
                decode_ms_list.append(e2e_ms / n_new)

    def _m(lst: list) -> float:
        return statistics.mean(lst) if lst else float("nan")

    e2e_mean    = _m(e2e_ms_list)
    decode_mean = _m(decode_ms_list)
    # prefill ~ total - (decode_per_tok * (n_new - 1))
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", required=True,
        help="HF model path to load. For pruned settings, pass the saved "
             "pruned checkpoint dir (not the original model ID).",
    )
    ap.add_argument(
        "--base-model", default=None,
        help="Original HF model ID (metadata/validation). If not set, defaults to --model.",
    )
    ap.add_argument(
        "--plan", default=None,
        help="Pruning plan JSON path (metadata in default mode). 'NONE' or omit for baseline.",
    )
    ap.add_argument(
        "--apply-plan-inside-benchmark", action="store_true", default=False,
        help="LEGACY: apply pruning plan inside this process. Peak memory includes "
             "full original model. Default=False: load saved pruned checkpoint directly.",
    )
    ap.add_argument("--label",          required=True)
    ap.add_argument("--method",         default="baseline")
    ap.add_argument("--selector",       default="none")
    ap.add_argument("--dataset",        default="none")
    ap.add_argument("--target-pct",     type=float, default=0.0)
    ap.add_argument("--actual-pct",     type=float, default=0.0,
                    help="Estimated actual %; overridden from plan JSON if available.")
    ap.add_argument("--out-json",       required=True)
    ap.add_argument("--dtype",          default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-warmup",       type=int, default=2)
    ap.add_argument("--n-bench",        type=int, default=5)
    ap.add_argument("--batch-size",     type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--dry-run",        action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    plan_path: Optional[str] = args.plan
    if plan_path in (None, "", "NONE", "none"):
        plan_path = None

    base_model = args.base_model or args.model

    # Determine mode label for JSON
    if args.apply_plan_inside_benchmark:
        mode = "apply_inside"
    elif plan_path:
        mode = "saved_ckpt"
    else:
        mode = "direct"

    print("[bench] benchmark_moe_speed_memory.py")
    print(f"[bench]   label                          : {args.label}")
    print(f"[bench]   loaded_model_path              : {args.model}")
    print(f"[bench]   base_model_name                : {base_model}")
    print(f"[bench]   plan                           : {plan_path or '(none - baseline)'}")
    print(f"[bench]   applying_plan_inside_benchmark : {args.apply_plan_inside_benchmark}")
    print(f"[bench]   mode                           : {mode}")
    print(f"[bench]   method / selector              : {args.method} / {args.selector}")
    print(f"[bench]   target / actual (est)          : {args.target_pct}% / {args.actual_pct}%")
    print(f"[bench]   dtype / warmup / bench         : {args.dtype} / {args.n_warmup} / {args.n_bench}")
    print(f"[bench]   out_json                       : {args.out_json}")

    if args.dry_run:
        result: Dict[str, Any] = {
            "label": args.label, "method": args.method, "selector": args.selector,
            "dataset": args.dataset, "target_pct": args.target_pct,
            "actual_pct": args.actual_pct, "status": "dry_run",
            "loaded_model_path": args.model, "base_model_name": base_model,
            "applying_plan_inside_benchmark": args.apply_plan_inside_benchmark,
            "mode": mode,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(result, fh, indent=2)
        print("[bench] DRY_RUN: wrote stub JSON, exiting.")
        return

    # ── Validation: in default mode, pruned settings must load the pruned ckpt ─
    if plan_path and not args.apply_plan_inside_benchmark:
        if args.base_model is not None and args.model == args.base_model:
            sys.exit(
                f"ERROR: --model == --base-model == {args.model} for a pruned setting "
                f"without --apply-plan-inside-benchmark. "
                f"Pass the saved pruned checkpoint path as --model, not the original model."
            )

    # ── Load plan data (for metadata / apply-inside mode) ────────────────────
    plan_data: Optional[Dict] = None
    actual_pct = args.actual_pct
    if plan_path is not None:
        if os.path.isfile(plan_path):
            with open(plan_path) as fh:
                plan_data = json.load(fh)
            actual_pct = compute_actual_pct_from_plan(plan_data)
            print(f"[bench]   actual_pct (from plan)         : {actual_pct:.3f}%")
        else:
            print(f"[bench] NOTE: plan JSON not found (metadata only): {plan_path}")

    # ── Load bench_meta.json (Stage 1 metadata, for shapes + moe_dim) ────────
    bench_meta: Dict[str, Any] = {}
    if not args.apply_plan_inside_benchmark:
        candidate = os.path.join(args.model, ".bench_meta.json")
        if os.path.isfile(candidate):
            try:
                with open(candidate) as f:
                    bench_meta = json.load(f)
                saved_dim = bench_meta.get("saved_moe_intermediate_size")
                print(f"[bench]   bench_meta loaded              : {candidate}")
                print(f"[bench]   saved_moe_intermediate_size   : {saved_dim}")
            except Exception as e:
                print(f"[bench] WARNING: cannot read .bench_meta.json: {e}")

    # ── Count checkpoint shards ───────────────────────────────────────────────
    n_shards      = bench_meta.get("n_shards", 0)
    ckpt_size_gib = bench_meta.get("checkpoint_size_gib", float("nan"))
    if n_shards == 0 and os.path.isdir(args.model):
        shard_files = (
            glob.glob(os.path.join(args.model, "model*.safetensors")) +
            glob.glob(os.path.join(args.model, "pytorch_model*.bin"))
        )
        n_shards = len(shard_files)
        if shard_files:
            ckpt_size_gib = round(
                sum(os.path.getsize(f) for f in shard_files) / (1024 ** 3), 3
            )

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # ── Load model ────────────────────────────────────────────────────────────
    # Fresh process = fresh CUDA context; reset before load so load-phase
    # peak is also captured if needed.
    reset_all_peak_memory()

    print(f"\n[bench] Loading tokenizer from: {args.model}")
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

    params_before = count_params(model)
    load_alloc, load_reserved = current_allocated_and_reserved_mib()
    print(f"[bench] Loaded in {load_sec:.1f}s  params={params_before:,}  "
          f"alloc={load_alloc:.0f} MiB  reserved={load_reserved:.0f} MiB")

    # ── Opt-in: apply plan inside benchmark (legacy, not recommended) ─────────
    n_layers_pruned = 0
    params_after    = params_before
    if args.apply_plan_inside_benchmark and plan_data is not None:
        print("[bench] LEGACY: applying pruning plan inside benchmark process ...")
        reset_all_peak_memory()
        n_layers_pruned = apply_pruning_plan(model, plan_data)
        cuda_sync_all()
        params_after = count_params(model)
        total_param_reduction = round(
            100.0 * (params_before - params_after) / max(params_before, 1), 3
        )
        print(f"[bench] {n_layers_pruned} layers pruned. "
              f"Reduction: {total_param_reduction:.3f}%  "
              f"({params_before:,} -> {params_after:,})")
        # Re-read memory after in-process pruning
        load_alloc, load_reserved = current_allocated_and_reserved_mib()
    else:
        # Checkpoint is already pruned; params_before == params of the loaded ckpt
        params_after = params_before
        total_param_reduction = 0.0  # computed from expert_pct in this mode

    # ── Read MoE dim and shapes ──────────────────────────────────────────────
    saved_moe_dim = bench_meta.get("saved_moe_intermediate_size")
    if saved_moe_dim is None:
        saved_moe_dim = getattr(model.config, "moe_intermediate_size", None)

    sample_gate = bench_meta.get("sample_gate_proj_shape")
    sample_up   = bench_meta.get("sample_up_proj_shape")
    sample_down = bench_meta.get("sample_down_proj_shape")
    if args.apply_plan_inside_benchmark and (
        sample_gate is None or sample_up is None or sample_down is None
    ):
        live = sample_expert_shapes_from_model(model)
        sample_gate = sample_gate or live.get("sample_gate_proj_shape")
        sample_up   = sample_up   or live.get("sample_up_proj_shape")
        sample_down = sample_down or live.get("sample_down_proj_shape")

    print("[bench]   saved_moe_intermediate_size : " + str(saved_moe_dim))
    if sample_gate:
        print("[bench]   sample_gate_proj_shape    : " + str(sample_gate))
    if sample_up:
        print("[bench]   sample_up_proj_shape      : " + str(sample_up))
    if sample_down:
        print("[bench]   sample_down_proj_shape    : " + str(sample_down))

    # ── Reset peak stats to isolate inference from load ──────────────────────
    # Everything after this point counts toward peak_inference_*.
    # Memory already resident (model weights) shows as "current" but peak starts fresh.
    reset_all_peak_memory()

    print("\n[bench] Benchmarking (" + str(args.n_warmup) + " warmup + " + str(args.n_bench) + " measured) ...")
    timing = benchmark_generation(
        model, tokenizer, DEFAULT_PROMPT,
        max_new_tokens=args.max_new_tokens,
        n_warmup=args.n_warmup,
        n_bench=args.n_bench,
        batch_size=args.batch_size,
    )
    cuda_sync_all()
    peak_inference = peak_memory_snapshot()

    print(
        "[bench] prefill={:.1f}ms  decode={:.2f}ms/tok  tok/s={:.1f}  "
        "peak_inference_alloc={:.0f} MiB  load_alloc={:.0f} MiB".format(
            timing["prefill_latency_ms_mean"],
            timing["decode_latency_ms_mean"],
            timing["tokens_per_sec_mean"],
            peak_inference["peak_allocated_mib_total"],
            load_alloc,
        )
    )

    # ── Pruned-row warnings ──────────────────────────────────────────────────
    if plan_path and not args.apply_plan_inside_benchmark:
        if saved_moe_dim is not None and saved_moe_dim == 768:
            print("[bench] WARNING: saved_moe_intermediate_size=768 for pruned row -- "
                  "checkpoint may not have been pruned (ratio too small?).")

    # ── Assemble result ──────────────────────────────────────────────────────
    expert_pct = bench_meta.get("actual_pct", actual_pct)

    result_dict: Dict[str, Any] = {
        # Identity
        "label":                               args.label,
        "method":                              args.method,
        "selector":                            args.selector,
        "dataset":                             args.dataset,
        "target_pct":                          args.target_pct,
        "actual_pct":                          actual_pct,
        # Checkpoint provenance
        "loaded_model_path":                   args.model,
        "base_model_name":                     base_model,
        "pruning_plan_path":                   plan_path or "",
        "applying_plan_inside_benchmark":      args.apply_plan_inside_benchmark,
        "mode":                                mode,
        # MoE dim + shapes
        "saved_moe_intermediate_size":         saved_moe_dim,
        "sample_gate_proj_shape":              sample_gate,
        "sample_up_proj_shape":                sample_up,
        "sample_down_proj_shape":              sample_down,
        "num_checkpoint_shards":               n_shards,
        "checkpoint_size_gib":                 ckpt_size_gib,
        # Param reduction
        "expert_param_reduction_pct":          expert_pct,
        "total_model_param_reduction_pct":     total_param_reduction,
        "active_expert_flop_reduction_pct":    expert_pct,
        # Timing
        "prompt_len":                          timing["prompt_len"],
        "generated_tokens":                    timing["generated_tokens"],
        "batch_size":                          timing["batch_size"],
        "prefill_latency_ms_mean":             timing["prefill_latency_ms_mean"],
        "decode_latency_ms_mean":              timing["decode_latency_ms_mean"],
        "end_to_end_latency_ms_mean":          timing["end_to_end_latency_ms_mean"],
        "tokens_per_sec_mean":                 timing["tokens_per_sec_mean"],
        # Load memory (captured before benchmark reset)
        "memory_after_load_allocated_mib_total": round(load_alloc,    1),
        "memory_after_load_reserved_mib_total":  round(load_reserved, 1),
        # Inference peak (reset before benchmark -- isolated from load)
        "peak_inference_allocated_mib_total":    peak_inference["peak_allocated_mib_total"],
        "peak_inference_reserved_mib_total":     peak_inference["peak_reserved_mib_total"],
        # Per-GPU inference peak
        "peak_allocated_mib_gpu0":               peak_inference["peak_allocated_mib_gpu0"],
        "peak_allocated_mib_gpu1":               peak_inference["peak_allocated_mib_gpu1"],
        # Legacy aliases (same values, for backward compat)
        "peak_allocated_mib_total":              peak_inference["peak_allocated_mib_total"],
        "peak_reserved_mib_total":               peak_inference["peak_reserved_mib_total"],
        # Other
        "params_before":                         params_before,
        "params_after":                          params_after,
        "n_layers_pruned":                       n_layers_pruned,
        "load_sec":                              round(load_sec, 2),
        "n_warmup":                              timing["n_warmup"],
        "n_bench":                               timing["n_bench"],
        "model_name":                            args.model,
        "plan_path":                             plan_path or "",
        "status":                                "ok",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(result_dict, fh, indent=2, default=str)
    print("\n[bench] Result written: " + args.out_json)


if __name__ == "__main__":
    main()
