#!/usr/bin/env python3
"""Time true prefill and cached single-token decode for one verified checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.heterogeneous_moe_checkpoint import (
    PLAN_FILENAME, inspect_plan_shapes, load_heterogeneous_checkpoint,
)


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _sync() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _reset_peaks() -> None:
    _sync()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _memory(prefix: str, *, peak: bool = False) -> dict:
    result = {}
    alloc_fn = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    reserve_fn = torch.cuda.max_memory_reserved if peak else torch.cuda.memory_reserved
    total_alloc = total_reserved = 0
    for index in range(torch.cuda.device_count()):
        allocated = int(alloc_fn(index)); reserved = int(reserve_fn(index))
        result[f"{prefix}_allocated_bytes_gpu{index}"] = allocated
        result[f"{prefix}_reserved_bytes_gpu{index}"] = reserved
        total_alloc += allocated; total_reserved += reserved
    result[f"{prefix}_allocated_bytes_total"] = total_alloc
    result[f"{prefix}_reserved_bytes_total"] = total_reserved
    return result


def _stats(values: list[float], prefix: str) -> dict:
    return {
        f"{prefix}_mean_ms": statistics.mean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_min_ms": min(values), f"{prefix}_max_ms": max(values),
    }


def _exact_inputs(tokenizer, batch: int, length: int, device) -> dict:
    ids = tokenizer("A controlled benchmark prompt for structured MoE pruning.",
                    add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id]
    repeated = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    input_ids = torch.tensor([repeated] * batch, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def benchmark_case(model, tokenizer, *, batch: int, prompt_len: int,
                   decode_tokens: int, warmups: int, repetitions: int) -> dict:
    device = model.get_input_embeddings().weight.device
    inputs = _exact_inputs(tokenizer, batch, prompt_len, device)
    prefill = []
    with torch.inference_mode():
        for iteration in range(warmups + repetitions):
            _sync(); start = time.perf_counter()
            output = model(**inputs, use_cache=True)
            _sync(); elapsed = (time.perf_counter() - start) * 1000.0
            if iteration >= warmups: prefill.append(elapsed)
            del output

    decode_steps = []
    with torch.inference_mode():
        for iteration in range(warmups + repetitions):
            context = model(**inputs, use_cache=True)
            cache = context.past_key_values
            next_token = context.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mask = inputs["attention_mask"]
            per_step = []
            for _ in range(decode_tokens):
                mask = torch.cat((mask, torch.ones(
                    (batch, 1), dtype=mask.dtype, device=mask.device
                )), dim=1)
                _sync(); start = time.perf_counter()
                output = model(input_ids=next_token, attention_mask=mask,
                               past_key_values=cache, use_cache=True)
                _sync(); per_step.append((time.perf_counter() - start) * 1000.0)
                cache = output.past_key_values
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if iteration >= warmups: decode_steps.extend(per_step)
            del context, output, cache
    result = {
        "batch_size": batch, "prompt_length_tokens": prompt_len,
        "decode_tokens_per_repetition": decode_tokens,
        "warmup_repetitions": warmups, "timed_repetitions": repetitions,
        "prefill_samples": len(prefill), "decode_step_samples": len(decode_steps),
    }
    result.update(_stats(prefill, "prefill_latency"))
    result.update(_stats(decode_steps, "decode_latency_per_token"))
    result["prefill_tokens_per_second_median"] = (
        batch * prompt_len * 1000.0 / result["prefill_latency_median_ms"]
    )
    result["decode_tokens_per_second_median"] = (
        batch * 1000.0 / result["decode_latency_per_token_median_ms"]
    )
    return result


def _nvidia_info() -> str:
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--cases", default="1x128,1x512,1x2048,2x512,4x512")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    verification_path = os.path.join(args.checkpoint, "checkpoint_verification.json")
    with open(verification_path, encoding="utf-8") as handle:
        verification = json.load(handle)
    if not verification.get("successful_reload") or not verification.get(
        "exact_logits_after_reload"
    ):
        raise ValueError("checkpoint has not passed reload/logit verification")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    dtype = _dtype(args.dtype)
    plan_path = os.path.join(args.checkpoint, PLAN_FILENAME)
    if os.path.isfile(plan_path):
        model, plan = load_heterogeneous_checkpoint(
            args.checkpoint, device_map="auto", dtype=dtype
        )
        shape_audit = inspect_plan_shapes(model, plan)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True,
        )
        model.eval(); shape_audit = []
    _sync(); load_seconds = time.perf_counter() - started
    after_load = _memory("after_load")
    _reset_peaks()
    cases = []
    for text in args.cases.split(","):
        batch_text, length_text = text.lower().split("x", 1)
        case = benchmark_case(
            model, tokenizer, batch=int(batch_text), prompt_len=int(length_text),
            decode_tokens=args.decode_tokens, warmups=args.warmups,
            repetitions=args.repetitions,
        )
        cases.append(case)
        print(
            f"[systems] {args.label} b={case['batch_size']} l={case['prompt_length_tokens']} "
            f"prefill={case['prefill_latency_median_ms']:.2f}ms "
            f"decode={case['decode_latency_per_token_median_ms']:.2f}ms/token"
        )
    peak = _memory("peak_inference", peak=True)
    payload = {
        "schema_version": 1, "label": args.label,
        "checkpoint": os.path.realpath(args.checkpoint),
        "checkpoint_storage_bytes": verification["serialized_weight_bytes"],
        "checkpoint_payload_bytes": verification[
            "checkpoint_payload_bytes_excluding_verification_manifest"
        ],
        "load_time_seconds": load_seconds, **after_load, **peak,
        "dtype": args.dtype, "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "inference_engine": "transformers eager/SDPA as configured by checkpoint",
        "nvidia_smi": _nvidia_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "kernel_environment": {key: os.environ.get(key, "") for key in (
            "TORCH_CUDNN_V8_API_ENABLED", "PYTORCH_CUDA_ALLOC_CONF",
            "NVIDIA_TF32_OVERRIDE",
        )},
        "reduced_intermediate_dimensions_executed": (
            bool(shape_audit) and all(row["actual_width"] == row["expected_width"]
                                      for row in shape_audit)
        ) if os.path.isfile(plan_path) else True,
        "shape_audit": shape_audit, "cases": cases,
        "moe_gemm_microbenchmark": {
            "available": False,
            "reason": "No architecture-stable isolated Qwen3 packed-MoE GEMM hook is present; end-to-end prefill/decode are reported.",
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[systems] OK: {args.output}")


if __name__ == "__main__":
    main()
