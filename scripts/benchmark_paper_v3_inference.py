#!/usr/bin/env python3
"""Time true prefill and cached single-token decode for one verified checkpoint."""
from __future__ import annotations

import argparse
import hashlib
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
    PLAN_FILENAME, find_decoder_layers, inspect_plan_shapes,
    load_heterogeneous_checkpoint,
)
from src.tokenizer_policy import resolve_tokenizer_policy
from src.system_evidence import (
    shape_integers,
    validate_packed_moe_shapes,
    validate_unpacked_moe_shapes,
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
    result["prefill_latency_samples_ms"] = prefill
    result["decode_latency_per_token_samples_ms"] = decode_steps
    return result


def collect_moe_execution_evidence(model, tokenizer) -> dict:
    """Run one untimed forward and record packed or unpacked expert execution."""
    records: dict[int, dict] = {}
    handles = []
    for layer_idx, layer in enumerate(find_decoder_layers(model)):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        if isinstance(gate_up, torch.Tensor) and isinstance(down, torch.Tensor):
            layout = validate_packed_moe_shapes(gate_up.shape, down.shape)
            record = {
                "layer_idx": layer_idx, "module_class": type(experts).__name__,
                "layout": "packed", "call_count": 0,
                "expert_count": layout["expert_count"],
                "gate_up_proj_shape": list(gate_up.shape),
                "down_proj_shape": list(down.shape),
                "intermediate_width": layout["intermediate_width"],
                "device": str(gate_up.device), "input_hidden_shapes": [],
                "routing_count_summaries": [],
            }
            records[layer_idx] = record

            def packed_pre_hook(_module, inputs, *, target=record):
                target["call_count"] += 1
                tensor_inputs = [
                    value for value in inputs if isinstance(value, torch.Tensor)
                ]
                if tensor_inputs:
                    shape = list(tensor_inputs[0].shape)
                    if shape not in target["input_hidden_shapes"]:
                        target["input_hidden_shapes"].append(shape)
                route = next((
                    value for value in tensor_inputs[1:] if value.dtype in (
                        torch.int8, torch.int16, torch.int32, torch.int64,
                    )
                ), None)
                if route is not None:
                    counts = torch.bincount(
                        route.reshape(-1).long(),
                        minlength=target["gate_up_proj_shape"][0],
                    ).detach().cpu()
                    positive = counts[counts > 0].float()
                    target["routing_count_summaries"].append({
                        "routed_assignments": int(counts.sum()),
                        "active_experts": int((counts > 0).sum()),
                        "tokens_per_active_expert_min": (
                            int(positive.min()) if positive.numel() else 0
                        ),
                        "tokens_per_active_expert_median": (
                            float(positive.median()) if positive.numel() else 0.0
                        ),
                        "tokens_per_active_expert_max": (
                            int(positive.max()) if positive.numel() else 0
                        ),
                    })

            handles.append(experts.register_forward_pre_hook(packed_pre_hook))
            continue

        try:
            expert_modules = list(experts)
        except TypeError:
            expert_modules = []
        expert_shapes = []
        for expert in expert_modules:
            weights = {
                name: getattr(getattr(expert, f"{name}_proj", None), "weight", None)
                for name in ("gate", "up", "down")
            }
            if not all(
                isinstance(weight, torch.Tensor) for weight in weights.values()
            ):
                expert_shapes = []
                break
            expert_shapes.append({
                name: list(weight.shape) for name, weight in weights.items()
            })
        if not expert_shapes:
            continue
        layout = validate_unpacked_moe_shapes(expert_shapes)
        first_gate = expert_modules[0].gate_proj.weight
        record = {
            "layer_idx": layer_idx, "module_class": type(experts).__name__,
            "expert_module_class": type(expert_modules[0]).__name__,
            "layout": "unpacked", "call_count": 0,
            "expert_count": layout["expert_count"],
            "executed_expert_indices": [],
            "gate_proj_shape_per_expert": layout["gate_proj_shape"],
            "up_proj_shape_per_expert": layout["up_proj_shape"],
            "down_proj_shape_per_expert": layout["down_proj_shape"],
            "intermediate_width": layout["intermediate_width"],
            "device": str(first_gate.device), "input_hidden_shapes": [],
        }
        records[layer_idx] = record

        for expert_idx, expert in enumerate(expert_modules):
            def unpacked_pre_hook(
                _module, inputs, *, target=record, target_expert_idx=expert_idx,
            ):
                target["call_count"] += 1
                if target_expert_idx not in target["executed_expert_indices"]:
                    target["executed_expert_indices"].append(target_expert_idx)
                tensor_input = next((
                    value for value in inputs if isinstance(value, torch.Tensor)
                ), None)
                if tensor_input is not None:
                    shape = list(tensor_input.shape)
                    if shape not in target["input_hidden_shapes"]:
                        target["input_hidden_shapes"].append(shape)

            handles.append(expert.register_forward_pre_hook(unpacked_pre_hook))
    if not records:
        raise RuntimeError(
            "no supported packed or unpacked MoE expert modules found for "
            "runtime evidence"
        )
    device = model.get_input_embeddings().weight.device
    inputs = _exact_inputs(tokenizer, batch=1, length=128, device=device)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        _sync()
    finally:
        for handle in handles:
            handle.remove()
    for record in records.values():
        if record["call_count"] <= 0:
            raise AssertionError(
                f"MoE layer {record['layer_idx']} was not executed"
            )
        if record["layout"] == "packed":
            layout = validate_packed_moe_shapes(
                record["gate_up_proj_shape"], record["down_proj_shape"]
            )
            record["inferred_executed_gemm_dimensions"] = {
                "gate_up": {
                    "M": "routed tokens per expert", "K": layout["hidden_size"],
                    "N": 2 * layout["intermediate_width"],
                },
                "down": {
                    "M": "routed tokens per expert",
                    "K": layout["intermediate_width"], "N": layout["hidden_size"],
                },
            }
        else:
            record["executed_expert_indices"].sort()
            record["executed_expert_count"] = len(record["executed_expert_indices"])
            if record["executed_expert_count"] <= 0:
                raise AssertionError(
                    f"unpacked MoE layer {record['layer_idx']} routed no experts"
                )
            width = record["intermediate_width"]
            hidden = record["gate_proj_shape_per_expert"][1]
            record["inferred_executed_gemm_dimensions"] = {
                "gate": {"M": "routed tokens", "K": hidden, "N": width},
                "up": {"M": "routed tokens", "K": hidden, "N": width},
                "down": {"M": "routed tokens", "K": width, "N": hidden},
            }
    all_executed = all(row["call_count"] > 0 for row in records.values())
    layout_counts = {
        layout: sum(row["layout"] == layout for row in records.values())
        for layout in ("packed", "unpacked")
    }
    return {
        "evidence_type": (
            "runtime forward-pre-hooks on executed packed or unpacked expert modules"
        ),
        "probe_batch_size": 1, "probe_prompt_length_tokens": 128,
        "all_moe_layers_executed": all_executed,
        # Retained for backward compatibility with already-frozen systems tables.
        "all_packed_moe_layers_executed": (
            all_executed and layout_counts["unpacked"] == 0
        ),
        "layout_counts": layout_counts,
        "layers": [records[key] for key in sorted(records)],
    }


def collect_operator_profile(
    model, tokenizer, *, expected_widths: dict[int, int], trace_path: str,
) -> dict:
    """Capture one CUDA operator trace and retain GEMM-like input shapes."""
    from torch.profiler import ProfilerActivity, profile

    device = model.get_input_embeddings().weight.device
    inputs = _exact_inputs(tokenizer, batch=1, length=128, device=device)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    _sync()
    with profile(activities=activities, record_shapes=True) as prof:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        _sync()
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    prof.export_chrome_trace(trace_path)
    operation_rows = []
    shape_values = set()
    needles = ("mm", "matmul", "bmm", "addmm", "einsum", "linear")
    for event in prof.key_averages(group_by_input_shape=True):
        key = str(getattr(event, "key", ""))
        if not any(needle in key.lower() for needle in needles):
            continue
        shapes = getattr(event, "input_shapes", [])
        shape_values.update(shape_integers(shapes))
        operation_rows.append({
            "operator": key, "input_shapes": shapes,
            "calls": int(getattr(event, "count", 0)),
            "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            "device_time_total_us": float(
                getattr(event, "device_time_total", 0.0)
            ),
        })
    unique_expected = sorted(set(expected_widths.values()))
    matched = sorted(width for width in unique_expected if width in shape_values)
    hasher = hashlib.sha256()
    with open(trace_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return {
        "evidence_type": "torch.profiler CUDA/CPU operator trace with input shapes",
        "trace_path": os.path.realpath(trace_path),
        "trace_sha256": digest, "trace_size_bytes": os.path.getsize(trace_path),
        "gemm_like_operator_groups": operation_rows,
        "expected_unique_pruned_widths": unique_expected,
        "matched_unique_pruned_widths_in_operator_shapes": matched,
        "all_expected_pruned_widths_observed": (
            set(matched) == set(unique_expected) if unique_expected else True
        ),
    }


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
    parser.add_argument("--tokenizer-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--cases", default="1x128,1x512,1x2048,2x512,4x512")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--profile-gemm-shapes", action="store_true")
    parser.add_argument("--profiler-trace")
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
    tokenizer_policy = resolve_tokenizer_policy(
        args.tokenizer_audit, args.checkpoint, label=args.label,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True,
        fix_mistral_regex=tokenizer_policy["fix_mistral_regex"],
    )
    dtype = _dtype(args.dtype)
    plan_path = os.path.join(args.checkpoint, PLAN_FILENAME)
    if os.path.isfile(plan_path):
        model, plan = load_heterogeneous_checkpoint(
            args.checkpoint, device_map="auto", dtype=dtype
        )
        shape_audit = inspect_plan_shapes(model, plan)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, dtype=dtype, device_map="auto",
            trust_remote_code=True,
        )
        model.eval(); shape_audit = []
    _sync(); load_seconds = time.perf_counter() - started
    after_load = _memory("after_load")
    execution_evidence = collect_moe_execution_evidence(model, tokenizer)
    expected_widths = {
        int(row["layer_idx"]): int(row["expected_width"]) for row in shape_audit
    }
    runtime_widths = {
        int(row["layer_idx"]): int(row["intermediate_width"])
        for row in execution_evidence["layers"]
    }
    runtime_widths_match = (
        not expected_widths
        or (set(runtime_widths) == set(expected_widths) and all(
            runtime_widths[layer_idx] == width
            for layer_idx, width in expected_widths.items()
        ))
    )
    if expected_widths and not runtime_widths_match:
        raise AssertionError("runtime MoE widths differ from pruning plan")
    all_moe_layers_executed = execution_evidence.get(
        "all_moe_layers_executed",
        execution_evidence.get("all_packed_moe_layers_executed", False),
    )
    operator_profile = {
        "enabled": False,
        "reason": "Pass --profile-gemm-shapes for a CUDA operator trace.",
    }
    if args.profile_gemm_shapes:
        trace_path = args.profiler_trace or os.path.splitext(args.output)[0] + ".trace.json.gz"
        operator_profile = collect_operator_profile(
            model, tokenizer, expected_widths=expected_widths,
            trace_path=trace_path,
        )
        operator_profile["enabled"] = True
        if expected_widths and not operator_profile[
            "all_expected_pruned_widths_observed"
        ]:
            print(
                "[systems] WARNING: operator trace did not expose every pruned "
                "width; runtime module shapes remain validated, but do not claim "
                "kernel-profiler confirmation from this run."
            )
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
        "schema_version": 2, "label": args.label,
        "checkpoint": os.path.realpath(args.checkpoint),
        "checkpoint_storage_bytes": verification["serialized_weight_bytes"],
        "checkpoint_payload_bytes": verification[
            "checkpoint_payload_bytes_excluding_verification_manifest"
        ],
        "load_time_seconds": load_seconds, **after_load, **peak,
        "successful_load": True,
        "dtype": args.dtype, "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "inference_engine": "transformers eager/SDPA as configured by checkpoint",
        "model_class": type(model).__name__,
        "attention_implementation": str(
            getattr(model.config, "_attn_implementation", "")
        ),
        "hf_device_map": {
            str(key): str(value)
            for key, value in (getattr(model, "hf_device_map", {}) or {}).items()
        },
        "nvidia_smi": _nvidia_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "kernel_environment": {key: os.environ.get(key, "") for key in (
            "TORCH_CUDNN_V8_API_ENABLED", "PYTORCH_CUDA_ALLOC_CONF",
            "NVIDIA_TF32_OVERRIDE",
        )},
        "reduced_intermediate_dimensions_executed": (
            bool(shape_audit) and all(row["actual_width"] == row["expected_width"]
                                      for row in shape_audit)
            and runtime_widths_match
            and all_moe_layers_executed
        ) if os.path.isfile(plan_path) else all_moe_layers_executed,
        "tokenizer_policy": tokenizer_policy,
        "shape_audit": shape_audit,
        "runtime_moe_execution_evidence": execution_evidence,
        "operator_profile_evidence": operator_profile,
        "cases": cases,
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
