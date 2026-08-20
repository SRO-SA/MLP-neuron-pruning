#!/usr/bin/env python3
"""Export and rigorously reload one baseline or heterogeneous MoE checkpoint."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment_provenance import file_sha256
from src.heterogeneous_moe_checkpoint import (
    PLAN_FILENAME,
    apply_plan_physical,
    count_parameters,
    inspect_plan_shapes,
    load_heterogeneous_checkpoint,
)


def _directory_files(path: str) -> list[str]:
    result = []
    for root, _, files in os.walk(path):
        for filename in files:
            result.append(os.path.join(root, filename))
    return sorted(result)


def _weight_files(path: str) -> list[str]:
    return [file for file in _directory_files(path) if file.endswith(
        (".safetensors", ".bin")
    )]


def _logits(model, tokenizer, prompt: str) -> torch.Tensor:
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[:, -1, :]
    return logits.detach().float().cpu()


def _sha_manifest(path: str) -> dict[str, str]:
    return {
        os.path.relpath(file, path): file_sha256(file)
        for file in _directory_files(path)
        if os.path.basename(file) != "checkpoint_verification.json"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", default="")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--expected-layer-channels", type=int, default=0)
    parser.add_argument("--expected-expert-neurons", type=int, default=0)
    parser.add_argument("--expected-plan-sha256", default="")
    parser.add_argument(
        "--verification-prompt",
        default="Structured pruning should preserve useful language model behavior.",
    )
    args = parser.parse_args()
    if os.path.exists(args.checkpoint_dir):
        raise FileExistsError(
            f"refusing to overwrite checkpoint: {args.checkpoint_dir}"
        )
    temporary = f"{args.checkpoint_dir}.incomplete.{os.getpid()}"
    if os.path.exists(temporary):
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    os.makedirs(os.path.dirname(args.checkpoint_dir) or ".", exist_ok=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    plan = None
    plan_sha = ""
    if args.plan:
        if not os.path.isfile(args.plan):
            raise FileNotFoundError(args.plan)
        plan_sha = file_sha256(args.plan)
        if args.expected_plan_sha256 and plan_sha != args.expected_plan_sha256:
            raise ValueError("source pruning-plan hash does not match manifest")
        with open(args.plan, encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("pruning_mode") != "packed_same_channel":
            raise ValueError("paper checkpoint requires packed_same_channel pruning")
        if int(plan.get("channel_alignment", -1)) != 16:
            raise ValueError("paper checkpoint requires channel alignment 16")
        counted = sum(len(row.get("prune_idx", [])) for row in plan["layers"])
        if counted != args.expected_layer_channels:
            raise ValueError(
                f"plan contains {counted} layer-channels, expected "
                f"{args.expected_layer_channels}"
            )
        for row in plan["layers"]:
            if len(row.get("prune_idx", [])) % 16:
                raise ValueError(
                    f"layer {row['layer_idx']} removal count violates alignment"
                )

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    source_model_revision = str(getattr(model.config, "_commit_hash", "") or "")
    tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", "") or ""
    )
    source_load_seconds = time.perf_counter() - load_started
    before_counts = count_parameters(model)
    pruning_audit = {
        "layers": [], "removed_layer_channels": 0, "removed_expert_neurons": 0,
    }
    if plan is not None:
        pruning_audit = apply_plan_physical(model, plan)
        if pruning_audit["removed_layer_channels"] != args.expected_layer_channels:
            raise AssertionError("removed layer-channel total differs from manifest")
        if pruning_audit["removed_expert_neurons"] != args.expected_expert_neurons:
            raise AssertionError("removed expert-neuron total differs from manifest")
        shape_audit_before = inspect_plan_shapes(model, plan)
        if not all(row["no_original_width_padding"] for row in shape_audit_before):
            raise AssertionError("physical pruning retained original-width padding")
        model.config.heterogeneous_moe_widths = {
            str(row["layer_idx"]): row["new_width"]
            for row in pruning_audit["layers"]
        }
        model.config.heterogeneous_pruning_plan_file = PLAN_FILENAME
    else:
        shape_audit_before = []
    after_counts = count_parameters(model)
    logits_before = _logits(model, tokenizer, args.verification_prompt)

    os.makedirs(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    if plan is not None:
        shutil.copy2(args.plan, os.path.join(temporary, PLAN_FILENAME))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    reload_started = time.perf_counter()
    if plan is None:
        reloaded = AutoModelForCausalLM.from_pretrained(
            temporary, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        reloaded_plan = None
    else:
        reloaded, reloaded_plan = load_heterogeneous_checkpoint(
            temporary, device_map="auto", dtype=dtype
        )
    reloaded.eval()
    reload_seconds = time.perf_counter() - reload_started
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        temporary, trust_remote_code=True
    )
    logits_after = _logits(reloaded, reloaded_tokenizer, args.verification_prompt)
    exact_logits = torch.equal(logits_before, logits_after)
    max_logit_difference = float((logits_before - logits_after).abs().max())
    if not exact_logits:
        raise AssertionError(
            "logits changed after save/reload: "
            f"max_abs_difference={max_logit_difference:.9g}"
        )
    reloaded_counts = count_parameters(reloaded)
    if reloaded_counts != after_counts:
        raise AssertionError(
            f"parameter counts changed after reload: {after_counts} -> {reloaded_counts}"
        )
    if plan is not None:
        if reloaded_plan != plan:
            raise AssertionError("saved/reloaded pruning plan differs from source")
        shape_audit_after = inspect_plan_shapes(reloaded, plan)
        if shape_audit_after != shape_audit_before:
            raise AssertionError("expert tensor shapes changed after reload")
    else:
        shape_audit_after = []

    weight_files = _weight_files(temporary)
    if not weight_files:
        raise AssertionError("checkpoint contains no serialized weight files")
    serialized_weight_bytes = sum(os.path.getsize(file) for file in weight_files)
    payload_bytes = sum(os.path.getsize(file) for file in _directory_files(temporary))
    file_hashes = _sha_manifest(temporary)
    verification = {
        "schema_version": 1,
        "label": args.label,
        "base_model": args.model,
        "source_model_revision": source_model_revision,
        "tokenizer_revision": tokenizer_revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "dtype": args.dtype,
        "plan_path": args.plan,
        "plan_sha256": plan_sha,
        "source_load_seconds": source_load_seconds,
        "reload_seconds": reload_seconds,
        "removed_layer_channels": pruning_audit["removed_layer_channels"],
        "removed_expert_neurons": pruning_audit["removed_expert_neurons"],
        "parameters_before": before_counts,
        "parameters_after": after_counts,
        "parameters_reloaded": reloaded_counts,
        "shape_audit_before_save": shape_audit_before,
        "shape_audit_after_reload": shape_audit_after,
        "serialized_weight_bytes": serialized_weight_bytes,
        "checkpoint_payload_bytes_excluding_verification_manifest": payload_bytes,
        "weight_file_count": len(weight_files),
        "successful_reload": True,
        "exact_logits_after_reload": exact_logits,
        "max_logit_difference": max_logit_difference,
        "no_hidden_original_width_padding": (
            all(row["no_original_width_padding"] for row in shape_audit_after)
            if plan is not None else True
        ),
        "checkpoint_file_sha256": file_hashes,
    }
    with open(
        os.path.join(temporary, "checkpoint_verification.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(verification, handle, indent=2)
    os.replace(temporary, args.checkpoint_dir)
    print(
        f"[checkpoint] OK: {args.label} weights={serialized_weight_bytes} bytes "
        f"reload=True exact_logits=True dir={args.checkpoint_dir}"
    )


if __name__ == "__main__":
    main()
