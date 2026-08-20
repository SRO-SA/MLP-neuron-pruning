#!/usr/bin/env python3
"""Evaluate one verified checkpoint with a pinned lm-eval protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.heterogeneous_moe_checkpoint import (
    PLAN_FILENAME, inspect_plan_shapes, load_heterogeneous_checkpoint,
)
from src.tokenizer_policy import resolve_tokenizer_policy

DEFAULT_TASKS = (
    "hellaswag,mathqa,openbookqa,piqa,winogrande,arc_easy,arc_challenge"
)


def harness_identity() -> dict:
    import lm_eval

    try:
        version = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(lm_eval, "__version__", "unknown")
    source = os.path.realpath(os.path.dirname(lm_eval.__file__))
    revision = ""
    try:
        revision = subprocess.check_output(
            ["git", "-C", source, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"package_version": version, "git_revision": revision,
            "source_path": source}


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path: str, dtype):
    if os.path.isfile(os.path.join(path, PLAN_FILENAME)):
        model, plan = load_heterogeneous_checkpoint(
            path, device_map="auto", dtype=dtype
        )
        shapes = inspect_plan_shapes(model, plan)
        if not all(row["no_original_width_padding"] for row in shapes):
            raise AssertionError("heterogeneous checkpoint contains width padding")
        return model, plan, shapes
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, None, []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokenizer-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-harness-identity", required=True,
                        help="Exact lm-eval git commit, or package version if not a git checkout")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    identity = harness_identity()
    actual_identity = identity["git_revision"] or identity["package_version"]
    if actual_identity != args.expected_harness_identity:
        raise RuntimeError(
            "lm-eval identity mismatch: "
            f"expected={args.expected_harness_identity!r}, actual={actual_identity!r}"
        )
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if args.include_optional:
        tasks.extend(task for task in ("boolq", "rte") if task not in tasks)

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    dtype = _dtype(args.dtype)
    verification_path = os.path.join(
        args.checkpoint, "checkpoint_verification.json"
    )
    with open(verification_path, encoding="utf-8") as handle:
        checkpoint_verification = json.load(handle)
    tokenizer_policy = resolve_tokenizer_policy(
        args.tokenizer_audit, args.checkpoint, label=args.label,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, use_fast=True,
        fix_mistral_regex=tokenizer_policy["fix_mistral_regex"],
    )
    model, plan, shape_audit = load_checkpoint(args.checkpoint, dtype)
    lm = HFLM(
        pretrained=model, tokenizer=tokenizer, backend="causal",
        batch_size=args.batch_size, dtype=dtype,
    )
    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=args.num_fewshot,
        batch_size=args.batch_size, limit=args.limit,
        log_samples=True, apply_chat_template=False,
        fewshot_as_multiturn=False, bootstrap_iters=10000,
        random_seed=args.seed, numpy_random_seed=args.seed,
        torch_random_seed=args.seed, fewshot_random_seed=args.seed,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no rank-0 results")
    results["paper_v3_protocol"] = {
        "schema_version": 1, "checkpoint": os.path.realpath(args.checkpoint),
        "label": args.label, "harness": identity, "tasks": tasks,
        "task_versions": results.get("versions", {}),
        "num_fewshot": args.num_fewshot, "batch_size": args.batch_size,
        "dtype": args.dtype, "seed_python": args.seed,
        "seed_numpy": args.seed, "seed_torch": args.seed,
        "seed_fewshot": args.seed, "apply_chat_template": False,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_class": type(tokenizer).__name__,
        "fix_mistral_regex": tokenizer_policy["fix_mistral_regex"],
        "tokenizer_audit_sha256": tokenizer_policy["tokenizer_audit_sha256"],
        "tokenizer_files_combined_sha256": tokenizer_policy[
            "tokenizer_files_combined_sha256"
        ],
        "source_model_revision": checkpoint_verification.get(
            "source_model_revision", ""
        ),
        "tokenizer_revision": checkpoint_verification.get(
            "tokenizer_revision", ""
        ),
        "checkpoint_verification_sha256": _sha256(verification_path),
        "limit": args.limit, "heterogeneous_plan_present": plan is not None,
        "shape_audit": shape_audit,
        "task_configs_sha256": _canonical_sha256(results.get("configs", {})),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=handle_non_serializable)
    print(f"[downstream] OK: {args.label} tasks={len(tasks)} output={args.output}")


if __name__ == "__main__":
    main()
