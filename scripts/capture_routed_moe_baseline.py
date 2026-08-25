#!/usr/bin/env python3
"""Capture immutable baseline inputs and fixed routing for local MoE audits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from scripts.run_paper_v3_lm_eval import load_checkpoint
from src.experiment_provenance import file_sha256, normalize_document
from src.heterogeneous_moe_checkpoint import find_decoder_layers
from src.moe_set_certification import matched_plan_validation
from src.routed_moe_perturbation import (
    canonical_sha256,
    compute_fixed_routing,
    fixed_routed_moe_output,
    shard_path,
)
from src.tokenizer_policy import resolve_tokenizer_policy


DEFAULT_LABELS = (
    "rmsnorm_alloc__ellipsoid_rank__p95__target6",
    "certified_hybrid__downnorm_refinement_slack0p25__target6",
    "certified_hybrid__downnorm_refinement_slack2__target6",
    "rmsnorm_alloc__downnorm_rank__p95__target6",
)


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _manifest_specs(path: str, labels: tuple[str, ...]):
    specs = json.loads(Path(path).read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in specs}
    required = ("baseline_unpruned", *labels)
    missing = set(required) - set(by_label)
    if missing:
        raise ValueError(f"checkpoint manifest lacks labels: {sorted(missing)}")
    plans = {}
    for label in labels:
        spec = by_label[label]
        plan_path = Path(spec["plan_path"])
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        actual_hash = file_sha256(str(plan_path))
        if actual_hash != spec["plan_sha256"]:
            raise ValueError(f"{label}: pruning-plan hash mismatch")
        plans[label] = json.loads(plan_path.read_text(encoding="utf-8"))
    validation = matched_plan_validation(
        plans, expected_total=2288, expected_alignment=16,
    )
    if validation.get("validation_passed") is not True:
        raise AssertionError("target-6 plans failed matched-plan validation")
    return by_label, plans, validation


def _heldout_documents(args) -> list[dict]:
    dataset = load_dataset(
        args.dataset_repo, args.dataset_config, split=args.dataset_split,
        revision=args.dataset_revision or None, streaming=True,
        trust_remote_code=True,
    )
    selected = []
    eligible_index = -1
    for source_index, row in enumerate(dataset):
        raw_text = str(row.get("text", "")).strip()
        if len(raw_text) <= args.minimum_characters:
            continue
        eligible_index += 1
        if eligible_index < args.skip_documents:
            continue
        text = normalize_document(raw_text)
        selected.append({
            "document_index": len(selected),
            "source_row_index": source_index,
            "eligible_document_index": eligible_index,
            "source_document_id": str(
                row.get("id") or row.get("url") or row.get("timestamp") or ""
            ),
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        if len(selected) == args.num_documents:
            break
    if len(selected) != args.num_documents:
        raise RuntimeError(
            f"held-out stream yielded {len(selected)}/{args.num_documents} documents"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--tokenizer-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--dataset-repo", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--dataset-license", default="ODC-BY")
    parser.add_argument(
        "--model-revision", default="",
        help="Pinned upstream model commit; used if checkpoint verification lacks it",
    )
    parser.add_argument("--skip-documents", type=int, default=4096)
    parser.add_argument("--num-documents", type=int, default=32)
    parser.add_argument("--minimum-characters", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = tuple(value.strip() for value in args.labels.split(",") if value.strip())
    if labels != DEFAULT_LABELS:
        raise ValueError(f"this frozen audit requires labels={DEFAULT_LABELS}")
    by_label, plans, matched = _manifest_specs(args.checkpoint_manifest, labels)
    baseline = by_label["baseline_unpruned"]
    for spec in [baseline, *(by_label[label] for label in labels)]:
        if not Path(spec["checkpoint_dir"]).is_dir():
            raise FileNotFoundError(spec["checkpoint_dir"])
    policy = resolve_tokenizer_policy(
        args.tokenizer_audit, baseline["checkpoint_dir"], label="baseline_unpruned",
    )
    if not args.dataset_revision:
        dataset_info = HfApi().dataset_info(args.dataset_repo)
        args.dataset_revision = str(dataset_info.sha)
    documents = _heldout_documents(args)
    if args.dry_run:
        print(
            f"[routed-moe-capture] DRY RUN documents={len(documents)} "
            f"eligible_range={documents[0]['eligible_document_index']}.."
            f"{documents[-1]['eligible_document_index']} labels={len(labels)}"
        )
        return

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + f".incomplete.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        baseline["checkpoint_dir"], use_fast=True, trust_remote_code=True,
        fix_mistral_regex=policy["fix_mistral_regex"],
    )
    model, baseline_plan, shape_audit = load_checkpoint(
        baseline["checkpoint_dir"], _dtype(args.dtype),
    )
    if baseline_plan is not None or shape_audit:
        raise AssertionError("baseline capture checkpoint is unexpectedly pruned")
    model.eval()
    layers = find_decoder_layers(model)
    layer_indices = sorted(int(row["layer_idx"]) for row in next(iter(plans.values()))["layers"])
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("duplicate layer indices in pruning plan")

    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for layer_index in layer_indices:
        norm = getattr(layers[layer_index], "post_attention_layernorm", None)
        if norm is None:
            raise AttributeError(f"layer {layer_index} lacks post_attention_layernorm")

        def hook(_module, inputs, output_value, index=layer_index):
            if len(inputs) != 1:
                raise AssertionError(f"layer {index}: RMSNorm received {len(inputs)} inputs")
            captured[index] = (
                inputs[0].detach().to("cpu", copy=True),
                output_value.detach().to("cpu", copy=True),
            )

        handles.append(norm.register_forward_hook(hook))

    input_device = model.get_input_embeddings().weight.device
    manifest_documents = []
    total_tokens = 0
    maximum_native_replay_difference = 0.0
    try:
        with torch.inference_mode():
            for document in documents:
                encoded = tokenizer(
                    document["text"], return_tensors="pt", truncation=True,
                    max_length=args.max_seq_len, add_special_tokens=False,
                    padding=False,
                )
                input_ids = encoded["input_ids"]
                token_count = int(input_ids.numel())
                if token_count < 2:
                    raise ValueError(f"document {document['document_index']} has too few tokens")
                token_hash = hashlib.sha256(
                    input_ids.cpu().numpy().astype("<i8", copy=False).tobytes()
                ).hexdigest()
                captured.clear()
                model(**{key: value.to(input_device) for key, value in encoded.items()},
                      use_cache=False)
                if set(captured) != set(layer_indices):
                    raise AssertionError(
                        f"document {document['document_index']}: captured layers differ"
                    )
                shard_rows = []
                for layer_index in layer_indices:
                    residual_cpu, y_cpu = captured[layer_index]
                    mlp = layers[layer_index].mlp
                    mlp_device = next(mlp.parameters()).device
                    y = y_cpu.to(mlp_device)
                    expert_ids, routing_weights = compute_fixed_routing(mlp, y)
                    base_output = fixed_routed_moe_output(
                        mlp, y, expert_ids, routing_weights,
                    )
                    native_output = mlp(y)
                    if isinstance(native_output, tuple):
                        native_output = native_output[0]
                    native_difference = float(
                        (native_output.float() - base_output.float()).abs().max()
                    )
                    maximum_native_replay_difference = max(
                        maximum_native_replay_difference, native_difference,
                    )
                    if not torch.allclose(
                        native_output.float(), base_output.float(),
                        rtol=5e-5, atol=1e-6,
                    ):
                        raise AssertionError(
                            f"layer {layer_index}: fixed-route replay differs from "
                            f"native MoE output; max={native_difference}"
                        )
                    residual_norm = torch.linalg.vector_norm(
                        residual_cpu.float(), dim=-1,
                    ).reshape(-1)
                    base_norm = torch.linalg.vector_norm(
                        base_output.float(), dim=-1,
                    ).reshape(-1).cpu()
                    path = shard_path(
                        temporary, document["document_index"], layer_index,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "schema_version": 1,
                        "document_index": document["document_index"],
                        "layer_index": layer_index,
                        "y": y_cpu,
                        "base_output": base_output.detach().cpu(),
                        "expert_ids": expert_ids.detach().to(torch.int16).cpu(),
                        "routing_weights": routing_weights.detach().float().cpu(),
                        "residual_norm": residual_norm,
                        "base_output_norm": base_norm,
                    }, path)
                    shard_rows.append({
                        "layer_index": layer_index,
                        "path": str(path.relative_to(temporary)).replace("\\", "/"),
                        "sha256": file_sha256(str(path)),
                    })
                    del y, expert_ids, routing_weights, base_output, native_output
                total_tokens += token_count
                manifest_documents.append({
                    **{key: value for key, value in document.items() if key != "text"},
                    "token_count": token_count,
                    "token_ids_sha256": token_hash,
                    "shards": shard_rows,
                })
                print(
                    f"[routed-moe-capture] document={document['document_index'] + 1}/"
                    f"{len(documents)} tokens={token_count} layers={len(layer_indices)}"
                )
    finally:
        for handle in handles:
            handle.remove()

    verification_path = Path(baseline["checkpoint_dir"]) / "checkpoint_verification.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file() else {}
    )
    model_revision = (
        args.model_revision
        or baseline.get("model_revision")
        or verification.get("source_model_revision")
        or ""
    )
    if not model_revision:
        raise ValueError(
            "exact upstream model revision is unavailable; pass --model-revision"
        )
    manifest = {
        "schema_version": 1,
        "comparison_type": "local_same_input_fixed_route_routed_moe",
        "end_to_end_trace_used": False,
        "seed": args.seed,
        "model": baseline.get("model", "Qwen/Qwen3-30B-A3B"),
        "model_revision": model_revision,
        "baseline_checkpoint_verification_sha256": (
            file_sha256(str(verification_path)) if verification_path.is_file() else ""
        ),
        "baseline_checkpoint": str(Path(baseline["checkpoint_dir"]).resolve()),
        "checkpoint_manifest_path": str(Path(args.checkpoint_manifest).resolve()),
        "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
        "tokenizer_policy": policy,
        "dataset": {
            "repo": args.dataset_repo,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "revision": args.dataset_revision,
            "license": args.dataset_license,
            "eligible_document_offset": args.skip_documents,
            "minimum_characters": args.minimum_characters,
            "document_count": len(manifest_documents),
            "total_token_count": total_tokens,
            "max_sequence_length": args.max_seq_len,
            "preprocessing": (
                "strip raw text; require raw stripped length > minimum; then collapse whitespace; "
                "fast tokenizer; add_special_tokens=False; truncate to max_seq_len"
            ),
            "operating_point_c4_eligible_indices": [0, 1023],
            "heldout_eligible_indices": [
                manifest_documents[0]["eligible_document_index"],
                manifest_documents[-1]["eligible_document_index"],
            ],
            "disjoint_from_operating_point_evaluation": (
                manifest_documents[0]["eligible_document_index"] > 1023
            ),
            "document_set_sha256": canonical_sha256([
                row["text_sha256"] for row in manifest_documents
            ]),
        },
        "documents": manifest_documents,
        "layer_indices": layer_indices,
        "evaluated_plan_labels": list(labels),
        "plan_hashes": {
            label: by_label[label]["plan_sha256"] for label in labels
        },
        "matched_plan_validation": matched,
        "capture_tensor_storage": {
            "y": args.dtype, "base_output": args.dtype,
            "routing_weights": "float32", "norms": "float32",
        },
        "fixed_route_replay_validation": {
            "compared_against_native_unpruned_moe": True,
            "rtol": 5e-5,
            "atol": 1e-6,
            "maximum_absolute_difference": maximum_native_replay_difference,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_worktree_status": _git_value("status", "--porcelain"),
        },
    }
    manifest_path = temporary / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.move(str(temporary), str(output))
    print(
        f"[routed-moe-capture] OK documents={len(manifest_documents)} "
        f"tokens={total_tokens} layers={len(layer_indices)} output={output}"
    )


if __name__ == "__main__":
    main()
