#!/usr/bin/env python3
"""Evaluate one physical checkpoint against an immutable same-input capture."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts.run_paper_v3_lm_eval import load_checkpoint
from src.experiment_provenance import file_sha256
from src.heterogeneous_moe_checkpoint import find_decoder_layers
from src.moe_set_certification import (
    certificate_for_plan,
    load_score_bundle,
    selected_by_layer,
)
from src.routed_moe_perturbation import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    distribution,
    expert_set_bounds,
    fixed_routed_moe_output,
    route_conditioned_bounds,
    safe_ratio,
    violation_mask,
)


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(prefix: dict, arrays: dict[str, np.ndarray]) -> dict:
    row = dict(prefix)
    for metric, values in arrays.items():
        for statistic, value in distribution(values).items():
            row[f"{metric}_{statistic}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--score-bundle", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--relative-tolerance", type=float,
                        default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--absolute-tolerance", type=float,
                        default=DEFAULT_ABSOLUTE_TOLERANCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    capture_root = Path(args.capture_dir)
    capture_path = capture_root / "capture_manifest.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if capture.get("comparison_type") != "local_same_input_fixed_route_routed_moe":
        raise ValueError("capture is not a local same-input fixed-route audit")
    if capture.get("end_to_end_trace_used") is not False:
        raise ValueError("capture unexpectedly contains an end-to-end trace")
    if args.label not in capture["evaluated_plan_labels"]:
        raise ValueError(f"label not authorized by capture: {args.label}")

    specs = json.loads(Path(args.checkpoint_manifest).read_text(encoding="utf-8"))
    matches = [row for row in specs if row["label"] == args.label]
    if len(matches) != 1:
        raise ValueError(f"checkpoint spec matches={len(matches)} for {args.label}")
    spec = matches[0]
    checkpoint = Path(spec["checkpoint_dir"])
    plan_path = Path(spec["plan_path"])
    if not checkpoint.is_dir() or not plan_path.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_dir() else plan_path)
    plan_hash = file_sha256(str(plan_path))
    if plan_hash != spec["plan_sha256"]:
        raise ValueError("source plan hash differs from checkpoint manifest")
    if capture["plan_hashes"][args.label] != plan_hash:
        raise ValueError("source plan hash differs from immutable capture")
    source_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sum(len(row["prune_idx"]) for row in source_plan["layers"]) != 2288:
        raise ValueError("source plan is not the frozen 2,288-channel target-6 plan")
    score_manifest = json.loads(Path(args.score_manifest).read_text(encoding="utf-8"))
    recorded_bundle_hash = (
        score_manifest.get("npz_sha256")
        or score_manifest.get("score_bundle_sha256")
        or score_manifest.get("bundle_sha256")
        or score_manifest.get("score_npz_sha256")
    )
    score_hash = file_sha256(args.score_bundle)
    if recorded_bundle_hash and recorded_bundle_hash != score_hash:
        raise ValueError("score bundle hash differs from score manifest")
    if score_manifest.get("model") != capture.get("model"):
        raise ValueError("score bundle model differs from baseline capture")
    if (
        score_manifest.get("model_revision")
        and score_manifest["model_revision"] != capture.get("model_revision")
    ):
        raise ValueError("score bundle model revision differs from baseline capture")
    scores = load_score_bundle(args.score_bundle)
    certificate = certificate_for_plan(source_plan, scores)

    if args.dry_run:
        print(
            f"[routed-moe-eval] DRY RUN label={args.label} "
            f"documents={len(capture['documents'])} layers={len(capture['layer_indices'])} "
            f"C={certificate['strict_global_unpropagated_certificate']:.9g}"
        )
        return

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + f".incomplete.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)

    model, loaded_plan, shape_audit = load_checkpoint(str(checkpoint), _dtype(args.dtype))
    if loaded_plan is None:
        raise AssertionError("physical pruned checkpoint did not load its pruning plan")
    if selected_by_layer(loaded_plan) != selected_by_layer(source_plan):
        raise AssertionError("physical checkpoint channel identities differ from source plan")
    if not shape_audit or not all(row["no_original_width_padding"] for row in shape_audit):
        raise AssertionError("physical checkpoint shape audit failed")
    model.eval()
    layers = find_decoder_layers(model)
    selected = selected_by_layer(source_plan)
    certificate_layers = {
        int(row["layer_idx"]): row for row in certificate["layers"]
    }

    columns: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "document_index", "layer_index", "token_index", "actual_perturbation",
            "strict_bound", "route_conditioned_bound", "strict_bound_ratio",
            "route_conditioned_bound_ratio", "perturbation_over_base_moe_norm",
            "perturbation_over_residual_norm", "strict_violation", "route_violation",
        )
    }
    layer_buckets: dict[int, dict[str, list[np.ndarray]]] = {}
    document_buckets: dict[int, dict[str, list[np.ndarray]]] = {}
    metric_names = (
        "actual_perturbation", "strict_bound_ratio",
        "route_conditioned_bound_ratio", "perturbation_over_base_moe_norm",
        "perturbation_over_residual_norm",
    )
    strict_violation_count = 0
    route_violation_count = 0
    max_strict_excess = 0.0
    max_route_excess = 0.0

    with torch.inference_mode():
        for document in capture["documents"]:
            document_index = int(document["document_index"])
            document_buckets[document_index] = {key: [] for key in metric_names}
            for shard_record in document["shards"]:
                layer_index = int(shard_record["layer_index"])
                path = capture_root / shard_record["path"]
                if file_sha256(str(path)) != shard_record["sha256"]:
                    raise ValueError(f"capture shard hash mismatch: {path}")
                shard = torch.load(path, map_location="cpu", weights_only=True)
                if int(shard["document_index"]) != document_index:
                    raise ValueError(f"capture document mismatch: {path}")
                if int(shard["layer_index"]) != layer_index:
                    raise ValueError(f"capture layer mismatch: {path}")
                mlp = layers[layer_index].mlp
                device = next(mlp.parameters()).device
                y = shard["y"].to(device)
                expert_ids = shard["expert_ids"].long().to(device)
                weights = shard["routing_weights"].to(device)
                pruned_output = fixed_routed_moe_output(
                    mlp, y, expert_ids, weights,
                ).detach().float().cpu()
                base_output = shard["base_output"].float()
                actual = torch.linalg.vector_norm(
                    base_output - pruned_output, dim=-1,
                ).reshape(-1).numpy().astype(np.float64)
                expert_sums = expert_set_bounds(
                    scores[layer_index]["ellipsoid"], selected[layer_index],
                )
                strict = float(expert_sums.max(initial=0.0))
                route = route_conditioned_bounds(
                    expert_sums,
                    shard["expert_ids"].reshape(-1, shard["expert_ids"].shape[-1]).numpy(),
                    shard["routing_weights"].reshape(
                        -1, shard["routing_weights"].shape[-1]
                    ).numpy(),
                )
                strict_ratio = safe_ratio(actual, strict)
                route_ratio = safe_ratio(actual, route)
                base_relative = safe_ratio(
                    actual, shard["base_output_norm"].reshape(-1).numpy(),
                )
                residual_relative = safe_ratio(
                    actual, shard["residual_norm"].reshape(-1).numpy(),
                )
                strict_violations = violation_mask(
                    actual, strict, relative_tolerance=args.relative_tolerance,
                    absolute_tolerance=args.absolute_tolerance,
                )
                route_violations = violation_mask(
                    actual, route, relative_tolerance=args.relative_tolerance,
                    absolute_tolerance=args.absolute_tolerance,
                )
                strict_violation_count += int(strict_violations.sum())
                route_violation_count += int(route_violations.sum())
                max_strict_excess = max(
                    max_strict_excess,
                    float(np.maximum(actual - strict, 0.0).max(initial=0.0)),
                )
                max_route_excess = max(
                    max_route_excess,
                    float(np.maximum(actual - route, 0.0).max(initial=0.0)),
                )
                values = {
                    "actual_perturbation": actual,
                    "strict_bound_ratio": strict_ratio,
                    "route_conditioned_bound_ratio": route_ratio,
                    "perturbation_over_base_moe_norm": base_relative,
                    "perturbation_over_residual_norm": residual_relative,
                }
                layer_buckets.setdefault(
                    layer_index, {key: [] for key in metric_names}
                )
                for key, value in values.items():
                    layer_buckets[layer_index][key].append(value)
                    document_buckets[document_index][key].append(value)
                count = actual.size
                columns["document_index"].append(np.full(count, document_index, np.int16))
                columns["layer_index"].append(np.full(count, layer_index, np.int16))
                columns["token_index"].append(np.arange(count, dtype=np.int16))
                columns["actual_perturbation"].append(actual.astype(np.float32))
                columns["strict_bound"].append(np.full(count, strict, np.float64))
                columns["route_conditioned_bound"].append(route.astype(np.float64))
                columns["strict_bound_ratio"].append(strict_ratio.astype(np.float64))
                columns["route_conditioned_bound_ratio"].append(route_ratio.astype(np.float64))
                columns["perturbation_over_base_moe_norm"].append(base_relative.astype(np.float64))
                columns["perturbation_over_residual_norm"].append(residual_relative.astype(np.float64))
                columns["strict_violation"].append(strict_violations.astype(np.uint8))
                columns["route_violation"].append(route_violations.astype(np.uint8))
            print(
                f"[routed-moe-eval] {args.label} document={document_index + 1}/"
                f"{len(capture['documents'])}"
            )

    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    token_path = temporary / "token_observations.npz"
    np.savez_compressed(token_path, **arrays)
    layer_rows = []
    for layer_index in sorted(layer_buckets):
        layer_arrays = {
            key: np.concatenate(values)
            for key, values in layer_buckets[layer_index].items()
        }
        cert_row = certificate_layers[layer_index]
        layer_rows.append(_metric_row({
            "label": args.label,
            "layer_index": layer_index,
            "removed_channels": len(selected[layer_index]),
            "strict_set_bound": cert_row["strict_max_expert_sum"],
            "p95_expert_set_bound": cert_row["p95_expert_sum"],
            "mean_expert_set_bound": cert_row["mean_expert_sum"],
            "older_channelwise_max_bound": cert_row[
                "older_sum_channelwise_expert_max"
            ],
            "normalized_down_norm_objective": cert_row[
                "normalized_down_norm_objective"
            ],
        }, layer_arrays))
    document_rows = []
    for document_index in sorted(document_buckets):
        doc_arrays = {
            key: np.concatenate(values)
            for key, values in document_buckets[document_index].items()
        }
        document_rows.append({
            "label": args.label,
            "document_index": document_index,
            "text_sha256": capture["documents"][document_index]["text_sha256"],
            **{
                f"mean_{key}": float(value.mean())
                for key, value in doc_arrays.items()
            },
        })
    _write_csv(temporary / "layer_summary.csv", layer_rows)
    _write_csv(temporary / "document_summary.csv", document_rows)
    overall_metrics = {
        key: distribution(arrays[key]) for key in metric_names
    }
    result = {
        "schema_version": 1,
        "label": args.label,
        "comparison_type": "local_same_input_fixed_route_routed_moe",
        "end_to_end_trace_used": False,
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
        "source_plan_path": str(plan_path.resolve()),
        "source_plan_sha256": plan_hash,
        "score_bundle_path": str(Path(args.score_bundle).resolve()),
        "score_bundle_sha256": score_hash,
        "score_manifest_sha256": file_sha256(args.score_manifest),
        "capture_manifest_sha256": file_sha256(str(capture_path)),
        "corpus": capture["dataset"],
        "documents": len(capture["documents"]),
        "tokens_per_layer": int(sum(row["token_count"] for row in capture["documents"])),
        "evaluated_layers": len(capture["layer_indices"]),
        "token_layer_observations": int(arrays["actual_perturbation"].size),
        "bound_evaluation_unit": "one routed-MoE output perturbation per token and layer",
        "removed_layer_channels": 2288,
        "actual_pruning_percentage": float(spec.get("actual_pct", source_plan.get("actual_pct", 0.0))),
        "strict_global_unpropagated_certificate_C": certificate[
            "strict_global_unpropagated_certificate"
        ],
        "normalized_down_norm_objective": certificate[
            "normalized_down_norm_objective"
        ],
        "set_level_certificate": certificate,
        "overall": overall_metrics,
        "numerical_tolerance": {
            "relative": args.relative_tolerance,
            "absolute": args.absolute_tolerance,
            "violation_rule": "actual > bound*(1+relative)+absolute",
        },
        "strict_bound_violations": strict_violation_count,
        "maximum_strict_bound_excess": max_strict_excess,
        "route_conditioned_bound_violations": route_violation_count,
        "maximum_route_conditioned_bound_excess": max_route_excess,
        "shape_audit": shape_audit,
        "no_hidden_original_width_padding": all(
            row["no_original_width_padding"] for row in shape_audit
        ),
        "artifacts": {
            "token_observations": "token_observations.npz",
            "token_observations_sha256": file_sha256(str(token_path)),
            "layer_summary": "layer_summary.csv",
            "document_summary": "document_summary.csv",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    (temporary / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    shutil.move(str(temporary), str(output))
    if strict_violation_count or route_violation_count:
        raise AssertionError(
            f"bound violations: strict={strict_violation_count} "
            f"route={route_violation_count}; result retained at {output}"
        )
    print(
        f"[routed-moe-eval] OK label={args.label} observations="
        f"{arrays['actual_perturbation'].size} strict_max_ratio="
        f"{overall_metrics['strict_bound_ratio']['max']:.8g} route_max_ratio="
        f"{overall_metrics['route_conditioned_bound_ratio']['max']:.8g}"
    )


if __name__ == "__main__":
    main()
