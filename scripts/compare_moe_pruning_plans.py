#!/usr/bin/env python3
"""Compare target-2 legacy/ellipsoid pruning plans without pruning or PPL.

The plan files provide allocations and selected indices.  The score-comparison
JSON produced by ``--moe-score-comparison-only`` supplies full-scope score
statistics and Spearman correlations without storing every expert/channel row.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, Iterable, Mapping, Tuple


OLD_SELECTOR = "rmsnorm_bound"
ELLIPSOID_SELECTOR = "rmsnorm_ellipsoid_bound"


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _find_one(pattern: str, label: str) -> str:
    matches = sorted(glob.glob(pattern, recursive=True))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {label} matching {pattern!r}; found {matches}"
        )
    return matches[0]


def _plan_path(run_dir: str, experiment: str) -> str:
    return _find_one(
        os.path.join(run_dir, experiment, "pruning_plans", "*.json"),
        f"{experiment} pruning plan",
    )


def _layer_map(plan: Mapping[str, object]) -> Dict[int, dict]:
    layers: Dict[int, dict] = {}
    for row in plan.get("layers", []):
        layer_idx = int(row["layer_idx"])
        if layer_idx in layers:
            raise ValueError(f"plan repeats layer {layer_idx}")
        indices = sorted(int(value) for value in row.get("prune_idx", []))
        declared = int(row.get("pruned_channels", len(indices)))
        if declared != len(indices):
            raise ValueError(
                f"layer {layer_idx}: pruned_channels={declared}, indices={len(indices)}"
            )
        layers[layer_idx] = {**row, "prune_idx": indices}
    if not layers:
        raise ValueError("plan contains no layers")
    return layers


def _selected_identity_set(layers: Mapping[int, dict]) -> set[Tuple[int, int]]:
    return {
        (layer_idx, channel)
        for layer_idx, row in layers.items()
        for channel in row["prune_idx"]
    }


def _jaccard(left: Iterable[object], right: Iterable[object]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _plan_summary(
    plan: Mapping[str, object],
    layers: Mapping[int, dict],
    num_experts_by_layer: Mapping[int, int],
) -> dict:
    num_experts = int(plan.get("num_experts_per_layer", 0))
    total_layer_channels = sum(len(row["prune_idx"]) for row in layers.values())
    declared_total = int(plan.get("total_selected_layer_channels", -1))
    if total_layer_channels != declared_total:
        raise ValueError(
            f"plan total mismatch: declared={declared_total}, counted={total_layer_channels}"
        )
    removed_expert_neurons = sum(
        len(row["prune_idx"]) * int(num_experts_by_layer[layer_idx])
        for layer_idx, row in layers.items()
    )
    actual_pct = float(plan["actual_pct"])
    target_pct = float(plan["target_pct"])
    return {
        "selector": plan.get("selector"),
        "target_pct": target_pct,
        "actual_pct": actual_pct,
        "overshot_requested_target": actual_pct > target_pct + 1e-12,
        "total_layer_channels_removed": total_layer_channels,
        "num_experts_per_layer": num_experts,
        "total_expert_neurons_removed": removed_expert_neurons,
        "layers_pruned": sum(bool(row["prune_idx"]) for row in layers.values()),
        "layers_hitting_64_channel_cap": sum(
            len(row["prune_idx"]) == 64 for row in layers.values()
        ),
        "per_layer_removed_channel_counts": {
            str(layer_idx): len(row["prune_idx"])
            for layer_idx, row in sorted(layers.items())
        },
    }


def _score_metrics(
    score_payload: Mapping[str, object],
) -> tuple[dict, Dict[int, dict], Dict[int, int]]:
    if score_payload.get("legacy_selector") != OLD_SELECTOR:
        raise ValueError("score JSON is not for rmsnorm_bound")
    if score_payload.get("new_selector") != ELLIPSOID_SELECTOR:
        raise ValueError("score JSON is not for rmsnorm_ellipsoid_bound")
    global_metrics = dict(score_payload.get("global", {}).get("metrics", {}))
    layer_metrics = {
        int(row["layer_idx"]): dict(row.get("metrics", {}))
        for row in score_payload.get("layers", [])
    }
    num_experts_by_layer = {
        int(row["layer_idx"]): int(row["num_experts"])
        for row in score_payload.get("layers", [])
    }
    return global_metrics, layer_metrics, num_experts_by_layer


def compare_plans(old_plan_path: str, ellipsoid_plan_path: str, score_json_path: str) -> dict:
    old_plan = _load_json(old_plan_path)
    ellipsoid_plan = _load_json(ellipsoid_plan_path)
    score_payload = _load_json(score_json_path)
    if old_plan.get("selector") != OLD_SELECTOR:
        raise ValueError(f"old plan selector is {old_plan.get('selector')!r}")
    if ellipsoid_plan.get("selector") != ELLIPSOID_SELECTOR:
        raise ValueError(
            f"ellipsoid plan selector is {ellipsoid_plan.get('selector')!r}"
        )
    for field in ("model_id", "target_pct", "pruning_mode", "aggregation_mode", "channel_alignment"):
        if old_plan.get(field) != ellipsoid_plan.get(field):
            raise ValueError(
                f"plan field {field!r} differs: old={old_plan.get(field)!r}, "
                f"ellipsoid={ellipsoid_plan.get(field)!r}"
            )

    old_layers = _layer_map(old_plan)
    ellipsoid_layers = _layer_map(ellipsoid_plan)
    if set(old_layers) != set(ellipsoid_layers):
        raise ValueError("plan layer sets differ")
    (
        global_score_metrics,
        layer_score_metrics,
        num_experts_by_layer,
    ) = _score_metrics(score_payload)
    if set(layer_score_metrics) != set(old_layers):
        raise ValueError(
            "score-comparison layer set does not match plan layer set; rerun "
            "score comparison with the same full model"
        )

    old_nonzero = {li for li, row in old_layers.items() if row["prune_idx"]}
    ellipsoid_nonzero = {
        li for li, row in ellipsoid_layers.items() if row["prune_idx"]
    }
    shared = old_nonzero & ellipsoid_nonzero
    per_layer = []
    for layer_idx in sorted(old_layers):
        old_indices = old_layers[layer_idx]["prune_idx"]
        ellipsoid_indices = ellipsoid_layers[layer_idx]["prune_idx"]
        overlap = sorted(set(old_indices) & set(ellipsoid_indices))
        metrics = layer_score_metrics[layer_idx]
        num_experts = num_experts_by_layer[layer_idx]
        per_layer.append({
            "layer_idx": layer_idx,
            "num_experts": num_experts,
            "old_removed_channels": len(old_indices),
            "ellipsoid_removed_channels": len(ellipsoid_indices),
            "old_removed_expert_neurons": len(old_indices) * num_experts,
            "ellipsoid_removed_expert_neurons": (
                len(ellipsoid_indices) * num_experts
            ),
            "allocation_membership": (
                "both" if layer_idx in shared else
                "old_only" if layer_idx in old_nonzero else
                "ellipsoid_only" if layer_idx in ellipsoid_nonzero else
                "neither"
            ),
            "selected_channel_overlap_count": len(overlap),
            "selected_channel_overlap_indices": overlap,
            "selected_channel_jaccard": _jaccard(old_indices, ellipsoid_indices),
            "spearman_old_vs_ellipsoid": metrics.get(
                "spearman_ellipsoid_vs_legacy"
            ),
            "old_score_min": metrics.get("legacy_min"),
            "old_score_median": metrics.get("legacy_median"),
            "old_score_p95": metrics.get("legacy_p95"),
            "old_score_max": metrics.get("legacy_max"),
            "ellipsoid_score_min": metrics.get("ellipsoid_min"),
            "ellipsoid_score_median": metrics.get("ellipsoid_median"),
            "ellipsoid_score_p95": metrics.get("ellipsoid_p95"),
            "ellipsoid_score_max": metrics.get("ellipsoid_max"),
            "ellipsoid_to_old_ratio_min": metrics.get(
                "ellipsoid_to_legacy_ratio_min"
            ),
            "ellipsoid_to_old_ratio_median": metrics.get(
                "ellipsoid_to_legacy_ratio_median"
            ),
            "ellipsoid_to_old_ratio_p95": metrics.get(
                "ellipsoid_to_legacy_ratio_p95"
            ),
            "ellipsoid_to_old_ratio_max": metrics.get(
                "ellipsoid_to_legacy_ratio_max"
            ),
        })

    old_global = _selected_identity_set(old_layers)
    ellipsoid_global = _selected_identity_set(ellipsoid_layers)
    return {
        "old_plan_path": old_plan_path,
        "ellipsoid_plan_path": ellipsoid_plan_path,
        "score_comparison_json_path": score_json_path,
        "plan_invariants": {
            field: old_plan.get(field)
            for field in (
                "model_id", "target_pct", "pruning_mode",
                "aggregation_mode", "channel_alignment", "max_layer_frac",
            )
        },
        "old_plan": _plan_summary(
            old_plan, old_layers, num_experts_by_layer
        ),
        "ellipsoid_plan": _plan_summary(
            ellipsoid_plan, ellipsoid_layers, num_experts_by_layer
        ),
        "layers_selected_only_by_old": sorted(old_nonzero - ellipsoid_nonzero),
        "layers_selected_only_by_ellipsoid": sorted(ellipsoid_nonzero - old_nonzero),
        "layers_selected_by_both": sorted(shared),
        "global_selected_channel_overlap_count": len(old_global & ellipsoid_global),
        "global_selected_channel_jaccard": _jaccard(old_global, ellipsoid_global),
        "global_spearman_old_vs_ellipsoid": global_score_metrics.get(
            "spearman_ellipsoid_vs_legacy"
        ),
        "global_score_metrics": global_score_metrics,
        "per_layer": per_layer,
        "summary_dimension_field_note": (
            "old_moe_dim/new_moe_dim in selector_baseline_summary.csv come from "
            "the experiment row's _old_inter_summary/_new_inter_summary values. "
            "For packed global pruning, old_inter is initialized from the first "
            "physically pruned layer and new_inter is overwritten for each pruned "
            "layer, so the final new value is the last pruned layer's width in "
            "iteration order. They are not the heterogeneous model-wide width "
            "distribution; the plan and per-layer CSV are authoritative."
        ),
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--score-comparison-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    old_plan_path = _plan_path(args.run_dir, "rmsnorm_bound_target2")
    ellipsoid_plan_path = _plan_path(
        args.run_dir, "rmsnorm_ellipsoid_bound_target2"
    )
    report = compare_plans(
        old_plan_path, ellipsoid_plan_path, args.score_comparison_json
    )
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "target2_plan_comparison.json")
    csv_path = os.path.join(args.output_dir, "target2_plan_comparison_per_layer.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _write_csv(csv_path, report["per_layer"])

    print("[plan-compare] old plan       :", report["old_plan_path"])
    print("[plan-compare] ellipsoid plan :", report["ellipsoid_plan_path"])
    for label in ("old_plan", "ellipsoid_plan"):
        row = report[label]
        print(
            f"[plan-compare] {label}: actual={row['actual_pct']:.4f}% "
            f"layer_channels={row['total_layer_channels_removed']} "
            f"expert_neurons={row['total_expert_neurons_removed']} "
            f"cap64_layers={row['layers_hitting_64_channel_cap']} "
            f"overshot={row['overshot_requested_target']}"
        )
    print(
        "[plan-compare] global selected-channel Jaccard:",
        f"{report['global_selected_channel_jaccard']:.6f}",
    )
    print(
        "[plan-compare] global score Spearman:",
        report["global_spearman_old_vs_ellipsoid"],
    )
    print("[plan-compare] JSON:", json_path)
    print("[plan-compare] CSV :", csv_path)


if __name__ == "__main__":
    main()
