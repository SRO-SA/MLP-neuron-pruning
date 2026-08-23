"""Strict set-level MoE certificates and deterministic hybrid refinement.

The expert-channel ellipsoid scores are mathematical local bounds.  This
module only combines those already-certified local quantities; it does not
claim end-to-end Transformer propagation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_TOLERANCE = 1e-8


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def plan_layer_map(plan: dict) -> dict[int, dict]:
    rows = {int(row["layer_idx"]): row for row in plan.get("layers", [])}
    if len(rows) != len(plan.get("layers", [])):
        raise ValueError("pruning plan contains duplicate layer indices")
    if not rows:
        raise ValueError("pruning plan contains no layers")
    return rows


def selected_by_layer(plan: dict) -> dict[int, tuple[int, ...]]:
    result = {}
    for layer_idx, row in plan_layer_map(plan).items():
        selected = tuple(sorted(int(value) for value in row.get("prune_idx", [])))
        if len(selected) != len(set(selected)):
            raise ValueError(f"layer {layer_idx} has duplicate selected indices")
        old_width = int(row["old_intermediate"])
        if any(value < 0 or value >= old_width for value in selected):
            raise ValueError(f"layer {layer_idx} has out-of-range selected indices")
        if int(row.get("pruned_channels", len(selected))) != len(selected):
            raise ValueError(f"layer {layer_idx} pruned_channels is inconsistent")
        if int(row["new_intermediate"]) != old_width - len(selected):
            raise ValueError(f"layer {layer_idx} new_intermediate is inconsistent")
        result[layer_idx] = selected
    return result


def matched_plan_validation(
    plans: Mapping[str, dict],
    *,
    expected_total: int = 2288,
    expected_alignment: int = 16,
) -> dict:
    """Validate that plans differ only in within-layer channel identities."""
    if len(plans) < 2:
        raise ValueError("at least two plans are required")
    labels = list(plans)
    reference_label = labels[0]
    reference = plans[reference_label]
    reference_rows = plan_layer_map(reference)
    reference_selected = selected_by_layer(reference)
    reference_counts = {
        layer: len(values) for layer, values in reference_selected.items()
    }
    total = sum(reference_counts.values())
    if total != expected_total:
        raise ValueError(f"reference total={total}, expected={expected_total}")

    invariant_fields = (
        "model_id", "pruning_mode", "aggregation_mode", "channel_alignment",
        "max_layer_frac", "num_experts_per_layer", "target_pct", "actual_pct",
        "transformers_version", "torch_version", "max_expert_frac",
    )
    comparisons = []
    for label, plan in plans.items():
        rows = plan_layer_map(plan)
        selected = selected_by_layer(plan)
        counts = {layer: len(values) for layer, values in selected.items()}
        if set(rows) != set(reference_rows):
            raise ValueError(f"{label}: layer set differs from {reference_label}")
        if counts != reference_counts:
            raise ValueError(f"{label}: per-layer allocation differs")
        if sum(counts.values()) != expected_total:
            raise ValueError(f"{label}: total removed channels differs")
        for field in invariant_fields:
            if plan.get(field) != reference.get(field):
                raise ValueError(f"{label}: invariant {field} differs")
        if int(plan.get("channel_alignment", -1)) != expected_alignment:
            raise ValueError(f"{label}: channel alignment differs")
        for layer_idx, row in rows.items():
            ref = reference_rows[layer_idx]
            for field in ("old_intermediate", "new_intermediate"):
                if int(row[field]) != int(ref[field]):
                    raise ValueError(f"{label}: layer {layer_idx} {field} differs")
            if len(selected[layer_idx]) % expected_alignment != 0:
                raise ValueError(f"{label}: layer {layer_idx} violates alignment")
            cap = int(float(plan["max_layer_frac"]) * int(row["old_intermediate"]))
            if len(selected[layer_idx]) > cap:
                raise ValueError(
                    f"{label}: layer {layer_idx} exceeds max-layer cap {cap}"
                )
            if int(row["new_intermediate"]) <= 0:
                raise ValueError(f"{label}: layer {layer_idx} retains no channels")

        same = sum(
            len(set(selected[layer]) & set(reference_selected[layer]))
            for layer in selected
        )
        union = sum(
            len(set(selected[layer]) | set(reference_selected[layer]))
            for layer in selected
        )
        comparisons.append({
            "label": label,
            "selector": plan.get("selector"),
            "same_selected_as_reference": same,
            "global_selected_jaccard": float(same / union) if union else 1.0,
        })

    num_experts = int(reference.get("num_experts_per_layer", 1))
    declared_total = int(reference.get("total_selected_layer_channels", total))
    if declared_total != total:
        raise ValueError("reference declared selected-channel total is inconsistent")
    return {
        "validation_passed": True,
        "reference_label": reference_label,
        "labels": labels,
        "invariant_fields": list(invariant_fields),
        "channel_alignment": expected_alignment,
        "total_removed_layer_channels": total,
        "total_removed_expert_neurons": total * num_experts,
        "per_layer_counts": {str(key): value for key, value in reference_counts.items()},
        "minimum_retained_width": min(
            int(row["new_intermediate"]) for row in reference_rows.values()
        ),
        "comparisons": comparisons,
    }


def load_score_bundle(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Load a compact all-expert score bundle produced by the collector."""
    bundle: dict[int, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if not key.startswith("layer_"):
                continue
            prefix, metric = key.rsplit("__", 1)
            layer_idx = int(prefix.removeprefix("layer_"))
            bundle.setdefault(layer_idx, {})[metric] = np.asarray(archive[key])
    for layer_idx, metrics in bundle.items():
        if set(metrics) != {"ellipsoid", "down_norm"}:
            raise ValueError(f"layer {layer_idx}: incomplete score metrics {set(metrics)}")
        if metrics["ellipsoid"].ndim != 2:
            raise ValueError(f"layer {layer_idx}: ellipsoid scores must be [expert,channel]")
        if metrics["down_norm"].shape != metrics["ellipsoid"].shape:
            raise ValueError(f"layer {layer_idx}: down-norm shape differs")
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError(f"layer {layer_idx}: score bundle contains non-finite data")
        if any((value < 0).any() for value in metrics.values()):
            raise ValueError(f"layer {layer_idx}: score bundle contains negative bounds")
    if not bundle:
        raise ValueError(f"score bundle is empty: {path}")
    return bundle


def normalized_down_norm(scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Return layer-normalized p95 down norms and the recorded scale."""
    aggregated = np.quantile(scores.astype(np.float64), 0.95, axis=0, method="linear")
    scale = float(np.max(aggregated))
    if scale <= 0.0:
        return np.zeros_like(aggregated), scale
    return aggregated / scale, scale


def certificate_for_selection(
    selected: Mapping[int, tuple[int, ...] | list[int] | set[int]],
    scores: Mapping[int, Mapping[str, np.ndarray]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Compute strict set-level and conservative channelwise certificates."""
    layer_rows = []
    expert_rows = []
    strict_total = 0.0
    old_total = 0.0
    objective_total = 0.0
    violations = 0
    max_violation = 0.0
    for layer_idx in sorted(selected):
        if layer_idx not in scores:
            raise ValueError(f"layer {layer_idx} is missing from score bundle")
        bounds = np.asarray(scores[layer_idx]["ellipsoid"], dtype=np.float64)
        down = np.asarray(scores[layer_idx]["down_norm"], dtype=np.float64)
        indices = np.asarray(sorted(int(value) for value in selected[layer_idx]), dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= bounds.shape[1]):
            raise ValueError(f"layer {layer_idx}: selected index out of score range")
        expert_sums = bounds[:, indices].sum(axis=1, dtype=np.float64)
        strict = float(expert_sums.max(initial=0.0))
        old = float(bounds[:, indices].max(axis=0).sum(dtype=np.float64)) if indices.size else 0.0
        allowed = tolerance * max(1.0, abs(old))
        violation = strict - old
        if violation > allowed:
            violations += 1
            max_violation = max(max_violation, violation)
        normalized, scale = normalized_down_norm(down)
        objective = float(normalized[indices].sum(dtype=np.float64))
        strict_total += strict
        old_total += old
        objective_total += objective
        layer_rows.append({
            "layer_idx": layer_idx,
            "removed_channels": int(indices.size),
            "num_experts": int(bounds.shape[0]),
            "strict_max_expert_sum": strict,
            "p95_expert_sum": _quantile(expert_sums, 0.95),
            "mean_expert_sum": float(expert_sums.mean()) if expert_sums.size else 0.0,
            "median_expert_sum": _quantile(expert_sums, 0.5),
            "min_expert_sum": float(expert_sums.min()) if expert_sums.size else 0.0,
            "older_sum_channelwise_expert_max": old,
            "strict_over_older": float(strict / old) if old else 0.0,
            "down_norm_scale": scale,
            "normalized_down_norm_objective": objective,
        })
        for expert_idx, value in enumerate(expert_sums.tolist()):
            expert_rows.append({
                "layer_idx": layer_idx,
                "expert_idx": expert_idx,
                "selected_set_bound": float(value),
            })
    if violations:
        raise AssertionError(
            f"strict set certificate exceeded older expression in {violations} layers; "
            f"max excess={max_violation}"
        )
    all_expert_values = np.asarray(
        [row["selected_set_bound"] for row in expert_rows], dtype=np.float64
    )
    strict_layer_values = np.asarray(
        [row["strict_max_expert_sum"] for row in layer_rows], dtype=np.float64
    )
    return {
        "strict_global_unpropagated_certificate": strict_total,
        "older_global_channelwise_max_certificate": old_total,
        "strict_over_older": strict_total / old_total if old_total else 0.0,
        "normalized_down_norm_objective": objective_total,
        "numerical_tolerance": tolerance,
        "inequality_violations": violations,
        "maximum_inequality_violation": max_violation,
        "distribution_across_layer_expert_sets": {
            "count": int(all_expert_values.size),
            "min": float(all_expert_values.min()) if all_expert_values.size else 0.0,
            "median": _quantile(all_expert_values, 0.5) if all_expert_values.size else 0.0,
            "mean": float(all_expert_values.mean()) if all_expert_values.size else 0.0,
            "p95": _quantile(all_expert_values, 0.95) if all_expert_values.size else 0.0,
            "max": float(all_expert_values.max()) if all_expert_values.size else 0.0,
        },
        "distribution_across_strict_layer_certificates": {
            "count": int(strict_layer_values.size),
            "min": float(strict_layer_values.min()) if strict_layer_values.size else 0.0,
            "median": _quantile(strict_layer_values, 0.5) if strict_layer_values.size else 0.0,
            "mean": float(strict_layer_values.mean()) if strict_layer_values.size else 0.0,
            "p95": _quantile(strict_layer_values, 0.95) if strict_layer_values.size else 0.0,
            "max": float(strict_layer_values.max()) if strict_layer_values.size else 0.0,
        },
        "layers": layer_rows,
        "expert_set_bounds": expert_rows,
    }


def certificate_for_plan(
    plan: dict,
    scores: Mapping[int, Mapping[str, np.ndarray]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    return certificate_for_selection(selected_by_layer(plan), scores, tolerance=tolerance)


@dataclass(frozen=True)
class SwapCandidate:
    layer_idx: int
    remove_selected: int
    add_selected: int
    objective_improvement: float
    initial_certificate_delta: float


def _selection_digest(selected: Mapping[int, set[int]]) -> str:
    payload = json.dumps(
        {str(key): sorted(value) for key, value in sorted(selected.items())},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def refine_with_certificate_slack(
    ellipsoid_plan: dict,
    down_norm_plan: dict,
    scores: Mapping[int, Mapping[str, np.ndarray]],
    rho: float,
    *,
    seed: int = 42,
    tolerance: float = DEFAULT_TOLERANCE,
    max_passes: int = 16,
) -> tuple[dict[int, set[int]], dict]:
    """Greedily move toward the down-norm endpoint under a strict budget.

    There is no random sampling.  ``seed`` is recorded for protocol stability;
    ties are resolved by certificate class, efficiency, objective improvement,
    layer index, removed ID, then added ID.
    """
    if rho < 0:
        raise ValueError("certificate slack must be non-negative")
    validation = matched_plan_validation(
        {"ellipsoid": ellipsoid_plan, "down_norm": down_norm_plan},
        expected_total=sum(len(v) for v in selected_by_layer(ellipsoid_plan).values()),
        expected_alignment=int(ellipsoid_plan["channel_alignment"]),
    )
    start = {layer: set(values) for layer, values in selected_by_layer(ellipsoid_plan).items()}
    endpoint = {layer: set(values) for layer, values in selected_by_layer(down_norm_plan).items()}
    base_report = certificate_for_selection(start, scores, tolerance=tolerance)
    base_certificate = float(base_report["strict_global_unpropagated_certificate"])
    threshold = (1.0 + float(rho)) * base_certificate

    current = {layer: set(values) for layer, values in start.items()}
    expert_sums = {}
    layer_strict = {}
    normalized_scores = {}
    for layer_idx, selected in current.items():
        bounds = np.asarray(scores[layer_idx]["ellipsoid"], dtype=np.float64)
        idx = np.asarray(sorted(selected), dtype=np.int64)
        sums = bounds[:, idx].sum(axis=1, dtype=np.float64)
        expert_sums[layer_idx] = sums
        layer_strict[layer_idx] = float(sums.max(initial=0.0))
        normalized_scores[layer_idx] = normalized_down_norm(
            np.asarray(scores[layer_idx]["down_norm"])
        )[0]
    current_certificate = float(sum(layer_strict.values()))

    candidates = []
    for layer_idx in sorted(current):
        bounds = np.asarray(scores[layer_idx]["ellipsoid"], dtype=np.float64)
        norm_down = normalized_scores[layer_idx]
        for removed in sorted(start[layer_idx] - endpoint[layer_idx]):
            for added in sorted(endpoint[layer_idx] - start[layer_idx]):
                improvement = float(norm_down[removed] - norm_down[added])
                if improvement <= tolerance:
                    continue
                proposed = expert_sums[layer_idx] - bounds[:, removed] + bounds[:, added]
                delta = float(proposed.max(initial=0.0) - layer_strict[layer_idx])
                candidates.append(SwapCandidate(
                    layer_idx, removed, added, improvement, delta
                ))

    def sort_key(candidate: SwapCandidate):
        delta = candidate.initial_certificate_delta
        if delta <= 0:
            return (0, -candidate.objective_improvement, candidate.layer_idx,
                    candidate.remove_selected, candidate.add_selected)
        efficiency = candidate.objective_improvement / delta
        return (1, -efficiency, -candidate.objective_improvement,
                candidate.layer_idx, candidate.remove_selected,
                candidate.add_selected)

    candidates.sort(key=sort_key)
    accepted = []
    for pass_index in range(max_passes):
        accepted_this_pass = 0
        for candidate in candidates:
            layer_idx = candidate.layer_idx
            if candidate.remove_selected not in current[layer_idx]:
                continue
            if candidate.add_selected in current[layer_idx]:
                continue
            bounds = np.asarray(scores[layer_idx]["ellipsoid"], dtype=np.float64)
            proposed_sums = (
                expert_sums[layer_idx]
                - bounds[:, candidate.remove_selected]
                + bounds[:, candidate.add_selected]
            )
            proposed_strict = float(proposed_sums.max(initial=0.0))
            proposed_global = (
                current_certificate - layer_strict[layer_idx] + proposed_strict
            )
            if proposed_global > threshold + tolerance * max(1.0, threshold):
                continue
            before = current_certificate
            current[layer_idx].remove(candidate.remove_selected)
            current[layer_idx].add(candidate.add_selected)
            expert_sums[layer_idx] = proposed_sums
            layer_strict[layer_idx] = proposed_strict
            current_certificate = proposed_global
            accepted.append({
                "pass": pass_index,
                "layer_idx": layer_idx,
                "removed_from_prune_set": candidate.remove_selected,
                "added_to_prune_set": candidate.add_selected,
                "objective_improvement": candidate.objective_improvement,
                "strict_certificate_before": before,
                "strict_certificate_after": current_certificate,
            })
            accepted_this_pass += 1
        if accepted_this_pass == 0:
            break

    final_report = certificate_for_selection(current, scores, tolerance=tolerance)
    if final_report["strict_global_unpropagated_certificate"] > (
        threshold + tolerance * max(1.0, threshold)
    ):
        raise AssertionError("refined plan exceeds strict certificate threshold")
    if {layer: len(value) for layer, value in current.items()} != {
        layer: len(value) for layer, value in start.items()
    }:
        raise AssertionError("refinement changed fixed per-layer counts")
    audit = {
        "rho": float(rho),
        "seed": int(seed),
        "randomness_used": False,
        "tie_breaking": (
            "certificate-decreasing first; then descending objective/certificate "
            "efficiency; descending objective improvement; layer; removed ID; added ID"
        ),
        "max_passes": int(max_passes),
        "candidate_count": len(candidates),
        "accepted_swap_count": len(accepted),
        "base_strict_certificate": base_certificate,
        "strict_certificate_threshold": threshold,
        "final_strict_certificate": final_report["strict_global_unpropagated_certificate"],
        "base_down_norm_objective": base_report["normalized_down_norm_objective"],
        "final_down_norm_objective": final_report["normalized_down_norm_objective"],
        "selection_sha256": _selection_digest(current),
        "matched_plan_validation": validation,
        "accepted_swaps": accepted,
    }
    return current, audit


def clone_plan_with_selection(
    base_plan: dict,
    selected: Mapping[int, set[int] | tuple[int, ...] | list[int]],
    *,
    plan_name: str,
    metadata: dict,
) -> dict:
    """Clone a trusted physical plan while changing only selected identities."""
    plan = copy.deepcopy(base_plan)
    rows = plan_layer_map(plan)
    for layer_idx, values in selected.items():
        row = rows[layer_idx]
        indices = sorted(int(value) for value in values)
        if len(indices) != len(row.get("prune_idx", [])):
            raise ValueError(f"layer {layer_idx}: fixed allocation count changed")
        row["prune_idx"] = indices
        row["pruned_channels"] = len(indices)
        row["new_intermediate"] = int(row["old_intermediate"]) - len(indices)
    plan["selector"] = "certified_hybrid_fixed_plan"
    plan["allocation_source"] = "rmsnorm_bound"
    plan["ranking_source"] = "fixed_plan"
    plan["allocation_ranking_experiment_name"] = plan_name
    plan["certified_hybrid"] = metadata
    plan["total_selected_layer_channels"] = sum(
        len(row.get("prune_idx", [])) for row in plan["layers"]
    )
    return plan
