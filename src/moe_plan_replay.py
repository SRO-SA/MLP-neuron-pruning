"""Pure helpers for fixed-layer-allocation MoE pruning-plan replay.

This module deliberately contains no model or torch dependencies.  It takes the
already-computed, p95-aggregated layer score vectors from ``moe_pruning`` and
selects the lowest-scoring channels while preserving every source-plan layer
count exactly.  Physical pruning remains in the existing pipeline.
"""
from __future__ import annotations

import math
import os
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PACKED_PRUNING_MODE = "packed_same_channel"
SUPPORTED_ALLOCATION_RANKING_SOURCES = {
    "rmsnorm_bound",
    "rmsnorm_ellipsoid_bound",
    "down_norm",
    "activation_score",
    "fixed_plan",
}


def _as_int_list(values: Iterable[object]) -> List[int]:
    return [int(value) for value in values]


def _source_layers_by_index(plan: Mapping[str, object]) -> Dict[int, Mapping[str, object]]:
    raw_layers = plan.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("source plan must contain a non-empty 'layers' list")
    layers: Dict[int, Mapping[str, object]] = {}
    for raw in raw_layers:
        if not isinstance(raw, Mapping) or "layer_idx" not in raw:
            raise ValueError("every source plan layer must contain layer_idx")
        layer_idx = int(raw["layer_idx"])
        if layer_idx in layers:
            raise ValueError(f"source plan repeats layer {layer_idx}")
        layers[layer_idx] = raw
    return layers


def normalize_aligned_allocation_counts(
    counts_by_layer: Mapping[int, int],
    *,
    exact_total_layer_channels: int,
    channel_alignment: int,
) -> Dict[int, int]:
    """Scale an existing aligned count vector down to an exact aligned total.

    Hamilton (largest-remainder) apportionment is applied in units of aligned
    channel blocks.  This preserves the source vector's relative layer shape
    as closely as possible without inventing a new cross-layer score scale.
    The operation is deliberately downward-only for this diagnostic.
    """
    if channel_alignment <= 0:
        raise ValueError("channel_alignment must be positive")
    target = int(exact_total_layer_channels)
    if target < 0 or target % channel_alignment:
        raise ValueError(
            f"exact total {target} must be non-negative and aligned to "
            f"{channel_alignment}"
        )
    counts = {int(layer): int(count) for layer, count in counts_by_layer.items()}
    if not counts:
        raise ValueError("allocation count vector must not be empty")
    if any(count < 0 or count % channel_alignment for count in counts.values()):
        raise ValueError("all source allocation counts must be non-negative/aligned")
    source_total = sum(counts.values())
    if target > source_total:
        raise ValueError(
            f"exact total {target} exceeds source allocation total {source_total}; "
            "upward extrapolation is not supported"
        )
    if target == source_total:
        return dict(counts)
    source_blocks = {
        layer: count // channel_alignment for layer, count in counts.items()
    }
    source_block_total = sum(source_blocks.values())
    target_blocks = target // channel_alignment
    ideals = {
        layer: blocks * target_blocks / source_block_total
        for layer, blocks in source_blocks.items()
    }
    allocated = {
        layer: min(source_blocks[layer], int(math.floor(ideals[layer])))
        for layer in source_blocks
    }
    remaining = target_blocks - sum(allocated.values())
    order = sorted(
        source_blocks,
        key=lambda layer: (
            -(ideals[layer] - math.floor(ideals[layer])),
            -source_blocks[layer],
            layer,
        ),
    )
    while remaining:
        progressed = False
        for layer in order:
            if allocated[layer] < source_blocks[layer]:
                allocated[layer] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise AssertionError("unable to apportion requested exact total")
    result = {
        layer: allocated[layer] * channel_alignment for layer in allocated
    }
    if sum(result.values()) != target:
        raise AssertionError("normalized allocation total does not match request")
    if any(result[layer] > counts[layer] for layer in result):
        raise AssertionError("normalized allocation exceeded a source layer count")
    return result


def validate_fixed_allocation_source_plan_static(
    source_plan: Mapping[str, object],
    *,
    source_plan_path: str,
    expected_source_selector: str,
    alternate_selector: str,
    pruning_mode: str,
    channel_alignment: int,
    max_layer_frac: float,
    max_expert_frac: float,
    target_pct: float,
    allow_same_selector: bool = False,
) -> Dict[int, Mapping[str, object]]:
    """Validate every source/config invariant that needs no loaded model."""
    if not source_plan_path or not os.path.isfile(source_plan_path):
        raise FileNotFoundError(
            f"fixed-allocation source plan not found: {source_plan_path!r}"
        )
    if pruning_mode != PACKED_PRUNING_MODE:
        raise ValueError(
            "fixed-allocation replay requires physical pruning mode "
            f"{PACKED_PRUNING_MODE!r}; got {pruning_mode!r}"
        )
    source_mode = str(source_plan.get("pruning_mode", ""))
    if source_mode != PACKED_PRUNING_MODE or source_mode != pruning_mode:
        raise ValueError(
            "source/current physical pruning mode mismatch: "
            f"source={source_mode!r}, current={pruning_mode!r}"
        )
    if channel_alignment <= 0:
        raise ValueError("channel_alignment must be positive")
    source_alignment = int(source_plan.get("channel_alignment", -1))
    if source_alignment != channel_alignment:
        raise ValueError(
            "source/current channel alignment mismatch: "
            f"source={source_alignment}, current={channel_alignment}"
        )
    source_selector = str(source_plan.get("selector", ""))
    if source_selector != expected_source_selector:
        raise ValueError(
            "source selector mismatch: "
            f"plan={source_selector!r}, expected={expected_source_selector!r}"
        )
    if alternate_selector == source_selector and not allow_same_selector:
        raise ValueError(
            "alternate channel selector must differ from the source allocation "
            f"selector; both are {source_selector!r}"
        )
    source_target = float(source_plan.get("target_pct", float("nan")))
    if abs(source_target - float(target_pct)) > 1e-9:
        raise ValueError(
            f"source target_pct={source_target} does not match current {target_pct}"
        )
    source_max_layer_frac = float(source_plan.get("max_layer_frac", float("nan")))
    if abs(source_max_layer_frac - float(max_layer_frac)) > 1e-12:
        raise ValueError(
            "source/current layer-cap fraction mismatch: "
            f"source={source_max_layer_frac}, current={max_layer_frac}"
        )

    source_layers = _source_layers_by_index(source_plan)
    counted_total = 0
    for layer_idx, source_layer in source_layers.items():
        old_intermediate = int(source_layer.get("old_intermediate", -1))
        if old_intermediate <= 0:
            raise ValueError(
                f"layer {layer_idx} has invalid old_intermediate={old_intermediate}"
            )
        source_indices = _as_int_list(source_layer.get("prune_idx", []))
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(f"layer {layer_idx} source indices contain duplicates")
        if any(index < 0 or index >= old_intermediate for index in source_indices):
            raise ValueError(f"layer {layer_idx} source index is out of range")
        required_count = int(source_layer.get("pruned_channels", len(source_indices)))
        if required_count != len(source_indices):
            raise ValueError(
                f"layer {layer_idx} pruned_channels={required_count} but "
                f"prune_idx has {len(source_indices)} entries"
            )
        if required_count % channel_alignment != 0:
            raise ValueError(
                f"layer {layer_idx} required count {required_count} violates "
                f"alignment {channel_alignment}"
            )
        new_intermediate = old_intermediate - required_count
        declared_new = int(source_layer.get("new_intermediate", new_intermediate))
        if declared_new != new_intermediate:
            raise ValueError(
                f"layer {layer_idx} new_intermediate={declared_new}, expected "
                f"{new_intermediate} from source indices"
            )
        if new_intermediate <= 0 or new_intermediate % channel_alignment != 0:
            raise ValueError(
                f"layer {layer_idx} replay would produce invalid width "
                f"{new_intermediate} for alignment {channel_alignment}"
            )
        aligned_cap = (
            int(old_intermediate * float(max_layer_frac)) // channel_alignment
        ) * channel_alignment
        if required_count > aligned_cap:
            raise ValueError(
                f"layer {layer_idx} requires {required_count} channels, above "
                f"the current aligned layer cap {aligned_cap}"
            )
        expert_cap = max(1, int(old_intermediate * float(max_expert_frac)))
        if required_count > expert_cap:
            raise ValueError(
                f"layer {layer_idx} requires {required_count} channels, above "
                f"the current max_expert_frac cap {expert_cap}"
            )
        counted_total += required_count

    declared_total = int(source_plan.get("total_selected_layer_channels", -1))
    if declared_total != counted_total:
        raise ValueError(
            "source plan total_selected_layer_channels mismatch: "
            f"declared={declared_total}, counted={counted_total}"
        )
    return source_layers


def validate_fixed_allocation_request(
    source_plan: Mapping[str, object],
    *,
    source_plan_path: str,
    expected_source_selector: str,
    alternate_selector: str,
    pruning_mode: str,
    channel_alignment: int,
    max_layer_frac: float,
    max_expert_frac: float,
    target_pct: float,
    layer_sizes: Mapping[int, int],
    allow_same_selector: bool = False,
) -> Dict[int, Mapping[str, object]]:
    """Validate source/config and loaded-model replay invariants."""
    source_layers = validate_fixed_allocation_source_plan_static(
        source_plan,
        source_plan_path=source_plan_path,
        expected_source_selector=expected_source_selector,
        alternate_selector=alternate_selector,
        pruning_mode=pruning_mode,
        channel_alignment=channel_alignment,
        max_layer_frac=max_layer_frac,
        max_expert_frac=max_expert_frac,
        target_pct=target_pct,
        allow_same_selector=allow_same_selector,
    )
    current_layer_ids = {int(layer_idx) for layer_idx in layer_sizes}
    if set(source_layers) != current_layer_ids:
        missing = sorted(current_layer_ids - set(source_layers))
        extra = sorted(set(source_layers) - current_layer_ids)
        raise ValueError(
            "source plan layers do not match current MoE layers: "
            f"missing={missing}, extra={extra}"
        )
    for layer_idx, source_layer in source_layers.items():
        old_intermediate = int(source_layer["old_intermediate"])
        current_size = int(layer_sizes[layer_idx])
        if old_intermediate != current_size:
            raise ValueError(
                f"layer {layer_idx} width mismatch: source={old_intermediate}, "
                f"current={current_size}"
            )
    return source_layers


def build_fixed_allocation_selection(
    source_plan: Mapping[str, object],
    *,
    source_plan_path: str,
    expected_source_selector: str,
    alternate_selector: str,
    pruning_mode: str,
    channel_alignment: int,
    max_layer_frac: float,
    max_expert_frac: float,
    target_pct: float,
    scores_by_layer: Mapping[int, Sequence[float]],
    layer_sizes: Mapping[int, int],
    num_experts_by_layer: Mapping[int, int],
    total_expert_neurons: int,
    allow_same_selector: bool = False,
    exact_total_layer_channels: int | None = None,
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[str, object]]:
    """Select alternate-ranked channels with source per-layer counts fixed."""
    source_layers = validate_fixed_allocation_request(
        source_plan,
        source_plan_path=source_plan_path,
        expected_source_selector=expected_source_selector,
        alternate_selector=alternate_selector,
        pruning_mode=pruning_mode,
        channel_alignment=channel_alignment,
        max_layer_frac=max_layer_frac,
        max_expert_frac=max_expert_frac,
        target_pct=target_pct,
        layer_sizes=layer_sizes,
        allow_same_selector=allow_same_selector,
    )
    if set(int(key) for key in scores_by_layer) != set(source_layers):
        raise ValueError("alternate score layers do not match source plan layers")
    if set(int(key) for key in num_experts_by_layer) != set(source_layers):
        raise ValueError("expert-count layers do not match source plan layers")
    if total_expert_neurons <= 0:
        raise ValueError("total_expert_neurons must be positive")

    source_counts = {
        layer_idx: len(source_layer.get("prune_idx", []))
        for layer_idx, source_layer in source_layers.items()
    }
    effective_counts = (
        normalize_aligned_allocation_counts(
            source_counts,
            exact_total_layer_channels=int(exact_total_layer_channels),
            channel_alignment=channel_alignment,
        )
        if exact_total_layer_channels is not None else source_counts
    )

    selection: Dict[Tuple[int, int], List[int]] = {}
    layer_audit: List[Dict[str, object]] = []
    total_layer_channels = 0
    total_removed_expert_neurons = 0
    for layer_idx in sorted(source_layers):
        source_layer = source_layers[layer_idx]
        original_source_indices = sorted(
            _as_int_list(source_layer.get("prune_idx", []))
        )
        required_count = effective_counts[layer_idx]
        # The exact-total diagnostic uses only the normalized count vector.
        # This truncated reference is retained solely for overlap bookkeeping;
        # it is not presented as a recomputed source-selector ranking.
        source_indices = original_source_indices[:required_count]
        scores = [float(value) for value in scores_by_layer[layer_idx]]
        if len(scores) != int(layer_sizes[layer_idx]):
            raise ValueError(
                f"layer {layer_idx} alternate score length {len(scores)} does "
                f"not match width {layer_sizes[layer_idx]}"
            )
        if any(value != value or value in (float("inf"), float("-inf")) for value in scores):
            raise ValueError(f"layer {layer_idx} alternate scores are not finite")
        # Sorting (score, channel index) gives deterministic, stable tie-breaking.
        selected = sorted(
            (index for index, _ in sorted(enumerate(scores), key=lambda item: (item[1], item[0]))[:required_count])
        )
        selection[(layer_idx, -1)] = selected
        overlap = sorted(set(source_indices).intersection(selected))
        union_count = len(set(source_indices).union(selected))
        num_experts = int(num_experts_by_layer[layer_idx])
        removed_expert_neurons = required_count * num_experts
        total_layer_channels += required_count
        total_removed_expert_neurons += removed_expert_neurons
        layer_audit.append({
            "layer_idx": layer_idx,
            "required_pruned_channels": required_count,
            "source_plan_pruned_channels": len(original_source_indices),
            "source_indices": source_indices,
            "source_plan_indices": original_source_indices,
            "selected_indices": selected,
            "overlap_indices": overlap,
            "overlap_count": len(overlap),
            "jaccard": (len(overlap) / union_count if union_count else 1.0),
            "num_experts": num_experts,
            "removed_expert_neurons": removed_expert_neurons,
        })

    source_total = int(source_plan["total_selected_layer_channels"])
    expected_total = sum(effective_counts.values())
    if total_layer_channels != expected_total:
        raise AssertionError(
            f"replay selected {total_layer_channels} layer-channels; "
            f"source requires {expected_total}"
        )
    for row in layer_audit:
        layer_idx = int(row["layer_idx"])
        if int(row["required_pruned_channels"]) != effective_counts[layer_idx]:
            raise AssertionError(
                f"layer {layer_idx} allocation changed during replay"
            )

    actual_pct = 100.0 * total_removed_expert_neurons / total_expert_neurons
    audit: Dict[str, object] = {
        "plan_kind": "fixed_layer_allocation_replay",
        "source_plan_path": source_plan_path,
        "source_allocation_selector": expected_source_selector,
        "alternate_channel_selector": alternate_selector,
        "fixed_allocation": expected_source_selector,
        "channel_selector": alternate_selector,
        "pruning_mode": pruning_mode,
        "channel_alignment": channel_alignment,
        "max_expert_frac": float(max_expert_frac),
        "target_pct": float(target_pct),
        "source_total_layer_channels": source_total,
        "source_plan_total_layer_channels": source_total,
        "effective_allocation_total_layer_channels": expected_total,
        "exact_total_layer_channels": (
            int(exact_total_layer_channels)
            if exact_total_layer_channels is not None else None
        ),
        "allocation_count_normalization": (
            "hamilton_largest_remainder_aligned_downscale"
            if exact_total_layer_channels is not None else "none"
        ),
        "total_selected_layer_channels": total_layer_channels,
        "total_removed_expert_neurons": total_removed_expert_neurons,
        "actual_pct": round(actual_pct, 6),
        "overshot_requested_target": actual_pct > float(target_pct) + 1e-12,
        "layers": layer_audit,
    }
    return selection, audit


def build_allocation_ranking_selection(
    allocation_plan: Mapping[str, object],
    *,
    allocation_plan_path: str,
    allocation_source: str,
    ranking_source: str,
    pruning_mode: str,
    channel_alignment: int,
    max_layer_frac: float,
    max_expert_frac: float,
    target_pct: float,
    scores_by_layer: Mapping[int, Sequence[float]],
    layer_sizes: Mapping[int, int],
    num_experts_by_layer: Mapping[int, int],
    total_expert_neurons: int,
    experiment_name: str,
    ranking_plan: Mapping[str, object] | None = None,
    ranking_plan_path: str | None = None,
    exact_total_layer_channels: int | None = None,
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[str, object]]:
    """Apply plan-derived layer counts and an independent within-layer ranking.

    Named allocation sources require a plan created by that selector.  The
    special ``fixed_plan`` source accepts any valid plan and uses only its count
    vector.  ``fixed_plan`` may also be used as the ranking source, in which case
    a second plan supplies channel identities and must have identical counts.
    """
    if allocation_source not in SUPPORTED_ALLOCATION_RANKING_SOURCES:
        raise ValueError(f"unsupported allocation_source={allocation_source!r}")
    if ranking_source not in SUPPORTED_ALLOCATION_RANKING_SOURCES:
        raise ValueError(f"unsupported ranking_source={ranking_source!r}")
    plan_selector = str(allocation_plan.get("selector", ""))
    expected_allocation_selector = (
        plan_selector if allocation_source == "fixed_plan" else allocation_source
    )

    effective_scores: Mapping[int, Sequence[float]] = scores_by_layer
    if ranking_source == "fixed_plan":
        if ranking_plan is None or not ranking_plan_path:
            raise ValueError("ranking_source='fixed_plan' requires moe_ranking_plan")
        ranking_layers = _source_layers_by_index(ranking_plan)
        allocation_layers = _source_layers_by_index(allocation_plan)
        if set(ranking_layers) != set(allocation_layers):
            raise ValueError("ranking/allocation fixed plans have different layers")
        synthetic_scores: Dict[int, List[float]] = {}
        for layer_idx, allocation_layer in allocation_layers.items():
            required = len(allocation_layer.get("prune_idx", []))
            ranking_indices = _as_int_list(
                ranking_layers[layer_idx].get("prune_idx", [])
            )
            if len(ranking_indices) != required:
                raise ValueError(
                    f"layer {layer_idx} fixed ranking count {len(ranking_indices)} "
                    f"does not match allocation count {required}"
                )
            width = int(layer_sizes[layer_idx])
            if len(set(ranking_indices)) != len(ranking_indices) or any(
                index < 0 or index >= width for index in ranking_indices
            ):
                raise ValueError(f"layer {layer_idx} fixed ranking indices invalid")
            selected_set = set(ranking_indices)
            synthetic_scores[layer_idx] = [
                float(index - width) if index in selected_set else float(index + width)
                for index in range(width)
            ]
        effective_scores = synthetic_scores

    selection, audit = build_fixed_allocation_selection(
        allocation_plan,
        source_plan_path=allocation_plan_path,
        expected_source_selector=expected_allocation_selector,
        alternate_selector=ranking_source,
        pruning_mode=pruning_mode,
        channel_alignment=channel_alignment,
        max_layer_frac=max_layer_frac,
        max_expert_frac=max_expert_frac,
        target_pct=target_pct,
        scores_by_layer=effective_scores,
        layer_sizes=layer_sizes,
        num_experts_by_layer=num_experts_by_layer,
        total_expert_neurons=total_expert_neurons,
        allow_same_selector=True,
        exact_total_layer_channels=exact_total_layer_channels,
    )
    audit.update({
        "plan_kind": "allocation_ranking_experiment",
        "experiment_name": experiment_name,
        "allocation_source": allocation_source,
        "ranking_source": ranking_source,
        "allocation_plan_path": allocation_plan_path,
        "ranking_plan_path": ranking_plan_path or "",
    })
    audit.pop("fixed_allocation", None)
    audit.pop("channel_selector", None)

    score_lookup = {
        int(layer_idx): [float(value) for value in values]
        for layer_idx, values in effective_scores.items()
    }
    for row in audit["layers"]:
        layer_idx = int(row["layer_idx"])
        values = score_lookup[layer_idx]
        abs_max = max(abs(value) for value in values) if values else 0.0
        median = statistics.median(values) if values else float("nan")
        row.update({
            "allocation_source": allocation_source,
            "ranking_source": ranking_source,
            "ranking_score_min": min(values) if values else None,
            "ranking_score_median": median if math.isfinite(median) else None,
            "ranking_score_max": max(values) if values else None,
            "ranking_score_scale_abs_max": abs_max,
        })

    if (
        allocation_source == ranking_source
        and allocation_source != "fixed_plan"
        and exact_total_layer_channels is None
    ):
        changed_layers = [
            int(row["layer_idx"])
            for row in audit["layers"]
            if row["source_indices"] != row["selected_indices"]
        ]
        if changed_layers:
            raise AssertionError(
                "same allocation/ranking source failed to reproduce source plan "
                f"indices in layers {changed_layers}"
            )
    return selection, audit


def validate_derived_replay_plan(
    derived_plan: Mapping[str, object],
    source_plan: Mapping[str, object],
) -> None:
    """Post-save validation used by dry tests and GPU wrapper scripts."""
    replay = derived_plan.get("fixed_allocation_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("derived plan has no fixed_allocation_replay audit block")
    derived_layers = _source_layers_by_index(derived_plan)
    source_layers = _source_layers_by_index(source_plan)
    if set(derived_layers) != set(source_layers):
        raise ValueError("derived/source layer sets differ")
    for layer_idx in source_layers:
        derived_count = len(derived_layers[layer_idx].get("prune_idx", []))
        source_count = len(source_layers[layer_idx].get("prune_idx", []))
        if derived_count != source_count:
            raise ValueError(
                f"layer {layer_idx} replay count changed: "
                f"source={source_count}, derived={derived_count}"
            )
    derived_total = sum(
        len(layer.get("prune_idx", [])) for layer in derived_layers.values()
    )
    source_total = int(source_plan.get("total_selected_layer_channels", -1))
    if derived_total != source_total:
        raise ValueError(
            f"derived total {derived_total} differs from source {source_total}"
        )
    if int(replay.get("total_selected_layer_channels", -1)) != source_total:
        raise ValueError("derived replay audit total differs from source")


def validate_derived_allocation_ranking_plan(
    derived_plan: Mapping[str, object],
    allocation_plan: Mapping[str, object],
) -> None:
    """Validate a saved allocation/ranking plan against its allocation plan."""
    audit = derived_plan.get("allocation_ranking")
    if not isinstance(audit, Mapping):
        raise ValueError("derived plan has no allocation_ranking audit block")
    derived_layers = _source_layers_by_index(derived_plan)
    allocation_layers = _source_layers_by_index(allocation_plan)
    audit_layers = {
        int(row["layer_idx"]): row for row in audit.get("layers", [])
    }
    if not (
        set(derived_layers) == set(allocation_layers) == set(audit_layers)
    ):
        raise ValueError("derived/allocation/audit layer sets differ")
    for layer_idx in sorted(allocation_layers):
        allocation_indices = _as_int_list(
            allocation_layers[layer_idx].get("prune_idx", [])
        )
        derived_indices = _as_int_list(
            derived_layers[layer_idx].get("prune_idx", [])
        )
        audit_row = audit_layers[layer_idx]
        expected_count = int(audit_row["required_pruned_channels"])
        if len(derived_indices) != expected_count:
            raise ValueError(
                f"layer {layer_idx} allocation count changed: "
                f"expected={expected_count}, derived={len(derived_indices)}"
            )
        if sorted(derived_indices) != sorted(audit_row.get("selected_indices", [])):
            raise ValueError(f"layer {layer_idx} derived/audit indices differ")
        if sorted(allocation_indices)[:expected_count] != sorted(
            audit_row.get("source_indices", [])
        ):
            raise ValueError(f"layer {layer_idx} source/audit indices differ")
    source_total = int(allocation_plan.get("total_selected_layer_channels", -1))
    expected_total = int(audit.get("total_selected_layer_channels", -1))
    derived_total = sum(
        len(layer.get("prune_idx", [])) for layer in derived_layers.values()
    )
    if derived_total != expected_total:
        raise ValueError(
            f"derived total {derived_total} differs from expected {expected_total}"
        )
    declared_exact = audit.get("exact_total_layer_channels")
    if declared_exact is None and expected_total != source_total:
        raise ValueError("allocation/ranking audit total differs from source")
    if declared_exact is not None and expected_total != int(declared_exact):
        raise ValueError("allocation/ranking audit total differs from exact total")
    if (
        audit.get("allocation_source") == audit.get("ranking_source")
        and audit.get("allocation_source") != "fixed_plan"
        and audit.get("exact_total_layer_channels") is None
    ):
        changed = [
            layer_idx for layer_idx in audit_layers
            if sorted(audit_layers[layer_idx]["source_indices"])
            != sorted(audit_layers[layer_idx]["selected_indices"])
        ]
        if changed:
            raise ValueError(
                f"same-source allocation/ranking changed channels: {changed}"
            )
