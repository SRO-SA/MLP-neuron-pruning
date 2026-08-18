"""Pure helpers for fixed-layer-allocation MoE pruning-plan replay.

This module deliberately contains no model or torch dependencies.  It takes the
already-computed, p95-aggregated layer score vectors from ``moe_pruning`` and
selects the lowest-scoring channels while preserving every source-plan layer
count exactly.  Physical pruning remains in the existing pipeline.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PACKED_PRUNING_MODE = "packed_same_channel"


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
    if alternate_selector == source_selector:
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
    )
    if set(int(key) for key in scores_by_layer) != set(source_layers):
        raise ValueError("alternate score layers do not match source plan layers")
    if set(int(key) for key in num_experts_by_layer) != set(source_layers):
        raise ValueError("expert-count layers do not match source plan layers")
    if total_expert_neurons <= 0:
        raise ValueError("total_expert_neurons must be positive")

    selection: Dict[Tuple[int, int], List[int]] = {}
    layer_audit: List[Dict[str, object]] = []
    total_layer_channels = 0
    total_removed_expert_neurons = 0
    for layer_idx in sorted(source_layers):
        source_layer = source_layers[layer_idx]
        source_indices = sorted(_as_int_list(source_layer.get("prune_idx", [])))
        required_count = len(source_indices)
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
            "source_indices": source_indices,
            "selected_indices": selected,
            "overlap_indices": overlap,
            "overlap_count": len(overlap),
            "jaccard": (len(overlap) / union_count if union_count else 1.0),
            "num_experts": num_experts,
            "removed_expert_neurons": removed_expert_neurons,
        })

    expected_total = int(source_plan["total_selected_layer_channels"])
    if total_layer_channels != expected_total:
        raise AssertionError(
            f"replay selected {total_layer_channels} layer-channels; "
            f"source requires {expected_total}"
        )
    for row in layer_audit:
        layer_idx = int(row["layer_idx"])
        source_count = len(source_layers[layer_idx].get("prune_idx", []))
        if int(row["required_pruned_channels"]) != source_count:
            raise AssertionError(f"layer {layer_idx} allocation changed during replay")

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
        "source_total_layer_channels": expected_total,
        "total_selected_layer_channels": total_layer_channels,
        "total_removed_expert_neurons": total_removed_expert_neurons,
        "actual_pct": round(actual_pct, 6),
        "overshot_requested_target": actual_pct > float(target_pct) + 1e-12,
        "layers": layer_audit,
    }
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
