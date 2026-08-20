"""Physical heterogeneous MoE-width checkpoint helpers.

Stock Qwen configuration exposes one global ``moe_intermediate_size`` even
though a globally allocated pruning plan can produce a different width in each
decoder layer.  These helpers retain the original config, store the exact plan,
reshape each expert container before state-dict loading, and never pad a pruned
layer back to the original width.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn


PLAN_FILENAME = "heterogeneous_pruning_plan.json"


def find_decoder_layers(model: Any):
    for root_name in ("model", "transformer"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for layers_name in ("layers", "h", "blocks"):
            layers = getattr(root, layers_name, None)
            if layers is not None:
                return layers
    raise RuntimeError("cannot locate decoder layers")


def plan_counts(plan: dict) -> dict[str, int]:
    layer_channels = sum(len(row.get("prune_idx", [])) for row in plan["layers"])
    return {
        "removed_layer_channels": layer_channels,
        "declared_removed_layer_channels": int(
            plan.get("total_selected_layer_channels", layer_channels)
        ),
    }


def _packed_container(mlp: Any):
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return None
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if isinstance(gate_up, torch.Tensor) and isinstance(down, torch.Tensor):
        return experts
    return None


def _unpacked_experts(mlp: Any) -> list[Any]:
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return [mlp]
    try:
        result = list(experts)
    except TypeError:
        return []
    return [expert for expert in result if hasattr(expert, "down_proj")]


def _set_width_attributes(module: Any, width: int) -> None:
    for name in ("intermediate_size", "moe_intermediate_size"):
        if hasattr(module, name):
            setattr(module, name, int(width))


def _replace_parameter(module: Any, name: str, value: torch.Tensor) -> None:
    old = getattr(module, name)
    setattr(module, name, nn.Parameter(value.contiguous(), requires_grad=old.requires_grad))


def _empty_like_shape(parameter: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(
        shape, dtype=parameter.dtype, device=parameter.device,
        memory_format=torch.contiguous_format,
    )


def apply_plan_physical(model: Any, plan: dict) -> dict:
    """Apply plans through the pruning pipeline's trusted physical slicers."""
    from .moe_pruning import prune_expert_channels, prune_packed_experts_global

    layers = find_decoder_layers(model)
    audit_layers = []
    total_expert_neurons = 0
    for row in plan["layers"]:
        layer_idx = int(row["layer_idx"])
        prune = sorted(int(value) for value in row.get("prune_idx", []))
        old_width = int(row["old_intermediate"])
        if len(set(prune)) != len(prune) or any(i < 0 or i >= old_width for i in prune):
            raise ValueError(f"invalid prune indices in layer {layer_idx}")
        new_width = old_width - len(prune)
        mlp = getattr(layers[layer_idx], "mlp", None)
        if mlp is None:
            raise RuntimeError(f"layer {layer_idx} has no MLP")
        packed = _packed_container(mlp)
        prune_tensor = torch.tensor(prune, dtype=torch.long)
        if packed is not None:
            gate_up = packed.gate_up_proj
            down = packed.down_proj
            if gate_up.ndim != 3 or down.ndim != 3:
                raise AssertionError(f"layer {layer_idx} packed tensors are not rank 3")
            if gate_up.shape[1] != 2 * old_width or down.shape[2] != old_width:
                raise AssertionError(f"layer {layer_idx} packed width mismatch")
            actual_new_width = prune_packed_experts_global(
                packed, prune_tensor, alignment=int(plan.get("channel_alignment", 1))
            )
            if actual_new_width != new_width:
                raise AssertionError(f"layer {layer_idx} trusted slicer width mismatch")
            num_experts = int(gate_up.shape[0])
            layout = "packed"
            _set_width_attributes(packed, new_width)
        else:
            experts = _unpacked_experts(mlp)
            if not experts:
                raise RuntimeError(f"layer {layer_idx} expert layout unsupported")
            for expert in experts:
                gate = expert.gate_proj
                up = expert.up_proj
                down = expert.down_proj
                if gate.weight.shape[0] != old_width or up.weight.shape[0] != old_width:
                    raise AssertionError(f"layer {layer_idx} unpacked input width mismatch")
                if down.weight.shape[1] != old_width:
                    raise AssertionError(f"layer {layer_idx} unpacked down width mismatch")
                prune_expert_channels(expert, prune_tensor)
                gate.out_features = up.out_features = new_width
                down.in_features = new_width
                _set_width_attributes(expert, new_width)
            num_experts = len(experts)
            layout = "unpacked"
        _set_width_attributes(mlp, new_width)
        total_expert_neurons += len(prune) * num_experts
        audit_layers.append({
            "layer_idx": layer_idx, "layout": layout, "num_experts": num_experts,
            "old_width": old_width, "new_width": new_width,
            "removed_layer_channels": len(prune),
            "removed_expert_neurons": len(prune) * num_experts,
        })
    return {
        "layers": audit_layers,
        "removed_layer_channels": sum(
            row["removed_layer_channels"] for row in audit_layers
        ),
        "removed_expert_neurons": total_expert_neurons,
    }


def resize_empty_model_to_plan(model: Any, plan: dict) -> None:
    """Resize a meta/empty model to saved heterogeneous shapes before load."""
    layers = find_decoder_layers(model)
    for row in plan["layers"]:
        layer_idx = int(row["layer_idx"])
        old_width = int(row["old_intermediate"])
        new_width = old_width - len(row.get("prune_idx", []))
        mlp = getattr(layers[layer_idx], "mlp")
        packed = _packed_container(mlp)
        if packed is not None:
            gate_up = packed.gate_up_proj
            down = packed.down_proj
            _replace_parameter(packed, "gate_up_proj", _empty_like_shape(
                gate_up, (gate_up.shape[0], 2 * new_width, gate_up.shape[2])
            ))
            _replace_parameter(packed, "down_proj", _empty_like_shape(
                down, (down.shape[0], down.shape[1], new_width)
            ))
            _set_width_attributes(packed, new_width)
        else:
            experts = _unpacked_experts(mlp)
            if not experts:
                raise RuntimeError(f"layer {layer_idx} expert layout unsupported")
            for expert in experts:
                gate, up, down = expert.gate_proj, expert.up_proj, expert.down_proj
                gate.weight = nn.Parameter(_empty_like_shape(
                    gate.weight, (new_width, gate.weight.shape[1])
                ))
                up.weight = nn.Parameter(_empty_like_shape(
                    up.weight, (new_width, up.weight.shape[1])
                ))
                down.weight = nn.Parameter(_empty_like_shape(
                    down.weight, (down.weight.shape[0], new_width)
                ))
                gate.out_features = up.out_features = new_width
                down.in_features = new_width
                _set_width_attributes(expert, new_width)
        _set_width_attributes(mlp, new_width)


def inspect_plan_shapes(model: Any, plan: dict) -> list[dict]:
    layers = find_decoder_layers(model)
    result = []
    for row in plan["layers"]:
        layer_idx = int(row["layer_idx"])
        expected = int(row["old_intermediate"]) - len(row.get("prune_idx", []))
        mlp = getattr(layers[layer_idx], "mlp")
        packed = _packed_container(mlp)
        if packed is not None:
            actual = int(packed.down_proj.shape[2])
            gate_up_shape = list(packed.gate_up_proj.shape)
            down_shape = list(packed.down_proj.shape)
            num_experts = int(packed.down_proj.shape[0])
            layout = "packed"
        else:
            experts = _unpacked_experts(mlp)
            widths = {int(expert.down_proj.weight.shape[1]) for expert in experts}
            if len(widths) != 1:
                raise AssertionError(f"layer {layer_idx} experts have different widths")
            actual = next(iter(widths))
            gate_up_shape = [list(experts[0].gate_proj.weight.shape),
                             list(experts[0].up_proj.weight.shape)]
            down_shape = list(experts[0].down_proj.weight.shape)
            num_experts = len(experts)
            layout = "unpacked"
        if actual != expected:
            raise AssertionError(
                f"layer {layer_idx} width={actual}, expected={expected}"
            )
        result.append({
            "layer_idx": layer_idx, "layout": layout, "num_experts": num_experts,
            "expected_width": expected, "actual_width": actual,
            "gate_up_shape": gate_up_shape, "down_shape": down_shape,
            "no_original_width_padding": actual == expected,
        })
    return result


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    moe = 0
    for layer in find_decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            moe += sum(parameter.numel() for parameter in experts.parameters())
    return {"total": int(total), "moe_experts": int(moe)}


def load_heterogeneous_checkpoint(
    checkpoint_dir: str, *, device_map: str | dict = "auto",
    dtype: torch.dtype | None = None,
):
    """Reload a saved heterogeneous checkpoint without width padding."""
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoModelForCausalLM

    plan_path = os.path.join(checkpoint_dir, PLAN_FILENAME)
    if not os.path.isfile(plan_path):
        raise FileNotFoundError(f"heterogeneous plan missing: {plan_path}")
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True,
            torch_dtype=dtype,
        )
        resize_empty_model_to_plan(model, plan)
        model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, checkpoint=checkpoint_dir, device_map=device_map, dtype=dtype,
    )
    model.eval()
    inspect_plan_shapes(model, plan)
    return model, plan
