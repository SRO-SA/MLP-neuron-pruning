"""CPU-only helpers for interpreting systems profiler evidence."""
from __future__ import annotations


def shape_integers(value) -> set[int]:
    """Collect all integer dimensions from a nested profiler-shape value."""
    found = set()
    if isinstance(value, (list, tuple)):
        for item in value:
            found.update(shape_integers(item))
    elif isinstance(value, int):
        found.add(value)
    return found


def validate_packed_moe_shapes(gate_up_shape, down_shape) -> dict:
    """Validate a packed SwiGLU expert container and describe its GEMM widths."""
    gate_up_shape = tuple(int(value) for value in gate_up_shape)
    down_shape = tuple(int(value) for value in down_shape)
    if len(gate_up_shape) != 3 or len(down_shape) != 3:
        raise ValueError(
            f"packed expert tensors must be rank 3: "
            f"gate_up={gate_up_shape} down={down_shape}"
        )
    experts, doubled_width, hidden = gate_up_shape
    down_experts, down_hidden, width = down_shape
    if (
        experts != down_experts
        or hidden != down_hidden
        or doubled_width != 2 * width
    ):
        raise ValueError(
            f"incompatible packed expert shapes: "
            f"gate_up={gate_up_shape} down={down_shape}"
        )
    return {
        "layout": "packed",
        "expert_count": experts,
        "hidden_size": hidden,
        "intermediate_width": width,
    }


def validate_unpacked_moe_shapes(expert_shapes) -> dict:
    """Validate unpacked gate/up/down expert weights without importing torch."""
    normalized = [
        {
            name: tuple(int(value) for value in shapes[name])
            for name in ("gate", "up", "down")
        }
        for shapes in expert_shapes
    ]
    if not normalized:
        raise ValueError("unpacked expert container is empty")
    reference = normalized[0]
    gate, up, down = reference["gate"], reference["up"], reference["down"]
    if len(gate) != 2 or len(up) != 2 or len(down) != 2:
        raise ValueError(f"unpacked expert weights must be rank 2: {reference}")
    width, hidden = gate
    if up != (width, hidden) or down != (hidden, width):
        raise ValueError(f"incompatible unpacked expert shapes: {reference}")
    for expert_idx, shapes in enumerate(normalized[1:], start=1):
        if shapes != reference:
            raise ValueError(
                f"heterogeneous experts within one layer are unsupported: "
                f"expert0={reference} expert{expert_idx}={shapes}"
            )
    return {
        "layout": "unpacked",
        "expert_count": len(normalized),
        "hidden_size": hidden,
        "intermediate_width": width,
        "gate_proj_shape": list(gate),
        "up_proj_shape": list(up),
        "down_proj_shape": list(down),
    }
