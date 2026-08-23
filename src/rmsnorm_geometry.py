"""Weight-only RMSNorm geometry bounds for SwiGLU channels.

The functions in this module are deliberately architecture-agnostic.  They
operate on the three projection tensors and the RMSNorm scale immediately
preceding an MLP/MoE block.  All accumulation is performed in float32 on CPU
so callers can score one expert at a time without retaining a model-wide FP32
copy of the weights.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _validated_float32_inputs(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Validate SwiGLU/RMSNorm shapes and return detached float32 CPU tensors."""
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise AssertionError(
            "gate, up, and down must be rank-2 tensors; got "
            f"gate={tuple(gate.shape)}, up={tuple(up.shape)}, "
            f"down={tuple(down.shape)}"
        )
    if tuple(gate.shape) != tuple(up.shape):
        raise AssertionError(
            f"gate/up shape mismatch: {tuple(gate.shape)} != {tuple(up.shape)}"
        )

    d_ff, d_model = gate.shape
    if tuple(down.shape) != (d_model, d_ff):
        raise AssertionError(
            "down must have shape [d_model, d_ff]; got "
            f"{tuple(down.shape)}, expected {(d_model, d_ff)}"
        )
    if gamma is None:
        raise AssertionError("RMSNorm gamma is required for RMSNorm geometry bounds")
    if gamma.ndim != 1 or tuple(gamma.shape) != (d_model,):
        raise AssertionError(
            f"gamma must have shape [d_model]={d_model}; got {tuple(gamma.shape)}"
        )

    gate32 = gate.detach().to(device="cpu", dtype=torch.float32)
    up32 = up.detach().to(device="cpu", dtype=torch.float32)
    down32 = down.detach().to(device="cpu", dtype=torch.float32)
    gamma32 = gamma.detach().to(device="cpu", dtype=torch.float32)
    if not all(torch.isfinite(t).all() for t in (gate32, up32, down32, gamma32)):
        raise AssertionError("RMSNorm geometry inputs must all be finite")
    return gate32, up32, down32, gamma32, d_ff, d_model


def compute_rmsnorm_ellipsoid_bound_from_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Return the exact coordinate-wise RMSNorm ellipsoid bound per channel.

    For ``r = diag(gamma) q`` with ``||q||_2 <= sqrt(d_model)``, the score is

        d_model / 2 * (
            ||gamma * g_i|| ||gamma * u_i||
            + |(gamma * g_i) dot (gamma * u_i)|
        ) * ||d_i||.
    """
    gate32, up32, down32, gamma32, d_ff, d_model = _validated_float32_inputs(
        gate, up, down, gamma
    )
    weighted_gate = gate32 * gamma32.unsqueeze(0)
    weighted_up = up32 * gamma32.unsqueeze(0)

    gate_norms = weighted_gate.norm(dim=1)
    up_norms = weighted_up.norm(dim=1)
    dot_products = (weighted_gate * weighted_up).sum(dim=1).abs()
    down_norms = down32.norm(dim=0)
    scores = (
        (float(d_model) / 2.0)
        * (gate_norms * up_norms + dot_products)
        * down_norms
    )
    if tuple(scores.shape) != (d_ff,):
        raise AssertionError(f"ellipsoid scores have wrong shape: {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("ellipsoid score computation produced non-finite values")
    return scores


def compute_rmsnorm_ellipsoid_and_down_norm_from_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ellipsoid and down-column scores from one FP32 weight copy.

    This helper is used by the set-certificate collector.  It deliberately
    materializes only one expert's FP32 weights at a time and returns CPU
    score vectors, never a model-wide FP32 weight copy.
    """
    gate32, up32, down32, gamma32, d_ff, d_model = _validated_float32_inputs(
        gate, up, down, gamma
    )
    weighted_gate = gate32 * gamma32.unsqueeze(0)
    weighted_up = up32 * gamma32.unsqueeze(0)
    down_norms = down32.norm(dim=0)
    ellipsoid = (
        (float(d_model) / 2.0)
        * (
            weighted_gate.norm(dim=1) * weighted_up.norm(dim=1)
            + (weighted_gate * weighted_up).sum(dim=1).abs()
        )
        * down_norms
    )
    if tuple(ellipsoid.shape) != (d_ff,) or tuple(down_norms.shape) != (d_ff,):
        raise AssertionError("ellipsoid/down-norm score shapes are invalid")
    if not torch.isfinite(ellipsoid).all() or not torch.isfinite(down_norms).all():
        raise FloatingPointError("ellipsoid/down-norm computation is non-finite")
    return ellipsoid, down_norms


def compute_rmsnorm_bound_triplet_from_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(legacy, valid_sphere, ellipsoid)`` from one FP32 weight copy.

    ``legacy`` exactly mirrors the historical MoE selector's unscaled formula.
    It is included only for reproducibility/ranking diagnostics.  ``sphere``
    and ``ellipsoid`` are the two mathematically valid RMSNorm bounds.
    """
    gate32, up32, down32, gamma32, d_ff, d_model = _validated_float32_inputs(
        gate, up, down, gamma
    )
    gate_norms = gate32.norm(dim=1)
    up_norms = up32.norm(dim=1)
    dot_products = (gate32 * up32).sum(dim=1).abs()
    down_norms = down32.norm(dim=0)
    legacy = ((gate_norms * up_norms + dot_products) / 2.0) * down_norms
    sphere = (
        float(d_model) * float(gamma32.abs().max()) ** 2 * legacy
    )

    weighted_gate = gate32 * gamma32.unsqueeze(0)
    weighted_up = up32 * gamma32.unsqueeze(0)
    ellipsoid = (
        (float(d_model) / 2.0)
        * (
            weighted_gate.norm(dim=1) * weighted_up.norm(dim=1)
            + (weighted_gate * weighted_up).sum(dim=1).abs()
        )
        * down_norms
    )
    for name, scores in (
        ("legacy", legacy),
        ("sphere", sphere),
        ("ellipsoid", ellipsoid),
    ):
        if tuple(scores.shape) != (d_ff,):
            raise AssertionError(f"{name} scores have wrong shape: {tuple(scores.shape)}")
        if not torch.isfinite(scores).all():
            raise FloatingPointError(f"{name} score computation produced non-finite values")
    return legacy, sphere, ellipsoid


def compute_rmsnorm_sphere_bound_from_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Return the valid gamma-infinity sphere bound per channel.

    This is the conservative comparison bound obtained from
    ``||r|| <= sqrt(d_model) * ||gamma||_inf``.  It is separate from any
    historical selector implementation so reproducibility code need not be
    changed to test the mathematical inequality ``ellipsoid <= sphere``.
    """
    gate32, up32, down32, gamma32, d_ff, d_model = _validated_float32_inputs(
        gate, up, down, gamma
    )
    gate_norms = gate32.norm(dim=1)
    up_norms = up32.norm(dim=1)
    dot_products = (gate32 * up32).sum(dim=1).abs()
    down_norms = down32.norm(dim=0)
    radius_sq = float(d_model) * float(gamma32.abs().max()) ** 2
    scores = radius_sq * (
        (gate_norms * up_norms + dot_products) / 2.0
    ) * down_norms
    if tuple(scores.shape) != (d_ff,):
        raise AssertionError(f"sphere scores have wrong shape: {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("sphere score computation produced non-finite values")
    return scores


def compute_observed_channel_contribution_max_from_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    routed_inputs: torch.Tensor,
) -> torch.Tensor:
    """Return each channel's maximum observed output-vector contribution norm.

    For routed normalized inputs ``r``, channel ``i`` contributes
    ``SiLU(r·g_i) (r·u_i) d_i``.  Its vector 2-norm is therefore the absolute
    scalar activation times ``||d_i||_2``.  Maxima are taken over the supplied
    routed-input sample and accumulated in float32 on CPU.
    """
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise AssertionError("gate/up/down must be rank-2")
    if tuple(gate.shape) != tuple(up.shape):
        raise AssertionError("gate/up shapes differ")
    d_ff, d_model = gate.shape
    if tuple(down.shape) != (d_model, d_ff):
        raise AssertionError("down shape is incompatible with gate/up")
    if routed_inputs is None or routed_inputs.ndim != 2:
        raise AssertionError("routed_inputs must have shape [n_samples, d_model]")
    if routed_inputs.shape[0] <= 0 or routed_inputs.shape[1] != d_model:
        raise AssertionError(
            f"routed_inputs shape {tuple(routed_inputs.shape)} is incompatible "
            f"with d_model={d_model}"
        )
    gate32 = gate.detach().to(device="cpu", dtype=torch.float32)
    up32 = up.detach().to(device="cpu", dtype=torch.float32)
    down32 = down.detach().to(device="cpu", dtype=torch.float32)
    inputs32 = routed_inputs.detach().to(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        activations = F.silu(inputs32 @ gate32.T) * (inputs32 @ up32.T)
        observed = activations.abs().amax(dim=0) * down32.norm(dim=0)
    if tuple(observed.shape) != (d_ff,) or not torch.isfinite(observed).all():
        raise FloatingPointError("observed channel contributions are invalid")
    return observed
