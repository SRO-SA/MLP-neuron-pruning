"""Same-input routed-MoE perturbation and statistical utilities.

The routines in this module deliberately operate on a fixed post-RMSNorm
input and fixed routing decisions.  They do not propagate a pruned hidden
state through later Transformer layers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_RELATIVE_TOLERANCE = 5e-5
DEFAULT_ABSOLUTE_TOLERANCE = 1e-6


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_fixed_routing(mlp, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k expert IDs and normalized weights used by the MoE router."""
    router = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
    if router is None:
        raise AttributeError(f"{type(mlp).__name__} has no supported router")
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = router(flat)
    top_k = int(getattr(mlp, "top_k", getattr(mlp, "num_experts_per_tok", 0)))
    if top_k <= 0:
        config = getattr(mlp, "config", None)
        top_k = int(getattr(config, "num_experts_per_tok", 0))
    if top_k <= 0:
        raise ValueError("could not determine num_experts_per_tok")
    probabilities = torch.softmax(logits.float(), dim=-1)
    weights, expert_ids = torch.topk(probabilities, k=top_k, dim=-1)
    normalize = bool(getattr(
        mlp, "norm_topk_prob",
        getattr(getattr(mlp, "config", None), "norm_topk_prob", True),
    ))
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    if not torch.allclose(
        weights.sum(dim=-1), torch.ones_like(weights[:, 0]),
        rtol=1e-5, atol=1e-6,
    ):
        raise AssertionError("fixed routing weights are not normalized")
    return expert_ids, weights


def fixed_routed_moe_output(
    mlp, hidden_states: torch.Tensor, expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only the routed experts with identities and weights held fixed."""
    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, original_shape[-1])
    expert_ids = expert_ids.reshape(flat.shape[0], -1).to(flat.device)
    routing_weights = routing_weights.reshape(flat.shape[0], -1).to(flat.device)
    if expert_ids.shape != routing_weights.shape:
        raise ValueError("expert IDs and routing weights have incompatible shapes")
    experts = getattr(mlp, "experts", None)
    if experts is None:
        raise AttributeError(f"{type(mlp).__name__} has no experts")

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if isinstance(gate_up, torch.Tensor) and isinstance(down, torch.Tensor):
        # Qwen3's packed container already accepts fixed top-k expert IDs and
        # returns one expert output per token/slot.  Using it retains the exact
        # physical pruned GEMM path.
        expert_outputs = experts(flat, expert_ids)
        if expert_outputs.ndim == 2:
            expert_outputs = expert_outputs.reshape(
                flat.shape[0], expert_ids.shape[1], flat.shape[1]
            )
        expected = (flat.shape[0], expert_ids.shape[1], flat.shape[1])
        if tuple(expert_outputs.shape) != expected:
            raise AssertionError(
                f"packed expert output shape={tuple(expert_outputs.shape)}, "
                f"expected={expected}"
            )
        result = (
            expert_outputs
            * routing_weights.to(expert_outputs.dtype).unsqueeze(-1)
        ).sum(dim=1)
        return result.reshape(*original_shape[:-1], original_shape[-1])

    try:
        unpacked = list(experts)
    except TypeError as error:
        raise TypeError("unsupported expert container") from error
    result = torch.zeros_like(flat)
    for expert_idx, expert in enumerate(unpacked):
        positions = (expert_ids == expert_idx).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_indices, slot_indices = positions[:, 0], positions[:, 1]
        x = flat.index_select(0, token_indices)
        expert_output = expert.down_proj(
            F.silu(expert.gate_proj(x)) * expert.up_proj(x)
        )
        weighted = expert_output * routing_weights[
            token_indices, slot_indices
        ].to(expert_output.dtype).unsqueeze(-1)
        result.index_add_(0, token_indices, weighted)
    return result.reshape(*original_shape[:-1], original_shape[-1])


def expert_set_bounds(
    ellipsoid_scores: np.ndarray, selected_indices: Iterable[int],
) -> np.ndarray:
    scores = np.asarray(ellipsoid_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("ellipsoid scores must have shape [expert, channel]")
    selected = np.asarray(sorted(int(index) for index in selected_indices), dtype=np.int64)
    if selected.size and (selected.min() < 0 or selected.max() >= scores.shape[1]):
        raise ValueError("selected channel is outside the score tensor")
    return scores[:, selected].sum(axis=1, dtype=np.float64)


def route_conditioned_bounds(
    expert_sums: np.ndarray, expert_ids: np.ndarray,
    routing_weights: np.ndarray,
) -> np.ndarray:
    sums = np.asarray(expert_sums, dtype=np.float64)
    ids = np.asarray(expert_ids, dtype=np.int64)
    weights = np.asarray(routing_weights, dtype=np.float64)
    if ids.shape != weights.shape:
        raise ValueError("expert IDs and weights differ in shape")
    if ids.size and (ids.min() < 0 or ids.max() >= sums.size):
        raise ValueError("routed expert ID outside certificate tensor")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise AssertionError("route-conditioned weights are not normalized")
    return (weights * sums[ids]).sum(axis=1, dtype=np.float64)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.broadcast_to(np.asarray(denominator, dtype=np.float64), num.shape)
    result = np.zeros_like(num)
    positive = den > 0
    result[positive] = num[positive] / den[positive]
    result[~positive & (num > 0)] = np.inf
    return result


def violation_mask(
    actual: np.ndarray, bound: np.ndarray | float, *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
) -> np.ndarray:
    actual_array = np.asarray(actual, dtype=np.float64)
    bound_array = np.broadcast_to(
        np.asarray(bound, dtype=np.float64), actual_array.shape
    )
    return actual_array > (
        bound_array * (1.0 + relative_tolerance) + absolute_tolerance
    )


def distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: 0.0 for key in ("mean", "median", "p95", "p99", "max")} | {
            "count": 0
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Spearman inputs must be matched one-dimensional arrays")
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def paired_bootstrap_mean_difference(
    first: np.ndarray, second: np.ndarray, *, resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float | int]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("paired bootstrap requires matched document vectors")
    delta = left - right
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        draws[start:stop] = delta[indices].mean(axis=1)
    return {
        "difference": float(delta.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "documents": int(delta.size),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def shard_path(root: str | Path, document_index: int, layer_index: int) -> Path:
    return Path(root) / "shards" / f"doc_{document_index:04d}" / f"layer_{layer_index:02d}.pt"
