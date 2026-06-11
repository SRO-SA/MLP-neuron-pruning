"""
pruning.py
==========
Physical structured MLP pruning: removes selected intermediate neurons
from every transformer layer by creating smaller Linear weight tensors.

What "removing neuron i" means
--------------------------------
The SwiGLU MLP computes:

    m(r) = Σ_i  SiLU(r · w_gate_i) * (r · w_up_i) * w_down_i

Neuron i is identified by:
  gate_proj.weight[i, :]   row i   of gate_proj  (shape [d_ff, d_model])
  up_proj.weight[i, :]     row i   of up_proj    (shape [d_ff, d_model])
  down_proj.weight[:, i]   col i   of down_proj  (shape [d_model, d_ff])

Removing neuron i therefore means:
  • delete row    i from gate_proj.weight   → new shape [d_ff', d_model]
  • delete row    i from up_proj.weight     → new shape [d_ff', d_model]
  • delete column i from down_proj.weight  → new shape [d_model, d_ff']

where d_ff' = d_ff - |pruned neurons|.

After replacing the weight tensors the MLP intermediate size is d_ff'.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .model_utils import (
    clone_model,
    get_mlp_module,
    get_mlp_weights,
    get_transformer_layers,
)
from .scoring import ScoringMethod, compute_scores, get_keep_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-layer pruning
# ---------------------------------------------------------------------------

def prune_layer_mlp(layer, keep_indices: torch.Tensor) -> None:
    """
    Physically reduce the MLP of *layer* to keep only the neurons in
    *keep_indices*.  Modifies *layer* **in-place**.

    Parameters
    ----------
    layer        : transformer decoder layer with .mlp.{gate,up,down}_proj
    keep_indices : sorted 1-D LongTensor of neuron indices to KEEP
    """
    mlp = get_mlp_module(layer)
    weights = get_mlp_weights(layer)

    w_gate  = weights["gate"]   # [d_ff, d_model]
    w_up    = weights["up"]     # [d_ff, d_model]
    w_down  = weights["down"]   # [d_model, d_ff]
    d_ff    = weights["d_ff"]
    d_model = weights["d_model"]
    n_keep  = keep_indices.numel()

    # Validate keep_indices
    assert keep_indices.min() >= 0 and keep_indices.max() < d_ff, (
        f"keep_indices out of range [0, {d_ff}): "
        f"min={keep_indices.min()}, max={keep_indices.max()}"
    )

    # -----------------------------------------------------------------------
    # Slice the weight tensors
    # -----------------------------------------------------------------------
    # gate_proj & up_proj: keep selected ROWS     [d_ff, d_model] → [n_keep, d_model]
    new_w_gate = w_gate[keep_indices, :]          # [n_keep, d_model]
    new_w_up   = w_up[keep_indices,   :]          # [n_keep, d_model]

    # down_proj: keep selected COLUMNS            [d_model, d_ff] → [d_model, n_keep]
    new_w_down = w_down[:, keep_indices]          # [d_model, n_keep]

    # Post-slice shape assertions
    assert new_w_gate.shape == (n_keep, d_model), (
        f"new gate_proj shape mismatch: expected ({n_keep}, {d_model}), "
        f"got {tuple(new_w_gate.shape)}"
    )
    assert new_w_up.shape == (n_keep, d_model), (
        f"new up_proj shape mismatch: expected ({n_keep}, {d_model}), "
        f"got {tuple(new_w_up.shape)}"
    )
    assert new_w_down.shape == (d_model, n_keep), (
        f"new down_proj shape mismatch: expected ({d_model}, {n_keep}), "
        f"got {tuple(new_w_down.shape)}"
    )

    # -----------------------------------------------------------------------
    # Replace Linear layers with new (smaller) ones
    # -----------------------------------------------------------------------
    device = w_gate.device
    dtype  = w_gate.dtype

    def _make_linear(weight: torch.Tensor, in_f: int, out_f: int) -> nn.Linear:
        lin = nn.Linear(in_f, out_f, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            lin.weight.copy_(weight)
        return lin

    mlp.gate_proj = _make_linear(new_w_gate, d_model, n_keep)
    mlp.up_proj   = _make_linear(new_w_up,   d_model, n_keep)
    mlp.down_proj = _make_linear(new_w_down, n_keep,  d_model)

    # -----------------------------------------------------------------------
    # Diagnostic print
    # -----------------------------------------------------------------------
    n_pruned = d_ff - n_keep
    logger.debug(
        "  Layer pruned: d_ff %d → %d  (removed %d neurons, %.1f%%)",
        d_ff, n_keep, n_pruned, 100.0 * n_pruned / d_ff,
    )


# ---------------------------------------------------------------------------
# Full-model pruning
# ---------------------------------------------------------------------------

def prune_model(
    model,
    prune_ratio: float,
    method: ScoringMethod,
    seed: int = 42,
) -> Dict:
    """
    Clone *model*, compute per-layer neuron scores, and physically prune each
    MLP to remove the lowest-scoring *prune_ratio* fraction of neurons.

    Parameters
    ----------
    model       : original (unmodified) AutoModelForCausalLM
    prune_ratio : fraction of d_ff neurons to prune per layer, e.g. 0.2
    method      : scoring method name
    seed        : RNG seed (for 'random')

    Returns
    -------
    pruned_model : deep-copied and pruned model
    info         : dict with per-layer diagnostics
    """
    if prune_ratio == 0.0:
        # Nothing to prune; return a clone with a diagnostic dict
        pruned = clone_model(model)
        layers = get_transformer_layers(pruned)
        per_layer = []
        for i, layer in enumerate(layers):
            w = get_mlp_weights(layer)
            per_layer.append({
                "layer_idx":  i,
                "d_ff_before": w["d_ff"],
                "d_ff_after":  w["d_ff"],
                "n_pruned":    0,
            })
        return pruned, {"per_layer": per_layer, "prune_ratio": 0.0, "method": method}

    pruned = clone_model(model)
    layers = get_transformer_layers(pruned)
    n_layers = len(layers)
    per_layer: List[Dict] = []

    print(f"\n{'─'*60}")
    print(f"Pruning  method={method}  ratio={prune_ratio:.0%}  layers={n_layers}")
    print(f"{'─'*60}")

    with torch.no_grad():
        for i, layer in enumerate(layers):
            weights = get_mlp_weights(layer)
            d_ff    = weights["d_ff"]
            d_model = weights["d_model"]

            # Score all neurons for this layer
            scores      = compute_scores(layer, method, seed=seed)  # [d_ff]
            keep_indices = get_keep_indices(scores, prune_ratio, seed=seed)
            n_keep  = keep_indices.numel()
            n_prune = d_ff - n_keep

            print(
                f"  Layer {i:2d}:  gate_proj {list(weights['gate'].shape)}"
                f"  up_proj {list(weights['up'].shape)}"
                f"  down_proj {list(weights['down'].shape)}"
            )
            print(
                f"           d_model={d_model}  d_ff={d_ff}"
                f"  prune={n_prune}  keep={n_keep}"
            )

            # Physically remove neurons
            prune_layer_mlp(layer, keep_indices.to(weights["gate"].device))

            # Verify forward plumbing still holds
            _verify_mlp_shapes(layer, d_model, n_keep)

            per_layer.append({
                "layer_idx":   i,
                "d_ff_before": d_ff,
                "d_ff_after":  n_keep,
                "n_pruned":    n_prune,
            })

    # Optionally update config intermediate_size to match pruned width.
    # (All layers are pruned to the same width when d_ff is uniform.)
    if hasattr(pruned.config, "intermediate_size"):
        last_d_ff_after = per_layer[-1]["d_ff_after"]
        if all(p["d_ff_after"] == last_d_ff_after for p in per_layer):
            pruned.config.intermediate_size = last_d_ff_after

    print(f"{'─'*60}\nPruning complete.\n")

    return pruned, {"per_layer": per_layer, "prune_ratio": prune_ratio, "method": method}


# ---------------------------------------------------------------------------
# Post-prune shape verifier
# ---------------------------------------------------------------------------

def _verify_mlp_shapes(layer, expected_d_model: int, expected_d_ff: int) -> None:
    """Assert that the pruned MLP has the expected shapes."""
    mlp = get_mlp_module(layer)
    assert mlp.gate_proj.weight.shape == (expected_d_ff, expected_d_model), (
        f"gate_proj shape after pruning: {mlp.gate_proj.weight.shape} "
        f"expected ({expected_d_ff}, {expected_d_model})"
    )
    assert mlp.up_proj.weight.shape == (expected_d_ff, expected_d_model), (
        f"up_proj shape after pruning: {mlp.up_proj.weight.shape} "
        f"expected ({expected_d_ff}, {expected_d_model})"
    )
    assert mlp.down_proj.weight.shape == (expected_d_model, expected_d_ff), (
        f"down_proj shape after pruning: {mlp.down_proj.weight.shape} "
        f"expected ({expected_d_model}, {expected_d_ff})"
    )


# ---------------------------------------------------------------------------
# Quick forward-pass sanity check
# ---------------------------------------------------------------------------

def verify_forward_pass(model, tokenizer, device: str) -> bool:
    """
    Run a single forward pass on a short prompt to confirm the pruned model
    does not crash.  Returns True on success, False (with log) on failure.
    """
    prompt = "Hello, my name is"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        assert out.logits is not None
        logger.info("Forward-pass sanity check: PASSED (logits shape %s)", list(out.logits.shape))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Forward-pass sanity check FAILED: %s", exc)
        return False
