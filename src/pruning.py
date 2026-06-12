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

  ⚠  DO NOT delete row i from down_proj.  Neuron i is column i of down_proj.
     row i of down_proj is a completely different concept (output dim i).

where d_ff' = d_ff - |pruned neurons|.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .model_utils import (
    clone_model,
    get_mlp_module,
    get_mlp_weights,
    get_transformer_layers,
)
from .scoring import ScoringMethod, compute_scores, get_keep_indices, get_prune_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-layer pruning
# ---------------------------------------------------------------------------

def prune_layer_mlp(layer, keep_indices: torch.Tensor) -> None:
    """
    Physically reduce the MLP of *layer* to keep only the neurons in
    *keep_indices*.  Modifies *layer* **in-place**.

    Shape contract (enforced by assertions):
        gate_proj.weight : [d_ff, d_model] → [n_keep, d_model]    (drop rows)
        up_proj.weight   : [d_ff, d_model] → [n_keep, d_model]    (drop rows)
        down_proj.weight : [d_model, d_ff] → [d_model, n_keep]    (drop COLUMNS)
    """
    mlp    = get_mlp_module(layer)
    weights = get_mlp_weights(layer)

    w_gate  = weights["gate"]   # [d_ff, d_model]
    w_up    = weights["up"]     # [d_ff, d_model]
    w_down  = weights["down"]   # [d_model, d_ff]
    d_ff    = weights["d_ff"]
    d_model = weights["d_model"]
    n_keep  = keep_indices.numel()

    assert keep_indices.min() >= 0 and keep_indices.max() < d_ff, (
        f"keep_indices out of range [0, {d_ff}): "
        f"min={keep_indices.min()}, max={keep_indices.max()}"
    )

    # ── Slice weight tensors ─────────────────────────────────────────────────
    ki = keep_indices.to(w_gate.device)

    # gate & up: keep selected ROWS    [d_ff, d_model] → [n_keep, d_model]
    new_w_gate = w_gate[ki, :]       # [n_keep, d_model]
    new_w_up   = w_up[ki,   :]       # [n_keep, d_model]

    # down: keep selected COLUMNS      [d_model, d_ff] → [d_model, n_keep]
    #   ⚠  Neuron i = COLUMN i of down_proj, NOT row i.
    new_w_down = w_down[:, ki]       # [d_model, n_keep]

    # Post-slice shape assertions
    assert new_w_gate.shape == (n_keep, d_model), (
        f"gate_proj post-prune shape: {tuple(new_w_gate.shape)} ≠ ({n_keep}, {d_model})"
    )
    assert new_w_up.shape == (n_keep, d_model), (
        f"up_proj post-prune shape: {tuple(new_w_up.shape)} ≠ ({n_keep}, {d_model})"
    )
    assert new_w_down.shape == (d_model, n_keep), (
        f"down_proj post-prune shape: {tuple(new_w_down.shape)} ≠ ({d_model}, {n_keep})"
    )

    # ── Replace Linear modules ────────────────────────────────────────────────
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

    logger.debug(
        "  Layer pruned: d_ff %d → %d  (removed %d, %.1f%%)",
        d_ff, n_keep, d_ff - n_keep, 100.0 * (d_ff - n_keep) / d_ff,
    )


# ---------------------------------------------------------------------------
# Zero-masking (used for equivalence test)
# ---------------------------------------------------------------------------

def zero_mask_layer(
    layer,
    prune_indices: torch.Tensor,
) -> None:
    """
    Zero out the weights corresponding to neurons *prune_indices*
    WITHOUT changing the tensor shapes.  Modifies *layer* **in-place**.

    After zero-masking:
      gate_proj.weight[prune_indices, :] = 0   → SiLU(0) = 0
      up_proj.weight[prune_indices, :]   = 0   → 0 * SiLU(...) = 0
      down_proj.weight[:, prune_indices] = 0   → even if a_i ≠ 0, contributes 0

    The MLP output should be identical to physically removing those neurons,
    because the zeroed rows/column contribute exactly 0 to the computation.
    """
    mlp = get_mlp_module(layer)

    device = mlp.gate_proj.weight.device
    pi     = prune_indices.to(device)

    with torch.no_grad():
        mlp.gate_proj.weight[pi, :] = 0.0   # zero rows in gate_proj
        mlp.up_proj.weight[pi, :]   = 0.0   # zero rows in up_proj
        mlp.down_proj.weight[:, pi] = 0.0   # zero COLUMNS in down_proj (not rows!)


def compare_zero_mask_vs_physical(
    model,
    tokenizer,
    device: str,
    method: ScoringMethod,
    prune_ratio: float = 0.05,
    layer_idx: int = 0,
    test_text: str = "The quick brown fox jumps over the lazy dog in the summer.",
) -> Dict:
    """
    Equivalence test: zero-masking and physical pruning should produce
    IDENTICAL logits (up to floating-point rounding).

    Procedure
    ---------
    1. Compute keep/prune indices from the ORIGINAL model's layer {layer_idx}.
    2. Clone A: zero-mask the pruned neurons (keep full tensor shape).
    3. Clone B: physically remove the pruned neurons (smaller tensors).
    4. Run both models on the same input.
    5. Compare logit tensors.

    Interpretation
    --------------
    max_logit_diff ≈ 0  →  physical pruning is CORRECT
    max_logit_diff >> 0 →  BUG in physical pruning
                           (wrong dimension sliced, or wrong neuron mapping)
    """
    orig_layers = get_transformer_layers(model)
    orig_layer  = orig_layers[layer_idx]

    # Compute indices from original weights
    scores       = compute_scores(orig_layer, method)
    keep_indices = get_keep_indices(scores, prune_ratio)
    prune_indices = get_prune_indices(scores, prune_ratio)

    # ── Clone A: zero-mask ────────────────────────────────────────────────────
    model_zm = clone_model(model)
    layers_zm = get_transformer_layers(model_zm)
    zero_mask_layer(layers_zm[layer_idx], prune_indices)

    # ── Clone B: physical prune ───────────────────────────────────────────────
    model_ph = clone_model(model)
    layers_ph = get_transformer_layers(model_ph)
    prune_layer_mlp(layers_ph[layer_idx], keep_indices.to(device))

    # ── Compare logits ────────────────────────────────────────────────────────
    enc = tokenizer(test_text, return_tensors="pt").to(device)

    model_zm.eval()
    model_ph.eval()

    with torch.no_grad():
        out_zm = model_zm(**enc)
        out_ph = model_ph(**enc)

    diff      = (out_zm.logits - out_ph.logits).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    # Also compare the pruned-layer outputs directly via hooks
    zm_layer_out  = {}
    ph_layer_out  = {}

    def _hook_zm(module, inp, out):
        zm_layer_out["out"] = out.detach()

    def _hook_ph(module, inp, out):
        ph_layer_out["out"] = out.detach()

    hzm = get_mlp_module(get_transformer_layers(model_zm)[layer_idx]).register_forward_hook(_hook_zm)
    hph = get_mlp_module(get_transformer_layers(model_ph)[layer_idx]).register_forward_hook(_hook_ph)

    with torch.no_grad():
        model_zm(**enc)
        model_ph(**enc)

    hzm.remove()
    hph.remove()

    mlp_diff = (zm_layer_out["out"] - ph_layer_out["out"]).abs()
    max_mlp_diff = mlp_diff.max().item()

    result = {
        "layer_idx":      layer_idx,
        "method":         method,
        "prune_ratio":    prune_ratio,
        "n_pruned":       prune_indices.numel(),
        "n_kept":         keep_indices.numel(),
        "max_logit_diff": max_diff,
        "mean_logit_diff": mean_diff,
        "max_mlp_layer_diff": max_mlp_diff,
        "is_consistent":  max_diff < 1e-3,
    }
    return result


# ---------------------------------------------------------------------------
# Full-model pruning
# ---------------------------------------------------------------------------

def prune_model(
    model,
    prune_ratio: float,
    method: ScoringMethod,
    seed: int = 42,
) -> Tuple[object, Dict]:
    """
    Clone *model*, score all neurons, and physically prune each MLP to
    remove the lowest-scoring *prune_ratio* fraction.

    Every call deep-copies the original model; the original is never modified.

    Returns
    -------
    pruned_model : a new, smaller model
    info         : dict with per-layer diagnostics
    """
    if prune_ratio == 0.0:
        pruned = clone_model(model)
        layers = get_transformer_layers(pruned)
        per_layer = []
        for i, layer in enumerate(layers):
            w = get_mlp_weights(layer)
            per_layer.append({"layer_idx": i, "d_ff_before": w["d_ff"],
                               "d_ff_after": w["d_ff"], "n_pruned": 0})
        return pruned, {"per_layer": per_layer, "prune_ratio": 0.0, "method": method}

    pruned   = clone_model(model)
    layers   = get_transformer_layers(pruned)
    n_layers = len(layers)
    per_layer: List[Dict] = []

    print("\n" + chr(8212)*62)
    print(f"Pruning  method={method}  ratio={prune_ratio:.1%}  layers={n_layers}")
    print(chr(8212)*62)

    with torch.no_grad():
        for i, layer in enumerate(layers):
            weights = get_mlp_weights(layer)
            d_ff    = weights["d_ff"]
            d_model = weights["d_model"]

            scores       = compute_scores(layer, method, seed=seed)
            keep_indices = get_keep_indices(scores, prune_ratio, seed=seed)
            n_keep  = keep_indices.numel()
            n_prune = d_ff - n_keep

            print(
                f"  Layer {i:2d}: "
                f"gate{list(weights['gate'].shape)} "
                f"up{list(weights['up'].shape)} "
                f"down{list(weights['down'].shape)}"
                f"  d_model={d_model} d_ff={d_ff}"
                f"  prune={n_prune} keep={n_keep}"
            )

            prune_layer_mlp(layer, keep_indices.to(weights["gate"].device))
            _verify_mlp_shapes(layer, d_model, n_keep)

            per_layer.append({
                "layer_idx":   i,
                "d_ff_before": d_ff,
                "d_ff_after":  n_keep,
                "n_pruned":    n_prune,
            })

    # Update config intermediate_size if all layers pruned uniformly
    if hasattr(pruned.config, "intermediate_size"):
        d_ff_values = set(p["d_ff_after"] for p in per_layer)
        if len(d_ff_values) == 1:
            pruned.config.intermediate_size = d_ff_values.pop()

    print(chr(8212)*62 + "\nPruning complete.\n")
    return pruned, {"per_layer": per_layer, "prune_ratio": prune_ratio, "method": method}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_mlp_shapes(layer, expected_d_model: int, expected_d_ff: int) -> None:
    """Assert that the pruned MLP has the expected shapes."""
    mlp = get_mlp_module(layer)
    assert mlp.gate_proj.weight.shape == (expected_d_ff, expected_d_model), (
        f"gate_proj post-prune: {mlp.gate_proj.weight.shape} != "
        f"({expected_d_ff}, {expected_d_model})"
    )
    assert mlp.up_proj.weight.shape == (expected_d_ff, expected_d_model), (
        f"up_proj post-prune: {mlp.up_proj.weight.shape}"
    )
    assert mlp.down_proj.weight.shape == (expected_d_model, expected_d_ff), (
        f"down_proj post-prune: {mlp.down_proj.weight.shape} != "
        f"({expected_d_model}, {expected_d_ff})"
    )


def prune_model_by_layer_indices(
    model,
    prune_indices_per_layer: List[torch.Tensor],
    label: str = "",
) -> Tuple[object, Dict]:
    """
    Clone *model* and physically prune each layer using the provided
    per-layer prune indices.

    Unlike prune_model() which applies a uniform ratio, this function
    accepts an explicit list of which neurons to remove from each layer.
    Layers with an empty prune_indices tensor are left unchanged.
    the caller computes which neurons are "certified safe" and passes
    them here; this function handles the physical removal.

    Parameters
    ----------
    model                   : unpruned model (never modified)
    prune_indices_per_layer : List[LongTensor]  one per transformer layer;
                              each tensor holds the neuron indices to PRUNE
    label                   : optional descriptive string for logging

    Returns
    -------
    pruned_model : deep-copy with selected neurons removed
    info         : dict with per_layer diagnostics
    """
    pruned = clone_model(model)
    layers = get_transformer_layers(pruned)

    if len(prune_indices_per_layer) != len(layers):
        raise ValueError(
            f"prune_indices_per_layer has {len(prune_indices_per_layer)} entries "
            f"but model has {len(layers)} transformer layers"
        )

    per_layer: List[Dict] = []
    n_pruned_total = sum(len(pi) for pi in prune_indices_per_layer)

    if label:
        print(f"\n  Pruning [{label}]  total_neurons_removed={n_pruned_total}")

    with torch.no_grad():
        for i, layer in enumerate(layers):
            pi      = prune_indices_per_layer[i]
            w       = get_mlp_weights(layer)
            d_ff    = w["d_ff"]
            d_model = w["d_model"]

            if len(pi) == 0:
                per_layer.append({
                    "layer_idx":   i,
                    "d_ff_before": d_ff,
                    "d_ff_after":  d_ff,
                    "n_pruned":    0,
                })
                continue

            # Build keep_indices as complement of prune_indices
            prune_set    = set(pi.tolist())
            keep_indices = torch.tensor(
                [j for j in range(d_ff) if j not in prune_set],
                dtype=torch.long,
            )
            n_keep = keep_indices.numel()

            prune_layer_mlp(layer, keep_indices.to(w["gate"].device))
            _verify_mlp_shapes(layer, d_model, n_keep)

            per_layer.append({
                "layer_idx":   i,
                "d_ff_before": d_ff,
                "d_ff_after":  n_keep,
                "n_pruned":    int(len(pi)),
            })

    # Update config intermediate_size if all pruned layers are uniform
    if hasattr(pruned.config, "intermediate_size"):
        vals = set(p["d_ff_after"] for p in per_layer)
        if len(vals) == 1:
            pruned.config.intermediate_size = vals.pop()

    return pruned, {"per_layer": per_layer, "label": label}


def verify_forward_pass(model, tokenizer, device: str) -> bool:
    """Run a single forward pass to confirm the pruned model does not crash."""
    try:
        inputs = tokenizer("Hello, my name is", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        assert out.logits is not None
        logger.info("Forward-pass check: PASSED  logits shape=%s", list(out.logits.shape))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Forward-pass check FAILED: %s", exc)
        return False
