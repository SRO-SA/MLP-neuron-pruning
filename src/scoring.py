"""
scoring.py
==========
Static neuron importance scores for SwiGLU MLP pruning.

All scores are derived purely from model weights (no forward passes,
no activations).  Lower score ⟹ neuron contributes less ⟹ prune first.

───────────────────────────────────────────────────────────────────────────
Theory: RMSNorm-bounded neuron contribution
───────────────────────────────────────────────────────────────────────────
The MLP input r comes from a RMSNorm layer with weight γ.  For any unit
vector v ∈ ℝ^d_model, RMSNorm produces:

    r_k = x_k / RMS(x) * γ_k

By definition RMS(x) = ||x||_2 / sqrt(d_model), so:

    |r_k| ≤ |γ_k|                 (normalised component-wise)
    ||r||_2 ≤ sqrt(d_model) * ||γ||_∞  =: R

Neuron i's contribution to the MLP output (for a single token vector r):

    c_i(r)  = SiLU(r · w_gate_i) * (r · w_up_i) * w_down_i  ∈ ℝ^d_model

Upper-bounding the scalar pre-SwiGLU product:

    |SiLU(r · w_gate_i) * (r · w_up_i)|
      ≤ |r · w_gate_i| * |r · w_up_i|          (since |SiLU(x)| ≤ |x|)
      ≤ ||r||_2² * ||w_gate_i||_2 * ||w_up_i||_2    (Cauchy-Schwarz × 2)

A tighter bound exploits the dot-product term:

    |r · w_gate_i| * |r · w_up_i|
      ≤ R² * (||w_gate_i|| * ||w_up_i|| + |w_gate_i · w_up_i|) / 2

  (from the AM–product inequality applied to the Cauchy-Schwarz bound and
   the direct inner-product bound, averaged for tightness)

The full neuron contribution norm is then:

    ||c_i(r)||_2
      ≤ R² * (||w_gate_i|| * ||w_up_i|| + |w_gate_i · w_up_i|) / 2
            * ||w_down_i||_2

This gives the proposed score (method = 'rmsnorm_bound_angle'):

    score_i = R² * ((||w_gate_i|| * ||w_up_i|| + |w_gate_i · w_up_i|) / 2)
                  * ||w_down_i||

where  R = sqrt(d_model) * ||γ||_∞

───────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from .model_utils import get_mlp_weights, get_rmsnorm_before_mlp

logger = logging.getLogger(__name__)

ScoringMethod = Literal["random", "down_norm", "product_norm", "rmsnorm_bound_angle"]


def compute_scores(
    layer,
    method: ScoringMethod,
    seed: int = 42,
) -> torch.Tensor:
    """
    Compute a 1-D importance score tensor of length d_ff for *layer*.

    Lower score  ⟹  neuron is less important  ⟹  pruned first.

    Parameters
    ----------
    layer   : a transformer decoder layer (must have .mlp + .post_attention_layernorm)
    method  : one of 'random', 'down_norm', 'product_norm', 'rmsnorm_bound_angle'
    seed    : RNG seed (only used for 'random')

    Returns
    -------
    scores : torch.Tensor  shape [d_ff],  dtype float32, on CPU
    """
    weights = get_mlp_weights(layer)
    w_gate  = weights["gate"].float().cpu()   # [d_ff, d_model]
    w_up    = weights["up"].float().cpu()     # [d_ff, d_model]
    w_down  = weights["down"].float().cpu()   # [d_model, d_ff]
    d_ff    = weights["d_ff"]
    d_model = weights["d_model"]

    # -----------------------------------------------------------------------
    # Neuron-wise vector norms
    #   gate_row_norm[i] = ||gate_proj.weight[i, :]||_2
    #   up_row_norm[i]   = ||up_proj.weight[i, :]||_2
    #   down_col_norm[i] = ||down_proj.weight[:, i]||_2
    # -----------------------------------------------------------------------
    gate_row_norm = w_gate.norm(dim=1)   # [d_ff]
    up_row_norm   = w_up.norm(dim=1)     # [d_ff]
    down_col_norm = w_down.norm(dim=0)   # [d_ff]  (norm over d_model axis)

    assert gate_row_norm.shape == (d_ff,), f"gate_row_norm shape {gate_row_norm.shape}"
    assert up_row_norm.shape   == (d_ff,), f"up_row_norm shape {up_row_norm.shape}"
    assert down_col_norm.shape == (d_ff,), f"down_col_norm shape {down_col_norm.shape}"

    # -----------------------------------------------------------------------
    # Method A: random
    # -----------------------------------------------------------------------
    if method == "random":
        rng = torch.Generator()
        rng.manual_seed(seed)
        scores = torch.rand(d_ff, generator=rng)
        return scores

    # -----------------------------------------------------------------------
    # Method B: down_norm
    #   score_i = ||w_down_i||_2
    # -----------------------------------------------------------------------
    if method == "down_norm":
        return down_col_norm

    # -----------------------------------------------------------------------
    # Method C: product_norm
    #   score_i = ||w_gate_i|| * ||w_up_i|| * ||w_down_i||
    # -----------------------------------------------------------------------
    if method == "product_norm":
        scores = gate_row_norm * up_row_norm * down_col_norm
        return scores

    # -----------------------------------------------------------------------
    # Method D: rmsnorm_bound_angle  (proposed)
    #
    #   R     = sqrt(d_model) * ||gamma||_inf
    #   dot_i = |w_gate_i · w_up_i|
    #   score_i = R² * (||w_gate_i|| * ||w_up_i|| + dot_i) / 2 * ||w_down_i||
    #
    # The (||·||·||·|| + dot) / 2 term is the average of the Cauchy-Schwarz
    # outer-product bound and the direct inner-product bound.  It is tighter
    # than the pure norm product and rewards gate/up vectors that are
    # approximately parallel (high cosine similarity → big output).
    # -----------------------------------------------------------------------
    if method == "rmsnorm_bound_angle":
        rmsnorm = get_rmsnorm_before_mlp(layer)
        gamma   = rmsnorm.weight.float().cpu()   # [d_model]

        # R = sqrt(d_model) * ||gamma||_inf
        R_squared = float(d_model) * float(gamma.abs().max()) ** 2

        # |w_gate_i · w_up_i|  — element-wise dot products across neuron rows
        dot_gate_up = (w_gate * w_up).sum(dim=1).abs()   # [d_ff]

        # (||w_gate_i|| * ||w_up_i|| + |dot_i|) / 2
        mixed_term = (gate_row_norm * up_row_norm + dot_gate_up) / 2.0

        scores = R_squared * mixed_term * down_col_norm
        return scores

    raise ValueError(
        f"Unknown scoring method '{method}'. "
        "Choose from: random, down_norm, product_norm, rmsnorm_bound_angle"
    )


def get_keep_indices(
    scores: torch.Tensor,
    prune_ratio: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    Return the indices of neurons to KEEP after pruning *prune_ratio* fraction.

    Neurons with the **lowest** scores are pruned first.

    Parameters
    ----------
    scores      : 1-D importance scores, length d_ff
    prune_ratio : fraction in [0, 1); 0.2 ⟹ prune 20% of neurons
    seed        : only used if prune_ratio == 0 (returns all indices)

    Returns
    -------
    keep_indices : sorted 1-D LongTensor of length ceil((1 - prune_ratio) * d_ff)
    """
    d_ff = scores.numel()
    n_keep = max(1, int(round(d_ff * (1.0 - prune_ratio))))

    # argsort ascending: lowest-scored neurons first
    sorted_indices = torch.argsort(scores)          # [d_ff] ascending
    keep_indices   = sorted_indices[d_ff - n_keep:] # keep the top n_keep
    keep_indices   = keep_indices.sort().values      # sort for nicer weight layout

    assert keep_indices.numel() == n_keep, (
        f"Expected {n_keep} keep indices, got {keep_indices.numel()}"
    )
    return keep_indices
