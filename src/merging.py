"""
merging.py
==========
Compensated neuron merging and down-projection reconstruction for SwiGLU MLP pruning.

MOTIVATION
----------
Pure deletion removes neuron i and loses a_i(r) * d_i from the MLP output.
If neuron i is close to kept neuron j in activation space:

    a_i(r) ≈ beta * a_j(r)

then:
    a_i(r)*d_i + a_j(r)*d_j  ≈  a_j(r) * (d_j + beta*d_i)

So we can remove i and compensate by updating d_j BEFORE physical pruning:

    down_proj[:, j] += beta * down_proj[:, i]

This conserves the linear part of the output transformation at the cost of
a small approximation error from the nonlinear activation.

MERGE STRATEGIES
----------------
A. merge_weight_similarity
   beta_u = (u_i · u_j) / (||u_j||² + ε)
   dist   = ||g_i − g_j|| / (||g_i|| + ε) + ||u_i − beta_u·u_j|| / (||u_i|| + ε)

B. merge_activation_similarity
   beta_a = (a_i · a_j) / (||a_j||² + ε)  where a_i = SiLU(R@g_i)*(R@u_i) over calibration

RECONSTRUCTION STRATEGY
-----------------------
C. down_reconstruction
   Let A = SiLU(R @ W_gate.T) * (R @ W_up.T)  [N, d_ff]   (all neurons, original weights)
       Y = A @ W_down.T                          [N, d_model] (original MLP output)
   After selecting keep set K:
       A_K = A[:, K]                             [N, k]
   Find B [k, d_model] minimising ||A_K @ B - Y||²

   Variants:
     lstsq  — minimum-norm least squares (via torch.linalg.lstsq)
     ridge  — ridge regression at lambda ∈ {1e-6, 1e-5, 1e-4, 1e-3}
              Uses kernel trick (N×N inversion) when N < k for efficiency.

KEY: All down_proj updates are applied BEFORE any physical pruning.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model_utils import (
    clone_model,
    get_mlp_module,
    get_mlp_weights,
    get_transformer_layers,
)

logger = logging.getLogger(__name__)

EPS = 1e-8

# ---------------------------------------------------------------------------
# Calibration prompts split — used by run_bound_merge_stable_mode
#   TRAIN  → compute merge assignments (beta)
#   HELDOUT → evaluate reconstruction generalization (detect overfitting)
# ---------------------------------------------------------------------------
RECONSTRUCTION_TRAIN_PROMPTS = [
    "The transformer architecture was introduced in the paper Attention Is All You Need.",
    "Python is a high-level, general-purpose programming language.",
    "The human brain is the central organ of the human nervous system.",
    "Machine learning automates analytical model building using statistical methods.",
    "Quantum mechanics describes the physical properties of matter at atomic scale.",
    "The Internet is a global system of interconnected computer networks.",
    "Neural networks are computing systems loosely inspired by biological brains.",
    "The capital of France is Paris; the city has a population of about two million.",
    "Deep learning uses multiple layers to extract progressively higher-level features.",
    "Language models are trained to predict the next token in sequences of text.",
    "Gradient descent iteratively minimizes a loss function during neural network training.",
    "Attention mechanisms allow sequence models to focus on relevant input positions.",
]
RECONSTRUCTION_HELDOUT_PROMPTS = [
    "The solar system consists of eight major planets orbiting the Sun.",
    "Photosynthesis converts sunlight into chemical energy stored as glucose.",
    "The Pythagorean theorem relates the lengths of the sides of a right triangle.",
    "Computers process information using binary representations of numerical data.",
]


# ===========================================================================
# Calibration data collection — inputs
# ===========================================================================

def collect_mlp_inputs(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_seq_len: int = 128,
) -> List[torch.Tensor]:
    """
    Collect MLP input hidden states (post-RMSNorm) for all transformer layers.
    Uses register_forward_pre_hook — 2-arg signature: hook(module, inputs).

    Returns
    -------
    List[Tensor]  —  one [N_tokens, d_model] float32 CPU tensor per layer.
    """
    layers   = get_transformer_layers(model)
    n_layers = len(layers)
    captured = [[] for _ in range(n_layers)]
    handles  = []

    try:
        for idx, layer in enumerate(layers):
            def _make_hook(i):
                def _hook(module, inputs):
                    captured[i].append(inputs[0].detach().float().cpu())
                return _hook
            handles.append(
                get_mlp_module(layer).register_forward_pre_hook(_make_hook(idx))
            )
        model.eval()
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="  Collecting calibration inputs", leave=False):
                enc = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=max_seq_len,
                ).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()

    result = []
    for i in range(n_layers):
        if captured[i]:
            all_r = torch.cat(
                [x.reshape(-1, x.shape[-1]) for x in captured[i]], dim=0
            )
        else:
            w = get_mlp_weights(layers[i])
            all_r = torch.zeros(1, w["d_model"])
        result.append(all_r)

    return result


# ===========================================================================
# Calibration data collection — outputs
# ===========================================================================

def collect_mlp_outputs(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_seq_len: int = 128,
) -> List[torch.Tensor]:
    """
    Collect MLP output tensors for all transformer layers.
    Uses register_forward_hook — 3-arg signature: hook(module, inputs, output).

    Returns
    -------
    List[Tensor]  —  one [N_tokens, d_model] float32 CPU tensor per layer.
    """
    layers   = get_transformer_layers(model)
    n_layers = len(layers)
    captured = [[] for _ in range(n_layers)]
    handles  = []

    try:
        for idx, layer in enumerate(layers):
            def _make_hook(i):
                def _hook(module, inputs, output):
                    captured[i].append(output.detach().float().cpu())
                return _hook
            handles.append(
                get_mlp_module(layer).register_forward_hook(_make_hook(idx))
            )
        model.eval()
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="  Collecting calibration outputs", leave=False):
                enc = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=max_seq_len,
                ).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()

    result = []
    for i in range(n_layers):
        if captured[i]:
            all_out = torch.cat(
                [x.reshape(-1, x.shape[-1]) for x in captured[i]], dim=0
            )
        else:
            w = get_mlp_weights(layers[i])
            all_out = torch.zeros(1, w["d_model"])
        result.append(all_out)

    return result


# ===========================================================================
# Merge target selection
# ===========================================================================

def compute_merge_assignments_weight(
    layer,
    prune_indices: torch.Tensor,
    keep_indices:  torch.Tensor,
    eps: float = EPS,
) -> List[dict]:
    """
    For each pruned neuron i, find the best kept neuron j by weight-space distance.

        beta_u  = (u_i · u_j) / (||u_j||² + ε)
        dist(i,j) = ||g_i − g_j|| / (||g_i|| + ε)
                  + ||u_i − beta_u · u_j|| / (||u_i|| + ε)

    Update rule:  down_proj[:, j] += beta_u * down_proj[:, i]

    Returns List[dict] with keys: i, j, beta, dist
    """
    if len(prune_indices) == 0:
        return []

    w = get_mlp_weights(layer)
    w_gate = w["gate"].detach().float().cpu()   # [d_ff, d_model]
    w_up   = w["up"].detach().float().cpu()     # [d_ff, d_model]

    G_prune = w_gate[prune_indices]
    U_prune = w_up[prune_indices]
    G_keep  = w_gate[keep_indices]
    U_keep  = w_up[keep_indices]

    prune_list = prune_indices.tolist()
    keep_list  = keep_indices.tolist()
    n_prune    = len(prune_list)

    norms_g_p  = G_prune.norm(dim=1)
    norms_g_k  = G_keep.norm(dim=1)
    norms_u_p  = U_prune.norm(dim=1)
    norms_u_k  = U_keep.norm(dim=1)
    norms_u_ksq = norms_u_k ** 2

    # Gate distance (vectorised)
    dots_g    = G_prune @ G_keep.T
    dist_g_sq = (
        norms_g_p.unsqueeze(1) ** 2 + norms_g_k.unsqueeze(0) ** 2 - 2.0 * dots_g
    ).clamp(min=0.0)
    dist_g_norm = dist_g_sq.sqrt() / (norms_g_p.unsqueeze(1) + eps)

    # Up residual distance (vectorised, no O(n²d) memory)
    dots_u   = U_prune @ U_keep.T
    betas_u  = dots_u / (norms_u_ksq.unsqueeze(0) + eps)
    dist_u_sq = (
        norms_u_p.unsqueeze(1) ** 2
        - 2.0 * betas_u * dots_u
        + betas_u ** 2 * norms_u_ksq.unsqueeze(0)
    ).clamp(min=0.0)
    dist_u_norm = dist_u_sq.sqrt() / (norms_u_p.unsqueeze(1) + eps)

    total_dist   = dist_g_norm + dist_u_norm
    best_j_local = total_dist.argmin(dim=1)

    assignments = []
    for idx in range(n_prune):
        jl   = int(best_j_local[idx])
        assignments.append({
            "i":    prune_list[idx],
            "j":    keep_list[jl],
            "beta": round(float(betas_u[idx, jl]), 8),
            "dist": round(float(total_dist[idx, jl]), 8),
        })

    return assignments


def compute_merge_assignments_activation(
    layer,
    prune_indices: torch.Tensor,
    keep_indices:  torch.Tensor,
    all_r:         torch.Tensor,
    eps:           float = EPS,
    ridge_lambda:  float = 0.0,
    clip_value:    Optional[float] = None,
) -> List[dict]:
    """
    For each pruned neuron i, find the best kept neuron j by activation-space similarity.

        a_i = SiLU(R @ g_i) * (R @ u_i)   [N_tokens]

    Stabilized beta computation:
        beta_raw  = (a_i · a_j) / (||a_j||² + ε + ridge_lambda)
        beta_used = clip(beta_raw, -clip_value, clip_value)  if clip_value is not None

    ridge_lambda > 0 shrinks beta toward zero (prevents large updates for weakly-aligned pairs).
    clip_value   sets a hard ceiling on |beta| (prevents single outlier neurons from dominating).

    Update rule:  down_proj[:, j] += beta_used * down_proj[:, i]

    Parameters
    ----------
    all_r       : [N_tokens, d_model] CPU float32 MLP inputs from calibration.
    ridge_lambda: regularization added to denominator (default 0.0 = no ridge)
    clip_value  : if set, beta is clipped to [-clip_value, +clip_value]

    Returns List[dict] with keys: i, j, beta, beta_raw, residual
    """
    if len(prune_indices) == 0:
        return []

    w = get_mlp_weights(layer)
    w_gate = w["gate"].detach().float().cpu()
    w_up   = w["up"].detach().float().cpu()

    prune_list = prune_indices.tolist()
    keep_list  = keep_indices.tolist()
    n_prune    = len(prune_list)

    R = all_r.float().cpu()

    with torch.no_grad():
        A_all = F.silu(R @ w_gate.T) * (R @ w_up.T)   # [N, d_ff]

    A_prune = A_all[:, prune_indices]   # [N, n_prune]
    A_keep  = A_all[:, keep_indices]    # [N, n_keep]

    dots_a      = A_prune.T @ A_keep                              # [n_prune, n_keep]
    norms_k_sq  = A_keep.norm(dim=0) ** 2                        # [n_keep]
    # Ridge regularization: larger lambda → smaller, more stable beta
    betas_a     = dots_a / (norms_k_sq.unsqueeze(0) + eps + ridge_lambda)  # [n_prune, n_keep]

    norms_p     = A_prune.norm(dim=0)
    residual_sq = (
        norms_p.unsqueeze(1) ** 2
        - 2.0 * betas_a * dots_a
        + betas_a ** 2 * norms_k_sq.unsqueeze(0)
    ).clamp(min=0.0)
    residual_norm = residual_sq.sqrt() / (norms_p.unsqueeze(1) + eps)
    best_j_local  = residual_norm.argmin(dim=1)

    assignments = []
    for idx in range(n_prune):
        jl        = int(best_j_local[idx])
        beta_raw  = float(betas_a[idx, jl])
        # Apply clipping after ridge regularization
        beta_used = float(max(-clip_value, min(clip_value, beta_raw))) if clip_value is not None else beta_raw
        assignments.append({
            "i":        prune_list[idx],
            "j":        keep_list[jl],
            "beta":     round(beta_used, 8),
            "beta_raw": round(beta_raw, 8),
            "residual": round(float(residual_norm[idx, jl]), 8),
        })

    return assignments


# ===========================================================================
# Apply compensation and prune (pairwise merge)
# ===========================================================================

def apply_merge_and_prune(
    model,
    prune_indices_per_layer: List[torch.Tensor],
    assignments_per_layer:   List[List[dict]],
    label: str = "",
) -> Tuple[object, dict]:
    """
    Accumulate beta compensation on down_proj, then physically prune.

    Multiple pruned neurons can target the same j — accumulation is safe
    because we always read prune columns and write keep columns (no aliasing).

    Returns (pruned_model, info).
    """
    from .pruning import prune_model_by_layer_indices

    merged = clone_model(model)
    layers = get_transformer_layers(merged)

    with torch.no_grad():
        for layer_idx, (layer, pi, assignments) in enumerate(
            zip(layers, prune_indices_per_layer, assignments_per_layer)
        ):
            if len(pi) == 0 or not assignments:
                continue
            w    = get_mlp_weights(layer)
            down = w["down"]
            for asn in assignments:
                col_i = down.data[:, asn["i"]].clone()
                down.data[:, asn["j"]].add_(asn["beta"] * col_i)

    pruned_model, info = prune_model_by_layer_indices(
        merged, prune_indices_per_layer, label=label
    )
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pruned_model, info


# ===========================================================================
# Down-projection reconstruction (new)
# ===========================================================================

def _compute_reconstruction_for_layer(
    layer,
    r:            torch.Tensor,        # [N, d_model] calibration MLP inputs
    keep_indices: torch.Tensor,        # [k] keep neuron indices
    method:       str   = "lstsq",
    ridge_lambda: float = 1e-5,
    eps:          float = EPS,
) -> Tuple[torch.Tensor, dict]:
    """
    Solve for new down_proj weights B [k, d_model] that best reconstruct
    the ORIGINAL layer's MLP output from only the surviving neurons.

    Setup
    -----
    A  = SiLU(R @ W_gate.T) * (R @ W_up.T)   [N, d_ff]   (original weights)
    Y  = A @ W_down.T                          [N, d_model] (original output)
    A_K = A[:, keep_indices]                   [N, k]

    Goal: min_B  ||A_K @ B - Y||²

    Methods
    -------
    lstsq : minimum-norm least squares via torch.linalg.lstsq
    ridge : B = A_K.T @ (A_K @ A_K.T + λI)^{-1} @ Y   (kernel trick, N×N)
            equivalent to (A_K.T A_K + λI)^{-1} A_K.T Y but cheaper when N < k.

    Sanity info returned
    --------------------
    delete_relative_err : ||Y - A_K W_down_K.T|| / ||Y||  (pure deletion baseline)
    recon_relative_err  : ||Y - A_K B|| / ||Y||            (after reconstruction)
    improvement_pct     : relative improvement over deletion
    """
    w = get_mlp_weights(layer)
    w_gate = w["gate"].detach().float().cpu()   # [d_ff, d_model]
    w_up   = w["up"].detach().float().cpu()
    w_down = w["down"].detach().float().cpu()   # [d_model, d_ff]

    R = r.float().cpu()     # [N, d_model]
    N = R.shape[0]
    k = keep_indices.shape[0]

    with torch.no_grad():
        A   = F.silu(R @ w_gate.T) * (R @ w_up.T)   # [N, d_ff]
        Y   = A @ w_down.T                           # [N, d_model]
        A_K = A[:, keep_indices]                     # [N, k]

        # Deletion baseline
        Y_del    = A_K @ w_down[:, keep_indices].T   # [N, d_model]
        del_err  = float((Y - Y_del).norm() / (Y.norm() + eps))

        if method == "lstsq":
            result = torch.linalg.lstsq(A_K, Y, rcond=None)
            B = result.solution   # [k, d_model]

        elif method.startswith("ridge"):
            if N < k:
                # Kernel (dual) form: N×N inversion — efficient when N << k
                AAt  = A_K @ A_K.T                                    # [N, N]
                reg  = ridge_lambda * torch.eye(N, dtype=AAt.dtype)
                coeffs = torch.linalg.solve(AAt + reg, Y)             # [N, d_model]
                B    = A_K.T @ coeffs                                  # [k, d_model]
            else:
                # Primal form: k×k inversion
                AtA  = A_K.T @ A_K                                    # [k, k]
                AtY  = A_K.T @ Y                                       # [k, d_model]
                reg  = ridge_lambda * torch.eye(k, dtype=AtA.dtype)
                B    = torch.linalg.solve(AtA + reg, AtY)             # [k, d_model]
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

        # Sanity check
        Y_hat     = A_K @ B                                           # [N, d_model]
        recon_err = float((Y - Y_hat).norm() / (Y.norm() + eps))

    sanity = {
        "N_tokens":           N,
        "k_kept":             k,
        "delete_relative_err": round(del_err, 6),
        "recon_relative_err":  round(recon_err, 6),
        "improvement_pct":    round(100.0 * (del_err - recon_err) / (del_err + eps), 2),
    }
    return B, sanity


def apply_down_reconstruction(
    model,
    prune_indices_per_layer:  List[torch.Tensor],
    keep_indices_per_layer:   List[torch.Tensor],
    all_r_per_layer:          List[torch.Tensor],
    method:                   str   = "lstsq",
    ridge_lambda:             float = 1e-5,
    eps:                      float = EPS,
) -> Tuple[object, dict]:
    """
    Prune gate/up rows and refit down_proj weights via least squares.

    Steps
    -----
    1. Clone model.
    2. For each layer: compute optimal B [k, d_model] on calibration data,
       set clone.down_proj.weight[:, keep_indices] = B.T.
    3. Call prune_model_by_layer_indices on the clone (selects keep columns).

    The clone's down_proj for pruned columns is irrelevant (they get removed).
    Only keep_indices columns matter, and they now hold the reconstructed weights.

    Returns (pruned_model, info) where info["sanity_per_layer"] has per-layer errors.
    """
    from .pruning import prune_model_by_layer_indices

    recon_clone = clone_model(model)
    orig_layers  = get_transformer_layers(model)
    clone_layers = get_transformer_layers(recon_clone)

    sanity_per_layer = []

    with torch.no_grad():
        for layer_idx, (orig_l, clone_l, pi, ki, r) in enumerate(
            zip(orig_layers, clone_layers,
                prune_indices_per_layer, keep_indices_per_layer, all_r_per_layer)
        ):
            if len(pi) == 0:
                sanity_per_layer.append({"layer_idx": layer_idx, "n_pruned": 0})
                continue
            if len(ki) == 0:
                sanity_per_layer.append({"layer_idx": layer_idx, "note": "all neurons pruned"})
                continue

            B, sanity = _compute_reconstruction_for_layer(
                orig_l, r, ki, method=method, ridge_lambda=ridge_lambda, eps=eps
            )
            sanity["layer_idx"] = layer_idx
            sanity_per_layer.append(sanity)

            # Set clone's down_proj columns for keep_indices to B.T
            # prune_model_by_layer_indices will then select exactly these columns.
            w_clone = get_mlp_weights(clone_l)
            down    = w_clone["down"]       # [d_model, d_ff]  (actual param ref)
            B_T = B.T.to(dtype=down.dtype).to(device=down.device)  # [d_model, k]
            down.data[:, ki] = B_T

    pruned_model, info = prune_model_by_layer_indices(
        recon_clone, prune_indices_per_layer,
        label=f"recon_{method}",
    )
    info["sanity_per_layer"] = sanity_per_layer

    del recon_clone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pruned_model, info


# ===========================================================================
# Diagnostic helpers for --debug-merge
# ===========================================================================

def _debug_beta_stats(all_asns_per_layer: List[List[dict]]) -> dict:
    """Aggregate beta statistics across all layers."""
    betas = [
        asn["beta"]
        for layer_asns in all_asns_per_layer
        for asn in layer_asns
    ]
    if not betas:
        return {"n": 0}

    bt = torch.tensor(betas, dtype=torch.float32)
    abt = bt.abs()
    sorted_abs = abt.sort().values
    n = len(betas)
    return {
        "n":              n,
        "mean_beta":      round(float(bt.mean()), 6),
        "mean_abs_beta":  round(float(abt.mean()), 6),
        "median_abs_beta": round(float(sorted_abs[n // 2]), 6),
        "max_abs_beta":   round(float(abt.max()), 6),
        "pct_lt_1e-6":    round(100.0 * float((abt < 1e-6).sum()) / n, 2),
        "pct_lt_1e-4":    round(100.0 * float((abt < 1e-4).sum()) / n, 2),
        "pct_lt_1e-2":    round(100.0 * float((abt < 1e-2).sum()) / n, 2),
        "pct_lt_0.1":     round(100.0 * float((abt < 0.1).sum()) / n, 2),
    }


def _debug_update_magnitudes(
    layers,
    all_asns_per_layer: List[List[dict]],
    eps: float = EPS,
) -> dict:
    """
    For each assignment (i → j with beta), compute the down_proj column update:
        update_j += beta * d_i

    Accumulate updates for all pruned neurons targeting the same j,
    then report statistics on ||update_j|| and ||update_j|| / ||d_j||.
    """
    all_abs_update_norms  = []
    all_relative_updates  = []

    for layer, asns in zip(layers, all_asns_per_layer):
        if not asns:
            continue
        w    = get_mlp_weights(layer)
        down = w["down"].detach().float().cpu()  # [d_model, d_ff]

        # Accumulate updates per target
        updates: defaultdict = defaultdict(lambda: torch.zeros(down.shape[0]))
        for asn in asns:
            updates[asn["j"]] += asn["beta"] * down[:, asn["i"]].clone()

        for j, upd in updates.items():
            u_norm = float(upd.norm())
            d_norm = float(down[:, j].norm())
            all_abs_update_norms.append(u_norm)
            all_relative_updates.append(u_norm / (d_norm + eps))

    if not all_abs_update_norms:
        return {"n_targets": 0}

    un = torch.tensor(all_abs_update_norms)
    rn = torch.tensor(all_relative_updates)
    return {
        "n_targets":          len(all_abs_update_norms),
        "mean_update_norm":   round(float(un.mean()), 6),
        "median_update_norm": round(float(un.median()), 6),
        "max_update_norm":    round(float(un.max()), 6),
        "mean_rel_update":    round(float(rn.mean()), 6),
        "median_rel_update":  round(float(rn.median()), 6),
    }


def _debug_isolated_layer_comparison(
    layers,
    prune_indices_per_layer: List[torch.Tensor],
    keep_indices_per_layer:  List[torch.Tensor],
    all_r_per_layer:         List[torch.Tensor],
    weight_asns_per_layer:   List[List[dict]],
    act_asns_per_layer:      List[List[dict]],
    eps: float = EPS,
) -> List[dict]:
    """
    For each layer: compute MLP outputs for original, pure_delete, merge_weight,
    merge_activation on the SAME calibration inputs. This is the 'fair' per-layer
    comparison that isolates reconstruction quality from error accumulation
    across layers.

    Returns list of per-layer dicts with relative errors.
    """
    layer_results = []

    for layer_idx, (layer, pi, ki, r, w_asns, a_asns) in enumerate(zip(
        layers, prune_indices_per_layer, keep_indices_per_layer,
        all_r_per_layer, weight_asns_per_layer, act_asns_per_layer,
    )):
        if len(pi) == 0:
            layer_results.append({"layer_idx": layer_idx, "n_pruned": 0})
            continue

        w      = get_mlp_weights(layer)
        w_gate = w["gate"].detach().float().cpu()
        w_up   = w["up"].detach().float().cpu()
        w_down = w["down"].detach().float().cpu()   # [d_model, d_ff]

        R = r.float().cpu()

        with torch.no_grad():
            A   = F.silu(R @ w_gate.T) * (R @ w_up.T)   # [N, d_ff]
            Y   = A @ w_down.T                           # [N, d_model]
            Y_norm = float(Y.norm()) + eps

            # Pure deletion
            A_K    = A[:, ki]
            Y_del  = A_K @ w_down[:, ki].T
            del_err = float((Y - Y_del).norm()) / Y_norm

            # Merge weight
            w_down_mw = w_down.clone()
            for asn in w_asns:
                col_i = w_down_mw[:, asn["i"]].clone()
                w_down_mw[:, asn["j"]].add_(asn["beta"] * col_i)
            Y_mw   = A_K @ w_down_mw[:, ki].T
            mw_err = float((Y - Y_mw).norm()) / Y_norm

            # Merge activation
            w_down_ma = w_down.clone()
            for asn in a_asns:
                col_i = w_down_ma[:, asn["i"]].clone()
                w_down_ma[:, asn["j"]].add_(asn["beta"] * col_i)
            Y_ma   = A_K @ w_down_ma[:, ki].T
            ma_err = float((Y - Y_ma).norm()) / Y_norm

        layer_results.append({
            "layer_idx":          layer_idx,
            "n_pruned":           int(len(pi)),
            "delete_rel_err":     round(del_err, 6),
            "merge_weight_rel_err":  round(mw_err, 6),
            "merge_act_rel_err":  round(ma_err, 6),
            "weight_vs_delete":   round(100.0 * (del_err - mw_err) / (del_err + eps), 2),
            "act_vs_delete":      round(100.0 * (del_err - ma_err) / (del_err + eps), 2),
        })

    return layer_results


def _debug_logit_comparison(
    model_dict:  Dict[str, object],
    tokenizer,
    device:      str,
    test_prompt: str = "The capital of France is",
) -> dict:
    """
    Run each model through test_prompt and compare final-token logits.

    model_dict: {"original": model, "pure_delete": model, ...}
    Returns dict with ||logits_a - logits_b|| norms for all pairs vs. original.
    """
    enc = tokenizer(
        test_prompt, return_tensors="pt",
        truncation=True, max_length=64,
    )

    logits = {}
    for name, m in model_dict.items():
        enc_dev = {k: v.to(device) for k, v in enc.items()}
        m.eval()
        with torch.no_grad():
            out = m(**enc_dev)
        # Last token logits, float32 CPU
        logits[name] = out.logits[0, -1, :].detach().float().cpu()

    orig = logits.get("original")
    result = {"prompt": test_prompt}
    for name, lg in logits.items():
        result[f"logit_norm_{name}"] = round(float(lg.norm()), 4)
        if orig is not None and name != "original":
            diff = float((orig - lg).norm())
            result[f"diff_vs_original_{name}"] = round(diff, 6)
            result[f"cosine_vs_original_{name}"] = round(
                float(F.cosine_similarity(orig.unsqueeze(0), lg.unsqueeze(0))), 6
            )

    return result



def _debug_end_to_end_mlp_comparison(
    orig_outputs:  list,
    other_outputs: dict,
    eps: float = 1e-8,
) -> dict:
    """
    Compare per-layer MLP outputs from full model forward passes.
    orig_outputs: List[Tensor [N, d_model]] from the original model.
    other_outputs: {method_name: List[Tensor [N, d_model]]} for pruned models.
    Returns aggregate and per-layer relative errors.
    """
    import torch
    n_layers = len(orig_outputs)
    per_layer = []
    agg = {name: [] for name in other_outputs}

    for layer_idx in range(n_layers):
        Y_orig = orig_outputs[layer_idx].float()
        Y_norm = float(Y_orig.norm()) + eps
        entry  = {"layer_idx": layer_idx}
        for name, out_list in other_outputs.items():
            Y_other = out_list[layer_idx].float()
            n_min   = min(Y_orig.shape[0], Y_other.shape[0])
            err     = float((Y_orig[:n_min] - Y_other[:n_min]).norm()) / (
                float(Y_orig[:n_min].norm()) + eps
            )
            entry[f"rel_err_{name}"] = round(err, 6)
            agg[name].append(err)
        per_layer.append(entry)

    aggregate = {}
    for name, errs in agg.items():
        t = torch.tensor(errs)
        aggregate[name] = {
            "mean_rel_err":   round(float(t.mean()), 6),
            "max_rel_err":    round(float(t.max()), 6),
            "median_rel_err": round(float(t.median()), 6),
        }

    return {"aggregate": aggregate, "per_layer": per_layer}


# ===========================================================================
# Diagnostics helper
# ===========================================================================

def _build_layer_diagnostics(all_scores, prune_indices_per_layer, assignments_per_layer):
    layers_diag = []
    for layer_idx, (scores, pi, assignments) in enumerate(
        zip(all_scores, prune_indices_per_layer, assignments_per_layer)
    ):
        if len(pi) == 0:
            layers_diag.append({"layer_idx": layer_idx, "n_pruned": 0})
            continue
        neuron_detail = []
        for asn in assignments:
            metric_val = asn.get("dist", asn.get("residual", 0.0))
            neuron_detail.append({
                "layer_idx":   layer_idx,
                "i":           asn["i"],
                "j":           asn["j"],
                "beta":        round(asn.get("beta", 0.0), 8),
                "metric":      round(metric_val, 8),
                "bound_score": round(float(scores[asn["i"]]), 8),
            })
        betas   = [a.get("beta", 0.0) for a in assignments]
        metrics = [a.get("dist", a.get("residual", 0.0)) for a in assignments]
        targets = [a["j"] for a in assignments]
        from collections import Counter
        tc = Counter(targets)
        layers_diag.append({
            "layer_idx":              layer_idx,
            "n_pruned":               int(len(pi)),
            "n_unique_targets":       len(set(targets)),
            "max_neurons_per_target": max(tc.values()) if tc else 0,
            "avg_beta":               round(float(sum(betas) / max(len(betas), 1)), 8),
            "avg_metric":             round(float(sum(metrics) / max(len(metrics), 1)), 8),
            "max_metric":             round(float(max(metrics)) if metrics else 0.0, 8),
            "neurons":                neuron_detail,
        })
    return layers_diag


# ===========================================================================
# --debug-merge entry point
# ===========================================================================

def run_debug_merge_mode(model, tokenizer, cfg, device, output_dir="results"):
    """
    Diagnose whether pairwise merging improves MLP output reconstruction.
    Reports beta statistics, update magnitudes, isolated per-layer errors,
    end-to-end logit diffs, and end-to-end MLP output errors.
    No PPL evaluation — diagnostic only.
    """
    import json, os, time
    import torch
    from .bound_analysis import CALIBRATION_PROMPTS, _k, compute_bound_scores_and_R, select_by_budget
    from .pruning import prune_model_by_layer_indices

    os.makedirs(output_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    layers = get_transformer_layers(model)
    ALPHAS = [1e-4, 1e-3, 1e-2]

    print("\nComputing bound scores ...")
    all_scores = [compute_bound_scores_and_R(l)[0] for l in layers]

    print("Collecting calibration inputs ...")
    all_r_per_layer = collect_mlp_inputs(
        model, tokenizer, CALIBRATION_PROMPTS, device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )
    print("Collecting original MLP outputs ...")
    orig_mlp_outputs = collect_mlp_outputs(
        model, tokenizer, CALIBRATION_PROMPTS, device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )

    all_debug = []

    for alpha in ALPHAS:
        prune_indices_per_layer = []
        for scores in all_scores:
            pi, _ = select_by_budget(scores, alpha, float(scores.sum()))
            prune_indices_per_layer.append(pi)

        total_pruned = sum(len(pi) for pi in prune_indices_per_layer)
        if total_pruned == 0:
            print(f"\nalpha={_k(alpha)}: no neurons selected - skipping")
            continue

        keep_indices_per_layer = []
        for scores, pi in zip(all_scores, prune_indices_per_layer):
            p_set = set(pi.tolist())
            ki    = torch.tensor([j for j in range(scores.numel()) if j not in p_set], dtype=torch.long)
            keep_indices_per_layer.append(ki)

        total_n = sum(s.numel() for s in all_scores)
        pct     = 100.0 * total_pruned / total_n

        print(f"\n{'=' * 64}")
        print(f"alpha={_k(alpha)}  pruned={total_pruned} ({pct:.3f}%)")
        print(f"{'=' * 64}")

        print("  Computing weight assignments ...")
        weight_asns = [
            compute_merge_assignments_weight(l, pi, ki) if len(pi) > 0 else []
            for l, pi, ki in zip(layers, prune_indices_per_layer, keep_indices_per_layer)
        ]
        print("  Computing activation assignments ...")
        act_asns = [
            compute_merge_assignments_activation(l, pi, ki, r) if len(pi) > 0 else []
            for l, pi, ki, r in zip(layers, prune_indices_per_layer, keep_indices_per_layer, all_r_per_layer)
        ]

        # 1. Beta statistics
        beta_stats_weight = _debug_beta_stats(weight_asns)
        beta_stats_act    = _debug_beta_stats(act_asns)
        print(f"\n  Beta stats (weight): mean_abs={beta_stats_weight['mean_abs_beta']:.4f}  "
              f"median_abs={beta_stats_weight['median_abs_beta']:.4f}  "
              f"max_abs={beta_stats_weight['max_abs_beta']:.4f}  "
              f"pct<0.01={beta_stats_weight['pct_lt_1e-2']:.1f}%  "
              f"pct<0.1={beta_stats_weight['pct_lt_0.1']:.1f}%")
        print(f"  Beta stats (act):    mean_abs={beta_stats_act['mean_abs_beta']:.4f}  "
              f"median_abs={beta_stats_act['median_abs_beta']:.4f}  "
              f"max_abs={beta_stats_act['max_abs_beta']:.4f}  "
              f"pct<0.01={beta_stats_act['pct_lt_1e-2']:.1f}%  "
              f"pct<0.1={beta_stats_act['pct_lt_0.1']:.1f}%")

        # 2. Update magnitudes
        upd_weight = _debug_update_magnitudes(layers, weight_asns)
        upd_act    = _debug_update_magnitudes(layers, act_asns)
        if upd_weight.get("n_targets", 0) > 0:
            print(f"\n  Down updates (weight): mean_norm={upd_weight['mean_update_norm']:.4f}  "
                  f"mean_rel={upd_weight['mean_rel_update']:.4f}")
        if upd_act.get("n_targets", 0) > 0:
            print(f"  Down updates (act):    mean_norm={upd_act['mean_update_norm']:.4f}  "
                  f"mean_rel={upd_act['mean_rel_update']:.4f}")

        # 3. Isolated per-layer comparison
        print("\n  Computing isolated per-layer reconstruction quality ...")
        layer_comp = _debug_isolated_layer_comparison(
            layers, prune_indices_per_layer, keep_indices_per_layer,
            all_r_per_layer, weight_asns, act_asns,
        )
        active = [l for l in layer_comp if l.get("n_pruned", 0) > 0]
        if active:
            avg_del = sum(l["delete_rel_err"] for l in active) / len(active)
            avg_mw  = sum(l["merge_weight_rel_err"] for l in active) / len(active)
            avg_ma  = sum(l["merge_act_rel_err"] for l in active) / len(active)
            print(f"  Isolated mean rel-err ({len(active)} layers):")
            print(f"    pure_delete:       {avg_del:.6f}")
            print(f"    merge_weight:      {avg_mw:.6f}  ({100*(avg_del-avg_mw)/(avg_del+EPS):+.2f}%)")
            print(f"    merge_activation:  {avg_ma:.6f}  ({100*(avg_del-avg_ma)/(avg_del+EPS):+.2f}%)")

        # 4 & 5. End-to-end comparison
        print("\n  Building models for end-to-end comparison ...")
        end2end_models = {"original": model}
        built_models   = {}
        logit_info     = {}
        e2e_comp       = {}

        try:
            built_models["pure_delete"], _ = prune_model_by_layer_indices(
                model, prune_indices_per_layer, label=f"dbg_del_a{_k(alpha)}"
            )
            built_models["merge_weight"], _ = apply_merge_and_prune(
                model, prune_indices_per_layer, weight_asns, label=f"dbg_mw_a{_k(alpha)}"
            )
            built_models["merge_act"], _ = apply_merge_and_prune(
                model, prune_indices_per_layer, act_asns, label=f"dbg_ma_a{_k(alpha)}"
            )
            end2end_models.update(built_models)

            logit_info = _debug_logit_comparison(end2end_models, tokenizer, device)
            print(f"\n  Logit comparison ('{logit_info['prompt']}'):")
            for name in built_models:
                dk = f"diff_vs_original_{name}"
                ck = f"cosine_vs_original_{name}"
                if dk in logit_info:
                    print(f"    {name:<22}  diff={logit_info[dk]:.4f}  cosine={logit_info[ck]:.4f}")

            print("\n  Collecting end-to-end MLP outputs from pruned models ...")
            other_outputs = {}
            for name, m in built_models.items():
                other_outputs[name] = collect_mlp_outputs(
                    m, tokenizer, CALIBRATION_PROMPTS, device,
                    max_seq_len=cfg.get("max_seq_len", 128),
                )

            e2e_comp = _debug_end_to_end_mlp_comparison(orig_mlp_outputs, other_outputs)
            print(f"\n  End-to-end MLP aggregate errors:")
            for name, stats in e2e_comp["aggregate"].items():
                print(f"    {name:<22}  mean={stats['mean_rel_err']:.6f}  max={stats['max_rel_err']:.6f}")

        except Exception as exc:
            logger.error("End-to-end comparison failed for alpha=%s: %s", _k(alpha), exc, exc_info=True)
            logit_info = {"error": str(exc)}
            e2e_comp   = {"error": str(exc)}
        finally:
            for m in built_models.values():
                del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        alpha_report = {
            "alpha":                       alpha,
            "total_pruned":                total_pruned,
            "pct_pruned":                  round(pct, 4),
            "beta_stats":                  {"weight": beta_stats_weight, "activation": beta_stats_act},
            "update_magnitudes":           {"weight": upd_weight, "activation": upd_act},
            "isolated_layer_comparison":   layer_comp,
            "logit_comparison":            logit_info,
            "end2end_mlp_comparison":      e2e_comp,
        }
        all_debug.append(alpha_report)
        p = os.path.join(output_dir, f"debug_merge_alpha{_k(alpha)}_{ts}.json")
        with open(p, "w") as f:
            json.dump(alpha_report, f, indent=2, default=str)
        print(f"\n  Debug report: {p}")

    full_path = os.path.join(output_dir, f"debug_merge_full_{ts}.json")
    with open(full_path, "w") as f:
        json.dump({"timestamp": ts, "alphas": all_debug}, f, indent=2, default=str)
    logger.info("Full debug report: %s", full_path)
    print(f"\nFull debug report: {full_path}\n")


# ===========================================================================
# --bound-merge entry point (updated with reconstruction methods)
# ===========================================================================

def run_bound_merge_mode(model, tokenizer, cfg, device, output_dir="results"):
    """
    Compare pure deletion, pairwise merging, and down-projection reconstruction.

    Methods per alpha:
      pure_delete
      merge_weight_similarity
      merge_activation_similarity
      down_reconstruction_lstsq
      down_reconstruction_ridge_1e-6 / 1e-5 / 1e-4 / 1e-3
    """
    import json, os, time
    import torch
    from .bound_analysis import CALIBRATION_PROMPTS, _k, compute_bound_scores_and_R, select_by_budget
    from .evaluation import evaluate_perplexity, load_eval_dataset, run_generation_tests
    from .flops import estimate_mlp_flops
    from .model_utils import count_parameters
    from .pruning import prune_model_by_layer_indices, verify_forward_pass

    os.makedirs(output_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    layers = get_transformer_layers(model)
    ALPHAS = [1e-4, 1e-3, 1e-2]

    def _build_pure_delete(model, pi, ki, all_r, wa, aa):
        return prune_model_by_layer_indices(model, pi, label="pure_delete")

    def _build_weight(model, pi, ki, all_r, wa, aa):
        return apply_merge_and_prune(model, pi, wa, label="merge_weight")

    def _build_act(model, pi, ki, all_r, wa, aa):
        return apply_merge_and_prune(model, pi, aa, label="merge_act")

    def _make_recon(method, lam):
        def _build(model, pi, ki, all_r, wa, aa):
            return apply_down_reconstruction(model, pi, ki, all_r, method=method, ridge_lambda=lam)
        return _build

    METHODS = [
        ("pure_delete",                    _build_pure_delete),
        ("merge_weight_similarity",        _build_weight),
        ("merge_activation_similarity",    _build_act),
        ("down_reconstruction_lstsq",      _make_recon("lstsq", 1e-5)),
        ("down_reconstruction_ridge_1e-6", _make_recon("ridge", 1e-6)),
        ("down_reconstruction_ridge_1e-5", _make_recon("ridge", 1e-5)),
        ("down_reconstruction_ridge_1e-4", _make_recon("ridge", 1e-4)),
        ("down_reconstruction_ridge_1e-3", _make_recon("ridge", 1e-3)),
    ]

    print("\nComputing bound scores for all layers ...")
    all_scores = [compute_bound_scores_and_R(l)[0] for l in layers]

    print("Collecting calibration hidden states ...")
    all_r_per_layer = collect_mlp_inputs(
        model, tokenizer, CALIBRATION_PROMPTS, device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )

    use_fallback = cfg.get("use_fallback_corpus", True)
    n_eval       = cfg.get("bound_analysis_eval_samples", 64)
    eval_texts   = load_eval_dataset(n_eval, use_fallback_corpus=use_fallback)

    baseline_params = count_parameters(model)
    baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))

    print("Computing baseline PPL ...")
    bp           = evaluate_perplexity(
        model, tokenizer, texts=eval_texts,
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 4),
        device=device,
    )
    baseline_ppl = bp["perplexity"]
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    all_results = []

    for alpha in ALPHAS:
        prune_indices_per_layer = []
        for scores in all_scores:
            pi, _ = select_by_budget(scores, alpha, float(scores.sum()))
            prune_indices_per_layer.append(pi)

        total_pruned = sum(len(pi) for pi in prune_indices_per_layer)
        total_n      = sum(s.numel() for s in all_scores)
        pct          = 100.0 * total_pruned / total_n if total_n else 0.0

        if total_pruned == 0:
            print(f"alpha={_k(alpha)}: 0 neurons selected - skipping\n")
            continue

        keep_indices_per_layer = []
        for scores, pi in zip(all_scores, prune_indices_per_layer):
            p_set = set(pi.tolist())
            ki    = torch.tensor(
                [j for j in range(scores.numel()) if j not in p_set], dtype=torch.long
            )
            keep_indices_per_layer.append(ki)

        print(f"{'=' * 72}")
        print(f"alpha={_k(alpha)}  pruned={total_pruned} ({pct:.3f}%)")
        print(f"{'=' * 72}\n")

        weight_asns = [
            compute_merge_assignments_weight(l, pi, ki) if len(pi) > 0 else []
            for l, pi, ki in zip(layers, prune_indices_per_layer, keep_indices_per_layer)
        ]
        act_asns = [
            compute_merge_assignments_activation(l, pi, ki, r) if len(pi) > 0 else []
            for l, pi, ki, r in zip(
                layers, prune_indices_per_layer, keep_indices_per_layer, all_r_per_layer
            )
        ]

        for method_name, builder in METHODS:
            print(f"\n  [{method_name}]")
            row = {
                "alpha":        alpha,
                "method":       method_name,
                "total_pruned": total_pruned,
                "pct_pruned":   round(pct, 4),
                "baseline_ppl": round(baseline_ppl, 4),
            }
            pruned_model = None
            try:
                pruned_model, build_info = builder(
                    model, prune_indices_per_layer, keep_indices_per_layer,
                    all_r_per_layer, weight_asns, act_asns,
                )
                fp_ok    = verify_forward_pass(pruned_model, tokenizer, device)
                ppl_info = evaluate_perplexity(
                    pruned_model, tokenizer, texts=eval_texts,
                    max_seq_len=cfg.get("max_seq_len", 512),
                    batch_size=cfg.get("batch_size", 4),
                    device=device,
                )
                ppl = ppl_info["perplexity"]

                p_params = count_parameters(pruned_model)
                p_flops  = estimate_mlp_flops(pruned_model, seq_len=cfg.get("max_seq_len", 512))
                flop_red = 100.0 * (1.0 - p_flops["total_flops"] / baseline_flops["total_flops"])
                mlp_red  = 100.0 * (1.0 - p_params["mlp"] / baseline_params["mlp"])

                gens = run_generation_tests(pruned_model, tokenizer, device=device)

                row.update({
                    "perplexity":          round(ppl, 4),
                    "perplexity_delta":    round(ppl - baseline_ppl, 4),
                    "forward_pass_ok":     fp_ok,
                    "mlp_params_red_pct":  round(mlp_red, 4),
                    "flops_red_pct":       round(flop_red, 4),
                    "generation_examples": gens,
                })
                if "sanity_per_layer" in build_info:
                    row["sanity_per_layer"] = build_info["sanity_per_layer"]

                print(
                    f"    PPL={ppl:.4f}  dPPL={ppl - baseline_ppl:+.4f}"
                    f"  MLP-{mlp_red:.2f}%  FLOPs-{flop_red:.2f}%"
                )
                if "sanity_per_layer" in build_info:
                    active_s = [s for s in build_info["sanity_per_layer"]
                                if s.get("n_pruned", 1) > 0 and "recon_relative_err" in s]
                    if active_s:
                        avg_recon = sum(s["recon_relative_err"] for s in active_s) / len(active_s)
                        avg_del   = sum(s["delete_relative_err"] for s in active_s) / len(active_s)
                        print(
                            f"    Isolated layer err: delete={avg_del:.6f}  "
                            f"recon={avg_recon:.6f}  "
                            f"improvement={100*(avg_del-avg_recon)/(avg_del+EPS):.2f}%"
                        )

            except Exception as exc:
                logger.error("Method [%s] alpha=%s failed: %s", method_name, _k(alpha), exc, exc_info=True)
                row["notes"] = f"ERROR: {exc}"
            finally:
                if pruned_model is not None:
                    del pruned_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results.append(row)

        diag = {
            "alpha":                  alpha,
            "total_pruned":           total_pruned,
            "weight_diagnostics":     _build_layer_diagnostics(
                all_scores, prune_indices_per_layer, weight_asns
            ),
            "activation_diagnostics": _build_layer_diagnostics(
                all_scores, prune_indices_per_layer, act_asns
            ),
        }
        diag_path = os.path.join(output_dir, f"merge_diagnostics_alpha{_k(alpha)}_{ts}.json")
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2, default=str)
        print(f"\n  Diagnostics: {diag_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("BOUND MERGE + RECONSTRUCTION SUMMARY")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'─' * 80}")
    print(
        f"  {'alpha':>9}  {'method':<38}  {'pruned':>7}"
        f"  {'PPL':>9}  {'dPPL':>9}  {'MLP-':>7}"
    )
    print(f"{'─' * 80}")
    for r in all_results:
        if "perplexity" in r:
            print(
                f"  {_k(r['alpha']):>9}  {r['method']:<38}"
                f"  {r['total_pruned']:>7}"
                f"  {r['perplexity']:>9.4f}"
                f"  {r['perplexity_delta']:>+9.4f}"
                f"  {r.get('mlp_params_red_pct', 0.0):>6.2f}%"
            )
        else:
            print(f"  {_k(r['alpha']):>9}  {r['method']:<38}  {r.get('notes', 'ERROR')}")
    print(f"{'=' * 80}\n")

    report = {
        "timestamp":    ts,
        "mode":         "bound_merge",
        "baseline_ppl": baseline_ppl,
        "results":      all_results,
    }
    path = os.path.join(output_dir, f"bound_merge_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Bound merge report: %s", path)
    print(f"Report saved to: {path}\n")


# ===========================================================================
# Held-out reconstruction evaluator
# ===========================================================================

def _compute_held_out_reconstruction(
    layers,
    prune_indices_per_layer,
    keep_indices_per_layer,
    assignments_per_layer,
    train_r_per_layer,
    heldout_r_per_layer,
    eps: float = EPS,
) -> list:
    """
    Evaluate merge assignment quality on both training and held-out calibration data.

    For each pruned layer, compute:
      - In-sample reconstruction error  (using train_r — same data used to compute beta)
      - Held-out reconstruction error   (using heldout_r — unseen at assignment time)

    A large gap between in-sample and held-out error indicates overfitting of beta
    to the (small) calibration set.

    Errors are relative:  ||Y - Y_approx|| / ||Y||
    where Y = A_full @ w_down.T  (full MLP output including pruned neurons).
    """
    import torch.nn.functional as F

    layer_results = []
    for layer_idx, (layer, pi, ki, asns, r_tr, r_ho) in enumerate(zip(
        layers,
        prune_indices_per_layer,
        keep_indices_per_layer,
        assignments_per_layer,
        train_r_per_layer,
        heldout_r_per_layer,
    )):
        if len(pi) == 0:
            layer_results.append({"layer_idx": layer_idx, "n_pruned": 0})
            continue

        w       = get_mlp_weights(layer)
        w_gate  = w["gate"].detach().float().cpu()   # [d_ff, d_model]
        w_up    = w["up"].detach().float().cpu()     # [d_ff, d_model]
        w_down  = w["down"].detach().float().cpu()   # [d_model, d_ff]

        # Build compensated down projection (same weight as used during pruning)
        w_down_comp = w_down.clone()
        for asn in asns:
            i_col = w_down[:, asn["i"]].clone()
            w_down_comp[:, asn["j"]].add_(asn["beta"] * i_col)

        def _layer_errors(R: torch.Tensor):
            """
            Returns (delete_err, merge_err) for a given input matrix R [N, d_model].
            """
            with torch.no_grad():
                A = F.silu(R @ w_gate.T) * (R @ w_up.T)  # [N, d_ff]
                Y = A @ w_down.T                           # [N, d_model] full output
                Y_del   = A[:, ki] @ w_down[:, ki].T      # delete-only approximation
                Y_merge = A[:, ki] @ w_down_comp[:, ki].T # merge-compensated approx
                Y_norm  = float(Y.norm()) + eps
                del_err   = float((Y - Y_del).norm())   / Y_norm
                merge_err = float((Y - Y_merge).norm()) / Y_norm
            return del_err, merge_err

        tr_del,  tr_mrg  = _layer_errors(r_tr.float().cpu())
        ho_del,  ho_mrg  = _layer_errors(r_ho.float().cpu())

        def _imp(d, m):
            return round(100.0 * (d - m) / (d + eps), 2)

        layer_results.append({
            "layer_idx":              layer_idx,
            "n_pruned":               int(len(pi)),
            "train_delete_err":       round(tr_del, 6),
            "train_merge_err":        round(tr_mrg, 6),
            "train_improvement_pct":  _imp(tr_del, tr_mrg),
            "heldout_delete_err":     round(ho_del, 6),
            "heldout_merge_err":      round(ho_mrg, 6),
            "heldout_improvement_pct": _imp(ho_del, ho_mrg),
            # Positive = merge is overfitting train, negative = generalizes better
            "overfit_gap_pct":        round(_imp(tr_del, tr_mrg) - _imp(ho_del, ho_mrg), 2),
        })

    return layer_results


# ===========================================================================
# --bound-merge-stable entry point
# ===========================================================================

def run_bound_merge_stable_mode(model, tokenizer, cfg, device, output_dir="results"):
    """
    Test stabilized activation-merge variants.

    Variants compared (per alpha):
      A. activation_merge_clip_{c}  — original beta, clipped to [-c, c]
         c ∈ {2.0, 1.0, 0.5, 0.25}
      B. activation_merge_ridge_lam{l}  — ridge-regularized beta
         lambda ∈ {1e-4, 1e-3, 1e-2, 0.1, 1.0}
      C. activation_merge_ridge_lam{l}_clip{c}  — ridge + clipping
         (lambda, clip) ∈ {(1e-2, 0.5), (0.1, 0.5), (1.0, 0.5)}

    Plus baselines:  pure_delete, merge_activation_original

    For each variant, reports:
      - Beta stats: mean/median/max abs, raw vs clipped
      - Update magnitudes on down_proj
      - In-sample AND held-out MLP reconstruction error (overfitting detection)
      - WikiText-2 perplexity

    Train/held-out split:
      RECONSTRUCTION_TRAIN_PROMPTS  → compute beta assignments
      RECONSTRUCTION_HELDOUT_PROMPTS → evaluate reconstruction generalization
    """
    import json, os, time
    import torch

    from .bound_analysis import _k, compute_bound_scores_and_R, select_by_budget
    from .evaluation import evaluate_perplexity, load_eval_dataset, run_generation_tests
    from .flops import estimate_mlp_flops
    from .model_utils import count_parameters
    from .pruning import prune_model_by_layer_indices, verify_forward_pass

    os.makedirs(output_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    layers = get_transformer_layers(model)
    ALPHAS = [1e-4, 1e-3, 1e-2]

    # ------------------------------------------------------------------
    # Variant specs: (name, ridge_lambda, clip_value)
    # ------------------------------------------------------------------
    VARIANTS = [
        # baseline
        ("pure_delete",                          None,  None),
        ("merge_activation_original",            0.0,   None),
        # A — clipped beta only
        ("merge_activation_clip_2.0",            0.0,   2.0),
        ("merge_activation_clip_1.0",            0.0,   1.0),
        ("merge_activation_clip_0.5",            0.0,   0.5),
        ("merge_activation_clip_0.25",           0.0,   0.25),
        # B — ridge only
        ("merge_activation_ridge_1e-4",          1e-4,  None),
        ("merge_activation_ridge_1e-3",          1e-3,  None),
        ("merge_activation_ridge_1e-2",          1e-2,  None),
        ("merge_activation_ridge_0.1",           0.1,   None),
        ("merge_activation_ridge_1.0",           1.0,   None),
        # C — ridge + clip
        ("merge_activation_ridge_1e-2_clip_0.5", 1e-2,  0.5),
        ("merge_activation_ridge_0.1_clip_0.5",  0.1,   0.5),
        ("merge_activation_ridge_1.0_clip_0.5",  1.0,   0.5),
    ]

    # ------------------------------------------------------------------
    # Compute scores + collect calibration inputs
    # ------------------------------------------------------------------
    print("\nComputing bound scores for all layers ...")
    all_scores = [compute_bound_scores_and_R(l)[0] for l in layers]

    print("Collecting TRAIN calibration inputs ...")
    train_r_per_layer = collect_mlp_inputs(
        model, tokenizer, RECONSTRUCTION_TRAIN_PROMPTS, device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )
    print("Collecting HELD-OUT calibration inputs ...")
    heldout_r_per_layer = collect_mlp_inputs(
        model, tokenizer, RECONSTRUCTION_HELDOUT_PROMPTS, device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )

    # ------------------------------------------------------------------
    # Load eval dataset + baseline
    # ------------------------------------------------------------------
    use_fallback = cfg.get("use_fallback_corpus", True)
    n_eval       = cfg.get("bound_analysis_eval_samples", 64)
    eval_texts   = load_eval_dataset(n_eval, use_fallback_corpus=use_fallback)

    baseline_params = count_parameters(model)
    baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))

    print("Computing baseline PPL ...")
    bp           = evaluate_perplexity(
        model, tokenizer, texts=eval_texts,
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 4),
        device=device,
    )
    baseline_ppl = bp["perplexity"]
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    all_results = []

    # ------------------------------------------------------------------
    # Main loop over alphas
    # ------------------------------------------------------------------
    for alpha in ALPHAS:
        prune_indices_per_layer = []
        for scores in all_scores:
            pi, _ = select_by_budget(scores, alpha, float(scores.sum()))
            prune_indices_per_layer.append(pi)

        total_pruned = sum(len(pi) for pi in prune_indices_per_layer)
        total_n      = sum(s.numel() for s in all_scores)
        pct          = 100.0 * total_pruned / total_n if total_n else 0.0

        if total_pruned == 0:
            print(f"alpha={_k(alpha)}: 0 neurons selected — skipping\n")
            continue

        keep_indices_per_layer = []
        for scores, pi in zip(all_scores, prune_indices_per_layer):
            p_set = set(pi.tolist())
            ki    = torch.tensor(
                [j for j in range(scores.numel()) if j not in p_set], dtype=torch.long
            )
            keep_indices_per_layer.append(ki)

        print(f"\n{'=' * 72}")
        print(f"alpha={_k(alpha)}  pruned={total_pruned} ({pct:.3f}%)")
        print(f"{'=' * 72}")

        # ------------------------------------------------------------------
        # Loop over variants
        # ------------------------------------------------------------------
        for vname, ridge_lam, clip_val in VARIANTS:
            print(f"\n  [{vname}]")
            row = {
                "alpha":        alpha,
                "method":       vname,
                "total_pruned": total_pruned,
                "pct_pruned":   round(pct, 4),
                "baseline_ppl": round(baseline_ppl, 4),
                "ridge_lambda": ridge_lam,
                "clip_value":   clip_val,
            }
            pruned_model = None
            try:
                # --- Build pruned model ---
                if vname == "pure_delete":
                    pruned_model, _ = prune_model_by_layer_indices(
                        model, prune_indices_per_layer, label=f"stable_del_a{_k(alpha)}"
                    )
                    asns_per_layer = [[] for _ in layers]
                else:
                    asns_per_layer = [
                        compute_merge_assignments_activation(
                            l, pi, ki, r,
                            ridge_lambda=ridge_lam,
                            clip_value=clip_val,
                        ) if len(pi) > 0 else []
                        for l, pi, ki, r in zip(
                            layers, prune_indices_per_layer,
                            keep_indices_per_layer, train_r_per_layer
                        )
                    ]
                    pruned_model, _ = apply_merge_and_prune(
                        model, prune_indices_per_layer, asns_per_layer,
                        label=f"stable_{vname[:20]}_a{_k(alpha)}"
                    )

                # --- Beta stats ---
                if vname != "pure_delete":
                    beta_stats = _debug_beta_stats(asns_per_layer)
                    row["beta_mean_abs"]   = beta_stats["mean_abs_beta"]
                    row["beta_median_abs"] = beta_stats["median_abs_beta"]
                    row["beta_max_abs"]    = beta_stats["max_abs_beta"]
                    row["beta_pct_lt_0.1"] = beta_stats["pct_lt_0.1"]

                    # Raw beta stats (before clipping) if clip was applied
                    if clip_val is not None:
                        raw_betas = [asn["beta_raw"] for asns in asns_per_layer for asn in asns]
                        if raw_betas:
                            row["beta_raw_mean_abs"] = round(
                                sum(abs(b) for b in raw_betas) / len(raw_betas), 6
                            )
                            row["beta_raw_max_abs"]  = round(max(abs(b) for b in raw_betas), 6)
                            row["beta_pct_clipped"]  = round(
                                100.0 * sum(1 for b in raw_betas if abs(b) > clip_val) / len(raw_betas), 2
                            )

                    # Update magnitudes on down_proj
                    upd = _debug_update_magnitudes(layers, asns_per_layer)
                    row["down_update_mean_norm"]  = upd.get("mean_update_norm", 0.0)
                    row["down_update_mean_rel"]   = upd.get("mean_rel_update", 0.0)

                    # Train / held-out reconstruction
                    ho_results = _compute_held_out_reconstruction(
                        layers,
                        prune_indices_per_layer,
                        keep_indices_per_layer,
                        asns_per_layer,
                        train_r_per_layer,
                        heldout_r_per_layer,
                    )
                    active_ho = [r for r in ho_results if r.get("n_pruned", 0) > 0]
                    if active_ho:
                        row["train_merge_err_mean"]      = round(
                            sum(r["train_merge_err"]   for r in active_ho) / len(active_ho), 6
                        )
                        row["train_delete_err_mean"]     = round(
                            sum(r["train_delete_err"]  for r in active_ho) / len(active_ho), 6
                        )
                        row["heldout_merge_err_mean"]    = round(
                            sum(r["heldout_merge_err"]  for r in active_ho) / len(active_ho), 6
                        )
                        row["heldout_delete_err_mean"]   = round(
                            sum(r["heldout_delete_err"] for r in active_ho) / len(active_ho), 6
                        )
                        row["train_improvement_pct"]     = round(
                            sum(r["train_improvement_pct"]   for r in active_ho) / len(active_ho), 2
                        )
                        row["heldout_improvement_pct"]   = round(
                            sum(r["heldout_improvement_pct"] for r in active_ho) / len(active_ho), 2
                        )
                        row["overfit_gap_pct"]           = round(
                            sum(r["overfit_gap_pct"] for r in active_ho) / len(active_ho), 2
                        )
                        row["per_layer_reconstruction"]  = active_ho

                        print(
                            f"    train:  del={row['train_delete_err_mean']:.6f}  "
                            f"merge={row['train_merge_err_mean']:.6f}  "
                            f"imp={row['train_improvement_pct']:+.2f}%"
                        )
                        print(
                            f"    heldout:del={row['heldout_delete_err_mean']:.6f}  "
                            f"merge={row['heldout_merge_err_mean']:.6f}  "
                            f"imp={row['heldout_improvement_pct']:+.2f}%  "
                            f"overfit={row['overfit_gap_pct']:+.2f}%"
                        )
                        if clip_val is not None and "beta_pct_clipped" in row:
                            print(
                                f"    beta clipped: {row['beta_pct_clipped']:.1f}% of neurons  "
                                f"raw_max={row.get('beta_raw_max_abs', 0.0):.4f}"
                            )
                        print(
                            f"    beta: mean_abs={row['beta_mean_abs']:.4f}  "
                            f"max_abs={row['beta_max_abs']:.4f}  "
                            f"down_update_rel={row['down_update_mean_rel']:.4f}"
                        )

                # --- PPL ---
                fp_ok    = verify_forward_pass(pruned_model, tokenizer, device)
                ppl_info = evaluate_perplexity(
                    pruned_model, tokenizer, texts=eval_texts,
                    max_seq_len=cfg.get("max_seq_len", 512),
                    batch_size=cfg.get("batch_size", 4),
                    device=device,
                )
                ppl = ppl_info["perplexity"]

                p_params = count_parameters(pruned_model)
                p_flops  = estimate_mlp_flops(pruned_model, seq_len=cfg.get("max_seq_len", 512))
                flop_red = 100.0 * (1.0 - p_flops["total_flops"] / baseline_flops["total_flops"])
                mlp_red  = 100.0 * (1.0 - p_params["mlp"] / baseline_params["mlp"])

                row.update({
                    "perplexity":         round(ppl, 4),
                    "perplexity_delta":   round(ppl - baseline_ppl, 4),
                    "forward_pass_ok":    fp_ok,
                    "mlp_params_red_pct": round(mlp_red, 4),
                    "flops_red_pct":      round(flop_red, 4),
                })
                print(
                    f"    PPL={ppl:.4f}  dPPL={ppl - baseline_ppl:+.4f}"
                    f"  MLP-{mlp_red:.2f}%  FLOPs-{flop_red:.2f}%"
                )

            except Exception as exc:
                logger.error("Variant [%s] alpha=%s failed: %s", vname, _k(alpha), exc, exc_info=True)
                row["notes"] = f"ERROR: {exc}"
            finally:
                if pruned_model is not None:
                    del pruned_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results.append(row)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("STABLE ACTIVATION MERGE SUMMARY")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'─' * 90}")
    hdr = (f"  {'alpha':>9}  {'method':<44}  {'PPL':>9}  {'dPPL':>9}"
           f"  {'HO-imp':>7}  {'overfit':>7}")
    print(hdr)
    print(f"{'─' * 90}")
    for r in all_results:
        if "perplexity" in r:
            ho_imp  = r.get("heldout_improvement_pct", float("nan"))
            overfit = r.get("overfit_gap_pct",         float("nan"))
            print(
                f"  {_k(r['alpha']):>9}  {r['method']:<44}"
                f"  {r['perplexity']:>9.4f}  {r['perplexity_delta']:>+9.4f}"
                f"  {ho_imp:>+7.2f}%  {overfit:>+7.2f}%"
            )
        else:
            print(f"  {_k(r['alpha']):>9}  {r['method']:<44}  {r.get('notes', 'ERROR')}")
    print(f"{'=' * 90}\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    report = {
        "timestamp":    ts,
        "mode":         "bound_merge_stable",
        "baseline_ppl": baseline_ppl,
        "results":      all_results,
    }
    path = os.path.join(output_dir, f"bound_merge_stable_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # CSV (summary without per-layer reconstruction lists)
    import csv
    csv_path = os.path.join(output_dir, f"bound_merge_stable_{ts}.csv")
    csv_keys = [
        "alpha", "method", "ridge_lambda", "clip_value",
        "total_pruned", "pct_pruned", "baseline_ppl",
        "perplexity", "perplexity_delta",
        "beta_mean_abs", "beta_median_abs", "beta_max_abs",
        "beta_raw_max_abs", "beta_pct_clipped",
        "down_update_mean_norm", "down_update_mean_rel",
        "train_delete_err_mean", "train_merge_err_mean", "train_improvement_pct",
        "heldout_delete_err_mean", "heldout_merge_err_mean", "heldout_improvement_pct",
        "overfit_gap_pct",
        "mlp_params_red_pct", "flops_red_pct",
        "forward_pass_ok", "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logger.info("Stable merge report: %s", path)
    print(f"Report saved to: {path}")
    print(f"CSV saved to:    {csv_path}\n")
