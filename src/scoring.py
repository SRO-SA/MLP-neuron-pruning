"""
scoring.py
==========
Static neuron importance scores for SwiGLU MLP pruning, plus activation-based
diagnostic scores and correlation utilities.

───────────────────────────────────────────────────────────────────────────
Theory: RMSNorm-bounded neuron contribution
───────────────────────────────────────────────────────────────────────────
The MLP input r comes from a RMSNorm layer with weight γ.  For any input x:

    r_k = x_k / RMS(x) * γ_k     →    ||r||_2 ≤ R = √d_model * ||γ||_∞

Neuron i's contribution to the MLP output (single token vector r):

    c_i(r) = SiLU(r · w_gate_i) * (r · w_up_i) * w_down_i   ∈ ℝ^d_model

Upper-bounding with |SiLU(x)| ≤ |x| and Cauchy-Schwarz:

    ||c_i(r)|| ≤ R² * (||w_gate_i|| * ||w_up_i|| + |w_gate_i · w_up_i|) / 2
                     * ||w_down_i||

The (norm_product + dot_product)/2 term is tighter than the pure norm product:
it additionally penalises neurons where the gate and up vectors are near-orthogonal
(dot ≈ 0 → less output even if both norms are large).

───────────────────────────────────────────────────────────────────────────
Activation-based score (diagnostic)
───────────────────────────────────────────────────────────────────────────
Given calibration token vectors {r_t}, neuron i's actual average contribution is:

    activation_score_i = mean_t(|SiLU(r_t · w_gate_i) * (r_t · w_up_i)|) * ||w_down_i||

This is a data-driven lower bound on the true importance, and serves as a
reference to evaluate whether the static scores are well-calibrated.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .model_utils import (
    get_mlp_module,
    get_mlp_weights,
    get_rmsnorm_before_mlp,
    get_transformer_layers,
)

logger = logging.getLogger(__name__)

ScoringMethod = Literal["random", "down_norm", "product_norm", "rmsnorm_bound_angle"]

ALL_STATIC_METHODS: List[str] = [
    "random", "down_norm", "product_norm", "rmsnorm_bound_angle"
]


# ===========================================================================
# ── Static scoring ──────────────────────────────────────────────────────────
# ===========================================================================

def compute_scores(
    layer,
    method: ScoringMethod,
    seed: int = 42,
) -> torch.Tensor:
    """
    Compute a 1-D importance score tensor of length d_ff for *layer*.

    Lower score ⟹ neuron is less important ⟹ pruned first.

    Returns
    -------
    scores : torch.Tensor  shape [d_ff], dtype float32, on CPU
    """
    weights = get_mlp_weights(layer)
    w_gate  = weights["gate"].float().cpu()   # [d_ff, d_model]
    w_up    = weights["up"].float().cpu()     # [d_ff, d_model]
    w_down  = weights["down"].float().cpu()   # [d_model, d_ff]
    d_ff    = weights["d_ff"]
    d_model = weights["d_model"]

    # Neuron-wise vector norms
    gate_row_norm = w_gate.norm(dim=1)   # [d_ff]  ||w_gate_i||
    up_row_norm   = w_up.norm(dim=1)     # [d_ff]  ||w_up_i||
    down_col_norm = w_down.norm(dim=0)   # [d_ff]  ||w_down_i|| (norm over d_model axis)

    assert gate_row_norm.shape == (d_ff,)
    assert up_row_norm.shape   == (d_ff,)
    assert down_col_norm.shape == (d_ff,)

    # ── A: random ────────────────────────────────────────────────────────────
    if method == "random":
        rng = torch.Generator()
        rng.manual_seed(seed)
        return torch.rand(d_ff, generator=rng)

    # ── B: down_norm ─────────────────────────────────────────────────────────
    # score_i = ||w_down_i||_2
    # Lower = neuron outputs less to the residual stream regardless of input.
    if method == "down_norm":
        return down_col_norm

    # ── C: product_norm ──────────────────────────────────────────────────────
    # score_i = ||w_gate_i|| * ||w_up_i|| * ||w_down_i||
    # Product of all three norms; similar to down_norm but weights by gate/up capacity.
    if method == "product_norm":
        return gate_row_norm * up_row_norm * down_col_norm

    # ── D: rmsnorm_bound_angle (proposed) ────────────────────────────────────
    # score_i = R² * (||w_gate_i|| * ||w_up_i|| + |w_gate_i · w_up_i|) / 2 * ||w_down_i||
    #
    # Key difference from product_norm: the dot product term rewards neurons
    # where gate/up vectors are PARALLEL (cos_sim ≈ 1).  For near-orthogonal
    # gate/up pairs, the (mixed_term / product_norm) ratio ≈ 0.5, lowering
    # the score and making them preferentially pruned.
    if method == "rmsnorm_bound_angle":
        rmsnorm  = get_rmsnorm_before_mlp(layer)
        gamma    = rmsnorm.weight.float().cpu()   # [d_model]

        # R² = d_model * ||γ||_∞²
        # (R is a constant across all neurons in this layer; does not affect ranking)
        R_squared = float(d_model) * float(gamma.abs().max()) ** 2

        # |w_gate_i · w_up_i|  (element-wise multiply then sum along d_model)
        dot_gate_up = (w_gate * w_up).sum(dim=1).abs()   # [d_ff]

        mixed_term = (gate_row_norm * up_row_norm + dot_gate_up) / 2.0

        return R_squared * mixed_term * down_col_norm

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

    Neurons with the LOWEST scores are pruned first.

    Returns
    -------
    keep_indices : sorted 1-D LongTensor of length round((1 - prune_ratio) * d_ff)
    """
    d_ff   = scores.numel()
    n_keep = max(1, int(round(d_ff * (1.0 - prune_ratio))))

    sorted_indices = torch.argsort(scores)           # ascending: lowest score first
    keep_indices   = sorted_indices[d_ff - n_keep:]  # top n_keep (highest scores) → kept
    keep_indices   = keep_indices.sort().values       # sort ascending for stable weight layout

    assert keep_indices.numel() == n_keep
    return keep_indices


def get_prune_indices(
    scores: torch.Tensor,
    prune_ratio: float,
    seed: int = 42,
) -> torch.Tensor:
    """Return the indices to PRUNE (complement of get_keep_indices)."""
    d_ff         = scores.numel()
    keep         = get_keep_indices(scores, prune_ratio, seed)
    keep_set     = set(keep.tolist())
    prune_idx    = torch.tensor(
        [i for i in range(d_ff) if i not in keep_set], dtype=torch.long
    )
    return prune_idx


# ===========================================================================
# -- Activation-based diagnostic score
# ===========================================================================

def compute_activation_scores_all_layers(
    model,
    tokenizer,
    prompts,
    device,
    max_seq_len=128,
    chunk_size=256,
):
    """
    Compute activation-based neuron importance scores for ALL transformer layers.

    For neuron i in a given layer:
        activation_score_i = mean_t( |SiLU(r_t . w_gate_i) * (r_t . w_up_i)| )
                             * ||w_down_i||

    where r_t is the MLP input (post-RMSNorm hidden state) at token t.

    Parameters
    ----------
    model   : unpruned model
    prompts : list of calibration text strings
    device  : 'cuda' or 'cpu'

    Returns
    -------
    scores_per_layer : List[Tensor]  one [d_ff] tensor per layer
    """
    layers   = get_transformer_layers(model)
    n_layers = len(layers)

    # Capture MLP inputs via forward pre-hooks.
    # register_forward_pre_hook calls hook(module, inputs) — 2 args, NOT 3.
    captured = [[] for _ in range(n_layers)]
    handles  = []

    try:
        for i, layer in enumerate(layers):
            def _make_hook(idx):
                def _hook(module, inputs):
                    captured[idx].append(inputs[0].detach().float().cpu())
                return _hook
            handles.append(get_mlp_module(layer).register_forward_pre_hook(_make_hook(i)))

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                enc = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=max_seq_len,
                ).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()

    scores_per_layer = []

    for i, layer in enumerate(layers):
        if not captured[i]:
            scores_per_layer.append(torch.zeros(1))
            continue

        # Flatten to [N_total, d_model]
        all_r = torch.cat(
            [x.reshape(-1, x.shape[-1]) for x in captured[i]], dim=0
        )

        w      = get_mlp_weights(layer)
        w_gate = w["gate"].float().cpu()   # [d_ff, d_model]
        w_up   = w["up"].float().cpu()     # [d_ff, d_model]
        w_down = w["down"].float().cpu()   # [d_model, d_ff]
        d_ff   = w["d_ff"]

        # Accumulate |SiLU(g_i) * u_i| in chunks (memory control)
        sum_abs_act = torch.zeros(d_ff)
        n_tokens    = 0

        for start in range(0, all_r.shape[0], chunk_size):
            r_chunk      = all_r[start : start + chunk_size]     # [C, d_model]
            g            = r_chunk @ w_gate.T                    # [C, d_ff]
            u            = r_chunk @ w_up.T                      # [C, d_ff]
            a            = F.silu(g) * u                         # [C, d_ff]
            sum_abs_act += a.abs().sum(dim=0)                    # [d_ff]
            n_tokens    += r_chunk.shape[0]

        mean_abs_act  = sum_abs_act / max(n_tokens, 1)           # [d_ff]
        down_col_norm = w_down.norm(dim=0)                       # [d_ff]
        act_score     = mean_abs_act * down_col_norm             # [d_ff]

        scores_per_layer.append(act_score)

    return scores_per_layer


# ===========================================================================
# -- Correlation utilities
# ===========================================================================

def _rank(v):
    """Return 0-based ranks for elements of 1-D tensor v."""
    n          = v.numel()
    sorted_idx = torch.argsort(v)
    ranks      = torch.zeros(n, dtype=torch.float32)
    ranks[sorted_idx] = torch.arange(n, dtype=torch.float32)
    return ranks


def pearson_corr(x, y):
    """Pearson correlation between two 1-D tensors."""
    xf = x.float() - x.float().mean()
    yf = y.float() - y.float().mean()
    denom = xf.norm() * yf.norm()
    if denom < 1e-10:
        return 0.0
    return float((xf @ yf) / denom)


def spearman_corr(x, y):
    """Spearman rank correlation between two 1-D tensors."""
    return pearson_corr(_rank(x.float()), _rank(y.float()))


def compute_score_correlations(layer, activation_scores=None, seed=42):
    """
    Compute pairwise Pearson and Spearman correlations between all scoring
    methods (and optionally activation_scores) for a single layer.

    Returns
    -------
    corr : dict  { method_a: { method_b: {'pearson': float, 'spearman': float} } }
    """
    methods   = ["down_norm", "product_norm", "rmsnorm_bound_angle"]
    score_map = {m: compute_scores(layer, m, seed=seed) for m in methods}

    if activation_scores is not None:
        score_map["activation"] = activation_scores.float().cpu()

    all_keys = list(score_map.keys())
    corr     = {}

    for ka in all_keys:
        corr[ka] = {}
        for kb in all_keys:
            if ka == kb:
                corr[ka][kb] = {"pearson": 1.0, "spearman": 1.0}
            else:
                corr[ka][kb] = {
                    "pearson":  pearson_corr(score_map[ka], score_map[kb]),
                    "spearman": spearman_corr(score_map[ka], score_map[kb]),
                }

    return corr
