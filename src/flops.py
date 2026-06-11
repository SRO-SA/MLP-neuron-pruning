"""
flops.py
========
Theoretical MLP FLOPs estimation for a SwiGLU transformer model.

FLOP counting convention
-------------------------
A single Linear layer  y = x W^T  of shape [seq, d_in] @ [d_in, d_out]
costs:
    FLOPs = 2 * seq * d_in * d_out
(one multiply + one add per element of the output matrix)

For one SwiGLU MLP block (seq tokens, input dim d_model, inner dim d_ff):
    gate_proj : 2 * seq * d_model * d_ff
    up_proj   : 2 * seq * d_model * d_ff
    SiLU + ⊙  : 2 * seq * d_ff          (element-wise, negligible)
    down_proj  : 2 * seq * d_ff   * d_model

Total per layer ≈ 2 * seq * d_ff * (2 * d_model + d_model) + tiny
                = 2 * seq * d_ff * (2 * d_model + d_model)   [simplified below]

We keep it simple and symmetric:
    mlp_flops_per_layer = 2 * seq * d_ff * d_model  (gate)
                        + 2 * seq * d_ff * d_model  (up)
                        + 2 * seq * d_ff * d_model  (down)
                        = 6 * seq * d_ff * d_model
"""

from __future__ import annotations

from typing import Dict, List

from .model_utils import get_mlp_weights, get_transformer_layers


def estimate_mlp_flops(
    model,
    seq_len: int = 512,
) -> Dict[str, int | float | List]:
    """
    Estimate total and per-layer theoretical MLP FLOPs for one forward pass
    over a sequence of *seq_len* tokens.

    Returns
    -------
    dict with keys:
        'total_flops'        : int   – summed over all layers
        'per_layer_flops'    : list  – one entry per layer
        'per_layer_d_ff'     : list  – current d_ff per layer (post-pruning)
        'seq_len'            : int
    """
    layers      = get_transformer_layers(model)
    per_layer_flops: List[int]  = []
    per_layer_d_ff:  List[int]  = []
    total_flops = 0

    for layer in layers:
        w       = get_mlp_weights(layer)
        d_model = w["d_model"]
        d_ff    = w["d_ff"]

        # 6 * seq * d_ff * d_model  (gate + up + down, each 2 * seq * d_ff * d_model)
        flops = 6 * seq_len * d_ff * d_model

        per_layer_flops.append(flops)
        per_layer_d_ff.append(d_ff)
        total_flops += flops

    return {
        "total_flops":     total_flops,
        "per_layer_flops": per_layer_flops,
        "per_layer_d_ff":  per_layer_d_ff,
        "seq_len":         seq_len,
    }
