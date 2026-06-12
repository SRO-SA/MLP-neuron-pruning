"""
selectors.py
============
Pruning scoring functions for structured MLP channel selection.

Each selector produces a 1-D float32 tensor of shape [d_ff] — one score per
MLP neuron (channel).  Lower score = less important = pruned first.

Selectors
---------
rmsnorm_bound  :  The proposed method.  score_i = R^2 * (||gate_i||*||up_i|| +
                  |gate_i·up_i|)/2 * ||down_i||, R = sqrt(d_model)*||gamma||_inf.
                  Implemented in bound_analysis.py; imported here as a wrapper.

down_norm      :  score_i = ||down_proj[:, i]||_2.  The column norm of the down
                  projection.  A simpler weight-norm baseline that ignores gate/up.

activation_score : score_i = mean_x(|SiLU(gate(x)_i) * up(x)_i|) * ||down_i||.
                  Requires calibration data.  The mean is over all calibration
                  tokens; multiplied by ||down_i|| to account for output magnitude.
                  If no calibration data is supplied, falls back to down_norm.

random_seedN   :  score_i = rand_i  with torch.Generator seeded to N.
                  N is extracted from the name "random_seed0", "random_seed1", etc.
                  Default seed is 0 if no suffix is found.
"""
from __future__ import annotations

import logging
from typing import Optional, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mlp_weights(layer):
    """Return (gate_w, up_w, down_w) as float32 CPU tensors."""
    from .model_utils import get_mlp_weights
    w = get_mlp_weights(layer)
    gate = w["gate_proj"].detach().float().cpu()  # [d_ff, d_model]
    up   = w["up_proj"].detach().float().cpu()    # [d_ff, d_model]
    down = w["down_proj"].detach().float().cpu()  # [d_model, d_ff]
    return gate, up, down


# ---------------------------------------------------------------------------
# down_norm
# ---------------------------------------------------------------------------

def compute_down_norm_scores(layer) -> torch.Tensor:
    """
    score_i = ||down_proj[:, i]||_2

    The column norm of the down-projection weight.  Simple, fast, no forward
    pass required.  Shape: [d_ff].
    """
    _, _, down = _get_mlp_weights(layer)
    # down: [d_model, d_ff]  →  norm over dim=0  →  [d_ff]
    return down.norm(dim=0)


# ---------------------------------------------------------------------------
# activation_score
# ---------------------------------------------------------------------------

def compute_activation_scores(
    layer,
    calib_inputs: Optional[torch.Tensor],
    device: str = "cpu",
) -> torch.Tensor:
    """
    score_i = mean_x( |SiLU(gate(x)_i) * up(x)_i| ) * ||down_proj[:, i]||_2

    Approximates the expected contribution magnitude of neuron i to the output.

    Parameters
    ----------
    layer        : transformer layer with .mlp.gate_proj / .mlp.up_proj / .mlp.down_proj
    calib_inputs : [N, d_model] float tensor of MLP input activations (calibration)
                   If None, falls back to down_norm.
    device       : device for computation

    Returns
    -------
    Tensor of shape [d_ff], float32, CPU.
    """
    if calib_inputs is None:
        logger.warning(
            "activation_score: no calibration data supplied; falling back to down_norm"
        )
        return compute_down_norm_scores(layer)

    from .model_utils import get_mlp_weights
    w    = get_mlp_weights(layer)
    gate = w["gate_proj"]   # [d_ff, d_model]
    up   = w["up_proj"]     # [d_ff, d_model]
    down = w["down_proj"]   # [d_model, d_ff]

    # Move weights and inputs to device for speed
    gate = gate.to(device=device, dtype=torch.float32)
    up   = up.to(device=device,   dtype=torch.float32)
    down = down.to(device=device, dtype=torch.float32)
    x    = calib_inputs.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # gate(x): [N, d_ff]
        gate_out = x @ gate.T       # [N, d_ff]
        up_out   = x @ up.T         # [N, d_ff]
        # SwiGLU activation: SiLU(gate) * up
        act      = F.silu(gate_out) * up_out   # [N, d_ff]
        # Mean absolute activation per neuron
        mean_act = act.abs().mean(dim=0)        # [d_ff]
        # Column norms of down_proj
        down_col_norms = down.norm(dim=0)       # [d_ff]

    scores = (mean_act * down_col_norms).detach().float().cpu()
    return scores


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------

def compute_random_scores(
    layer,
    seed: int = 0,
) -> torch.Tensor:
    """
    score_i = uniform random ∈ [0, 1)  with a fixed generator seed.

    Deterministic given the seed — the same seed always produces the same
    selection for a given layer size.

    Parameters
    ----------
    layer : transformer layer (only used to read d_ff)
    seed  : RNG seed (default 0)

    Returns
    -------
    Tensor of shape [d_ff], float32, CPU.
    """
    from .model_utils import get_mlp_weights
    w    = get_mlp_weights(layer)
    d_ff = w["d_ff"]
    gen  = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(d_ff, generator=gen)


def parse_random_seed(selector_name: str) -> int:
    """
    Parse seed from selector names like "random_seed0", "random_seed2".
    Falls back to seed=0 if no suffix is found.
    """
    prefix = "random_seed"
    if selector_name.startswith(prefix):
        try:
            return int(selector_name[len(prefix):])
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# rmsnorm_bound (wrapper to existing function)
# ---------------------------------------------------------------------------

def compute_rmsnorm_bound_scores(layer) -> torch.Tensor:
    """
    Wrapper around compute_bound_scores_and_R from bound_analysis.py.
    Returns scores of shape [d_ff], float32, CPU.
    """
    from .bound_analysis import compute_bound_scores_and_R
    with torch.no_grad():
        scores, _ = compute_bound_scores_and_R(layer)
    return scores.detach().float().cpu()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_scores_for_selector(
    selector_name: str,
    layer,
    calib_inputs: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Return a [d_ff] float32 score tensor for the given selector name.

    Selector names
    --------------
    "rmsnorm_bound"    → compute_rmsnorm_bound_scores
    "down_norm"        → compute_down_norm_scores
    "activation_score" → compute_activation_scores (needs calib_inputs)
    "random"           → compute_random_scores (seed=0)
    "random_seed0",
    "random_seed1",
    "random_seed2"     → compute_random_scores (seed=N)
    """
    name = selector_name.lower().strip()
    if name in ("rmsnorm_bound", "rmsnorm_bound_angle"):
        return compute_rmsnorm_bound_scores(layer)
    if name == "down_norm":
        return compute_down_norm_scores(layer)
    if name == "activation_score":
        return compute_activation_scores(layer, calib_inputs, device=device)
    if name.startswith("random"):
        seed = parse_random_seed(name)
        return compute_random_scores(layer, seed=seed)
    raise ValueError(
        f"Unknown selector '{selector_name}'. "
        f"Valid: rmsnorm_bound, down_norm, activation_score, "
        f"random, random_seed0, random_seed1, random_seed2"
    )


def gather_scores_for_selector(
    selector_name: str,
    layers: list,
    calib_inputs_per_layer: Optional[List[Optional[torch.Tensor]]] = None,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Gather per-layer scores for a given selector across all layers.

    Parameters
    ----------
    selector_name            : one of the valid selector names above
    layers                   : list of transformer layer modules
    calib_inputs_per_layer   : list of [N, d_model] tensors, one per layer
                               (required for activation_score; ignored otherwise)
    device                   : device for activation_score computation

    Returns
    -------
    List of [d_ff] float32 CPU tensors, one per layer.
    """
    if calib_inputs_per_layer is None:
        calib_inputs_per_layer = [None] * len(layers)

    scores = []
    with torch.no_grad():
        for li, (lyr, calib) in enumerate(zip(layers, calib_inputs_per_layer)):
            try:
                s = get_scores_for_selector(
                    selector_name, lyr, calib_inputs=calib, device=device
                )
                scores.append(s.detach().float().cpu())
            except Exception as exc:
                logger.error(
                    "gather_scores_for_selector: layer %d selector '%s' failed: %s",
                    li, selector_name, exc,
                )
                # Fall back to zeros (layer will be skipped by cap logic)
                from .model_utils import get_mlp_weights
                d_ff = get_mlp_weights(lyr)["d_ff"]
                scores.append(torch.zeros(d_ff))
    return scores
