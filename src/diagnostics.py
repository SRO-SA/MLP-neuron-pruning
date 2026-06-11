"""
diagnostics.py
==============
Diagnostic mode: run calibration prompts through the *unpruned* model and
log per-layer MLP output norms.

This is useful to understand how much each MLP block actually contributes to
the hidden-state residual stream.  Layers where the ratio

    ||MLP(r)|| / ||hidden_state||

is consistently small could in principle be candidates for whole-layer
skipping — but we do NOT skip them here; we only log the diagnostics.

Implementation note
--------------------
We use PyTorch forward hooks to capture:
  1. The MLP *input*  (= post-attention hidden state, post RMSNorm)
  2. The MLP *output* (= the MLP block output before residual add)

The residual-update ratio is:
    ||MLP_output|| / ||hidden_state_before_mlp||

where hidden_state_before_mlp is the residual-stream vector fed into the
post_attention_layernorm (i.e. before the layernorm, since we hook the mlp
module input which comes after the layernorm).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .model_utils import get_mlp_module, get_transformer_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration prompts
# ---------------------------------------------------------------------------
DEFAULT_CALIBRATION_PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "The speed of light in a vacuum is approximately",
    "Python is a programming language that",
    "The human brain contains approximately",
    "Transformers in deep learning were introduced by",
    "The Great Wall of China was built",
    "Water boils at 100 degrees Celsius at",
]


# ---------------------------------------------------------------------------
# Hook-based diagnostic runner
# ---------------------------------------------------------------------------

def run_diagnostics(
    model,
    tokenizer,
    prompts: Optional[List[str]] = None,
    max_seq_len: int = 128,
    device: Optional[str] = None,
) -> Dict:
    """
    Run calibration prompts through the model and collect per-layer
    MLP output statistics.

    Returns
    -------
    stats : dict  { layer_idx: { 'mlp_out_norm_mean', 'hidden_norm_mean',
                                  'ratio_mean', 'ratio_std' } }
    """
    if prompts is None:
        prompts = DEFAULT_CALIBRATION_PROMPTS

    if device is None:
        device = next(model.parameters()).device

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Per-layer accumulators  [list of per-sample means]
    mlp_out_norms:    List[List[float]] = [[] for _ in range(n_layers)]
    hidden_norms:     List[List[float]] = [[] for _ in range(n_layers)]
    ratios:           List[List[float]] = [[] for _ in range(n_layers)]

    # -----------------------------------------------------------------------
    # Register forward hooks
    # -----------------------------------------------------------------------
    # hook_data[i] = dict storing the captured tensors during the forward pass
    hook_data: List[Dict] = [{} for _ in range(n_layers)]
    hooks = []

    for i, layer in enumerate(layers):
        mlp = get_mlp_module(layer)

        def make_hooks(idx):
            def pre_hook(module, inp):
                # inp is a tuple; first element is the MLP input tensor
                hook_data[idx]["mlp_in"] = inp[0].detach()

            def post_hook(module, inp, out):
                hook_data[idx]["mlp_out"] = out.detach()

            return pre_hook, post_hook

        pre_h, post_h = make_hooks(i)
        hooks.append(mlp.register_forward_pre_hook(pre_h))
        hooks.append(mlp.register_forward_hook(post_h))

    # -----------------------------------------------------------------------
    # Run calibration prompts
    # -----------------------------------------------------------------------
    model.eval()
    print("\n" + "=" * 60)
    print("DIAGNOSTIC MODE — per-layer MLP contribution")
    print("=" * 60)

    try:
        for prompt in tqdm(prompts, desc="Calibration prompts"):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            with torch.no_grad():
                model(**enc)

            # Harvest hook data for this prompt
            for i in range(n_layers):
                mlp_in  = hook_data[i].get("mlp_in")
                mlp_out = hook_data[i].get("mlp_out")

                if mlp_in is None or mlp_out is None:
                    continue

                # Average over batch and sequence dimensions → scalar per sample
                mlp_out_norm = mlp_out.float().norm(dim=-1).mean().item()
                hidden_norm  = mlp_in.float().norm(dim=-1).mean().item()
                ratio        = (mlp_out_norm / hidden_norm) if hidden_norm > 1e-8 else 0.0

                mlp_out_norms[i].append(mlp_out_norm)
                hidden_norms[i].append(hidden_norm)
                ratios[i].append(ratio)

    finally:
        for h in hooks:
            h.remove()

    # -----------------------------------------------------------------------
    # Summarise
    # -----------------------------------------------------------------------
    stats = {}
    print(f"\n{'─'*60}")
    print(f"{'Layer':>5}  {'||MLP_out|| mean':>18}  {'||h|| mean':>12}  {'ratio mean':>12}  {'ratio std':>10}")
    print(f"{'─'*60}")

    for i in range(n_layers):
        if not ratios[i]:
            continue
        import statistics
        mo_mean  = statistics.mean(mlp_out_norms[i])
        h_mean   = statistics.mean(hidden_norms[i])
        r_mean   = statistics.mean(ratios[i])
        r_std    = statistics.stdev(ratios[i]) if len(ratios[i]) > 1 else 0.0

        stats[i] = {
            "mlp_out_norm_mean": mo_mean,
            "hidden_norm_mean":  h_mean,
            "ratio_mean":        r_mean,
            "ratio_std":         r_std,
        }
        print(f"{i:>5}  {mo_mean:>18.4f}  {h_mean:>12.4f}  {r_mean:>12.4f}  {r_std:>10.4f}")

    print(f"{'─'*60}\n")
    return stats
