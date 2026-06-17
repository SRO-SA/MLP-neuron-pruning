"""
moe_pruning.py
==============
Expert-wise structured SwiGLU channel pruning for MoE transformer models.

Supported architecture: Qwen3MoeModel (Qwen3-30B-A3B and compatible variants).
Will also work on any MoE model that follows the pattern:
    layer.mlp.gate          : Linear(hidden_size, num_experts)   [router]
    layer.mlp.experts       : ModuleList of SwiGLU MLPs
    layer.mlp.shared_expert : SwiGLU MLP or None
    each expert             : .gate_proj, .up_proj, .down_proj (Linear)

Key design decisions
--------------------
1. DO NOT prune router weights (gate linear layer).
2. DO NOT remove entire experts.
3. Per-expert pruning: remove up to `max_expert_frac` (default 20%) of that
   expert's MLP channels.
4. Router-aware calibration: route each calibration token to its assigned
   expert(s) to collect per-expert MLP input activations.
5. Per-expert residual reconstruction: solve the ridge regression using only
   the tokens that were routed to that expert.
6. Skip experts with fewer than `min_expert_tokens` (default 128) routed tokens
   — mark them as skipped in the report.
7. Score function: rmsnorm_bound_angle (same as dense pruning), ignoring the
   RMSNorm weight (no RMSNorm precedes individual expert layers typically).
   Falls back to down_norm if bound scores fail.

Usage
-----
    python run_experiment.py \\
        --config configs/moe_qwen3_30b.yaml \\
        --moe-target-pruning

Config keys (in YAML)
---------------------
    scaling_models: ["Qwen/Qwen3-30B-A3B"]
    target_pruning_percents: [2.0]
    scaling_methods: ["pure_delete"]
    max_expert_frac: 0.20
    min_expert_tokens: 128
    reconstruction_eval_samples: 64
    eval_datasets: ["wikitext2"]
    moe_smoke_test: true   # if true: run only first 4 layers to verify correctness
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_EXPERT_FRAC    = 0.20
DEFAULT_MIN_EXPERT_TOKENS  = 128
BEST_RESIDUAL_LAM          = 1e-2
BEST_RESIDUAL_TAU          = 1.0

MOE_MAIN_CSV_KEYS = [
    "model", "target_pruning_percent", "eval_dataset",
    "layer_index", "expert_index",
    "selector", "method",
    "d_ff_before", "d_ff_after",
    "n_pruned", "pruning_percent",
    "n_routed_tokens",
    "skipped",
    "baseline_ppl", "compressed_ppl", "delta_ppl",
    "relative_ppl_increase_percent",
    "reconstruction_time_seconds", "peak_gpu_memory_MB",
    "dtype", "notes",
]

MOE_SUMMARY_CSV_KEYS = [
    "model", "target_pruning_percent", "eval_dataset",
    "selector", "method",
    "total_experts", "experts_pruned", "experts_skipped",
    "total_mlp_neurons_before", "total_mlp_neurons_pruned",
    "actual_pruning_percent",
    "baseline_ppl", "compressed_ppl", "delta_ppl",
    "relative_ppl_increase_percent",
    "damage_reduction_percent",
    "reconstruction_time_seconds", "peak_gpu_memory_MB",
    "dtype", "notes",
]


# ---------------------------------------------------------------------------
# Architecture discovery
# ---------------------------------------------------------------------------

class MoELayerInfo:
    """Metadata about a single MoE layer."""
    __slots__ = [
        "layer_idx", "layer_module",
        "is_moe", "is_dense",
        "router_module",
        "expert_modules",       # List[expert_module | PackedExpertView]
        "shared_expert_module",
        "num_experts",
        "top_k",
        "experts_packed",       # True when mlp.experts is a packed Qwen3MoeExperts tensor
        "experts_container",    # the Qwen3MoeExperts object when packed; else None
    ]

    def __init__(self, layer_idx: int, layer_module):
        self.layer_idx            = layer_idx
        self.layer_module         = layer_module
        self.is_moe               = False
        self.is_dense             = False
        self.router_module        = None
        self.expert_modules       = []
        self.shared_expert_module = None
        self.num_experts          = 0
        self.top_k                = 2  # default
        self.experts_packed       = False
        self.experts_container    = None


# ---------------------------------------------------------------------------
# Packed-expert support (Qwen3MoeExperts layout)
# ---------------------------------------------------------------------------

class PackedExpertView:
    """
    Virtual per-expert view into a packed Qwen3MoeExperts tensor block.

    Qwen3MoeExperts stores all experts in two fused parameters:
        gate_up_proj : [num_experts, 2 * moe_intermediate, hidden_dim]
                       (gate projection comes first, then up projection)
        down_proj    : [num_experts, hidden_dim, moe_intermediate]

    This class slices those tensors for a single expert so that
    get_expert_weights / get_expert_scores work identically to the
    unpacked (ModuleList) case.

    For expert i:
        gate slice: gate_up_proj[i, :moe_inter, :]   shape [moe_inter, hidden]
        up   slice: gate_up_proj[i, moe_inter:, :]   shape [moe_inter, hidden]
        down slice: down_proj[i]                      shape [hidden, moe_inter]

    These are views (not copies) — they share memory with the container.
    """

    def __init__(self, container, expert_idx: int, moe_intermediate: int):
        self.container        = container
        self.idx              = expert_idx
        self.moe_intermediate = moe_intermediate
        self.hidden_dim       = container.gate_up_proj.shape[2]

    # ── virtual weight properties ─────────────────────────────────────────
    @property
    def gate_weight(self) -> torch.Tensor:
        """[moe_intermediate, hidden_dim] view of gate projection."""
        return self.container.gate_up_proj[self.idx, :self.moe_intermediate, :]

    @property
    def up_weight(self) -> torch.Tensor:
        """[moe_intermediate, hidden_dim] view of up projection."""
        return self.container.gate_up_proj[self.idx, self.moe_intermediate:, :]

    @property
    def down_weight(self) -> torch.Tensor:
        """[hidden_dim, moe_intermediate] view of down projection."""
        return self.container.down_proj[self.idx]

    def __repr__(self):
        return (f"PackedExpertView(idx={self.idx}, "
                f"d_ff={self.moe_intermediate}, d_model={self.hidden_dim})")


def inspect_experts_container(experts) -> None:
    """
    Diagnostic helper: print type, parameters, buffers, and child modules
    of an experts container.  Call this when expert layout is unknown.
    """
    print(f"\n  Experts type   : {type(experts).__name__}")
    print(f"  Experts params :")
    for name, param in experts.named_parameters(recurse=False):
        print(f"    {name:30s}  shape={list(param.shape)}  dtype={param.dtype}")
    for name, buf in experts.named_buffers(recurse=False):
        print(f"    [buf] {name:26s}  shape={list(buf.shape)}")
    print(f"  Experts children:")
    for cname, cmod in experts.named_children():
        print(f"    {cname:30s}  {type(cmod).__name__}")
    for attr in ("num_experts", "intermediate_dim", "hidden_dim",
                 "is_concatenated", "has_gate"):
        if hasattr(experts, attr):
            print(f"  .{attr:28s} = {getattr(experts, attr)}")
    print()


def _detect_experts_layout(experts) -> str:
    """
    Detect whether experts is Layout A (iterable ModuleList) or
    Layout B (packed Qwen3MoeExperts with gate_up_proj + down_proj).

    Returns
    -------
    "unpacked"          — Layout A: experts is iterable, each item has gate_proj / up_proj / down_proj
    "packed_gate_up"    — Layout B: experts has .gate_up_proj [n,2i,h] + .down_proj [n,h,i]
    "unknown"           — neither layout recognised
    """
    # Layout B check first (more specific)
    if (hasattr(experts, "gate_up_proj") and
            hasattr(experts, "down_proj") and
            isinstance(getattr(experts, "gate_up_proj", None), torch.Tensor)):
        gu = experts.gate_up_proj
        if gu.ndim == 3:
            return "packed_gate_up"

    # Layout A: try iterating
    try:
        items = list(experts)
        if items and hasattr(items[0], "gate_proj"):
            return "unpacked"
    except (TypeError, RuntimeError):
        pass

    return "unknown"


def discover_moe_architecture(model) -> Tuple[List[MoELayerInfo], dict]:
    """
    Auto-discover MoE layer structure in the model.

    Handles:
    - Qwen3MoeModel  (Qwen3-30B-A3B)
    - Any model where layer.mlp has `.experts` (ModuleList) and `.gate`
    - Dense layers in mixed MoE/dense models (some Qwen3 variants)

    Returns
    -------
    layer_infos : List[MoELayerInfo]
    arch_info   : dict with summary statistics
    """
    from .model_utils import get_transformer_layers
    layers     = get_transformer_layers(model)
    layer_infos: List[MoELayerInfo] = []
    n_moe = 0
    n_dense = 0
    total_experts = 0

    for li, layer in enumerate(layers):
        info = MoELayerInfo(li, layer)
        mlp = getattr(layer, "mlp", None)

        if mlp is None:
            logger.warning("Layer %d has no .mlp attribute — treating as dense", li)
            info.is_dense = True
            layer_infos.append(info)
            continue

        # Check for MoE indicators
        experts = getattr(mlp, "experts", None)
        router  = (getattr(mlp, "gate", None) or
                   getattr(mlp, "router", None) or
                   getattr(mlp, "expert_router", None))

        if experts is not None and router is not None:
            info.is_moe            = True
            info.router_module     = router
            info.shared_expert_module = getattr(mlp, "shared_expert", None)
            info.experts_container = experts

            layout = _detect_experts_layout(experts)

            if layout == "unpacked":
                # Layout A: ModuleList of independent expert modules
                info.experts_packed  = False
                info.expert_modules  = list(experts)
                info.num_experts     = len(info.expert_modules)

            elif layout == "packed_gate_up":
                # Layout B: Qwen3MoeExperts — packed [n_exp, 2*inter, hidden] tensors
                gu = experts.gate_up_proj          # [n_exp, 2*inter, hidden]
                n_exp   = gu.shape[0]
                inter   = gu.shape[1] // 2        # moe_intermediate
                info.experts_packed = True
                info.num_experts    = n_exp
                # Build virtual per-expert views
                info.expert_modules = [
                    PackedExpertView(experts, ei, inter)
                    for ei in range(n_exp)
                ]
                logger.info(
                    "Layer %d: packed Qwen3MoeExperts detected — "
                    "n_exp=%d, moe_inter=%d, hidden=%d",
                    li, n_exp, inter, gu.shape[2],
                )

            else:
                logger.warning(
                    "Layer %d: unknown experts layout %s — printing diagnostics",
                    li, type(experts).__name__,
                )
                inspect_experts_container(experts)
                raise RuntimeError(
                    f"Unsupported experts layout in layer {li}: "
                    f"{type(experts).__name__}. "
                    "See diagnostic output above for parameter names/shapes."
                )

            cfg = getattr(model, "config", None)
            info.top_k = getattr(cfg, "num_experts_per_tok",
                         getattr(cfg, "top_k", 2))
            n_moe         += 1
            total_experts += info.num_experts
        else:
            info.is_dense = True
            n_dense += 1

        layer_infos.append(info)

    arch_info = {
        "n_moe_layers":    n_moe,
        "n_dense_layers":  n_dense,
        "total_layers":    len(layers),
        "total_experts":   total_experts,
        "model_class":     type(model).__name__,
    }
    logger.info(
        "MoE architecture: %d MoE layers (%d total experts), %d dense layers",
        n_moe, total_experts, n_dense,
    )
    return layer_infos, arch_info


# ---------------------------------------------------------------------------
# Expert weight access
# ---------------------------------------------------------------------------

def get_expert_weights(expert_module) -> dict:
    """
    Extract gate_proj, up_proj, down_proj from an expert module.

    Handles both:
    - Layout A (unpacked): expert_module.gate_proj / up_proj / down_proj are nn.Linear
    - Layout B (packed): expert_module is a PackedExpertView with tensor properties

    Returns dict with 'd_model', 'd_ff', 'gate_proj', 'up_proj', 'down_proj'.
    Shapes follow the dense-MLP convention:
        gate_proj : [d_ff, d_model]
        up_proj   : [d_ff, d_model]
        down_proj : [d_model, d_ff]
    """
    if isinstance(expert_module, PackedExpertView):
        pv = expert_module
        gate = pv.gate_weight   # [moe_inter, hidden] = [d_ff, d_model]
        up   = pv.up_weight     # [moe_inter, hidden] = [d_ff, d_model]
        down = pv.down_weight   # [hidden, moe_inter] = [d_model, d_ff]
        d_ff, d_model = gate.shape
        return {
            "d_model":   d_model,
            "d_ff":      d_ff,
            "gate_proj": gate,
            "up_proj":   up,
            "down_proj": down,
        }

    # Layout A: independent nn.Linear expert modules
    gate = getattr(expert_module, "gate_proj", None)
    up   = getattr(expert_module, "up_proj",   None)
    down = getattr(expert_module, "down_proj", None)
    if gate is None or up is None or down is None:
        raise AttributeError(
            f"Expert module {type(expert_module).__name__} missing "
            "gate_proj / up_proj / down_proj"
        )
    d_ff, d_model = gate.weight.shape
    return {
        "d_model":    d_model,
        "d_ff":       d_ff,
        "gate_proj":  gate.weight,   # [d_ff, d_model]
        "up_proj":    up.weight,     # [d_ff, d_model]
        "down_proj":  down.weight,   # [d_model, d_ff]
    }


def get_expert_scores(expert_module) -> torch.Tensor:
    """
    Compute RMSNorm-bound-angle scores for an expert.
    Falls back to down_norm if the bound score computation fails.
    Returns [d_ff] float32 CPU tensor.
    """
    try:
        w = get_expert_weights(expert_module)
        gate = w["gate_proj"].detach().float().cpu()   # [d_ff, d_model]
        up   = w["up_proj"].detach().float().cpu()     # [d_ff, d_model]
        down = w["down_proj"].detach().float().cpu()   # [d_model, d_ff]

        # RMSNorm-bound-angle score (without gamma since no pre-expert norm)
        # score_i = (||gate_i|| * ||up_i|| + |gate_i · up_i|) / 2 * ||down_i||
        gate_norms = gate.norm(dim=1)         # [d_ff]
        up_norms   = up.norm(dim=1)           # [d_ff]
        dot_prods  = (gate * up).sum(dim=1).abs()  # [d_ff]
        down_norms = down.norm(dim=0)         # [d_ff]  (column norms)

        scores = ((gate_norms * up_norms + dot_prods) / 2.0) * down_norms
        return scores
    except Exception as exc:
        logger.warning("Expert score fallback to down_norm: %s", exc)
        w = get_expert_weights(expert_module)
        return w["down_proj"].detach().float().cpu().norm(dim=0)


# ---------------------------------------------------------------------------
# Router-aware calibration
# ---------------------------------------------------------------------------

def collect_expert_activations(
    model,
    tokenizer,
    layer_infos:    List[MoELayerInfo],
    prompts:        List[str],
    device:         str,
    max_seq_len:    int = 512,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Route calibration tokens through the model and collect MLP input activations
    per expert.

    Handles two expert layouts:

    Layout A (unpacked ModuleList):
        Registers a forward hook on each expert's gate_proj module.
        The hook fires only for tokens routed to that expert — exactly the
        activations needed for per-expert residual reconstruction.

    Layout B (packed Qwen3MoeExperts):
        Registers a pre-hook on the MoE block to capture hidden_states, and a
        forward hook on the router (gate) to capture top-k routing decisions.
        After the forward pass, reconstructs per-expert slices:
            expert_inputs[layer_idx, expert_idx] = hidden_states[routed_mask]

    Returns
    -------
    Dict mapping (layer_idx, expert_idx) → Tensor of shape [n_routed, d_model]
    Only MoE layers are populated; dense layers are skipped.
    """
    expert_inputs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    hooks = []

    for info in layer_infos:
        if not info.is_moe:
            continue

        if not info.experts_packed:
            # ── Layout A: hook each expert's gate_proj ───────────────────────
            for ei, expert in enumerate(info.expert_modules):
                gate_proj = getattr(expert, "gate_proj", None)
                if gate_proj is None:
                    continue
                key = (info.layer_idx, ei)
                expert_inputs[key] = []

                def _make_hook(k):
                    def hook_fn(module, inp, out):
                        # inp[0]: [n_tokens, d_model] (already routed)
                        expert_inputs[k].append(inp[0].detach().float().cpu())
                    return hook_fn

                h = gate_proj.register_forward_hook(_make_hook(key))
                hooks.append(h)

        else:
            # ── Layout B: hook MoE block input + router output ───────────────
            n_exp = info.num_experts
            top_k = info.top_k
            for ei in range(n_exp):
                key = (info.layer_idx, ei)
                expert_inputs[key] = []

            # Storage for this layer's per-prompt data
            layer_hidden: List[torch.Tensor] = []     # [n_tok, d_model] per prompt
            layer_routing: List[torch.Tensor] = []    # [n_tok, top_k] per prompt

            mlp_module    = info.layer_module.mlp
            router_module = info.router_module

            def _make_pre_hook(lh):
                def pre_hook(module, args):
                    # args[0]: [batch, seq, d_model] or [n_tok, d_model]
                    h = args[0].detach().float().cpu()
                    if h.dim() == 3:
                        h = h.reshape(-1, h.shape[-1])
                    lh.append(h)
                return pre_hook

            def _make_router_hook(lr):
                def router_hook(module, inp, out):
                    # out may be (routing_weights, selected_experts) or just logits
                    if isinstance(out, (tuple, list)):
                        # Typically (routing_weights, selected_experts)
                        # selected_experts: [n_tok, top_k] int64
                        if len(out) >= 2:
                            sel = out[1]
                            if isinstance(sel, torch.Tensor) and sel.dtype in (
                                    torch.int32, torch.int64, torch.long):
                                lr.append(sel.detach().cpu())
                                return
                        # Fallback: logits → top_k
                        logits = out[0] if isinstance(out[0], torch.Tensor) else out
                        topk = torch.topk(logits.float(), k=min(top_k, logits.shape[-1]),
                                          dim=-1)
                        lr.append(topk.indices.detach().cpu())
                    elif isinstance(out, torch.Tensor):
                        if out.dtype in (torch.int32, torch.int64, torch.long):
                            lr.append(out.detach().cpu())
                        else:
                            topk = torch.topk(out.float(),
                                              k=min(top_k, out.shape[-1]), dim=-1)
                            lr.append(topk.indices.detach().cpu())
                return router_hook

            h1 = mlp_module.register_forward_pre_hook(_make_pre_hook(layer_hidden))
            h2 = router_module.register_forward_hook(_make_router_hook(layer_routing))
            hooks.extend([h1, h2])

            # Store references so we can reconstruct after calibration
            info._calib_hidden  = layer_hidden
            info._calib_routing = layer_routing

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=max_seq_len,
            )
            input_ids = enc["input_ids"].to(device)
            try:
                _ = model(input_ids=input_ids)
            except Exception as exc:
                logger.warning("calibration forward pass failed: %s", exc)

    for h in hooks:
        h.remove()

    # ── Reconstruct per-expert activations for packed layers ─────────────────
    for info in layer_infos:
        if not info.is_moe or not info.experts_packed:
            continue
        hidden_list  = getattr(info, "_calib_hidden",  [])
        routing_list = getattr(info, "_calib_routing", [])
        if not hidden_list or not routing_list:
            logger.warning("Layer %d: no calibration data captured", info.layer_idx)
            continue

        # Align lengths (router may fire multiple times per prompt due to block structure)
        # Concatenate all and match by token count
        all_hidden = torch.cat(hidden_list, dim=0)   # [total_tokens, d_model]

        # routing tensors may be [n_tok, top_k] or [n_tok]
        try:
            all_routing = torch.cat(routing_list, dim=0)  # [total_tokens, top_k] or similar
        except RuntimeError:
            # Shape mismatch — skip
            logger.warning("Layer %d: routing tensor shapes mismatch, skipping", info.layer_idx)
            continue

        if all_routing.dim() == 1:
            all_routing = all_routing.unsqueeze(-1)  # [n_tok, 1]

        n_tok = min(all_hidden.shape[0], all_routing.shape[0])
        all_hidden   = all_hidden[:n_tok]
        all_routing  = all_routing[:n_tok]

        for ei in range(info.num_experts):
            # mask: token is routed to expert ei if any column of routing == ei
            mask = (all_routing == ei).any(dim=-1)  # [n_tok] bool
            if mask.any():
                key = (info.layer_idx, ei)
                expert_inputs[key].append(all_hidden[mask])

        # Clean up temporary storage
        info._calib_hidden  = []
        info._calib_routing = []

    # Concatenate results
    result: Dict[Tuple[int, int], torch.Tensor] = {}
    for key, tensors in expert_inputs.items():
        if tensors:
            result[key] = torch.cat(tensors, dim=0)

    return result


# ---------------------------------------------------------------------------
# Expert pruning (physical weight removal)
# ---------------------------------------------------------------------------

def prune_expert_channels(
    expert_module,
    prune_indices: torch.Tensor,
) -> None:
    """
    Physically remove MLP channels from an expert module IN-PLACE.

    gate_proj.weight : [d_ff, d_model]  → remove rows prune_indices
    up_proj.weight   : [d_ff, d_model]  → remove rows prune_indices
    down_proj.weight : [d_model, d_ff]  → remove columns prune_indices

    Biases are handled if present.
    """
    keep_mask = torch.ones(
        expert_module.gate_proj.weight.shape[0], dtype=torch.bool
    )
    keep_mask[prune_indices] = False
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    # gate_proj
    old_g = expert_module.gate_proj.weight.data
    expert_module.gate_proj.weight = torch.nn.Parameter(old_g[keep_indices, :])
    if expert_module.gate_proj.bias is not None:
        expert_module.gate_proj.bias = torch.nn.Parameter(
            expert_module.gate_proj.bias.data[keep_indices]
        )

    # up_proj
    old_u = expert_module.up_proj.weight.data
    expert_module.up_proj.weight = torch.nn.Parameter(old_u[keep_indices, :])
    if expert_module.up_proj.bias is not None:
        expert_module.up_proj.bias = torch.nn.Parameter(
            expert_module.up_proj.bias.data[keep_indices]
        )

    # down_proj (column removal)
    old_d = expert_module.down_proj.weight.data
    expert_module.down_proj.weight = torch.nn.Parameter(old_d[:, keep_indices])
    if expert_module.down_proj.bias is not None:
        pass  # down_proj bias is [d_model], independent of d_ff — no change



def prune_packed_experts_global(
    experts_container,
    prune_indices: torch.Tensor,
) -> int:
    """
    Globally prune the SAME channels from ALL experts in a packed tensor block.

    Because gate_up_proj and down_proj are fused across experts, per-expert
    variable-width pruning is not possible without unpacking.  This function
    instead removes a shared set of channel indices from every expert.

    Parameters
    ----------
    experts_container : Qwen3MoeExperts (has .gate_up_proj and .down_proj params)
    prune_indices     : 1-D int64 tensor of channel indices to prune,
                        in [0, moe_intermediate)

    Modifies
    --------
    experts_container.gate_up_proj : [n_exp, 2*moe_inter, hidden]
                                   → [n_exp, 2*new_inter, hidden]
    experts_container.down_proj    : [n_exp, hidden, moe_inter]
                                   → [n_exp, hidden, new_inter]
    experts_container.intermediate_dim (if present) updated to new_inter.

    Returns
    -------
    new_intermediate : int  (number of channels remaining)
    """
    gu = experts_container.gate_up_proj.data   # [n_exp, 2*moe_inter, hidden]
    dp = experts_container.down_proj.data       # [n_exp, hidden, moe_inter]

    moe_inter = gu.shape[1] // 2
    n_exp     = gu.shape[0]

    keep_mask = torch.ones(moe_inter, dtype=torch.bool)
    keep_mask[prune_indices] = False
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]  # [new_inter]
    new_inter = len(keep_idx)

    # gate_up rows: gate occupies [:moe_inter], up occupies [moe_inter:]
    gate_keep    = keep_idx                     # rows to keep in gate portion
    up_keep      = keep_idx + moe_inter         # rows to keep in up portion
    gate_up_keep = torch.cat([gate_keep, up_keep])  # [2*new_inter]

    new_gu = gu[:, gate_up_keep, :]   # [n_exp, 2*new_inter, hidden]
    new_dp = dp[:, :, keep_idx]       # [n_exp, hidden, new_inter]

    experts_container.gate_up_proj = torch.nn.Parameter(new_gu)
    experts_container.down_proj    = torch.nn.Parameter(new_dp)

    # Update metadata attribute if it exists
    if hasattr(experts_container, "intermediate_dim"):
        experts_container.intermediate_dim = new_inter

    logger.info(
        "prune_packed_experts_global: %d experts, moe_inter %d → %d "
        "(pruned %d channels)",
        n_exp, moe_inter, new_inter, moe_inter - new_inter,
    )
    return new_inter


# ---------------------------------------------------------------------------
# Expert residual reconstruction
# ---------------------------------------------------------------------------

def apply_expert_residual_reconstruction(
    expert_module,
    prune_indices:  torch.Tensor,
    keep_indices:   torch.Tensor,
    calib_inputs:   torch.Tensor,
    ridge_lambda:   float = 1e-2,
    tau:            float = 1.0,
) -> dict:
    """
    Residual down-projection reconstruction for a single expert.

    Computes the lost signal E = A_P @ W_P.T and solves a ridge regression
    over kept activations to update W_down.

    Parameters
    ----------
    expert_module : expert module with gate_proj / up_proj / down_proj
    prune_indices : indices to prune (original d_ff indexing)
    keep_indices  : indices to keep (original d_ff indexing)
    calib_inputs  : [N, d_model] float32 CPU tensor of routed activations
    ridge_lambda  : ridge regularization coefficient
    tau           : update scale (1.0 = full update)

    Returns
    -------
    dict with: n_pruned, n_kept, n_tokens, coverage_pct, status
    """
    import torch.nn.functional as F

    # Move weights to CPU float32 for reconstruction
    W_gate = expert_module.gate_proj.weight.data.detach().float().cpu()  # [d_ff, d_model]
    W_up   = expert_module.up_proj.weight.data.detach().float().cpu()    # [d_ff, d_model]
    W_down = expert_module.down_proj.weight.data.detach().float().cpu()  # [d_model, d_ff]

    X = calib_inputs.float()  # [N, d_model]
    N = X.shape[0]

    if N == 0:
        return {"n_pruned": len(prune_indices), "n_kept": len(keep_indices),
                "n_tokens": 0, "status": "skipped_no_tokens"}

    with torch.no_grad():
        # Compute SwiGLU activations for ALL neurons
        gate_out = X @ W_gate.T                   # [N, d_ff]
        up_out   = X @ W_up.T                     # [N, d_ff]
        act_all  = F.silu(gate_out) * up_out      # [N, d_ff]

        # Activations for pruned and kept neurons
        A_P = act_all[:, prune_indices]           # [N, n_pruned]
        A_K = act_all[:, keep_indices]            # [N, n_kept]
        W_P = W_down[:, prune_indices]            # [d_model, n_pruned]
        W_K = W_down[:, keep_indices]             # [d_model, n_kept]

        # Lost signal
        E = A_P @ W_P.T                           # [N, d_model]

        # Ridge solve in dual form (N×N) when N < n_kept
        n_kept = A_K.shape[1]
        AAt = A_K @ A_K.T                         # [N, N]
        lam_scaled = ridge_lambda * AAt.diagonal().mean().item()
        reg = lam_scaled * torch.eye(N)
        try:
            # Solve (AAt + reg) @ B = E  →  B: [N, d_model]
            B = torch.linalg.solve(AAt + reg, E)   # [N, d_model]
            Delta = A_K.T @ B                       # [n_kept, d_model]
            # New W_K columns
            W_new_K = W_K + tau * Delta.T           # [d_model, n_kept]
            # Update only kept columns of down_proj in-place
            expert_module.down_proj.weight.data[:, keep_indices] = (
                W_new_K.to(expert_module.down_proj.weight.data.device,
                           dtype=expert_module.down_proj.weight.data.dtype)
            )
            status = "ok"
        except Exception as exc:
            logger.warning("Expert ridge solve failed: %s", exc)
            status = f"ridge_failed: {exc}"

    total_ff = len(prune_indices) + len(keep_indices)
    coverage_pct = 100.0 * len(keep_indices) / total_ff if total_ff > 0 else 0.0
    return {
        "n_pruned":      len(prune_indices),
        "n_kept":        len(keep_indices),
        "n_tokens":      N,
        "coverage_pct":  round(coverage_pct, 2),
        "status":        status,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_moe_target_pruning_mode(
    cfg:                    dict,
    device:                 str,
    output_dir:             str          = "results",
    models_override:        Optional[List[str]]   = None,
    targets_override:       Optional[List[float]] = None,
    methods_override:       Optional[List[str]]   = None,
    n_eval_override:        Optional[int]         = None,
    eval_datasets_override: Optional[List[str]]   = None,
) -> None:
    """
    Expert-wise structured MLP channel pruning for MoE models.

    Protocol
    --------
    For each model × target_percent × method:
      1. Discover MoE architecture (layers, experts, router).
      2. Calibrate: route prompts, collect per-expert MLP input activations.
      3. Score: compute per-channel importance for each expert.
      4. Select: globally select target_n channels (across all MoE experts),
         subject to max_expert_frac per expert.
      5. Prune: physically remove channels from expert gate/up/down weights.
      6. Optionally reconstruct: solve residual correction for down_proj.
      7. Evaluate: perplexity on eval datasets.
    """
    from .evaluation import evaluate_perplexity, load_all_eval_datasets
    from .flops import estimate_mlp_flops
    from .model_utils import (
        count_parameters,
        get_transformer_layers,
        load_model_and_tokenizer,
    )
    from .pruning import verify_forward_pass
    from .merging import RECONSTRUCTION_TRAIN_PROMPTS

    def _auto_dtype(dev: str) -> str:
        if dev != "cpu" and torch.cuda.is_available():
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"

    os.makedirs(output_dir, exist_ok=True)
    ts            = time.strftime("%Y%m%d_%H%M%S")
    main_csv_path = os.path.join(output_dir, f"moe_target_pruning_{ts}.csv")
    json_path     = os.path.join(output_dir, f"moe_target_pruning_{ts}.json")

    # ── Config ────────────────────────────────────────────────────────────────
    model_list     = (models_override
                      or cfg.get("scaling_models", ["Qwen/Qwen3-30B-A3B"]))
    TARGET_PCTS    = [float(t) for t in (
        targets_override or cfg.get("target_pruning_percents", [2.0]))]
    METHODS        = (methods_override
                      or cfg.get("scaling_methods", ["pure_delete"]))
    n_eval         = int(n_eval_override
                         or cfg.get("reconstruction_eval_samples", 64))
    max_seq        = int(cfg.get("max_seq_len", 512))
    batch_sz       = int(cfg.get("batch_size", 4))
    use_fb         = bool(cfg.get("use_fallback_corpus", False))
    dtype_cfg      = str(cfg.get("scaling_dtype", "auto"))
    max_exp_frac   = float(cfg.get("max_expert_frac", DEFAULT_MAX_EXPERT_FRAC))
    min_exp_tokens = int(cfg.get("min_expert_tokens", DEFAULT_MIN_EXPERT_TOKENS))
    smoke_test     = bool(cfg.get("moe_smoke_test", False))
    EVAL_DATASETS  = [str(d) for d in (
        eval_datasets_override or cfg.get("eval_datasets", ["wikitext2"]))]

    print(f"\n{'=' * 90}")
    print("MOE TARGET-PRUNING EXPERIMENT")
    print(f"  Models         : {model_list}")
    print(f"  Target percents: {TARGET_PCTS}%")
    print(f"  Methods        : {METHODS}")
    print(f"  Eval datasets  : {EVAL_DATASETS}")
    print(f"  n_eval         : {n_eval}")
    print(f"  max_expert_frac: {max_exp_frac:.0%}")
    print(f"  min_exp_tokens : {min_exp_tokens}")
    if smoke_test:
        print("  SMOKE TEST MODE: only first 4 MoE layers will be processed")
    print(f"{'=' * 90}\n")

    # Load eval datasets once
    print(f"Loading evaluation datasets: {EVAL_DATASETS} ...")
    all_eval_corpora = load_all_eval_datasets(
        EVAL_DATASETS, max_samples=n_eval, use_fallback_corpus=use_fb,
    )
    for _dn, _txts in all_eval_corpora.items():
        print(f"  {_dn}: {len(_txts)} samples")

    all_results: List[Dict] = []

    def _flush_csv(path, rows, keys):
        if not rows:
            return
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)

    for model_name in model_list:
        print(f"\n{'#' * 90}")
        print(f"MODEL: {model_name}")
        print(f"{'#' * 90}")
        model     = None
        tokenizer = None
        try:
            dtype_str = _auto_dtype(device) if dtype_cfg == "auto" else dtype_cfg
            model, tokenizer, _ = load_model_and_tokenizer(
                model_name=model_name, fallback_name=None,
                device=device, dtype_str=dtype_str,
            )
            model.eval()

            # Discover architecture
            layer_infos, arch_info = discover_moe_architecture(model)

            # ── Architecture sanity log ──────────────────────────────────────
            first_moe = next((i for i in layer_infos if i.is_moe), None)
            print(f"\n  {'─' * 60}")
            print(f"  ARCHITECTURE DISCOVERY")
            print(f"  {'─' * 60}")
            print(f"  Model class         : {arch_info['model_class']}")
            print(f"  Total layers        : {arch_info['total_layers']}")
            print(f"  MoE layers          : {arch_info['n_moe_layers']}")
            print(f"  Dense layers        : {arch_info['n_dense_layers']}")
            print(f"  Total experts       : {arch_info['total_experts']}")
            if first_moe is not None:
                print(f"  First MoE layer idx : {first_moe.layer_idx}")
                # Router info
                rtr = first_moe.router_module
                if rtr is not None:
                    r_shape = (list(rtr.weight.shape)
                               if hasattr(rtr, "weight") else "no .weight")
                    print(f"  Router type         : {type(rtr).__name__}  shape={r_shape}")
                else:
                    print(f"  Router              : not detected")
                # First expert shapes
                if first_moe.experts_packed and first_moe.experts_container is not None:
                    ec = first_moe.experts_container
                    print(f"  Expert layout       : PACKED (Qwen3MoeExperts)")
                    for pname in ("gate_up_proj", "down_proj"):
                        p = getattr(ec, pname, None)
                        if isinstance(p, torch.Tensor):
                            print(f"  experts.{pname:18s}: {list(p.shape)}")
                    inter = first_moe.expert_modules[0].moe_intermediate if first_moe.expert_modules else "?"
                    print(f"  moe_intermediate    : {inter}")
                    print(f"  NOTE: physical pruning will use global same-channel mode")
                elif first_moe.expert_modules:
                    e0 = first_moe.expert_modules[0]
                    if isinstance(e0, PackedExpertView):
                        print(f"  Expert layout       : PACKED (virtual views)")
                        print(f"  expert[0].gate      : {list(e0.gate_weight.shape)}")
                        print(f"  expert[0].up        : {list(e0.up_weight.shape)}")
                        print(f"  expert[0].down      : {list(e0.down_weight.shape)}")
                    else:
                        print(f"  Expert layout       : UNPACKED (ModuleList)")
                        for pname in ("gate_proj", "up_proj", "down_proj"):
                            pm = getattr(e0, pname, None)
                            if pm is not None and hasattr(pm, "weight"):
                                print(f"  expert[0].{pname:9s} : {list(pm.weight.shape)}")
                # num_experts_per_tok / top_k
                cfg_m = getattr(model, "config", None)
                epk = getattr(cfg_m, "num_experts_per_tok",
                              getattr(cfg_m, "top_k", "?"))
                print(f"  num_experts_per_tok : {epk}")
                # Shared expert
                if first_moe.shared_expert_module is not None:
                    se = first_moe.shared_expert_module
                    for pname in ("gate_proj", "up_proj", "down_proj"):
                        pm = getattr(se, pname, None)
                        if pm is not None and hasattr(pm, "weight"):
                            print(f"  shared.{pname:9s}    : {list(pm.weight.shape)}")
            else:
                print(f"  WARNING: no MoE layers found — cannot proceed")
            print(f"  {'─' * 60}\n")

            if smoke_test:
                # Only keep first 4 MoE layers for smoke test
                moe_layers = [info for info in layer_infos if info.is_moe][:4]
                print(f"  SMOKE TEST: limiting to {len(moe_layers)} MoE layers")
            else:
                moe_layers = [info for info in layer_infos if info.is_moe]

            if not moe_layers:
                print("  No MoE layers found — aborting")
                continue

            def _expert_d_ff(exp) -> int:
                if isinstance(exp, PackedExpertView):
                    return exp.moe_intermediate
                w = get_expert_weights(exp)
                return w["d_ff"]

            total_expert_neurons = sum(
                sum(_expert_d_ff(exp) for exp in info.expert_modules)
                for info in moe_layers
            )
            print(f"  Total expert MLP neurons (prunable): {total_expert_neurons:,}")

            # Per-dataset baselines
            baseline_ppl_per_ds = {}
            for _ds in EVAL_DATASETS:
                _bp = evaluate_perplexity(
                    model, tokenizer, texts=all_eval_corpora[_ds],
                    max_seq_len=max_seq, batch_size=batch_sz, device=device,
                )
                baseline_ppl_per_ds[_ds] = _bp["perplexity"]
                print(f"  Baseline PPL ({_ds}): {_bp['perplexity']:.4f}")

            baseline_params = count_parameters(model)

            # Router-aware calibration
            print("\n  Collecting router-aware expert activations ...")
            calib_prompts = list(RECONSTRUCTION_TRAIN_PROMPTS)
            expert_activations = collect_expert_activations(
                model, tokenizer, moe_layers, calib_prompts,
                device=device, max_seq_len=max_seq,
            )
            routed_counts = {k: v.shape[0] for k, v in expert_activations.items()}
            n_routed_mean = (
                sum(routed_counts.values()) / len(routed_counts)
                if routed_counts else 0
            )
            print(f"  Mean routed tokens per expert: {n_routed_mean:.0f}")

            # ── Per-target loop ──────────────────────────────────────────────
            for target_pct in TARGET_PCTS:
                target_n = round(target_pct / 100.0 * total_expert_neurons)
                print(f"\n  Target: {target_pct:.1f}%  →  {target_n:,} neurons")

                # Score all experts
                all_expert_scores: List[Tuple[int, int, torch.Tensor]] = []
                for info in moe_layers:
                    for ei, expert in enumerate(info.expert_modules):
                        scores = get_expert_scores(expert)
                        all_expert_scores.append((info.layer_idx, ei, scores))

                # Global selection with per-expert cap
                flat_scores  = np.concatenate([s.numpy() for _, _, s in all_expert_scores])
                flat_layer   = np.concatenate([
                    np.full(len(s), li, dtype=np.int32)
                    for li, _, s in all_expert_scores
                ])
                flat_expert  = np.concatenate([
                    np.full(len(s), ei, dtype=np.int32)
                    for _, ei, s in all_expert_scores
                ])
                flat_neuron  = np.concatenate([
                    np.arange(len(s), dtype=np.int32)
                    for _, _, s in all_expert_scores
                ])
                order = np.argsort(flat_scores, kind="stable")

                # Per-expert caps
                expert_sizes = {(li, ei): int(len(s))
                                for li, ei, s in all_expert_scores}
                expert_caps  = {k: max(1, int(max_exp_frac * v))
                                for k, v in expert_sizes.items()}
                per_expert_pruned: Dict[Tuple[int, int], List[int]] = {
                    k: [] for k in expert_sizes
                }
                selected = 0
                for oi in order:
                    if selected >= target_n:
                        break
                    li  = int(flat_layer[oi])
                    ei  = int(flat_expert[oi])
                    ni  = int(flat_neuron[oi])
                    key = (li, ei)
                    if len(per_expert_pruned[key]) >= expert_caps[key]:
                        continue
                    per_expert_pruned[key].append(ni)
                    selected += 1

                actual_pruned = selected
                actual_pct    = 100.0 * actual_pruned / total_expert_neurons

                print(f"    Selected {actual_pruned:,} neurons ({actual_pct:.3f}%)")

                for method in METHODS:
                    print(f"\n    [method={method}]")
                    rows: List[Dict] = []
                    t_recon_total = 0.0
                    peak_gpu_mb   = 0.0

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    t0_method = time.perf_counter()
                    experts_pruned   = 0
                    experts_skipped  = 0

                    # Build a copy of the model
                    import copy
                    pruned_model = copy.deepcopy(model)
                    pruned_layers_map = {
                        info.layer_idx: info
                        for info in moe_layers
                    }

                    # Access experts from pruned_model
                    from .model_utils import get_transformer_layers as _get_layers
                    pruned_tf_layers = _get_layers(pruned_model)

                    for (li, ei), prune_list in per_expert_pruned.items():
                        if not prune_list:
                            continue

                        prune_idx = torch.tensor(sorted(prune_list), dtype=torch.long)
                        d_ff_orig = expert_sizes[(li, ei)]
                        keep_mask = torch.ones(d_ff_orig, dtype=torch.bool)
                        keep_mask[prune_idx] = False
                        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

                        n_routed  = routed_counts.get((li, ei), 0)
                        skipped   = n_routed < min_exp_tokens
                        calib_inp = expert_activations.get((li, ei), None)

                        if skipped:
                            experts_skipped += 1
                            logger.info(
                                "Layer %d Expert %d: skipped (only %d routed tokens < %d)",
                                li, ei, n_routed, min_exp_tokens,
                            )
                            row_exp = {
                                "model": model_name, "target_pruning_percent": target_pct,
                                "layer_index": li, "expert_index": ei,
                                "selector": "rmsnorm_bound", "method": method,
                                "d_ff_before": d_ff_orig, "d_ff_after": d_ff_orig,
                                "n_pruned": 0, "pruning_percent": 0.0,
                                "n_routed_tokens": n_routed,
                                "skipped": True, "dtype": dtype_str,
                            }
                            rows.append(row_exp)
                            continue

                        # Get the expert from the pruned model
                        pruned_layer = pruned_tf_layers[li]
                        pruned_mlp   = getattr(pruned_layer, "mlp", None)
                        if pruned_mlp is None or not hasattr(pruned_mlp, "experts"):
                            continue
                        expert = pruned_mlp.experts[ei]

                        # Reconstruct first (before physical pruning) if method != pure_delete
                        recon_info = {}
                        if method != "pure_delete" and calib_inp is not None:
                            t_r0 = time.perf_counter()
                            recon_info = apply_expert_residual_reconstruction(
                                expert, prune_idx, keep_idx, calib_inp,
                                ridge_lambda=BEST_RESIDUAL_LAM,
                                tau=BEST_RESIDUAL_TAU,
                            )
                            t_recon_total += time.perf_counter() - t_r0

                        # Physical pruning
                        prune_expert_channels(expert, prune_idx)
                        experts_pruned += 1

                        d_ff_new = d_ff_orig - len(prune_idx)
                        row_exp = {
                            "model": model_name, "target_pruning_percent": target_pct,
                            "layer_index": li, "expert_index": ei,
                            "selector": "rmsnorm_bound", "method": method,
                            "d_ff_before": d_ff_orig, "d_ff_after": d_ff_new,
                            "n_pruned": len(prune_idx),
                            "pruning_percent": round(100.0 * len(prune_idx) / d_ff_orig, 2),
                            "n_routed_tokens": n_routed,
                            "skipped": False,
                            "reconstruction_time_seconds": round(t_recon_total, 2),
                            "dtype": dtype_str,
                            "notes": recon_info.get("status", ""),
                        }
                        rows.append(row_exp)

                    t1_method = time.perf_counter()
                    if torch.cuda.is_available():
                        peak_gpu_mb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 2)
                        )
                        torch.cuda.empty_cache()

                    # Verify forward pass
                    fp_ok = verify_forward_pass(pruned_model, tokenizer, device)
                    if not fp_ok:
                        print("    WARNING: forward pass failed after pruning")

                    # Evaluate PPL on each dataset
                    for _ds in EVAL_DATASETS:
                        cur_bppl = baseline_ppl_per_ds[_ds]
                        ppl_info = evaluate_perplexity(
                            pruned_model, tokenizer,
                            texts=all_eval_corpora[_ds],
                            max_seq_len=max_seq, batch_size=batch_sz, device=device,
                        )
                        ppl   = ppl_info["perplexity"]
                        delta = ppl - cur_bppl
                        rel   = 100.0 * delta / cur_bppl if cur_bppl > 0 else 0.0

                        summary = {
                            "model": model_name,
                            "target_pruning_percent": target_pct,
                            "eval_dataset": _ds,
                            "selector": "rmsnorm_bound",
                            "method": method,
                            "total_experts":    sum(len(info.expert_modules)
                                                    for info in moe_layers),
                            "experts_pruned":   experts_pruned,
                            "experts_skipped":  experts_skipped,
                            "total_mlp_neurons_before": total_expert_neurons,
                            "total_mlp_neurons_pruned": actual_pruned,
                            "actual_pruning_percent":   round(actual_pct, 4),
                            "baseline_ppl":             round(cur_bppl, 4),
                            "compressed_ppl":           round(ppl, 4),
                            "delta_ppl":                round(delta, 4),
                            "relative_ppl_increase_percent": round(rel, 4),
                            "damage_reduction_percent": float("nan"),
                            "reconstruction_time_seconds": round(t_recon_total, 2),
                            "peak_gpu_memory_MB":        round(peak_gpu_mb, 1),
                            "dtype": dtype_str,
                            "notes": "" if fp_ok else "forward_pass_failed",
                        }
                        print(
                            f"    [{_ds}] baseline={cur_bppl:.4f}  "
                            f"compressed={ppl:.4f}  delta={delta:+.4f}  "
                            f"rel={rel:+.2f}%"
                        )
                        all_results.append(summary)
                        _flush_csv(main_csv_path, [summary], MOE_SUMMARY_CSV_KEYS)

                    del pruned_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as exc:
            logger.error("Failed %s: %s", model_name, exc, exc_info=True)
            print(f"  *** ERROR -- {model_name}: {exc} ***")
            all_results.append({"model": model_name, "notes": f"ERROR: {exc}"})
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # JSON report
    report = {
        "timestamp": ts,
        "mode": "moe_target_pruning",
        "models": model_list,
        "target_percents": TARGET_PCTS,
        "methods": METHODS,
        "max_expert_frac": max_exp_frac,
        "min_expert_tokens": min_exp_tokens,
        "note": "Router weights and expert routing are NOT modified. "
                "Only MLP channels within each expert are pruned.",
        "results": all_results,
    }
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\nMoE Summary CSV : {main_csv_path}")
    print(f"MoE JSON report : {json_path}\n")
