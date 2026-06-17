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
    # Local caches for packed-expert calibration (keyed by layer_idx)
    # These stay local — never attached to MoELayerInfo slots.
    hidden_cache:  Dict[int, List[torch.Tensor]] = {}
    routing_cache: Dict[int, List[torch.Tensor]] = {}

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
            # Use local dicts (keyed by layer_idx) instead of attaching to
            # MoELayerInfo — which has __slots__ and rejects dynamic attributes.
            n_exp = info.num_experts
            top_k = info.top_k
            li    = info.layer_idx
            for ei in range(n_exp):
                expert_inputs[(li, ei)] = []

            hidden_cache[li]  = []   # List of [n_tok, d_model] CPU tensors
            routing_cache[li] = []   # List of [n_tok, top_k]   CPU int tensors

            mlp_module    = info.layer_module.mlp
            router_module = info.router_module

            def _make_pre_hook(layer_idx):
                def pre_hook(module, args):
                    h = args[0].detach().float().cpu()
                    if h.dim() == 3:
                        h = h.reshape(-1, h.shape[-1])
                    hidden_cache[layer_idx].append(h)
                return pre_hook

            def _make_router_hook(layer_idx, top_k_):
                def router_hook(module, inp, out):
                    if isinstance(out, (tuple, list)):
                        if len(out) >= 2:
                            sel = out[1]
                            if isinstance(sel, torch.Tensor) and sel.dtype in (
                                    torch.int32, torch.int64, torch.long):
                                routing_cache[layer_idx].append(sel.detach().cpu())
                                return
                        logits = out[0] if isinstance(out[0], torch.Tensor) else out
                        topk = torch.topk(logits.float(),
                                          k=min(top_k_, logits.shape[-1]), dim=-1)
                        routing_cache[layer_idx].append(topk.indices.detach().cpu())
                    elif isinstance(out, torch.Tensor):
                        if out.dtype in (torch.int32, torch.int64, torch.long):
                            routing_cache[layer_idx].append(out.detach().cpu())
                        else:
                            topk = torch.topk(out.float(),
                                              k=min(top_k_, out.shape[-1]), dim=-1)
                            routing_cache[layer_idx].append(topk.indices.detach().cpu())
                return router_hook

            h1 = mlp_module.register_forward_pre_hook(_make_pre_hook(li))
            h2 = router_module.register_forward_hook(_make_router_hook(li, top_k))
            hooks.extend([h1, h2])

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
        li            = info.layer_idx
        hidden_list   = hidden_cache.get(li, [])
        routing_list  = routing_cache.get(li, [])

        if not hidden_list or not routing_list:
            logger.warning("Layer %d: no calibration data captured", li)
            continue

        all_hidden = torch.cat(hidden_list, dim=0).cpu()    # [total_tokens, d_model]
        del hidden_list                                      # free CPU memory

        try:
            all_routing = torch.cat(routing_list, dim=0).cpu()
        except RuntimeError:
            logger.warning("Layer %d: routing tensor shape mismatch, skipping", li)
            del routing_cache[li]
            continue
        del routing_list

        if all_routing.dim() == 1:
            all_routing = all_routing.unsqueeze(-1)          # [n_tok, 1]

        n_tok       = min(all_hidden.shape[0], all_routing.shape[0])
        all_hidden  = all_hidden[:n_tok]
        all_routing = all_routing[:n_tok]

        for ei in range(info.num_experts):
            mask = (all_routing == ei).any(dim=-1)           # [n_tok] bool
            if mask.any():
                expert_inputs[(li, ei)].append(all_hidden[mask].clone())

        # Free layer-level caches
        del hidden_cache[li], routing_cache[li]
        del all_hidden, all_routing

    # ── Concatenate per-expert lists into tensors ─────────────────────────────
    result: Dict[Tuple[int, int], torch.Tensor] = {}
    for key, tensors in expert_inputs.items():
        if tensors:
            result[key] = torch.cat(tensors, dim=0)          # [n_routed, d_model]

    # ── Routing statistics log ────────────────────────────────────────────────
    import statistics as _stat
    # ── Per-layer routing stats ───────────────────────────────────────────────
    layer_idx_to_info = {i.layer_idx: i for i in layer_infos}
    packed_layers = [i for i in layer_infos if i.is_moe and i.experts_packed]
    n_skipped_total = 0
    for info in packed_layers:
        li = info.layer_idx
        counts = [
            result.get((li, ei), torch.empty(0)).shape[0]
            for ei in range(info.num_experts)
        ]
        total_tok = sum(counts)
        n_zero    = sum(1 for c in counts if c == 0)
        if counts:
            mn  = min(counts)
            med = int(_stat.median(counts))
            mx  = max(counts)
            print(
                f"    Layer {li:3d}: total_tokens={total_tok:6d}  "
                f"per_expert min={mn:4d} med={med:5d} max={mx:5d}  "
                f"zero_routed={n_zero}"
            )
        n_skipped_total += n_zero
    if packed_layers:
        print(f"    Total experts with zero routed tokens: {n_skipped_total}")
        first_li   = packed_layers[0].layer_idx
        first_info = packed_layers[0]
        counts_0   = [
            result.get((first_li, ei), torch.empty(0)).shape[0]
            for ei in range(first_info.num_experts)
        ]
        print(f"    Layer {first_li} first 8 expert counts: {counts_0[:8]}")

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
    alignment: int = 1,
) -> int:
    """
    Globally prune the SAME channels from ALL experts in a packed tensor block.

    Because gate_up_proj and down_proj are fused across experts, per-expert
    variable-width pruning is not possible without unpacking.  This function
    removes a shared set of channel indices from every expert simultaneously.

    The caller is responsible for ensuring len(prune_indices) is chosen so
    that (moe_intermediate - len(prune_indices)) % alignment == 0.

    Parameters
    ----------
    experts_container : Qwen3MoeExperts (has .gate_up_proj and .down_proj)
    prune_indices     : 1-D int64 tensor of channel indices to prune,
                        in [0, moe_intermediate)
    alignment         : new_intermediate must be divisible by this value.

    Returns
    -------
    new_intermediate : int
    """
    gu = experts_container.gate_up_proj.data   # [n_exp, 2*moe_inter, hidden]
    dp = experts_container.down_proj.data       # [n_exp, hidden, moe_inter]

    moe_inter = gu.shape[1] // 2
    n_exp     = gu.shape[0]

    keep_mask = torch.ones(moe_inter, dtype=torch.bool)
    keep_mask[prune_indices] = False
    keep_idx  = keep_mask.nonzero(as_tuple=True)[0]  # [new_inter]
    new_inter = len(keep_idx)

    if alignment > 1 and new_inter % alignment != 0:
        raise RuntimeError(
            f"prune_packed_experts_global: new_intermediate={new_inter} is not "
            f"divisible by alignment={alignment}. "
            f"Prune {new_inter % alignment} extra channels to fix."
        )

    # gate_up rows: gate occupies [:moe_inter], up occupies [moe_inter:]
    gate_keep    = keep_idx
    up_keep      = keep_idx + moe_inter
    gate_up_keep = torch.cat([gate_keep, up_keep])   # [2*new_inter]

    # Slice and FORCE CONTIGUOUS (required by grouped_mm kernel)
    new_gu = gu[:, gate_up_keep, :].contiguous()   # [n_exp, 2*new_inter, hidden]
    new_dp = dp[:, :, keep_idx].contiguous()        # [n_exp, hidden, new_inter]

    experts_container.gate_up_proj = torch.nn.Parameter(new_gu)
    experts_container.down_proj    = torch.nn.Parameter(new_dp)

    if hasattr(experts_container, "intermediate_dim"):
        experts_container.intermediate_dim = new_inter

    # Diagnostic prints — verify alignment and strides
    gu_s = list(experts_container.gate_up_proj.shape)
    dp_s = list(experts_container.down_proj.shape)
    gu_st = list(experts_container.gate_up_proj.stride())
    dp_st = list(experts_container.down_proj.stride())
    gu_c  = experts_container.gate_up_proj.is_contiguous()
    dp_c  = experts_container.down_proj.is_contiguous()
    print(f"        gate_up: shape={gu_s}  stride={gu_st}  contiguous={gu_c}")
    print(f"        down:    shape={dp_s}  stride={dp_st}  contiguous={dp_c}")
    print(f"        new_intermediate={new_inter}  "
          f"new_intermediate%{alignment}={new_inter % alignment}")

    logger.info(
        "prune_packed_experts_global: %d experts, moe_inter %d → %d "
        "(pruned %d channels, alignment=%d)",
        n_exp, moe_inter, new_inter, moe_inter - new_inter, alignment,
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

def _log_gpu_memory(label: str = "") -> None:
    """Log current/peak GPU memory for all visible devices."""
    if not torch.cuda.is_available():
        return
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        peak  = torch.cuda.max_memory_allocated(i) / 1024**2
        parts.append(f"GPU{i}: {alloc:.0f}/{peak:.0f} MB")
    tag = f" [{label}]" if label else ""
    print(f"  [mem{tag}] {' | '.join(parts)}")


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
    import gc
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
    inplace_prune  = bool(cfg.get("moe_inplace_pruning", True))
    device_map_cfg = str(cfg.get("device_map", "auto"))
    chan_align     = int(cfg.get("moe_channel_alignment", 16))
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
    print(f"  chan_alignment : {chan_align}")
    if smoke_test:
        print("  SMOKE TEST MODE: only first 4 MoE layers will be processed")
    if inplace_prune:
        print("  INPLACE PRUNING: model pruned in-place (no deepcopy)")
        if len(TARGET_PCTS) > 1:
            print(f"  WARNING: moe_inplace_pruning=True but {len(TARGET_PCTS)} "
                  "target_percents specified — only first will be valid")
        if len(METHODS) > 1:
            print(f"  WARNING: moe_inplace_pruning=True but {len(METHODS)} "
                  "methods specified — only first method runs per process")
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
            _dmap = device_map_cfg if device_map_cfg != "none" else None
            model, tokenizer, _ = load_model_and_tokenizer(
                model_name=model_name, fallback_name=None,
                device=device, dtype_str=dtype_str,
                device_map=_dmap,
            )
            model.eval()
            _log_gpu_memory("after model load")

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
            _log_gpu_memory("after baseline PPL")

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
            _log_gpu_memory("after calibration")

            # ── Per-target loop ──────────────────────────────────────────────
            for target_pct in TARGET_PCTS:
                target_n = round(target_pct / 100.0 * total_expert_neurons)
                print(f"\n  Target: {target_pct:.1f}%  →  {target_n:,} neurons")

                # ── Score all experts ────────────────────────────────────────
                # Packed layers: average across all experts → single (ei=-1) entry
                # (global same-channel pruning requires a single shared score).
                layer_idx_to_info: Dict[int, "MoELayerInfo"] = {
                    i.layer_idx: i for i in moe_layers
                }
                all_expert_scores: List[Tuple[int, int, torch.Tensor]] = []
                print("  Computing expert scores ...")
                for info in moe_layers:
                    if info.experts_packed:
                        per_exp_s = []
                        for ei, exp in enumerate(info.expert_modules):
                            try:
                                per_exp_s.append(get_expert_scores(exp).float())
                            except Exception as _se:
                                logger.warning(
                                    "score layer=%d ei=%d: %s",
                                    info.layer_idx, ei, _se,
                                )
                        if per_exp_s:
                            avg_s = torch.stack(per_exp_s, dim=0).mean(dim=0)
                            all_expert_scores.append((info.layer_idx, -1, avg_s))
                    else:
                        for ei, exp in enumerate(info.expert_modules):
                            try:
                                s = get_expert_scores(exp)
                                all_expert_scores.append((info.layer_idx, ei, s))
                            except Exception as _se:
                                logger.warning(
                                    "score layer=%d ei=%d: %s",
                                    info.layer_idx, ei, _se,
                                )

                # ── Global selection — correct accounting for packed layers ───
                # For packed layers (ei=-1), selecting one channel removes
                # num_experts expert-neurons (not 1).  We weight accordingly
                # so target_n (in expert-neurons) is honoured correctly.
                #
                # entry_weight[(li, ei)] = expert-neurons removed per channel.
                entry_weight: Dict[Tuple[int, int], int] = {}
                for _li, _ei, _s in all_expert_scores:
                    if _ei == -1:
                        entry_weight[(_li, _ei)] = layer_idx_to_info[_li].num_experts
                    else:
                        entry_weight[(_li, _ei)] = 1

                # Keep per-layer avg scores for alignment adjustment later
                layer_avg_scores: Dict[int, torch.Tensor] = {
                    _li: _s for _li, _ei, _s in all_expert_scores if _ei == -1
                }

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

                expert_sizes = {(li, ei): int(len(s))
                                for li, ei, s in all_expert_scores}
                # Cap = max layer-channels per entry (not per expert-neuron)
                expert_caps  = {k: max(1, int(max_exp_frac * v))
                                for k, v in expert_sizes.items()}
                per_expert_pruned: Dict[Tuple[int, int], List[int]] = {
                    k: [] for k in expert_sizes
                }

                removed_expert_neurons = 0
                selected_layer_channels = 0
                for oi in order:
                    if removed_expert_neurons >= target_n:
                        break
                    li  = int(flat_layer[oi])
                    ei  = int(flat_expert[oi])
                    ni  = int(flat_neuron[oi])
                    key = (li, ei)
                    if len(per_expert_pruned[key]) >= expert_caps[key]:
                        continue
                    per_expert_pruned[key].append(ni)
                    wt = entry_weight[key]
                    removed_expert_neurons  += wt
                    selected_layer_channels += 1  # counts channel-slots, not expert-neurons

                # ── Alignment adjustment for packed layers ────────────────────
                # After selection, new_inter = old_inter - k_selected may not
                # be divisible by chan_align.  Round new_inter DOWN to the
                # nearest multiple of chan_align by pruning the next-lowest-
                # scoring channels (using the stored averaged score vector).
                for key, prune_list in per_expert_pruned.items():
                    _li, _ei = key
                    if _ei != -1:
                        continue  # unpacked: no alignment needed
                    old_inter = expert_sizes[key]
                    k = len(prune_list)
                    new_inter_raw = old_inter - k
                    new_inter_aligned = (new_inter_raw // chan_align) * chan_align
                    extra = new_inter_raw - new_inter_aligned  # channels to add
                    if extra == 0:
                        continue
                    if new_inter_aligned <= 0:
                        print(f"    WARNING: alignment={chan_align} would reduce "
                              f"layer {_li} to 0 channels — skipping alignment")
                        continue
                    # Find extra lowest-scoring channels NOT already in prune_list
                    pruned_set = set(prune_list)
                    avg_s = layer_avg_scores[_li]  # [old_inter]
                    # Sort remaining indices by score (ascending = lowest first)
                    remaining = [
                        (float(avg_s[ch]), ch)
                        for ch in range(old_inter)
                        if ch not in pruned_set
                    ]
                    remaining.sort()
                    for _, ch in remaining[:extra]:
                        prune_list.append(ch)
                        removed_expert_neurons += entry_weight[key]
                        selected_layer_channels += 1

                actual_pruned = removed_expert_neurons
                actual_pct    = 100.0 * actual_pruned / total_expert_neurons
                n_packed_layers = sum(
                    1 for (_li, _ei) in per_expert_pruned if _ei == -1
                )
                print(f"    selected_layer_channels  : {selected_layer_channels:,}")
                print(f"    removed_expert_neurons   : {actual_pruned:,}")
                print(f"    actual_pct               : {actual_pct:.3f}%  "
                      f"(requested {target_pct:.1f}%)")

                # ── Release calibration caches before pruning ─────────────────
                _calib_ref = expert_activations   # keep local ref for recon
                del expert_activations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _log_gpu_memory("after cache release, before pruning")

                for _method_idx, method in enumerate(METHODS):
                    if inplace_prune and _method_idx > 0:
                        print("  [skipping] moe_inplace_pruning=True — "
                              "only first method runs per process")
                        break

                    print(f"\n    [method={method}]")
                    rows: List[Dict] = []
                    t_recon_total = 0.0
                    peak_gpu_mb   = 0.0
                    experts_pruned  = 0
                    experts_skipped = 0

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # ── In-place pruning — no deepcopy ────────────────────────
                    for (li, ei), prune_list in per_expert_pruned.items():
                        if not prune_list:
                            continue

                        info      = layer_idx_to_info[li]
                        d_ff_orig = expert_sizes[(li, ei)]
                        prune_idx = torch.tensor(sorted(prune_list), dtype=torch.long)

                        if ei == -1:
                            # ── Packed layer: global same-channel pruning ─────
                            ec              = info.experts_container
                            n_pruned_actual = len(prune_idx)
                            new_inter_exp   = d_ff_orig - n_pruned_actual
                            removed_en      = n_pruned_actual * info.num_experts
                            new_inter = prune_packed_experts_global(
                                ec, prune_idx, alignment=chan_align
                            )
                            for pv in info.expert_modules:
                                if isinstance(pv, PackedExpertView):
                                    pv.moe_intermediate = new_inter
                            experts_pruned += info.num_experts
                            rows.append({
                                "model": model_name,
                                "target_pruning_percent": target_pct,
                                "layer_index": li, "expert_index": -1,
                                "selector": "rmsnorm_bound_avg",
                                "method": method,
                                "d_ff_before": d_ff_orig,
                                "d_ff_after":  new_inter,
                                "n_pruned": n_pruned_actual,
                                "pruning_percent": round(
                                    100.0 * n_pruned_actual / d_ff_orig, 2),
                                "n_routed_tokens": sum(
                                    routed_counts.get((li, ej), 0)
                                    for ej in range(info.num_experts)
                                ),
                                "skipped": False, "dtype": dtype_str,
                                "old_moe_intermediate": d_ff_orig,
                                "new_moe_intermediate": new_inter,
                                "moe_channel_alignment": chan_align,
                                "removed_expert_neurons": removed_en,
                                "actual_expert_neuron_pct": round(
                                    100.0 * removed_en / total_expert_neurons, 4),
                            })
                            print(
                                f"      Layer {li}: "
                                f"pruned {n_pruned_actual} channels from "
                                f"{info.num_experts} packed experts "
                                f"({d_ff_orig}→{new_inter})  "
                                f"new_inter%{chan_align}={new_inter % chan_align}  "
                                f"removed_expert_neurons={removed_en:,}"
                            )

                        else:
                            # ── Unpacked layer: per-expert in-place ───────────
                            n_routed  = routed_counts.get((li, ei), 0)
                            if n_routed < min_exp_tokens:
                                experts_skipped += 1
                                logger.info(
                                    "Layer %d Expert %d: skipped "
                                    "(%d routed < %d)",
                                    li, ei, n_routed, min_exp_tokens,
                                )
                                rows.append({
                                    "model": model_name,
                                    "target_pruning_percent": target_pct,
                                    "layer_index": li, "expert_index": ei,
                                    "selector": "rmsnorm_bound",
                                    "method": method,
                                    "d_ff_before": d_ff_orig,
                                    "d_ff_after": d_ff_orig,
                                    "n_pruned": 0, "pruning_percent": 0.0,
                                    "n_routed_tokens": n_routed,
                                    "skipped": True, "dtype": dtype_str,
                                })
                                continue

                            expert    = info.expert_modules[ei]
                            keep_mask = torch.ones(d_ff_orig, dtype=torch.bool)
                            keep_mask[prune_idx] = False
                            keep_idx  = keep_mask.nonzero(as_tuple=True)[0]
                            calib_inp = _calib_ref.get((li, ei), None)

                            recon_info: Dict = {}
                            if method != "pure_delete" and calib_inp is not None:
                                t_r0 = time.perf_counter()
                                recon_info = apply_expert_residual_reconstruction(
                                    expert, prune_idx, keep_idx, calib_inp,
                                    ridge_lambda=BEST_RESIDUAL_LAM,
                                    tau=BEST_RESIDUAL_TAU,
                                )
                                t_recon_total += time.perf_counter() - t_r0

                            prune_expert_channels(expert, prune_idx)
                            experts_pruned += 1

                            d_ff_new = d_ff_orig - len(prune_idx)
                            rows.append({
                                "model": model_name,
                                "target_pruning_percent": target_pct,
                                "layer_index": li, "expert_index": ei,
                                "selector": "rmsnorm_bound",
                                "method": method,
                                "d_ff_before": d_ff_orig, "d_ff_after": d_ff_new,
                                "n_pruned": len(prune_idx),
                                "pruning_percent": round(
                                    100.0 * len(prune_idx) / d_ff_orig, 2),
                                "n_routed_tokens": n_routed,
                                "skipped": False,
                                "reconstruction_time_seconds": round(
                                    t_recon_total, 2),
                                "dtype": dtype_str,
                                "notes": recon_info.get("status", ""),
                            })

                    _log_gpu_memory("after pruning")

                    # ── Forward check ─────────────────────────────────────────
                    # If forward pass fails, record error and skip PPL eval.
                    fp_ok = verify_forward_pass(model, tokenizer, device)
                    _log_gpu_memory("after forward check")

                    _has_packed = any(i.experts_packed for i in moe_layers)
                    _sel_str    = ("rmsnorm_bound_avg"
                                   if _has_packed else "rmsnorm_bound")

                    if not fp_ok:
                        print("    ERROR: forward pass failed — skipping PPL eval")
                        for _ds in EVAL_DATASETS:
                            err_row = {
                                "model": model_name,
                                "target_pruning_percent": target_pct,
                                "eval_dataset": _ds,
                                "selector": _sel_str,
                                "method": method,
                                "total_experts":    sum(len(i.expert_modules)
                                                        for i in moe_layers),
                                "experts_pruned":   experts_pruned,
                                "experts_skipped":  experts_skipped,
                                "total_mlp_neurons_before": total_expert_neurons,
                                "total_mlp_neurons_pruned": actual_pruned,
                                "actual_pruning_percent":   round(actual_pct, 4),
                                "requested_target_pct":     target_pct,
                                "selected_layer_channels":  selected_layer_channels,
                                "removed_expert_neurons":   actual_pruned,
                                "moe_channel_alignment":    chan_align,
                                "baseline_ppl":    round(baseline_ppl_per_ds[_ds], 4),
                                "compressed_ppl":  float("nan"),
                                "delta_ppl":       float("nan"),
                                "relative_ppl_increase_percent": float("nan"),
                                "damage_reduction_percent":       float("nan"),
                                "reconstruction_time_seconds":    round(t_recon_total, 2),
                                "peak_gpu_memory_MB": round(peak_gpu_mb, 1),
                                "dtype": dtype_str,
                                "notes": "forward_pass_failed",
                            }
                            all_results.append(err_row)
                            _flush_csv(main_csv_path, [err_row], MOE_SUMMARY_CSV_KEYS)
                        _log_gpu_memory("after forward fail")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue  # next method

                    # ── PPL eval ─────────────────────────────────────────────
                    for _ds in EVAL_DATASETS:
                        cur_bppl = baseline_ppl_per_ds[_ds]
                        ppl_info = evaluate_perplexity(
                            model, tokenizer,
                            texts=all_eval_corpora[_ds],
                            max_seq_len=max_seq, batch_size=batch_sz,
                            device=device,
                        )
                        ppl   = ppl_info["perplexity"]
                        delta = ppl - cur_bppl
                        rel   = 100.0 * delta / cur_bppl if cur_bppl > 0 else 0.0
                        if torch.cuda.is_available():
                            peak_gpu_mb = (
                                torch.cuda.max_memory_allocated() / 1024**2
                            )

                        summary = {
                            "model": model_name,
                            "target_pruning_percent": target_pct,
                            "eval_dataset": _ds,
                            "selector": _sel_str,
                            "method": method,
                            "total_experts":    sum(len(i.expert_modules)
                                                    for i in moe_layers),
                            "experts_pruned":   experts_pruned,
                            "experts_skipped":  experts_skipped,
                            "total_mlp_neurons_before": total_expert_neurons,
                            "total_mlp_neurons_pruned": actual_pruned,
                            "actual_pruning_percent":   round(actual_pct, 4),
                            "requested_target_pct":     target_pct,
                            "selected_layer_channels":  selected_layer_channels,
                            "removed_expert_neurons":   actual_pruned,
                            "moe_channel_alignment":    chan_align,
                            "baseline_ppl":             round(cur_bppl, 4),
                            "compressed_ppl":           round(ppl, 4),
                            "delta_ppl":                round(delta, 4),
                            "relative_ppl_increase_percent": round(rel, 4),
                            "damage_reduction_percent": float("nan"),
                            "reconstruction_time_seconds": round(t_recon_total, 2),
                            "peak_gpu_memory_MB":        round(peak_gpu_mb, 1),
                            "dtype": dtype_str,
                            "notes": "",
                        }
                        print(
                            f"    [{_ds}] baseline={cur_bppl:.4f}  "
                            f"compressed={ppl:.4f}  delta={delta:+.4f}  "
                            f"rel={rel:+.2f}%"
                        )
                        all_results.append(summary)
                        _flush_csv(main_csv_path, [summary], MOE_SUMMARY_CSV_KEYS)

                    _log_gpu_memory("after eval")
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
