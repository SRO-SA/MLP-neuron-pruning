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
    "relative_delta_pct",
    "reconstruction_time_seconds", "peak_gpu_memory_MB",
    "dtype", "notes",
]

MOE_SUMMARY_CSV_KEYS = [
    # Identity
    "model", "target_pruning_percent", "eval_dataset",
    "pruning_mode", "aggregation_mode", "selector", "method",
    "physical_pruning", "speedup_expected", "same_channel_across_experts",
    # Layer counts
    "smoke_layers_used", "total_moe_layers",
    # Accounting
    "total_experts", "experts_pruned", "experts_skipped",
    "total_mlp_neurons_before", "total_mlp_neurons_pruned",
    "requested_target_pct", "actual_pruning_percent",
    "selected_layer_channels", "removed_expert_neurons",
    "old_intermediate", "new_intermediate", "moe_channel_alignment",
    # PPL
    "baseline_ppl", "compressed_ppl", "delta_ppl",
    "relative_delta_pct", "damage_reduction_percent",
    # Diagnostics
    "forward_check", "reconstruction_time_seconds",
    "peak_gpu_memory_MB", "gpu0_peak_mb", "gpu1_peak_mb",
    "time_sec", "dtype", "notes",
    # Structural / residual stats
    "shape_changed",
    "residual_stable_experts", "residual_skipped_experts",
    "residual_failed_experts", "residual_time_sec",
    "csv_path", "json_path",
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
# Activation-based expert scoring
# ---------------------------------------------------------------------------

def compute_activation_scores_for_expert(
    expert_module,
    calib_inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Activation-weighted importance score for SwiGLU expert channels.

    score_i = mean_over_tokens( |SiLU(x @ gate_i.T) * (x @ up_i.T)| ) * ||down[:, i]||

    This measures how much each intermediate neuron actually fires on
    calibration data, weighted by how large a change its removal causes
    in the output (down column norm).

    Args:
        expert_module: module with gate/up/down weights accessible via
                       get_expert_weights().
        calib_inputs:  [N, d_model] float tensor of calibration hidden states
                       routed to this expert.

    Returns:
        [d_ff] float32 CPU tensor of scores.
    """
    import torch.nn.functional as _F
    if calib_inputs is None or calib_inputs.shape[0] == 0:
        # Fallback to weight-only score if no calibration data
        return get_expert_scores(expert_module)
    try:
        w      = get_expert_weights(expert_module)
        gate   = w["gate_proj"].detach().float()   # [d_ff, d_model]
        up     = w["up_proj"].detach().float()     # [d_ff, d_model]
        down   = w["down_proj"].detach().float()   # [d_model, d_ff]
        X      = calib_inputs.detach().float()     # [N, d_model]

        # Move everything to the same device (CPU keeps memory usage low)
        X    = X.cpu()
        gate = gate.cpu()
        up   = up.cpu()
        down = down.cpu()

        with torch.no_grad():
            gate_out  = X @ gate.T                       # [N, d_ff]
            up_out    = X @ up.T                         # [N, d_ff]
            act       = _F.silu(gate_out) * up_out       # [N, d_ff]
            act_score = act.abs().mean(dim=0)             # [d_ff]
            down_norms = down.norm(dim=0)                 # [d_ff] column norms
            scores    = act_score * down_norms
        return scores
    except Exception as exc:
        logger.warning("activation_score fallback to rmsnorm_bound: %s", exc)
        return get_expert_scores(expert_module)


def _score_expert_moe(
    expert_module,
    selector: str,
    calib_inputs: "Optional[torch.Tensor]" = None,
) -> torch.Tensor:
    """
    Dispatcher: returns [d_ff] importance scores for one expert.

    selector options:
      "rmsnorm_bound"    – weight-only RMSNorm-bounded SwiGLU score
      "activation_score" – activation × down-column-norm score (needs calib)
    """
    if selector == "activation_score":
        return compute_activation_scores_for_expert(expert_module, calib_inputs)
    else:
        # default: rmsnorm_bound (weight-only, no calib needed)
        return get_expert_scores(expert_module)


# ---------------------------------------------------------------------------
# Packed-expert residual reconstruction
# ---------------------------------------------------------------------------

def apply_packed_residual_for_layer(
    experts_container,
    prune_idx:   torch.Tensor,
    keep_idx:    torch.Tensor,
    expert_activations_for_layer: "Dict[int, Optional[torch.Tensor]]",
    min_tokens:   int   = 16,
    ridge_lambda: float = 1e-2,
    tau:          float = 1.0,
    solve_on_cpu: bool  = True,
) -> "Dict":
    """
    Ridge-regression residual reconstruction for packed MoE experts.

    MUST be called BEFORE prune_packed_experts_global because it reads
    the full (pre-pruning) gate_up_proj and down_proj.

    For each expert e in the packed layer we want to minimise:
        || A_K · ΔD^T - A_P · W_P^T ||_F
    where:
        A = SiLU(X_e @ G_e^T) ⊙ (X_e @ U_e^T)   # [N, d_ff] activations
        A_K = A[:, keep_idx],  A_P = A[:, prune_idx]
        W_P = down_proj[e, :, prune_idx]           # pruned columns

    Dual-form ridge solve (efficient when N < n_kept):
        (A_K A_K^T + λ I) B = E,   E = A_P W_P^T   # [N, d_model]
        ΔD = A_K^T B                                # [n_kept, d_model]

    Update: down_proj[e, :, keep_idx] += τ · ΔD^T  (in-place, pre-pruning)

    Returns stats dict with n_stable, n_skipped, n_failed, mean_tokens.
    """
    import torch.nn.functional as _F

    gu     = experts_container.gate_up_proj.data   # [n_exp, 2*inter, d_model]
    dp     = experts_container.down_proj.data       # [n_exp, d_model, inter]
    inter  = gu.shape[1] // 2
    n_exp  = gu.shape[0]

    n_stable = n_skipped = n_failed = 0
    n_tokens_list: "List[int]" = []

    for ei in range(n_exp):
        X_raw = expert_activations_for_layer.get(ei, None)
        if X_raw is None or X_raw.shape[0] < min_tokens:
            n_skipped += 1
            continue

        N = X_raw.shape[0]
        try:
            with torch.no_grad():
                X      = X_raw.detach().float()
                gate_e = gu[ei, :inter, :].detach().float()   # [inter, d_model]
                up_e   = gu[ei, inter:, :].detach().float()   # [inter, d_model]
                down_e = dp[ei].detach().float()               # [d_model, inter]

                if solve_on_cpu:
                    X      = X.cpu()
                    gate_e = gate_e.cpu()
                    up_e   = up_e.cpu()
                    down_e = down_e.cpu()

                # SwiGLU activations [N, inter]
                act_all = _F.silu(X @ gate_e.T) * (X @ up_e.T)

                _prune = prune_idx.to(act_all.device)
                _keep  = keep_idx.to(act_all.device)

                A_P = act_all[:, _prune]      # [N, n_pruned]
                A_K = act_all[:, _keep]       # [N, n_kept]
                W_P = down_e[:, _prune.to(down_e.device)]     # [d_model, n_pruned]

                # Target residual: what the pruned neurons were contributing
                E   = A_P @ W_P.T             # [N, d_model]

                # Dual-form ridge (N×N system, efficient when N is small)
                AAt = A_K @ A_K.T             # [N, N]
                lam = ridge_lambda * float(AAt.diagonal().mean())
                reg = lam * torch.eye(N, dtype=torch.float32, device=AAt.device)

                B     = torch.linalg.solve(AAt + reg, E)  # [N, d_model]
                Delta = A_K.T @ B                          # [n_kept, d_model]

                W_K     = down_e[:, _keep.to(down_e.device)]  # [d_model, n_kept]
                W_K_new = W_K + tau * Delta.T                  # [d_model, n_kept]

                # Write back at original dtype and device
                dp[ei, :, keep_idx.to(dp.device)] = W_K_new.to(
                    device=dp.device, dtype=dp.dtype
                )

            n_stable += 1
            n_tokens_list.append(N)

        except Exception as exc:
            logger.warning(
                "apply_packed_residual: expert %d solve failed: %s", ei, exc
            )
            n_failed += 1

    mean_toks = (
        round(sum(n_tokens_list) / len(n_tokens_list))
        if n_tokens_list else 0
    )
    return {
        "n_stable":  n_stable,
        "n_skipped": n_skipped,
        "n_failed":  n_failed,
        "mean_tokens": mean_toks,
    }




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

def _aggregate_expert_scores(
    stacked: "torch.Tensor",
    aggregation: str,
    routing_weights: "Optional[torch.Tensor]" = None,
) -> "torch.Tensor":
    """Aggregate [n_experts, d_ff] per-expert scores → [d_ff] layer score.

    aggregation options
    -------------------
    p95               : 95th-percentile over experts (conservative — channel
                        must be weak for 95% of experts to be pruned)
    max               : maximum over experts (even more conservative)
    router_weighted_mean : route-traffic-weighted average
    mean              : plain average (fallback / least conservative)
    """
    if aggregation == "max":
        return stacked.max(dim=0).values
    if aggregation == "p95":
        return torch.quantile(stacked.float(), 0.95, dim=0)
    if aggregation == "router_weighted_mean" and routing_weights is not None:
        w = routing_weights.float()
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = torch.ones_like(w) / max(len(w), 1)
        return (stacked.float() * w.unsqueeze(1)).sum(dim=0)
    # fallback: plain mean
    return stacked.float().mean(dim=0)


def _print_per_layer_distribution(
    per_expert_pruned: Dict,
    expert_sizes: Dict,
    moe_layers: List,
    routed_counts: Dict,
    total_expert_neurons: int,
    chan_align: int,
    pruning_mode: str = "packed_same_channel",
) -> None:
    """Print a per-layer summary of channel pruning decisions.

    Three cases:
      packed_same_channel : (li, -1) key — physical reshape, old_i→new_i
      per_expert_mask     : (li, ei) keys on packed layer — mask-only, old_i=new_i
      unpacked per-expert : (li, ei) keys on individual modules — physical per-expert
    """
    li_to_info = {i.layer_idx: i for i in moe_layers}
    pruned_layers = sorted(
        set(_li for (_li, _ei), plist in per_expert_pruned.items() if plist),
        key=lambda x: x,
    )
    if not pruned_layers:
        return

    is_mask_only = (pruning_mode == "per_expert_mask")

    print("\n  Per-layer pruning distribution:")
    if is_mask_only:
        hdr = (
            f"    {'layer':>6}  {'n_exp':>6}  {'old_i':>6}  "
            f"{'masked':>7}  {'new_i':>6}  {'shp':>5}  "
            f"{'rem_en':>9}  {'lyr%':>6}  "
            f"{'0rt':>4}  {'min_rt':>6}  {'med_rt':>6}  {'max_rt':>6}"
        )
    else:
        hdr = (
            f"    {'layer':>6}  {'n_exp':>6}  {'old_i':>6}  "
            f"{'pruned':>7}  {'new_i':>6}  "
            f"{'rem_en':>9}  {'lyr%':>6}  "
            f"{'0rt':>4}  {'min_rt':>6}  {'med_rt':>6}  {'max_rt':>6}"
        )
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))

    for _li in pruned_layers:
        info  = li_to_info[_li]
        n_exp = info.num_experts
        packed = info.experts_packed

        if is_mask_only and packed:
            # per_expert_mask on packed layer: (li, ei) keys, no shape change.
            # Aggregate across all experts in this layer.
            total_masked = sum(
                len(per_expert_pruned.get((_li, _ei), []))
                for _ei in range(n_exp)
            )
            n_exp_masked = sum(
                1 for _ei in range(n_exp)
                if per_expert_pruned.get((_li, _ei), [])
            )
            old_i = expert_sizes.get((_li, 0), 0)
            new_i = old_i   # mask-only: shape unchanged
            rem   = total_masked   # each (ei, ch) pair = 1 expert-neuron
            pct   = 100.0 * rem / total_expert_neurons if total_expert_neurons else 0.0
            cnts  = [routed_counts.get((_li, ej), 0) for ej in range(n_exp)]
            z     = sum(1 for c in cnts if c == 0)
            nz    = sorted(c for c in cnts if c > 0) or [0]
            mn, mx, med = nz[0], nz[-1], nz[len(nz)//2]
            print(
                f"    {_li:>6}  {n_exp_masked:>6}  {old_i:>6}  "
                f"{total_masked:>7}  {new_i:>6}  {'N':>5}  "
                f"{rem:>9,}  {pct:>5.2f}%  "
                f"{z:>4}  {mn:>6}  {med:>6}  {mx:>6}"
            )

        elif packed:
            # packed_same_channel: (li, -1) key, physical shape change.
            key   = (_li, -1)
            plist = per_expert_pruned.get(key, [])
            old_i = expert_sizes.get(key, 0)
            n_pr  = len(plist)
            new_i = old_i - n_pr
            rem   = n_pr * n_exp
            pct   = 100.0 * rem / total_expert_neurons if total_expert_neurons else 0.0
            cnts  = [routed_counts.get((_li, ej), 0) for ej in range(n_exp)]
            z     = sum(1 for c in cnts if c == 0)
            nz    = sorted(c for c in cnts if c > 0) or [0]
            mn, mx, med = nz[0], nz[-1], nz[len(nz)//2]
            print(
                f"    {_li:>6}  {n_exp:>6}  {old_i:>6}  "
                f"{n_pr:>7}  {new_i:>6}  "
                f"{rem:>9,}  {pct:>5.2f}%  "
                f"{z:>4}  {mn:>6}  {med:>6}  {mx:>6}"
            )

        else:
            # Unpacked: one line per pruned expert.
            for ei in range(n_exp):
                key   = (_li, ei)
                plist = per_expert_pruned.get(key, [])
                if not plist:
                    continue
                old_i = expert_sizes.get(key, 0)
                n_pr  = len(plist)
                new_i = old_i - n_pr if not is_mask_only else old_i
                rem   = n_pr
                pct   = 100.0 * rem / total_expert_neurons if total_expert_neurons else 0.0
                n_rt  = routed_counts.get((_li, ei), 0)
                print(
                    f"    {_li:>6}  {1:>6}  {old_i:>6}  "
                    f"{n_pr:>7}  {new_i:>6}  "
                    f"{rem:>9,}  {pct:>5.2f}%  "
                    f"{'n/a':>4}  {n_rt:>6}  {n_rt:>6}  {n_rt:>6}"
                )
    print()


def _print_moe_summary_table(
    all_results: List[Dict],
    n_smoke_layers: int,
    total_moe_layers: int,
    main_csv_path: str,
    json_path: str,
) -> None:
    """Print final MoE experiment summary table."""
    rows = [r for r in all_results if "baseline_ppl" in r and "model" in r]
    if not rows:
        return

    W = 190
    print(f"\n{'=' * W}")
    print("MOE EXPERIMENT SUMMARY")
    print(f"{'=' * W}")

    hdr = (
        f"  {'model':>22}  {'s/t':>5}  {'tgt%':>5}  {'act%':>6}  "
        f"{'sel_ch':>6}  {'rem_en':>9}  {'old_i':>5}  {'new_i':>5}  "
        f"{'aln':>3}  {'selector':>18}  {'agg':>8}  {'p_mode':>22}  {'method':>18}  "
        f"{'shp':>5}  {'r_stbl':>6}  {'r_skip':>6}  {'r_fail':>6}  "
        f"{'dataset':>10}  {'bPPL':>8}  {'cPPL':>9}  {'dPPL':>8}  {'rel%':>7}  "
        f"{'fwd':>4}  {'status':>16}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in rows:
        mdl    = str(r.get("model",""))[-22:]
        st     = f"{n_smoke_layers}/{total_moe_layers}"
        tgt    = r.get("requested_target_pct", r.get("target_pruning_percent", 0.0))
        act    = r.get("actual_pruning_percent", 0.0)
        sel    = r.get("selected_layer_channels", "?")
        rem    = r.get("removed_expert_neurons", "?")
        old_i  = r.get("old_intermediate", "?")
        new_i  = r.get("new_intermediate", "?")
        aln    = r.get("moe_channel_alignment", "?")
        selstr = str(r.get("selector",""))[:18]
        agg    = str(r.get("aggregation_mode",""))[:8]
        pm     = str(r.get("pruning_mode",""))[:22]
        meth   = str(r.get("method",""))[:18]
        shp    = "Y" if r.get("shape_changed", False) else "N"
        r_stbl = r.get("residual_stable_experts", "-")
        r_skip = r.get("residual_skipped_experts", "-")
        r_fail = r.get("residual_failed_experts", "-")
        ds     = str(r.get("eval_dataset",""))[:10]
        bppl   = r.get("baseline_ppl", float("nan"))
        cppl   = r.get("compressed_ppl", float("nan"))
        dppl   = r.get("delta_ppl", float("nan"))
        rel    = r.get("relative_delta_pct", float("nan"))
        fwd    = "OK" if r.get("forward_check", True) else "FAIL"
        st_s   = str(r.get("notes","") or "ok")[:16]
        try:
            print(
                f"  {mdl:>22}  {st:>5}  {tgt:>5.1f}  {act:>6.3f}  "
                f"{str(sel):>6}  {str(rem):>9}  {str(old_i):>5}  {str(new_i):>5}  "
                f"{str(aln):>3}  {selstr:>18}  {agg:>8}  {pm:>22}  {meth:>18}  "
                f"{str(shp):>5}  {str(r_stbl):>6}  {str(r_skip):>6}  {str(r_fail):>6}  "
                f"{ds:>10}  {bppl:>8.4f}  {cppl:>9.4f}  {dppl:>8.4f}  {rel:>7.2f}  "
                f"{fwd:>4}  {st_s:>16}"
            )
        except (TypeError, ValueError):
            print(f"  {mdl}  (row format error)")

    print(f"{'=' * W}")
    print(f"  CSV : {main_csv_path}")
    print(f"  JSON: {json_path}")
    print(f"{'=' * W}\n")


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
    chan_agg       = str(cfg.get("moe_same_channel_aggregation", "p95"))
    pruning_mode   = str(cfg.get("moe_pruning_mode", "packed_same_channel"))
    moe_selector   = str(cfg.get("moe_selector", "rmsnorm_bound"))
    resid_lambda   = float(cfg.get("residual_lambda", 1e-2))
    resid_tau      = float(cfg.get("residual_tau", 1.0))
    min_resid_tok  = int(cfg.get("min_residual_tokens_per_expert", 16))
    resid_on_cpu   = bool(cfg.get("solve_residual_on_cpu", True))
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
    print(f"  aggregation    : {chan_agg}")
    print(f"  pruning_mode   : {pruning_mode}")
    print(f"  moe_selector   : {moe_selector}")
    if smoke_test:
        print("  SMOKE TEST MODE: only first 4 MoE layers will be processed")
    if inplace_prune:
        print("  INPLACE PRUNING: model pruned in-place (no deepcopy)")
        if len(TARGET_PCTS) > 1:
            raise RuntimeError(
                f"moe_inplace_pruning=True supports only ONE target percent per "
                f"process, but {len(TARGET_PCTS)} were specified: {TARGET_PCTS}.\n"
                f"Run separate processes (one per target), or set\n"
                f"  moe_inplace_pruning: false\nto use deepcopy mode."
            )
        if len(METHODS) > 1:
            print(f"  WARNING: moe_inplace_pruning=True but {len(METHODS)} "
                  "methods — only the first method will run per process")
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
                    if info.experts_packed and pruning_mode == "packed_same_channel":
                        # Aggregate per-expert scores to a single layer-level score.
                        # Selection then picks globally across layers using this score.
                        per_exp_s: List[torch.Tensor] = []
                        rt_weights: List[float] = []
                        _total_rt = max(
                            sum(routed_counts.get((info.layer_idx, ej), 0)
                                for ej in range(info.num_experts)),
                            1,
                        )
                        for ei, exp in enumerate(info.expert_modules):
                            try:
                                _calib_ei = (
                                    expert_activations.get((info.layer_idx, ei))
                                    if moe_selector == "activation_score"
                                    else None
                                )
                                per_exp_s.append(
                                    _score_expert_moe(exp, moe_selector, _calib_ei).float()
                                )
                                rt_weights.append(
                                    routed_counts.get((info.layer_idx, ei), 0)
                                    / _total_rt
                                )
                            except Exception as _se:
                                logger.warning(
                                    "score layer=%d ei=%d: %s",
                                    info.layer_idx, ei, _se,
                                )
                        if per_exp_s:
                            stacked_s = torch.stack(per_exp_s, dim=0)  # [n_exp, d_ff]
                            rt_t      = torch.tensor(rt_weights, dtype=torch.float32)
                            agg_s     = _aggregate_expert_scores(stacked_s, chan_agg, rt_t)
                            all_expert_scores.append((info.layer_idx, -1, agg_s))
                    else:
                        # per_expert_mask mode (packed or unpacked) OR unpacked
                        # packed_same_channel: use true per-expert scores.
                        for ei, exp in enumerate(info.expert_modules):
                            try:
                                _calib_ei = (
                                    expert_activations.get((info.layer_idx, ei))
                                    if moe_selector == "activation_score"
                                    else None
                                )
                                s = _score_expert_moe(exp, moe_selector, _calib_ei)
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
                _print_per_layer_distribution(
                    per_expert_pruned, expert_sizes, moe_layers,
                    routed_counts, total_expert_neurons, chan_align,
                    pruning_mode=pruning_mode,
                )

                # ── Release calibration caches before pruning ─────────────────
                # pure_delete needs no per-expert activations during pruning;
                # reconstruction methods do.  Only hold a reference if needed.
                _first_method = METHODS[0] if METHODS else "pure_delete"
                _needs_calib  = (_first_method != "pure_delete")
                if _needs_calib:
                    _calib_ref = expert_activations   # keep alive for recon
                else:
                    _calib_ref = None
                del expert_activations   # release GPU/CPU memory regardless
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
                    _t_start = time.perf_counter()

                    # ── In-place pruning — no deepcopy ────────────────────────
                    # Determine summary intermediate sizes (updated per-layer)
                    _old_inter_summary = 0
                    _new_inter_summary = 0
                    # For per_expert_mask, shape never changes — pre-populate from
                    # expert_sizes so the summary always shows the correct value
                    # even when layers_to_mask ends up empty.
                    if pruning_mode == "per_expert_mask":
                        _any_key = next(
                            (k for k in expert_sizes if k[1] >= 0), None
                        )
                        if _any_key:
                            _old_inter_summary = expert_sizes[_any_key]
                            _new_inter_summary = _old_inter_summary
                    # Residual reconstruction counters
                    _resid_stable  = 0
                    _resid_skipped = 0
                    _resid_failed  = 0

                    if pruning_mode == "per_expert_mask":
                        # ── MASK-ONLY: zero weights without changing shapes ────
                        # Groups by layer so we do one container pass per layer.
                        layers_to_mask: Dict[int, Dict[int, List[int]]] = {}
                        for (li, ei), prune_list in per_expert_pruned.items():
                            if prune_list:
                                layers_to_mask.setdefault(li, {})[ei] = prune_list

                        total_masked_en = 0
                        for li, exp_dict in sorted(layers_to_mask.items()):
                            info = layer_idx_to_info[li]
                            if info.experts_packed and info.experts_container is not None:
                                ec    = info.experts_container
                                gu    = ec.gate_up_proj.data  # [n_exp, 2*i, h]
                                dp    = ec.down_proj.data     # [n_exp, h, i]
                                inter = gu.shape[1] // 2
                                if _old_inter_summary == 0:
                                    _old_inter_summary = inter
                                    _new_inter_summary = inter  # unchanged
                                for ei, prune_list in exp_dict.items():
                                    _prune_idx_e = torch.tensor(
                                        sorted(prune_list), dtype=torch.long
                                    )
                                    _keep_mask_e = torch.ones(inter, dtype=torch.bool)
                                    _keep_mask_e[_prune_idx_e] = False
                                    _keep_idx_e = _keep_mask_e.nonzero(as_tuple=True)[0]

                                    # ── residual_mask_moe: compensate before zeroing ──
                                    if method == "residual_mask_moe" and _calib_ref is not None:
                                        _X_e = _calib_ref.get((li, ei))
                                        if _X_e is not None and _X_e.shape[0] >= min_resid_tok:
                                            import torch.nn.functional as _F
                                            try:
                                                with torch.no_grad():
                                                    _X_f  = _X_e.detach().float()
                                                    _gw_e = gu[ei, :inter, :].detach().float()
                                                    _uw_e = gu[ei, inter:, :].detach().float()
                                                    _dw_e = dp[ei].detach().float()
                                                    if resid_on_cpu:
                                                        _X_f  = _X_f.cpu()
                                                        _gw_e = _gw_e.cpu()
                                                        _uw_e = _uw_e.cpu()
                                                        _dw_e = _dw_e.cpu()
                                                    _dev = _X_f.device
                                                    _act = _F.silu(_X_f @ _gw_e.T) * (_X_f @ _uw_e.T)
                                                    _AP  = _act[:, _prune_idx_e.to(_dev)]
                                                    _AK  = _act[:, _keep_idx_e.to(_dev)]
                                                    _WP  = _dw_e[:, _prune_idx_e.to(_dw_e.device)]
                                                    _E   = _AP @ _WP.T
                                                    _AAt = _AK @ _AK.T
                                                    _N   = _X_f.shape[0]
                                                    _lam = resid_lambda * float(_AAt.diagonal().mean())
                                                    _reg = _lam * torch.eye(_N, dtype=torch.float32, device=_AAt.device)
                                                    _B   = torch.linalg.solve(_AAt + _reg, _E)
                                                    _D   = _AK.T @ _B
                                                    _WK  = _dw_e[:, _keep_idx_e.to(_dw_e.device)]
                                                    _WKn = _WK + resid_tau * _D.T
                                                    dp[ei, :, _keep_idx_e.to(dp.device)] = _WKn.to(
                                                        device=dp.device, dtype=dp.dtype
                                                    )
                                                _resid_stable += 1
                                            except Exception as _re:
                                                logger.warning(
                                                    "residual_mask_moe layer=%d ei=%d: %s",
                                                    li, ei, _re,
                                                )
                                                _resid_failed += 1
                                        else:
                                            _resid_skipped += 1

                                    for ch in prune_list:
                                        gu[ei, ch, :].zero_()
                                        gu[ei, ch + inter, :].zero_()
                                        dp[ei, :, ch].zero_()
                                        total_masked_en += 1
                                    experts_pruned += 1
                            else:
                                for ei, prune_list in exp_dict.items():
                                    expert   = info.expert_modules[ei]
                                    d_ff_exp = expert_sizes.get((li, ei), 0)
                                    if _old_inter_summary == 0:
                                        _old_inter_summary = d_ff_exp
                                        _new_inter_summary = d_ff_exp
                                    for ch in prune_list:
                                        if hasattr(expert, "gate_proj"):
                                            expert.gate_proj.weight.data[ch, :].zero_()
                                        if hasattr(expert, "up_proj"):
                                            expert.up_proj.weight.data[ch, :].zero_()
                                        if hasattr(expert, "down_proj"):
                                            expert.down_proj.weight.data[:, ch].zero_()
                                        total_masked_en += 1
                                    experts_pruned += 1

                            n_masked_this_layer = sum(len(v) for v in exp_dict.values())
                            print(
                                f"      Layer {li}: masked {n_masked_this_layer} "
                                f"expert-channel units across "
                                f"{len(exp_dict)} experts  "
                                f"[no shape change]"
                            )
                            rows.append({
                                "model": model_name,
                                "layer_index": li, "expert_index": sorted(exp_dict.keys()),
                                "method": method, "dtype": dtype_str,
                                "n_masked_units": n_masked_this_layer,
                            })

                        # Update actual_pruned to reflect mask-only count
                        actual_pruned = total_masked_en
                        actual_pct    = (
                            100.0 * actual_pruned / total_expert_neurons
                            if total_expert_neurons else 0.0
                        )

                    else:
                        # ── packed_same_channel: physical in-place pruning ─────
                        for (li, ei), prune_list in per_expert_pruned.items():
                            if not prune_list:
                                continue

                            info      = layer_idx_to_info[li]
                            d_ff_orig = expert_sizes[(li, ei)]
                            prune_idx = torch.tensor(sorted(prune_list), dtype=torch.long)

                            if ei == -1:
                                # Packed layer: global same-channel pruning
                                ec              = info.experts_container
                                n_pruned_actual = len(prune_idx)
                                removed_en      = n_pruned_actual * info.num_experts

                                # keep_idx in old d_ff space (needed for residual)
                                _km = torch.ones(d_ff_orig, dtype=torch.bool)
                                _km[prune_idx] = False
                                _keep_idx_packed = _km.nonzero(as_tuple=True)[0]

                                # ── residual_full_moe: compensate BEFORE pruning ──
                                if method == "residual_full_moe" and _calib_ref is not None:
                                    _t_rs = time.perf_counter()
                                    _layer_acts = {
                                        _ei2: _calib_ref.get((li, _ei2))
                                        for _ei2 in range(info.num_experts)
                                    }
                                    _rs = apply_packed_residual_for_layer(
                                        ec, prune_idx, _keep_idx_packed, _layer_acts,
                                        min_tokens=min_resid_tok,
                                        ridge_lambda=resid_lambda,
                                        tau=resid_tau,
                                        solve_on_cpu=resid_on_cpu,
                                    )
                                    t_recon_total += time.perf_counter() - _t_rs
                                    _resid_stable  += _rs["n_stable"]
                                    _resid_skipped += _rs["n_skipped"]
                                    _resid_failed  += _rs["n_failed"]
                                    print(
                                        f"      [residual_full_moe layer {li}] "
                                        f"stable={_rs['n_stable']}  "
                                        f"skip={_rs['n_skipped']}  "
                                        f"fail={_rs['n_failed']}  "
                                        f"mean_tok={_rs['mean_tokens']}"
                                    )

                                new_inter = prune_packed_experts_global(
                                    ec, prune_idx, alignment=chan_align
                                )
                                for pv in info.expert_modules:
                                    if isinstance(pv, PackedExpertView):
                                        pv.moe_intermediate = new_inter
                                experts_pruned += info.num_experts
                                _old_inter_summary = _old_inter_summary or d_ff_orig
                                _new_inter_summary = new_inter
                                rows.append({
                                    "model": model_name,
                                    "target_pruning_percent": target_pct,
                                    "layer_index": li, "expert_index": -1,
                                    "pruning_mode": pruning_mode,
                                    "physical_pruning": True,
                                    "speedup_expected": True,
                                    "same_channel_across_experts": True,
                                    "aggregation_mode": chan_agg,
                                    "selector": f"rmsnorm_bound_{chan_agg}",
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
                                # Unpacked layer: per-expert in-place
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
                                        "pruning_mode": pruning_mode,
                                        "physical_pruning": True,
                                        "speedup_expected": True,
                                        "same_channel_across_experts": False,
                                        "aggregation_mode": "N/A",
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
                                calib_inp = (
                                    _calib_ref.get((li, ei), None)
                                    if _calib_ref is not None else None
                                )

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
                                    "pruning_mode": pruning_mode,
                                    "physical_pruning": True,
                                    "speedup_expected": True,
                                    "same_channel_across_experts": False,
                                    "aggregation_mode": "N/A",
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
                    if pruning_mode == "packed_same_channel" and _has_packed:
                        _sel_str = f"{moe_selector}_{chan_agg}"
                    elif pruning_mode == "per_expert_mask":
                        _sel_str = f"{moe_selector}_per_expert"
                    else:
                        _sel_str = moe_selector
                    _phys       = (pruning_mode == "packed_same_channel")
                    _t_end      = time.perf_counter()
                    _gpu0_peak  = (
                        torch.cuda.max_memory_allocated(0) / 1024**2
                        if torch.cuda.is_available() and torch.cuda.device_count() > 0
                        else 0.0
                    )
                    _gpu1_peak  = (
                        torch.cuda.max_memory_allocated(1) / 1024**2
                        if torch.cuda.is_available() and torch.cuda.device_count() > 1
                        else 0.0
                    )

                    if not fp_ok:
                        print("    ERROR: forward pass failed — skipping PPL eval")
                        for _ds in EVAL_DATASETS:
                            err_row = {
                                "model": model_name,
                                "target_pruning_percent": target_pct,
                                "eval_dataset": _ds,
                                "pruning_mode":  pruning_mode,
                                "aggregation_mode": chan_agg,
                                "selector": _sel_str,
                                "method": method,
                                "physical_pruning": _phys,
                                "speedup_expected": _phys,
                                "same_channel_across_experts": (
                                    pruning_mode == "packed_same_channel"
                                ),
                                "smoke_layers_used": len(moe_layers),
                                "total_moe_layers": len(
                                    [i for i in layer_infos if i.is_moe]
                                ),
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
                                "old_intermediate": _old_inter_summary,
                                "new_intermediate": _new_inter_summary,
                                "moe_channel_alignment": chan_align,
                                "baseline_ppl":    round(baseline_ppl_per_ds[_ds], 4),
                                "compressed_ppl":  float("nan"),
                                "delta_ppl":       float("nan"),
                                "relative_delta_pct": float("nan"),
                                "damage_reduction_percent":       float("nan"),
                                "forward_check":   False,
                                "reconstruction_time_seconds": round(t_recon_total, 2),
                                "shape_changed": pruning_mode == "packed_same_channel",
                                "residual_stable_experts":  _resid_stable,
                                "residual_skipped_experts": _resid_skipped,
                                "residual_failed_experts":  _resid_failed,
                                "residual_time_sec": round(t_recon_total, 2),
                                "peak_gpu_memory_MB": round(peak_gpu_mb, 1),
                                "gpu0_peak_mb":  round(_gpu0_peak, 1),
                                "gpu1_peak_mb":  round(_gpu1_peak, 1),
                                "time_sec":      round(_t_end - _t_start, 1),
                                "dtype": dtype_str,
                                "notes": "forward_pass_failed",
                                "csv_path": main_csv_path,
                                "json_path": json_path,
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
                            "pruning_mode":  pruning_mode,
                            "aggregation_mode": chan_agg,
                            "selector": _sel_str,
                            "method": method,
                            "physical_pruning": _phys,
                            "speedup_expected": _phys,
                            "same_channel_across_experts": (
                                pruning_mode == "packed_same_channel"
                            ),
                            "smoke_layers_used": len(moe_layers),
                            "total_moe_layers": len(
                                [i for i in layer_infos if i.is_moe]
                            ),
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
                            "old_intermediate": _old_inter_summary,
                            "new_intermediate": _new_inter_summary,
                            "moe_channel_alignment":    chan_align,
                            "baseline_ppl":             round(cur_bppl, 4),
                            "compressed_ppl":           round(ppl, 4),
                            "delta_ppl":                round(delta, 4),
                            "relative_delta_pct": round(rel, 4),
                            "damage_reduction_percent": float("nan"),
                            "forward_check":            True,
                            "reconstruction_time_seconds": round(t_recon_total, 2),
                            "shape_changed": pruning_mode == "packed_same_channel",
                            "residual_stable_experts":  _resid_stable,
                            "residual_skipped_experts": _resid_skipped,
                            "residual_failed_experts":  _resid_failed,
                            "residual_time_sec": round(t_recon_total, 2),
                            "peak_gpu_memory_MB":        round(peak_gpu_mb, 1),
                            "gpu0_peak_mb":  round(_gpu0_peak, 1),
                            "gpu1_peak_mb":  round(_gpu1_peak, 1),
                            "time_sec":      round(_t_end - _t_start, 1),
                            "dtype": dtype_str,
                            "notes": "",
                            "csv_path": main_csv_path,
                            "json_path": json_path,
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

                if inplace_prune:
                    print(
                        "\n  NOTE: model has been modified in-place. "
                        "No additional target percentages will run from this "
                        "model state. Use separate processes for each target."
                    )
                    break  # exit per-target loop — model is consumed

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

    _total_moe = len([i for i in (layer_infos if "layer_infos" in dir() else [])
                      if getattr(i, "is_moe", False)])
    _print_moe_summary_table(
        all_results,
        n_smoke_layers=len(moe_layers) if "moe_layers" in dir() else 0,
        total_moe_layers=_total_moe,
        main_csv_path=main_csv_path,
        json_path=json_path,
    )
    print(f"MoE Summary CSV : {main_csv_path}")
    print(f"MoE JSON report : {json_path}\n")
