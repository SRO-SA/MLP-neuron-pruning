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
7. The historical rmsnorm_bound score remains unchanged for reproducibility.
   The optional rmsnorm_ellipsoid_bound score uses the containing decoder
   layer's post_attention_layernorm weight (experts share that normalized input).

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
    allocation_source: rmsnorm_bound       # optional plan-anchored interface
    ranking_source: rmsnorm_ellipsoid_bound
    moe_allocation_plan: results/.../plan.json
    moe_ranking_plan: null                 # required only for fixed_plan ranking
    moe_smoke_test: true   # if true: run only first 4 layers to verify correctness
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from .moe_residual_methods import (  # noqa: E402
    RESIDUAL_METHODS,
    BEST_RESIDUAL_LAM,
    _empty_residual_stats,
    make_pruning_plan_path,
    build_pruning_plan,
    save_pruning_plan,
    load_pruning_plan,
    apply_pruning_plan_to_selection,
    apply_residual_dispatch_unpacked,
)
from .rmsnorm_geometry import (
    compute_observed_channel_contribution_max_from_weights,
    compute_rmsnorm_bound_triplet_from_weights,
    compute_rmsnorm_ellipsoid_bound_from_weights,
    compute_rmsnorm_sphere_bound_from_weights,
)
from .moe_plan_replay import (
    SUPPORTED_ALLOCATION_RANKING_SOURCES,
    build_allocation_ranking_selection,
    build_fixed_allocation_selection,
    validate_fixed_allocation_source_plan_static,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_EXPERT_FRAC    = 0.20
DEFAULT_MIN_EXPERT_TOKENS  = 128
BEST_RESIDUAL_LAM          = 1e-2
BEST_RESIDUAL_TAU          = 1.0

SUPPORTED_MOE_SELECTORS = {
    "rmsnorm_bound",
    "rmsnorm_ellipsoid_bound",
    "down_norm",
    "random",
    "activation_score",
}

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
    # ── Identity ──────────────────────────────────────────────────────────────
    "model",
    "model_revision",
    "transformers_version",
    "torch_version",
    "expert_layout",
    # ── Experiment config ────────────────────────────────────────────────────
    "total_moe_layers",
    "processed_moe_layers",
    "target_pct",
    "actual_pct",
    "selected_layer_channels",
    "removed_expert_neurons",
    "selector",
    "aggregation_mode",
    "pruning_mode",
    "method",
    "requested_method",
    "actual_method",
    "residual_variant",
    "residual_lambda",
    "eval_dataset",
    "n_eval",
    "moe_calib_samples",
    # ── Pruning plan ──────────────────────────────────────────────────────────
    "pruning_plan_path",
    "loaded_pruning_plan",
    "fixed_allocation_replay",
    "experiment_label",
    "source_plan_path",
    "source_allocation_selector",
    "alternate_channel_selector",
    "replay_source_overlap_channels",
    "allocation_source",
    "ranking_source",
    "allocation_aggregation_mode",
    "ranking_aggregation_mode",
    "exact_total_layer_channels",
    "allocation_plan_path",
    "ranking_plan_path",
    "evaluation_max_seq_len",
    "evaluation_batch_size",
    "evaluation_num_texts",
    "evaluation_corpus_sha256",
    "evaluation_preprocessing",
    "evaluation_protocol_label",
    "seed",
    "baseline_eval_tokens",
    "pruned_eval_tokens",
    "evaluation_token_count_match",
    "fresh_model_process_required",
    "process_id",
    "model_load_instance_id",
    "score_comparison_csv_path",
    "score_comparison_json_path",
    "per_example_nll_path",
    "mean_nll_difference",
    "mean_nll_difference_ci95_lower",
    "mean_nll_difference_ci95_upper",
    "paired_bootstrap_resamples",
    "bound_tightness_json_path",
    "expert_bound_scores_npz_path",
    # ── Results ───────────────────────────────────────────────────────────────
    "baseline_ppl",
    "compressed_ppl",
    "delta_ppl",
    "relative_delta_pct",
    "forward_check",
    "status",
    # ── Parameter / FLOP accounting ───────────────────────────────────────────
    "expert_param_reduction_pct",
    "total_model_param_reduction_pct",
    "estimated_active_expert_flop_reduction_pct",
    # ── Residual audit (binary flags) ─────────────────────────────────────────
    "residual_applied",
    "residual_fallback_used",
    # ── Residual per-expert counts ────────────────────────────────────────────
    "residual_total_candidate_experts",
    "residual_attempted_experts",
    "residual_applied_experts",
    "residual_stable_experts",
    "residual_rejected_experts",
    "residual_skipped_experts",
    "residual_failed_experts",
    "residual_coverage_pct",
    # ── Residual skip reasons ─────────────────────────────────────────────────
    "residual_skip_too_few_tokens",
    "residual_skip_ill_conditioned",
    "residual_skip_non_finite",
    "residual_skip_update_too_large",
    "residual_skip_not_improved",
    # ── Residual quality metrics ──────────────────────────────────────────────
    "mean_err_delete",
    "mean_err_resid",
    "mean_local_improvement_pct",
    "mean_update_norm",
    "max_update_norm",
    # ── Timing / memory ──────────────────────────────────────────────────────
    "residual_time_sec",
    # ── Paths ────────────────────────────────────────────────────────────────
    "csv_path",
    "json_path",
    "per_layer_csv_path",
]


MOE_DRYRUN_CSV_KEYS = [
    "model", "target_pruning_percent", "pruning_mode",
    "aggregation_mode", "selector", "max_layer_frac",
    "moe_channel_alignment",
    "layer_idx", "num_experts", "old_intermediate",
    "selected_channels", "new_intermediate",
    "removed_expert_neurons", "layer_pruning_pct",
    "zero_routed_experts", "min_routed_tokens",
    "median_routed_tokens", "max_routed_tokens",
    "score_min", "score_median", "score_p95", "score_max",
    "actual_overall_pct", "target_pct", "timestamp",
]


MOE_PER_LAYER_CSV_KEYS = [
    "layer_idx",
    "allocation_source",
    "ranking_source",
    "num_experts",
    "old_intermediate",
    "new_intermediate",
    "pruned_channels",
    "removed_expert_neurons",
    "actual_layer_pruning_pct",
    "shape_changed",
    "zero_routed_experts",
    "min_routed_tokens",
    "median_routed_tokens",
    "max_routed_tokens",
    "score_min",
    "score_median",
    "score_p95",
    "score_max",
    "score_scale_abs_max",
    "allocation_reference_channel_ids",
    "selected_channel_ids",
    "ranking_overlap_count",
    "ranking_jaccard_with_allocation",
    "expert_params_before",
    "expert_params_after",
    "removed_expert_params",
    "expert_param_reduction_pct",
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
    _type_name = type(experts).__name__
    print(f"    [detect] experts type = {_type_name}")

    _has_gu = hasattr(experts, "gate_up_proj")
    _has_dp = hasattr(experts, "down_proj")
    print(f"    [detect] has gate_up_proj={_has_gu}  has down_proj={_has_dp}")

    if _has_gu and _has_dp:
        _gu_obj    = getattr(experts, "gate_up_proj", None)
        _is_tensor = isinstance(_gu_obj, torch.Tensor)
        _gu_type   = type(_gu_obj).__name__ if _gu_obj is not None else "None"
        print(f"    [detect] gate_up_proj type={_gu_type}  isinstance(Tensor)={_is_tensor}")
        if _is_tensor:
            gu = _gu_obj
            print(f"    [detect] gate_up_proj.shape={list(gu.shape)}  ndim={gu.ndim}")
            if gu.ndim == 3:
                print("    [detect] -> packed_gate_up")
                return "packed_gate_up"
            else:
                print(f"    [detect] ndim={gu.ndim} != 3, falling through")

    try:
        items = list(experts)
        if items and hasattr(items[0], "gate_proj"):
            print(f"    [detect] -> unpacked (iterable, {len(items)} experts)")
            return "unpacked"
    except (TypeError, RuntimeError) as _det_e:
        print(f"    [detect] not iterable: {_det_e}")

    print("    [detect] -> unknown layout")
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


def get_moe_input_rmsnorm_weight(layer_info: MoELayerInfo) -> torch.Tensor:
    """Return the RMSNorm scale immediately preceding a MoE/MLP block.

    Qwen3 experts do not contain individual RMSNorm modules.  Their shared
    input is the output of the containing decoder layer's
    ``post_attention_layernorm``.  A missing or incompatible weight is a hard
    error for the ellipsoid selector; silently reverting to the legacy score
    would make the requested experiment invalid.
    """
    layer = layer_info.layer_module
    rmsnorm = getattr(layer, "post_attention_layernorm", None)
    if rmsnorm is None:
        raise AssertionError(
            f"MoE layer {layer_info.layer_idx} has no post_attention_layernorm; "
            "rmsnorm_ellipsoid_bound requires the norm immediately before the MoE/MLP"
        )
    gamma = getattr(rmsnorm, "weight", None)
    if not isinstance(gamma, torch.Tensor):
        raise AssertionError(
            f"MoE layer {layer_info.layer_idx} post_attention_layernorm has no tensor weight"
        )
    if gamma.ndim != 1:
        raise AssertionError(
            f"MoE layer {layer_info.layer_idx} RMSNorm gamma must be rank 1; "
            f"got {tuple(gamma.shape)}"
        )
    return gamma


def compute_rmsnorm_ellipsoid_scores_for_expert(
    expert_module,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Compute the coordinate-wise RMSNorm ellipsoid bound for one expert."""
    w = get_expert_weights(expert_module)
    if tuple(gamma.shape) != (w["d_model"],):
        raise AssertionError(
            f"gamma shape {tuple(gamma.shape)} is incompatible with expert "
            f"d_model={w['d_model']}"
        )
    return compute_rmsnorm_ellipsoid_bound_from_weights(
        w["gate_proj"], w["up_proj"], w["down_proj"], gamma
    )


def compute_rmsnorm_sphere_scores_for_expert(
    expert_module,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Compute the valid gamma-infinity sphere bound for diagnostics/tests."""
    w = get_expert_weights(expert_module)
    if tuple(gamma.shape) != (w["d_model"],):
        raise AssertionError(
            f"gamma shape {tuple(gamma.shape)} is incompatible with expert "
            f"d_model={w['d_model']}"
        )
    return compute_rmsnorm_sphere_bound_from_weights(
        w["gate_proj"], w["up_proj"], w["down_proj"], gamma
    )


def compute_rmsnorm_bound_triplet_for_expert(
    expert_module,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute legacy/sphere/ellipsoid scores with one expert FP32 transfer."""
    w = get_expert_weights(expert_module)
    if tuple(gamma.shape) != (w["d_model"],):
        raise AssertionError(
            f"gamma shape {tuple(gamma.shape)} is incompatible with expert "
            f"d_model={w['d_model']}"
        )
    return compute_rmsnorm_bound_triplet_from_weights(
        w["gate_proj"], w["up_proj"], w["down_proj"], gamma
    )

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
    rmsnorm_gamma: "Optional[torch.Tensor]" = None,
) -> torch.Tensor:
    """
    Dispatcher: returns [d_ff] importance scores for one expert.

    selector options:
      "rmsnorm_bound"    – weight-only RMSNorm-bounded SwiGLU score (default)
      "rmsnorm_ellipsoid_bound" – exact coordinate-wise RMSNorm geometry bound
      "down_norm"        – L2 norm of each down_proj column (simple baseline)
      "random"           – uniform random scores (random baseline; caller sets seed)
      "activation_score" – activation × down-column-norm score (needs calib)
    """
    if selector == "activation_score":
        return compute_activation_scores_for_expert(expert_module, calib_inputs)
    elif selector == "rmsnorm_ellipsoid_bound":
        if rmsnorm_gamma is None:
            raise AssertionError(
                "rmsnorm_ellipsoid_bound requires the containing layer RMSNorm gamma"
            )
        return compute_rmsnorm_ellipsoid_scores_for_expert(
            expert_module, rmsnorm_gamma
        )
    elif selector == "down_norm":
        w = get_expert_weights(expert_module)
        # down_proj: [d_model, d_ff] — column norms give per-channel importance
        return w["down_proj"].detach().float().cpu().norm(dim=0)  # [d_ff]
    elif selector == "random":
        w = get_expert_weights(expert_module)
        # Uniform random — reproducibility controlled by torch.manual_seed at call site
        return torch.rand(w["d_ff"])
    elif selector == "rmsnorm_bound":
        # Historical selector kept byte-for-byte in get_expert_scores for
        # reproducibility of the trusted Qwen3-30B-A3B baseline.
        return get_expert_scores(expert_module)
    raise ValueError(
        f"Unknown MoE selector {selector!r}. "
        f"Valid: {', '.join(sorted(SUPPORTED_MOE_SELECTORS))}"
    )


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

    mean_toks   = round(sum(n_tokens_list) / len(n_tokens_list)) if n_tokens_list else 0
    median_toks = int(sorted(n_tokens_list)[len(n_tokens_list) // 2]) if n_tokens_list else 0
    min_toks    = min(n_tokens_list) if n_tokens_list else 0
    max_toks    = max(n_tokens_list) if n_tokens_list else 0
    return {
        "n_stable":      n_stable,
        "n_skipped":     n_skipped,
        "n_failed":      n_failed,
        "mean_tokens":   mean_toks,
        "median_tokens": median_toks,
        "min_tokens":    min_toks,
        "max_tokens":    max_toks,
    }




# ---------------------------------------------------------------------------
# Router-aware calibration
# ---------------------------------------------------------------------------

def apply_unpacked_residual_for_layer(
    expert_modules:               list,
    prune_idx:                    "torch.Tensor",
    keep_idx:                     "torch.Tensor",
    expert_activations_for_layer: "Dict[int, Optional[torch.Tensor]]",
    min_tokens:   int   = 16,
    ridge_lambda: float = 1e-2,
    tau:          float = 1.0,
    solve_on_cpu: bool  = True,
) -> "Dict":
    """
    Ridge-regression residual reconstruction for UNPACKED MoE experts (nn.ModuleList).

    MUST be called BEFORE prune_expert_channels because it updates down_proj
    in-place at the original (pre-pruning) intermediate size.

    For each expert e with nn.Linear gate_proj / up_proj / down_proj:

        gate_proj.weight : [d_ff, d_model]
        up_proj.weight   : [d_ff, d_model]
        down_proj.weight : [d_model, d_ff]   (column removal on prune)

    Computes:
        A  = SiLU(X_e @ gate^T) * (X_e @ up^T)   # [N, d_ff]  SwiGLU activations
        A_K = A[:, keep_idx]   A_P = A[:, prune_idx]
        W_P = down_proj.weight[:, prune_idx]        # [d_model, n_pruned]
        E   = A_P @ W_P^T                            # [N, d_model]  — contribution of pruned channels

    Dual-form ridge (cheap when N << d_ff):
        (A_K A_K^T + λI) B = E                       # [N, N] system
        ΔD = A_K^T B                                  # [n_kept, d_model]

    Update (in-place, before physical pruning):
        down_proj.weight[:, keep_idx] += τ * ΔD^T    # [d_model, n_kept]

    Returns stats dict (n_stable, n_skipped, n_failed, mean_tokens, ...).
    """
    import torch.nn.functional as _F

    n_stable = n_skipped = n_failed = 0
    n_tokens_list: "List[int]" = []

    for ei, exp in enumerate(expert_modules):
        X_raw = expert_activations_for_layer.get(ei, None)
        if X_raw is None or X_raw.shape[0] < min_tokens:
            n_skipped += 1
            continue

        N = X_raw.shape[0]
        try:
            with torch.no_grad():
                X      = X_raw.detach().float()
                gate_w = exp.gate_proj.weight.data.detach().float()   # [d_ff, d_model]
                up_w   = exp.up_proj.weight.data.detach().float()      # [d_ff, d_model]
                down_w = exp.down_proj.weight.data.detach().float()    # [d_model, d_ff]

                if solve_on_cpu:
                    X      = X.cpu()
                    gate_w = gate_w.cpu()
                    up_w   = up_w.cpu()
                    down_w = down_w.cpu()

                # SwiGLU activations [N, d_ff]
                act_all = _F.silu(X @ gate_w.T) * (X @ up_w.T)

                _prune = prune_idx.to(act_all.device)
                _keep  = keep_idx.to(act_all.device)

                A_P = act_all[:, _prune]                                  # [N, n_pruned]
                A_K = act_all[:, _keep]                                   # [N, n_kept]
                W_P = down_w[:, _prune.to(down_w.device)]                 # [d_model, n_pruned]

                # Residual target: contribution of pruned channels
                E   = A_P @ W_P.T                                          # [N, d_model]

                # Dual-form ridge solve
                AAt = A_K @ A_K.T                                          # [N, N]
                lam = ridge_lambda * float(AAt.diagonal().mean())
                reg = lam * torch.eye(N, dtype=torch.float32, device=AAt.device)

                B     = torch.linalg.solve(AAt + reg, E)                   # [N, d_model]
                Delta = A_K.T @ B                                           # [n_kept, d_model]

                W_K     = down_w[:, _keep.to(down_w.device)]               # [d_model, n_kept]
                W_K_new = W_K + tau * Delta.T                               # [d_model, n_kept]

                # Write back at original dtype and device
                _keep_dev = _keep.to(exp.down_proj.weight.device)
                exp.down_proj.weight.data[:, _keep_dev] = W_K_new.to(
                    device=exp.down_proj.weight.device,
                    dtype=exp.down_proj.weight.dtype,
                )

            n_stable += 1
            n_tokens_list.append(N)

        except Exception as exc:
            logger.warning(
                "apply_unpacked_residual: expert %d solve failed: %s", ei, exc
            )
            n_failed += 1

    mean_toks   = round(sum(n_tokens_list) / len(n_tokens_list)) if n_tokens_list else 0
    median_toks = int(sorted(n_tokens_list)[len(n_tokens_list) // 2]) if n_tokens_list else 0
    min_toks    = min(n_tokens_list) if n_tokens_list else 0
    max_toks    = max(n_tokens_list) if n_tokens_list else 0
    return {
        "n_stable":      n_stable,
        "n_skipped":     n_skipped,
        "n_failed":      n_failed,
        "mean_tokens":   mean_toks,
        "median_tokens": median_toks,
        "min_tokens":    min_toks,
        "max_tokens":    max_toks,
    }


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


_BOUND_COMPARE_BOTTOM_FRACTIONS = (0.01, 0.02, 0.04, 0.06, 0.08)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Small dependency-free equivalent of scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Return Spearman correlation, or None when either rank vector is constant."""
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    if len(ra) < 2 or float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _distribution_fields(prefix: str, values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _bound_comparison_metrics(
    legacy: np.ndarray,
    sphere: np.ndarray,
    ellipsoid: np.ndarray,
) -> Dict[str, object]:
    """Compute compact ranking/distribution diagnostics for one score scope."""
    legacy = np.asarray(legacy, dtype=np.float64)
    sphere = np.asarray(sphere, dtype=np.float64)
    ellipsoid = np.asarray(ellipsoid, dtype=np.float64)
    if not (legacy.shape == sphere.shape == ellipsoid.shape):
        raise AssertionError(
            f"comparison score shapes differ: {legacy.shape}, {sphere.shape}, "
            f"{ellipsoid.shape}"
        )
    n = len(legacy)
    metrics: Dict[str, object] = {"num_channels": int(n)}
    metrics.update(_distribution_fields("legacy", legacy))
    metrics.update(_distribution_fields("sphere", sphere))
    metrics.update(_distribution_fields("ellipsoid", ellipsoid))

    positive_legacy = legacy > 0.0
    legacy_ratio = ellipsoid[positive_legacy] / legacy[positive_legacy]
    if len(legacy_ratio):
        metrics.update(
            _distribution_fields("ellipsoid_to_legacy_ratio", legacy_ratio)
        )
    else:
        metrics.update({
            "ellipsoid_to_legacy_ratio_min": None,
            "ellipsoid_to_legacy_ratio_median": None,
            "ellipsoid_to_legacy_ratio_p95": None,
            "ellipsoid_to_legacy_ratio_max": None,
        })

    positive_sphere = sphere > 0.0
    ratio = ellipsoid[positive_sphere] / sphere[positive_sphere]
    if len(ratio):
        metrics.update(_distribution_fields("ellipsoid_to_sphere_ratio", ratio))
    else:
        metrics.update({
            "ellipsoid_to_sphere_ratio_min": None,
            "ellipsoid_to_sphere_ratio_median": None,
            "ellipsoid_to_sphere_ratio_p95": None,
            "ellipsoid_to_sphere_ratio_max": None,
        })
    tolerance = 1e-5 * np.maximum(1.0, np.abs(sphere))
    metrics["ellipsoid_gt_sphere_count"] = int(
        np.count_nonzero(ellipsoid > sphere + tolerance)
    )
    metrics["spearman_ellipsoid_vs_legacy"] = _spearman_correlation(
        ellipsoid, legacy
    )
    metrics["spearman_ellipsoid_vs_sphere"] = _spearman_correlation(
        ellipsoid, sphere
    )

    legacy_ranks = _rankdata_average(legacy)
    sphere_ranks = _rankdata_average(sphere)
    ellipsoid_ranks = _rankdata_average(ellipsoid)
    metrics["channels_changed_rank_vs_legacy"] = int(
        np.count_nonzero(legacy_ranks != ellipsoid_ranks)
    )
    metrics["channels_changed_rank_vs_sphere"] = int(
        np.count_nonzero(sphere_ranks != ellipsoid_ranks)
    )

    for fraction in _BOUND_COMPARE_BOTTOM_FRACTIONS:
        label = f"bottom_{int(round(100 * fraction))}pct"
        k = max(1, int(np.ceil(fraction * n)))
        ellipsoid_bottom = set(np.argsort(ellipsoid, kind="mergesort")[:k].tolist())
        legacy_bottom = set(np.argsort(legacy, kind="mergesort")[:k].tolist())
        sphere_bottom = set(np.argsort(sphere, kind="mergesort")[:k].tolist())
        metrics[f"{label}_count"] = k
        metrics[f"{label}_overlap_vs_legacy"] = (
            len(ellipsoid_bottom & legacy_bottom) / k
        )
        metrics[f"{label}_overlap_vs_sphere"] = (
            len(ellipsoid_bottom & sphere_bottom) / k
        )
    return metrics


def _bottom_selection_details(
    identities: List[object],
    legacy: np.ndarray,
    sphere: np.ndarray,
    ellipsoid: np.ndarray,
) -> Dict[str, object]:
    """Return compact selected-channel identities for requested bottom fractions."""
    result: Dict[str, object] = {}
    for fraction in _BOUND_COMPARE_BOTTOM_FRACTIONS:
        label = f"bottom_{int(round(100 * fraction))}pct"
        k = max(1, int(np.ceil(fraction * len(identities))))
        entry: Dict[str, object] = {"count": k}
        for name, values in (
            ("legacy", legacy),
            ("sphere", sphere),
            ("ellipsoid", ellipsoid),
        ):
            idx = np.argsort(values, kind="mergesort")[:k]
            entry[name] = [identities[int(i)] for i in idx]
        result[label] = entry
    return result


def _cross_layer_bound_scale_summary(
    json_layers: List[Dict[str, object]],
) -> Dict[str, object]:
    """Summarize how bound magnitudes vary from layer to layer."""
    legacy_medians = np.asarray([
        float(row["metrics"]["legacy_median"]) for row in json_layers
    ], dtype=np.float64)
    ellipsoid_medians = np.asarray([
        float(row["metrics"]["ellipsoid_median"]) for row in json_layers
    ], dtype=np.float64)

    def _layer_spread(prefix: str, values: np.ndarray) -> Dict[str, object]:
        positive = values[values > 0.0]
        result: Dict[str, object] = _distribution_fields(prefix, values)
        result[f"{prefix}_cross_layer_max_to_min_positive"] = (
            float(positive.max() / positive.min()) if len(positive) else None
        )
        mean = float(values.mean())
        result[f"{prefix}_cross_layer_cv"] = (
            float(values.std() / mean) if mean != 0.0 else None
        )
        return result

    summary: Dict[str, object] = {}
    summary.update(_layer_spread("legacy_layer_median", legacy_medians))
    summary.update(_layer_spread("ellipsoid_layer_median", ellipsoid_medians))
    positive = legacy_medians > 0.0
    layer_ratios = ellipsoid_medians[positive] / legacy_medians[positive]
    if len(layer_ratios):
        summary.update(
            _distribution_fields(
                "ellipsoid_to_legacy_layer_median_ratio", layer_ratios
            )
        )
    summary["interpretation"] = (
        "These are raw bound magnitudes across layers. A certified local bound "
        "does not by itself establish cross-layer comparability."
    )
    return summary


def collect_moe_bound_comparison_scores(
    moe_layers: List[MoELayerInfo],
    aggregation: str = "p95",
) -> List[Dict[str, object]]:
    """Compute legacy, valid-sphere, and ellipsoid layer scores.

    Expert weights are converted to FP32 one expert at a time.  Only the small
    per-channel score vectors are retained for layer aggregation.
    """
    records: List[Dict[str, object]] = []
    for info in moe_layers:
        gamma = get_moe_input_rmsnorm_weight(info)
        legacy_scores: List[torch.Tensor] = []
        sphere_scores: List[torch.Tensor] = []
        ellipsoid_scores: List[torch.Tensor] = []
        for expert in info.expert_modules:
            legacy, sphere, ellipsoid = compute_rmsnorm_bound_triplet_for_expert(
                expert, gamma
            )
            legacy_scores.append(legacy.float())
            sphere_scores.append(sphere.float())
            ellipsoid_scores.append(ellipsoid.float())
        if not legacy_scores:
            raise AssertionError(f"MoE layer {info.layer_idx} contains no experts")
        legacy = _aggregate_expert_scores(torch.stack(legacy_scores), aggregation)
        sphere = _aggregate_expert_scores(torch.stack(sphere_scores), aggregation)
        ellipsoid = _aggregate_expert_scores(
            torch.stack(ellipsoid_scores), aggregation
        )
        records.append({
            "layer_idx": info.layer_idx,
            "num_experts": info.num_experts,
            "d_ff": int(len(ellipsoid)),
            "legacy": legacy.detach().float().cpu(),
            "sphere": sphere.detach().float().cpu(),
            "ellipsoid": ellipsoid.detach().float().cpu(),
        })
    return records


def save_moe_bound_comparison_diagnostics(
    records: List[Dict[str, object]],
    output_dir: str,
    timestamp: str,
    model_name: str,
    aggregation: str,
) -> Tuple[str, str]:
    """Save compact layer/global bound diagnostics to CSV and JSON."""
    if not records:
        raise AssertionError("No bound-comparison score records were provided")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"moe_bound_comparison_{timestamp}.csv")
    json_path = os.path.join(output_dir, f"moe_bound_comparison_{timestamp}.json")

    csv_rows: List[Dict[str, object]] = []
    json_layers: List[Dict[str, object]] = []
    global_legacy: List[np.ndarray] = []
    global_sphere: List[np.ndarray] = []
    global_ellipsoid: List[np.ndarray] = []
    global_identities: List[object] = []

    for record in records:
        li = int(record["layer_idx"])
        legacy = record["legacy"].numpy()
        sphere = record["sphere"].numpy()
        ellipsoid = record["ellipsoid"].numpy()
        identities: List[object] = list(range(len(legacy)))
        metrics = _bound_comparison_metrics(legacy, sphere, ellipsoid)
        csv_rows.append({
            "scope": "layer",
            "layer_idx": li,
            "num_experts": int(record["num_experts"]),
            "d_ff": int(record["d_ff"]),
            **metrics,
        })
        json_layers.append({
            "layer_idx": li,
            "num_experts": int(record["num_experts"]),
            "d_ff": int(record["d_ff"]),
            "metrics": metrics,
            "bottom_channels": _bottom_selection_details(
                identities, legacy, sphere, ellipsoid
            ),
        })
        global_legacy.append(legacy)
        global_sphere.append(sphere)
        global_ellipsoid.append(ellipsoid)
        global_identities.extend([[li, ch] for ch in range(len(legacy))])

    legacy_all = np.concatenate(global_legacy)
    sphere_all = np.concatenate(global_sphere)
    ellipsoid_all = np.concatenate(global_ellipsoid)
    global_metrics = _bound_comparison_metrics(
        legacy_all, sphere_all, ellipsoid_all
    )
    csv_rows.append({
        "scope": "global",
        "layer_idx": "ALL",
        "num_experts": sum(int(r["num_experts"]) for r in records),
        "d_ff": sum(int(r["d_ff"]) for r in records),
        **global_metrics,
    })

    fieldnames: List[str] = []
    for row in csv_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "timestamp": timestamp,
        "model": model_name,
        "aggregation": aggregation,
        "legacy_selector": "rmsnorm_bound",
        "valid_sphere_bound": "gamma_infinity_sphere",
        "new_selector": "rmsnorm_ellipsoid_bound",
        "global": {
            "metrics": global_metrics,
            "bottom_channels": _bottom_selection_details(
                global_identities, legacy_all, sphere_all, ellipsoid_all
            ),
        },
        "cross_layer_scale": _cross_layer_bound_scale_summary(json_layers),
        "layers": json_layers,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return csv_path, json_path


def _tightness_distribution(values: np.ndarray) -> Dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def save_moe_bound_tightness_diagnostics(
    moe_layers: List[MoELayerInfo],
    expert_activations: Dict[Tuple[int, int], torch.Tensor],
    per_expert_pruned: Dict[Tuple[int, int], List[int]],
    *,
    output_dir: str,
    timestamp: str,
    model_name: str,
    aggregation: str,
    save_expert_scores: bool,
    max_inputs_per_expert: int = 8,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> Tuple[str, str]:
    """Measure sampled expert-channel tightness before physical pruning.

    One ratio is retained per expert-channel: the maximum observed routed-input
    contribution norm divided by that expert-specific bound.  The compressed
    NPZ keeps the pre-aggregation expert bounds and observations; the JSON is a
    compact statistical report.
    """
    os.makedirs(output_dir, exist_ok=True)
    if max_inputs_per_expert <= 0:
        raise ValueError("max_inputs_per_expert must be positive")
    json_path = os.path.join(output_dir, f"moe_bound_tightness_{timestamp}.json")
    npz_path = (
        os.path.join(output_dir, f"moe_expert_bounds_{timestamp}.npz")
        if save_expert_scores else ""
    )
    all_values: Dict[str, List[np.ndarray]] = {
        "ellipsoid_all": [], "ellipsoid_pruned": [], "ellipsoid_retained": [],
        "sphere_all": [], "sphere_pruned": [], "sphere_retained": [],
    }
    npz_arrays: Dict[str, np.ndarray] = {}
    layers_payload: List[Dict[str, object]] = []
    total_ellipsoid_violations = 0
    total_sphere_violations = 0
    sampled_experts = 0
    sampled_routed_inputs = 0

    for info in moe_layers:
        gamma = get_moe_input_rmsnorm_weight(info)
        selected = set(per_expert_pruned.get((info.layer_idx, -1), []))
        layer_ellipsoid: List[np.ndarray] = []
        layer_sphere: List[np.ndarray] = []
        layer_observed: List[np.ndarray] = []
        layer_ellipsoid_ratios: List[np.ndarray] = []
        layer_sphere_ratios: List[np.ndarray] = []
        layer_ellipsoid_violations = 0
        layer_sphere_violations = 0
        layer_input_count = 0
        layer_expert_indices: List[int] = []
        for expert_idx, expert in enumerate(info.expert_modules):
            routed = expert_activations.get((info.layer_idx, expert_idx))
            if routed is None or routed.shape[0] == 0:
                continue
            if routed.shape[0] > max_inputs_per_expert:
                sample_indices = torch.linspace(
                    0, routed.shape[0] - 1,
                    steps=max_inputs_per_expert,
                ).round().long()
                routed = routed.index_select(0, sample_indices)
            weights = get_expert_weights(expert)
            _, sphere, ellipsoid = compute_rmsnorm_bound_triplet_from_weights(
                weights["gate_proj"], weights["up_proj"],
                weights["down_proj"], gamma,
            )
            observed = compute_observed_channel_contribution_max_from_weights(
                weights["gate_proj"], weights["up_proj"],
                weights["down_proj"], routed,
            )
            observed_np = observed.numpy().astype(np.float32, copy=False)
            sphere_np = sphere.numpy().astype(np.float32, copy=False)
            ellipsoid_np = ellipsoid.numpy().astype(np.float32, copy=False)

            def _ratio(bound: np.ndarray) -> np.ndarray:
                ratio = np.zeros_like(observed_np, dtype=np.float32)
                positive = bound > 0.0
                ratio[positive] = observed_np[positive] / bound[positive]
                ratio[~positive & (observed_np > absolute_tolerance)] = np.inf
                return ratio

            ellipsoid_ratio = _ratio(ellipsoid_np)
            sphere_ratio = _ratio(sphere_np)
            ellipsoid_violation = observed_np > (
                ellipsoid_np * (1.0 + relative_tolerance) + absolute_tolerance
            )
            sphere_violation = observed_np > (
                sphere_np * (1.0 + relative_tolerance) + absolute_tolerance
            )
            layer_ellipsoid_violations += int(ellipsoid_violation.sum())
            layer_sphere_violations += int(sphere_violation.sum())
            layer_input_count += int(routed.shape[0])
            sampled_experts += 1
            sampled_routed_inputs += int(routed.shape[0])
            layer_expert_indices.append(expert_idx)
            layer_ellipsoid.append(ellipsoid_np)
            layer_sphere.append(sphere_np)
            layer_observed.append(observed_np)
            layer_ellipsoid_ratios.append(ellipsoid_ratio)
            layer_sphere_ratios.append(sphere_ratio)

        if not layer_ellipsoid:
            continue
        ellipsoid_stack = np.stack(layer_ellipsoid)
        sphere_stack = np.stack(layer_sphere)
        observed_stack = np.stack(layer_observed)
        ellipsoid_ratio_stack = np.stack(layer_ellipsoid_ratios)
        sphere_ratio_stack = np.stack(layer_sphere_ratios)
        pruned_mask = np.zeros(ellipsoid_stack.shape[1], dtype=bool)
        if selected:
            pruned_mask[list(selected)] = True
        for name, values in (
            ("ellipsoid", ellipsoid_ratio_stack),
            ("sphere", sphere_ratio_stack),
        ):
            all_values[f"{name}_all"].append(values.reshape(-1))
            all_values[f"{name}_pruned"].append(values[:, pruned_mask].reshape(-1))
            all_values[f"{name}_retained"].append(values[:, ~pruned_mask].reshape(-1))
        total_ellipsoid_violations += layer_ellipsoid_violations
        total_sphere_violations += layer_sphere_violations
        layer_key = f"layer_{info.layer_idx}"
        if save_expert_scores:
            npz_arrays[f"{layer_key}_ellipsoid_bound"] = ellipsoid_stack
            npz_arrays[f"{layer_key}_sphere_bound"] = sphere_stack
            npz_arrays[f"{layer_key}_observed_max"] = observed_stack
            npz_arrays[f"{layer_key}_pruned_mask"] = pruned_mask
            npz_arrays[f"{layer_key}_expert_indices"] = np.asarray(
                layer_expert_indices, dtype=np.int32
            )
        layers_payload.append({
            "layer_idx": info.layer_idx,
            "sampled_experts": int(ellipsoid_stack.shape[0]),
            "sampled_routed_inputs": layer_input_count,
            "pruned_channels": len(selected),
            "ellipsoid": _tightness_distribution(ellipsoid_ratio_stack),
            "ellipsoid_pruned": _tightness_distribution(
                ellipsoid_ratio_stack[:, pruned_mask]
            ),
            "ellipsoid_retained": _tightness_distribution(
                ellipsoid_ratio_stack[:, ~pruned_mask]
            ),
            "sphere": _tightness_distribution(sphere_ratio_stack),
            "sphere_pruned": _tightness_distribution(
                sphere_ratio_stack[:, pruned_mask]
            ),
            "sphere_retained": _tightness_distribution(
                sphere_ratio_stack[:, ~pruned_mask]
            ),
            "ellipsoid_numerical_violations": layer_ellipsoid_violations,
            "sphere_numerical_violations": layer_sphere_violations,
        })

    if not layers_payload:
        raise AssertionError("no routed expert inputs were available for tightness")
    if save_expert_scores:
        np.savez_compressed(npz_path, **npz_arrays)
    global_payload: Dict[str, object] = {}
    for name, chunks in all_values.items():
        nonempty = [chunk for chunk in chunks if len(chunk)]
        global_payload[name] = _tightness_distribution(
            np.concatenate(nonempty) if nonempty else np.asarray([])
        )
    global_payload.update({
        "ellipsoid_numerical_violations": total_ellipsoid_violations,
        "sphere_numerical_violations": total_sphere_violations,
    })
    payload = {
        "timestamp": timestamp,
        "model": model_name,
        "observed_ratio_definition": (
            "per expert-channel max over sampled routed inputs of "
            "||SiLU(r.g_i)(r.u_i)d_i||_2 divided by the expert-specific bound"
        ),
        "expert_aggregation_for_ranking": aggregation,
        "certificate_scope": (
            "Expert-specific sphere and ellipsoid scores are valid sampled-input "
            "upper bounds. p95 aggregation is a channel-selection heuristic, not "
            "a worst-case certificate across every expert; max is the conservative "
            "all-expert aggregation."
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "sampled_experts": sampled_experts,
        "sampled_routed_inputs": sampled_routed_inputs,
        "max_inputs_per_expert": max_inputs_per_expert,
        "input_sampling": (
            "deterministic evenly spaced routed inputs per expert, capped at "
            f"{max_inputs_per_expert}"
        ),
        "expert_scores_npz_path": npz_path,
        "global": global_payload,
        "layers": layers_payload,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path, npz_path


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
            # Unpacked: check for same-channel mode ((li, -1) keys) first.
            key_sc = (_li, -1)
            if key_sc in per_expert_pruned:
                # same_channel_all_experts: one summary line per layer
                plist = per_expert_pruned[key_sc]
                old_i = expert_sizes.get(key_sc, 0)
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
                # per-expert unpacked: one line per pruned expert
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
        tgt    = r.get("target_pct", r.get("requested_target_pct", r.get("target_pruning_percent", 0.0)))
        act    = r.get("actual_pct", r.get("actual_pruning_percent", 0.0))
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
        st_s   = str(r.get("status", r.get("notes", "")) or "ok")[:16]
        ep_red = r.get("expert_param_reduction_pct", float("nan"))
        tm_red = r.get("total_model_param_reduction_pct", float("nan"))
        fl_red = r.get("estimated_active_expert_flop_reduction_pct", float("nan"))
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
    # Post-table: param/FLOP reduction from last result row
    _last = rows[-1] if rows else {}
    _ep = _last.get("expert_param_reduction_pct")
    _tm = _last.get("total_model_param_reduction_pct")
    _fl = _last.get("estimated_active_expert_flop_reduction_pct")
    if _ep is not None:
        try:
            print(
                f"  Expert param red.: {float(_ep):.3f}%")
        except (TypeError, ValueError):
            pass
    if _tm is not None and _fl is not None:
        try:
            print(
                f"  Total model red. : {float(_tm):.3f}%   "
                f"Active FLOP red. : {float(_fl):.3f}%")
        except (TypeError, ValueError):
            pass
    print(f"  CSV : {main_csv_path}")
    print(f"  JSON: {json_path}")
    print(f"{'=' * W}\n")


def _print_dryrun_table(
    per_expert_pruned, expert_sizes, moe_layers,
    all_expert_scores, routed_counts,
    total_expert_neurons, chan_align,
    pruning_mode, aggregation_mode, selector,
    target_pct, actual_pruned, actual_pct,
    max_layer_frac, model_name, ts,
):
    """Enhanced per-layer table with score quantiles for dry-run analysis."""
    li_to_info = {i.layer_idx: i for i in moe_layers}

    # Build per-layer aggregated score tensor
    layer_score_agg = {}
    for _li, _ei, _s in all_expert_scores:
        sf = _s.float().cpu()
        if _ei == -1:
            layer_score_agg[_li] = sf
        else:
            if _li not in layer_score_agg:
                layer_score_agg[_li] = sf
            else:
                layer_score_agg[_li] = torch.stack(
                    [layer_score_agg[_li], sf]
                ).max(dim=0).values

    pruned_layers = sorted(
        set(_li for (_li, _ei), plist in per_expert_pruned.items() if plist)
    )

    SEP = "─"
    print()
    print("  DRY-RUN SELECTION TABLE")
    hdr = (
        f"  {'layer':>5}  {'n_exp':>5}  {'old_i':>5}  "
        f"{'sel_ch':>6}  {'new_i':>5}  {'rem_en':>8}  {'lyr%':>5}  "
        f"{'0rt':>4}  {'min_rt':>6}  {'med_rt':>6}  {'max_rt':>6}  "
        f"{'s_min':>10}  {'s_med':>10}  {'s_p95':>10}  {'s_max':>10}"
    )
    print(hdr)
    print("  " + SEP * (len(hdr) - 2))

    rows = []
    for _li in pruned_layers:
        info  = li_to_info[_li]
        n_exp = info.num_experts
        cnts  = [routed_counts.get((_li, ej), 0) for ej in range(n_exp)]
        z     = sum(1 for c in cnts if c == 0)
        nz    = sorted(c for c in cnts if c > 0) or [0]
        mn_rt, mx_rt, med_rt = nz[0], nz[-1], nz[len(nz) // 2]
        s_vec = layer_score_agg.get(_li, torch.tensor([0.0]))
        s_min = float(s_vec.min())
        s_max = float(s_vec.max())
        s_med = float(s_vec.median())
        s_p95 = float(torch.quantile(s_vec, 0.95))
        if info.experts_packed and pruning_mode == "packed_same_channel":
            key    = (_li, -1)
            plist  = per_expert_pruned.get(key, [])
            old_i  = expert_sizes.get(key, 0)
            sel_ch = len(plist)
            new_i  = old_i - sel_ch
            rem_en = sel_ch * n_exp
        else:
            sel_ch = sum(len(per_expert_pruned.get((_li, _ei), [])) for _ei in range(n_exp))
            old_i  = expert_sizes.get((_li, 0), 0)
            new_i  = old_i if pruning_mode == "per_expert_mask" else old_i - sel_ch
            rem_en = sel_ch
        lyr_pct = 100.0 * sel_ch / old_i if old_i else 0.0
        try:
            print(
                f"  {_li:>5}  {n_exp:>5}  {old_i:>5}  "
                f"{sel_ch:>6}  {new_i:>5}  {rem_en:>8,}  {lyr_pct:>4.1f}%  "
                f"{z:>4}  {mn_rt:>6}  {med_rt:>6}  {mx_rt:>6}  "
                f"{s_min:>10.4f}  {s_med:>10.4f}  {s_p95:>10.4f}  {s_max:>10.4f}"
            )
        except Exception:
            print(f"  layer={_li}  (format error)")
        rows.append({
            "model": model_name, "target_pruning_percent": target_pct,
            "pruning_mode": pruning_mode, "aggregation_mode": aggregation_mode,
            "selector": selector, "max_layer_frac": max_layer_frac,
            "moe_channel_alignment": chan_align,
            "layer_idx": _li, "num_experts": n_exp, "old_intermediate": old_i,
            "selected_channels": sel_ch, "new_intermediate": new_i,
            "removed_expert_neurons": rem_en,
            "layer_pruning_pct": round(lyr_pct, 3),
            "zero_routed_experts": z,
            "min_routed_tokens": mn_rt, "median_routed_tokens": med_rt,
            "max_routed_tokens": mx_rt,
            "score_min": round(s_min, 6), "score_median": round(s_med, 6),
            "score_p95": round(s_p95, 6), "score_max": round(s_max, 6),
            "actual_overall_pct": round(actual_pct, 4),
            "target_pct": target_pct, "timestamp": ts,
        })
    print("  " + SEP * (len(hdr) - 2))
    print(f"  Layers selected: {len(pruned_layers)}  actual_pct={actual_pct:.3f}%")
    print()
    return rows


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



# ---------------------------------------------------------------------------
# Parameter-count extraction helper
# ---------------------------------------------------------------------------
# count_parameters(model) returns {"total": int, "mlp": int}.
# Older call-sites in moe_pruning.py treated the result as a scalar, which
# causes "TypeError: '>' not supported between instances of 'dict' and 'int'".
# _extract_total_param_count normalises any return value to a plain int.

def _extract_total_param_count(x) -> int:
    """Return the total parameter count as a plain int regardless of input type.

    Handles:
      int / float  → cast to int directly
      dict         → look for 'total', 'total_params', 'num_params', 'params' keys
    """
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, dict):
        for key in ("total", "total_params", "num_params", "params"):
            if key in x:
                return int(x[key])
        raise ValueError(
            f"_extract_total_param_count: dict has no recognised key. "
            f"Available keys: {list(x.keys())}"
        )
    raise TypeError(
        f"_extract_total_param_count: unsupported type {type(x).__name__!r}. "
        f"Value: {x!r}"
    )


# Expert parameter / FLOP helpers  (called before and after physical pruning)
# ---------------------------------------------------------------------------

def _count_moe_expert_params(moe_layers_list) -> int:
    """Count total MLP parameters across all MoE experts (gate + up + down)."""
    total = 0
    for info in moe_layers_list:
        if info.experts_packed and info.experts_container is not None:
            ec = info.experts_container
            total += ec.gate_up_proj.numel() + ec.down_proj.numel()
        else:
            for exp in info.expert_modules:
                w = get_expert_weights(exp)
                for k in ("gate_proj", "up_proj", "down_proj"):
                    t = w.get(k)
                    if t is not None:
                        total += t.numel()
    return total


def _estimate_moe_active_flops(moe_layers_list) -> float:
    """
    Approximate active expert FLOPs per token (proportional; for relative comparison).

    Each active expert per token: gate_proj + up_proj + down_proj
    FLOPs ≈ top_k × 6 × intermediate × hidden  (2 MACs per element × 3 projections)
    """
    total = 0.0
    for info in moe_layers_list:
        top_k = info.top_k if (info.top_k and info.top_k > 0) else 2
        if info.experts_packed and info.experts_container is not None:
            ec = info.experts_container
            _, two_inter, hidden = ec.gate_up_proj.shape
            intermediate = two_inter // 2
            total += top_k * 6.0 * intermediate * hidden
        elif info.expert_modules:
            # Unpacked: compute once per layer (top_k already accounts for active experts)
            try:
                w0 = get_expert_weights(info.expert_modules[0])
                gw = w0.get("gate_proj")
                if gw is not None:
                    d_ff, d_model = gw.shape
                    total += top_k * 6.0 * d_ff * d_model
            except Exception:
                pass
    return total


def _build_per_layer_rows(
    per_expert_pruned:  Dict,
    expert_sizes:       Dict,
    moe_layers:         List,
    all_expert_scores:  List,
    routed_counts:      Dict,
    pruning_mode:       str,
) -> List[Dict]:
    """
    Build per-layer statistics for the per-layer CSV.
    Computed analytically from selection (before physical pruning).
    """
    score_lookup = {(_li, _ei): _s for _li, _ei, _s in all_expert_scores}
    rows: List[Dict] = []

    for info in moe_layers:
        li          = info.layer_idx
        num_experts = info.num_experts

        if pruning_mode == "packed_same_channel":
            key       = (li, -1)
            old_inter = expert_sizes.get(key, 0)
            prune_ch  = len(per_expert_pruned.get(key, []))
            new_inter = old_inter - prune_ch
            removed_en = prune_ch * num_experts
            layer_pct  = 100.0 * prune_ch / old_inter if old_inter > 0 else 0.0
            shp_changed = prune_ch > 0

            # Routing stats
            all_counts = [routed_counts.get((li, _ei), 0) for _ei in range(num_experts)]
            zero_rt    = sum(1 for c in all_counts if c == 0)
            min_rt     = min(all_counts) if all_counts else 0
            med_rt     = sorted(all_counts)[len(all_counts) // 2] if all_counts else 0
            max_rt     = max(all_counts) if all_counts else 0

            # Score stats (packed aggregated score)
            agg_s = score_lookup.get(key)
            if agg_s is not None:
                sv    = agg_s.float().cpu().numpy()
                s_min = float(sv.min())
                s_med = float(np.median(sv))
                s_p95 = float(np.percentile(sv, 95))
                s_max = float(sv.max())
            else:
                s_min = s_med = s_p95 = s_max = float("nan")

            # Expert params (analytical — avoids re-reading tensor after pruning)
            if info.experts_packed and info.experts_container is not None:
                ec     = info.experts_container
                hidden = ec.gate_up_proj.shape[2]
                ep_bef = num_experts * 3 * old_inter * hidden
                ep_aft = num_experts * 3 * new_inter * hidden
            elif old_inter > 0 and info.expert_modules:
                # Unpacked Layout A: derive hidden_dim from first expert
                try:
                    _w0    = get_expert_weights(info.expert_modules[0])
                    hidden = _w0["d_model"]
                    ep_bef = num_experts * 3 * old_inter * hidden
                    ep_aft = num_experts * 3 * new_inter * hidden
                except Exception:
                    ep_bef = ep_aft = 0
            else:
                ep_bef = ep_aft = 0

        else:
            # per_expert_mask or unpacked
            old_inter   = expert_sizes.get((li, 0), 0)
            prune_ch    = sum(len(per_expert_pruned.get((li, _ei), []))
                              for _ei in range(num_experts))
            new_inter   = old_inter
            removed_en  = prune_ch
            layer_pct   = (100.0 * prune_ch / (num_experts * old_inter)
                           if old_inter > 0 else 0.0)
            shp_changed = False
            all_counts  = [routed_counts.get((li, _ei), 0) for _ei in range(num_experts)]
            zero_rt     = sum(1 for c in all_counts if c == 0)
            min_rt      = min(all_counts) if all_counts else 0
            med_rt      = sorted(all_counts)[len(all_counts) // 2] if all_counts else 0
            max_rt      = max(all_counts) if all_counts else 0
            s_min = s_med = s_p95 = s_max = float("nan")
            ep_bef = ep_aft = 0

        removed_ep  = ep_bef - ep_aft
        ep_red      = 100.0 * removed_ep / ep_bef if ep_bef > 0 else 0.0

        def _fmt(v):
            return round(float(v), 6) if (v == v) else ""  # "" for NaN

        rows.append({
            "layer_idx":              li,
            "num_experts":            num_experts,
            "old_intermediate":       old_inter,
            "new_intermediate":       new_inter,
            "pruned_channels":        prune_ch,
            "removed_expert_neurons": removed_en,
            "actual_layer_pruning_pct": round(layer_pct, 4),
            "shape_changed":          shp_changed,
            "zero_routed_experts":    zero_rt,
            "min_routed_tokens":      min_rt,
            "median_routed_tokens":   med_rt,
            "max_routed_tokens":      max_rt,
            "score_min":    _fmt(s_min),
            "score_median": _fmt(s_med),
            "score_p95":    _fmt(s_p95),
            "score_max":    _fmt(s_max),
            "expert_params_before":   ep_bef,
            "expert_params_after":    ep_aft,
            "removed_expert_params":  removed_ep,
            "expert_param_reduction_pct": round(ep_red, 4),
        })

    return rows



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
    from .paired_bootstrap import (
        paired_bootstrap_nll_difference,
        write_paired_nll_csv,
    )
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
    dry_run        = bool(cfg.get("moe_selection_dry_run", False))
    score_comparison_only = bool(cfg.get("moe_score_comparison_only", False))
    max_layer_frac = float(cfg.get("moe_max_layer_channel_prune_frac", 1.0))
    # "uniform" = per-layer budget fixed from architecture+target_pct (selector-independent);
    # "global"  = global greedy across all layers (original behaviour, may vary per selector).
    moe_budget_mode = str(cfg.get("moe_budget_mode", "global"))
    resid_lambda   = float(cfg.get("residual_lambda", 1e-2))
    resid_tau      = float(cfg.get("residual_tau", 1.0))
    min_resid_tok  = int(cfg.get("min_residual_tokens_per_expert", 16))
    resid_on_cpu      = bool(cfg.get("solve_residual_on_cpu", True))
    moe_calib_samples = int(cfg.get("moe_calib_samples", n_eval))
    moe_calib_dataset = str(cfg.get("moe_calib_dataset", "wikitext2"))
    EVAL_DATASETS  = [str(d) for d in (
        eval_datasets_override or cfg.get("eval_datasets", ["wikitext2"]))]
    if moe_selector not in SUPPORTED_MOE_SELECTORS:
        raise ValueError(
            f"Unknown moe_selector {moe_selector!r}. "
            f"Valid: {', '.join(sorted(SUPPORTED_MOE_SELECTORS))}"
        )
    if score_comparison_only and chan_agg != "p95":
        raise ValueError(
            "moe_score_comparison_only currently requires "
            "moe_same_channel_aggregation: p95"
        )
    # ── Residual extended config ───────────────────────────────────────────────
    resid_imp_margin   = float(cfg.get("residual_improvement_margin", 1.0))
    resid_alpha_clip   = float(cfg.get("residual_alpha_clip", 2.0))
    resid_merge_metric = str(cfg.get("residual_merge_metric", "ls_scalar"))
    resid_norm_clip    = cfg.get("residual_update_norm_clip", None)
    resid_max_rel_norm = cfg.get("residual_max_relative_update_norm", None)
    # ── Pruning plan save/load ────────────────────────────────────────────────
    save_plan_cfg  = cfg.get("save_pruning_plan", False)
    load_plan_cfg  = cfg.get("load_pruning_plan", None)
    fixed_allocation_plan_cfg = cfg.get("moe_fixed_allocation_plan", None)
    fixed_allocation_selector_cfg = cfg.get(
        "moe_fixed_allocation_selector", None
    )
    allocation_source_cfg = cfg.get("allocation_source", None)
    ranking_source_cfg = cfg.get("ranking_source", None)
    allocation_plan_cfg = cfg.get("moe_allocation_plan", None)
    ranking_plan_cfg = cfg.get("moe_ranking_plan", None)
    allocation_ranking_name_cfg = str(
        cfg.get("allocation_ranking_experiment_name", "")
    )
    exact_total_layer_channels_cfg = cfg.get(
        "exact_total_layer_channels", None
    )
    if exact_total_layer_channels_cfg is not None:
        exact_total_layer_channels_cfg = int(exact_total_layer_channels_cfg)
        if exact_total_layer_channels_cfg <= 0:
            raise ValueError("exact_total_layer_channels must be positive")
        if exact_total_layer_channels_cfg % chan_align:
            raise ValueError(
                "exact_total_layer_channels must respect channel alignment: "
                f"{exact_total_layer_channels_cfg} % {chan_align} != 0"
            )
    collect_per_example_nll = bool(cfg.get("collect_per_example_nll", False))
    paired_bootstrap_resamples = int(cfg.get("paired_bootstrap_resamples", 10000))
    save_bound_tightness = bool(cfg.get("save_bound_tightness", False))
    save_expert_bound_scores = bool(cfg.get("save_expert_bound_scores", False))
    bound_tightness_max_inputs = int(
        cfg.get("bound_tightness_max_inputs_per_expert", 8)
    )
    if (save_bound_tightness or save_expert_bound_scores) and (
        moe_selector != "rmsnorm_ellipsoid_bound"
    ):
        raise ValueError(
            "bound tightness/expert-score capture requires "
            "moe_selector=rmsnorm_ellipsoid_bound"
        )
    allocation_ranking_enabled = bool(
        allocation_source_cfg or ranking_source_cfg or allocation_plan_cfg
        or ranking_plan_cfg
    )
    if allocation_ranking_enabled:
        if score_comparison_only:
            raise ValueError(
                "allocation/ranking experiments cannot use score-comparison-only mode"
            )
        if not save_plan_cfg:
            raise ValueError(
                "allocation/ranking experiments require save_pruning_plan: true"
            )
        if fixed_allocation_plan_cfg or fixed_allocation_selector_cfg:
            raise ValueError(
                "allocation_source/ranking_source cannot be combined with the "
                "legacy moe_fixed_allocation_* interface"
            )
        if not allocation_source_cfg or not ranking_source_cfg:
            raise ValueError(
                "allocation_source and ranking_source must be provided together"
            )
        if allocation_source_cfg not in SUPPORTED_ALLOCATION_RANKING_SOURCES:
            raise ValueError(
                f"unsupported allocation_source={allocation_source_cfg!r}"
            )
        if ranking_source_cfg not in SUPPORTED_ALLOCATION_RANKING_SOURCES:
            raise ValueError(f"unsupported ranking_source={ranking_source_cfg!r}")
        if not allocation_plan_cfg or not os.path.isfile(str(allocation_plan_cfg)):
            raise FileNotFoundError(
                f"allocation plan not found: {allocation_plan_cfg!r}"
            )
        if ranking_source_cfg != "fixed_plan" and moe_selector != ranking_source_cfg:
            raise ValueError(
                "moe_selector must equal ranking_source for selector-ranked "
                f"experiments: {moe_selector!r} != {ranking_source_cfg!r}"
            )
        if ranking_source_cfg == "fixed_plan":
            if not ranking_plan_cfg or not os.path.isfile(str(ranking_plan_cfg)):
                raise FileNotFoundError(
                    f"ranking_source='fixed_plan' plan not found: {ranking_plan_cfg!r}"
                )
            _ranking_plan = load_pruning_plan(str(ranking_plan_cfg))
        else:
            _ranking_plan = None
        if pruning_mode != "packed_same_channel" or moe_budget_mode != "global":
            raise ValueError(
                "allocation/ranking experiments require global "
                "packed_same_channel pruning"
            )
        if chan_agg not in {"p95", "max"}:
            raise ValueError(
                "allocation/ranking experiments require p95 or max aggregation"
            )
        if chan_agg == "max" and ranking_source_cfg != "rmsnorm_ellipsoid_bound":
            raise ValueError(
                "the bounded aggregation ablation permits max only for "
                "rmsnorm_ellipsoid_bound ranking"
            )
        if len(TARGET_PCTS) != 1:
            raise ValueError("allocation/ranking experiments require one target")
        _allocation_plan = load_pruning_plan(str(allocation_plan_cfg))
        if str(_allocation_plan.get("aggregation_mode", "")) != "p95":
            raise ValueError(
                "allocation plan must use the preserved p95 aggregation; got "
                f"{_allocation_plan.get('aggregation_mode')!r}"
            )
        _expected_allocation_selector = (
            str(_allocation_plan.get("selector", ""))
            if allocation_source_cfg == "fixed_plan"
            else str(allocation_source_cfg)
        )
        validate_fixed_allocation_source_plan_static(
            _allocation_plan,
            source_plan_path=str(allocation_plan_cfg),
            expected_source_selector=_expected_allocation_selector,
            alternate_selector=str(ranking_source_cfg),
            pruning_mode=pruning_mode,
            channel_alignment=chan_align,
            max_layer_frac=max_layer_frac,
            max_expert_frac=max_exp_frac,
            target_pct=float(TARGET_PCTS[0]),
            allow_same_selector=True,
        )
    else:
        _allocation_plan = None
        _ranking_plan = None
    if bool(fixed_allocation_plan_cfg) != bool(fixed_allocation_selector_cfg):
        raise ValueError(
            "moe_fixed_allocation_plan and moe_fixed_allocation_selector must "
            "be provided together"
        )
    if fixed_allocation_plan_cfg:
        if load_plan_cfg:
            raise ValueError(
                "fixed-allocation replay cannot be combined with load_pruning_plan"
            )
        if score_comparison_only:
            raise ValueError(
                "fixed-allocation replay cannot be combined with "
                "moe_score_comparison_only"
            )
        if not os.path.isfile(str(fixed_allocation_plan_cfg)):
            raise FileNotFoundError(
                "fixed-allocation source plan not found: "
                f"{fixed_allocation_plan_cfg!r}"
            )
        if pruning_mode != "packed_same_channel":
            raise ValueError(
                "fixed-allocation replay requires "
                "moe_pruning_mode: packed_same_channel"
            )
        if moe_budget_mode != "global":
            raise ValueError(
                "fixed-allocation replay requires moe_budget_mode: global; "
                "the per-layer counts are replayed from a global source plan"
            )
        if str(fixed_allocation_selector_cfg) == moe_selector:
            raise ValueError(
                "fixed-allocation source selector and alternate channel selector "
                f"must differ; both are {moe_selector!r}"
            )
        _fixed_source_plan = load_pruning_plan(str(fixed_allocation_plan_cfg))
        if len(TARGET_PCTS) != 1:
            raise ValueError(
                "fixed-allocation replay requires exactly one target; got "
                f"{TARGET_PCTS}"
            )
        validate_fixed_allocation_source_plan_static(
            _fixed_source_plan,
            source_plan_path=str(fixed_allocation_plan_cfg),
            expected_source_selector=str(fixed_allocation_selector_cfg),
            alternate_selector=moe_selector,
            pruning_mode=pruning_mode,
            channel_alignment=chan_align,
            max_layer_frac=max_layer_frac,
            max_expert_frac=max_exp_frac,
            target_pct=float(TARGET_PCTS[0]),
        )
    else:
        _fixed_source_plan = None
    # Unified residual config dict passed to apply_residual_dispatch_unpacked
    residual_cfg = {
        "residual_lambda":                    resid_lambda,
        "residual_tau":                       resid_tau,
        "residual_min_tokens":                min_resid_tok,
        "solve_residual_on_cpu":              resid_on_cpu,
        "residual_improvement_margin":        resid_imp_margin,
        "residual_alpha_clip":                resid_alpha_clip,
        "residual_merge_metric":              resid_merge_metric,
        "residual_update_norm_clip":          resid_norm_clip,
        "residual_max_relative_update_norm":  resid_max_rel_norm,
    }

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
    print(f"  moe_budget_mode: {moe_budget_mode}")
    if allocation_ranking_enabled:
        print(
            "  ALLOC/RANK     : "
            f"allocation_source={allocation_source_cfg}  "
            f"ranking_source={ranking_source_cfg}"
        )
        print(f"  ALLOCATION PLAN: {allocation_plan_cfg}")
    if fixed_allocation_plan_cfg:
        print(
            "  FIXED ALLOCATION: "
            f"{fixed_allocation_selector_cfg} allocation + "
            f"{moe_selector} channel ranking"
        )
        print(f"  SOURCE PLAN    : {fixed_allocation_plan_cfg}")
    if score_comparison_only:
        print("  SCORE COMPARE  : legacy vs sphere vs ellipsoid; no pruning, no PPL")
    if dry_run:
        print("  DRY-RUN MODE   : selection only — no pruning, no PPL eval")
    if max_layer_frac < 1.0:
        _cap_ex = (int(768 * max_layer_frac) // chan_align) * chan_align
        print(f"  max_layer_frac : {max_layer_frac:.0%} per layer"
              f" (cap ~{_cap_ex} ch for d_ff=768 align={chan_align})")
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

    # A weight-only comparison does not need evaluation text or PPL.
    if score_comparison_only:
        all_eval_corpora = {}
        eval_corpus_sha256 = {}
        print("Skipping evaluation dataset loading (score-comparison-only mode).")
    else:
        print(f"Loading evaluation datasets: {EVAL_DATASETS} ...")
        all_eval_corpora = load_all_eval_datasets(
            EVAL_DATASETS, max_samples=n_eval, use_fallback_corpus=use_fb,
        )
        eval_corpus_sha256 = {
            _dn: hashlib.sha256("\0".join(_txts).encode("utf-8")).hexdigest()
            for _dn, _txts in all_eval_corpora.items()
        }
        for _dn, _txts in all_eval_corpora.items():
            print(f"  {_dn}: {len(_txts)} samples")

    all_results: List[Dict] = []
    comparison_output_paths: List[Tuple[str, str]] = []

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
            _model_load_instance_id = (
                f"pid={os.getpid()};loaded_ns={time.time_ns()};model={model_name}"
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

            if score_comparison_only:
                print("\n  Computing weight-only RMSNorm bound comparison ...")
                _cmp_records = collect_moe_bound_comparison_scores(
                    moe_layers, aggregation=chan_agg
                )
                _cmp_csv, _cmp_json = save_moe_bound_comparison_diagnostics(
                    _cmp_records, output_dir, ts, model_name, chan_agg
                )
                comparison_output_paths.append((_cmp_csv, _cmp_json))
                _global_legacy = np.concatenate([
                    r["legacy"].numpy() for r in _cmp_records
                ])
                _global_sphere = np.concatenate([
                    r["sphere"].numpy() for r in _cmp_records
                ])
                _global_ellipsoid = np.concatenate([
                    r["ellipsoid"].numpy() for r in _cmp_records
                ])
                _gm = _bound_comparison_metrics(
                    _global_legacy, _global_sphere, _global_ellipsoid
                )
                print(
                    "  [score-compare] global Spearman ellipsoid vs legacy: "
                    f"{_gm['spearman_ellipsoid_vs_legacy']}"
                )
                print(
                    "  [score-compare] ellipsoid/sphere median ratio: "
                    f"{_gm['ellipsoid_to_sphere_ratio_median']}"
                )
                print(
                    "  [score-compare] ellipsoid > sphere violations: "
                    f"{_gm['ellipsoid_gt_sphere_count']}"
                )
                print(f"  [score-compare] CSV : {_cmp_csv}")
                print(f"  [score-compare] JSON: {_cmp_json}")
                continue

            # Per-dataset baselines (skipped for dry-run)
            if dry_run:
                baseline_ppl_per_ds = {_ds: float("nan") for _ds in EVAL_DATASETS}
                baseline_eval_info_per_ds = {
                    _ds: {"perplexity": float("nan"), "n_tokens": 0}
                    for _ds in EVAL_DATASETS
                }
                print("  [dry-run] Skipping baseline PPL evaluation.")
            else:
                baseline_ppl_per_ds = {}
                baseline_eval_info_per_ds = {}
                for _ds in EVAL_DATASETS:
                    _bp = evaluate_perplexity(
                        model, tokenizer, texts=all_eval_corpora[_ds],
                        max_seq_len=max_seq, batch_size=batch_sz, device=device,
                        collect_per_example=collect_per_example_nll,
                    )
                    baseline_ppl_per_ds[_ds] = _bp["perplexity"]
                    baseline_eval_info_per_ds[_ds] = dict(_bp)
                    print(f"  Baseline PPL ({_ds}): {_bp['perplexity']:.4f}")

            baseline_params_raw = count_parameters(model)
            if isinstance(baseline_params_raw, dict):
                logger.info(
                    "count_parameters returned dict: keys=%s  value=%s",
                    list(baseline_params_raw.keys()), baseline_params_raw,
                )
            else:
                logger.info(
                    "count_parameters returned %s: %s",
                    type(baseline_params_raw).__name__, baseline_params_raw,
                )
            baseline_params = baseline_params_raw   # kept for any legacy references
            _total_model_params_before = _extract_total_param_count(baseline_params_raw)
            _all_moe_for_count = [_i for _i in layer_infos if _i.is_moe]
            _expert_params_before = _count_moe_expert_params(_all_moe_for_count)
            _active_flops_before  = _estimate_moe_active_flops(_all_moe_for_count)
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
                _target_score_comparison_csv = ""
                _target_score_comparison_json = ""
                _target_bound_tightness_json = ""
                _target_expert_bound_scores_npz = ""
                print(f"\n  Target: {target_pct:.1f}%  →  {target_n:,} neurons")

                # ── Score all experts ────────────────────────────────────────
                # Packed layers: average across all experts → single (ei=-1) entry
                # (global same-channel pruning requires a single shared score).
                layer_idx_to_info: Dict[int, "MoELayerInfo"] = {
                    i.layer_idx: i for i in moe_layers
                }
                all_expert_scores: List[Tuple[int, int, torch.Tensor]] = []
                bound_comparison_records: List[Dict[str, object]] = []
                print("  Computing expert scores ...")
                for info in moe_layers:
                    _layer_gamma = (
                        get_moe_input_rmsnorm_weight(info)
                        if moe_selector == "rmsnorm_ellipsoid_bound"
                        else None
                    )
                    if pruning_mode == "packed_same_channel":
                        # Aggregate per-expert scores to a single layer-level score.
                        # Works for both packed (Layout B) and unpacked (Layout A):
                        # the same channel indices are removed from every expert.
                        # Selection then picks globally across layers using this score.
                        per_exp_s: List[torch.Tensor] = []
                        _legacy_per_exp: List[torch.Tensor] = []
                        _sphere_per_exp: List[torch.Tensor] = []
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
                                if moe_selector == "rmsnorm_ellipsoid_bound":
                                    (
                                        _legacy_score,
                                        _sphere_score,
                                        _selected_score,
                                    ) = compute_rmsnorm_bound_triplet_for_expert(
                                        exp, _layer_gamma
                                    )
                                else:
                                    _selected_score = _score_expert_moe(
                                        exp, moe_selector, _calib_ei, _layer_gamma
                                    ).float()
                                per_exp_s.append(_selected_score)
                                if moe_selector == "rmsnorm_ellipsoid_bound":
                                    _legacy_per_exp.append(_legacy_score)
                                    _sphere_per_exp.append(_sphere_score)
                                rt_weights.append(
                                    routed_counts.get((info.layer_idx, ei), 0)
                                    / _total_rt
                                )
                            except Exception as _se:
                                if moe_selector == "rmsnorm_ellipsoid_bound":
                                    raise
                                logger.warning(
                                    "score layer=%d ei=%d: %s",
                                    info.layer_idx, ei, _se,
                                )
                        if per_exp_s:
                            stacked_s = torch.stack(per_exp_s, dim=0)  # [n_exp, d_ff]
                            rt_t      = torch.tensor(rt_weights, dtype=torch.float32)
                            agg_s     = _aggregate_expert_scores(stacked_s, chan_agg, rt_t)
                            all_expert_scores.append((info.layer_idx, -1, agg_s))
                            if moe_selector == "rmsnorm_ellipsoid_bound":
                                _legacy_agg = _aggregate_expert_scores(
                                    torch.stack(_legacy_per_exp, dim=0), chan_agg, rt_t
                                )
                                _sphere_agg = _aggregate_expert_scores(
                                    torch.stack(_sphere_per_exp, dim=0), chan_agg, rt_t
                                )
                                bound_comparison_records.append({
                                    "layer_idx": info.layer_idx,
                                    "num_experts": info.num_experts,
                                    "d_ff": int(len(agg_s)),
                                    "legacy": _legacy_agg.detach().float().cpu(),
                                    "sphere": _sphere_agg.detach().float().cpu(),
                                    "ellipsoid": agg_s.detach().float().cpu(),
                                })
                    else:
                        # per_expert_mask mode: score each expert independently.
                        for ei, exp in enumerate(info.expert_modules):
                            try:
                                _calib_ei = (
                                    expert_activations.get((info.layer_idx, ei))
                                    if moe_selector == "activation_score"
                                    else None
                                )
                                s = _score_expert_moe(
                                    exp, moe_selector, _calib_ei, _layer_gamma
                                )
                                all_expert_scores.append((info.layer_idx, ei, s))
                            except Exception as _se:
                                if moe_selector == "rmsnorm_ellipsoid_bound":
                                    raise
                                logger.warning(
                                    "score layer=%d ei=%d: %s",
                                    info.layer_idx, ei, _se,
                                )

                if bound_comparison_records:
                    _cmp_csv, _cmp_json = save_moe_bound_comparison_diagnostics(
                        bound_comparison_records,
                        output_dir,
                        ts,
                        model_name,
                        chan_agg,
                    )
                    _target_score_comparison_csv = _cmp_csv
                    _target_score_comparison_json = _cmp_json
                    print(f"  [score-compare] CSV : {_cmp_csv}")
                    print(f"  [score-compare] JSON: {_cmp_json}")

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
                _fixed_replay_audit: Optional[Dict[str, object]] = None
                _allocation_ranking_audit: Optional[Dict[str, object]] = None

                if allocation_ranking_enabled:
                    per_expert_pruned, _allocation_ranking_audit = (
                        build_allocation_ranking_selection(
                            _allocation_plan,
                            allocation_plan_path=str(allocation_plan_cfg),
                            allocation_source=str(allocation_source_cfg),
                            ranking_source=str(ranking_source_cfg),
                            pruning_mode=pruning_mode,
                            channel_alignment=chan_align,
                            max_layer_frac=max_layer_frac,
                            max_expert_frac=max_exp_frac,
                            target_pct=target_pct,
                            scores_by_layer={
                                li: scores.tolist()
                                for li, scores in layer_avg_scores.items()
                            },
                            layer_sizes={
                                li: expert_sizes[(li, -1)]
                                for li in layer_avg_scores
                            },
                            num_experts_by_layer={
                                li: layer_idx_to_info[li].num_experts
                                for li in layer_avg_scores
                            },
                            total_expert_neurons=total_expert_neurons,
                            experiment_name=(
                                allocation_ranking_name_cfg
                                or f"{allocation_source_cfg}_alloc__"
                                   f"{ranking_source_cfg}_rank"
                            ),
                            ranking_plan=_ranking_plan,
                            ranking_plan_path=(
                                str(ranking_plan_cfg) if ranking_plan_cfg else None
                            ),
                            exact_total_layer_channels=(
                                exact_total_layer_channels_cfg
                            ),
                        )
                    )
                    selected_layer_channels = int(
                        _allocation_ranking_audit[
                            "total_selected_layer_channels"
                        ]
                    )
                    removed_expert_neurons = int(
                        _allocation_ranking_audit[
                            "total_removed_expert_neurons"
                        ]
                    )
                    _allocation_ranking_audit.update({
                        "allocation_aggregation_mode": "p95",
                        "ranking_aggregation_mode": chan_agg,
                    })
                    print(
                        "    [allocation-ranking] "
                        f"allocation_source={allocation_source_cfg}  "
                        f"ranking_source={ranking_source_cfg}  "
                        f"layer_channels={selected_layer_channels:,}"
                    )
                elif fixed_allocation_plan_cfg:
                    # ── Fixed-layer-allocation replay ───────────────────────
                    # Reuse the normal p95-aggregated score vectors, but take
                    # every layer's channel count from the source plan.  The
                    # selected indices continue through the unchanged physical
                    # pruning and evaluation paths below.
                    per_expert_pruned, _fixed_replay_audit = (
                        build_fixed_allocation_selection(
                            _fixed_source_plan,
                            source_plan_path=str(fixed_allocation_plan_cfg),
                            expected_source_selector=str(
                                fixed_allocation_selector_cfg
                            ),
                            alternate_selector=moe_selector,
                            pruning_mode=pruning_mode,
                            channel_alignment=chan_align,
                            max_layer_frac=max_layer_frac,
                            max_expert_frac=max_exp_frac,
                            target_pct=target_pct,
                            scores_by_layer={
                                li: scores.tolist()
                                for li, scores in layer_avg_scores.items()
                            },
                            layer_sizes={
                                li: expert_sizes[(li, -1)]
                                for li in layer_avg_scores
                            },
                            num_experts_by_layer={
                                li: layer_idx_to_info[li].num_experts
                                for li in layer_avg_scores
                            },
                            total_expert_neurons=total_expert_neurons,
                        )
                    )
                    selected_layer_channels = int(
                        _fixed_replay_audit["total_selected_layer_channels"]
                    )
                    removed_expert_neurons = int(
                        _fixed_replay_audit["total_removed_expert_neurons"]
                    )
                    print(
                        "    [fixed-allocation] "
                        f"fixed_allocation={fixed_allocation_selector_cfg}  "
                        f"channel_selector={moe_selector}"
                    )
                    print(
                        "    [fixed-allocation] validated source total: "
                        f"{selected_layer_channels:,} layer-channels"
                    )
                elif moe_budget_mode == "uniform":
                    # ── Uniform-budget mode ──────────────────────────────────
                    # Per-layer budget is determined solely from architecture +
                    # target_pct, so actual_pct is identical across all selectors
                    # (the selector only determines WHICH channels are pruned,
                    # not HOW MANY per layer).
                    #
                    #   n_prune_l = round(d_ff_l * target_pct/100 / align) * align
                    #   n_prune_l = min(n_prune_l, floor(d_ff_l * max_layer_frac / align) * align)
                    #
                    # Note: round() (not floor()) so that target=2% on d_ff=768, align=16
                    # snaps to 16 ch (2.083%) instead of flooring to 0.
                    #
                    # Uses layer_avg_scores (already computed above) for ranking.
                    # No global greedy pass; no alignment-adjustment needed.
                    for key, prune_list in per_expert_pruned.items():
                        _li, _ei = key
                        if _ei != -1:
                            continue  # uniform mode only applies to packed same-channel
                        d_ff_l  = expert_sizes[key]
                        # Fixed budget for this layer (aligned, capped).
                        # Use round() not floor() so small targets (e.g. 2% on d_ff=768, align=16)
                        # snap to the nearest aligned non-zero count (16 ch = 2.08%) instead of
                        # flooring to 0.  All selectors get the exact same n_prune_l, making
                        # actual_pct selector-independent.
                        n_prune_l = int(round(d_ff_l * target_pct / 100.0 / chan_align)) * chan_align
                        if max_layer_frac < 1.0:
                            _cap = (int(d_ff_l * max_layer_frac) // chan_align) * chan_align
                            n_prune_l = min(n_prune_l, _cap)
                        n_prune_l = min(n_prune_l, d_ff_l)  # can't prune more than exists
                        if n_prune_l <= 0:
                            continue
                        # Select lowest-scoring n_prune_l channels (ascending sort)
                        avg_s  = layer_avg_scores[_li]  # [d_ff_l] aggregated layer score
                        sorted_ch = np.argsort(avg_s.numpy(), kind="stable")
                        for ch in sorted_ch[:n_prune_l]:
                            prune_list.append(int(ch))
                        wt = entry_weight[key]
                        removed_expert_neurons  += n_prune_l * wt
                        selected_layer_channels += n_prune_l
                else:
                    # ── Global greedy mode (default) ─────────────────────────
                    # Select channels globally across layers in ascending score
                    # order until the target expert-neuron count is reached.
                    # Budget distribution varies by score — different selectors
                    # may produce different actual_pct values.
                    order = np.argsort(flat_scores, kind="stable")
                    for oi in order:
                        if removed_expert_neurons >= target_n:
                            break
                        li  = int(flat_layer[oi])
                        ei  = int(flat_expert[oi])
                        ni  = int(flat_neuron[oi])
                        key = (li, ei)
                        if len(per_expert_pruned[key]) >= expert_caps[key]:
                            continue
                        # Per-layer cap for packed same-channel
                        if ei == -1 and max_layer_frac < 1.0:
                            _d_ff_l = expert_sizes[key]
                            _layer_cap = (int(_d_ff_l * max_layer_frac) // chan_align) * chan_align
                            if _layer_cap > 0 and len(per_expert_pruned[key]) >= _layer_cap:
                                continue
                        per_expert_pruned[key].append(ni)
                        wt = entry_weight[key]
                        removed_expert_neurons  += wt
                        selected_layer_channels += 1  # counts channel-slots, not expert-neurons

                    # ── Alignment adjustment for packed layers ───────────────
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
                        # Cap extra by max_layer_frac to avoid exceeding per-layer limit
                        if max_layer_frac < 1.0:
                            _layer_cap = (int(old_inter * max_layer_frac) // chan_align) * chan_align
                            extra = min(extra, max(0, _layer_cap - k))
                        if extra == 0:
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

                # ── Pruning plan save/load ─────────────────────────────────
                _plan_path      = ""
                _loaded_plan    = False
                if save_plan_cfg and target_pct > 0:
                    _plan = build_pruning_plan(
                        model_id=model_name,
                        target_pct=target_pct,
                        actual_pct=actual_pct,
                        selector=moe_selector,
                        aggr_mode=chan_agg,
                        pruning_mode=pruning_mode,
                        chan_align=chan_align,
                        max_layer_frac=max_layer_frac,
                        per_expert_pruned=per_expert_pruned,
                        expert_sizes=expert_sizes,
                        num_experts_per_layer=(
                            moe_layers[0].num_experts if moe_layers else 0
                        ),
                    )
                    if _fixed_replay_audit is not None:
                        _plan["plan_kind"] = "fixed_layer_allocation_replay"
                        _plan["fixed_allocation_replay"] = _fixed_replay_audit
                        _plan["source_plan_path"] = str(
                            fixed_allocation_plan_cfg
                        )
                        _plan["source_allocation_selector"] = str(
                            fixed_allocation_selector_cfg
                        )
                        _plan["alternate_channel_selector"] = moe_selector
                        _plan["max_expert_frac"] = max_exp_frac
                    if _allocation_ranking_audit is not None:
                        _plan["plan_kind"] = "allocation_ranking_experiment"
                        _plan["allocation_ranking"] = _allocation_ranking_audit
                        _plan["allocation_source"] = str(allocation_source_cfg)
                        _plan["ranking_source"] = str(ranking_source_cfg)
                        _plan["allocation_plan_path"] = str(allocation_plan_cfg)
                        _plan["ranking_plan_path"] = (
                            str(ranking_plan_cfg) if ranking_plan_cfg else ""
                        )
                        _plan["max_expert_frac"] = max_exp_frac
                    _plan_path = make_pruning_plan_path(
                        output_dir, model_name, moe_calib_dataset,
                        n_eval, moe_calib_samples, moe_selector,
                        chan_agg, target_pct, chan_align,
                    )
                    if _fixed_replay_audit is not None:
                        _plan_path = _plan_path.replace(
                            ".json",
                            f"_fixedalloc_{fixed_allocation_selector_cfg}"
                            f"_rank_{moe_selector}.json",
                        )
                    if _allocation_ranking_audit is not None:
                        _plan_path = _plan_path.replace(
                            ".json",
                            f"_{allocation_ranking_name_cfg or allocation_source_cfg + '_alloc__' + ranking_source_cfg + '_rank'}.json",
                        )
                    save_pruning_plan(_plan, _plan_path)
                    print(f"    [plan] saved  → {_plan_path}")

                if load_plan_cfg:
                    _lp = load_pruning_plan(load_plan_cfg)
                    per_expert_pruned = apply_pruning_plan_to_selection(
                        _lp, per_expert_pruned
                    )
                    _plan_path   = load_plan_cfg
                    _loaded_plan = True
                    # Recompute actual_pruned from loaded plan
                    actual_pruned = sum(
                        len(v) * entry_weight.get((li2, -1), 1)
                        for (li2, ei2), v in per_expert_pruned.items()
                        if ei2 == -1
                    )
                    actual_pct = 100.0 * actual_pruned / max(total_expert_neurons, 1)
                    selected_layer_channels = sum(
                        len(v)
                        for (li2, ei2), v in per_expert_pruned.items()
                        if ei2 == -1
                    )
                    print(
                        f"    [plan] loaded ← {load_plan_cfg}  "
                        f"actual_pruned={actual_pruned:,}  "
                        f"actual_pct={actual_pct:.3f}%"
                    )

                _replay_overlap_channels = (
                    sum(
                        int(layer["overlap_count"])
                        for layer in _fixed_replay_audit["layers"]
                    )
                    if _fixed_replay_audit is not None else 0
                )
                _selection_audit = (
                    _allocation_ranking_audit or _fixed_replay_audit
                )
                if _allocation_ranking_audit is not None:
                    _replay_overlap_channels = sum(
                        int(layer["overlap_count"])
                        for layer in _allocation_ranking_audit["layers"]
                    )
                _replay_result_fields = {
                    "fixed_allocation_replay": _fixed_replay_audit is not None,
                    "experiment_label": (
                        f"fixed_allocation={fixed_allocation_selector_cfg} "
                        f"channel_selector={moe_selector}"
                        if _fixed_replay_audit is not None else ""
                    ),
                    "source_plan_path": (
                        str(fixed_allocation_plan_cfg)
                        if _fixed_replay_audit is not None else ""
                    ),
                    "source_allocation_selector": (
                        str(fixed_allocation_selector_cfg)
                        if _fixed_replay_audit is not None else ""
                    ),
                    "alternate_channel_selector": (
                        moe_selector if _fixed_replay_audit is not None else ""
                    ),
                    "replay_source_overlap_channels": _replay_overlap_channels,
                    "allocation_source": (
                        str(allocation_source_cfg)
                        if _allocation_ranking_audit is not None
                        else str(fixed_allocation_selector_cfg or moe_selector)
                    ),
                    "ranking_source": (
                        str(ranking_source_cfg)
                        if _allocation_ranking_audit is not None
                        else moe_selector
                    ),
                    "allocation_plan_path": (
                        str(allocation_plan_cfg)
                        if _allocation_ranking_audit is not None else ""
                    ),
                    "ranking_plan_path": (
                        str(ranking_plan_cfg)
                        if (_allocation_ranking_audit is not None and ranking_plan_cfg)
                        else ""
                    ),
                    "allocation_aggregation_mode": "p95",
                    "ranking_aggregation_mode": chan_agg,
                    "exact_total_layer_channels": (
                        exact_total_layer_channels_cfg or ""
                    ),
                }
                if _allocation_ranking_audit is not None:
                    _replay_result_fields.update({
                        "fixed_allocation_replay": False,
                        "experiment_label": str(
                            _allocation_ranking_audit["experiment_name"]
                        ),
                        "source_plan_path": str(allocation_plan_cfg),
                        "source_allocation_selector": str(
                            allocation_source_cfg
                        ),
                        "alternate_channel_selector": str(ranking_source_cfg),
                    })

                # ── Fail-fast: verify packed_same_channel used (li,-1) keys ─
                if pruning_mode == "packed_same_channel" and target_pct > 0:
                    _n_layer_ch_keys = sum(
                        1 for (_li2, _ei2) in per_expert_pruned
                        if _ei2 == -1 and per_expert_pruned[(_li2, _ei2)]
                    )
                    if _n_layer_ch_keys == 0:
                        _sample_keys = list(per_expert_pruned.keys())[:5]
                        raise RuntimeError(
                            "packed_same_channel scoring produced NO (layer,-1) "
                            "entries.  per_expert_pruned sample keys: "
                            f"{_sample_keys}  "
                            f"selected_layer_channels={selected_layer_channels}  "
                            f"removed_expert_neurons={actual_pruned}  "
                            "Scoring took per-expert path. Check [detect] output."
                        )
                    print(
                        f"    [validation] packed_same_channel: "
                        f"{_n_layer_ch_keys} layers selected  "
                        f"selected_layer_channels={selected_layer_channels}  "
                        f"removed_expert_neurons={actual_pruned}"
                    )

                if save_bound_tightness:
                    (
                        _target_bound_tightness_json,
                        _target_expert_bound_scores_npz,
                    ) = save_moe_bound_tightness_diagnostics(
                        moe_layers,
                        expert_activations,
                        per_expert_pruned,
                        output_dir=output_dir,
                        timestamp=ts,
                        model_name=model_name,
                        aggregation=chan_agg,
                        save_expert_scores=save_expert_bound_scores,
                        max_inputs_per_expert=bound_tightness_max_inputs,
                    )
                    print(
                        "    [bound-tightness] JSON: "
                        f"{_target_bound_tightness_json}"
                    )
                    if _target_expert_bound_scores_npz:
                        print(
                            "    [bound-tightness] expert scores: "
                            f"{_target_expert_bound_scores_npz}"
                        )

                _print_per_layer_distribution(
                    per_expert_pruned, expert_sizes, moe_layers,
                    routed_counts, total_expert_neurons, chan_align,
                    pruning_mode=pruning_mode,
                )

                # ── Pre-prune per-layer stats (analytical, before modifying tensors) ─
                _per_layer_rows = _build_per_layer_rows(
                    per_expert_pruned, expert_sizes, moe_layers,
                    all_expert_scores, routed_counts, pruning_mode,
                )
                if _selection_audit is not None:
                    _audit_by_layer = {
                        int(row["layer_idx"]): row
                        for row in _selection_audit["layers"]
                    }
                    for _layer_row in _per_layer_rows:
                        _audit_row = _audit_by_layer[int(_layer_row["layer_idx"])]
                        _layer_row.update({
                            "allocation_source": _replay_result_fields[
                                "allocation_source"
                            ],
                            "ranking_source": _replay_result_fields[
                                "ranking_source"
                            ],
                            "score_scale_abs_max": _audit_row.get(
                                "ranking_score_scale_abs_max", ""
                            ),
                            "allocation_reference_channel_ids": _audit_row[
                                "source_indices"
                            ],
                            "selected_channel_ids": _audit_row[
                                "selected_indices"
                            ],
                            "ranking_overlap_count": _audit_row[
                                "overlap_count"
                            ],
                            "ranking_jaccard_with_allocation": _audit_row[
                                "jaccard"
                            ],
                        })
                _pruned_ep = sum(
                    r.get("removed_expert_params", 0) for r in _per_layer_rows
                )
                _expert_params_after_sel = _expert_params_before - _pruned_ep
                _expert_param_red_pct    = (
                    100.0 * _pruned_ep / _expert_params_before
                    if _expert_params_before > 0 else 0.0
                )
                # per-layer CSV path derived from main CSV
                _per_layer_csv_path = main_csv_path.replace(".csv", "_per_layer.csv")

                # ── Dry-run: save selection analysis, skip pruning+PPL ──────────
                if dry_run:
                    _dr_rows = _print_dryrun_table(
                        per_expert_pruned, expert_sizes, moe_layers,
                        all_expert_scores, routed_counts,
                        total_expert_neurons, chan_align,
                        pruning_mode, chan_agg, moe_selector,
                        target_pct, actual_pruned, actual_pct,
                        max_layer_frac, model_name, ts,
                    )
                    _dr_csv = os.path.join(
                        output_dir,
                        f"moe_dryrun_{ts}_{target_pct}pct.csv",
                    )
                    _flush_csv(_dr_csv, _dr_rows, MOE_DRYRUN_CSV_KEYS)
                    _dr_json = _dr_csv.replace(".csv", ".json")
                    import json as _json
                    with open(_dr_json, "w") as _jf:
                        _json.dump({
                            "model": model_name, "target_pct": target_pct,
                            "actual_pct": round(actual_pct, 4),
                            "pruning_mode": pruning_mode,
                            "aggregation_mode": chan_agg,
                            "selector": moe_selector,
                            "max_layer_frac": max_layer_frac,
                            **_replay_result_fields,
                            "fixed_allocation_audit": _fixed_replay_audit,
                            "rows": _dr_rows,
                        }, _jf, indent=2)
                    print(f"  [dry-run] CSV  saved: {_dr_csv}")
                    print(f"  [dry-run] JSON saved: {_dr_json}")
                    del expert_activations
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _log_gpu_memory("dry-run after cache release")
                    continue   # skip method loop — no pruning, no PPL

                # ── Release calibration caches before pruning ─────────────────
                # pure_delete needs no per-expert activations during pruning;
                # reconstruction methods do.  Only hold a reference if needed.
                # Keep calibration data alive if ANY method needs it.
                _needs_calib = any(m in RESIDUAL_METHODS for m in METHODS)
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
                    # Residual reconstruction counters (extended)
                    _resid_stable        = 0
                    _resid_skipped       = 0
                    _resid_failed        = 0
                    _resid_rejected      = 0
                    _resid_attempted     = 0
                    _resid_total_cand    = 0
                    _resid_skip_few_tok  = 0
                    _resid_skip_ill_cond = 0
                    _resid_skip_non_fin  = 0
                    _resid_skip_too_lrg  = 0
                    _resid_skip_not_imp  = 0
                    _err_del_wsum        = 0.0
                    _err_res_wsum        = 0.0
                    _err_cnt             = 0
                    _upd_norms_all: list = []
                    # Residual tracking (for CSV/JSON audit fields)
                    _residual_applied       = False  # True if residual was run
                    _residual_fallback_used = False  # True if fell back unexpectedly
                    _plan_path              = getattr(locals(), "_plan_path", "")
                    _loaded_plan_flag       = getattr(locals(), "_loaded_plan", False)
                    _resid_variant          = method  # captured per-method

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
                                # Same-channel pruning across ALL experts in the layer.
                                # Handles both packed Layout B (tensor slice) and
                                # unpacked Layout A (loop over individual modules).
                                ec              = info.experts_container
                                n_pruned_actual = len(prune_idx)
                                removed_en      = n_pruned_actual * info.num_experts

                                # keep_idx in old d_ff space (needed for residual path)
                                _km = torch.ones(d_ff_orig, dtype=torch.bool)
                                _km[prune_idx] = False
                                _keep_idx_packed = _km.nonzero(as_tuple=True)[0]

                                # ── Debug: shapes BEFORE pruning ──────────────
                                if info.experts_packed and ec is not None:
                                    print(
                                        f"      [before] layer {li}: "
                                        f"gate_up_proj {list(ec.gate_up_proj.shape)}  "
                                        f"down_proj {list(ec.down_proj.shape)}  "
                                        f"old_intermediate={d_ff_orig}"
                                    )
                                else:
                                    try:
                                        _e0 = info.expert_modules[0]
                                        print(
                                            f"      [before] layer {li} (unpacked): "
                                            f"gate_proj {list(_e0.gate_proj.weight.shape)}  "
                                            f"down_proj {list(_e0.down_proj.weight.shape)}  "
                                            f"old_intermediate={d_ff_orig}"
                                        )
                                    except Exception:
                                        print(
                                            f"      [before] layer {li}: "
                                            f"old_intermediate={d_ff_orig}"
                                        )

                                if info.experts_packed and ec is not None:
                                    # ── PACKED TENSOR PATH (Layout B) ─────────
                                    if method in ("residual_full_moe", "residual_ridge_moe") and _calib_ref is not None:
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
                                        _residual_applied = True
                                        print(
                                            f"      [residual_full_moe layer {li}] (packed) "
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
                                    print(
                                        f"      [after]  layer {li}: "
                                        f"gate_up_proj {list(ec.gate_up_proj.shape)}  "
                                        f"down_proj {list(ec.down_proj.shape)}  "
                                        f"new_intermediate={new_inter}  "
                                        f"pruned_channels={n_pruned_actual}"
                                    )
                                else:
                                    # ── UNPACKED SAME-CHANNEL PATH (Layout A) ──
                                    # Apply identical prune_idx to every expert so
                                    # the same channels are removed globally.

                                    # Step 1: residual reconstruction (BEFORE pruning)
                                    if method in RESIDUAL_METHODS and method != "residual_mask_moe" and _calib_ref is not None:
                                        _t_rs = time.perf_counter()
                                        _layer_acts_u = {
                                            _ei2: _calib_ref.get((li, _ei2))
                                            for _ei2 in range(info.num_experts)
                                        }
                                        try:
                                            _rs = apply_residual_dispatch_unpacked(
                                                method,
                                                info.expert_modules,
                                                prune_idx, _keep_idx_packed,
                                                _layer_acts_u,
                                                residual_cfg,
                                            )
                                            t_recon_total      += time.perf_counter() - _t_rs
                                            _resid_stable      += _rs["n_stable"]
                                            _resid_skipped     += _rs["n_skipped"]
                                            _resid_failed      += _rs["n_failed"]
                                            _resid_rejected    += _rs.get("n_rejected", 0)
                                            _resid_attempted   += _rs.get("n_attempted", 0)
                                            _resid_total_cand  += _rs.get("n_total_candidate", 0)
                                            _resid_skip_few_tok  += _rs.get("skip_too_few_tokens", 0)
                                            _resid_skip_ill_cond += _rs.get("skip_ill_conditioned", 0)
                                            _resid_skip_non_fin  += _rs.get("skip_non_finite", 0)
                                            _resid_skip_too_lrg  += _rs.get("skip_update_too_large", 0)
                                            _resid_skip_not_imp  += _rs.get("skip_not_improved", 0)
                                            _ed = _rs.get("mean_err_delete", float("nan"))
                                            _er = _rs.get("mean_err_resid",  float("nan"))
                                            if _ed == _ed:  # not nan
                                                _err_del_wsum += _ed
                                                _err_res_wsum += _er if _er == _er else 0.0
                                                _err_cnt      += 1
                                            if "mean_update_norm" in _rs and _rs["mean_update_norm"] == _rs["mean_update_norm"]:
                                                _upd_norms_all.append(_rs["mean_update_norm"])
                                            _residual_applied = True
                                            print(
                                                f"      [{method} layer {li}] (unpacked) "
                                                f"stable={_rs['n_stable']}  "
                                                f"skip={_rs['n_skipped']}  "
                                                f"fail={_rs['n_failed']}  "
                                                f"reject={_rs.get('n_rejected', 0)}  "
                                                f"mean_tok={_rs['mean_tokens']:.0f}"
                                            )
                                        except ValueError as _ve:
                                            print(f"      [ERROR] layer {li}: {_ve}")
                                            _notes_str = "failed_method_not_supported"
                                            _residual_fallback_used = True
                                    elif method in RESIDUAL_METHODS and method != "residual_mask_moe" and _calib_ref is None:
                                        print(
                                            f"      [warn] layer {li}: "
                                            f"{method} skipped — no calibration data"
                                        )
                                        _residual_fallback_used = True

                                    # Step 2: physical pruning (same prune_idx for all experts)
                                    _unpacked_ok = 0
                                    for ei_u in range(info.num_experts):
                                        expert_u = info.expert_modules[ei_u]
                                        try:
                                            prune_expert_channels(expert_u, prune_idx)
                                            _unpacked_ok += 1
                                        except Exception as _upe:
                                            logger.warning(
                                                "unpacked same-ch prune layer=%d ei=%d: %s",
                                                li, ei_u, _upe,
                                            )
                                    new_inter = d_ff_orig - n_pruned_actual
                                    try:
                                        _e0n = info.expert_modules[0]
                                        print(
                                            f"      [after]  layer {li} (unpacked): "
                                            f"gate_proj {list(_e0n.gate_proj.weight.shape)}  "
                                            f"down_proj {list(_e0n.down_proj.weight.shape)}  "
                                            f"new_intermediate={new_inter}  "
                                            f"pruned_channels={n_pruned_actual}  "
                                            f"experts_pruned={_unpacked_ok}/{info.num_experts}"
                                        )
                                    except Exception:
                                        print(
                                            f"      [after]  layer {li} (unpacked): "
                                            f"new_intermediate={new_inter}  "
                                            f"pruned_channels={n_pruned_actual}"
                                        )

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
                                    "selector": f"{moe_selector}_{chan_agg}",
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
                                    f"{info.num_experts} experts "
                                    f"({d_ff_orig}->{new_inter})  "
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
                                        "selector": moe_selector,
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
                                    "selector": moe_selector,
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

                    # ── Post-prune param / FLOP accounting ────────────────────
                    _total_model_params_after  = _extract_total_param_count(
                        count_parameters(model)
                    )
                    _total_model_param_red_pct = (
                        100.0 * (_total_model_params_before - _total_model_params_after)
                        / _total_model_params_before
                        if _total_model_params_before > 0 else 0.0
                    )
                    _active_flops_after    = _estimate_moe_active_flops(_all_moe_for_count)
                    _active_flop_red_pct   = (
                        100.0 * (_active_flops_before - _active_flops_after)
                        / _active_flops_before
                        if _active_flops_before > 0 else 0.0
                    )

                    # Save per-layer CSV (once per run; derived from main CSV name)
                    if _per_layer_rows:
                        _flush_csv(
                            _per_layer_csv_path, _per_layer_rows,
                            MOE_PER_LAYER_CSV_KEYS,
                        )

                    # Compact layer pruning summary
                    _pruned_pl = [_r for _r in _per_layer_rows
                                  if _r.get("pruned_channels", 0) > 0]
                    print()
                    print("  ── Layer pruning summary ──────────────────────────────────")
                    print(f"  Layers pruned          : {len(_pruned_pl)}/{len(_per_layer_rows)}")
                    print(f"  Selected layer-channels: {selected_layer_channels:,}")
                    print(f"  Removed expert-neurons : {actual_pruned:,}")
                    if _pruned_pl:
                        _chs = [_r["pruned_channels"] for _r in _pruned_pl]
                        _mid = sorted(_chs)[len(_chs) // 2]
                        print(f"  Channels min/med/max   : {min(_chs)}/{_mid}/{max(_chs)}")
                        _top10 = sorted(_pruned_pl,
                                        key=lambda _rr: _rr["pruned_channels"],
                                        reverse=True)[:10]
                        print("  Top-10 most-pruned layers:")
                        for _pr in _top10:
                            print(
                                f"    layer {_pr['layer_idx']:3d}: "
                                f"{_pr['pruned_channels']} ch  "
                                f"({_pr['actual_layer_pruning_pct']:.1f}%  of  "
                                f"{_pr['old_intermediate']})"
                            )
                    print(
                        f"  Expert param reduction : {_expert_param_red_pct:.3f}%"
                        f"  ({_pruned_ep:,} / {_expert_params_before:,})"
                    )
                    print(f"  Total model reduction  : {_total_model_param_red_pct:.3f}%")
                    print(f"  Active expert FLOP red.: {_active_flop_red_pct:.3f}%")

                    # ── Post-pruning validation (packed_same_channel) ─────────
                    _prune_valid = True
                    if pruning_mode == "packed_same_channel" and target_pct > 0:
                        _pruned_pl_check  = [_r for _r in _per_layer_rows
                                             if _r.get("pruned_channels", 0) > 0]
                        _shape_chg_check  = any(_r.get("shape_changed", False)
                                                for _r in _per_layer_rows)
                        _ep_red_check     = _expert_param_red_pct > 0
                        _flop_red_check   = _active_flop_red_pct > 0
                        _chan_check       = selected_layer_channels > 0
                        _neu_check        = actual_pruned > 0
                        _fails = []
                        if not _chan_check:
                            _fails.append(
                                f"selected_layer_channels={selected_layer_channels} (expected >0)")
                        if not _neu_check:
                            _fails.append(
                                f"removed_expert_neurons={actual_pruned} (expected >0)")
                        if not _pruned_pl_check:
                            _fails.append(
                                "per-layer CSV has no pruned layers (pruned_channels>0)")
                        if not _shape_chg_check:
                            _fails.append(
                                "no layer has shape_changed=True in per-layer CSV")
                        if not _ep_red_check:
                            _fails.append(
                                f"expert_param_reduction_pct={_expert_param_red_pct:.4f}% (expected >0)")
                        if not _flop_red_check:
                            _fails.append(
                                f"active_expert_flop_reduction_pct={_active_flop_red_pct:.4f}% (expected >0)")
                        if _fails:
                            _prune_valid = False
                            _sep = "=" * 60
                            print(f"  {_sep}")
                            print(
                                f"  PRUNING VALIDATION FAILED "
                                f"(packed_same_channel target={target_pct}%)"
                            )
                            for _fmsg in _fails:
                                print(f"    FAIL: {_fmsg}")
                            print(
                                "  Physical pruning did NOT happen correctly."
                            )
                            print(
                                "  Skipping PPL evaluation."
                            )
                            print(
                                "  Check [detect] / [before] / [after] output above."
                            )
                            print(f"  {_sep}")
                        else:
                            print(
                                f"  [validation] packed_same_channel OK: "
                                f"{len(_pruned_pl_check)} layers pruned  "
                                f"param_red={_expert_param_red_pct:.3f}%  "
                                f"flop_red={_active_flop_red_pct:.3f}%"
                            )

                    # ── Forward check ─────────────────────────────────────────
                    # Only run forward pass if pruning validation passed.
                    # Separate status: failed_invalid_physical_pruning vs failed_forward_check
                    fp_ok = True
                    if not _prune_valid:
                        fp_ok = False
                        _notes_str = "failed_invalid_physical_pruning"
                        print("    ERROR: pruning validation failed — skipping forward check and PPL eval")
                    else:
                        fp_ok = verify_forward_pass(model, tokenizer, device)
                        if not fp_ok:
                            _notes_str = "failed_forward_check"
                            print("    ERROR: forward pass failed — skipping PPL eval")
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

                    # ── Residual summary stats — computed once, used by err_row and summary ──
                    _VARIANT_MAP = {
                        "pure_delete":                                          "none",
                        "residual_full_moe":                                    "full",
                        "residual_ridge_moe":                                   "ridge",
                        "residual_ridge_only_if_improves_moe":                  "ridge_only_if_improves",
                        "residual_nearest_channel_merge_moe":                   "nearest_channel_merge",
                        "residual_nearest_channel_merge_only_if_improves_moe":  "nearest_channel_merge_only_if_improves",
                        "residual_mask_moe":                                    "mask",
                        "residual_diag_moe":                                    "diag",
                    }
                    _cov_pct = (
                        100.0 * _resid_stable / _resid_total_cand
                        if _resid_total_cand > 0 else 0.0
                    )
                    _mean_e_del = (
                        _err_del_wsum / _err_cnt
                        if _err_cnt > 0 else float("nan")
                    )
                    _mean_e_res = (
                        _err_res_wsum / _err_cnt
                        if _err_cnt > 0 else float("nan")
                    )
                    _mean_imp = (
                        100.0 * (_mean_e_del - _mean_e_res)
                        / (_mean_e_del + 1e-12)
                        if _err_cnt > 0 else float("nan")
                    )
                    _mean_upd = (
                        sum(_upd_norms_all) / len(_upd_norms_all)
                        if _upd_norms_all else float("nan")
                    )
                    _max_upd = (
                        max(_upd_norms_all)
                        if _upd_norms_all else float("nan")
                    )
                    _actual_method = (
                        method
                        if (_residual_applied or method == "pure_delete")
                        else "pure_delete"
                    )
                    # Override _resid_variant with a readable slug
                    _resid_variant = _VARIANT_MAP.get(method, method)

                    if not fp_ok:
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
                                "notes": locals().get("_notes_str", "forward_pass_failed"),
                                "csv_path": main_csv_path,
                                "json_path": json_path,
                            }

                            err_row.update({
                                "target_pct": target_pct,
                                "actual_pct": round(actual_pct, 4),
                                "processed_moe_layers": len(moe_layers),
                                "n_eval": n_eval,
                                "moe_calib_samples": moe_calib_samples,
                                "status": locals().get("_notes_str", "failed_forward_check"),
                                "expert_param_reduction_pct": round(_expert_param_red_pct, 4),
                                "total_model_param_reduction_pct": round(_total_model_param_red_pct, 4),
                                "estimated_active_expert_flop_reduction_pct": round(_active_flop_red_pct, 4),
                                "residual_lambda": resid_lambda,
                                "per_layer_csv_path": (
                                    os.path.relpath(_per_layer_csv_path)
                                    if _per_layer_rows else ""
                                ),
                                "requested_method":       method,
                                "actual_method":          _actual_method,
                                "residual_applied":       _residual_applied,
                                "residual_fallback_used": _residual_fallback_used,
                                "residual_stable_experts":  _resid_stable,
                                "residual_skipped_experts": _resid_skipped,
                                "residual_failed_experts":  _resid_failed,
                                "residual_rejected_experts": _resid_rejected,
                                "residual_time_sec":      round(t_recon_total, 2),
                                "expert_layout":          "unpacked",
                                "model_revision":         "",
                                "transformers_version":   "",
                                "torch_version":          torch.__version__,
                                "pruning_plan_path":      _plan_path,
                                "loaded_pruning_plan":    _loaded_plan_flag,
                                "residual_variant":       _resid_variant,
                                "residual_total_candidate_experts": _resid_total_cand,
                                "residual_attempted_experts": _resid_attempted,
                                "residual_applied_experts":   _resid_stable,
                                "residual_coverage_pct":      round(_cov_pct, 2),
                                "residual_skip_too_few_tokens":    _resid_skip_few_tok,
                                "residual_skip_ill_conditioned":   _resid_skip_ill_cond,
                                "residual_skip_non_finite":        _resid_skip_non_fin,
                                "residual_skip_update_too_large":  _resid_skip_too_lrg,
                                "residual_skip_not_improved":      _resid_skip_not_imp,
                                "mean_err_delete":            _mean_e_del,
                                "mean_err_resid":             _mean_e_res,
                                "mean_local_improvement_pct": _mean_imp,
                                "mean_update_norm":           _mean_upd,
                                "max_update_norm":            _max_upd,
                            })
                            err_row.update(_replay_result_fields)
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
                            collect_per_example=collect_per_example_nll,
                        )
                        ppl   = ppl_info["perplexity"]
                        _baseline_tokens = int(
                            baseline_eval_info_per_ds[_ds]["n_tokens"]
                        )
                        _pruned_tokens = int(ppl_info["n_tokens"])
                        if _baseline_tokens != _pruned_tokens:
                            raise AssertionError(
                                f"evaluation token count changed for {_ds}: "
                                f"baseline={_baseline_tokens}, "
                                f"pruned={_pruned_tokens}"
                            )
                        delta = ppl - cur_bppl
                        rel   = 100.0 * delta / cur_bppl if cur_bppl > 0 else 0.0
                        _paired_nll_path = ""
                        _paired_nll_stats = {
                            "mean_nll_difference": float("nan"),
                            "ci_lower": float("nan"),
                            "ci_upper": float("nan"),
                            "n_resamples": 0,
                        }
                        if collect_per_example_nll:
                            _baseline_examples = baseline_eval_info_per_ds[_ds].get(
                                "per_example"
                            )
                            _pruned_examples = ppl_info.get("per_example")
                            if not isinstance(_baseline_examples, list) or not isinstance(
                                _pruned_examples, list
                            ):
                                raise AssertionError(
                                    "paired NLL collection requested but per-example "
                                    "records are missing"
                                )
                            _paired_nll_path = os.path.join(
                                output_dir, f"paired_nll_{_ds}_{ts}.csv"
                            )
                            write_paired_nll_csv(
                                _paired_nll_path,
                                dataset=_ds,
                                corpus_sha256=eval_corpus_sha256[_ds],
                                baseline_examples=_baseline_examples,
                                pruned_examples=_pruned_examples,
                            )
                            _paired_nll_stats = paired_bootstrap_nll_difference(
                                [float(row["nll_sum"]) for row in _pruned_examples],
                                [float(row["nll_sum"]) for row in _baseline_examples],
                                [int(row["n_tokens"]) for row in _baseline_examples],
                                n_resamples=paired_bootstrap_resamples,
                                seed=int(cfg.get("seed", 42)),
                            )
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
                            "evaluation_max_seq_len": max_seq,
                            "evaluation_batch_size": batch_sz,
                            "evaluation_num_texts": len(all_eval_corpora[_ds]),
                            "evaluation_corpus_sha256": eval_corpus_sha256[_ds],
                            "evaluation_preprocessing": (
                                "same_text_list; tokenizer padding+truncation; "
                                "padding labels=-100; shifted nonpadding token NLL"
                            ),
                            "evaluation_protocol_label": str(
                                cfg.get("evaluation_protocol_label", "")
                            ),
                            "seed": int(cfg.get("seed", 42)),
                            "baseline_eval_tokens": _baseline_tokens,
                            "pruned_eval_tokens": _pruned_tokens,
                            "evaluation_token_count_match": True,
                            "fresh_model_process_required": True,
                            "process_id": os.getpid(),
                            "model_load_instance_id": _model_load_instance_id,
                            "score_comparison_csv_path": _target_score_comparison_csv,
                            "score_comparison_json_path": _target_score_comparison_json,
                            "per_example_nll_path": _paired_nll_path,
                            "mean_nll_difference": _paired_nll_stats[
                                "mean_nll_difference"
                            ],
                            "mean_nll_difference_ci95_lower": _paired_nll_stats[
                                "ci_lower"
                            ],
                            "mean_nll_difference_ci95_upper": _paired_nll_stats[
                                "ci_upper"
                            ],
                            "paired_bootstrap_resamples": _paired_nll_stats[
                                "n_resamples"
                            ],
                            "bound_tightness_json_path": (
                                _target_bound_tightness_json
                            ),
                            "expert_bound_scores_npz_path": (
                                _target_expert_bound_scores_npz
                            ),
                        }
                        summary.update({
                            "target_pct": target_pct,
                            "actual_pct": round(actual_pct, 4),
                            "processed_moe_layers": len(moe_layers),
                            "n_eval": n_eval,
                            "moe_calib_samples": moe_calib_samples,
                            "status": "ok",
                            "expert_param_reduction_pct": round(_expert_param_red_pct, 4),
                            "total_model_param_reduction_pct": round(_total_model_param_red_pct, 4),
                            "estimated_active_expert_flop_reduction_pct": round(_active_flop_red_pct, 4),
                            "residual_lambda": resid_lambda,
                            "per_layer_csv_path": (
                                os.path.relpath(_per_layer_csv_path)
                                if _per_layer_rows else ""
                            ),
                            "requested_method":       method,
                            "actual_method":          _actual_method,
                            "residual_applied":       _residual_applied,
                            "residual_fallback_used": _residual_fallback_used,
                            "residual_variant":       _resid_variant,
                            "residual_stable_experts":  _resid_stable,
                            "residual_skipped_experts": _resid_skipped,
                            "residual_failed_experts":  _resid_failed,
                            "residual_rejected_experts": _resid_rejected,
                            "residual_time_sec":      round(t_recon_total, 2),
                            "expert_layout":          "unpacked",
                            "model_revision":         "",
                            "transformers_version":   "",
                            "torch_version":          torch.__version__,
                            "pruning_plan_path":      _plan_path,
                            "loaded_pruning_plan":    _loaded_plan_flag,
                            "residual_total_candidate_experts": _resid_total_cand,
                            "residual_attempted_experts": _resid_attempted,
                            "residual_applied_experts":   _resid_stable,
                            "residual_coverage_pct":      round(_cov_pct, 2),
                            "residual_skip_too_few_tokens":    _resid_skip_few_tok,
                            "residual_skip_ill_conditioned":   _resid_skip_ill_cond,
                            "residual_skip_non_finite":        _resid_skip_non_fin,
                            "residual_skip_update_too_large":  _resid_skip_too_lrg,
                            "residual_skip_not_improved":      _resid_skip_not_imp,
                            "mean_err_delete":            _mean_e_del,
                            "mean_err_resid":             _mean_e_res,
                            "mean_local_improvement_pct": _mean_imp,
                            "mean_update_norm":           _mean_upd,
                            "max_update_norm":            _max_upd,
                        })
                        summary.update(_replay_result_fields)
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

    if score_comparison_only:
        print("\nMOE score comparison complete (no pruning, no PPL).")
        for _csv_path, _json_path in comparison_output_paths:
            print(f"  CSV : {_csv_path}")
            print(f"  JSON: {_json_path}")
        return

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
        "method_framing": (
            "Global layer-budget allocation followed by certified "
            "ellipsoid-based within-layer channel selection. Raw certified "
            "bounds are not assumed comparable across layers."
            if ranking_source_cfg == "rmsnorm_ellipsoid_bound"
            else "Allocation/ranking control experiment; existing selector "
                 "definitions are unchanged."
        ),
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
