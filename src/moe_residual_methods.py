"""
moe_residual_methods.py
=======================
Residual reconstruction methods for MoE expert-channel pruning.

All public functions work on UNPACKED expert layouts (nn.ModuleList of nn.Linear
modules). Each function takes the same inputs and returns a unified stats dict.

Pruning plan save/load helpers are also provided here so that all methods at the
same target percentage can use the exact same channel-selection indices.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default lambda used by residual_full_moe (kept as constant for backward compat)
BEST_RESIDUAL_LAM = 1e-2

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# Methods that require calibration activations to be collected before pruning.
# pure_delete does NOT require activations.
RESIDUAL_METHODS: set = {
    "residual_full_moe",
    "residual_ridge_moe",
    "residual_only_if_improves_moe",
    "residual_ridge_only_if_improves_moe",
    "residual_nearest_channel_merge_moe",
    "residual_nearest_channel_merge_only_if_improves_moe",
    "residual_diag_moe",
    "residual_mask_moe",
}


def _empty_residual_stats() -> Dict:
    """Return a zeroed-out stats dict for pure_delete (no residual applied)."""
    return {
        "n_total_candidate": 0,
        "n_attempted":       0,
        "n_stable":          0,
        "n_skipped":         0,
        "n_failed":          0,
        "n_rejected":        0,
        "skip_too_few_tokens":    0,
        "skip_ill_conditioned":   0,
        "skip_non_finite":        0,
        "skip_update_too_large":  0,
        "skip_not_improved":      0,
        "mean_tokens":            0.0,
        "mean_err_delete":        float("nan"),
        "mean_err_resid":         float("nan"),
        "mean_local_improvement_pct": float("nan"),
        "mean_update_norm":       float("nan"),
        "max_update_norm":        float("nan"),
    }


# ---------------------------------------------------------------------------
# Pruning plan helpers
# ---------------------------------------------------------------------------

def make_pruning_plan_path(
    results_dir: str,
    model_name: str,
    dataset: str,
    n_eval: int,
    calib_samples: int,
    selector: str,
    aggr_mode: str,
    target_pct: float,
    align: int,
) -> str:
    """Generate a canonical path for a pruning-plan JSON file."""
    model_slug = model_name.replace("/", "_").replace("-", "_")
    fname = (
        f"{model_slug}_{dataset}_n{n_eval}_calib{calib_samples}"
        f"_{selector}_{aggr_mode}_{target_pct:.1f}pct_align{align}.json"
    )
    plan_dir = os.path.join(results_dir, "pruning_plans")
    return os.path.join(plan_dir, fname)


def build_pruning_plan(
    model_id: str,
    target_pct: float,
    actual_pct: float,
    selector: str,
    aggr_mode: str,
    pruning_mode: str,
    chan_align: int,
    max_layer_frac: float,
    per_expert_pruned: Dict,
    expert_sizes: Dict,
    num_experts_per_layer: int,
) -> Dict:
    """Build a serialisable pruning-plan dict from the computed selection."""
    try:
        import transformers as _tf
        tv = _tf.__version__
    except Exception:
        tv = "unknown"
    layers = []
    for (li, ei), prune_list in sorted(per_expert_pruned.items()):
        if ei != -1:
            continue  # packed_same_channel uses (li, -1) keys only
        old_inter = expert_sizes.get((li, ei), 0)
        layers.append({
            "layer_idx":       li,
            "prune_idx":       sorted(prune_list),
            "old_intermediate": old_inter,
            "new_intermediate": old_inter - len(prune_list),
            "pruned_channels":  len(prune_list),
        })
    return {
        "model_id":           model_id,
        "transformers_version": tv,
        "torch_version":      torch.__version__,
        "target_pct":         target_pct,
        "actual_pct":         round(actual_pct, 4),
        "selector":           selector,
        "aggregation_mode":   aggr_mode,
        "pruning_mode":       pruning_mode,
        "channel_alignment":  chan_align,
        "max_layer_frac":     max_layer_frac,
        "num_layers":         len(layers),
        "num_experts_per_layer": num_experts_per_layer,
        "total_selected_layer_channels": sum(l["pruned_channels"] for l in layers),
        "layers":             layers,
    }


def save_pruning_plan(plan: Dict, path: str) -> None:
    """Save pruning plan to JSON, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(plan, fh, indent=2, default=str)
    logger.info("Pruning plan saved → %s", path)


def load_pruning_plan(path: str) -> Dict:
    """Load and return a pruning plan dict from JSON."""
    with open(path) as fh:
        return json.load(fh)


def apply_pruning_plan_to_selection(
    plan: Dict,
    per_expert_pruned: Dict,
) -> Dict:
    """
    Override per_expert_pruned with the channel indices from a loaded plan.
    Returns a new dict (does not mutate the original).
    """
    new_sel = dict(per_expert_pruned)
    for layer_info in plan.get("layers", []):
        li         = int(layer_info["layer_idx"])
        prune_list = [int(x) for x in layer_info["prune_idx"]]
        new_sel[(li, -1)] = prune_list
    return new_sel


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _swiglu_activations(
    X:      torch.Tensor,
    gate_w: torch.Tensor,
    up_w:   torch.Tensor,
) -> torch.Tensor:
    """SwiGLU forward pass → [N, d_ff] activation tensor."""
    return F.silu(X @ gate_w.T) * (X @ up_w.T)


# ---------------------------------------------------------------------------
# Core method: ridge residual (dual-form)
# ---------------------------------------------------------------------------

def apply_residual_ridge_unpacked(
    expert_modules:     list,
    prune_idx:          torch.Tensor,
    keep_idx:           torch.Tensor,
    expert_activations: Dict,                   # {ei: Tensor[N, d_model]}
    min_tokens:         int   = 16,
    ridge_lambda:       float = 1e-2,
    tau:                float = 1.0,
    solve_on_cpu:       bool  = True,
    norm_clip:          Optional[float] = None,
    max_rel_norm:       Optional[float] = None,
    only_if_improves:   bool  = False,
    improvement_margin: float = 1.0,
) -> Dict:
    """
    Ridge-regression residual reconstruction for unpacked MoE experts.

    For each expert e with sufficient routed tokens:

        act_all = SiLU(X_e @ gate^T) * (X_e @ up^T)   # [N, d_ff]
        A_K = act_all[:, keep_idx]   A_P = act_all[:, prune_idx]
        W_P = down_proj.weight[:, prune_idx]
        Y_lost = A_P @ W_P^T                            # [N, d_model]

    Dual-form ridge solve (cheap when N < d_ff):

        (A_K A_K^T + λ·diag_mean(A_K A_K^T)·I) B = Y_lost
        ΔD = A_K^T B                                    # [n_kept, d_model]

    Update (in-place, before physical pruning):

        down_proj.weight[:, keep_idx] += τ · ΔD^T

    If only_if_improves=True, the update is applied only when:

        MSE(Y_orig, Y_resid) < MSE(Y_orig, Y_delete) * improvement_margin

    Returns a unified stats dict.
    """
    n_total      = len(expert_modules)
    n_stable     = n_skipped = n_failed = n_rejected = n_attempted = 0
    skip_few     = skip_ill  = skip_nfin = skip_large = skip_nimp  = 0
    tok_counts:   List[int]   = []
    err_del_list: List[float] = []
    err_res_list: List[float] = []
    upd_norms:    List[float] = []

    for ei, exp in enumerate(expert_modules):
        X_raw = expert_activations.get(ei, None)
        if X_raw is None or X_raw.shape[0] < min_tokens:
            n_skipped += 1
            skip_few  += 1
            continue

        N = X_raw.shape[0]
        n_attempted += 1
        try:
            with torch.no_grad():
                X      = X_raw.detach().float()
                gate_w = exp.gate_proj.weight.data.detach().float()  # [d_ff, d_model]
                up_w   = exp.up_proj.weight.data.detach().float()    # [d_ff, d_model]
                down_w = exp.down_proj.weight.data.detach().float()  # [d_model, d_ff]
                if solve_on_cpu:
                    X, gate_w, up_w, down_w = (
                        t.cpu() for t in (X, gate_w, up_w, down_w)
                    )

                act_all = _swiglu_activations(X, gate_w, up_w)        # [N, d_ff]
                _p = prune_idx.to(act_all.device)
                _k = keep_idx.to(act_all.device)

                A_P = act_all[:, _p]                                   # [N, n_pruned]
                A_K = act_all[:, _k]                                   # [N, n_kept]
                W_P = down_w[:, _p.to(down_w.device)]                 # [d_model, n_pruned]
                W_K = down_w[:, _k.to(down_w.device)]                 # [d_model, n_kept]

                # Residual target: contribution of pruned channels to output
                Y_lost = A_P @ W_P.T                                   # [N, d_model]

                # Dual-form ridge: (A_K A_K^T + λI) B = Y_lost
                AAt = A_K @ A_K.T                                      # [N, N]
                lam = ridge_lambda * float(AAt.diagonal().mean())
                reg = lam * torch.eye(N, dtype=torch.float32,
                                      device=AAt.device)
                B     = torch.linalg.solve(AAt + reg, Y_lost)          # [N, d_model]
                Delta = A_K.T @ B                                       # [n_kept, d_model]

                if not torch.isfinite(Delta).all():
                    n_failed  += 1
                    skip_nfin += 1
                    continue

                upd_norm = float(Delta.norm())
                upd_norms.append(upd_norm)

                # Optional hard norm clip
                if norm_clip is not None and upd_norm > norm_clip:
                    n_skipped  += 1
                    skip_large += 1
                    continue
                # Optional relative norm clip
                if max_rel_norm is not None:
                    ref = float(W_K.norm())
                    if ref > 1e-8 and upd_norm / ref > max_rel_norm:
                        n_skipped  += 1
                        skip_large += 1
                        continue

                W_K_new = W_K + tau * Delta.T                          # [d_model, n_kept]

                # Only-if-improves gate
                if only_if_improves:
                    Y_orig  = A_P @ W_P.T + A_K @ W_K.T               # [N, d_model]
                    Y_del   = A_K @ W_K.T
                    Y_res   = A_K @ W_K_new.T
                    e_del   = float((Y_orig - Y_del).pow(2).mean())
                    e_res   = float((Y_orig - Y_res).pow(2).mean())
                    err_del_list.append(e_del)
                    err_res_list.append(e_res)
                    if e_res >= e_del * improvement_margin:
                        n_rejected += 1
                        skip_nimp  += 1
                        continue

                # Write back
                _k_dev = _k.to(exp.down_proj.weight.device)
                exp.down_proj.weight.data[:, _k_dev] = W_K_new.to(
                    device=exp.down_proj.weight.device,
                    dtype=exp.down_proj.weight.dtype,
                )

            n_stable += 1
            tok_counts.append(N)

        except Exception as exc:
            logger.warning("residual_ridge expert=%d: %s", ei, exc)
            n_failed  += 1
            skip_ill  += 1

    mean_tok   = (sum(tok_counts) / len(tok_counts)) if tok_counts else 0.0
    mean_e_del = (sum(err_del_list) / len(err_del_list)) if err_del_list else float("nan")
    mean_e_res = (sum(err_res_list) / len(err_res_list)) if err_res_list else float("nan")
    mean_imp   = (
        100.0 * (mean_e_del - mean_e_res) / (mean_e_del + 1e-12)
        if err_del_list else float("nan")
    )
    mean_upd   = (sum(upd_norms) / len(upd_norms)) if upd_norms else float("nan")
    max_upd    = max(upd_norms) if upd_norms else float("nan")

    return {
        "n_total_candidate":  n_total,
        "n_attempted":        n_attempted,
        "n_stable":           n_stable,
        "n_skipped":          n_skipped,
        "n_failed":           n_failed,
        "n_rejected":         n_rejected,
        "skip_too_few_tokens":    skip_few,
        "skip_ill_conditioned":   skip_ill,
        "skip_non_finite":        skip_nfin,
        "skip_update_too_large":  skip_large,
        "skip_not_improved":      skip_nimp,
        "mean_tokens":            mean_tok,
        "mean_err_delete":            mean_e_del,
        "mean_err_resid":             mean_e_res,
        "mean_local_improvement_pct": mean_imp,
        "mean_update_norm":           mean_upd,
        "max_update_norm":            max_upd,
    }


# ---------------------------------------------------------------------------
# Core method: nearest-channel merge (channelwise / diag)
# ---------------------------------------------------------------------------

def apply_nearest_channel_merge_unpacked(
    expert_modules:     list,
    prune_idx:          torch.Tensor,
    keep_idx:           torch.Tensor,
    expert_activations: Dict,
    min_tokens:         int   = 16,
    alpha_clip:         float = 2.0,
    merge_metric:       str   = "ls_scalar",  # "ls_scalar" | "cosine"
    only_if_improves:   bool  = False,
    improvement_margin: float = 1.0,
    solve_on_cpu:       bool  = True,
) -> Dict:
    """
    Nearest-channel-merge residual for unpacked MoE experts.

    For each pruned channel p:
      1. Find nearest kept channel k by activation similarity.
      2. Compute scalar  alpha = (a_k · a_p) / (||a_k||^2 + eps)   [ls_scalar]
         or find k by    cosine sim between a_p and a_k             [cosine]
      3. Clip alpha to [-alpha_clip, +alpha_clip].
      4. Update:  down_proj.weight[:, k] += alpha * down_proj.weight[:, p]

    More numerically stable than full-matrix residual because each correction
    is a scalar update to a single kept column.

    If only_if_improves=True, the entire per-expert update is committed only
    when it reduces the local reconstruction MSE.
    """
    n_total      = len(expert_modules)
    n_stable     = n_skipped = n_failed = n_rejected = n_attempted = 0
    skip_few     = skip_ill  = skip_nfin = skip_nimp  = 0
    tok_counts:   List[int]   = []
    err_del_list: List[float] = []
    err_res_list: List[float] = []

    for ei, exp in enumerate(expert_modules):
        X_raw = expert_activations.get(ei, None)
        if X_raw is None or X_raw.shape[0] < min_tokens:
            n_skipped += 1
            skip_few  += 1
            continue

        N = X_raw.shape[0]
        n_attempted += 1
        try:
            with torch.no_grad():
                X      = X_raw.detach().float()
                gate_w = exp.gate_proj.weight.data.detach().float()
                up_w   = exp.up_proj.weight.data.detach().float()
                down_w = exp.down_proj.weight.data.detach().float()
                if solve_on_cpu:
                    X, gate_w, up_w, down_w = (
                        t.cpu() for t in (X, gate_w, up_w, down_w)
                    )

                act_all = _swiglu_activations(X, gate_w, up_w)        # [N, d_ff]
                _p = prune_idx.to(act_all.device)
                _k = keep_idx.to(act_all.device)

                A_P  = act_all[:, _p]                                  # [N, n_pruned]
                A_K  = act_all[:, _k]                                  # [N, n_kept]
                W_P  = down_w[:, _p.to(down_w.device)]                # [d_model, n_pruned]
                W_K  = down_w[:, _k.to(down_w.device)]                # [d_model, n_kept]

                n_pruned = A_P.shape[1]
                n_kept   = A_K.shape[1]
                if n_pruned == 0 or n_kept == 0:
                    n_skipped += 1
                    continue

                # Per-channel: find nearest kept channel and merge
                A_K_ssq = (A_K * A_K).sum(0)                          # [n_kept]
                W_K_new = W_K.clone()

                for pi in range(n_pruned):
                    a_p = A_P[:, pi]                                   # [N]
                    dot = A_K.T @ a_p                                  # [n_kept]
                    if merge_metric == "cosine":
                        a_p_norm = float(a_p.norm()) + 1e-8
                        a_k_norm = A_K_ssq.sqrt() + 1e-8
                        sim      = dot / (a_k_norm * a_p_norm)        # [n_kept]
                        k_best   = int(sim.abs().argmax())
                    else:  # ls_scalar: best scalar fit
                        k_best = int(dot.abs().argmax())

                    denom = float(A_K_ssq[k_best]) + 1e-8
                    alpha = float(dot[k_best]) / denom
                    alpha = max(-alpha_clip, min(alpha_clip, alpha))
                    W_K_new[:, k_best] = W_K_new[:, k_best] + alpha * W_P[:, pi]

                # Only-if-improves gate (whole-expert decision)
                if only_if_improves:
                    Y_orig  = A_P @ W_P.T + A_K @ W_K.T               # [N, d_model]
                    Y_del   = A_K @ W_K.T
                    Y_res   = A_K @ W_K_new.T
                    e_del   = float((Y_orig - Y_del).pow(2).mean())
                    e_res   = float((Y_orig - Y_res).pow(2).mean())
                    err_del_list.append(e_del)
                    err_res_list.append(e_res)
                    if e_res >= e_del * improvement_margin:
                        n_rejected += 1
                        skip_nimp  += 1
                        continue

                # Write back
                _k_dev = _k.to(exp.down_proj.weight.device)
                exp.down_proj.weight.data[:, _k_dev] = W_K_new.to(
                    device=exp.down_proj.weight.device,
                    dtype=exp.down_proj.weight.dtype,
                )

            n_stable += 1
            tok_counts.append(N)

        except Exception as exc:
            logger.warning("nearest_channel_merge expert=%d: %s", ei, exc)
            n_failed += 1
            skip_ill += 1

    mean_tok   = (sum(tok_counts) / len(tok_counts)) if tok_counts else 0.0
    mean_e_del = (sum(err_del_list) / len(err_del_list)) if err_del_list else float("nan")
    mean_e_res = (sum(err_res_list) / len(err_res_list)) if err_res_list else float("nan")
    mean_imp   = (
        100.0 * (mean_e_del - mean_e_res) / (mean_e_del + 1e-12)
        if err_del_list else float("nan")
    )

    return {
        "n_total_candidate":  n_total,
        "n_attempted":        n_attempted,
        "n_stable":           n_stable,
        "n_skipped":          n_skipped,
        "n_failed":           n_failed,
        "n_rejected":         n_rejected,
        "skip_too_few_tokens":    skip_few,
        "skip_ill_conditioned":   skip_ill,
        "skip_non_finite":        skip_nfin,
        "skip_update_too_large":  0,
        "skip_not_improved":      skip_nimp,
        "mean_tokens":            mean_tok,
        "mean_err_delete":            mean_e_del,
        "mean_err_resid":             mean_e_res,
        "mean_local_improvement_pct": mean_imp,
        "mean_update_norm":           float("nan"),
        "max_update_norm":            float("nan"),
    }


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def apply_residual_dispatch_unpacked(
    method:             str,
    expert_modules:     list,
    prune_idx:          torch.Tensor,
    keep_idx:           torch.Tensor,
    expert_activations: Dict,
    residual_cfg:       Dict,
) -> Dict:
    """
    Dispatch to the correct residual method for unpacked MoE experts.

    residual_cfg keys used:
        residual_lambda               float  (default: BEST_RESIDUAL_LAM)
        residual_tau                  float  (default: 1.0)
        residual_min_tokens           int    (default: 16)
        solve_residual_on_cpu         bool   (default: True)
        residual_update_norm_clip     float  (default: None)
        residual_max_relative_update_norm float (default: None)
        residual_improvement_margin   float  (default: 1.0)
        residual_alpha_clip           float  (default: 2.0)
        residual_merge_metric         str    (default: "ls_scalar")

    Raises ValueError for unknown methods so the caller can record
    status='failed_method_not_supported' instead of silently falling back.
    """
    min_tok   = int(residual_cfg.get("residual_min_tokens", 16))
    lam       = float(residual_cfg.get("residual_lambda", BEST_RESIDUAL_LAM))
    tau       = float(residual_cfg.get("residual_tau", 1.0))
    cpu       = bool(residual_cfg.get("solve_residual_on_cpu", True))
    clip      = residual_cfg.get("residual_update_norm_clip", None)
    rel_clip  = residual_cfg.get("residual_max_relative_update_norm", None)
    margin    = float(residual_cfg.get("residual_improvement_margin", 1.0))
    a_clip    = float(residual_cfg.get("residual_alpha_clip", 2.0))
    metric    = str(residual_cfg.get("residual_merge_metric", "ls_scalar"))

    if method in ("residual_full_moe", "residual_ridge_moe"):
        return apply_residual_ridge_unpacked(
            expert_modules, prune_idx, keep_idx, expert_activations,
            min_tokens=min_tok, ridge_lambda=lam, tau=tau,
            solve_on_cpu=cpu, norm_clip=clip, max_rel_norm=rel_clip,
            only_if_improves=False,
        )

    elif method == "residual_only_if_improves_moe":
        return apply_residual_ridge_unpacked(
            expert_modules, prune_idx, keep_idx, expert_activations,
            min_tokens=min_tok, ridge_lambda=BEST_RESIDUAL_LAM, tau=tau,
            solve_on_cpu=cpu, norm_clip=clip, max_rel_norm=rel_clip,
            only_if_improves=True, improvement_margin=margin,
        )

    elif method == "residual_ridge_only_if_improves_moe":
        return apply_residual_ridge_unpacked(
            expert_modules, prune_idx, keep_idx, expert_activations,
            min_tokens=min_tok, ridge_lambda=lam, tau=tau,
            solve_on_cpu=cpu, norm_clip=clip, max_rel_norm=rel_clip,
            only_if_improves=True, improvement_margin=margin,
        )

    elif method == "residual_nearest_channel_merge_moe":
        return apply_nearest_channel_merge_unpacked(
            expert_modules, prune_idx, keep_idx, expert_activations,
            min_tokens=min_tok, alpha_clip=a_clip, merge_metric=metric,
            only_if_improves=False, improvement_margin=margin,
            solve_on_cpu=cpu,
        )

    elif method == "residual_nearest_channel_merge_only_if_improves_moe":
        return apply_nearest_channel_merge_unpacked(
            expert_modules, prune_idx, keep_idx, expert_activations,
            min_tokens=min_tok, alpha_clip=a_clip, merge_metric=metric,
            only_if_improves=True, improvement_margin=margin,
            solve_on_cpu=cpu,
        )

    elif method in ("residual_diag_moe",):
        # Alias: same as nearest_channel_merge with ls_scalar
        return apply_nearest_channel_merge_unpacked(
            expert_modules, prune_idx, keep_idx, expert_activations,
            min_tokens=min_tok, alpha_clip=a_clip, merge_metric="ls_scalar",
            only_if_improves=False, improvement_margin=margin,
            solve_on_cpu=cpu,
        )

    else:
        raise ValueError(
            f"Unknown or unsupported residual method for unpacked experts: {method!r}. "
            f"Supported: {sorted(RESIDUAL_METHODS - {'residual_mask_moe'})}"
        )
