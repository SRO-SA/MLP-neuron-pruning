"""
merging.py
==========
Compensated neuron merging for SwiGLU MLP pruning.

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

TWO MERGE STRATEGIES
--------------------
A. merge_weight_similarity
   Find the best target j using weight-space distance:
     beta_u = (u_i · u_j) / (||u_j||² + ε)
     dist   = ||g_i − g_j|| / (||g_i|| + ε)
            + ||u_i − beta_u · u_j|| / (||u_i|| + ε)
   The second term is the residual after projecting u_i onto u_j's direction.

B. merge_activation_similarity
   Collect actual MLP-input hidden states {r_t} from calibration data.
   Compute activation vectors: a_i = SiLU(R @ g_i) * (R @ u_i)  [N_tokens]
     beta_a    = (a_i · a_j) / (||a_j||² + ε)
     residual  = ||a_i − beta_a · a_j|| / (||a_i|| + ε)
   Choose j with smallest residual.

PHYSICAL PRUNING
----------------
In both cases, after all compensation updates are applied (accumulating multiple
merges onto the same target), prune_model_by_layer_indices() does the physical
removal — dropping the pruned neurons' rows from gate/up and columns from down.

KEY: All down_proj compensation updates are applied BEFORE any physical pruning.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
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


# ===========================================================================
# Calibration data collection
# ===========================================================================

def collect_mlp_inputs(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_seq_len: int = 128,
) -> List[torch.Tensor]:
    """
    Collect MLP input hidden states (post-RMSNorm) for all transformer layers
    by registering forward pre-hooks.

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
            for prompt in tqdm(
                prompts, desc="  Collecting calibration states", leave=False
            ):
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
# Merge target selection
# ===========================================================================

def compute_merge_assignments_weight(
    layer,
    prune_indices: torch.Tensor,
    keep_indices:  torch.Tensor,
    eps: float = EPS,
) -> List[dict]:
    """
    For each pruned neuron i, find the best kept neuron j using weight-space
    similarity, then compute beta_u for the down-projection update.

        beta_u  = (u_i · u_j) / (||u_j||² + ε)
        dist(i,j) = ||g_i − g_j|| / (||g_i|| + ε)
                  + ||u_i − beta_u · u_j|| / (||u_i|| + ε)

    The update rule is:  down_proj[:, j] += beta_u * down_proj[:, i]

    All computations are fully vectorised over (n_prune × n_keep) pairs.

    Returns
    -------
    List[dict] with keys: i, j, beta, dist
    """
    if len(prune_indices) == 0:
        return []

    w = get_mlp_weights(layer)
    w_gate = w["gate"].detach().float().cpu()   # [d_ff, d_model]
    w_up   = w["up"].detach().float().cpu()     # [d_ff, d_model]

    G_prune = w_gate[prune_indices]    # [n_prune, d_model]
    U_prune = w_up[prune_indices]      # [n_prune, d_model]
    G_keep  = w_gate[keep_indices]     # [n_keep,  d_model]
    U_keep  = w_up[keep_indices]       # [n_keep,  d_model]

    n_prune = G_prune.shape[0]
    prune_list = prune_indices.tolist()
    keep_list  = keep_indices.tolist()

    norms_g_p = G_prune.norm(dim=1)    # [n_prune]
    norms_g_k = G_keep.norm(dim=1)     # [n_keep]
    norms_u_p = U_prune.norm(dim=1)    # [n_prune]
    norms_u_k = U_keep.norm(dim=1)     # [n_keep]

    # ── Gate distance ─────────────────────────────────────────────────────────
    # ||g_i − g_j||² = ||g_i||² + ||g_j||² − 2·(g_i·g_j)
    dots_g    = G_prune @ G_keep.T                          # [n_prune, n_keep]
    dist_g_sq = (
        norms_g_p.unsqueeze(1) ** 2
        + norms_g_k.unsqueeze(0) ** 2
        - 2.0 * dots_g
    ).clamp(min=0.0)
    dist_g_norm = dist_g_sq.sqrt() / (norms_g_p.unsqueeze(1) + eps)  # [n_prune, n_keep]

    # ── Up residual distance ──────────────────────────────────────────────────
    # beta_u[i,j] = (u_i·u_j) / (||u_j||² + ε)
    # ||u_i − beta_u·u_j||² = ||u_i||² − 2·beta_u·(u_i·u_j) + beta_u²·||u_j||²
    dots_u     = U_prune @ U_keep.T                         # [n_prune, n_keep]
    norms_u_ksq = norms_u_k ** 2                            # [n_keep]
    betas_u    = dots_u / (norms_u_ksq.unsqueeze(0) + eps)  # [n_prune, n_keep]

    dist_u_sq = (
        norms_u_p.unsqueeze(1) ** 2
        - 2.0 * betas_u * dots_u
        + betas_u ** 2 * norms_u_ksq.unsqueeze(0)
    ).clamp(min=0.0)
    dist_u_norm = dist_u_sq.sqrt() / (norms_u_p.unsqueeze(1) + eps)   # [n_prune, n_keep]

    total_dist    = dist_g_norm + dist_u_norm      # [n_prune, n_keep]
    best_j_local  = total_dist.argmin(dim=1)       # [n_prune]

    assignments = []
    for idx in range(n_prune):
        jl   = int(best_j_local[idx])
        i_g  = prune_list[idx]
        j_g  = keep_list[jl]
        beta = float(betas_u[idx, jl])
        dist = float(total_dist[idx, jl])
        assignments.append({
            "i":    i_g,
            "j":    j_g,
            "beta": round(beta, 8),
            "dist": round(dist, 8),
        })

    return assignments


def compute_merge_assignments_activation(
    layer,
    prune_indices: torch.Tensor,
    keep_indices:  torch.Tensor,
    all_r:         torch.Tensor,
    eps:           float = EPS,
) -> List[dict]:
    """
    For each pruned neuron i, find the best kept neuron j using activation-space
    similarity from calibration data.

    Activation vector for neuron i over calibration tokens:
        a_i = SiLU(R @ g_i) * (R @ u_i)   shape [N_tokens]

        beta_a  = (a_i · a_j) / (||a_j||² + ε)
        residual = ||a_i − beta_a · a_j|| / (||a_i|| + ε)

    The update rule is:  down_proj[:, j] += beta_a * down_proj[:, i]

    Parameters
    ----------
    all_r : [N_tokens, d_model] CPU float32 tensor of MLP inputs from calibration.

    Returns
    -------
    List[dict] with keys: i, j, beta, residual
    """
    if len(prune_indices) == 0:
        return []

    w = get_mlp_weights(layer)
    w_gate = w["gate"].detach().float().cpu()   # [d_ff, d_model]
    w_up   = w["up"].detach().float().cpu()     # [d_ff, d_model]

    prune_list = prune_indices.tolist()
    keep_list  = keep_indices.tolist()
    n_prune    = len(prune_list)

    R = all_r.float().cpu()    # [N_tokens, d_model]

    # Compute activations for all neurons: [N_tokens, d_ff]
    # Then select prune / keep subsets.
    with torch.no_grad():
        G_all = R @ w_gate.T   # [N_tokens, d_ff]
        U_all = R @ w_up.T     # [N_tokens, d_ff]
        A_all = F.silu(G_all) * U_all   # [N_tokens, d_ff]

    A_prune = A_all[:, prune_indices]   # [N_tokens, n_prune]
    A_keep  = A_all[:, keep_indices]    # [N_tokens, n_keep]

    # beta_a[i,j] = (a_i · a_j) / (||a_j||² + ε)
    dots_a      = A_prune.T @ A_keep                     # [n_prune, n_keep]
    norms_k_sq  = A_keep.norm(dim=0) ** 2                # [n_keep]
    betas_a     = dots_a / (norms_k_sq.unsqueeze(0) + eps)  # [n_prune, n_keep]

    # ||a_i − beta·a_j||² = ||a_i||² − 2·beta·(a_i·a_j) + beta²·||a_j||²
    norms_p     = A_prune.norm(dim=0)                    # [n_prune]
    residual_sq = (
        norms_p.unsqueeze(1) ** 2
        - 2.0 * betas_a * dots_a
        + betas_a ** 2 * norms_k_sq.unsqueeze(0)
    ).clamp(min=0.0)

    residual_norm = residual_sq.sqrt() / (norms_p.unsqueeze(1) + eps)  # [n_prune, n_keep]
    best_j_local  = residual_norm.argmin(dim=1)   # [n_prune]

    assignments = []
    for idx in range(n_prune):
        jl       = int(best_j_local[idx])
        i_g      = prune_list[idx]
        j_g      = keep_list[jl]
        beta     = float(betas_a[idx, jl])
        residual = float(residual_norm[idx, jl])
        assignments.append({
            "i":        i_g,
            "j":        j_g,
            "beta":     round(beta, 8),
            "residual": round(residual, 8),
        })

    return assignments


# ===========================================================================
# Apply compensation and prune
# ===========================================================================

def apply_merge_and_prune(
    model,
    prune_indices_per_layer: List[torch.Tensor],
    assignments_per_layer:   List[List[dict]],
    label: str = "",
) -> Tuple[object, dict]:
    """
    Apply beta compensation updates to down_proj, then physically prune.

    Steps
    -----
    1. Deep-copy the model.
    2. For each layer: for each (i, j, beta) accumulate
           down_proj[:, j] += beta * down_proj[:, i]
       across ALL assignments BEFORE physical removal.
    3. Call prune_model_by_layer_indices() to physically remove pruned neurons.

    IMPORTANT: Multiple pruned neurons can merge into the same kept target j.
    The updates are additive and order-independent (we only read prune columns,
    only write keep columns, so there is no aliasing).

    Returns
    -------
    pruned_model : model with compensated down_proj and neurons physically removed
    info         : dict from prune_model_by_layer_indices
    """
    from .pruning import prune_model_by_layer_indices

    # Step 1: Clone the original model so we can modify down_proj safely
    merged = clone_model(model)
    layers = get_transformer_layers(merged)

    with torch.no_grad():
        for layer_idx, (layer, pi, assignments) in enumerate(
            zip(layers, prune_indices_per_layer, assignments_per_layer)
        ):
            if len(pi) == 0 or not assignments:
                continue

            w    = get_mlp_weights(layer)
            down = w["down"]  # [d_model, d_ff]  — actual parameter tensor

            # Accumulate all compensation updates before any removal
            for asn in assignments:
                i_g  = asn["i"]
                j_g  = asn["j"]
                beta = asn["beta"]
                # Clone the source column to avoid aliasing
                # (safe because i_g is always in prune_indices, never updated)
                col_i = down.data[:, i_g].clone()
                down.data[:, j_g].add_(beta * col_i)

    # Step 2: Physically prune (prune_model_by_layer_indices clones merged internally)
    pruned_model, info = prune_model_by_layer_indices(
        merged, prune_indices_per_layer, label=label
    )

    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pruned_model, info


# ===========================================================================
# Diagnostics
# ===========================================================================

def _build_layer_diagnostics(
    all_scores:              List[torch.Tensor],
    prune_indices_per_layer: List[torch.Tensor],
    assignments_per_layer:   List[List[dict]],
) -> List[dict]:
    """Build per-layer and per-neuron diagnostic summary for a merge method."""
    layers_diag = []

    for layer_idx, (scores, pi, assignments) in enumerate(
        zip(all_scores, prune_indices_per_layer, assignments_per_layer)
    ):
        if len(pi) == 0:
            layers_diag.append({"layer_idx": layer_idx, "n_pruned": 0})
            continue

        # Per-neuron details
        neuron_detail = []
        for asn in assignments:
            # 'dist' for weight-sim, 'residual' for activation-sim
            metric_val = asn.get("dist", asn.get("residual", 0.0))
            neuron_detail.append({
                "layer_idx":    layer_idx,
                "i":            asn["i"],
                "j":            asn["j"],
                "beta":         round(asn.get("beta", 0.0), 8),
                "metric":       round(metric_val, 8),
                "bound_score":  round(float(scores[asn["i"]]), 8),
            })

        betas   = [a.get("beta", 0.0) for a in assignments]
        metrics = [a.get("dist", a.get("residual", 0.0)) for a in assignments]
        targets = [a["j"] for a in assignments]
        target_counts = Counter(targets)

        layers_diag.append({
            "layer_idx":              layer_idx,
            "n_pruned":               int(len(pi)),
            "n_unique_targets":       len(set(targets)),
            "max_neurons_per_target": max(target_counts.values()) if target_counts else 0,
            "avg_beta":               round(float(sum(betas) / max(len(betas), 1)), 8),
            "avg_metric":             round(float(sum(metrics) / max(len(metrics), 1)), 8),
            "max_metric":             round(float(max(metrics)) if metrics else 0.0, 8),
            "neurons":                neuron_detail,
        })

    return layers_diag


# ===========================================================================
# Entry point
# ===========================================================================

def run_bound_merge_mode(
    model,
    tokenizer,
    cfg:        dict,
    device:     str,
    output_dir: str = "results",
) -> None:
    """
    Compare pure deletion vs compensated merging for the same candidate neurons
    selected by cumul_score_sum at alpha = 1e-4, 1e-3, 1e-2.

    For each alpha, runs three methods:
      pure_delete                — remove neurons, no compensation
      merge_weight_similarity    — compensate using gate/up weight distance
      merge_activation_similarity — compensate using calibration activation similarity

    Saves PPL comparison table, generation examples, and per-neuron diagnostics.
    """
    from .bound_analysis import (
        CALIBRATION_PROMPTS,
        _k,
        compute_bound_scores_and_R,
        select_by_budget,
    )
    from .evaluation import evaluate_perplexity, load_eval_dataset, run_generation_tests
    from .flops import estimate_mlp_flops
    from .model_utils import count_parameters
    from .pruning import prune_model_by_layer_indices, verify_forward_pass

    os.makedirs(output_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    layers = get_transformer_layers(model)

    ALPHAS  = [1e-4, 1e-3, 1e-2]
    METHODS = [
        "pure_delete",
        "merge_weight_similarity",
        "merge_activation_similarity",
    ]

    # ── Pre-compute bound scores (weight-only, fast) ──────────────────────────
    print("\nComputing bound scores for all layers …")
    all_scores: List[torch.Tensor] = []
    for layer in layers:
        s, _ = compute_bound_scores_and_R(layer)
        all_scores.append(s)

    # ── Collect calibration hidden states (one pass, shared across all alphas) ─
    print("Collecting calibration hidden states (used for activation merge) …")
    all_r_per_layer = collect_mlp_inputs(
        model, tokenizer, CALIBRATION_PROMPTS, device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )

    # ── Dataset + baseline PPL ────────────────────────────────────────────────
    use_fallback = cfg.get("use_fallback_corpus", True)
    n_eval       = cfg.get("bound_analysis_eval_samples", 64)
    eval_texts   = load_eval_dataset(n_eval, use_fallback_corpus=use_fallback)

    baseline_params = count_parameters(model)
    baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))

    print("Computing baseline PPL …")
    bp           = evaluate_perplexity(
        model, tokenizer, texts=eval_texts,
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 4),
        device=device,
    )
    baseline_ppl = bp["perplexity"]
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    all_results: List[dict] = []

    for alpha in ALPHAS:
        # ── Select candidate neurons ──────────────────────────────────────────
        prune_indices_per_layer: List[torch.Tensor] = []
        for i, scores in enumerate(all_scores):
            ref = float(scores.sum())
            pi, _ = select_by_budget(scores, alpha, ref)
            prune_indices_per_layer.append(pi)

        total_pruned = sum(len(pi) for pi in prune_indices_per_layer)
        total_n      = sum(s.numel() for s in all_scores)
        pct          = 100.0 * total_pruned / total_n if total_n else 0.0

        if total_pruned == 0:
            print(f"alpha={_k(alpha)}: 0 neurons selected — skipping\n")
            continue

        # ── Compute keep_indices per layer ────────────────────────────────────
        keep_indices_per_layer: List[torch.Tensor] = []
        for i, scores in enumerate(all_scores):
            d_ff    = scores.numel()
            p_set   = set(prune_indices_per_layer[i].tolist())
            keep    = torch.tensor(
                [j for j in range(d_ff) if j not in p_set], dtype=torch.long
            )
            keep_indices_per_layer.append(keep)

        print(f"{'=' * 64}")
        print(f"alpha={_k(alpha)}  pruned={total_pruned} ({pct:.3f}%)")
        print(f"{'=' * 64}\n")

        # ── Compute merge assignments (shared across methods B and C) ─────────
        print("  Computing weight-similarity assignments …")
        weight_asns: List[List[dict]] = []
        for layer_idx, layer in enumerate(tqdm(layers, desc="    weight", leave=False)):
            pi = prune_indices_per_layer[layer_idx]
            ki = keep_indices_per_layer[layer_idx]
            weight_asns.append(
                compute_merge_assignments_weight(layer, pi, ki)
                if len(pi) > 0 else []
            )

        print("  Computing activation-similarity assignments …")
        act_asns: List[List[dict]] = []
        for layer_idx, layer in enumerate(tqdm(layers, desc="    activation", leave=False)):
            pi    = prune_indices_per_layer[layer_idx]
            ki    = keep_indices_per_layer[layer_idx]
            all_r = all_r_per_layer[layer_idx]
            act_asns.append(
                compute_merge_assignments_activation(layer, pi, ki, all_r)
                if len(pi) > 0 else []
            )

        # ── Run all three methods ─────────────────────────────────────────────
        for method in METHODS:
            print(f"\n  [{method}]")
            row: dict = {
                "alpha":         alpha,
                "method":        method,
                "total_pruned":  total_pruned,
                "pct_pruned":    round(pct, 4),
                "baseline_ppl":  round(baseline_ppl, 4),
            }
            pruned_model = None
            try:
                if method == "pure_delete":
                    pruned_model, _ = prune_model_by_layer_indices(
                        model, prune_indices_per_layer,
                        label=f"pure_delete_a{_k(alpha)}",
                    )
                elif method == "merge_weight_similarity":
                    pruned_model, _ = apply_merge_and_prune(
                        model, prune_indices_per_layer, weight_asns,
                        label=f"merge_weight_a{_k(alpha)}",
                    )
                elif method == "merge_activation_similarity":
                    pruned_model, _ = apply_merge_and_prune(
                        model, prune_indices_per_layer, act_asns,
                        label=f"merge_act_a{_k(alpha)}",
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
                p_flops  = estimate_mlp_flops(
                    pruned_model, seq_len=cfg.get("max_seq_len", 512)
                )
                flop_red = 100.0 * (
                    1.0 - p_flops["total_flops"] / baseline_flops["total_flops"]
                )
                mlp_red = 100.0 * (
                    1.0 - p_params["mlp"] / baseline_params["mlp"]
                )

                gens = run_generation_tests(pruned_model, tokenizer, device=device)

                row.update({
                    "perplexity":         round(ppl, 4),
                    "perplexity_delta":   round(ppl - baseline_ppl, 4),
                    "forward_pass_ok":    fp_ok,
                    "mlp_params_red_pct": round(mlp_red, 4),
                    "flops_red_pct":      round(flop_red, 4),
                    "generation_examples": gens,
                })
                print(
                    f"    PPL={ppl:.4f}  dPPL={ppl - baseline_ppl:+.4f}"
                    f"  MLP-{mlp_red:.2f}%  FLOPs-{flop_red:.2f}%"
                )

            except Exception as exc:
                logger.error(
                    "Method [%s] alpha=%s failed: %s",
                    method, _k(alpha), exc, exc_info=True,
                )
                row["notes"] = f"ERROR: {exc}"
            finally:
                if pruned_model is not None:
                    del pruned_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results.append(row)

        # ── Save per-neuron diagnostics for this alpha ────────────────────────
        diag = {
            "alpha":                  alpha,
            "total_pruned":           total_pruned,
            "pct_pruned":             round(pct, 4),
            "weight_diagnostics":     _build_layer_diagnostics(
                all_scores, prune_indices_per_layer, weight_asns
            ),
            "activation_diagnostics": _build_layer_diagnostics(
                all_scores, prune_indices_per_layer, act_asns
            ),
        }
        diag_path = os.path.join(
            output_dir, f"merge_diagnostics_alpha{_k(alpha)}_{ts}.json"
        )
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2, default=str)
        logger.info("Diagnostics saved: %s", diag_path)
        print(f"\n  Diagnostics: {diag_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("BOUND MERGE SUMMARY")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'─' * 80}")
    print(
        f"  {'alpha':>9}  {'method':<32}  {'pruned':>7}"
        f"  {'PPL':>9}  {'dPPL':>9}  {'MLP-':>7}"
    )
    print(f"{'─' * 80}")
    for r in all_results:
        if "perplexity" in r:
            print(
                f"  {_k(r['alpha']):>9}  {r['method']:<32}"
                f"  {r['total_pruned']:>7}"
                f"  {r['perplexity']:>9.4f}"
                f"  {r['perplexity_delta']:>+9.4f}"
                f"  {r.get('mlp_params_red_pct', 0.0):>6.2f}%"
            )
        else:
            print(
                f"  {_k(r['alpha']):>9}  {r['method']:<32}"
                f"  {r.get('notes', 'ERROR')}"
            )
    print(f"{'=' * 80}\n")

    # ── Save full report ───────────────────────────────────────────────────────
    report = {
        "timestamp":   ts,
        "mode":        "bound_merge",
        "baseline_ppl": baseline_ppl,
        "results":     all_results,
    }
    path = os.path.join(output_dir, f"bound_merge_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Bound merge report: %s", path)
    print(f"Report saved to: {path}\n")
