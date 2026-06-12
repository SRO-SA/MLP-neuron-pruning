"""
bound_analysis.py
=================
Score distribution analysis and threshold-based pruning for the
RMSNorm-bounded SwiGLU neuron contribution score.

KEY INSIGHT
-----------
The rmsnorm_bound_angle score S_i is a worst-case upper bound:

    ||c_i(r)|| <= S_i   for ALL inputs r with ||r||_2 <= R

This means:
  YES  If S_i is extremely small -> neuron i can be CERTIFIED as negligible.
  NO   The bottom 5% of neurons are NOT automatically safe to prune.
       (The bound may be far from tight; most neurons may have similar S_i.)

This module answers the key question:
  "How many MLP neurons are ACTUALLY near-zero under the worst-case bound?"

If very few neurons fall below any reasonable threshold, the bound is too
conservative to justify pruning and a data-driven approach (calibration)
is needed.

TWO THRESHOLD-BASED PRUNING MODES
----------------------------------
A. static_abs / static_rel:
   Prune neuron i if score_i < theta (absolute) or score_i / median < theta_rel.
   If no neurons in a layer satisfy this, prune ZERO from that layer.

B. cumul_score_sum / cumul_mlp_norm:
   Sort neurons by S_i ascending. Prune the largest prefix P such that
       sum_{i in P} S_i <= alpha * reference_norm_layer
   Two reference choices:
     - score_sum: reference = sum(S_i) for the layer (purely static)
     - mlp_output_norm: reference = E[||MLP(r)||] from calibration data
   If the sum of ALL S_i is already below the budget, prune all.
   If the smallest S_i exceeds the budget, prune zero.

CLI ENTRY POINTS
----------------
--bound-analysis                    Full pipeline (dist + calibration + PPL)
--bound-analysis --no-ppl           Distributions + count tables only
--bound-analysis --no-activation-verification
                                    PPL without activation verification
--bound-ppl-only                    Only cumul_score_sum α=1e-4/1e-3/1e-2 + PPL
--activation-verification-only      Hook-based activation scores + correlations
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model_utils import (
    get_mlp_module,
    get_mlp_weights,
    get_rmsnorm_before_mlp,
    get_transformer_layers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

ABS_THRESHOLDS = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
REL_THRESHOLDS = [1e-6,  1e-5,  1e-4, 1e-3, 1e-2]
ALPHA_VALUES   = [1e-6,  1e-5,  1e-4, 1e-3, 1e-2]


def _k(v: float) -> str:
    """Consistent string key for threshold / alpha values (e.g. '1.0e-06')."""
    return f"{v:.1e}"


CALIBRATION_PROMPTS: List[str] = [
    "The transformer architecture was introduced in the paper Attention Is All You Need.",
    "Python is a high-level, general-purpose programming language.",
    "The human brain is the central organ of the human nervous system.",
    "Machine learning automates analytical model building from data.",
    "Quantum mechanics describes physical properties at the atomic scale.",
    "The Internet is a global system of interconnected computer networks.",
    "Neural networks are computing systems inspired by biological brains.",
    "The capital of France is Paris and its population is about 2 million.",
]


# ===========================================================================
# Score computation
# ===========================================================================

def compute_bound_scores_and_R(layer) -> Tuple[torch.Tensor, float]:
    """
    Compute rmsnorm_bound_angle scores S_i and the input-bound radius R.

        S_i = R^2 * ((||w_gate_i|| * ||w_up_i|| + |w_gate_i . w_up_i|) / 2) * ||w_down_i||
        R   = sqrt(d_model) * ||gamma||_inf

    Shapes confirmed: gate/up are [d_ff, d_model], down is [d_model, d_ff],
    so neuron i = gate[i,:], up[i,:], down[:,i] (column).

    Returns
    -------
    scores : Tensor [d_ff]  CPU float32
    R      : float
    """
    w      = get_mlp_weights(layer)
    w_gate = w["gate"].float().cpu()   # [d_ff, d_model]
    w_up   = w["up"].float().cpu()     # [d_ff, d_model]
    w_down = w["down"].float().cpu()   # [d_model, d_ff]
    d_ff   = w["d_ff"]
    d_model = w["d_model"]

    gate_row_norm = w_gate.norm(dim=1)        # [d_ff]  ||w_gate_i||
    up_row_norm   = w_up.norm(dim=1)          # [d_ff]  ||w_up_i||
    down_col_norm = w_down.norm(dim=0)        # [d_ff]  ||w_down_i|| (column norms)

    assert down_col_norm.shape == (d_ff,), (
        f"down_col_norm shape {down_col_norm.shape} != ({d_ff},) "
        "-- check that norm is taken over dim=0 (d_model axis)"
    )

    rmsnorm   = get_rmsnorm_before_mlp(layer)
    gamma     = rmsnorm.weight.detach().float().cpu()  # [d_model]
    gamma_inf = float(gamma.abs().max())
    R         = math.sqrt(d_model) * gamma_inf
    R_sq      = float(d_model) * gamma_inf ** 2

    # |w_gate_i . w_up_i| via elementwise multiply then sum
    dot_gate_up = (w_gate * w_up).sum(dim=1).abs()   # [d_ff]
    mixed_term  = (gate_row_norm * up_row_norm + dot_gate_up) / 2.0
    scores      = R_sq * mixed_term * down_col_norm   # [d_ff]

    assert scores.shape == (d_ff,), f"scores shape {scores.shape} != ({d_ff},)"
    return scores, R


# ===========================================================================
# Distribution analysis
# ===========================================================================

def analyze_score_distribution(
    scores: torch.Tensor,
    layer_idx: int,
    R: float,
    d_ff: int,
    d_model: int,
) -> dict:
    """
    Compute full distribution statistics for one layer's bound scores.
    Returns a JSON-serialisable dict.
    """
    s      = scores.float()
    median = float(s.median())

    result: dict = {
        "layer_idx":    layer_idx,
        "d_model":      d_model,
        "d_ff":         d_ff,
        "R":            round(R, 6),
        "R_squared":    round(R ** 2, 4),
        "score_min":    float(s.min()),
        "score_max":    float(s.max()),
        "score_mean":   float(s.mean()),
        "score_median": median,
        "score_std":    float(s.std()),
        "score_sum":    float(s.sum()),
        "n_exactly_zero": int((s == 0).sum()),
    }

    for pct in [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0]:
        result[f"p{pct}"] = float(torch.quantile(s, pct / 100.0))

    # Absolute threshold counts
    result["below_abs"] = {}
    for thr in ABS_THRESHOLDS:
        n = int((s < thr).sum())
        result["below_abs"][_k(thr)] = {"n": n, "pct": round(100.0 * n / d_ff, 4)}

    # Relative threshold counts (score_i / median < threshold)
    result["below_rel"] = {}
    if median > 0:
        s_rel = s / median
        for thr in REL_THRESHOLDS:
            n = int((s_rel < thr).sum())
            result["below_rel"][_k(thr)] = {"n": n, "pct": round(100.0 * n / d_ff, 4)}
    else:
        for thr in REL_THRESHOLDS:
            result["below_rel"][_k(thr)] = {"n": 0, "pct": 0.0}

    # Cumulative budget counts (ref = total score sum)
    sorted_s = s.sort().values     # ascending
    cumsum   = sorted_s.cumsum(0)
    total    = float(s.sum())
    result["cumulative_budget_score_sum"] = {}
    for alpha in ALPHA_VALUES:
        budget   = alpha * total
        n_pruned = int((cumsum <= budget).sum())
        result["cumulative_budget_score_sum"][_k(alpha)] = {
            "budget":   round(budget, 8),
            "n_pruned": n_pruned,
            "pct":      round(100.0 * n_pruned / d_ff, 4),
        }

    return result


def run_distribution_analysis(model) -> List[dict]:
    """
    Compute and print score distribution for every transformer layer.
    Pure weight-based analysis; no forward passes needed.
    """
    layers = get_transformer_layers(model)

    print(f"\n{'=' * 80}")
    print("BOUND SCORE DISTRIBUTION (weight-only, no calibration)")
    print(f"{'=' * 80}")
    print(f"  {'Layer':>5}  {'d_ff':>5}  {'R':>8}  {'min':>10}  {'median':>10}  {'max':>10}")
    print(f"  {'-' * 60}")

    results = []
    for i, layer in enumerate(layers):
        scores, R = compute_bound_scores_and_R(layer)
        w         = get_mlp_weights(layer)
        stats     = analyze_score_distribution(scores, i, R, w["d_ff"], w["d_model"])
        results.append(stats)
        print(
            f"  {i:>5}  {stats['d_ff']:>5}  {R:>8.3f}"
            f"  {stats['score_min']:>10.3e}"
            f"  {stats['score_median']:>10.3e}"
            f"  {stats['score_max']:>10.3e}"
        )

    # Relative threshold table
    print(f"\n  Neurons below relative threshold (score / median < t):")
    print(f"  {'Layer':>5}  {'d_ff':>5}", end="")
    for thr in REL_THRESHOLDS:
        print(f"  {_k(thr):>10}", end="")
    print()
    print(f"  {'-' * 70}")
    for s in results:
        print(f"  {s['layer_idx']:>5}  {s['d_ff']:>5}", end="")
        for thr in REL_THRESHOLDS:
            cnt = s["below_rel"].get(_k(thr), {}).get("n", 0)
            print(f"  {cnt:>10}", end="")
        print()

    # Absolute threshold table
    print(f"\n  Neurons below absolute threshold:")
    print(f"  {'Layer':>5}  {'d_ff':>5}", end="")
    for thr in ABS_THRESHOLDS:
        print(f"  {_k(thr):>10}", end="")
    print()
    print(f"  {'-' * 90}")
    for s in results:
        print(f"  {s['layer_idx']:>5}  {s['d_ff']:>5}", end="")
        for thr in ABS_THRESHOLDS:
            cnt = s["below_abs"].get(_k(thr), {}).get("n", 0)
            print(f"  {cnt:>10}", end="")
        print()

    # Cumulative budget table (score_sum reference)
    print(f"\n  Cumulative budget: prune smallest until sum(pruned) <= alpha * sum(all):")
    print(f"  {'Layer':>5}  {'d_ff':>5}", end="")
    for alpha in ALPHA_VALUES:
        print(f"  {_k(alpha):>10}", end="")
    print()
    print(f"  {'-' * 70}")
    for s in results:
        print(f"  {s['layer_idx']:>5}  {s['d_ff']:>5}", end="")
        for alpha in ALPHA_VALUES:
            cnt = s["cumulative_budget_score_sum"].get(_k(alpha), {}).get("n_pruned", 0)
            print(f"  {cnt:>10}", end="")
        print()

    print(f"{'=' * 80}\n")
    return results


# ===========================================================================
# Calibration: MLP output norms
# ===========================================================================

def compute_mlp_output_norms_all_layers(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_seq_len: int = 128,
) -> List[float]:
    """
    Compute mean ||MLP(r)||_2 per layer via forward hooks on calibration prompts.

    Uses register_forward_hook (3-arg signature: module, inp, out) — correct
    for post-hooks, which receive the output tensor as the third argument.

    Returns
    -------
    mean_norms : List[float]  one value per transformer layer
    """
    layers   = get_transformer_layers(model)
    n_layers = len(layers)
    captured = [[] for _ in range(n_layers)]
    handles  = []

    try:
        for idx, layer in enumerate(layers):
            def _make_hook(i):
                def _hook(module, inp, out):
                    norms = out.detach().float().norm(dim=-1)   # [B, T]
                    captured[i].append(float(norms.mean()))
                return _hook
            handles.append(get_mlp_module(layer).register_forward_hook(_make_hook(idx)))

        model.eval()
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="  Calibrating MLP norms", leave=False):
                enc = tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=max_seq_len).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()

    return [
        float(sum(vals) / len(vals)) if vals else 0.0
        for vals in captured
    ]


# ===========================================================================
# Neuron selection
# ===========================================================================

def select_by_absolute_threshold(
    scores: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Return indices where score_i < threshold."""
    return (scores < threshold).nonzero(as_tuple=True)[0]


def select_by_relative_threshold(
    scores: torch.Tensor, rel_threshold: float
) -> torch.Tensor:
    """Return indices where score_i / median < rel_threshold."""
    median = float(scores.median())
    if median <= 0:
        return torch.tensor([], dtype=torch.long)
    return ((scores / median) < rel_threshold).nonzero(as_tuple=True)[0]


def select_by_budget(
    scores: torch.Tensor,
    alpha: float,
    reference_norm: float,
) -> Tuple[torch.Tensor, float]:
    """
    Sort neurons ascending by S_i; prune the largest prefix P such that:
        sum_{j in P} S_j <= alpha * reference_norm

    Returns (prune_indices_sorted_ascending, budget_used).
    Returns (empty, 0.0) if budget <= 0 or no neuron fits in budget.
    """
    budget = alpha * reference_norm
    if budget <= 0 or scores.numel() == 0:
        return torch.tensor([], dtype=torch.long), 0.0

    sorted_vals, sorted_idx = scores.sort()    # ascending
    cumsum  = sorted_vals.cumsum(0)
    n_prune = int((cumsum <= budget).sum())

    if n_prune == 0:
        return torch.tensor([], dtype=torch.long), 0.0

    prune_idx   = sorted_idx[:n_prune].sort().values
    budget_used = float(sorted_vals[:n_prune].sum())
    return prune_idx, budget_used


# ===========================================================================
# Activation contribution verification
# ===========================================================================

def verify_activation_contributions(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    prune_indices_per_layer: List[torch.Tensor],
    max_seq_len: int = 128,
    chunk_size: int = 256,
) -> dict:
    """
    For the theoretically-pruned neurons, compute their actual average
    activation contribution on calibration data:

        actual_i = mean_t( |SiLU(r_t . w_gate_i) * (r_t . w_up_i)| ) * ||w_down_i||

    This answers: "Are the bound-certified neurons also empirically unimportant?"

    Hook notes
    ----------
    Uses register_forward_pre_hook, which calls hook(module, inputs) — 2 args.
    The try/finally ensures hooks are removed even if forward passes crash,
    preventing stale hooks from corrupting subsequent runs.

    Returns a dict with per-layer stats.
    """
    layers   = get_transformer_layers(model)
    n_layers = len(layers)
    captured = [[] for _ in range(n_layers)]
    handles  = []

    try:
        for idx, layer in enumerate(layers):
            def _make_hook(i):
                # register_forward_pre_hook: hook(module, inputs) — 2 args only
                def _hook(module, inputs):
                    captured[i].append(inputs[0].detach().float().cpu())
                return _hook
            handles.append(
                get_mlp_module(layer).register_forward_pre_hook(_make_hook(idx))
            )

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                enc = tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=max_seq_len).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()

    per_layer = []
    for i, layer in enumerate(layers):
        pi = prune_indices_per_layer[i]
        if not captured[i]:
            per_layer.append({"layer_idx": i, "n_pruned": 0, "note": "no captures"})
            continue

        all_r  = torch.cat([x.reshape(-1, x.shape[-1]) for x in captured[i]], dim=0)
        w      = get_mlp_weights(layer)
        w_gate = w["gate"].float().cpu()
        w_up   = w["up"].float().cpu()
        w_down = w["down"].float().cpu()
        d_ff   = w["d_ff"]

        sum_abs = torch.zeros(d_ff)
        n_tok   = 0
        for start in range(0, all_r.shape[0], chunk_size):
            chunk    = all_r[start : start + chunk_size]
            g        = chunk @ w_gate.T
            u        = chunk @ w_up.T
            a        = F.silu(g) * u
            sum_abs += a.abs().sum(dim=0)
            n_tok   += chunk.shape[0]

        mean_abs  = sum_abs / max(n_tok, 1)
        act_score = mean_abs * w_down.norm(dim=0)   # [d_ff]  actual contribution
        total_act = float(act_score.sum())

        if len(pi) == 0:
            per_layer.append({
                "layer_idx":       i,
                "n_pruned":        0,
                "n_total":         d_ff,
                "pruned_act_mean": 0.0,
                "pruned_act_max":  0.0,
                "pruned_act_sum":  0.0,
                "total_act_sum":   round(total_act, 6),
                "pruned_fraction": 0.0,
            })
            continue

        pruned_act = act_score[pi]
        per_layer.append({
            "layer_idx":       i,
            "n_pruned":        int(len(pi)),
            "n_total":         d_ff,
            "pruned_act_mean": round(float(pruned_act.mean()), 8),
            "pruned_act_max":  round(float(pruned_act.max()), 8),
            "pruned_act_sum":  round(float(pruned_act.sum()), 8),
            "total_act_sum":   round(total_act, 8),
            "pruned_fraction": round(float(pruned_act.sum() / max(total_act, 1e-30)), 8),
        })

    return {"per_layer": per_layer}


# ===========================================================================
# Experiment configuration builder
# ===========================================================================

def build_experiment_configs(
    all_scores: List[torch.Tensor],
    mlp_norms: List[float],
) -> List[dict]:
    """
    Build all (rule, label, prune_indices_per_layer) experiment dicts.
    Covers all four pruning modes across all threshold/alpha values.
    """
    n_layers  = len(all_scores)
    total_n   = sum(s.numel() for s in all_scores)

    def _exp(rule: str, label: str, indices: List[torch.Tensor]) -> dict:
        total = sum(len(pi) for pi in indices)
        return {
            "rule":                   rule,
            "label":                  label,
            "prune_indices_per_layer": indices,
            "total_pruned":           total,
            "total_neurons":          total_n,
            "pct_pruned":             round(100.0 * total / total_n, 4) if total_n else 0.0,
        }

    exps: List[dict] = []

    # A: Absolute threshold
    for thr in ABS_THRESHOLDS:
        indices = [select_by_absolute_threshold(all_scores[i], thr)
                   for i in range(n_layers)]
        exps.append(_exp("static_abs", f"abs<{_k(thr)}", indices))

    # B: Relative threshold
    for thr in REL_THRESHOLDS:
        indices = [select_by_relative_threshold(all_scores[i], thr)
                   for i in range(n_layers)]
        exps.append(_exp("static_rel", f"rel<{_k(thr)}", indices))

    # C: Cumulative budget vs total score sum (static reference)
    for alpha in ALPHA_VALUES:
        indices = []
        for i in range(n_layers):
            ref = float(all_scores[i].sum())
            pi, _ = select_by_budget(all_scores[i], alpha, ref)
            indices.append(pi)
        exps.append(_exp("cumul_score_sum", f"cumul_score_sum a={_k(alpha)}", indices))

    # D: Cumulative budget vs calibrated MLP output norm
    for alpha in ALPHA_VALUES:
        indices = []
        for i in range(n_layers):
            ref = mlp_norms[i] if i < len(mlp_norms) else 0.0
            pi, _ = select_by_budget(all_scores[i], alpha, ref)
            indices.append(pi)
        exps.append(_exp("cumul_mlp_norm", f"cumul_mlp_norm a={_k(alpha)}", indices))

    return exps


def print_pruning_count_table(
    exps: List[dict],
    all_scores: List[torch.Tensor],
) -> None:
    """Print how many neurons each experiment prunes, before running PPL."""
    n_layers = len(all_scores)

    print(f"\n{'=' * 82}")
    print("PRUNING COUNT PREVIEW  (answers: how many neurons are certified near-zero?)")
    print(f"{'=' * 82}")
    print(f"  {'Rule/Threshold':<42}  {'Total':>7}  {'%':>7}  Non-zero layers")
    print(f"  {'-' * 80}")

    for exp in exps:
        counts  = [len(pi) for pi in exp["prune_indices_per_layer"]]
        nonzero = ", ".join(
            f"L{i}:{counts[i]}" for i in range(n_layers) if counts[i] > 0
        ) or "none"
        print(
            f"  {exp['label']:<42}"
            f"  {exp['total_pruned']:>7}"
            f"  {exp['pct_pruned']:>6.3f}%"
            f"  {nonzero}"
        )

    n_nonzero = sum(1 for e in exps if e["total_pruned"] > 0)
    print(f"\n  {n_nonzero} / {len(exps)} configurations prune at least one neuron.")
    print(f"{'=' * 82}\n")


# ===========================================================================
# Shared serializer helper
# ===========================================================================

def _ser(obj):
    """Recursively convert tensors and non-JSON types to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ser(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    return str(obj)


# ===========================================================================
# Entry points
# ===========================================================================

def run_bound_analysis_mode(
    model,
    tokenizer,
    cfg: dict,
    device: str,
    output_dir: str = "results",
    skip_ppl: bool = False,
    skip_activation: bool = False,
) -> None:
    """
    Full bound analysis pipeline.

    Phase 1: Score distribution (weight-only, fast)
    Phase 2: Calibration — MLP output norms
    Phase 3: Pruning count preview (no PPL — answers "how many certified?")
    Phase 4: PPL experiments for configs that prune > 0 neurons
             (skipped when skip_ppl=True)
    Phase 5: Save JSON report

    Parameters
    ----------
    skip_ppl : bool
        Skip Phase 4 entirely. Pass --no-ppl on CLI.
    skip_activation : bool
        Within Phase 4, skip activation verification.
        PPL results are saved regardless.  Pass --no-activation-verification on CLI.

    INTERPRETATION
    ---------------
    If Phase 3 shows near-zero pruned counts for all configs:
        The bound is too conservative to certify pruning in this model.
        This is a valid scientific finding, not a code failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    layers = get_transformer_layers(model)

    # ── Phase 1: Distribution analysis ────────────────────────────────────────
    dist_results = run_distribution_analysis(model)

    # Pre-compute scores and R for all layers (reuse in later phases)
    all_scores: List[torch.Tensor] = []
    all_R:      List[float]        = []
    for layer in layers:
        s, R = compute_bound_scores_and_R(layer)
        all_scores.append(s)
        all_R.append(R)

    # ── Phase 2: Calibration ───────────────────────────────────────────────────
    print("Computing calibrated MLP output norms …")
    mlp_norms = compute_mlp_output_norms_all_layers(
        model, tokenizer,
        prompts=CALIBRATION_PROMPTS,
        device=device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )
    print(f"\n  Layer  E[||MLP(r)||_2]")
    print(f"  {'-' * 25}")
    for i, n in enumerate(mlp_norms):
        print(f"  {i:>5}  {n:>14.4f}")

    # ── Phase 3: Pruning count preview ────────────────────────────────────────
    exps = build_experiment_configs(all_scores, mlp_norms)
    print_pruning_count_table(exps, all_scores)

    baseline_ppl = None

    if skip_ppl:
        print("  [--no-ppl] Skipping PPL experiments.\n")
    else:
        # ── Phase 4: PPL experiments ───────────────────────────────────────────
        from .pruning import prune_model_by_layer_indices, verify_forward_pass
        from .evaluation import evaluate_perplexity, load_eval_dataset
        from .flops import estimate_mlp_flops
        from .model_utils import count_parameters

        use_fallback = cfg.get("use_fallback_corpus", True)
        n_eval       = cfg.get("bound_analysis_eval_samples", 64)
        eval_texts   = load_eval_dataset(n_eval, use_fallback_corpus=use_fallback)

        baseline_params = count_parameters(model)
        baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))
        nonzero_exps    = [e for e in exps if e["total_pruned"] > 0]

        if not nonzero_exps:
            print("  No configurations prune any neurons.")
            print("  Skipping PPL experiments.\n")
        else:
            print(f"\n{'=' * 82}")
            print(
                f"PPL EXPERIMENTS  ({len(nonzero_exps)} configs, "
                f"eval_samples={n_eval}"
                + ("  [no activation verification]" if skip_activation else "")
                + ")"
            )
            print(f"{'=' * 82}\n")

            for exp in nonzero_exps:
                # Lazy baseline (compute once)
                if baseline_ppl is None:
                    print("  Computing baseline PPL …")
                    bp = evaluate_perplexity(
                        model, tokenizer, texts=eval_texts,
                        max_seq_len=cfg.get("max_seq_len", 512),
                        batch_size=cfg.get("batch_size", 4),
                        device=device,
                    )
                    baseline_ppl = bp["perplexity"]
                    print(f"  Baseline PPL: {baseline_ppl:.4f}\n")

                label   = exp["label"]
                indices = exp["prune_indices_per_layer"]
                counts  = [len(pi) for pi in indices]
                print(
                    f"  [{label}]  total_pruned={exp['total_pruned']}"
                    f" ({exp['pct_pruned']:.3f}%)"
                )

                pruned_model = None
                try:
                    pruned_model, _ = prune_model_by_layer_indices(
                        model, indices, label=label
                    )
                    fp_ok    = verify_forward_pass(pruned_model, tokenizer, device)
                    ppl_info = evaluate_perplexity(
                        pruned_model, tokenizer, texts=eval_texts,
                        max_seq_len=cfg.get("max_seq_len", 512),
                        batch_size=cfg.get("batch_size", 4),
                        device=device,
                    )
                    ppl = ppl_info["perplexity"]

                    pruned_params = count_parameters(pruned_model)
                    pruned_flops  = estimate_mlp_flops(
                        pruned_model, seq_len=cfg.get("max_seq_len", 512)
                    )
                    flop_red = 100.0 * (
                        1.0 - pruned_flops["total_flops"] / baseline_flops["total_flops"]
                    )
                    mlp_red = 100.0 * (
                        1.0 - pruned_params["mlp"] / baseline_params["mlp"]
                    )

                    # Store PPL result immediately — activation verification is separate
                    exp["perplexity"]         = round(ppl, 4)
                    exp["perplexity_delta"]   = round(ppl - baseline_ppl, 4)
                    exp["baseline_ppl"]       = round(baseline_ppl, 4)
                    exp["forward_pass_ok"]    = fp_ok
                    exp["mlp_params_before"]  = baseline_params["mlp"]
                    exp["mlp_params_after"]   = pruned_params["mlp"]
                    exp["mlp_params_red_pct"] = round(mlp_red, 4)
                    exp["flops_red_pct"]      = round(flop_red, 4)

                    layer_detail = " ".join(
                        f"L{i}:{counts[i]}" for i in range(len(counts)) if counts[i] > 0
                    )
                    print(
                        f"    PPL={ppl:.4f}  dPPL={ppl - baseline_ppl:+.4f}"
                        f"  MLP-{mlp_red:.2f}%  FLOPs-{flop_red:.2f}%"
                    )
                    if layer_detail:
                        print(f"    Layers: {layer_detail}")

                    # Activation verification: diagnostic only — failure does NOT
                    # invalidate the PPL result already stored above.
                    if not skip_activation:
                        try:
                            act_info = verify_activation_contributions(
                                model, tokenizer,
                                prompts=CALIBRATION_PROMPTS,
                                device=device,
                                prune_indices_per_layer=indices,
                            )
                            exp["activation_verification"] = act_info
                        except Exception as act_exc:
                            logger.warning(
                                "Activation verification failed for [%s]: %s",
                                label, act_exc,
                            )
                            exp["activation_verification"] = {"error": str(act_exc)}

                except Exception as exc:
                    logger.error("Experiment [%s] failed: %s", label, exc, exc_info=True)
                    exp["notes"] = f"ERROR: {exc}"
                finally:
                    if pruned_model is not None:
                        del pruned_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # ── Summary table ──────────────────────────────────────────────────────
        ran = [e for e in exps if e.get("perplexity") is not None]
        if baseline_ppl is not None and ran:
            print(f"\n{'=' * 90}")
            print("BOUND ANALYSIS SUMMARY")
            print(f"  Baseline PPL: {baseline_ppl:.4f}")
            print(f"{'─' * 90}")
            print(
                f"  {'Label':<44}  {'Pruned':>7}  {'%':>7}"
                f"  {'PPL':>9}  {'dPPL':>9}  {'MLP-':>7}  {'Fl-':>7}"
            )
            print(f"{'─' * 90}")
            for e in ran:
                print(
                    f"  {e['label']:<44}"
                    f"  {e['total_pruned']:>7}"
                    f"  {e['pct_pruned']:>6.3f}%"
                    f"  {e['perplexity']:>9.4f}"
                    f"  {e['perplexity_delta']:>+9.4f}"
                    f"  {e.get('mlp_params_red_pct', 0.0):>6.2f}%"
                    f"  {e.get('flops_red_pct', 0.0):>6.2f}%"
                )
            print(f"{'=' * 90}\n")
        elif not nonzero_exps:
            print(
                "\n  KEY FINDING: The RMSNorm worst-case bound found no neurons below\n"
                "  any tested threshold in this model. This means the bound is too\n"
                "  conservative to certify the removal of any neuron purely from\n"
                "  weight norms — every neuron's worst-case contribution exceeds the\n"
                "  tested safety budgets. Consider using the activation-based score\n"
                "  (--debug-pruning) or a data-calibrated pruning approach.\n"
            )

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "timestamp":    ts,
        "mode":         "bound_analysis",
        "skip_ppl":     skip_ppl,
        "distribution": dist_results,
        "mlp_norms":    mlp_norms,
        "experiments":  _ser(exps),
        "baseline_ppl": baseline_ppl,
    }
    path = os.path.join(output_dir, f"bound_analysis_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Bound analysis report -> %s", path)
    print(f"Full report saved to: {path}\n")


# ---------------------------------------------------------------------------

def run_bound_ppl_mode(
    model,
    tokenizer,
    cfg: dict,
    device: str,
    output_dir: str = "results",
) -> None:
    """
    Focused PPL mode: only cumul_score_sum at alpha = 1e-4, 1e-3, 1e-2.
    No activation verification.  Extra detail saved for alpha=1e-4.

    Use this after --bound-analysis has already confirmed which alpha values
    actually prune neurons, and you want clean PPL numbers without the full
    distribution overhead.
    """
    from .pruning import prune_model_by_layer_indices, verify_forward_pass
    from .evaluation import evaluate_perplexity, load_eval_dataset, run_generation_tests
    from .flops import estimate_mlp_flops
    from .model_utils import count_parameters

    os.makedirs(output_dir, exist_ok=True)
    ts      = time.strftime("%Y%m%d_%H%M%S")
    layers  = get_transformer_layers(model)
    ALPHAS  = [1e-4, 1e-3, 1e-2]

    # Pre-compute scores
    print("\nComputing bound scores for all layers …")
    all_scores: List[torch.Tensor] = []
    for layer in layers:
        s, _ = compute_bound_scores_and_R(layer)
        all_scores.append(s)

    # Dataset
    use_fallback = cfg.get("use_fallback_corpus", True)
    n_eval       = cfg.get("bound_analysis_eval_samples", 64)
    eval_texts   = load_eval_dataset(n_eval, use_fallback_corpus=use_fallback)

    baseline_params = count_parameters(model)
    baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))

    print("  Computing baseline PPL …")
    bp           = evaluate_perplexity(
        model, tokenizer, texts=eval_texts,
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 4),
        device=device,
    )
    baseline_ppl = bp["perplexity"]
    print(f"  Baseline PPL: {baseline_ppl:.4f}\n")

    results = []

    for alpha in ALPHAS:
        # Select neurons via cumulative budget (reference = layer score sum)
        indices      = []
        budget_used_per_layer = []
        for i, scores in enumerate(all_scores):
            ref = float(scores.sum())
            pi, bu = select_by_budget(scores, alpha, ref)
            indices.append(pi)
            budget_used_per_layer.append(bu)

        total_pruned = sum(len(pi) for pi in indices)
        label        = f"cumul_score_sum a={_k(alpha)}"

        if total_pruned == 0:
            print(f"  [{label}] 0 neurons pruned — skipping PPL")
            results.append({
                "alpha": alpha,
                "label": label,
                "total_pruned": 0,
                "total_neurons": sum(s.numel() for s in all_scores),
                "pct_pruned": 0.0,
                "notes": "0 neurons pruned, PPL skipped",
            })
            continue

        pct_pruned = 100.0 * total_pruned / sum(s.numel() for s in all_scores)
        print(f"  [{label}]  pruned={total_pruned} ({pct_pruned:.3f}%)")

        pruned_model = None
        row = {
            "alpha":         alpha,
            "label":         label,
            "total_pruned":  total_pruned,
            "total_neurons": sum(s.numel() for s in all_scores),
            "pct_pruned":    round(pct_pruned, 4),
        }

        try:
            pruned_model, _ = prune_model_by_layer_indices(
                model, indices, label=label
            )
            fp_ok    = verify_forward_pass(pruned_model, tokenizer, device)
            ppl_info = evaluate_perplexity(
                pruned_model, tokenizer, texts=eval_texts,
                max_seq_len=cfg.get("max_seq_len", 512),
                batch_size=cfg.get("batch_size", 4),
                device=device,
            )
            ppl = ppl_info["perplexity"]

            pruned_params = count_parameters(pruned_model)
            pruned_flops  = estimate_mlp_flops(
                pruned_model, seq_len=cfg.get("max_seq_len", 512)
            )
            flop_red = 100.0 * (
                1.0 - pruned_flops["total_flops"] / baseline_flops["total_flops"]
            )
            mlp_red = 100.0 * (
                1.0 - pruned_params["mlp"] / baseline_params["mlp"]
            )

            row.update({
                "baseline_ppl":       round(baseline_ppl, 4),
                "perplexity":         round(ppl, 4),
                "perplexity_delta":   round(ppl - baseline_ppl, 4),
                "forward_pass_ok":    fp_ok,
                "mlp_params_red_pct": round(mlp_red, 4),
                "flops_red_pct":      round(flop_red, 4),
            })
            print(
                f"    PPL={ppl:.4f}  dPPL={ppl - baseline_ppl:+.4f}"
                f"  MLP-{mlp_red:.2f}%  FLOPs-{flop_red:.2f}%"
            )

            # ── Extra detail for alpha=1e-4 ────────────────────────────────────
            if abs(alpha - 1e-4) < 1e-10 and total_pruned > 0:
                print("  [alpha=1e-4] Saving detailed pruning log …")

                # Generation examples before pruning (use original model)
                gens_before = run_generation_tests(model, tokenizer, device=device)

                # Generation examples after pruning
                gens_after = run_generation_tests(pruned_model, tokenizer, device=device)

                detail: dict = {
                    "alpha":          alpha,
                    "label":          label,
                    "baseline_ppl":   round(baseline_ppl, 4),
                    "pruned_ppl":     round(ppl, 4),
                    "ppl_delta":      round(ppl - baseline_ppl, 4),
                    "total_pruned":   total_pruned,
                    "pct_pruned":     round(pct_pruned, 4),
                    "per_layer": [],
                    "generation_before": gens_before,
                    "generation_after":  gens_after,
                }
                for i, (pi, bu) in enumerate(
                    zip(indices, budget_used_per_layer)
                ):
                    if len(pi) == 0:
                        detail["per_layer"].append({
                            "layer_idx": i,
                            "n_pruned":  0,
                        })
                        continue
                    pruned_scores = all_scores[i][pi].tolist()
                    detail["per_layer"].append({
                        "layer_idx":              i,
                        "n_pruned":               int(len(pi)),
                        "n_total":                int(all_scores[i].numel()),
                        "pruned_neuron_indices":  pi.tolist(),
                        "pruned_neuron_scores":   [round(v, 8) for v in pruned_scores],
                        "cumul_removed_bound":    round(bu, 8),
                        "layer_score_sum":        round(float(all_scores[i].sum()), 8),
                        "fraction_of_layer_sum":  round(bu / max(float(all_scores[i].sum()), 1e-30), 8),
                    })

                detail_path = os.path.join(
                    output_dir, f"bound_ppl_detail_alpha1e-04_{ts}.json"
                )
                with open(detail_path, "w") as f:
                    json.dump(_ser(detail), f, indent=2)
                logger.info("Alpha=1e-4 detail log -> %s", detail_path)
                print(f"    Detail log: {detail_path}")

        except Exception as exc:
            logger.error("PPL run [%s] failed: %s", label, exc, exc_info=True)
            row["notes"] = f"ERROR: {exc}"
        finally:
            if pruned_model is not None:
                del pruned_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(row)

    # Summary
    print(f"\n{'=' * 70}")
    print("BOUND PPL MODE SUMMARY")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'─' * 70}")
    for r in results:
        if "perplexity" in r:
            print(
                f"  alpha={_k(r['alpha'])}  pruned={r['total_pruned']} ({r['pct_pruned']:.3f}%)"
                f"  PPL={r['perplexity']:.4f}  dPPL={r['perplexity_delta']:+.4f}"
            )
        else:
            print(
                f"  alpha={_k(r['alpha'])}  {r.get('notes', 'skipped')}"
            )
    print(f"{'=' * 70}\n")

    # Save
    report = {
        "timestamp":    ts,
        "mode":         "bound_ppl_only",
        "baseline_ppl": baseline_ppl,
        "results":      results,
    }
    path = os.path.join(output_dir, f"bound_ppl_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Bound PPL report -> %s", path)
    print(f"Report saved to: {path}\n")


# ---------------------------------------------------------------------------

def run_activation_verification_mode(
    model,
    tokenizer,
    cfg: dict,
    device: str,
    output_dir: str = "results",
) -> None:
    """
    Compute activation-based neuron scores and compare them to the static
    RMSNorm bound scores.  No pruning is performed.

    Reports per-layer Pearson and Spearman correlation between bound scores
    and actual activation contributions, plus a bottom-20% rank overlap.

    High correlation would mean the static bound score is a good proxy for
    empirical importance. Low correlation means calibration data is needed.
    """
    from .scoring import (
        compute_activation_scores_all_layers,
        pearson_corr,
        spearman_corr,
    )

    os.makedirs(output_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    layers = get_transformer_layers(model)

    print(f"\n{'=' * 70}")
    print("ACTIVATION VERIFICATION MODE")
    print("Comparing static bound scores to calibration-data activation scores")
    print(f"{'=' * 70}\n")

    # Compute activation scores (uses fixed 2-arg pre-hook with try/finally)
    print("Computing activation-based scores via calibration data …")
    act_scores_per_layer = compute_activation_scores_all_layers(
        model, tokenizer,
        prompts=CALIBRATION_PROMPTS,
        device=device,
        max_seq_len=cfg.get("max_seq_len", 128),
    )

    # Compute static bound scores
    print("Computing static bound scores …")
    bound_scores_per_layer: List[torch.Tensor] = []
    for layer in layers:
        s, _ = compute_bound_scores_and_R(layer)
        bound_scores_per_layer.append(s)

    # Per-layer correlation analysis
    print(f"\n  {'Layer':>5}  {'d_ff':>5}  {'Pearson':>9}  {'Spearman':>10}  {'Bot-20% overlap':>16}")
    print(f"  {'-' * 60}")

    correlations = []
    for i, (bound_s, act_s) in enumerate(
        zip(bound_scores_per_layer, act_scores_per_layer)
    ):
        d_ff = bound_s.numel()
        if d_ff == 0 or act_s.numel() == 0:
            continue

        # Align lengths (act_s might be shorter if layer had no captures)
        min_len = min(d_ff, act_s.numel())
        b = bound_s[:min_len].float().cpu()
        a = act_s[:min_len].float().cpu()

        p = pearson_corr(b, a)
        s = spearman_corr(b, a)

        # Rank overlap: bottom 20% by bound vs bottom 20% by activation
        k = max(1, min_len // 5)
        bottom_bound = set(torch.argsort(b)[:k].tolist())
        bottom_act   = set(torch.argsort(a)[:k].tolist())
        overlap      = len(bottom_bound & bottom_act) / k

        correlations.append({
            "layer_idx":         i,
            "d_ff":              d_ff,
            "pearson":           round(p, 4),
            "spearman":          round(s, 4),
            "bottom20_overlap":  round(overlap, 4),
        })
        print(
            f"  {i:>5}  {d_ff:>5}"
            f"  {p:>9.4f}  {s:>10.4f}  {overlap:>16.4f}"
        )

    if correlations:
        n = len(correlations)
        mean_p = sum(c["pearson"] for c in correlations) / n
        mean_s = sum(c["spearman"] for c in correlations) / n
        mean_o = sum(c["bottom20_overlap"] for c in correlations) / n
        print(f"  {'-' * 60}")
        print(
            f"  {'Mean':>5}         "
            f"  {mean_p:>9.4f}  {mean_s:>10.4f}  {mean_o:>16.4f}"
        )
        print(f"\n  Interpretation:")
        print(f"    Spearman ≥ 0.8  → bound score is a good proxy for activation importance")
        print(f"    Spearman < 0.5  → calibration data is required for reliable pruning")
        print(f"    Bot-20% overlap → fraction of lowest-bound neurons also lowest-activation")

    summary = {
        "mean_pearson":         round(mean_p, 4) if correlations else None,
        "mean_spearman":        round(mean_s, 4) if correlations else None,
        "mean_bottom20_overlap": round(mean_o, 4) if correlations else None,
    }

    # Save
    report = {
        "timestamp":    ts,
        "mode":         "activation_verification_only",
        "correlations": correlations,
        "summary":      summary,
    }
    path = os.path.join(output_dir, f"activation_verification_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Activation verification report -> %s", path)
    print(f"\nReport saved to: {path}\n")
