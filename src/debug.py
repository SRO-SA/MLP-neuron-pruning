"""
debug.py
========
Full debug suite for diagnosing why static pruning scores underperform
random pruning.

Run via:
    python run_experiment.py --config configs/default.yaml --debug-pruning

═══════════════════════════════════════════════════════════════════════════
Eight checks performed
═══════════════════════════════════════════════════════════════════════════

[1] TENSOR SHAPE VALIDATION
    Confirms gate_proj/up_proj/down_proj have the expected shapes.
    What a violation means: the down_proj column / gate_proj row indexing
    is based on shapes [d_model, d_ff] and [d_ff, d_model] respectively.
    If they are transposed, our pruning is removing the wrong weights.

[2] PRUNING DIRECTION CHECK
    For each scoring method, verifies that pruned neurons have LOWER
    scores than kept neurons.
    What "inverted direction" means: a bug in get_keep_indices() is
    removing the IMPORTANT neurons instead of the unimportant ones.

[3] ZERO-MASK VS PHYSICAL PRUNING EQUIVALENCE
    Sets the pruned neurons' weights to zero (without changing shapes),
    then physically removes them, and compares logits.
    What a large difference means: prune_layer_mlp() removes the WRONG
    rows/columns — e.g., down_proj rows instead of columns.

[4] MODEL INDEPENDENCE
    Verifies the original model's parameter count is unchanged after each
    prune_model() call.  Rules out cumulative/in-place pruning bugs.

[5] TINY PRUNING RATIOS (0.1% – 2%)
    If even 0.1% pruning causes large PPL spikes with designed scores but
    not with random, the scoring is either inverted or the top-ranked
    "unimportant" neurons happen to be critical.

[6] SCORE CORRELATION ANALYSIS
    Computes Pearson and Spearman correlations between:
        down_norm, product_norm, rmsnorm_bound_angle, activation_score
    for a sample of layers.
    If rmsnorm_bound_angle has LOW or NEGATIVE correlation with
    activation_score, the theoretical score does not track actual
    neuron importance.

[7] PER-LAYER PRUNING SENSITIVITY
    Prunes ONE layer at a time at 1% and measures ΔPP.
    Reveals whether some layers are disproportionately sensitive.

[8] PERPLEXITY SANITY CHECK
    Evaluates the unmodified model on a tiny, fixed, padding-free text.
    Expected PPL for Qwen2.5-0.5B ≈ 10–30.
    High PPL here indicates a broken evaluation pipeline.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .evaluation import evaluate_on_fixed_text, evaluate_perplexity, load_eval_dataset
from .model_utils import (
    clone_model,
    count_parameters,
    get_mlp_weights,
    get_transformer_layers,
)
from .pruning import compare_zero_mask_vs_physical, prune_layer_mlp, prune_model
from .scoring import (
    ALL_STATIC_METHODS,
    compute_activation_scores_all_layers,
    compute_score_correlations,
    compute_scores,
    get_keep_indices,
    get_prune_indices,
    pearson_corr,
    spearman_corr,
)

logger = logging.getLogger(__name__)

# Calibration prompts for activation scoring
CALIBRATION_PROMPTS = [
    "The transformer architecture was introduced in 2017.",
    "Python is a high-level programming language.",
    "The speed of light is approximately 300,000 km/s.",
    "Machine learning is a subset of artificial intelligence.",
    "The human brain contains approximately 86 billion neurons.",
    "Water molecules consist of two hydrogen atoms and one oxygen atom.",
    "The French Revolution began in 1789 and ended in 1799.",
    "Neural networks learn by adjusting their weights through backpropagation.",
    "The Earth orbits the Sun at an average distance of 150 million kilometres.",
    "Quantum computers use qubits instead of classical bits.",
    "The periodic table organizes elements by atomic number.",
    "DNA encodes genetic information using four nucleotide bases.",
]

SECTION = "═" * 68
DIVIDER = "─" * 68


def _hdr(n: int, title: str) -> None:
    print(f"\n{SECTION}")
    print(f"[{n}] {title}")
    print(SECTION)


# ===========================================================================
# [1] TENSOR SHAPE VALIDATION
# ===========================================================================

def check_tensor_shapes(model) -> Dict[str, Any]:
    """
    Verify that all MLP weight tensors have the expected shapes.

    Expected (Qwen2.5 / LLaMA style):
        gate_proj.weight : [d_ff,    d_model]
        up_proj.weight   : [d_ff,    d_model]
        down_proj.weight : [d_model, d_ff]

    Neuron i = gate_proj[i,:], up_proj[i,:], down_proj[:,i]  ← column of down!
    """
    _hdr(1, "TENSOR SHAPE VALIDATION")

    cfg     = model.config
    d_model = getattr(cfg, "hidden_size", None)
    d_ff    = getattr(cfg, "intermediate_size", None)
    layers  = get_transformer_layers(model)

    print(f"  Config: hidden_size={d_model}, intermediate_size={d_ff}")
    print(f"  Layers: {len(layers)}\n")
    print(f"  {'Layer':>5}  {'gate_proj':>18}  {'up_proj':>18}  {'down_proj':>18}  {'Status':>10}")
    print(f"  {DIVIDER}")

    violations = {}

    for i, layer in enumerate(layers):
        w         = get_mlp_weights(layer)
        g_shape   = tuple(w["gate"].shape)
        u_shape   = tuple(w["up"].shape)
        d_shape   = tuple(w["down"].shape)
        d_m       = w["d_model"]
        d_f       = w["d_ff"]

        # Check: gate & up should be [d_ff, d_model]; down should be [d_model, d_ff]
        ok  = (g_shape[0] == d_f and g_shape[1] == d_m)
        ok &= (u_shape[0] == d_f and u_shape[1] == d_m)
        ok &= (d_shape[0] == d_m and d_shape[1] == d_f)

        # Also check the key property: down_proj is NOT transposed
        down_rows_eq_d_model = (d_shape[0] == d_m)
        down_cols_eq_d_ff    = (d_shape[1] == d_f)

        status = "✓" if ok else "✗ VIOLATION"

        if i == 0 or not ok:
            print(f"  {i:>5}  {str(g_shape):>18}  {str(u_shape):>18}  "
                  f"{str(d_shape):>18}  {status:>10}")
            if not down_rows_eq_d_model:
                print(f"         ⚠  down_proj.shape[0]={d_shape[0]} ≠ d_model={d_m}")
                print(f"            This means down_proj is TRANSPOSED vs expected!")
                print(f"            Neuron i would map to ROW i, not COLUMN i.")
                violations[f"layer_{i}_down_transposed"] = d_shape

        if not ok:
            violations[f"layer_{i}"] = {
                "gate": g_shape, "up": u_shape, "down": d_shape
            }

    if not violations:
        print(f"\n  ✓ All {len(layers)} layers have correct shapes.")
    else:
        print(f"\n  ✗ {len(violations)} violation(s) found: {violations}")

    return violations


# ===========================================================================
# [2] PRUNING DIRECTION CHECK
# ===========================================================================

def check_pruning_direction(
    model,
    methods: List[str],
    prune_ratio: float = 0.05,
    n_layers_to_check: int = 3,
) -> Dict[str, Any]:
    """
    For each method, verify that pruned neurons have LOWER scores than kept ones.

    Interpretation
    --------------
    ✓ CORRECT   →  max(pruned scores) ≤ min(kept scores)
                   The scoring and index selection are working as intended.
    ✗ INVERTED  →  pruned scores > kept scores
                   get_keep_indices() is keeping the WRONG end of the sorted array.
    ⚠ OVERLAP   →  pruned and kept score ranges overlap
                   Possible for methods with ties, but generally fine.
    """
    _hdr(2, "PRUNING DIRECTION CHECK")
    print(f"  Prune ratio: {prune_ratio:.1%}\n")

    layers     = get_transformer_layers(model)
    n          = len(layers)
    check_idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))[:n_layers_to_check]

    results = {}

    for method in methods:
        print(f"  ┌── Method: {method} {'─'*(50-len(method))}")
        all_ok = True

        for layer_idx in check_idxs:
            layer  = layers[layer_idx]
            scores = compute_scores(layer, method)
            d_ff   = scores.numel()

            keep_idx   = get_keep_indices(scores, prune_ratio)
            prune_idx  = get_prune_indices(scores, prune_ratio)

            if prune_idx.numel() == 0:
                print(f"  │  Layer {layer_idx:2d}: nothing to prune at this ratio")
                continue

            pruned_s = scores[prune_idx].sort().values
            kept_s   = scores[keep_idx].sort().values

            p_max = pruned_s.max().item()
            k_min = kept_s.min().item()

            if p_max <= k_min:
                direction = "✓ CORRECT"
            elif p_max <= k_min * 1.001:
                direction = "⚠ MARGINAL"
            else:
                direction = "✗ INVERTED!"
                all_ok = False

            print(f"  │  Layer {layer_idx:2d}: "
                  f"score min={scores.min():.4f} max={scores.max():.4f} "
                  f"mean={scores.mean():.4f}")
            print(f"  │           pruned top-5={[f'{x:.4f}' for x in pruned_s[:5].tolist()]}  "
                  f"max_pruned={p_max:.4f}")
            print(f"  │           kept   bot-5={[f'{x:.4f}' for x in kept_s[:5].tolist()]}   "
                  f"min_kept={k_min:.4f}")
            print(f"  │           → Direction: {direction}")

        print(f"  └── {'All layers OK' if all_ok else 'DIRECTION BUG DETECTED'}")
        results[method] = {"all_ok": all_ok}
        print()

    return results


# ===========================================================================
# [3] ZERO-MASK VS PHYSICAL PRUNING EQUIVALENCE
# ===========================================================================

def run_zero_mask_equivalence_test(
    model,
    tokenizer,
    device: str,
    methods: List[str],
    prune_ratio: float = 0.05,
    layer_idx: int = 0,
) -> Dict[str, Any]:
    """
    For each method, compare zero-masked logits vs physically pruned logits.

    If max_logit_diff is near 0 for all methods:
        → Physical pruning is correct.  The problem is in the SCORING.

    If max_logit_diff is large for designed methods but small for random:
        → Bug in prune_layer_mlp() — wrong row/column is being sliced.

    If max_logit_diff is large for ALL methods including random:
        → Systematic bug in prune_layer_mlp() itself.
    """
    _hdr(3, "ZERO-MASK VS PHYSICAL PRUNING EQUIVALENCE")
    print(f"  Layer: {layer_idx},  Prune ratio: {prune_ratio:.1%}\n")
    print(f"  {'Method':<25}  {'n_pruned':>8}  {'max_logit_diff':>16}  "
          f"{'max_mlp_diff':>14}  {'Status':>12}")
    print(f"  {DIVIDER}")

    results = {}

    for method in methods:
        res = compare_zero_mask_vs_physical(
            model, tokenizer, device, method, prune_ratio, layer_idx
        )
        status = "✓ CONSISTENT" if res["is_consistent"] else "✗ MISMATCH!"
        print(
            f"  {method:<25}  {res['n_pruned']:>8}  "
            f"{res['max_logit_diff']:>16.6f}  "
            f"{res['max_mlp_layer_diff']:>14.6f}  "
            f"{status:>12}"
        )
        results[method] = res

    print()
    n_bug = sum(1 for r in results.values() if not r["is_consistent"])
    if n_bug == 0:
        print("  ✓ All methods: physical pruning matches zero-masking.")
        print("    → Physical pruning code is CORRECT.")
        print("    → The PPL degradation must be due to the SCORING FUNCTION.")
    else:
        print(f"  ✗ {n_bug} method(s) show mismatch — BUG in physical pruning.")
        print("    Check: down_proj[:,i] vs down_proj[i,:]  (column vs row)")
        print("    Check: gate/up row slicing vs column slicing")

    return results


# ===========================================================================
# [4] MODEL INDEPENDENCE
# ===========================================================================

def check_model_independence(
    model,
    methods: List[str],
    prune_ratio: float = 0.05,
) -> bool:
    """
    Verify that prune_model() never modifies the original model.
    Checks parameter count before and after each call.

    If the original model changes across calls, experiments are cumulative
    (e.g. 10% would prune an already-5%-pruned model, causing 14.75% effective
    pruning and misleadingly bad results).
    """
    _hdr(4, "MODEL INDEPENDENCE CHECK")

    orig_params = count_parameters(model)["total"]
    print(f"  Original model: {orig_params:,} total parameters\n")
    print(f"  {'Run':<30}  {'orig_before':>14}  {'orig_after':>14}  {'pruned':>14}  {'OK':>6}")
    print(f"  {DIVIDER}")

    all_ok = True

    for method in methods:
        before_params = count_parameters(model)["total"]
        pruned_model, _ = prune_model(model, prune_ratio, method)
        after_params  = count_parameters(model)["total"]
        pruned_params = count_parameters(pruned_model)["total"]

        unchanged = (before_params == orig_params and after_params == orig_params)
        status    = "✓" if unchanged else "✗ MUTATED!"

        print(
            f"  {f'{method} r={prune_ratio:.0%}':<30}  "
            f"{before_params:>14,}  {after_params:>14,}  "
            f"{pruned_params:>14,}  {status:>6}"
        )

        if not unchanged:
            all_ok = False
            print(f"    ⚠  Original model changed from {before_params:,} to {after_params:,}!")

        del pruned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    if all_ok:
        print("  ✓ Original model unchanged across all runs.")
    else:
        print("  ✗ Original model was MUTATED — results are NOT independent!")
        print("    Check clone_model() and prune_layer_mlp() for in-place ops on original.")

    return all_ok


# ===========================================================================
# [5] TINY PRUNING RATIOS
# ===========================================================================

def test_tiny_pruning_ratios(
    model,
    tokenizer,
    methods: List[str],
    eval_texts: List[str],
    device: str,
    ratios: Optional[List[float]] = None,
    max_seq_len: int = 256,
    batch_size: int = 4,
    n_eval_samples: int = 64,
) -> List[Dict]:
    """
    Test very small pruning ratios to find the threshold at which each method
    diverges from random.

    If rmsnorm_bound_angle causes large PPL spikes even at 0.1% but random
    does not, the score is identifying and removing CRITICAL neurons —
    either the score is inverted or the top-ranked neurons happen to be outliers.
    """
    _hdr(5, "TINY PRUNING RATIOS")

    if ratios is None:
        ratios = [0.001, 0.005, 0.01, 0.02, 0.05]

    texts = eval_texts[:n_eval_samples]

    # Baseline
    print("  Computing baseline perplexity …")
    baseline = evaluate_perplexity(
        model, tokenizer, texts=texts,
        max_seq_len=max_seq_len, batch_size=batch_size, device=device,
    )
    base_ppl = baseline["perplexity"]
    print(f"  Baseline PPL: {base_ppl:.4f}\n")

    print(f"  {'Method':<25}  {'Ratio':>6}  {'PPL':>8}  {'ΔPPL':>8}  {'Ratio':>8}")
    print(f"  {DIVIDER}")

    rows = []
    for method in methods:
        for ratio in ratios:
            pruned, _ = prune_model(model, ratio, method)
            res = evaluate_perplexity(
                pruned, tokenizer, texts=texts,
                max_seq_len=max_seq_len, batch_size=batch_size, device=device,
            )
            ppl   = res["perplexity"]
            delta = ppl - base_ppl
            rel   = ppl / base_ppl

            print(f"  {method:<25}  {ratio:>6.3f}  {ppl:>8.3f}  {delta:>+8.3f}  {rel:>7.2f}×")

            rows.append({
                "method":       method,
                "prune_ratio":  ratio,
                "baseline_ppl": base_ppl,
                "perplexity":   ppl,
                "delta_ppl":    delta,
                "relative_ppl": rel,
            })

            del pruned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    return rows


# ===========================================================================
# [6] SCORE CORRELATION ANALYSIS
# ===========================================================================

def run_score_correlation_analysis(
    model,
    tokenizer,
    device: str,
    n_layers_to_check: int = 6,
) -> Dict[str, Any]:
    """
    For a sample of layers, compute pairwise Pearson and Spearman correlations
    between: down_norm, product_norm, rmsnorm_bound_angle, activation_score.

    Key question: Does rmsnorm_bound_angle correlate with activation_score?

    Interpretation
    --------------
    Spearman(rmsnorm_bound_angle, activation_score) > 0.7
        → Static score is a reasonable proxy for actual importance.
          Poor pruning performance might be due to other factors (e.g., outlier neurons).

    Spearman(rmsnorm_bound_angle, activation_score) ≈ 0 or < 0.3
        → Static score does NOT track actual neuron importance.
          The ranking is nearly random or wrong → explains why random pruning wins.

    Negative Spearman
        → Static score is INVERTED — it's calling important neurons unimportant!
    """
    _hdr(6, "SCORE CORRELATION ANALYSIS")

    layers  = get_transformer_layers(model)
    n       = len(layers)
    # Sample evenly across layers
    idxs    = [int(round(i * (n - 1) / max(n_layers_to_check - 1, 1)))
               for i in range(n_layers_to_check)]
    idxs    = sorted(set(idxs))

    print(f"  Computing activation scores on {len(CALIBRATION_PROMPTS)} calibration prompts …")
    act_scores = compute_activation_scores_all_layers(
        model, tokenizer, CALIBRATION_PROMPTS, device=device, max_seq_len=128
    )
    print()

    all_corrs = {}

    for li in idxs:
        layer = layers[li]
        corr  = compute_score_correlations(layer, activation_scores=act_scores[li])
        all_corrs[li] = corr

        print(f"  Layer {li:2d}:")
        pairs = [
            ("down_norm",           "activation"),
            ("product_norm",        "activation"),
            ("rmsnorm_bound_angle", "activation"),
            ("rmsnorm_bound_angle", "down_norm"),
            ("rmsnorm_bound_angle", "product_norm"),
        ]
        for a, b in pairs:
            if a in corr and b in corr[a]:
                p = corr[a][b]["pearson"]
                s = corr[a][b]["spearman"]
                flag = ""
                if abs(s) < 0.3:
                    flag = "  ⚠ WEAK"
                elif s < 0:
                    flag = "  ✗ NEGATIVE!"
                print(f"    {a:25s} ↔ {b:25s}  Pearson={p:+.3f}  Spearman={s:+.3f}{flag}")
        print()

    # Overall summary
    rba_act_spearmans = []
    for li in idxs:
        corr = all_corrs[li]
        if "rmsnorm_bound_angle" in corr and "activation" in corr["rmsnorm_bound_angle"]:
            rba_act_spearmans.append(corr["rmsnorm_bound_angle"]["activation"]["spearman"])

    if rba_act_spearmans:
        mean_s = sum(rba_act_spearmans) / len(rba_act_spearmans)
        print(f"  rmsnorm_bound_angle ↔ activation_score  mean Spearman = {mean_s:+.3f}")
        if mean_s > 0.7:
            print("  ✓ High correlation: static score is a good proxy for activation importance.")
        elif mean_s > 0.3:
            print("  ⚠ Moderate correlation: static score partially tracks activation importance.")
        elif mean_s > -0.1:
            print("  ✗ WEAK/NO correlation: static score does NOT identify important neurons.")
            print("    This explains why random pruning outperforms designed scores.")
        else:
            print("  ✗ NEGATIVE correlation: static score is INVERTED.")
            print("    Neurons that score LOW are actually the most activated ones.")

    return all_corrs


# ===========================================================================
# [7] PER-LAYER PRUNING SENSITIVITY
# ===========================================================================

def per_layer_pruning_test(
    model,
    tokenizer,
    eval_texts: List[str],
    method: str,
    prune_ratio: float = 0.01,
    device: str = "cpu",
    n_eval_samples: int = 30,
    max_seq_len: int = 256,
    batch_size: int = 4,
) -> List[Dict]:
    """
    Prune ONE layer at a time and measure the perplexity impact.

    This reveals:
    - Whether pruning is uniformly damaging across layers.
    - Whether a few specific layers are hyper-sensitive (and why).

    If some layers have ΔPPL ≈ 0 and others have ΔPPL >> 0, the sensitive
    layers contain neurons the score misidentifies as unimportant.
    """
    _hdr(7, "PER-LAYER PRUNING SENSITIVITY")
    print(f"  Method: {method},  Ratio: {prune_ratio:.1%},  Eval samples: {n_eval_samples}\n")

    layers  = get_transformer_layers(model)
    n_layers = len(layers)
    texts   = eval_texts[:n_eval_samples]

    # Baseline PPL
    print("  Computing baseline …")
    base = evaluate_perplexity(
        model, tokenizer, texts=texts,
        max_seq_len=max_seq_len, batch_size=batch_size, device=device,
    )
    base_ppl = base["perplexity"]
    print(f"  Baseline PPL: {base_ppl:.4f}\n")

    print(f"  {'Layer':>5}  {'d_ff_before':>12}  {'d_ff_after':>10}  "
          f"{'PPL':>8}  {'ΔPPL':>8}  {'rel':>6}")
    print(f"  {DIVIDER}")

    rows = []
    for i in range(n_layers):
        # Clone original and prune ONLY layer i
        m_single = clone_model(model)
        layers_s = get_transformer_layers(m_single)

        w_before  = get_mlp_weights(layers_s[i])
        d_ff_before = w_before["d_ff"]

        scores       = compute_scores(layers_s[i], method)
        keep_indices = get_keep_indices(scores, prune_ratio)
        d_ff_after   = keep_indices.numel()

        prune_layer_mlp(layers_s[i], keep_indices.to(device))

        res  = evaluate_perplexity(
            m_single, tokenizer, texts=texts,
            max_seq_len=max_seq_len, batch_size=batch_size, device=device,
        )
        ppl   = res["perplexity"]
        delta = ppl - base_ppl
        rel   = ppl / base_ppl

        sensitivity = ""
        if abs(delta) > 10:
            sensitivity = "  ← SENSITIVE"
        elif abs(delta) < 0.5:
            sensitivity = "  (insensitive)"

        print(f"  {i:>5}  {d_ff_before:>12}  {d_ff_after:>10}  "
              f"{ppl:>8.3f}  {delta:>+8.3f}  {rel:>5.2f}×{sensitivity}")

        rows.append({
            "layer_idx":    i,
            "method":       method,
            "prune_ratio":  prune_ratio,
            "d_ff_before":  d_ff_before,
            "d_ff_after":   d_ff_after,
            "perplexity":   ppl,
            "delta_ppl":    delta,
            "relative_ppl": rel,
        })

        del m_single
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


# ===========================================================================
# [8] PERPLEXITY SANITY CHECK
# ===========================================================================

def perplexity_sanity_check(
    model,
    tokenizer,
    device: str,
) -> Dict:
    """
    Evaluate PPL on a tiny, fixed, padding-free text.

    For Qwen2.5-0.5B, expected PPL ≈ 10–30.

    If PPL > 500 on this text:
        → The evaluation pipeline itself is broken (wrong tokenizer,
          wrong label alignment, etc.)
    If PPL is ~99 (as reported):
        → Might be a mismatch between the model and the evaluation text,
          OR the WikiText-2 data is not loading (fallback corpus used).
        → The fixed-text check disambiguates.
    """
    _hdr(8, "PERPLEXITY SANITY CHECK (fixed padding-free text)")

    result = evaluate_on_fixed_text(model, tokenizer, device, label="Qwen2.5-0.5B")

    # Also test on a very simple sentence
    simple_texts = [
        "The dog barked loudly.",
        "She went to the store to buy groceries.",
        "The capital of France is Paris.",
    ]
    print()
    for t in simple_texts:
        evaluate_on_fixed_text(model, tokenizer, device, text=t, label="simple")

    return result


# ===========================================================================
# Orchestrator: run_debug_mode
# ===========================================================================

def run_debug_mode(
    model,
    tokenizer,
    cfg: Dict,
    device: str,
    output_dir: str = "results",
) -> None:
    """
    Run all debug checks in sequence and save a report.

    Parameters
    ----------
    model      : the unpruned AutoModelForCausalLM
    tokenizer  : the tokenizer
    cfg        : the experiment config dict
    device     : 'cuda' or 'cpu'
    output_dir : directory to write the JSON report
    """
    os.makedirs(output_dir, exist_ok=True)
    methods = cfg.get("pruning_methods", ALL_STATIC_METHODS)
    seed    = cfg.get("seed", 42)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device":    device,
        "model":     getattr(model.config, "_name_or_path", "unknown"),
    }

    # ── Load a small eval set ─────────────────────────────────────────────────
    print("\nLoading evaluation dataset …")
    eval_texts = load_eval_dataset(max_samples=128)
    # Use a tiny subset for the slow per-layer test
    tiny_texts = eval_texts[:30]

    # ── [1] Shape validation ─────────────────────────────────────────────────
    report["shape_violations"] = check_tensor_shapes(model)

    # ── [2] Pruning direction ─────────────────────────────────────────────────
    report["pruning_direction"] = check_pruning_direction(
        model, methods, prune_ratio=0.05
    )

    # ── [3] Zero-mask equivalence ─────────────────────────────────────────────
    report["zero_mask_equivalence"] = run_zero_mask_equivalence_test(
        model, tokenizer, device, methods, prune_ratio=0.05, layer_idx=0
    )

    # ── [4] Model independence ────────────────────────────────────────────────
    report["model_independence"] = check_model_independence(
        model, methods[:2], prune_ratio=0.05   # limit to 2 methods to save time
    )

    # ── [8] PPL sanity (do this early — if it fails, skip slow tests) ─────────
    report["ppl_sanity"] = perplexity_sanity_check(model, tokenizer, device)

    # ── [5] Tiny ratios ───────────────────────────────────────────────────────
    report["tiny_ratio_results"] = test_tiny_pruning_ratios(
        model, tokenizer, methods, eval_texts,
        device=device, ratios=[0.001, 0.005, 0.01, 0.02, 0.05],
        n_eval_samples=64,
    )

    # ── [6] Correlation analysis ──────────────────────────────────────────────
    report["correlations"] = run_score_correlation_analysis(
        model, tokenizer, device, n_layers_to_check=6
    )

    # ── [7] Per-layer sensitivity ─────────────────────────────────────────────
    # Use the BEST static method vs random for comparison
    for m in ["rmsnorm_bound_angle", "random"]:
        key = f"per_layer_{m}"
        report[key] = per_layer_pruning_test(
            model, tokenizer, tiny_texts,
            method=m, prune_ratio=0.01, device=device, n_eval_samples=30,
        )

    # ── Save report ───────────────────────────────────────────────────────────
    ts          = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"debug_report_{ts}.json")

    # Correlations contain nested dicts with float values — serialize safely
    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serialisable(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        return str(obj)

    with open(report_path, "w") as f:
        json.dump(_make_serialisable(report), f, indent=2)

    print(f"\n{SECTION}")
    print("DEBUG REPORT COMPLETE")
    print(SECTION)
    print(f"\nFull report saved to: {report_path}")
    print()
    print("How to interpret results:")
    print("  [3] zero_mask_equivalence:")
    print("      is_consistent=true  → physical pruning code is CORRECT")
    print("      is_consistent=false → BUG in prune_layer_mlp (wrong row/col)")
    print()
    print("  [6] correlations (Spearman, rmsnorm_bound_angle ↔ activation):")
    print("      > 0.7  → score is a good proxy; poor results from other causes")
    print("      < 0.3  → score does NOT identify unimportant neurons")
    print("      < 0.0  → score is INVERTED; pruning removes the important neurons")
    print()
    print("  [5] tiny_ratio_results:")
    print("      If random stays near baseline but designed scores spike even at 0.1%,")
    print("      the scoring is hitting outlier / critical neurons first.")
