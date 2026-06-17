"""
target_pruning.py
=================
Fixed-percentage MLP neuron pruning experiment.

Unlike the alpha-based scaling experiment (scaling.py), this mode controls
pruning via a DIRECT percentage target rather than a score-mass budget (alpha).
This allows fair cross-model comparisons: Qwen2.5-3B and Qwen2.5-7B at exactly
the same fraction of MLP neurons removed.

Background
----------
Alpha (score-mass budget) does not correspond to the same pruning percentage
across models.  For example:
    Qwen2.5-3B  at alpha=1e-3  pruned 6.63% of MLP neurons
    Qwen2.5-7B  at alpha=1e-3  pruned 2.40% of MLP neurons
This is because alpha pruning selects neurons whose cumulative score sum <= alpha
× layer_total_sum.  The score distribution varies across architectures, so the
same alpha can yield very different pruning percentages.

Score-selection rule (unchanged from scaling.py)
-------------------------------------------------
    score_i = R^2 * ((||w_gate_i|| * ||w_up_i|| + |w_gate_i . w_up_i|) / 2)
              * ||w_down_i||
    where R = sqrt(d_model) * ||gamma||_inf   (RMSNorm weight bound)

Global selection algorithm
--------------------------
1. Compute scores for every MLP neuron in every layer (static, weight-based).
2. Sort ALL neurons globally by score ascending (lowest = least important).
3. Greedily select the lowest-scoring neurons until the target count is reached.
4. Per-layer safety cap: at most MAX_LAYER_FRAC (default 30%) of a single layer's
   neurons may be pruned.  If a layer hits its cap, skip it and continue with
   neurons from other layers.
5. Report whether the global target was reached and which layers hit the cap.

Local selection criterion for residual_local_topK (deterministic)
------------------------------------------------------------------
For each pruned layer, after computing E = A_P @ W_P.T (lost residual signal):
    score_j = ||(A_K[:, j])^T @ E||_2   for each KEPT neuron j
    — the L2 norm of the projection of kept neuron j's activations onto E

Top-K kept neurons by this score are used for the ridge solve.
Only those K down_proj columns are updated; all others remain as original W_K.
This criterion is deterministic given the calibration data and is written into
the JSON report for reproducibility.

Calibration / evaluation disjointness
--------------------------------------
Calibration: RECONSTRUCTION_TRAIN_PROMPTS (16 hardcoded short strings, NOT
             drawn from any evaluation dataset).
Evaluation:  WikiText-2 raw test split, first n_eval samples.
Overlap: IMPOSSIBLE by construction.  The two sources share no text.

Command-line usage
------------------
    python run_experiment.py --target-pruning-scaling \\
        --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B \\
        --target-pruning-percents 2 4 6 8 \\
        --methods pure_delete residual_full residual_local_top1024 residual_local_top2048 \\
        --n-eval 256

Or via YAML config (--target-pruning-scaling with --config):
    scaling_models: ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
    target_pruning_percents: [2, 4, 6, 8]
    scaling_methods: [pure_delete, residual_full, residual_local_top1024, residual_local_top2048]
    reconstruction_eval_samples: 256
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
BEST_RESIDUAL_LAM   = 1e-2
BEST_RESIDUAL_TAU   = 1.0
DEFAULT_MAX_LAYER_FRAC = 0.30    # per-layer cap: at most 30% of neurons pruned

# Alphas used ONLY for score-distribution diagnostics
DIAGNOSTIC_ALPHAS = [5e-4, 1e-3, 2e-3]

# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------
MAIN_CSV_KEYS = [
    "model", "target_pruning_percent", "eval_dataset", "selector",
    "actual_pruned_neurons", "total_mlp_neurons", "actual_pruning_percent",
    "mlp_channel_pruning_percent",
    "mlp_param_reduction_percent", "mlp_flop_reduction_percent",
    "total_model_param_reduction_percent",
    "estimated_total_forward_flop_reduction_percent",
    "method",
    "baseline_ppl", "compressed_ppl", "delta_ppl", "relative_ppl_increase_percent",
    "pure_delete_delta_ppl", "damage_reduction_percent",
    "reconstruction_time_seconds", "peak_gpu_memory_MB",
    "calibration_num_samples", "eval_num_samples",
    "calibration_num_tokens", "eval_num_tokens",
    "n_stable_layers", "n_unstable_layers",
    "cap_hit_layers_count", "target_reached",
    "dtype", "notes",
]

PER_LAYER_CSV_KEYS = [
    "model", "target_pruning_percent", "layer_index",
    "intermediate_size", "pruned_in_layer", "pruned_percent_in_layer",
    "mean_score_all_layer", "median_score_all_layer",
    "mean_score_pruned_layer", "max_score_pruned_layer",
]

DIAG_CSV_KEYS = [
    "model", "total_mlp_neurons",
    "score_min", "score_p0_1", "score_p1", "score_p5", "score_p10",
    "score_median", "score_p90", "score_max",
    "p1_over_median", "p5_over_median",
    "alpha_5e-4_neurons", "alpha_5e-4_pct",
    "alpha_1e-3_neurons", "alpha_1e-3_pct",
    "alpha_2e-3_neurons", "alpha_2e-3_pct",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _k(v: float) -> str:
    """Compact scientific notation string."""
    if v == 0:
        return "0"
    s = f"{v:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e")


def _parse_top_k(method: str) -> Optional[int]:
    """
    Parse method name to extract top_k for residual_local_topK methods.
    Returns None  for 'residual_full'
    Returns int   for 'residual_local_topK'
    Returns -1    for unknown / pure_delete / anything else
    """
    if method == "residual_full":
        return None
    if method.startswith("residual_local_top"):
        try:
            return int(method[len("residual_local_top"):])
        except ValueError:
            return -1
    return -1


def _flush_csv(path: str, rows: List[Dict], keys: List[str]) -> None:
    """Append rows to CSV; write header if file is new or empty."""
    if not rows:
        return
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Score gathering
# ---------------------------------------------------------------------------

def _gather_all_scores(
    layers: list,
    compute_bound_scores_fn,
) -> List[torch.Tensor]:
    """
    Compute per-layer RMSNorm-bound-angle scores (static; no forward pass).
    Returns list of detached 1-D float32 CPU tensors, shape [intermediate_size].
    """
    with torch.no_grad():
        return [compute_bound_scores_fn(layer)[0].detach().float().cpu()
                for layer in layers]


# ---------------------------------------------------------------------------
# Global target selection
# ---------------------------------------------------------------------------

def _select_global_target(
    all_scores:     List[torch.Tensor],
    target_n:       int,
    max_layer_frac: float = DEFAULT_MAX_LAYER_FRAC,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], dict]:
    """
    Select the target_n lowest-scoring MLP neurons globally across all layers,
    subject to a per-layer cap of floor(max_layer_frac * layer_size).

    Algorithm
    ---------
    1. Build a flat (score, layer_idx, neuron_idx) array sorted by score ascending.
       Uses numpy for efficiency (7B has ~530K neurons).
    2. Greedily select neurons in ascending score order.
       Skip neurons whose layer has already reached its cap.
    3. Stop at target_n selected or when no valid candidates remain.

    Returns
    -------
    prune_per_layer : List[LongTensor]  indices to prune per layer (sorted asc)
    keep_per_layer  : List[LongTensor]  indices to keep per layer (sorted asc)
    info            : dict with selection metadata
    """
    n_layers     = len(all_scores)
    layer_sizes  = [int(s.numel()) for s in all_scores]
    # Cap: at least 1 neuron must remain kept; cap from above at max_layer_frac
    caps = [max(1, int(max_layer_frac * sz)) for sz in layer_sizes]

    # Build flat sorted array
    flat_scores  = np.concatenate([s.detach().cpu().float().numpy() for s in all_scores])
    flat_layers  = np.concatenate(
        [np.full(sz, i, dtype=np.int32) for i, sz in enumerate(layer_sizes)]
    )
    flat_neurons = np.concatenate(
        [np.arange(sz, dtype=np.int32) for sz in layer_sizes]
    )
    order = np.argsort(flat_scores, kind="stable")   # ascending = least important first

    per_layer_pruned: List[List[int]] = [[] for _ in range(n_layers)]
    selected        = 0
    cap_hit_layers  = set()

    for oi in order:
        if selected >= target_n:
            break
        li = int(flat_layers[oi])
        ni = int(flat_neurons[oi])
        if len(per_layer_pruned[li]) >= caps[li]:
            cap_hit_layers.add(li)
            continue
        per_layer_pruned[li].append(ni)
        selected += 1

    prune_per_layer = [
        torch.tensor(sorted(pi), dtype=torch.long)
        for pi in per_layer_pruned
    ]
    # Build keep sets: complement of pruned, sorted
    keep_per_layer = [
        torch.tensor(
            [j for j in range(sz) if j not in set(pi)],
            dtype=torch.long,
        )
        for sz, pi in zip(layer_sizes, per_layer_pruned)
    ]

    info = {
        "actual_pruned":    selected,
        "target_n":         target_n,
        "per_layer_counts": [len(pi) for pi in per_layer_pruned],
        "caps":             caps,
        "cap_hit_layers":   sorted(cap_hit_layers),
        "target_reached":   selected >= target_n,
    }
    return prune_per_layer, keep_per_layer, info


# ---------------------------------------------------------------------------
# Score diagnostics
# ---------------------------------------------------------------------------

def _score_diagnostics(
    all_scores:          List[torch.Tensor],
    model_name:          str,
    total_mlp_neurons:   int,
    select_by_budget_fn,
) -> dict:
    """
    Compute score-distribution statistics and alpha-vs-pruning% comparisons.

    The alpha comparisons use select_by_budget_fn with per-layer total sums
    (matching the existing alpha-based selection logic exactly), so the neuron
    counts are directly comparable to --scaling-recon output.
    """
    flat = np.concatenate([s.detach().cpu().float().numpy() for s in all_scores])

    # Percentiles
    qs   = [0.1, 1.0, 5.0, 10.0, 50.0, 90.0]
    vals = np.percentile(flat, qs)
    p0_1, p1, p5, p10, median, p90 = vals

    eps = 1e-30
    diag: Dict = {
        "model":             model_name,
        "total_mlp_neurons": total_mlp_neurons,
        "score_min":         round(float(flat.min()),  8),
        "score_p0_1":        round(float(p0_1),        8),
        "score_p1":          round(float(p1),           8),
        "score_p5":          round(float(p5),           8),
        "score_p10":         round(float(p10),          8),
        "score_median":      round(float(median),       8),
        "score_p90":         round(float(p90),          8),
        "score_max":         round(float(flat.max()),  8),
        "p1_over_median":    round(float(p1  / (median + eps)), 6),
        "p5_over_median":    round(float(p5  / (median + eps)), 6),
    }

    # Alpha comparisons: use per-layer sums (matching existing behavior)
    with torch.no_grad():
        for alpha in DIAGNOSTIC_ALPHAS:
            count = 0
            for scores in all_scores:
                try:
                    pi, _ = select_by_budget_fn(scores, alpha, float(scores.sum()))
                    count += len(pi)
                except Exception:
                    pass
            pct    = 100.0 * count / total_mlp_neurons if total_mlp_neurons else 0.0
            a_key  = _k(alpha)
            diag[f"alpha_{a_key}_neurons"] = count
            diag[f"alpha_{a_key}_pct"]     = round(pct, 4)

    return diag


# ---------------------------------------------------------------------------
# Per-layer reporting
# ---------------------------------------------------------------------------

def _make_per_layer_rows(
    all_scores:      List[torch.Tensor],
    prune_per_layer: List[torch.Tensor],
    model_name:      str,
    target_pct:      float,
) -> List[Dict]:
    """Build one CSV row per (model, target_pct, layer)."""
    rows = []
    for li, (scores, pi) in enumerate(zip(all_scores, prune_per_layer)):
        sz         = int(scores.numel())
        n_pruned   = int(len(pi))
        prn_pct    = 100.0 * n_pruned / sz if sz else 0.0
        s_all      = scores.detach().cpu().float().numpy()

        row: Dict = {
            "model":                   model_name,
            "target_pruning_percent":  target_pct,
            "layer_index":             li,
            "intermediate_size":       sz,
            "pruned_in_layer":         n_pruned,
            "pruned_percent_in_layer": round(prn_pct, 4),
            "mean_score_all_layer":    round(float(s_all.mean()),       8),
            "median_score_all_layer":  round(float(np.median(s_all)),   8),
            "mean_score_pruned_layer": "",
            "max_score_pruned_layer":  "",
        }
        if n_pruned > 0:
            s_pruned = s_all[pi.detach().numpy()]
            row["mean_score_pruned_layer"] = round(float(s_pruned.mean()), 8)
            row["max_score_pruned_layer"]  = round(float(s_pruned.max()),  8)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Diagnostic plots (optional — silently skipped if matplotlib unavailable)
# ---------------------------------------------------------------------------

def _try_save_plots(
    all_scores:          List[torch.Tensor],
    model_name:          str,
    output_dir:          str,
    total_mlp_neurons:   int,
    select_by_budget_fn,
) -> None:
    """
    Save three diagnostic plots to output_dir:
      1. Histogram of log10(score)
      2. Cumulative score mass curve (Lorenz-style)
      3. Alpha vs pruning % curve
    Silently skips if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available — skipping diagnostic plots")
        return

    flat      = np.concatenate([s.detach().cpu().float().numpy() for s in all_scores])
    total_sum = float(flat.sum()) + 1e-30
    safe_name = model_name.replace("/", "_")

    # ── 1. log10(score) histogram ─────────────────────────────────────────
    log_s = np.log10(flat.clip(min=1e-30))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(log_s, bins=120, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(float(np.log10(np.median(flat) + 1e-30)),
               color="orange", linewidth=1.5, linestyle="--",
               label=f"median = {np.median(flat):.2e}")
    ax.set_xlabel("log₁₀(score)")
    ax.set_ylabel("neuron count")
    ax.set_title(f"{model_name}  —  score distribution  ({len(flat):,} neurons)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(output_dir, f"{safe_name}_score_hist.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", p)

    # ── 2. Cumulative score mass curve ────────────────────────────────────
    sorted_s   = np.sort(flat)
    cum_mass   = np.cumsum(sorted_s) / total_sum
    frac_n     = np.linspace(0.0, 100.0, len(sorted_s))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frac_n, cum_mass * 100.0, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Neurons pruned (%)")
    ax.set_ylabel("Cumulative score mass removed (%)")
    ax.set_title(f"{model_name}  —  cumulative score mass curve")
    ax.set_xlim(0, 20)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, f"{safe_name}_score_cumulative.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", p)

    # ── 3. Alpha vs pruning % curve ───────────────────────────────────────
    alpha_range = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3,
                   5e-3, 1e-2, 2e-2]
    alpha_pts, pct_pts = [], []
    for alpha in alpha_range:
        cnt = 0
        for scores in all_scores:
            try:
                pi, _ = select_by_budget_fn(scores, alpha, float(scores.sum()))
                cnt += len(pi)
            except Exception:
                pass
        alpha_pts.append(alpha)
        pct_pts.append(100.0 * cnt / total_mlp_neurons if total_mlp_neurons else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(alpha_pts, pct_pts, "o-", color="steelblue",
                linewidth=1.5, markersize=5)
    ax.set_xlabel("alpha (score-mass budget, per layer)")
    ax.set_ylabel("Actual pruning (%)")
    ax.set_title(f"{model_name}  —  alpha vs actual pruning %")
    ax.grid(alpha=0.3)
    for tgt in [2, 4, 6, 8]:
        ax.axhline(tgt, color="gray", linewidth=0.8, linestyle=":")
    fig.tight_layout()
    p = os.path.join(output_dir, f"{safe_name}_alpha_vs_pct.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", p)



# ---------------------------------------------------------------------------
# Physical shape verification
# ---------------------------------------------------------------------------

def _verify_pruned_shapes(
    layers_orig:     list,
    layers_pruned:   list,
    prune_per_layer: list,
    label:           str = "",
) -> int:
    """
    Confirm that gate_proj, up_proj, down_proj shapes are physically correct
    after pruning.  Returns total neurons removed across all layers.

    For each layer:
      gate_proj.weight : [d_ff_new, d_model]  where d_ff_new = d_ff - n_pruned
      up_proj.weight   : [d_ff_new, d_model]
      down_proj.weight : [d_model, d_ff_new]

    Logs a WARNING for any mismatch; does not raise (resilience over strictness).
    """
    from .model_utils import get_mlp_weights
    total_removed = 0
    for li, (ol, pl, pi) in enumerate(
            zip(layers_orig, layers_pruned, prune_per_layer)):
        try:
            ow = get_mlp_weights(ol)
            pw = get_mlp_weights(pl)
            n_pruned   = int(len(pi))
            d_ff_exp   = ow["d_ff"] - n_pruned
            d_ff_got   = pw["d_ff"]
            total_removed += n_pruned
            if d_ff_exp != d_ff_got:
                logger.warning(
                    "[%s] Layer %d shape mismatch: "
                    "expected d_ff=%d got d_ff=%d (pruned=%d)",
                    label, li, d_ff_exp, d_ff_got, n_pruned,
                )
            else:
                logger.debug(
                    "[%s] Layer %d OK: gate[%d,%d] up[%d,%d] down[%d,%d]",
                    label, li,
                    d_ff_got, ow["d_model"],
                    d_ff_got, ow["d_model"],
                    ow["d_model"], d_ff_got,
                )
        except Exception as exc:
            logger.warning("[%s] Layer %d shape check failed: %s", label, li, exc)
    return total_removed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_target_pruning_mode(
    cfg:             dict,
    device:          str,
    output_dir:      str          = "results",
    models_override: Optional[List[str]]   = None,
    targets_override: Optional[List[float]] = None,
    methods_override: Optional[List[str]]   = None,
    n_eval_override:  Optional[int]         = None,
    eval_datasets_override: Optional[List[str]] = None,
    selectors_override: Optional[List[str]] = None,
) -> None:
    """
    Fixed-percentage MLP pruning experiment.

    For each model x target_percent x method:
      1. Compute global neuron scores (static weights only — no forward pass)
      2. Select target_n lowest-scoring neurons (global, with 30% per-layer cap)
      3. Run pure_delete and/or residual reconstruction
      4. Evaluate PPL on WikiText-2 test split (disjoint from calibration)

    CLI args (models_override, etc.) take precedence over the YAML config when
    provided (i.e. not None).
    """
    from .merging import (
        RECONSTRUCTION_TRAIN_PROMPTS,
        RECONSTRUCTION_HELDOUT_PROMPTS,
        collect_mlp_inputs,
        apply_residual_down_reconstruction_timed,
    )
    from .bound_analysis import compute_bound_scores_and_R, select_by_budget
    from .selectors import gather_scores_for_selector
    from .evaluation import (evaluate_perplexity, load_eval_dataset,
                               load_all_eval_datasets)
    from .flops import estimate_mlp_flops
    from .model_utils import (
        count_parameters,
        get_transformer_layers,
        load_model_and_tokenizer,
    )
    from .pruning import prune_model_by_layer_indices, verify_forward_pass

    def _auto_dtype(dev: str) -> str:
        if dev != "cpu" and torch.cuda.is_available():
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"

    os.makedirs(output_dir, exist_ok=True)
    ts             = time.strftime("%Y%m%d_%H%M%S")
    main_csv_path  = os.path.join(output_dir, f"target_pruning_{ts}.csv")
    layer_csv_path = os.path.join(output_dir, f"target_pruning_per_layer_{ts}.csv")
    diag_csv_path  = os.path.join(output_dir, f"target_pruning_diagnostics_{ts}.csv")
    json_path      = os.path.join(output_dir, f"target_pruning_{ts}.json")

    # ── Config resolution: CLI overrides beat YAML ──────────────────────────
    model_list     = (models_override
                      or cfg.get("scaling_models", ["Qwen/Qwen2.5-0.5B"]))
    TARGET_PCTS    = (targets_override
                      or cfg.get("target_pruning_percents", [2.0, 4.0, 6.0, 8.0]))
    TARGET_PCTS    = [float(t) for t in TARGET_PCTS]
    METHODS        = (methods_override
                      or cfg.get("scaling_methods",
                                 ["pure_delete", "residual_full"]))
    n_eval         = int(n_eval_override
                         or cfg.get("reconstruction_eval_samples",
                                    cfg.get("bound_analysis_eval_samples", 256)))
    max_seq        = int(cfg.get("max_seq_len", 512))
    batch_sz       = int(cfg.get("batch_size", 4))
    use_fb         = bool(cfg.get("use_fallback_corpus", False))
    dtype_cfg      = str(cfg.get("scaling_dtype", "auto"))
    max_layer_frac = float(cfg.get("max_layer_prune_frac", DEFAULT_MAX_LAYER_FRAC))
    EVAL_DATASETS  = (eval_datasets_override
                      or cfg.get("eval_datasets", ["wikitext2"]))
    EVAL_DATASETS  = [str(d) for d in EVAL_DATASETS]
    SELECTORS      = (selectors_override
                      or cfg.get("selectors", ["rmsnorm_bound"]))
    SELECTORS      = [str(s) for s in SELECTORS]
    needs_act_sel  = any(s == "activation_score" for s in SELECTORS)

    n_train_prompts  = len(RECONSTRUCTION_TRAIN_PROMPTS)
    n_heldout_prompts = len(RECONSTRUCTION_HELDOUT_PROMPTS)
    n_calib_samples  = n_train_prompts + n_heldout_prompts
    needs_calib      = any(m != "pure_delete" for m in METHODS) or needs_act_sel

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("TARGET-PRUNING SCALING EXPERIMENT")
    print(f"  Models         : {model_list}")
    print(f"  Target percents: {TARGET_PCTS}%")
    print(f"  Methods        : {METHODS}")
    print(f"  Selectors      : {SELECTORS}")
    print(f"  n_eval         : {n_eval} samples per dataset")
    print(f"  Eval datasets  : {EVAL_DATASETS}")
    print(f"  Max layer frac : {max_layer_frac:.0%}")
    print(f"  lambda / tau   : {BEST_RESIDUAL_LAM} / {BEST_RESIDUAL_TAU}")
    print(f"{'=' * 90}")
    print()
    print("  Calibration source: RECONSTRUCTION_*_PROMPTS")
    print(f"    {n_train_prompts} train + {n_heldout_prompts} held-out = "
          f"{n_calib_samples} hardcoded strings (NOT from WikiText-2)")
    print(f"  Evaluation source : WikiText-2 raw test split, first {n_eval} samples")
    print("  Overlap check     : IMPOSSIBLE — calibration data is hardcoded text,")
    print("                      evaluation data is WikiText-2. No shared tokens.")
    print()

    # Load all eval corpora ONCE — same corpora for every model ensures comparability
    print(f"Loading evaluation datasets: {EVAL_DATASETS} ...")
    all_eval_corpora: Dict[str, list] = load_all_eval_datasets(
        EVAL_DATASETS, max_samples=n_eval, use_fallback_corpus=use_fb,
    )
    for _dn, _txts in all_eval_corpora.items():
        print(f"  {_dn}: {len(_txts)} samples "
              f"(approx {len(_txts) * max_seq:,} tokens)")
    print()
    # Backward-compat references (used for diag prints; per-dataset values used in rows)
    eval_texts    = all_eval_corpora[EVAL_DATASETS[0]]
    n_eval_actual = len(eval_texts)
    n_eval_tokens_approx = n_eval_actual * max_seq

    all_results:       List[Dict] = []
    all_layer_results: List[Dict] = []
    all_diag_results:  List[Dict] = []

    # ── Per-model loop ────────────────────────────────────────────────────────
    for model_name in model_list:
        print(f"\n{'#' * 90}")
        print(f"MODEL: {model_name}")
        print(f"{'#' * 90}")

        model     = None
        tokenizer = None

        try:
            dtype_str = _auto_dtype(device) if dtype_cfg == "auto" else dtype_cfg
            print(f"  dtype = {dtype_str}")

            model, tokenizer, _ = load_model_and_tokenizer(
                model_name    = model_name,
                fallback_name = None,
                device        = device,
                dtype_str     = dtype_str,
            )
            model.eval()

            cfg_m        = model.config
            n_layers     = getattr(cfg_m, "num_hidden_layers", None)
            hidden_size  = getattr(cfg_m, "hidden_size", None)
            inter_size   = getattr(cfg_m, "intermediate_size", None)

            print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Layers       : {n_layers}")
            print(f"  Hidden size  : {hidden_size}")
            print(f"  Intermediate : {inter_size}")

            # Per-dataset baseline PPLs (computed once per model)
            print("\n  Computing per-dataset baseline PPLs ...")
            layers = get_transformer_layers(model)
            baseline_ppl_per_ds: Dict[str, float] = {}
            for _ds in EVAL_DATASETS:
                _bp = evaluate_perplexity(
                    model, tokenizer, texts=all_eval_corpora[_ds],
                    max_seq_len=max_seq, batch_size=batch_sz, device=device,
                )
                baseline_ppl_per_ds[_ds] = _bp["perplexity"]
                print(f"    {_ds}: PPL = {_bp['perplexity']:.4f}")
            baseline_ppl    = baseline_ppl_per_ds[EVAL_DATASETS[0]]
            baseline_params = count_parameters(model)
            baseline_flops  = estimate_mlp_flops(model, seq_len=max_seq)

            # Static scores for rmsnorm_bound (used for diagnostics + total_mlp count)
            print("\n  Computing RMSNorm-bound-angle scores (for diagnostics) ...")
            all_scores    = _gather_all_scores(layers, compute_bound_scores_and_R)
            total_mlp     = sum(int(s.numel()) for s in all_scores)
            print(f"  Total MLP neurons : {total_mlp:,}")

            # Score diagnostics (once per model, always with rmsnorm_bound)
            print("  Computing score diagnostics ...")
            diag = _score_diagnostics(
                all_scores, model_name, total_mlp, select_by_budget
            )
            all_diag_results.append(diag)
            _flush_csv(diag_csv_path, [diag], DIAG_CSV_KEYS)
            print(f"    score median    : {diag['score_median']:.4e}")
            print(f"    alpha=5e-4 -> {diag.get('alpha_5e-4_neurons','?')} neurons "
                  f"({diag.get('alpha_5e-4_pct','?'):.2f}%)")
            print(f"    alpha=1e-3 -> {diag.get('alpha_1e-3_neurons','?')} neurons "
                  f"({diag.get('alpha_1e-3_pct','?'):.2f}%)")
            print(f"    alpha=2e-3 -> {diag.get('alpha_2e-3_neurons','?')} neurons "
                  f"({diag.get('alpha_2e-3_pct','?'):.2f}%)")

            _try_save_plots(all_scores, model_name, output_dir, total_mlp, select_by_budget)

            # Calibration inputs (once per model, only if residual methods present)
            train_r        = None
            heldout_r      = None
            n_calib_tokens = 0

            if needs_calib:
                print("  Collecting calibration inputs (train) ...")
                train_r = collect_mlp_inputs(
                    model, tokenizer, RECONSTRUCTION_TRAIN_PROMPTS, device,
                    max_seq_len=max_seq,
                )
                print("  Collecting calibration inputs (held-out) ...")
                heldout_r = collect_mlp_inputs(
                    model, tokenizer, RECONSTRUCTION_HELDOUT_PROMPTS, device,
                    max_seq_len=max_seq,
                )
                if train_r:
                    n_calib_tokens = int(train_r[0].shape[0])
                print(f"  Calibration: {n_train_prompts} train prompts, "
                      f"{n_calib_tokens} tokens per layer")
                print(f"  Disjointness confirmed: calibration = hardcoded prompts; "
                      f"evaluation = WikiText-2 test split")

            # ── Per-layer calibration activations for activation_score ───────
            calib_per_layer: List[Optional[torch.Tensor]] = [None] * len(layers)
            if needs_act_sel and train_r is not None:
                print("  Caching per-layer MLP inputs for activation_score ...")
                for _li, _r in enumerate(train_r):
                    if _r is not None:
                        calib_per_layer[_li] = _r.detach().float().cpu()

            # ── Per-target loop ──────────────────────────────────────────────
            for target_pct in TARGET_PCTS:
                print(f"\n  {'=' * 70}")
                print(f"  Target: {target_pct:.1f}%  of  {total_mlp:,}  MLP neurons")

                for selector_name in SELECTORS:
                    print(f"\n    [selector={selector_name}]")

                    # Gather scores for this selector
                    sel_scores = gather_scores_for_selector(
                        selector_name, layers,
                        calib_inputs_per_layer=calib_per_layer,
                        device=device,
                    )

                    target_n = round(target_pct / 100.0 * total_mlp)
                    prune_per_layer, keep_per_layer, sel_info = _select_global_target(
                        sel_scores, target_n, max_layer_frac=max_layer_frac
                    )

                    actual_pruned = sel_info["actual_pruned"]
                    actual_pct    = 100.0 * actual_pruned / total_mlp if total_mlp else 0.0
                    cap_hits      = sel_info["cap_hit_layers"]

                    print(f"      Requested: {target_n:,}  "
                          f"Actual: {actual_pruned:,} ({actual_pct:.3f}%)")
                    if cap_hits:
                        print(f"      Per-layer cap hit in {len(cap_hits)} layer(s)")
                    if not sel_info["target_reached"]:
                        logger.warning(
                            "Target %.1f%% NOT REACHED for %s selector=%s: "
                            "actual %.3f%% (cap hit in %d layers)",
                            target_pct, model_name, selector_name,
                            actual_pct, len(cap_hits),
                        )
                        print(f"      WARNING: target not reached due to per-layer cap")

                    # Per-layer CSV (one entry per selector per target%)
                    layer_rows = _make_per_layer_rows(
                        sel_scores, prune_per_layer, model_name, target_pct
                    )
                    all_layer_results.extend(layer_rows)
                    _flush_csv(layer_csv_path, layer_rows, PER_LAYER_CSV_KEYS)

                    # ─────────────────────────────────────────────────────────────
                    # Row factories and PPL helper
                    # ─────────────────────────────────────────────────────────────
                    def _base_row(method_: str, ds_name_: str, **extra) -> Dict:
                        """Build a base result row for (method, dataset)."""
                        r: Dict = {
                            "model":                    model_name,
                            "target_pruning_percent":   target_pct,
                            "eval_dataset":             ds_name_,
                            "selector":                 selector_name,
                            "actual_pruned_neurons":    actual_pruned,
                            "total_mlp_neurons":        total_mlp,
                            "actual_pruning_percent":   round(actual_pct, 4),
                            "method":                   method_,
                            "baseline_ppl":             round(
                                baseline_ppl_per_ds.get(ds_name_, baseline_ppl), 4),
                            "dtype":                    dtype_str,
                            "calibration_num_samples":  n_calib_samples,
                            "eval_num_samples":         len(all_eval_corpora[ds_name_]),
                            "calibration_num_tokens":   n_calib_tokens,
                            "eval_num_tokens":          (
                                len(all_eval_corpora[ds_name_]) * max_seq),
                            "cap_hit_layers_count":     len(cap_hits),
                            "target_reached":           sel_info["target_reached"],
                            "notes":                    "",
                        }
                        r.update(extra)
                        return r

                    def _fill_ppl(row: Dict, m, ds_name_: str) -> Tuple[float, float]:
                        """Evaluate m on dataset ds_name_; populate compression cols."""
                        cur_texts_   = all_eval_corpora[ds_name_]
                        cur_bppl_    = baseline_ppl_per_ds.get(ds_name_, baseline_ppl)
                        fp_ok    = verify_forward_pass(m, tokenizer, device)
                        ppl_info = evaluate_perplexity(
                            m, tokenizer, texts=cur_texts_,
                            max_seq_len=max_seq, batch_size=batch_sz, device=device,
                        )
                        ppl   = ppl_info["perplexity"]
                        delta = ppl - cur_bppl_
                        pp    = count_parameters(m)
                        pf    = estimate_mlp_flops(m, seq_len=max_seq)

                        mlp_par_red = round(
                            100 * (1 - pp["mlp"] / baseline_params["mlp"]), 4)
                        mlp_flp_red = round(
                            100 * (1 - pf["total_flops"]
                                   / baseline_flops["total_flops"]), 4)
                        total_par_red = round(
                            100 * (1 - pp["total"] / baseline_params["total"]), 4)

                        # Attention FLOPs ≈ 8 * seq * d_model^2 * n_layers (approx)
                        _n_ly  = getattr(cfg_m, "num_hidden_layers", 1)
                        _d_m   = getattr(cfg_m, "hidden_size", 1)
                        _attn  = 8 * max_seq * (_d_m ** 2) * _n_ly
                        _tb    = baseline_flops["total_flops"] + _attn
                        _mlp_s = baseline_flops["total_flops"] - pf["total_flops"]
                        est_tot_flp_red = round(
                            100 * _mlp_s / _tb if _tb > 0 else 0.0, 4)

                        row.update({
                            "compressed_ppl":                round(ppl,   4),
                            "delta_ppl":                     round(delta, 4),
                            "relative_ppl_increase_percent": round(
                                100.0 * delta / cur_bppl_, 4),
                            "mlp_channel_pruning_percent":   round(actual_pct, 4),
                            "mlp_param_reduction_percent":   mlp_par_red,
                            "mlp_flop_reduction_percent":    mlp_flp_red,
                            "total_model_param_reduction_percent":       total_par_red,
                            "estimated_total_forward_flop_reduction_percent":
                                                             est_tot_flp_red,
                            "notes": row.get("notes", "") or ("" if fp_ok
                                                              else "forward_pass_failed"),
                        })
                        return round(ppl, 4), round(delta, 4)

                    # ── Method loop (prune/reconstruct once; eval on all datasets) ──
                    all_target_rows: List[Dict] = []
                    # dppl_delete per dataset (needed for damage_reduction of resid methods)
                    dppl_delete_per_ds: Dict[str, Optional[float]] = {
                        ds: None for ds in EVAL_DATASETS
                    }

                    for method in METHODS:

                        # ── pure_delete ──────────────────────────────────────────
                        if method == "pure_delete":
                            print(f"    [pure_delete] pruning ... ", end="", flush=True)
                            pruned_model = None
                            try:
                                pruned_model, _ = prune_model_by_layer_indices(
                                    model, prune_per_layer,
                                    label=f"tp_del_{target_pct}",
                                )
                                # Physical shape verification (once per pruning)
                                _n_rm = _verify_pruned_shapes(
                                    layers,
                                    get_transformer_layers(pruned_model),
                                    prune_per_layer,
                                    label=f"pure_delete/{model_name}/{target_pct}%",
                                )
                                print(f"shape OK ({_n_rm:,} removed)")
                                # Evaluate on each dataset
                                for _ds in EVAL_DATASETS:
                                    row_del = _base_row("pure_delete", _ds)
                                    _, _dppl = _fill_ppl(row_del, pruned_model, _ds)
                                    row_del["pure_delete_delta_ppl"]    = _dppl
                                    row_del["damage_reduction_percent"] = float("nan")
                                    dppl_delete_per_ds[_ds] = _dppl
                                    print(
                                        f"      [{_ds}] PPL={row_del['compressed_ppl']:.4f}"
                                        f"  dPPL={_dppl:+.4f}"
                                        f"  rel={row_del['relative_ppl_increase_percent']:+.2f}%"
                                    )
                                    all_target_rows.append(row_del)
                            except Exception as exc:
                                logger.error("pure_delete %s t=%.1f%%: %s",
                                             model_name, target_pct, exc, exc_info=True)
                                for _ds in EVAL_DATASETS:
                                    r = _base_row("pure_delete", _ds)
                                    r["notes"] = f"ERROR: {exc}"
                                    all_target_rows.append(r)
                                print(f"FAILED: {exc}")
                            finally:
                                if pruned_model is not None:
                                    del pruned_model
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            continue

                        # ── residual methods ─────────────────────────────────────
                        top_k = _parse_top_k(method)
                        if top_k == -1:
                            logger.warning("Unknown method '%s' -- skipping", method)
                            continue
                        if train_r is None or heldout_r is None:
                            logger.error(
                                "Calibration inputs needed for '%s' but missing", method
                            )
                            continue

                        print(f"    [{method}] reconstructing ... ", end="", flush=True)
                        pruned_model = None
                        try:
                            pruned_model, build_info = apply_residual_down_reconstruction_timed(
                                model, prune_per_layer, keep_per_layer,
                                train_r, heldout_r,
                                ridge_lambda=BEST_RESIDUAL_LAM,
                                tau=BEST_RESIDUAL_TAU,
                                top_k=top_k,
                            )
                            n_s = build_info["n_stable_layers"]
                            n_u = build_info["n_unstable_layers"]
                            t_s   = build_info.get("total_recon_time_s", float("nan"))
                            gb_mb = build_info.get("peak_gpu_mb", float("nan"))
                            print(f"done  t={t_s:.1f}s  stable={n_s}/{n_s+n_u}")

                            if build_info.get("all_unstable", False):
                                for _ds in EVAL_DATASETS:
                                    r = _base_row(method, _ds,
                                                  ridge_lambda=BEST_RESIDUAL_LAM,
                                                  tau=BEST_RESIDUAL_TAU)
                                    r["n_stable_layers"]             = n_s
                                    r["n_unstable_layers"]           = n_u
                                    r["reconstruction_time_seconds"] = t_s
                                    r["peak_gpu_memory_MB"]          = gb_mb
                                    r["notes"] = "SKIPPED_PPL:all_layers_unstable"
                                    all_target_rows.append(r)
                                continue

                            # Evaluate on each dataset
                            for _ds in EVAL_DATASETS:
                                row_res = _base_row(method, _ds,
                                                    ridge_lambda=BEST_RESIDUAL_LAM,
                                                    tau=BEST_RESIDUAL_TAU)
                                row_res["n_stable_layers"]             = n_s
                                row_res["n_unstable_layers"]           = n_u
                                row_res["reconstruction_time_seconds"] = t_s
                                row_res["peak_gpu_memory_MB"]          = gb_mb
                                _, dppl_res = _fill_ppl(row_res, pruned_model, _ds)

                                # Damage reduction (per-dataset baseline)
                                dppl_del = dppl_delete_per_ds.get(_ds)
                                if dppl_del is not None and abs(dppl_del) >= 0.05:
                                    dmg = round(
                                        100.0 * (dppl_del - dppl_res) / dppl_del, 2)
                                    row_res["damage_reduction_percent"] = dmg
                                    row_res["pure_delete_delta_ppl"]    = round(dppl_del, 4)
                                else:
                                    row_res["damage_reduction_percent"] = float("nan")

                                dm   = row_res.get("damage_reduction_percent", float("nan"))
                                dm_s = f"  dmg_red={dm:+.1f}%" if dm == dm else ""
                                print(
                                    f"      [{_ds}] PPL={row_res['compressed_ppl']:.4f}"
                                    f"  dPPL={dppl_res:+.4f}"
                                    f"  rel={row_res['relative_ppl_increase_percent']:+.2f}%"
                                    + dm_s
                                )
                                all_target_rows.append(row_res)

                        except Exception as exc:
                            logger.error("[%s] %s t=%.1f%%: %s",
                                         method, model_name, target_pct, exc, exc_info=True)
                            for _ds in EVAL_DATASETS:
                                r = _base_row(method, _ds)
                                r["notes"] = f"ERROR: {exc}"
                                all_target_rows.append(r)
                            print(f"FAILED: {exc}")
                        finally:
                            if pruned_model is not None:
                                del pruned_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    # Flush after every (model, target_pct, selector)
                    _flush_csv(main_csv_path, all_target_rows, MAIN_CSV_KEYS)
                    all_results.extend(all_target_rows)
                    all_target_rows = []  # reset for next selector
                # end selector loop

        except torch.cuda.OutOfMemoryError as oom:
            logger.error("OOM for %s: %s", model_name, oom)
            print(f"  *** OOM -- skipping {model_name} ***")
            err = {"model": model_name, "notes": f"OOM: {oom}"}
            _flush_csv(main_csv_path, [err], MAIN_CSV_KEYS)
            all_results.append(err)

        except Exception as exc:
            logger.error("Failed %s: %s", model_name, exc, exc_info=True)
            print(f"  *** ERROR -- {model_name}: {exc} ***")
            err = {"model": model_name, "notes": f"ERROR: {exc}"}
            _flush_csv(main_csv_path, [err], MAIN_CSV_KEYS)
            all_results.append(err)

        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n  Released {model_name} from memory.\n")

    # ── Final summary table ───────────────────────────────────────────────────
    ppl_rows = [r for r in all_results if "compressed_ppl" in r]
    W = 204
    print(f"\n{'=' * W}")
    print(f"TARGET-PRUNING SUMMARY  (n_eval={n_eval_actual})")
    print(f"{'─' * W}")
    hdr = (
        f"  {'model':>22}  {'tgt%':>5}  {'act%':>6}  {'pruned':>8}  "
        f"{'mlp_par%':>8}  {'tot_par%':>8}  {'mlp_flp%':>8}  {'est_flp%':>8}  "
        f"{'dataset':>12}  {'selector':>18}  {'method':>30}  "
        f"{'bPPL':>8}  {'PPL':>9}  {'dPPL':>9}  {'rel%':>7}  {'dmg_red':>9}  "
        f"{'t_s':>6}  {'gpu_MB':>7}"
    )
    print(hdr)
    print(f"{'─' * W}")
    for r in ppl_rows:
        act_pct  = r.get("actual_pruning_percent",        float("nan"))
        n_prn    = r.get("actual_pruned_neurons",         "?")
        par_red  = r.get("mlp_param_reduction_percent",   float("nan"))
        tpar_red = r.get("total_model_param_reduction_percent", float("nan"))
        flp_red  = r.get("mlp_flop_reduction_percent",    float("nan"))
        eflp_red = r.get("estimated_total_forward_flop_reduction_percent",
                         float("nan"))
        dm       = r.get("damage_reduction_percent",      float("nan"))
        rel      = r.get("relative_ppl_increase_percent", float("nan"))
        t_s      = r.get("reconstruction_time_seconds",   float("nan"))
        gpu_mb   = r.get("peak_gpu_memory_MB",            float("nan"))

        act_s   = f"{act_pct:5.2f}%"   if act_pct  == act_pct   else "  nan%"
        par_s   = f"{par_red:+7.2f}%"  if par_red  == par_red   else "     nan%"
        tpar_s  = f"{tpar_red:+7.2f}%" if tpar_red == tpar_red  else "     nan%"
        flp_s   = f"{flp_red:+7.2f}%"  if flp_red  == flp_red   else "     nan%"
        eflp_s  = f"{eflp_red:+7.2f}%" if eflp_red == eflp_red  else "     nan%"
        dm_s    = f"{dm:+8.1f}%"       if dm == dm               else "      nan%"
        rel_s   = f"{rel:+7.2f}%"      if rel == rel             else "    nan%"
        t_s_s   = f"{t_s:6.1f}"        if t_s == t_s             else "   nan"
        gpu_s   = f"{gpu_mb:7.0f}"     if gpu_mb == gpu_mb       else "    nan"

        print(
            f"  {str(r.get('model',''))[-22:]:>22}  "
            f"{float(r.get('target_pruning_percent', 0)):>5.1f}  {act_s}  "
            f"{str(n_prn):>8}  {par_s}  {tpar_s}  {flp_s}  {eflp_s}  "
            f"{str(r.get('eval_dataset',''))[:12]:>12}  "
            f"{str(r.get('selector',''))[:18]:>18}  "
            f"{str(r.get('method',''))[:30]:>30}  "
            f"{float(r.get('baseline_ppl', 0)):>8.4f}  "
            f"{float(r.get('compressed_ppl', 0)):>9.4f}  "
            f"{float(r.get('delta_ppl', 0)):>+9.4f}  "
            f"{rel_s}  {dm_s}  {t_s_s}  {gpu_s}"
        )
    print(f"{'=' * W}\n")

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "timestamp":            ts,
        "mode":                 "target_pruning_scaling",
        "models":               model_list,
        "target_percents":      TARGET_PCTS,
        "methods":              METHODS,
        "n_eval":               n_eval_actual,
        "best_lambda":          BEST_RESIDUAL_LAM,
        "best_tau":             BEST_RESIDUAL_TAU,
        "max_layer_frac":       max_layer_frac,
        "local_selection_criterion": (
            "score_j = ||(A_K[:,j])^T @ E||_2  "
            "where E = A_P @ W_P.T  (lost residual signal)"
        ),
        "calibration_source":   "RECONSTRUCTION_*_PROMPTS (hardcoded, not WikiText-2)",
        "eval_source":          "WikiText-2 raw test split",
        "overlap_possible":     False,
        "results":              all_results,
        "diagnostics":          all_diag_results,
    }
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"Main CSV    : {main_csv_path}")
    print(f"Layer CSV   : {layer_csv_path}")
    print(f"Diagnostics : {diag_csv_path}")
    print(f"JSON report : {json_path}\n")
