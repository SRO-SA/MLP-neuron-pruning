"""
scaling.py
==========
Model-scaling experiment: pure_delete vs residual reconstruction across Qwen2.5 sizes.

Usage
-----
    python run_experiment.py --config configs/default.yaml --scaling-recon

For each model × alpha × method, runs structured MLP pruning and evaluates PPL.
Methods are configured via ``scaling_methods`` in the YAML config.

Supported methods
-----------------
  pure_delete              — prune and evaluate immediately
  residual_full            — residual correction using ALL kept neurons (best quality)
  residual_local_top128    — local correction: top-128 kept neurons by correlation
  residual_local_top256    — local correction: top-256
  residual_local_top512    — local correction: top-512  (good scalability/quality tradeoff)
  residual_local_top1024   — local correction: top-1024

Config keys
-----------
  scaling_models            : list of HF model IDs  (default: 0.5B, 1.5B)
  scaling_alphas            : list of prune-budget fractions
                              (default: [1e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2])
  scaling_methods           : list of method names (default: pure_delete + residual_full)
  scaling_dtype             : "bfloat16" | "float16" | "float32" | "auto"
                              "auto" -> bfloat16 on CUDA if supported, else float32
  reconstruction_eval_samples : WikiText-2 samples for PPL evaluation (default 256)
  max_seq_len               : sequence length for calibration and PPL (default 512)
  batch_size                : PPL evaluation batch size (default 4)
  use_fallback_corpus       : fall back to built-in corpus when WikiText-2 unavailable

Reported metrics
----------------
  recon_time_s  : total residual reconstruction time (seconds, all layers)
  peak_gpu_mb   : peak CUDA memory during reconstruction (MB)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BEST_RESIDUAL_LAM = 1e-2
BEST_RESIDUAL_TAU = 1.0

DEFAULT_SCALING_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]

DEFAULT_ALPHAS = [1e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]

# Default: pure_delete + full residual (backward-compatible)
DEFAULT_SCALING_METHODS = ["pure_delete", "residual_full"]

CSV_KEYS = [
    # Model identity & architecture
    "model_name", "n_params", "n_layers", "hidden_size",
    "intermediate_size", "total_mlp_neurons",
    # Pruning config
    "alpha", "n_pruned", "pct_pruned",
    # Method
    "method", "ridge_lambda", "tau",
    # PPL metrics
    "baseline_ppl", "perplexity", "perplexity_delta",
    "relative_ppl_increase_pct",
    # Cross-method comparison
    "damage_reduction_pct",
    # Compression metrics
    "mlp_params_red_pct", "flops_red_pct",
    # Reconstruction quality (residual methods only)
    "train_improvement_pct", "heldout_improvement_pct",
    "overfit_gap_pct", "update_norm_ratio_mean",
    # Stability (residual methods only)
    "n_stable_layers", "n_unstable_layers",
    # Timing & memory (residual methods only)
    "recon_time_s", "peak_gpu_mb",
    # Misc
    "forward_pass_ok", "dtype", "notes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_dtype(device: str) -> str:
    """Pick the best dtype: bfloat16 on CUDA if supported, else float32."""
    if device != "cpu" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return "float32"


def _get_arch_info(model) -> Dict:
    """Extract architecture metadata from model.config."""
    cfg           = model.config
    n_layers      = getattr(cfg, "num_hidden_layers", None)
    hidden_size   = getattr(cfg, "hidden_size", None)
    inter_size    = getattr(cfg, "intermediate_size", None)
    total_mlp     = (n_layers * inter_size) if (n_layers and inter_size) else None
    n_params      = sum(p.numel() for p in model.parameters())
    return {
        "n_params":          n_params,
        "n_layers":          n_layers,
        "hidden_size":       hidden_size,
        "intermediate_size": inter_size,
        "total_mlp_neurons": total_mlp,
    }


def _flush_csv(path: str, rows: List[Dict]) -> None:
    """Append *rows* to *path* (writes header if file is new/empty)."""
    if not rows:
        return
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_KEYS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _k(v: float) -> str:
    """Format float as compact scientific string."""
    if v == 0:
        return "0"
    s = f"{v:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e")


def _parse_top_k(method: str) -> Optional[int]:
    """
    Parse method name to extract top_k.
    Returns None for "residual_full", int for "residual_local_topK".
    Returns -1 for unrecognised residual methods (treated as full).
    """
    if method == "residual_full":
        return None
    if method.startswith("residual_local_top"):
        try:
            return int(method[len("residual_local_top"):])
        except ValueError:
            return -1
    return -1   # not a residual method


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_scaling_recon_mode(cfg: dict, device: str, output_dir: str = "results") -> None:
    """
    Sequential multi-model scaling experiment.

    Each model is loaded, evaluated, then deleted before the next is loaded.
    Partial results are written to CSV after every (model, alpha) pair.
    """
    from .merging import (
        RECONSTRUCTION_TRAIN_PROMPTS,
        RECONSTRUCTION_HELDOUT_PROMPTS,
        collect_mlp_inputs,
        apply_residual_down_reconstruction,
        apply_residual_down_reconstruction_timed,
    )
    from .bound_analysis import compute_bound_scores_and_R, select_by_budget
    from .evaluation import evaluate_perplexity, load_eval_dataset
    from .flops import estimate_mlp_flops
    from .model_utils import (
        count_parameters,
        get_transformer_layers,
        load_model_and_tokenizer,
    )
    from .pruning import prune_model_by_layer_indices, verify_forward_pass

    os.makedirs(output_dir, exist_ok=True)
    ts        = time.strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(output_dir, f"scaling_recon_{ts}.csv")
    json_path = os.path.join(output_dir, f"scaling_recon_{ts}.json")

    # ── Config ──────────────────────────────────────────────────────────────
    model_list  = cfg.get("scaling_models",  DEFAULT_SCALING_MODELS)
    ALPHAS      = cfg.get("scaling_alphas",  DEFAULT_ALPHAS)
    METHODS     = cfg.get("scaling_methods", DEFAULT_SCALING_METHODS)
    n_eval      = cfg.get("reconstruction_eval_samples",
                          cfg.get("bound_analysis_eval_samples", 256))
    max_seq     = cfg.get("max_seq_len", 512)
    batch_sz    = cfg.get("batch_size", 4)
    use_fb      = cfg.get("use_fallback_corpus", True)
    dtype_cfg   = cfg.get("scaling_dtype", "auto")

    print(f"\n{'=' * 80}")
    print(f"SCALING EXPERIMENT")
    print(f"  Models  : {model_list}")
    print(f"  Alphas  : {[_k(a) for a in ALPHAS]}")
    print(f"  Methods : {METHODS}")
    print(f"  n_eval  : {n_eval} WikiText-2 samples (test split)")
    print(f"{'=' * 80}")
    print()
    print("  Calibration/evaluation disjointness:")
    print("    Calibration = 16 fixed short strings (RECONSTRUCTION_*_PROMPTS)")
    print("    Evaluation  = WikiText-2 test split -- no overlap by construction")
    print()

    # Load evaluation texts once — same corpus for all models ensures comparability
    print("Loading WikiText-2 evaluation texts ...")
    eval_texts = load_eval_dataset(n_eval, use_fallback_corpus=use_fb)
    print(f"  Loaded {len(eval_texts)} samples.\n")

    all_results: List[Dict] = []

    # ── Per-model loop ───────────────────────────────────────────────────────
    for model_name in model_list:
        print(f"\n{'#' * 80}")
        print(f"MODEL: {model_name}")
        print(f"{'#' * 80}")

        model     = None
        tokenizer = None

        try:
            # Dtype selection
            if dtype_cfg == "auto":
                dtype_str = _auto_dtype(device)
            else:
                dtype_str = dtype_cfg
            print(f"  dtype = {dtype_str}")

            model, tokenizer, resolved_name = load_model_and_tokenizer(
                model_name=model_name,
                fallback_name=None,
                device=device,
                dtype_str=dtype_str,
            )
            model.eval()
            arch = _get_arch_info(model)

            print(f"  Parameters   : {arch['n_params']:,}")
            print(f"  Layers       : {arch['n_layers']}")
            print(f"  Hidden size  : {arch['hidden_size']}")
            print(f"  Intermediate : {arch['intermediate_size']}")
            print(f"  MLP neurons  : {arch['total_mlp_neurons']:,}")

            # Baseline
            print("\n  Computing baseline PPL ...")
            bp              = evaluate_perplexity(
                model, tokenizer, texts=eval_texts,
                max_seq_len=max_seq, batch_size=batch_sz, device=device,
            )
            baseline_ppl    = bp["perplexity"]
            baseline_params = count_parameters(model)
            baseline_flops  = estimate_mlp_flops(model, seq_len=max_seq)
            print(f"  Baseline PPL = {baseline_ppl:.4f}")

            # Scores and calibration inputs (computed once per model)
            print("\n  Computing pruning scores ...")
            layers     = get_transformer_layers(model)
            all_scores = [compute_bound_scores_and_R(l)[0] for l in layers]

            # Only collect calibration inputs when at least one residual method is requested
            needs_calib = any(m != "pure_delete" for m in METHODS)
            if needs_calib:
                print("  Collecting calibration inputs (train) ...")
                train_r = collect_mlp_inputs(
                    model, tokenizer, RECONSTRUCTION_TRAIN_PROMPTS, device,
                    max_seq_len=cfg.get("max_seq_len", 128),
                )
                print("  Collecting calibration inputs (held-out) ...")
                heldout_r = collect_mlp_inputs(
                    model, tokenizer, RECONSTRUCTION_HELDOUT_PROMPTS, device,
                    max_seq_len=cfg.get("max_seq_len", 128),
                )
            else:
                train_r   = None
                heldout_r = None

            # ── Per-alpha loop ────────────────────────────────────────────
            for alpha in ALPHAS:
                prune_per_layer = []
                for scores in all_scores:
                    pi, _ = select_by_budget(scores, alpha, float(scores.sum()))
                    prune_per_layer.append(pi)

                total_pruned = sum(len(pi) for pi in prune_per_layer)
                total_n      = sum(s.numel() for s in all_scores)
                pct          = 100.0 * total_pruned / total_n if total_n else 0.0

                if total_pruned == 0:
                    print(f"  alpha={_k(alpha)}: 0 neurons selected -- skipping")
                    continue

                keep_per_layer = []
                for scores, pi in zip(all_scores, prune_per_layer):
                    p_set = set(pi.tolist())
                    keep_per_layer.append(
                        torch.tensor(
                            [j for j in range(scores.numel()) if j not in p_set],
                            dtype=torch.long,
                        )
                    )

                print(f"\n  alpha={_k(alpha)}  pruned={total_pruned} ({pct:.3f}%)")

                # Row factory
                def _base_row(method: str, **extra) -> Dict:
                    r = {
                        "model_name":   model_name,
                        "dtype":        dtype_str,
                        "alpha":        alpha,
                        "n_pruned":     total_pruned,
                        "pct_pruned":   round(pct, 4),
                        "baseline_ppl": round(baseline_ppl, 4),
                        "method":       method,
                        **arch,
                    }
                    r.update(extra)
                    return r

                def _fill_ppl(row: Dict, m) -> tuple:
                    """Evaluate m, populate row, return (ppl, delta)."""
                    fp_ok    = verify_forward_pass(m, tokenizer, device)
                    ppl_info = evaluate_perplexity(
                        m, tokenizer, texts=eval_texts,
                        max_seq_len=max_seq, batch_size=batch_sz, device=device,
                    )
                    ppl   = ppl_info["perplexity"]
                    delta = ppl - baseline_ppl
                    pp    = count_parameters(m)
                    pf    = estimate_mlp_flops(m, seq_len=max_seq)
                    row.update({
                        "perplexity":              round(ppl,   4),
                        "perplexity_delta":        round(delta, 4),
                        "relative_ppl_increase_pct": round(
                            100.0 * delta / baseline_ppl, 4),
                        "forward_pass_ok":         fp_ok,
                        "mlp_params_red_pct":      round(
                            100 * (1 - pp["mlp"] / baseline_params["mlp"]), 4),
                        "flops_red_pct":           round(
                            100 * (1 - pf["total_flops"] /
                                   baseline_flops["total_flops"]), 4),
                    })
                    return round(ppl, 4), round(delta, 4)

                alpha_rows: List[Dict] = []
                dppl_delete: Optional[float] = None   # for damage-reduction calc

                # ── Method loop ───────────────────────────────────────────
                for method in METHODS:

                    # ── pure_delete ──────────────────────────────────────
                    if method == "pure_delete":
                        print(f"    [pure_delete]  ", end="", flush=True)
                        row_del      = _base_row("pure_delete")
                        pruned_model = None
                        try:
                            pruned_model, _ = prune_model_by_layer_indices(
                                model, prune_per_layer,
                                label=f"sc_del_{_k(alpha)}",
                            )
                            ppl_del, dppl_delete = _fill_ppl(row_del, pruned_model)
                            print(
                                f"PPL={ppl_del:.4f}  dPPL={dppl_delete:+.4f}"
                                f"  rel={row_del['relative_ppl_increase_pct']:+.2f}%"
                            )
                        except Exception as exc:
                            logger.error("pure_delete %s a=%s: %s",
                                         model_name, _k(alpha), exc, exc_info=True)
                            row_del["notes"] = f"ERROR: {exc}"
                            print(f"FAILED: {exc}")
                        finally:
                            if pruned_model is not None:
                                del pruned_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        alpha_rows.append(row_del)
                        continue

                    # ── residual methods ─────────────────────────────────
                    # Parse top_k from method name
                    top_k = _parse_top_k(method)
                    if top_k == -1:
                        logger.warning("Unknown method '%s' — skipping", method)
                        continue

                    if train_r is None or heldout_r is None:
                        logger.error("Calibration inputs missing for residual method '%s'", method)
                        continue

                    print(f"    [{method}]  ", end="", flush=True)
                    row_res      = _base_row(method,
                                             ridge_lambda=BEST_RESIDUAL_LAM,
                                             tau=BEST_RESIDUAL_TAU)
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
                        row_res["n_stable_layers"]   = n_s
                        row_res["n_unstable_layers"] = n_u
                        row_res["recon_time_s"]      = build_info.get("total_recon_time_s")
                        row_res["peak_gpu_mb"]       = build_info.get("peak_gpu_mb")

                        # Aggregate per-layer diagnostics
                        active_s = [
                            s for s in build_info["sanity_per_layer"]
                            if s.get("n_pruned", 0) > 0 and s.get("stable", False)
                        ]
                        if active_s:
                            na = len(active_s)
                            row_res["train_improvement_pct"]   = round(
                                sum(s["train_improvement_pct"]    for s in active_s) / na, 2)
                            row_res["heldout_improvement_pct"] = round(
                                sum(s["heldout_improvement_pct"]  for s in active_s) / na, 2)
                            row_res["overfit_gap_pct"]         = round(
                                sum(s["overfit_gap_pct"]          for s in active_s) / na, 2)
                            row_res["update_norm_ratio_mean"]  = round(
                                sum(s.get("update_norm_ratio", 0) for s in active_s) / na, 6)

                        if build_info.get("all_unstable", False):
                            row_res["notes"] = "SKIPPED_PPL:all_layers_unstable"
                            print(
                                f"all layers unstable "
                                f"(stable={n_s} unstable={n_u})"
                            )
                            alpha_rows.append(row_res)
                            _flush_csv(csv_path, alpha_rows)
                            all_results.extend(alpha_rows)
                            alpha_rows = []
                            continue

                        ppl_res, dppl_res = _fill_ppl(row_res, pruned_model)

                        # Damage reduction vs pure_delete
                        if dppl_delete is not None and abs(dppl_delete) > 1e-6:
                            row_res["damage_reduction_pct"] = round(
                                100.0 * (dppl_delete - dppl_res) / dppl_delete, 2)

                        t_s   = build_info.get("total_recon_time_s", float("nan"))
                        gb_mb = build_info.get("peak_gpu_mb", float("nan"))
                        dm_str = (
                            f"  dmg_red={row_res['damage_reduction_pct']:+.1f}%"
                            if "damage_reduction_pct" in row_res else ""
                        )
                        print(
                            f"PPL={ppl_res:.4f}  dPPL={dppl_res:+.4f}"
                            f"  rel={row_res['relative_ppl_increase_pct']:+.2f}%"
                            + dm_str
                            + f"  t={t_s:.1f}s  gpu={gb_mb:.0f}MB"
                        )

                    except Exception as exc:
                        logger.error("[%s] %s a=%s: %s",
                                     method, model_name, _k(alpha), exc, exc_info=True)
                        row_res["notes"] = f"ERROR: {exc}"
                        print(f"FAILED: {exc}")
                    finally:
                        if pruned_model is not None:
                            del pruned_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    alpha_rows.append(row_res)

                # ── end method loop ───────────────────────────────────────
                # Flush partial results
                _flush_csv(csv_path, alpha_rows)
                all_results.extend(alpha_rows)

        except torch.cuda.OutOfMemoryError as oom:
            logger.error("OOM for %s: %s", model_name, oom)
            print(f"  *** OOM -- skipping {model_name} ***")
            err_row = {"model_name": model_name, "notes": f"OOM: {oom}"}
            _flush_csv(csv_path, [err_row])
            all_results.append(err_row)

        except Exception as exc:
            logger.error("Failed %s: %s", model_name, exc, exc_info=True)
            print(f"  *** ERROR -- skipping {model_name}: {exc} ***")
            err_row = {"model_name": model_name, "notes": f"ERROR: {exc}"}
            _flush_csv(csv_path, [err_row])
            all_results.append(err_row)

        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n  Released {model_name} from memory.\n")

    # ── Final summary table ──────────────────────────────────────────────────
    ppl_rows = [r for r in all_results if "perplexity" in r]
    W = 160
    print(f"\n{'=' * W}")
    print(f"SCALING SUMMARY  (n_eval={n_eval})")
    print(f"{'─' * W}")
    print(
        f"  {'model':>22}  {'alpha':>7}  "
        f"{'pruned':>8}  {'tot_mlp':>8}  {'prn%':>6}  {'par%':>7}  {'flop%':>7}  "
        f"{'method':>30}  "
        f"{'bPPL':>8}  {'PPL':>9}  {'dPPL':>9}  {'rel%':>7}  {'dmg_red':>9}  "
        f"{'t_s':>6}  {'gpu_MB':>7}"
    )
    print(f"{'─' * W}")
    for r in ppl_rows:
        n_prn   = r.get("n_pruned",              "?")
        tot_mlp = r.get("total_mlp_neurons",     "?")
        pct_prn = r.get("pct_pruned",            float("nan"))
        par_red = r.get("mlp_params_red_pct",    float("nan"))
        flp_red = r.get("flops_red_pct",         float("nan"))
        dm      = r.get("damage_reduction_pct",  float("nan"))
        rel     = r.get("relative_ppl_increase_pct", float("nan"))
        t_s     = r.get("recon_time_s",          float("nan"))
        gpu_mb  = r.get("peak_gpu_mb",           float("nan"))

        par_s  = f"{par_red:+6.2f}%" if par_red == par_red else "    nan%"
        flp_s  = f"{flp_red:+6.2f}%" if flp_red == flp_red else "    nan%"
        dm_s   = f"{dm:+8.1f}%"      if dm  == dm           else "      nan%"
        rel_s  = f"{rel:+7.2f}%"     if rel == rel           else "    nan%"
        pct_s  = f"{pct_prn:5.2f}%"  if pct_prn == pct_prn  else "  nan%"
        t_s_s  = f"{t_s:6.1f}"       if t_s == t_s           else "   nan"
        gpu_s  = f"{gpu_mb:7.0f}"    if gpu_mb == gpu_mb     else "    nan"

        print(
            f"  {r['model_name'][-22:]:>22}  {_k(r['alpha']):>7}  "
            f"{n_prn:>8}  {tot_mlp:>8}  {pct_s}  {par_s}  {flp_s}  "
            f"{r['method']:>30}  "
            f"{r['baseline_ppl']:>8.4f}  {r['perplexity']:>9.4f}  "
            f"{r['perplexity_delta']:>+9.4f}  {rel_s}  {dm_s}  "
            f"{t_s_s}  {gpu_s}"
        )
    print(f"{'=' * W}\n")

    # ── JSON report ──────────────────────────────────────────────────────────
    report = {
        "timestamp":   ts,
        "mode":        "scaling_recon",
        "models":      model_list,
        "alphas":      ALPHAS,
        "methods":     METHODS,
        "n_eval":      n_eval,
        "best_lambda": BEST_RESIDUAL_LAM,
        "best_tau":    BEST_RESIDUAL_TAU,
        "results":     all_results,
    }
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"CSV    : {csv_path}")
    print(f"Report : {json_path}\n")
