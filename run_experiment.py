"""
run_experiment.py
=================
Entry point for the qwen_swiglu_pruning research prototype.

Usage
-----
Full experiment:
    python run_experiment.py --config configs/default.yaml

Debug mode (no pruning; runs all correctness + scoring checks):
    python run_experiment.py --config configs/default.yaml --debug-pruning

Diagnostic mode (no pruning; logs per-layer MLP norms):
    python run_experiment.py --config configs/default.yaml --diagnostics-only

Quick override of config values:
    python run_experiment.py --config configs/default.yaml \
        --pruning-ratios 0.0 0.1 --methods rmsnorm_bound_angle random
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.bound_analysis import run_bound_analysis_mode
from src.scaling import run_scaling_recon_mode
from src.target_pruning import run_target_pruning_mode
from src.moe_pruning import run_moe_target_pruning_mode
from src.benchmark import run_benchmark_mode
from src.merging import (
    run_bound_merge_mode,
    run_bound_merge_stable_mode,
    run_debug_merge_mode,
    run_reconstruction_merge_mode,
    run_residual_reconstruction_mode,
    run_reconstruction_best_mode,
)
from src.debug import run_debug_mode
from src.diagnostics import run_diagnostics
from src.evaluation import evaluate_perplexity, load_eval_dataset, run_generation_tests
from src.flops import estimate_mlp_flops
from src.model_utils import clone_model, count_parameters, load_model_and_tokenizer
from src.pruning import prune_model, verify_forward_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / device
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: Dict) -> str:
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "model_name",
    "pruning_method",
    "pruning_ratio",
    "total_params_before",
    "total_params_after",
    "mlp_params_before",
    "mlp_params_after",
    "mlp_params_reduction_pct",
    "mlp_flops_before",
    "mlp_flops_after",
    "mlp_flops_reduction_pct",
    "perplexity",
    "perplexity_delta",
    "forward_pass_ok",
    "notes",
]


def save_results(results: List[Dict], output_dir: str) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    stem = f"results_{ts}"

    csv_path  = os.path.join(output_dir, stem + ".csv")
    json_path = os.path.join(output_dir, stem + ".json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results → %s  and  %s", csv_path, json_path)
    return csv_path, json_path


def save_generations(gen_results: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"generations_{ts}.json")
    with open(path, "w") as f:
        json.dump(gen_results, f, indent=2)
    logger.info("Generations → %s", path)


def print_result_row(row: Dict) -> None:
    print(
        f"  method={row['pruning_method']:20s}  ratio={row['pruning_ratio']:.0%}"
        f"  PPL={row['perplexity']:.3f}  ΔPPL={row['perplexity_delta']:+.3f}"
        f"  MLP {row['mlp_params_before']:,} → {row['mlp_params_after']:,}"
        f"  FLOPs↓{row['mlp_flops_reduction_pct']:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: Dict, args) -> None:
    set_seed(cfg.get("seed", 42))
    device     = resolve_device(cfg)
    output_dir = cfg.get("output_dir", "results")

    # ── SCALING RECON MODE  (loads its own models — skip global model load) ───
    if args.scaling_recon:
        run_scaling_recon_mode(cfg, device=device, output_dir=output_dir)
        return

    # ── TARGET-PRUNING SCALING MODE  (loads its own models) ─────────────────
    if args.target_pruning_scaling:
        run_target_pruning_mode(
            cfg,
            device=device,
            output_dir=output_dir,
            models_override=args.models or None,
            targets_override=args.target_pruning_percents or None,
            methods_override=args.methods or None,
            n_eval_override=args.n_eval or None,
            eval_datasets_override=args.eval_datasets or None,
            selectors_override=args.selectors or None,
        )
        return

    # ── MOE TARGET-PRUNING MODE ─────────────────────────────────────────────
    if args.moe_target_pruning:
        run_moe_target_pruning_mode(
            cfg,
            device=device,
            output_dir=output_dir,
            models_override=args.models or None,
            targets_override=args.target_pruning_percents or None,
            methods_override=args.methods or None,
            n_eval_override=args.n_eval or None,
            eval_datasets_override=args.eval_datasets or None,
        )
        return

    # Load model
    model, tokenizer, resolved_name = load_model_and_tokenizer(
        model_name    = cfg["model_name"],
        fallback_name = cfg.get("model_name_fallback"),
        device        = device,
        dtype_str     = cfg.get("dtype", "float32"),
    )

    # ── BENCHMARK MODE ─────────────────────────────────────────────────────────
    if args.benchmark:
        run_benchmark_mode(
            cfg, device=device, output_dir=output_dir,
            model_configs=[("baseline", model, tokenizer)],
            prompt_lens=cfg.get("benchmark_prompt_lens", [128, 512, 1024]),
            max_new_tokens=int(cfg.get("benchmark_max_new_tokens", 128)),
            n_repeats=int(cfg.get("benchmark_n_repeats", 5)),
        )
        return

    # ── BOUND ANALYSIS MODE ────────────────────────────────────────────────────
    if args.bound_analysis:
        run_bound_analysis_mode(
            model, tokenizer, cfg,
            device=device,
            output_dir=output_dir,
            skip_ppl=args.no_ppl,
            skip_activation=args.no_activation_verification,
        )
        return

    # ── BOUND PPL ONLY MODE ────────────────────────────────────────────────────
    if args.bound_ppl_only:
        run_bound_analysis_mode(
            model, tokenizer, cfg,
            device=device, output_dir=output_dir,
            skip_ppl=False, skip_activation=True,
        )
        return

    # ── ACTIVATION VERIFICATION ONLY MODE ─────────────────────────────────────
    if args.activation_verification_only:
        run_bound_analysis_mode(
            model, tokenizer, cfg,
            device=device, output_dir=output_dir,
            skip_ppl=True, skip_activation=False,
        )
        return

    # ── BOUND MERGE MODE ───────────────────────────────────────────────────────
    if args.bound_merge:
        run_bound_merge_mode(model, tokenizer, cfg, device=device, output_dir=output_dir)
        return

    # ── RECONSTRUCTION MERGE MODE ─────────────────────────────────────────────
    if args.reconstruction_merge:
        run_residual_reconstruction_mode(model, tokenizer, cfg, device=device, output_dir=output_dir)
        return

    # ── RECONSTRUCTION BEST MODE ──────────────────────────────────────────────
    if args.reconstruction_best:
        run_reconstruction_best_mode(model, tokenizer, cfg, device=device, output_dir=output_dir)
        return

    # ── STABLE MERGE MODE ─────────────────────────────────────────────────────
    if args.bound_merge_stable:
        run_bound_merge_stable_mode(model, tokenizer, cfg, device=device, output_dir=output_dir)
        return

    # ── DEBUG MERGE MODE ───────────────────────────────────────────────────────
    if args.debug_merge:
        run_debug_merge_mode(model, tokenizer, cfg, device=device, output_dir=output_dir)
        return

    # ── DEBUG MODE ─────────────────────────────────────────────────────────────
    if args.debug_pruning:
        run_debug_mode(model, tokenizer, cfg, device=device, output_dir=output_dir)
        return

    # ── DIAGNOSTICS ONLY MODE ──────────────────────────────────────────────────
    if args.diagnostics_only:
        stats = run_diagnostics(model, tokenizer,
                                max_seq_len=cfg.get("max_seq_len", 128), device=device)
        diag_path = os.path.join(output_dir, "diagnostics.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Diagnostics → %s", diag_path)
        return

    # ── BASELINE ───────────────────────────────────────────────────────────────
    logger.info("Evaluating BASELINE model …")
    eval_texts = load_eval_dataset(
        cfg.get("max_eval_samples", 512),
        use_fallback_corpus=cfg.get("use_fallback_corpus", True),
    )

    baseline_ppl_info = evaluate_perplexity(
        model, tokenizer,
        texts       = eval_texts,
        max_samples = cfg.get("max_eval_samples", 512),
        max_seq_len = cfg.get("max_seq_len", 512),
        batch_size  = cfg.get("batch_size", 4),
        device      = device,
    )
    baseline_ppl = baseline_ppl_info["perplexity"]
    logger.info("Baseline perplexity: %.4f", baseline_ppl)

    logger.info("Running baseline generation tests …")
    baseline_gens = run_generation_tests(model, tokenizer, device=device)

    baseline_params = count_parameters(model)
    baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))

    # Record original param count for independence checks
    orig_total_params = baseline_params["total"]

    # ── EXPERIMENT GRID ────────────────────────────────────────────────────────
    methods = args.methods or cfg.get("pruning_methods", ["rmsnorm_bound_angle"])
    ratios  = [float(r) for r in (args.pruning_ratios or cfg.get("pruning_ratios", [0.0, 0.1, 0.2]))]
    seed    = cfg.get("seed", 42)

    all_results: List[Dict]   = []
    all_generations: Dict     = {"baseline": baseline_gens}

    print(f"\n{'═'*70}")
    print(f"Grid: {len(methods)} methods × {len(ratios)} ratios = {len(methods)*len(ratios)} runs")
    print(f"{'═'*70}\n")

    for method in methods:
        for ratio in ratios:
            run_tag = f"{method}_r{int(ratio*100):02d}pct"
            print(f"\n{'─'*70}")
            print(f"  Run: {run_tag}")
            print(f"{'─'*70}")

            # ── Independence assertion ──────────────────────────────────────
            current_params = count_parameters(model)["total"]
            if current_params != orig_total_params:
                logger.error(
                    "INDEPENDENCE BUG: original model changed from %d to %d params "
                    "before run (%s, %.0f%%)!",
                    orig_total_params, current_params, method, ratio * 100,
                )
            else:
                logger.debug(
                    "Independence OK: original model has %d params before run (%s, %.0f%%)",
                    current_params, method, ratio * 100,
                )

            try:
                pruned_model, prune_info = prune_model(
                    model      = model,
                    prune_ratio = ratio,
                    method     = method,
                    seed       = seed,
                )

                # Verify original is still intact after pruning
                post_params = count_parameters(model)["total"]
                if post_params != orig_total_params:
                    logger.error(
                        "INDEPENDENCE BUG: original model changed AFTER prune_model() "
                        "for (%s, %.0f%%). Before=%d After=%d",
                        method, ratio * 100, orig_total_params, post_params,
                    )

                fp_ok = verify_forward_pass(pruned_model, tokenizer, device)

                pruned_params = count_parameters(pruned_model)
                pruned_flops  = estimate_mlp_flops(pruned_model, seq_len=cfg.get("max_seq_len", 512))

                ppl_info = evaluate_perplexity(
                    pruned_model, tokenizer,
                    texts       = eval_texts,
                    max_samples = cfg.get("max_eval_samples", 512),
                    max_seq_len = cfg.get("max_seq_len", 512),
                    batch_size  = cfg.get("batch_size", 4),
                    device      = device,
                )
                ppl  = ppl_info["perplexity"]
                gens = run_generation_tests(pruned_model, tokenizer, device=device)
                all_generations[run_tag] = gens

                flop_before   = baseline_flops["total_flops"]
                flop_after    = pruned_flops["total_flops"]
                flop_red_pct  = 100.0 * (1.0 - flop_after / flop_before) if flop_before > 0 else 0.0
                mlp_before    = baseline_params["mlp"]
                mlp_after     = pruned_params["mlp"]
                mlp_red_pct   = 100.0 * (1.0 - mlp_after / mlp_before) if mlp_before > 0 else 0.0

                row = {
                    "model_name":              resolved_name,
                    "pruning_method":          method,
                    "pruning_ratio":           ratio,
                    "total_params_before":     baseline_params["total"],
                    "total_params_after":      pruned_params["total"],
                    "mlp_params_before":       mlp_before,
                    "mlp_params_after":        mlp_after,
                    "mlp_params_reduction_pct": mlp_red_pct,
                    "mlp_flops_before":        flop_before,
                    "mlp_flops_after":         flop_after,
                    "mlp_flops_reduction_pct": flop_red_pct,
                    "perplexity":              ppl,
                    "perplexity_delta":        ppl - baseline_ppl,
                    "forward_pass_ok":         fp_ok,
                    "notes":                   "",
                    "per_layer_d_ff":          prune_info["per_layer"],
                    "generation_examples":     gens,
                }
                all_results.append(row)
                print_result_row(row)

            except Exception as exc:  # noqa: BLE001
                logger.error("Run %s failed: %s", run_tag, exc, exc_info=True)
                row = {f: "" for f in RESULT_FIELDS}
                row.update({
                    "model_name":     resolved_name,
                    "pruning_method": method,
                    "pruning_ratio":  ratio,
                    "notes":          f"ERROR: {exc}",
                })
                all_results.append(row)

            del pruned_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── SAVE ───────────────────────────────────────────────────────────────────
    csv_path, json_path = save_results(all_results, output_dir)
    save_generations(all_generations, output_dir)

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("SUMMARY")
    print(f"{'═'*70}")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'─'*70}")
    print(f"  {'Method':<22} {'Ratio':>6}  {'PPL':>8}  {'ΔPPL':>8}  {'FLOPs↓':>8}  {'MLP↓':>8}")
    print(f"{'─'*70}")
    for r in all_results:
        try:
            print(
                f"  {str(r['pruning_method']):<22} {float(r['pruning_ratio']):>6.0%}"
                f"  {float(r['perplexity']):>8.3f}  {float(r['perplexity_delta']):>+8.3f}"
                f"  {float(r['mlp_flops_reduction_pct']):>7.1f}%"
                f"  {float(r['mlp_params_reduction_pct']):>7.1f}%"
            )

        except (ValueError, TypeError):
            print(f"  {r.get('pruning_method','')} ratio={r.get('pruning_ratio','')}  "
                  f"ERROR: {r.get('notes','')}")
    print(f"{'═'*70}")
    print(f"\nResults → {csv_path}\nDetails → {json_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen SwiGLU MLP Pruning Experiment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default="configs/default.yaml")

    # ── Main modes ──────────────────────────────────────────────────────────
    p.add_argument(
        "--bound-analysis", action="store_true",
        help=(
            "Score distributions + threshold-based PPL experiments.\n"
            "Combine with --no-ppl / --no-activation-verification to reduce scope."
        ),
    )
    p.add_argument(
        "--bound-ppl-only", action="store_true",
        help=(
            "Run only cumul_score_sum PPL experiments at alpha=1e-4/1e-3/1e-2.\n"
            "No activation verification. Saves extra detail for alpha=1e-4."
        ),
    )
    p.add_argument(
        "--activation-verification-only", action="store_true",
        help=(
            "Compute hook-based activation scores and correlations with bound scores.\n"
            "No pruning performed."
        ),
    )
    p.add_argument(
        "--bound-merge", action="store_true",
        help=(
            "Compare pure_delete vs merge_weight vs merge_activation vs down_reconstruction\n"
            "for cumulative-budget candidates at alpha=1e-4/1e-3/1e-2.\n"
            "Saves PPL comparison table and per-neuron diagnostics."
        ),
    )
    p.add_argument(
        "--reconstruction-merge", action="store_true",
        help=(
            "Full residual reconstruction grid: pure_delete + activation merge baselines\n"
            "+ residual_lam×tau grid (5×5=25 configs) for alpha in {1e-4,1e-3,1e-2}.\n"
            "Uses train/held-out calibration split. Reports all reconstruction metrics."
        ),
    )
    p.add_argument(
        "--reconstruction-best", action="store_true",
        help=(
            "Fast best-config evaluation: pure_delete + merge_act_ridge_1e-2 + \n"
            "resid_lam1e-2_tau1.0 for alpha in {1e-4,1e-3,1e-2}.\n"
            "Uses reconstruction_eval_samples (default 256) from WikiText-2.\n"
            "Reports full held-out reconstruction metrics and overfit diagnostics."
        ),
    )
    p.add_argument(
        "--scaling-recon", action="store_true",
        help=(
            "Multi-model scaling experiment: pure_delete + resid_lam1e-2_tau1.0\n"
            "across all models in scaling_models (default: 0.5B and 1.5B).\n"
            "Alphas: scaling_alphas (default: 1e-4,1e-3,2e-3,3e-3,5e-3,1e-2).\n"
            "Models loaded one at a time; partial CSV flushed after each (model, alpha).\n"
            "Uses bfloat16 automatically if CUDA supports it.\n"
            "Skips models gracefully on OOM."
        ),
    )
    p.add_argument(
        "--bound-merge-stable", action="store_true",
        help=(
            "Test stabilized activation-merge variants at alpha=1e-4/1e-3/1e-2.\n"
            "Variants: clipped beta (4 clip levels), ridge-regularized beta (5 lambdas),\n"
            "and ridge+clip combinations (3 configs).\n"
            "Reports: beta stats, update magnitudes, train+held-out reconstruction error\n"
            "(overfitting detection), and WikiText-2 PPL for each variant."
        ),
    )
    p.add_argument(
        "--debug-merge", action="store_true",
        help=(
            "Diagnose merging quality WITHOUT running PPL.\n"
            "Reports: beta statistics, down-proj update magnitudes,\n"
            "isolated per-layer MLP reconstruction errors,\n"
            "end-to-end logit diffs, end-to-end MLP output errors."
        ),
    )
    p.add_argument(
        "--diagnostics-only", action="store_true",
        help="Run diagnostic mode only (no pruning, no PPL).",
    )
    p.add_argument(
        "--debug-pruning", action="store_true",
        help=(
            "Run a tiny-ratio pruning diagnostic.\n"
            "Sweeps very small prune fractions to find where PPL first degrades."
        ),
    )
    p.add_argument(
        "--no-ppl", action="store_true",
        help="Skip PPL evaluation in --bound-analysis mode.",
    )
    p.add_argument(
        "--no-activation-verification", action="store_true",
        help="Skip activation verification in --bound-analysis mode.",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Override pruning methods from config (e.g. --methods random down_norm).",
    )
    p.add_argument(
        "--pruning-ratios", nargs="+", type=float, default=None,
        help="Override pruning ratios from config (e.g. --pruning-ratios 0.1 0.2).",
    )
    p.add_argument(
        "--target-pruning-scaling", action="store_true",
        help=(
            "Fixed-percentage MLP pruning experiment.\n"
            "Selects neurons by global score rank to reach exact target percentages,\n"
            "instead of using alpha (score-mass budget). Enables fair cross-model\n"
            "comparison at identical compression levels.\n"
            "Use --models, --target-pruning-percents, --methods, --n-eval to configure."
        ),
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Model names for --target-pruning-scaling "
            "(e.g. --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B). "
            "Overrides scaling_models in the config."
        ),
    )
    p.add_argument(
        "--target-pruning-percents", nargs="+", type=float, default=None,
        help=(
            "Target pruning percentages for --target-pruning-scaling "
            "(e.g. --target-pruning-percents 2 4 6 8). "
            "Overrides target_pruning_percents in the config."
        ),
    )
    p.add_argument(
        "--n-eval", type=int, default=None,
        help=(
            "Number of evaluation samples per dataset "
            "(e.g. --n-eval 256). "
            "Overrides reconstruction_eval_samples in the config."
        ),
    )
    p.add_argument(
        "--eval-datasets", nargs="+", default=None,
        dest="eval_datasets",
        help=(
            "Evaluation datasets for --target-pruning-scaling "
            "(e.g. --eval-datasets wikitext2 c4 wikitext103). "
            "Supported: wikitext2, c4, wikitext103, lambada. "
            "Overrides eval_datasets in the config."
        ),
    )
    p.add_argument(
        "--selectors", nargs="+", default=None,
        help=(
            "Pruning selector(s) for --target-pruning-scaling "
            "(e.g. --selectors rmsnorm_bound down_norm random_seed0). "
            "Supported: rmsnorm_bound, down_norm, activation_score, "
            "random_seed0, random_seed1, random_seed2. "
            "Overrides selectors in the config."
        ),
    )
    p.add_argument(
        "--moe-target-pruning", action="store_true",
        dest="moe_target_pruning",
        help=(
            "Expert-wise structured MLP channel pruning for MoE models "
            "(e.g. Qwen3-30B-A3B). "
            "Uses router-aware calibration and per-expert channel scoring. "
            "Does NOT prune router weights or remove entire experts. "
            "Use --models, --target-pruning-percents, --methods, --n-eval to configure."
        ),
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help=(
            "Run inference latency / throughput / memory benchmark. "
            "Requires --config with model_name set. "
            "Benchmarks prompt_lens 128/512/1024 with max_new_tokens=128. "
            "Use benchmark_prompt_lens, benchmark_max_new_tokens, "
            "benchmark_n_repeats in config to customise."
        ),
    )
    return p


def main():
    p    = parse_args()
    args = p.parse_args()

    with open(args.config) as fh:
        import yaml
        cfg = yaml.safe_load(fh)

    run_experiment(cfg, args)


if __name__ == "__main__":
    main()
