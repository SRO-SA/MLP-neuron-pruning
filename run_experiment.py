"""
run_experiment.py
=================
Entry point for the qwen_swiglu_pruning research prototype.

Usage
-----
Full experiment:
    python run_experiment.py --config configs/default.yaml

Diagnostic mode only (no pruning):
    python run_experiment.py --config configs/default.yaml --diagnostics-only

Override config values inline:
    python run_experiment.py --config configs/default.yaml \
        --pruning-ratios 0.0 0.1 0.2  --methods rmsnorm_bound_angle random

Results are saved to the output_dir specified in the config (default: results/).
"""

from __future__ import annotations

import argparse
import copy
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

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

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
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(cfg: Dict) -> str:
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

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


def save_results(results: List[Dict], output_dir: str, tag: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"results_{ts}{('_' + tag) if tag else ''}"

    csv_path  = os.path.join(output_dir, stem + ".csv")
    json_path = os.path.join(output_dir, stem + ".json")

    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # JSON (includes generation examples)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s and %s", csv_path, json_path)
    return csv_path, json_path


def save_generations(gen_results: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"generations_{ts}.json")
    with open(path, "w") as f:
        json.dump(gen_results, f, indent=2)
    logger.info("Generations saved to %s", path)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_result_row(row: Dict) -> None:
    print(
        f"  method={row['pruning_method']:20s}  ratio={row['pruning_ratio']:.0%}"
        f"  PPL={row['perplexity']:.3f}  ΔPPL={row['perplexity_delta']:+.3f}"
        f"  MLP params: {row['mlp_params_before']:,} → {row['mlp_params_after']:,}"
        f"  FLOPs↓ {row['mlp_flops_reduction_pct']:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: Dict, args) -> None:
    set_seed(cfg.get("seed", 42))
    device     = resolve_device(cfg)
    output_dir = cfg.get("output_dir", "results")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, tokenizer, resolved_name = load_model_and_tokenizer(
        model_name    = cfg["model_name"],
        fallback_name = cfg.get("model_name_fallback"),
        device        = device,
        dtype_str     = cfg.get("dtype", "float32"),
    )

    # -----------------------------------------------------------------------
    # Diagnostics-only mode
    # -----------------------------------------------------------------------
    if args.diagnostics_only:
        stats = run_diagnostics(
            model, tokenizer,
            max_seq_len = cfg.get("max_seq_len", 128),
            device      = device,
        )
        diag_path = os.path.join(output_dir, "diagnostics.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Diagnostics saved to %s", diag_path)
        return

    # -----------------------------------------------------------------------
    # Baseline: perplexity & generation on the original model
    # -----------------------------------------------------------------------
    logger.info("Evaluating BASELINE model …")
    eval_texts = load_eval_dataset(cfg.get("max_eval_samples", 512))

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

    # Baseline FLOPs and params
    baseline_params = count_parameters(model)
    baseline_flops  = estimate_mlp_flops(model, seq_len=cfg.get("max_seq_len", 512))

    # -----------------------------------------------------------------------
    # Experiment grid
    # -----------------------------------------------------------------------
    methods = args.methods or cfg.get("pruning_methods", ["rmsnorm_bound_angle"])
    ratios  = [float(r) for r in (args.pruning_ratios or cfg.get("pruning_ratios", [0.0, 0.1, 0.2]))]

    all_results: List[Dict]  = []
    all_generations: Dict    = {"baseline": baseline_gens}
    seed = cfg.get("seed", 42)

    print(f"\n{'═'*70}")
    print(f"Experiment grid: {len(methods)} methods × {len(ratios)} ratios "
          f"= {len(methods)*len(ratios)} runs")
    print(f"{'═'*70}\n")

    for method in methods:
        for ratio in ratios:
            run_tag = f"{method}_r{int(ratio*100):02d}"
            print(f"\n{'─'*70}")
            print(f"  Run: {run_tag}")
            print(f"{'─'*70}")

            try:
                # Prune a fresh clone for each (method, ratio) pair
                pruned_model, prune_info = prune_model(
                    model      = model,
                    prune_ratio = ratio,
                    method     = method,
                    seed       = seed,
                )

                # Forward-pass sanity check
                fp_ok = verify_forward_pass(pruned_model, tokenizer, device)

                # Params after pruning
                pruned_params = count_parameters(pruned_model)
                pruned_flops  = estimate_mlp_flops(
                    pruned_model, seq_len=cfg.get("max_seq_len", 512)
                )

                # Perplexity
                ppl_info = evaluate_perplexity(
                    pruned_model, tokenizer,
                    texts       = eval_texts,
                    max_samples = cfg.get("max_eval_samples", 512),
                    max_seq_len = cfg.get("max_seq_len", 512),
                    batch_size  = cfg.get("batch_size", 4),
                    device      = device,
                )
                ppl = ppl_info["perplexity"]

                # Generation
                gens = run_generation_tests(pruned_model, tokenizer, device=device)
                all_generations[run_tag] = gens

                # FLOP reduction
                flop_before = baseline_flops["total_flops"]
                flop_after  = pruned_flops["total_flops"]
                flop_red_pct = 100.0 * (1.0 - flop_after / flop_before) if flop_before > 0 else 0.0

                # Param reduction
                mlp_before = baseline_params["mlp"]
                mlp_after  = pruned_params["mlp"]
                mlp_red_pct = 100.0 * (1.0 - mlp_after / mlp_before) if mlp_before > 0 else 0.0

                row = {
                    "model_name":             resolved_name,
                    "pruning_method":         method,
                    "pruning_ratio":          ratio,
                    "total_params_before":    baseline_params["total"],
                    "total_params_after":     pruned_params["total"],
                    "mlp_params_before":      mlp_before,
                    "mlp_params_after":       mlp_after,
                    "mlp_params_reduction_pct": mlp_red_pct,
                    "mlp_flops_before":       flop_before,
                    "mlp_flops_after":        flop_after,
                    "mlp_flops_reduction_pct": flop_red_pct,
                    "perplexity":             ppl,
                    "perplexity_delta":       ppl - baseline_ppl,
                    "forward_pass_ok":        fp_ok,
                    "notes":                  "",
                    # Extended info (kept in JSON only)
                    "per_layer_d_ff":         prune_info["per_layer"],
                    "generation_examples":    gens,
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

            # Free memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    csv_path, json_path = save_results(all_results, output_dir)
    save_generations(all_generations, output_dir)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'═'*70}")
    print("SUMMARY")
    print(f"{'═'*70}")
    print(f"{'Method':<22} {'Ratio':>6}  {'PPL':>8}  {'ΔPPL':>8}  {'FLOPs↓':>8}  {'MLP↓':>8}")
    print(f"{'─'*70}")
    for r in all_results:
        try:
            print(
                f"{str(r['pruning_method']):<22} {float(r['pruning_ratio']):>6.0%}"
                f"  {float(r['perplexity']):>8.3f}  {float(r['perplexity_delta']):>+8.3f}"
                f"  {float(r['mlp_flops_reduction_pct']):>7.1f}%"
                f"  {float(r['mlp_params_reduction_pct']):>7.1f}%"
            )
        except (ValueError, TypeError):
            print(f"  {r.get('pruning_method','')} ratio={r.get('pruning_ratio','')}  ERROR: {r.get('notes','')}")
    print(f"{'═'*70}")
    print(f"\nResults → {csv_path}")
    print(f"Details  → {json_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Qwen SwiGLU MLP Pruning Experiment")
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    p.add_argument(
        "--diagnostics-only", action="store_true",
        help="Run diagnostic mode only (no pruning); log per-layer MLP norms",
    )
    p.add_argument(
        "--pruning-ratios", nargs="+", type=float, default=None,
        help="Override pruning_ratios from config (e.g. --pruning-ratios 0.0 0.1 0.2)",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Override pruning_methods from config (e.g. --methods random rmsnorm_bound_angle)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    run_experiment(cfg, args)
