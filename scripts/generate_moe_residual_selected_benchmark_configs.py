#!/usr/bin/env python3
"""
generate_moe_residual_selected_benchmark_configs.py
====================================================
Generate YAML configs for the MoE residual selected full benchmark.

Matrix: 4 targets × 3 methods × 2 datasets = 24 configs.

Methods:
  1. pure_delete
  2. residual_nearest_channel_merge_moe
  3. residual_ridge_moe  (residual_lambda=1e-3)

Targets:  2%, 4%, 6%, 8%
Datasets: wikitext2, c4
n_eval:   512  (full benchmark)

Per (target, dataset): pure_delete saves the pruning plan; others load it.
Plan path encodes dataset so wikitext2 and c4 plans stay separate.

Usage:
    python scripts/generate_moe_residual_selected_benchmark_configs.py
    python scripts/generate_moe_residual_selected_benchmark_configs.py --dry-run
    python scripts/generate_moe_residual_selected_benchmark_configs.py --out-dir configs/my_dir
"""
from __future__ import annotations

import argparse
import os

import yaml

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR  = os.path.join(REPO_ROOT, "configs", "moe_residual_selected_benchmark")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ── Benchmark dimensions ────────────────────────────────────────────────────────
TARGETS   = [2, 4, 6, 8]
DATASETS  = ["wikitext2", "c4"]
N_EVAL    = 512
CALIB_N   = 512
SELECTOR  = "rmsnorm_bound"
AGG_MODE  = "p95"
ALIGN     = 16
MAX_LAYER = 0.10
MIN_TOK   = 32

MODEL_ID   = "Qwen/Qwen3-30B-A3B"
MODEL_SLUG = MODEL_ID.replace("/", "_").replace("-", "_")

# ── Method definitions: (method_str, label_suffix, extra_cfg_dict) ──────────────
# Ordered so pure_delete is always first — it saves the pruning plan.
METHODS = [
    ("pure_delete",                        "pure_delete",        {}),
    ("residual_nearest_channel_merge_moe", "nearest_merge",      {}),
    ("residual_ridge_moe",                 "ridge_lam1e-3",      {"residual_lambda": 1e-3}),
]

EXPECTED_TOTAL = len(TARGETS) * len(DATASETS) * len(METHODS)  # 4 × 2 × 3 = 24


def plan_path(target_pct: int, dataset: str) -> str:
    """Canonical pruning plan path — mirrors make_pruning_plan_path()."""
    fname = (
        f"{MODEL_SLUG}_{dataset}_n{N_EVAL}_calib{CALIB_N}"
        f"_{SELECTOR}_{AGG_MODE}_{float(target_pct):.1f}pct_align{ALIGN}.json"
    )
    return os.path.join(RESULTS_DIR, "pruning_plans", fname)


def config_name(target_pct: int, dataset: str, label: str) -> str:
    return f"qwen3_30b_a3b_{dataset}_n{N_EVAL}_target{target_pct}_{label}.yaml"


def build_config(
    target_pct:    int,
    dataset:       str,
    method:        str,
    extra_cfg:     dict,
    is_pure_delete: bool,
) -> dict:
    cfg = {
        # Model
        "scaling_models":              [MODEL_ID],
        "scaling_dtype":               "auto",
        "device_map":                  "auto",
        "expected_expert_layout":      "unpacked",
        # Pruning
        "moe_pruning_mode":            "packed_same_channel",
        "target_pruning_percents":     [float(target_pct)],
        "scaling_methods":             [method],
        "moe_selector":                SELECTOR,
        "moe_same_channel_aggregation": AGG_MODE,
        "moe_channel_alignment":       ALIGN,
        "moe_max_layer_channel_prune_frac": MAX_LAYER,
        "max_expert_frac":             MAX_LAYER,
        "min_expert_tokens":           MIN_TOK,
        # Evaluation
        "eval_datasets":               [dataset],
        "moe_calib_dataset":           dataset,
        "reconstruction_eval_samples": N_EVAL,
        "moe_calib_samples":           CALIB_N,
        "max_seq_len":                 512,
        "batch_size":                  4,
        "use_fallback_corpus":         False,
        # Execution
        "moe_inplace_pruning":         True,
        "moe_smoke_test":              False,
        "seed":                        42,
        # Residual base config
        "residual_tau":                1.0,
        "min_residual_tokens_per_expert": MIN_TOK,
        "solve_residual_on_cpu":       True,
        "residual_alpha_clip":         2.0,
        "residual_merge_metric":       "ls_scalar",
        # Pruning plan: pure_delete saves, others load
        "save_pruning_plan": is_pure_delete,
        "load_pruning_plan": (None if is_pure_delete else plan_path(target_pct, dataset)),
    }
    cfg.update(extra_cfg)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print filenames only, do not write")
    ap.add_argument("--out-dir", default=CONFIG_DIR,
                    help=f"Output directory (default: {CONFIG_DIR})")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    written = 0

    for target_pct in TARGETS:
        for dataset in DATASETS:
            for method, label, extra in METHODS:
                is_pd = (method == "pure_delete")
                cfg   = build_config(target_pct, dataset, method, extra, is_pd)
                name  = config_name(target_pct, dataset, label)
                path  = os.path.join(args.out_dir, name)

                if args.dry_run:
                    plan = cfg.get("load_pruning_plan") or "(saves plan)"
                    print(f"  [would write] {name}")
                    print(f"               plan: {plan}")
                    continue

                with open(path, "w") as fh:
                    yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
                print(f"  wrote  {name}")
                written += 1

    total = EXPECTED_TOTAL
    if args.dry_run:
        print(f"\n  Would write {total} configs to {args.out_dir}")
    else:
        print(f"\n  {written}/{total} configs written to {args.out_dir}")


if __name__ == "__main__":
    main()
