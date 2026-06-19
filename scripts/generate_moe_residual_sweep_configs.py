#!/usr/bin/env python3
"""
generate_moe_residual_sweep_configs.py
======================================
Generate YAML configs for the MoE residual method sweep.

Targets  : 2%, 4%, 6%, 8%
Methods  : 10 variants (pure_delete + 9 residual methods)
Datasets : wikitext2
n_eval   : 64  (quick sweep)

Output:  configs/moe_residual_sweep/<name>.yaml

Pure-delete configs save the pruning plan so other methods can load it.
All non-pure-delete configs reference the same plan path.

Usage:
    python scripts/generate_moe_residual_sweep_configs.py
    python scripts/generate_moe_residual_sweep_configs.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR  = os.path.join(REPO_ROOT, "configs", "moe_residual_sweep")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ── Sweep dimensions ────────────────────────────────────────────────────────
TARGETS   = [2, 4, 6, 8]      # target pruning %
DATASET   = "wikitext2"
N_EVAL    = 64
CALIB_N   = 512
SELECTOR  = "rmsnorm_bound"
AGG_MODE  = "p95"
ALIGN     = 16
MAX_LAYER = 0.10
MIN_TOK   = 32

# Model slug for plan filenames (mirrors make_pruning_plan_path)
MODEL_ID   = "Qwen/Qwen3-30B-A3B"
MODEL_SLUG = MODEL_ID.replace("/", "_").replace("-", "_")

# ── Method definitions ───────────────────────────────────────────────────────
# Each entry: (method_str, lambda_or_None, label_suffix, extra_cfg_dict)
METHODS = [
    # baseline
    ("pure_delete",                        None,  "pure_delete",                    {}),
    # full residual (same as ridge with default lambda)
    ("residual_full_moe",                  None,  "residual_full",                  {}),
    # explicit ridge — lambda sweep
    ("residual_ridge_moe",                 1e-3,  "residual_ridge_lam1e-3",         {"residual_lambda": 1e-3}),
    ("residual_ridge_moe",                 1e-2,  "residual_ridge_lam1e-2",         {"residual_lambda": 1e-2}),
    ("residual_ridge_moe",                 1e-1,  "residual_ridge_lam1e-1",         {"residual_lambda": 1e-1}),
    # ridge + only-if-improves — lambda sweep
    ("residual_ridge_only_if_improves_moe", 1e-3, "residual_ridge_oii_lam1e-3",    {"residual_lambda": 1e-3, "residual_improvement_margin": 1.0}),
    ("residual_ridge_only_if_improves_moe", 1e-2, "residual_ridge_oii_lam1e-2",    {"residual_lambda": 1e-2, "residual_improvement_margin": 1.0}),
    ("residual_ridge_only_if_improves_moe", 1e-1, "residual_ridge_oii_lam1e-1",    {"residual_lambda": 1e-1, "residual_improvement_margin": 1.0}),
    # nearest-channel merge
    ("residual_nearest_channel_merge_moe", None,  "residual_nearest_merge",         {}),
    # nearest-channel merge + only-if-improves
    ("residual_nearest_channel_merge_only_if_improves_moe", None,
                                                  "residual_nearest_merge_oii",     {"residual_improvement_margin": 1.0}),
]


def plan_path(target_pct: int) -> str:
    """Canonical pruning plan path for a given target percentage."""
    fname = (
        f"{MODEL_SLUG}_{DATASET}_n{N_EVAL}_calib{CALIB_N}"
        f"_{SELECTOR}_{AGG_MODE}_{float(target_pct):.1f}pct_align{ALIGN}.json"
    )
    return os.path.join(RESULTS_DIR, "pruning_plans", fname)


def config_name(target_pct: int, label: str) -> str:
    return f"qwen3_30b_a3b_{DATASET}_n{N_EVAL}_target{target_pct}_{label}.yaml"


def build_config(
    target_pct:    int,
    method:        str,
    label:         str,
    extra_cfg:     dict,
    is_pure_delete: bool,
) -> dict:
    cfg = {
        # Model
        "scaling_models":   [MODEL_ID],
        "scaling_dtype":    "auto",
        "device_map":       "auto",
        "expected_expert_layout": "unpacked",
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
        "eval_datasets":               [DATASET],
        "moe_calib_dataset":           DATASET,
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
        # Pruning plan
        "save_pruning_plan": is_pure_delete,
        "load_pruning_plan": (None if is_pure_delete else plan_path(target_pct)),
    }
    cfg.update(extra_cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print filenames only, do not write")
    ap.add_argument("--out-dir", default=CONFIG_DIR,
                    help=f"Output directory (default: {CONFIG_DIR})")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    written = 0

    for target_pct in TARGETS:
        for method, lam, label, extra in METHODS:
            is_pd = (method == "pure_delete")
            cfg   = build_config(target_pct, method, label, extra, is_pd)
            name  = config_name(target_pct, label)
            path  = os.path.join(args.out_dir, name)

            if args.dry_run:
                print(f"  [would write] {name}")
                continue

            with open(path, "w") as fh:
                yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
            print(f"  wrote  {name}")
            written += 1

    total = len(TARGETS) * len(METHODS)
    if args.dry_run:
        print(f"\n  Would write {total} configs to {args.out_dir}")
    else:
        print(f"\n  {written}/{total} configs written to {args.out_dir}")
        print("  Next: bash scripts/run_moe_residual_method_sweep_quick.sh")


if __name__ == "__main__":
    main()
