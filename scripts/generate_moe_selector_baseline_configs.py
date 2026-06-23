#!/usr/bin/env python3
"""
generate_moe_selector_baseline_configs.py
==========================================
Generate YAML configs for the MoE selector baseline comparison.

Matrix: 4 selectors × 4 targets × 2 datasets = 32 configs.

All configs use the ``pure_delete`` pruning method.  Each selector saves its
own pruning plan because the channel-selection order differs between selectors.

Selectors:
  - rmsnorm_bound    : weight-only RMSNorm-bounded SwiGLU score (baseline)
  - down_norm        : L2 norm of each down_proj column (simple)
  - activation_score : activation × down-column-norm (needs calibration data)
  - random           : uniform random (random baseline; seed fixed)

Targets:  2%, 4%, 6%, 8%
Datasets: wikitext2, c4
n_eval:   512

Usage:
    python scripts/generate_moe_selector_baseline_configs.py
    python scripts/generate_moe_selector_baseline_configs.py --dry-run
    python scripts/generate_moe_selector_baseline_configs.py --out-dir configs/my_dir
"""
from __future__ import annotations

import argparse
import os

import yaml

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR  = os.path.join(REPO_ROOT, "configs", "moe_selector_baseline")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ── Benchmark dimensions ─────────────────────────────────────────────────────
TARGETS   = [2, 4, 6, 8]
DATASETS  = ["wikitext2", "c4"]
N_EVAL    = 512
CALIB_N   = 512
AGG_MODE  = "p95"
ALIGN     = 16
MAX_LAYER = 0.10
MIN_TOK   = 32

# selector name → (label_for_filename, needs_calib_data)
SELECTORS = [
    ("rmsnorm_bound",    "rmsnorm_bound",    False),
    ("down_norm",        "down_norm",        False),
    ("activation_score", "activation_score", True),
    ("random",           "random",           False),
]

METHOD = "pure_delete"  # all configs use pure_delete; only the selector varies

MODEL_ID   = "Qwen/Qwen3-30B-A3B"
MODEL_SLUG = MODEL_ID.replace("/", "_").replace("-", "_")

EXPECTED_TOTAL = len(SELECTORS) * len(TARGETS) * len(DATASETS)  # 4×4×2 = 32


def plan_path(target_pct: int, dataset: str, selector: str) -> str:
    """
    Canonical pruning plan path — mirrors make_pruning_plan_path() in
    src/moe_residual_methods.py.  Selector is included in the filename so
    each selector's plan is stored separately.
    """
    fname = (
        f"{MODEL_SLUG}_{dataset}_n{N_EVAL}_calib{CALIB_N}"
        f"_{selector}_{AGG_MODE}_{float(target_pct):.1f}pct_align{ALIGN}.json"
    )
    return os.path.join(RESULTS_DIR, "pruning_plans", fname)


def config_name(target_pct: int, dataset: str, selector_label: str) -> str:
    return f"qwen3_30b_a3b_{dataset}_n{N_EVAL}_target{target_pct}_sel_{selector_label}.yaml"


def build_config(
    target_pct:   int,
    dataset:      str,
    selector:     str,
    needs_calib:  bool,
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
        "scaling_methods":             [METHOD],
        "moe_selector":                selector,
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
        # activation_score selector: enable calibration-data collection
        "moe_selector_needs_calib":    needs_calib,
        # Pruning plan: every config saves its own plan (selector → different channels)
        "save_pruning_plan": True,
        "load_pruning_plan": None,
    }
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

    for selector, label, needs_calib in SELECTORS:
        for target_pct in TARGETS:
            for dataset in DATASETS:
                cfg  = build_config(target_pct, dataset, selector, needs_calib)
                name = config_name(target_pct, dataset, label)
                path = os.path.join(args.out_dir, name)

                if args.dry_run:
                    print(f"  [would write] {name}")
                    print(f"               selector={selector}  target={target_pct}%  ds={dataset}")
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
        if written != total:
            raise SystemExit(f"ERROR: expected {total} configs, wrote {written}")


if __name__ == "__main__":
    main()
