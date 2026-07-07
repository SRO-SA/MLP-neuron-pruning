#!/usr/bin/env python3
"""
generate_moe_selector_baseline_configs.py
==========================================
Generate YAML configs for the selector-baseline PPL benchmark.

Filename format (must match run_moe_selector_baseline_ppl.sh):
  qwen3_30b_a3b_{dataset}_n{n_eval}_target{target}_sel_{selector}.yaml

Usage:
  python3 scripts/generate_moe_selector_baseline_configs.py \
      --dataset wikitext2 \
      --selectors rmsnorm_bound,down_norm,activation_score,random \
      --targets 2,4,6,8 \
      --n-eval 128 \
      --config-dir configs/moe_selector_baseline

  python3 scripts/generate_moe_selector_baseline_configs.py \
      --dataset wikitext2 --n-eval 512   # full-run configs

Outputs N_selectors x N_targets YAML files in --config-dir.
Skips files that already exist unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# ── Optional yaml import ───────────────────────────────────────────────────────
try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False


# ── Selectors that require calibration data ────────────────────────────────────
_CALIB_SELECTORS = {"activation_score"}


def _model_slug(model_name: str) -> str:
    """
    Qwen/Qwen3-30B-A3B  ->  qwen3_30b_a3b
    Strips the org prefix, lowercases, replaces - with _.
    """
    name = model_name.split("/")[-1]
    name = name.lower().replace("-", "_")
    name = re.sub(r"_+", "_", name)
    return name


def _build_config(
    model: str,
    selector: str,
    target_pct: float,
    dataset: str,
    n_eval: int,
    channel_alignment: int,
    seed: int,
    max_seq_len: int,
    batch_size: int,
) -> dict:
    """Build the YAML config dict for one (selector, target) pair."""
    return {
        "scaling_models":                   [model],
        "scaling_dtype":                    "auto",
        "device_map":                       "auto",
        "expected_expert_layout":           "unpacked",
        "moe_pruning_mode":                 "packed_same_channel",
        "target_pruning_percents":          [float(target_pct)],
        "scaling_methods":                  ["pure_delete"],
        "moe_selector":                     selector,
        "moe_same_channel_aggregation":     "p95",
        "moe_channel_alignment":            channel_alignment,
        "moe_max_layer_channel_prune_frac": 0.1,
        "max_expert_frac":                  0.1,
        "min_expert_tokens":                32,
        "moe_budget_mode":                  "uniform",
        "eval_datasets":                    [dataset],
        "moe_calib_dataset":                dataset,
        "reconstruction_eval_samples":      n_eval,
        "moe_calib_samples":                n_eval,
        "max_seq_len":                      max_seq_len,
        "batch_size":                       batch_size,
        "use_fallback_corpus":              False,
        "moe_inplace_pruning":              True,
        "moe_smoke_test":                   False,
        "seed":                             seed,
        "moe_selector_needs_calib":         selector in _CALIB_SELECTORS,
        "save_pruning_plan":                True,
        "load_pruning_plan":                None,
    }


def _write_yaml(path: str, cfg: dict) -> None:
    if _YAML:
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: manual YAML serialisation (no PyYAML required)
        lines = []
        for k, v in cfg.items():
            if v is None:
                lines.append(f"{k}: null")
            elif isinstance(v, bool):
                lines.append(f"{k}: {'true' if v else 'false'}")
            elif isinstance(v, list):
                items = ", ".join(
                    ("null" if x is None else
                     f"'{x}'" if isinstance(x, str) else str(x))
                    for x in v
                )
                lines.append(f"{k}: [{items}]")
            elif isinstance(v, str):
                lines.append(f"{k}: {v}")
            else:
                lines.append(f"{k}: {v}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",             default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--model-slug",        default=None,
                    help="Override auto-derived filename slug (e.g. qwen3_30b_a3b).")
    ap.add_argument("--dataset",           default="wikitext2")
    ap.add_argument("--selectors",         default="rmsnorm_bound,down_norm,activation_score,random")
    ap.add_argument("--targets",           default="2,4,6,8",
                    help="Comma-separated integer target pruning percentages.")
    ap.add_argument("--n-eval",            type=int, default=512,
                    help="reconstruction_eval_samples and moe_calib_samples (also in filename).")
    ap.add_argument("--channel-alignment", type=int, default=16)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--max-seq-len",       type=int, default=512)
    ap.add_argument("--batch-size",        type=int, default=4)
    ap.add_argument("--config-dir",        default="configs/moe_selector_baseline")
    ap.add_argument("--overwrite",         action="store_true",
                    help="Overwrite existing config files.")
    ap.add_argument("--dry-run",           action="store_true",
                    help="Print planned files without writing.")
    args = ap.parse_args()

    slug      = args.model_slug or _model_slug(args.model)
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    targets   = [t.strip() for t in args.targets.split(",")  if t.strip()]

    if not selectors:
        print("ERROR: --selectors is empty.", file=sys.stderr)
        sys.exit(1)
    if not targets:
        print("ERROR: --targets is empty.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.config_dir, exist_ok=True)

    written = 0
    skipped = 0
    planned = 0

    print(f"[generate_configs] model_slug = {slug}")
    print(f"[generate_configs] dataset    = {args.dataset}")
    print(f"[generate_configs] n_eval     = {args.n_eval}")
    print(f"[generate_configs] selectors  = {selectors}")
    print(f"[generate_configs] targets    = {targets}")
    print(f"[generate_configs] config_dir = {args.config_dir}")
    if args.dry_run:
        print("[generate_configs] DRY RUN -- no files written.")
    print()

    for sel in selectors:
        for tgt in targets:
            tgt_int = int(float(tgt))   # "2.0" -> 2
            fname   = (
                f"{slug}_{args.dataset}_n{args.n_eval}"
                f"_target{tgt_int}_sel_{sel}.yaml"
            )
            fpath   = os.path.join(args.config_dir, fname)
            planned += 1

            if os.path.exists(fpath) and not args.overwrite:
                print(f"  SKIP  {fname}  (already exists; use --overwrite to replace)")
                skipped += 1
                continue

            cfg = _build_config(
                model             = args.model,
                selector          = sel,
                target_pct        = float(tgt_int),
                dataset           = args.dataset,
                n_eval            = args.n_eval,
                channel_alignment = args.channel_alignment,
                seed              = args.seed,
                max_seq_len       = args.max_seq_len,
                batch_size        = args.batch_size,
            )

            if args.dry_run:
                print(f"  WOULD WRITE  {fpath}")
            else:
                _write_yaml(fpath, cfg)
                print(f"  WRITE  {fpath}")
                written += 1

    print()
    if args.dry_run:
        print(f"[generate_configs] (dry-run) {planned} config(s) planned.")
    else:
        print(f"[generate_configs] {written} written, {skipped} skipped "
              f"(of {planned} planned).")


if __name__ == "__main__":
    main()
