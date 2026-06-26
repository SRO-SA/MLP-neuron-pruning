#!/usr/bin/env python3
"""
downstream_eval_summarize.py
============================
Post-process downstream evaluation outputs.

Modes
-----
Default (after a run):
    Reads downstream_summary.csv, computes comparison vs baseline,
    writes downstream_comparison_summary.csv, prints compact table.

--summarize-only (rebuild from raw lm_eval dirs):
    Scans RUN_DIR for *_lm_eval/ subdirectories, parses labels and
    lm_eval JSON outputs, rebuilds downstream_summary.csv from scratch,
    then runs the default mode steps.

Usage
-----
    # Called automatically by run_moe_downstream_eval.sh after each run:
    python3 scripts/downstream_eval_summarize.py \
        --run-dir results/downstream_eval_runs/20260624_213446 \
        --plan-dir results/pruning_plans \
        --orig-moe-dim 768 --moe-align 16 --model Qwen/Qwen3-30B-A3B

    # Rebuild summaries from an existing run without re-running lm_eval:
    SUMMARIZE_ONLY=1 RUN_DIR=results/downstream_eval_runs/20260624_213446 \\
        bash scripts/run_moe_downstream_eval.sh
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── CSV schemas ───────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "setting_label", "method", "selector", "dataset",
    "target_pct", "actual_pct", "moe_dim",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "pruning_plan_path", "model_path", "is_pruned",
    "requested_method", "actual_method", "residual_applied", "residual_fallback_used",
    "task", "metric", "value", "stderr",
    "num_fewshot", "limit", "batch_size", "status",
]

COMPARISON_FIELDS = [
    "method", "selector", "dataset", "target_pct", "actual_pct", "moe_dim",
    "requested_method", "actual_method", "residual_applied", "residual_fallback_used",
    "task", "metric",
    "baseline_value", "pruned_value", "delta", "relative_delta_pct",
    "baseline_stderr", "pruned_stderr",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_label(label: str) -> Dict[str, Any]:
    """Decompose a setting label string into its components."""
    if label == "baseline_no_pruning":
        return dict(method="baseline", selector="none", dataset="none",
                    target_pct=0.0, actual_pct=0.0, is_pruned=False)
    m = re.match(
        r'^(.+?)__(.+?)__(.+?)__target(\d+(?:\.\d+)?)pct__actual(\d+(?:\.\d+)?)pct$',
        label,
    )
    if m:
        return dict(
            method=m.group(1), selector=m.group(2), dataset=m.group(3),
            target_pct=float(m.group(4)), actual_pct=float(m.group(5)),
            is_pruned=True,
        )
    return dict(method="unknown", selector="unknown", dataset="unknown",
                target_pct=0.0, actual_pct=0.0, is_pruned=False)


def infer_moe_dim(actual_pct: float, orig_moe_dim: int, align: int) -> int:
    """
    Estimate saved moe_intermediate_size from the pruning %.
    Formula: moe_dim = orig_moe_dim - round(orig_moe_dim * actual_pct/100 / align) * align
    For Qwen3-30B-A3B (orig=768, align=16):
      2.1% -> 768-16=752, 4.2% -> 768-32=736, 6.2% -> 768-48=720, 8.3% -> 768-64=704
    """
    if actual_pct <= 0:
        return orig_moe_dim
    pruned = round(orig_moe_dim * actual_pct / 100.0 / align) * align
    return max(0, orig_moe_dim - pruned)


def find_plan_file(plan_dir: str, method: str, selector: str, dataset: str,
                   target_pct: float, model_slug: str) -> str:
    """Locate the pruning plan JSON for a given setting."""
    if not plan_dir or not os.path.isdir(plan_dir):
        return "NONE"
    target = int(float(target_pct))
    # Exact pattern first
    exact = os.path.join(
        plan_dir,
        f"{model_slug}_{dataset}_n512_calib512_{selector}_p95_{target}.0pct_align16.json",
    )
    if os.path.isfile(exact):
        return exact
    # Glob fallback (any n/calib values)
    hits = glob.glob(os.path.join(
        plan_dir, f"{model_slug}_{dataset}_n*_{selector}_*_{target}.0pct_align16.json",
    ))
    return hits[0] if hits else "NONE"


def read_plan_meta(plan_path: str) -> Dict[str, Any]:
    """Return actual_pct and method string from a plan JSON file."""
    if plan_path == "NONE" or not os.path.isfile(plan_path):
        return {}
    try:
        with open(plan_path) as f:
            d = json.load(f)
        layers   = d.get("layers", [])
        n_old    = sum(lc.get("old_intermediate", 0) for lc in layers)
        n_pruned = sum(len(lc.get("prune_idx", []))  for lc in layers)
        actual   = round(100.0 * n_pruned / n_old, 3) if n_old else 0.0
        method   = (d.get("method") or d.get("pruning_method") or "").strip()
        return {"actual_pct": actual, "method_from_plan": method}
    except Exception as e:
        print("[summarize] WARNING: cannot read plan {}: {}".format(plan_path, e))
        return {}


def parse_lm_eval_dir(lm_eval_dir: str) -> Tuple[List[Dict], Dict]:
    """
    Parse all lm_eval JSON files in a *_lm_eval/ directory.
    Returns (metric_rows, config_dict).
    """
    rows: List[Dict] = []
    config_info: Dict = {}

    for rf in sorted(set(glob.glob(
            os.path.join(lm_eval_dir, "**/*.json"), recursive=True))):
        try:
            with open(rf) as f:
                data = json.load(f)
        except Exception as e:
            print("[summarize] WARNING: cannot open {}: {}".format(rf, e))
            continue

        # Extract eval config (num_fewshot, limit, batch_size) from first file that has it
        if not config_info:
            raw_cfg = data.get("config", data.get("configs", {}))
            if isinstance(raw_cfg, dict):
                config_info = {
                    "num_fewshot": raw_cfg.get("num_fewshot", raw_cfg.get("fewshot", 0)),
                    "limit":       raw_cfg.get("limit", "none"),
                    "batch_size":  raw_cfg.get("batch_size", "auto"),
                }

        results = data.get("results", {})
        if not results:
            continue

        for task, metrics in results.items():
            for metric_key, val in metrics.items():
                if "_stderr," in metric_key:
                    continue
                if not (metric_key.endswith(",none") or metric_key == "acc"):
                    continue
                metric     = metric_key.replace(",none", "")
                stderr_key = metric + "_stderr,none"
                stderr     = metrics.get(stderr_key, "")
                rows.append({"task": task, "metric": metric, "value": val, "stderr": stderr})

    return rows, config_info


def build_comparison(summary_rows: List[Dict]) -> List[Dict]:
    """
    Join pruned rows against baseline rows by (task, metric).
    delta = pruned_value - baseline_value  (negative = accuracy degraded).
    """
    baseline: Dict[Tuple, Dict] = {}
    for row in summary_rows:
        if str(row.get("method", "")).lower() == "baseline":
            key = (row["task"], row["metric"])
            baseline[key] = row

    comp_rows: List[Dict] = []
    for row in summary_rows:
        if str(row.get("method", "")).lower() == "baseline":
            continue
        key  = (row["task"], row["metric"])
        base = baseline.get(key)
        if base is None:
            continue
        try:
            bv    = float(base["value"])
            pv    = float(row["value"])
            delta = round(pv - bv, 6)
            rel   = round(100.0 * delta / bv, 3) if bv != 0 else "NA"
        except (ValueError, TypeError):
            delta = rel = "NA"
        comp_rows.append({
            "method":                 row.get("actual_method") or row.get("method", ""),
            "selector":               row.get("selector", ""),
            "dataset":                row.get("dataset", ""),
            "target_pct":             row.get("target_pct", ""),
            "actual_pct":             row.get("actual_pct", ""),
            "moe_dim":                row.get("moe_dim", ""),
            "requested_method":       row.get("requested_method", row.get("method", "")),
            "actual_method":          row.get("actual_method", row.get("method", "")),
            "residual_applied":       row.get("residual_applied", "unknown"),
            "residual_fallback_used": row.get("residual_fallback_used", "unknown"),
            "task":                   row["task"],
            "metric":                 row["metric"],
            "baseline_value":         base["value"],
            "pruned_value":           row["value"],
            "delta":                  delta,
            "relative_delta_pct":     rel,
            "baseline_stderr":        base.get("stderr", ""),
            "pruned_stderr":          row.get("stderr", ""),
        })
    return comp_rows


def print_comparison_table(comp_rows: List[Dict]) -> None:
    if not comp_rows:
        print("[summarize] No comparison rows to display.")
        return
    W = 36
    hdr = (
        "  {:<14}  {:<9}  {:<{W}}  {:>7}  {:>7}  {:>9}  {:>9}  {:>8}".format(
            "Task", "Metric", "Method", "Target", "MoeDim",
            "Baseline", "Pruned", "Delta", W=W,
        )
    )
    print("\n[summarize] Comparison (pruned vs baseline):")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in comp_rows:
        method = str(r.get("actual_method") or r.get("method", ""))[:W]
        tgt    = str(r.get("target_pct", ""))
        moed   = str(r.get("moe_dim", ""))
        try:
            bv = float(r["baseline_value"])
            pv = float(r["pruned_value"])
            dv = float(r["delta"])
            print(
                "  {:<14}  {:<9}  {:<{W}}  {:>7}  {:>7}  {:>9.4f}  {:>9.4f}  {:>+8.4f}".format(
                    r["task"], r["metric"], method, tgt, moed, bv, pv, dv, W=W,
                )
            )
        except (ValueError, TypeError):
            print("  {:<14}  {:<9}  {:<{W}}  {:>7}  {:>7}  (parse error)".format(
                r["task"], r["metric"], method, tgt, moed, W=W,
            ))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",        required=True,
                    help="Run output directory (contains *_lm_eval/ subdirs)")
    ap.add_argument("--summarize-only", action="store_true", default=False,
                    help="Rebuild downstream_summary.csv from raw lm_eval dirs")
    ap.add_argument("--plan-dir",       default="results/pruning_plans",
                    help="Directory containing pruning plan JSON files")
    ap.add_argument("--orig-moe-dim",   type=int, default=768,
                    help="Original model moe_intermediate_size (default 768)")
    ap.add_argument("--moe-align",      type=int, default=16,
                    help="MoE intermediate size alignment granularity (default 16)")
    ap.add_argument("--model",          default="Qwen/Qwen3-30B-A3B",
                    help="HuggingFace model ID (used as model_path for baseline rows)")
    args = ap.parse_args()

    summary_csv    = os.path.join(args.run_dir, "downstream_summary.csv")
    comparison_csv = os.path.join(args.run_dir, "downstream_comparison_summary.csv")
    model_slug     = args.model.replace("/", "_").replace("-", "_")

    # ── SUMMARIZE_ONLY: rebuild from *_lm_eval dirs ───────────────────────────
    if args.summarize_only:
        print("[summarize] SUMMARIZE_ONLY: scanning {} ...".format(args.run_dir))
        lm_dirs = sorted(glob.glob(os.path.join(args.run_dir, "*_lm_eval")))
        if not lm_dirs:
            print("[summarize] ERROR: no *_lm_eval directories found in {}".format(args.run_dir))
            sys.exit(1)
        print("[summarize] Found {} lm_eval dir(s):".format(len(lm_dirs)))
        for d in lm_dirs:
            print("[summarize]   " + os.path.basename(d))

        summary_rows: List[Dict] = []
        missing_meta: List[str] = []

        for lm_dir in lm_dirs:
            label    = os.path.basename(lm_dir).replace("_lm_eval", "")
            linfo    = parse_label(label)
            plan_path = find_plan_file(
                args.plan_dir,
                linfo["method"], linfo["selector"], linfo["dataset"],
                linfo["target_pct"], model_slug,
            )
            plan_meta = read_plan_meta(plan_path)
            if plan_meta.get("actual_pct"):
                linfo["actual_pct"] = plan_meta["actual_pct"]

            moe_dim   = infer_moe_dim(linfo["actual_pct"], args.orig_moe_dim, args.moe_align)
            metrics, cfg_info = parse_lm_eval_dir(lm_dir)

            if not metrics:
                print("[summarize] WARNING: no metrics found in {}".format(lm_dir))
                missing_meta.append(label)
                continue

            limit_val = cfg_info.get("limit", "none")
            if limit_val is None:
                limit_val = "none"
            model_path = args.model if not linfo["is_pruned"] else "PRUNED({})".format(label)

            for m in metrics:
                summary_rows.append({
                    "setting_label":                  label,
                    "method":                         linfo["method"],
                    "selector":                       linfo["selector"],
                    "dataset":                        linfo["dataset"],
                    "target_pct":                     linfo["target_pct"],
                    "actual_pct":                     linfo["actual_pct"],
                    "moe_dim":                        moe_dim,
                    "expert_param_reduction_pct":     linfo["actual_pct"] if linfo["is_pruned"] else 0.0,
                    "total_model_param_reduction_pct": "NA",
                    "pruning_plan_path":              plan_path,
                    "model_path":                     model_path,
                    "is_pruned":                      str(linfo["is_pruned"]),
                    # Cannot recover residual metadata from lm_eval dirs alone
                    "requested_method":               linfo["method"],
                    "actual_method":                  "unknown",
                    "residual_applied":               "unknown",
                    "residual_fallback_used":         "unknown",
                    "task":                           m["task"],
                    "metric":                         m["metric"],
                    "value":                          m["value"],
                    "stderr":                         m["stderr"],
                    "num_fewshot":                    cfg_info.get("num_fewshot", 0),
                    "limit":                          limit_val,
                    "batch_size":                     cfg_info.get("batch_size", "auto"),
                    "status":                         "ok",
                })

        if not summary_rows:
            print("[summarize] ERROR: could not rebuild any rows from existing outputs.")
            if missing_meta:
                print("[summarize] Settings with no lm_eval metrics: " + str(missing_meta))
            sys.exit(1)

        os.makedirs(args.run_dir, exist_ok=True)
        with open(summary_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print("[summarize] Rebuilt downstream_summary.csv: {} rows -> {}".format(
            len(summary_rows), summary_csv))

        if missing_meta:
            print("[summarize] NOTE: no lm_eval output found for: " + str(missing_meta))
        print("[summarize] NOTE: the following fields were inferred (cannot be recovered):")
        print("[summarize]   model_path        -> set to 'PRUNED(label)' for pruned rows")
        print("[summarize]   total_model_param_reduction_pct -> 'NA' (requires full param count)")

    # ── Build comparison (default mode and after summarize-only rebuild) ───────
    if not os.path.isfile(summary_csv):
        print("[summarize] ERROR: {} not found. Run eval first.".format(summary_csv))
        sys.exit(1)

    with open(summary_csv, newline="") as fh:
        summary_rows = list(csv.DictReader(fh))

    if not summary_rows:
        print("[summarize] WARNING: {} is empty.".format(summary_csv))
        sys.exit(0)

    comp_rows = build_comparison(summary_rows)
    if comp_rows:
        with open(comparison_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comp_rows)
        print("[summarize] Comparison CSV: {} rows -> {}".format(len(comp_rows), comparison_csv))
    else:
        print("[summarize] WARNING: no comparison rows (need at least one baseline + one pruned setting).")

    print_comparison_table(comp_rows)


if __name__ == "__main__":
    main()
