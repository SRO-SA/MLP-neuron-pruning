#!/usr/bin/env python3
"""
summarize_moe_selector_ppl.py
==============================
Post-process a selector-baseline PPL benchmark run.

Reads per-run moe_target_pruning_*.csv files produced by run_experiment.py
--moe-target-pruning, combines them into two summary CSVs, prints a compact
attribution table, and validates completeness.

Outputs (written to --run-dir):
  selector_baseline_summary.csv    — one row per selector/target
  selector_attribution_summary.csv — ranking with delta vs rmsnorm_bound

Usage:
  python3 scripts/summarize_moe_selector_ppl.py \
      --run-dir results/moe_selector_baselines/20260707_120000 \
      --model Qwen/Qwen3-30B-A3B \
      --dataset wikitext2 \
      --selectors rmsnorm_bound,down_norm,activation_score,random \
      --targets 2,4,6,8 \
      --min-rows 16
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── Output CSV schemas ────────────────────────────────────────────────────────

BASELINE_SUMMARY_FIELDS = [
    "run_id",
    "model",
    "dataset",
    "selector",
    "method",
    "target_pct",
    "actual_pct",
    "moe_dim",
    "expert_param_reduction_pct",
    "total_model_param_reduction_pct",
    "ppl_base",
    "ppl_pruned",
    "ppl_delta",
    "ppl_rel_inc_pct",
    "forward_check",
    "pruning_plan_path",
    "checkpoint_path",
    "status",
]

ATTRIBUTION_FIELDS = [
    "selector",
    "target_pct",
    "actual_pct",
    "moe_dim",
    "ppl_rel_inc_pct",
    "delta_vs_rmsnorm_bound",
    "rank_at_target",
    "status",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_moe_dim(actual_pct: float, orig: int, align: int) -> int:
    if actual_pct <= 0:
        return orig
    pruned = round(orig * actual_pct / 100.0 / align) * align
    return max(0, orig - pruned)


def _flt(val: Any, fallback: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


def collect_run_rows(run_dir: str) -> List[Dict]:
    """
    Walk run_dir for */moe_target_pruning_*.csv files, read all rows,
    return a flat list of dicts (raw column values from the per-run CSVs).
    """
    pattern = os.path.join(run_dir, "*/moe_target_pruning_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        # Also try the run_dir itself (in case output_dir = run_dir directly)
        pattern2 = os.path.join(run_dir, "moe_target_pruning_*.csv")
        csv_files = sorted(glob.glob(pattern2))

    rows: List[Dict] = []
    for path in csv_files:
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    row["_src_csv"] = path
                    rows.append(row)
        except Exception as e:
            print(f"[summarize] WARNING: cannot read {path}: {e}", flush=True)
    return rows


def validate_rows(rows: List[Dict], min_rows: int, selectors: List[str]) -> None:
    """Fail clearly on any of the required validation conditions."""
    errors: List[str] = []

    # 1. Minimum row count
    if len(rows) < min_rows:
        errors.append(
            f"Expected at least {min_rows} row(s), found {len(rows)}. "
            f"Some runs may have failed."
        )

    # 2. PPL present and non-nan for every row
    for i, row in enumerate(rows):
        bp = _flt(row.get("baseline_ppl"))
        cp = _flt(row.get("compressed_ppl"))
        if math.isnan(bp) or math.isnan(cp):
            errors.append(
                f"Row {i}: selector={row.get('selector')!r} "
                f"target={row.get('target_pct')!r}: "
                f"PPL is missing (baseline_ppl={bp}, compressed_ppl={cp})."
            )

    # 3. forward_check must be truthy
    for i, row in enumerate(rows):
        fwd = str(row.get("forward_check", "True")).strip().lower()
        if fwd in ("false", "0", "fail"):
            errors.append(
                f"Row {i}: selector={row.get('selector')!r} "
                f"target={row.get('target_pct')!r}: "
                f"forward_check={row.get('forward_check')!r} — "
                f"shape may not have changed after pruning."
            )

    # 4. Selector name must be non-empty and in the known list
    for i, row in enumerate(rows):
        sel = str(row.get("selector", "")).strip()
        if not sel:
            errors.append(f"Row {i}: selector field is empty or missing.")
        elif selectors and sel not in selectors:
            errors.append(
                f"Row {i}: selector={sel!r} not in expected list {selectors}."
            )

    if errors:
        print("[summarize] VALIDATION FAILED:", flush=True)
        for e in errors:
            print(f"[summarize]   {e}", flush=True)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",      required=True)
    ap.add_argument("--model",        default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--dataset",      default="wikitext2")
    ap.add_argument("--selectors",    default="rmsnorm_bound,down_norm,activation_score,random")
    ap.add_argument("--targets",      default="2,4,6,8")
    ap.add_argument("--orig-moe-dim", type=int, default=768)
    ap.add_argument("--moe-align",    type=int, default=16)
    ap.add_argument("--min-rows",     type=int, default=16,
                    help="Minimum expected rows in combined summary (fail if fewer).")
    args = ap.parse_args()

    run_dir       = args.run_dir
    run_id        = os.path.basename(run_dir.rstrip("/"))
    summary_csv   = os.path.join(run_dir, "selector_baseline_summary.csv")
    attribution_csv = os.path.join(run_dir, "selector_attribution_summary.csv")

    known_selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    target_list     = [t.strip() for t in args.targets.split(",") if t.strip()]

    print(f"[summarize] Collecting CSVs from {run_dir} ...", flush=True)
    raw_rows = collect_run_rows(run_dir)
    print(f"[summarize] Found {len(raw_rows)} raw row(s).", flush=True)

    if not raw_rows:
        print(f"[summarize] ERROR: no moe_target_pruning_*.csv files found in {run_dir}",
              flush=True)
        print(f"[summarize] Expected pattern: {run_dir}/*/moe_target_pruning_*.csv",
              flush=True)
        sys.exit(1)

    # ── Validate ──────────────────────────────────────────────────────────────
    validate_rows(raw_rows, args.min_rows, known_selectors)
    print(f"[summarize] Validation OK: {len(raw_rows)} row(s), all PPL present.", flush=True)

    # ── Build selector_baseline_summary ──────────────────────────────────────
    summary_rows: List[Dict] = []
    for row in raw_rows:
        actual_pct = _flt(row.get("actual_pct", 0))
        moe_dim    = infer_moe_dim(actual_pct, args.orig_moe_dim, args.moe_align)
        bp  = _flt(row.get("baseline_ppl"))
        cp  = _flt(row.get("compressed_ppl"))
        dp  = _flt(row.get("delta_ppl"))
        rel = _flt(row.get("relative_delta_pct"))

        summary_rows.append({
            "run_id":                         run_id,
            "model":                          row.get("model", args.model),
            "dataset":                        row.get("eval_dataset", args.dataset),
            "selector":                       row.get("selector", ""),
            "method":                         row.get("method", ""),
            "target_pct":                     row.get("target_pct", ""),
            "actual_pct":                     round(actual_pct, 4),
            "moe_dim":                        moe_dim,
            "expert_param_reduction_pct":     round(_flt(row.get("expert_param_reduction_pct")), 4),
            "total_model_param_reduction_pct": round(_flt(row.get("total_model_param_reduction_pct")), 4),
            "ppl_base":                       round(bp, 6),
            "ppl_pruned":                     round(cp, 6),
            "ppl_delta":                      round(dp, 6),
            "ppl_rel_inc_pct":                round(rel, 4),
            "forward_check":                  row.get("forward_check", ""),
            "pruning_plan_path":              row.get("pruning_plan_path", "NONE"),
            "checkpoint_path":                "NONE",   # PPL-only; no HF ckpt saved
            "status":                         row.get("status", ""),
        })

    os.makedirs(run_dir, exist_ok=True)
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BASELINE_SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[summarize] Wrote {len(summary_rows)} rows → {summary_csv}", flush=True)

    # Validate selector column is present in CSV
    found_selectors = {r["selector"] for r in summary_rows}
    for sel in known_selectors:
        if sel not in found_selectors:
            # Only warn if we expected it from the run (not in SMOKE mode)
            print(f"[summarize] WARNING: expected selector {sel!r} not found in results.",
                  flush=True)

    # ── Build selector_attribution_summary ───────────────────────────────────
    # Group by (target_pct, selector) — pick best (lowest rel_inc) if duplicates
    best: Dict[Tuple[str, str], Dict] = {}
    for r in summary_rows:
        key = (str(r["target_pct"]), str(r["selector"]))
        existing = best.get(key)
        if existing is None or _flt(r["ppl_rel_inc_pct"]) < _flt(existing["ppl_rel_inc_pct"]):
            best[key] = r

    # Build attribution rows grouped by target
    attribution_rows: List[Dict] = []
    targets_found = sorted({str(r["target_pct"]) for r in summary_rows},
                           key=lambda x: _flt(x))

    for tgt in targets_found:
        # Collect all selectors for this target, sorted by ppl_rel_inc_pct
        rows_at_tgt = [
            best[(tgt, sel)]
            for sel in known_selectors
            if (tgt, sel) in best
        ]
        # Also include any unexpected selectors found in results
        for key, r in best.items():
            if key[0] == tgt and r["selector"] not in known_selectors:
                rows_at_tgt.append(r)

        rows_at_tgt.sort(key=lambda r: _flt(r["ppl_rel_inc_pct"]))

        # Find rmsnorm_bound baseline for this target
        ref_rel = None
        for r in rows_at_tgt:
            if r["selector"] == "rmsnorm_bound":
                ref_rel = _flt(r["ppl_rel_inc_pct"])
                break

        for rank, r in enumerate(rows_at_tgt, start=1):
            rel = _flt(r["ppl_rel_inc_pct"])
            delta_vs_ref = (
                round(rel - ref_rel, 4) if ref_rel is not None else "NA"
            )
            attribution_rows.append({
                "selector":              r["selector"],
                "target_pct":           r["target_pct"],
                "actual_pct":           r["actual_pct"],
                "moe_dim":              r["moe_dim"],
                "ppl_rel_inc_pct":      round(rel, 4),
                "delta_vs_rmsnorm_bound": delta_vs_ref,
                "rank_at_target":       rank,
                "status":               r["status"],
            })

    with open(attribution_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ATTRIBUTION_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(attribution_rows)
    print(f"[summarize] Wrote {len(attribution_rows)} rows → {attribution_csv}", flush=True)

    # ── Print compact ranking table ───────────────────────────────────────────
    print("\n[summarize] ── Selector Attribution: PPL relative increase ──────────────")
    for tgt in targets_found:
        rows_at_tgt = [r for r in attribution_rows if str(r["target_pct"]) == tgt]
        rows_at_tgt.sort(key=lambda r: int(r["rank_at_target"]))
        tgt_int = int(float(tgt))
        print(f"\n  Target {tgt_int}%  (moe_dim={rows_at_tgt[0]['moe_dim'] if rows_at_tgt else '?'}):")
        for r in rows_at_tgt:
            sel   = str(r["selector"])
            rel   = _flt(r["ppl_rel_inc_pct"])
            delta = r["delta_vs_rmsnorm_bound"]
            rank  = r["rank_at_target"]
            delta_str = (
                f"  [{delta:+.3f}% vs rmsnorm_bound]"
                if isinstance(delta, float) else ""
            )
            print(f"    {rank}. {sel:<22s}  {rel:+.3f}%{delta_str}")
    print()

    # ── Final validation: row count ───────────────────────────────────────────
    if len(summary_rows) < args.min_rows:
        print(f"[summarize] ERROR: summary has {len(summary_rows)} rows, "
              f"expected at least {args.min_rows}.", flush=True)
        # Check for selectors missing from summary
        missing_sels = sorted(known_selectors - found_selectors)
        if missing_sels:
            print(f"[summarize]   Missing selectors: {missing_sels}", flush=True)
        sys.exit(1)

    print(f"[summarize] OK: {len(summary_rows)} summary row(s), "
          f"{len(attribution_rows)} attribution row(s).", flush=True)


if __name__ == "__main__":
    main()
