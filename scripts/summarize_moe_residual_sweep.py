#!/usr/bin/env python3
"""
summarize_moe_residual_sweep.py

Reads all MoE pruning experiment CSVs from a results directory,
groups by target_pct, and prints a comparison table of residual methods.

Outputs:
  results/moe_residual_sweep_summary.csv
  results/moe_residual_sweep_summary.md

Usage:
  python3 scripts/summarize_moe_residual_sweep.py [--results-dir results] [--out-dir results] [--quiet]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ── Column display order and formatting ────────────────────────────────────────
DISPLAY_COLS = [
    "moe_pruning_method",
    "residual_lambda",
    "residual_coverage_pct",
    "residual_applied_experts",
    "residual_rejected_experts",
    "expert_param_reduction_pct",
    "compressed_ppl",
    "baseline_ppl",
    "delta_ppl",
    "relative_delta_pct",
    "forward_check",
    "status",
    "residual_fallback_used",
    "pruning_plan_path",
]

WARN_COLS = {
    "residual_fallback_used": lambda v: str(v).lower() in ("true", "1"),
    "status": lambda v: v.lower() not in ("ok", ""),
    "residual_coverage_pct": lambda v: _float_or_none(v) == 0.0,
}

def _float_or_none(s) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _pct(s) -> str:
    v = _float_or_none(s)
    return f"{v:.1f}%" if v is not None else str(s)

def _fmt(val: str, col: str) -> str:
    if col in ("compressed_ppl", "baseline_ppl", "delta_ppl"):
        v = _float_or_none(val)
        return f"{v:.4f}" if v is not None else val
    if col in ("residual_coverage_pct", "expert_param_reduction_pct", "relative_delta_pct"):
        return _pct(val)
    if col in ("residual_applied_experts", "residual_rejected_experts"):
        v = _float_or_none(val)
        return f"{int(v)}" if v is not None else val
    return val if val else "—"

def load_csvs(results_dir: Path) -> List[Dict]:
    rows = []
    for csv_file in sorted(results_dir.glob("*.csv")):
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_source_file"] = str(csv_file.name)
                    rows.append(row)
        except Exception as e:
            print(f"[summarize] WARNING: could not read {csv_file}: {e}", file=sys.stderr)
    return rows

def group_by_target(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for row in rows:
        target = row.get("target_pct") or row.get("expert_target_pct") or "unknown"
        groups[str(target)].append(row)
    return dict(groups)

def check_consistency(rows: List[Dict], target: str) -> List[str]:
    """Emit warnings about cross-method inconsistencies at the same target."""
    warnings = []

    # Check all methods share the same pruning_plan_path (non-empty)
    plan_paths = set(r.get("pruning_plan_path", "") for r in rows if r.get("pruning_plan_path", ""))
    if len(plan_paths) > 1:
        warnings.append(
            f"⚠  target={target}: Multiple pruning_plan_path values found: {plan_paths}"
        )

    # Check actual_pct is consistent across methods
    actual_pcts = set(r.get("actual_pct", "") for r in rows if r.get("actual_pct", ""))
    if len(actual_pcts) > 1:
        warnings.append(
            f"⚠  target={target}: Inconsistent actual_pct across methods: {actual_pcts}"
        )

    # Per-row warnings
    for row in rows:
        method = row.get("moe_pruning_method", "?")

        if str(row.get("residual_fallback_used", "")).lower() in ("true", "1"):
            warnings.append(
                f"⚠  target={target} method={method}: residual_fallback_used=True"
                " (calibration missing or method unsupported)"
            )

        status = row.get("status", "")
        if status.lower() not in ("ok", ""):
            warnings.append(
                f"⚠  target={target} method={method}: status={status!r}"
            )

        cov = _float_or_none(row.get("residual_coverage_pct", ""))
        if cov is not None and cov == 0.0 and method not in ("pure_delete",):
            warnings.append(
                f"⚠  target={target} method={method}: residual_coverage_pct=0"
                " (no experts had residual applied)"
            )

    return warnings

def find_winner(rows: List[Dict]) -> Optional[Dict]:
    """Return row with lowest relative_delta_pct among status=ok rows."""
    ok_rows = [r for r in rows if r.get("status", "").lower() in ("ok", "")]
    if not ok_rows:
        return None
    def sort_key(r):
        v = _float_or_none(r.get("relative_delta_pct", ""))
        return v if v is not None else float("inf")
    return min(ok_rows, key=sort_key)

def make_md_table(rows: List[Dict], cols: List[str]) -> str:
    # Filter to cols that exist in at least one row
    actual_cols = [c for c in cols if any(c in r for r in rows)]
    header = " | ".join(actual_cols)
    sep = " | ".join(["---"] * len(actual_cols))
    lines = [f"| {header} |", f"| {sep} |"]
    for row in rows:
        cells = [_fmt(row.get(c, ""), c) for c in actual_cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Summarize MoE residual sweep results.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result CSV files (default: results/)")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory for summary files (default: results/)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-target table output (only write files)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all CSVs ──────────────────────────────────────────────────────────
    rows = load_csvs(results_dir)
    if not rows:
        print(f"[summarize] No CSV files found in {results_dir}/", file=sys.stderr)
        sys.exit(0)

    print(f"[summarize] Loaded {len(rows)} rows from {results_dir}/")

    # ── Group by target ────────────────────────────────────────────────────────
    groups = group_by_target(rows)
    all_warnings: List[str] = []

    md_sections: List[str] = []
    md_sections.append("# MoE Residual Method Sweep Summary\n")

    # ── Per-target analysis ────────────────────────────────────────────────────
    for target in sorted(groups.keys(), key=lambda x: _float_or_none(x) or 0):
        target_rows = groups[target]
        warnings = check_consistency(target_rows, target)
        all_warnings.extend(warnings)

        winner = find_winner(target_rows)

        if not args.quiet:
            print(f"\n{'═'*70}")
            print(f"  Target pct: {target}%  ({len(target_rows)} runs)")
            print(f"{'═'*70}")

        # Determine cols present in this group
        avail_cols = [c for c in DISPLAY_COLS if any(c in r for r in target_rows)]

        # Print table
        if not args.quiet:
            col_widths = {c: max(len(c), max(len(_fmt(r.get(c, ""), c)) for r in target_rows))
                          for c in avail_cols}
            header_line = "  ".join(c.ljust(col_widths[c]) for c in avail_cols)
            print(header_line)
            print("-" * len(header_line))
            for row in target_rows:
                line = "  ".join(_fmt(row.get(c, ""), c).ljust(col_widths[c]) for c in avail_cols)
                print(line)

        if winner and not args.quiet:
            winner_method = winner.get("moe_pruning_method", "?")
            winner_rdp = winner.get("relative_delta_pct", "?")
            print(f"\n  ✓ Best method (lowest relative_delta_pct): {winner_method}"
                  f"  ({_pct(winner_rdp)})")

        if warnings and not args.quiet:
            print()
            for w in warnings:
                print(f"  {w}")

        # Add to markdown
        md_sections.append(f"\n## Target: {target}%\n")
        md_sections.append(make_md_table(target_rows, DISPLAY_COLS))
        if winner:
            md_sections.append(
                f"\n**Best method:** `{winner.get('moe_pruning_method', '?')}`"
                f" — relative_delta_pct = {_pct(winner.get('relative_delta_pct', ''))}\n"
            )
        if warnings:
            md_sections.append("\n**Warnings:**\n")
            for w in warnings:
                md_sections.append(f"- {w}\n")

    # ── Global warnings summary ────────────────────────────────────────────────
    if all_warnings:
        if not args.quiet:
            print(f"\n{'═'*70}")
            print("  WARNINGS SUMMARY")
            print(f"{'═'*70}")
            for w in all_warnings:
                print(f"  {w}")

    # ── Write summary CSV ──────────────────────────────────────────────────────
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen and k != "_source_file":
                all_keys.append(k)
                seen.add(k)

    csv_out = out_dir / "moe_residual_sweep_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"\n[summarize] Wrote {csv_out}")

    # ── Write summary Markdown ─────────────────────────────────────────────────
    md_out = out_dir / "moe_residual_sweep_summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections))
    print(f"[summarize] Wrote {md_out}")

if __name__ == "__main__":
    main()
