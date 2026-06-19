#!/usr/bin/env python3
"""
summarize_moe_residual_sweep.py

Reads result CSVs for a specific sweep run (via manifest) and prints
a comparison table grouped by target_pct.

Outputs:
  <out_dir>/summary.csv
  <out_dir>/summary.md

Usage:
  python3 scripts/summarize_moe_residual_sweep.py --manifest results/moe_residual_sweep_runs/<id>/sweep_manifest.json
  python3 scripts/summarize_moe_residual_sweep.py --run-dir  results/moe_residual_sweep_runs/<id>/
  python3 scripts/summarize_moe_residual_sweep.py --manifest ... --quiet
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ── Display columns (in order) ─────────────────────────────────────────────────
DISPLAY_COLS = [
    "moe_pruning_method",
    "requested_method",
    "actual_method",
    "residual_variant",
    "residual_lambda",
    "residual_coverage_pct",
    "residual_applied_experts",
    "residual_rejected_experts",
    "expert_param_reduction_pct",
    "actual_pct",
    "compressed_ppl",
    "baseline_ppl",
    "delta_ppl",
    "relative_delta_pct",
    "forward_check",
    "status",
    "residual_fallback_used",
]

def _float_or_none(s) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _pct(s, decimals: int = 1) -> str:
    v = _float_or_none(s)
    return f"{v:.{decimals}f}%" if v is not None else (str(s) if s else "—")

def _fmt(val: str, col: str) -> str:
    if not val:
        return "—"
    if col in ("compressed_ppl", "baseline_ppl", "delta_ppl"):
        v = _float_or_none(val)
        return f"{v:.4f}" if v is not None else val
    if col in ("residual_coverage_pct", "expert_param_reduction_pct", "relative_delta_pct", "actual_pct"):
        return _pct(val)
    if col in ("residual_applied_experts", "residual_rejected_experts"):
        v = _float_or_none(val)
        return f"{int(v)}" if v is not None else val
    if col == "residual_lambda":
        v = _float_or_none(val)
        if v is not None:
            return f"{v:.0e}" if v < 0.01 else f"{v:.3f}"
    return val

def _get_target(row: Dict) -> str:
    return row.get("target_pct") or row.get("expert_target_pct") or "unknown"

# ── Load CSVs from manifest ────────────────────────────────────────────────────
def load_from_manifest(manifest_path: str) -> tuple[List[Dict], List[str]]:
    """Load CSV rows listed in the manifest. Returns (rows, warnings)."""
    warnings = []
    with open(manifest_path) as f:
        manifest = json.load(f)

    csv_files = manifest.get("csv_files", [])
    if not csv_files:
        # Fallback: look for CSVs in the run dir's csvs/ subdirectory
        run_dir = os.path.dirname(manifest_path)
        csv_dir = os.path.join(run_dir, "csvs")
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "moe_target_pruning_*.csv")))
        if csv_files:
            warnings.append(
                f"manifest had no csv_files list; found {len(csv_files)} CSVs in {csv_dir}/"
            )
        else:
            warnings.append(f"No CSV files found. manifest={manifest_path}")

    rows = []
    for csv_path in csv_files:
        # Safety: skip per_layer CSVs, summary CSVs, unrelated files
        fname = os.path.basename(csv_path)
        if "_per_layer" in fname or "summary" in fname:
            warnings.append(f"Skipping non-main CSV: {fname}")
            continue
        if not fname.startswith("moe_target_pruning_"):
            warnings.append(f"Skipping unexpected CSV: {fname}")
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                file_rows = list(reader)
                rows.extend(file_rows)
        except Exception as e:
            warnings.append(f"Could not read {csv_path}: {e}")

    return rows, warnings

# ── Load CSVs from run directory directly ─────────────────────────────────────
def load_from_run_dir(run_dir: str) -> tuple[List[Dict], List[str]]:
    manifest = os.path.join(run_dir, "sweep_manifest.json")
    if os.path.exists(manifest):
        return load_from_manifest(manifest)
    csv_dir = os.path.join(run_dir, "csvs")
    warnings = [f"No manifest found; loading from {csv_dir}/"]
    rows = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "moe_target_pruning_*.csv"))):
        fname = os.path.basename(f)
        if "_per_layer" in fname or "summary" in fname:
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows, warnings

# ── Cross-method consistency checks ───────────────────────────────────────────
def check_consistency(rows: List[Dict], target: str) -> List[str]:
    warnings = []
    plan_paths = {r.get("pruning_plan_path", "") for r in rows if r.get("pruning_plan_path", "")}
    if len(plan_paths) > 1:
        warnings.append(f"Multiple pruning_plan_path values at target={target}: {plan_paths}")

    actual_pcts = {r.get("actual_pct", "") for r in rows if r.get("actual_pct", "")}
    if len(actual_pcts) > 1:
        warnings.append(f"Inconsistent actual_pct at target={target}: {actual_pcts}")

    for row in rows:
        method = row.get("moe_pruning_method") or row.get("requested_method") or "?"
        if str(row.get("residual_fallback_used", "")).lower() in ("true", "1"):
            warnings.append(
                f"target={target} method={method}: residual_fallback_used=True"
            )
        status = row.get("status", "")
        if status.lower() not in ("ok", ""):
            warnings.append(f"target={target} method={method}: status={status!r}")
        cov = _float_or_none(row.get("residual_coverage_pct", ""))
        if cov is not None and cov == 0.0 and method not in ("pure_delete",):
            warnings.append(
                f"target={target} method={method}: residual_coverage_pct=0"
            )
    return warnings

def find_winner(rows: List[Dict]) -> Optional[Dict]:
    ok_rows = [r for r in rows if r.get("status", "").lower() in ("ok", "")]
    if not ok_rows:
        return None
    def key(r):
        v = _float_or_none(r.get("relative_delta_pct", ""))
        return v if v is not None else float("inf")
    return min(ok_rows, key=key)

# ── Print fixed-width table ────────────────────────────────────────────────────
def print_table(rows: List[Dict], cols: List[str]) -> None:
    avail = [c for c in cols if any(c in r and r[c] for r in rows)]
    widths = {c: max(len(c), max((len(_fmt(r.get(c, ""), c)) for r in rows), default=0))
              for c in avail}
    header = "  ".join(c.ljust(widths[c]) for c in avail)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(_fmt(row.get(c, ""), c).ljust(widths[c]) for c in avail)
        print(line)

def make_md_table(rows: List[Dict], cols: List[str]) -> str:
    avail = [c for c in cols if any(c in r and r[c] for r in rows)]
    header = " | ".join(avail)
    sep = " | ".join(["---"] * len(avail))
    lines = [f"| {header} |", f"| {sep} |"]
    for row in rows:
        cells = [_fmt(row.get(c, ""), c) for c in avail]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="Path to sweep_manifest.json for a specific run")
    ap.add_argument("--run-dir", default=None,
                    help="Path to a sweep run directory (alternative to --manifest)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for summary files (default: same as run dir)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-target table output")
    args = ap.parse_args()

    if not args.manifest and not args.run_dir:
        ap.error("Provide --manifest or --run-dir")

    # Load rows
    if args.manifest:
        rows, load_warnings = load_from_manifest(args.manifest)
        default_out = os.path.dirname(args.manifest)
    else:
        rows, load_warnings = load_from_run_dir(args.run_dir)
        default_out = args.run_dir

    out_dir = Path(args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for w in load_warnings:
        print(f"[summarize] WARN: {w}", file=sys.stderr)

    if not rows:
        print("[summarize] No rows found. Nothing to summarize.", file=sys.stderr)
        sys.exit(0)

    print(f"[summarize] Loaded {len(rows)} rows.")

    # Group by target
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        t = _get_target(row)
        groups[t].append(row)

    if "unknown" in groups:
        print(f"[summarize] WARN: {len(groups['unknown'])} rows have unknown target_pct",
              file=sys.stderr)

    all_warnings: List[str] = []
    md_sections: List[str] = ["# MoE Residual Method Sweep Summary\n"]

    for target in sorted(groups, key=lambda x: _float_or_none(x) or 0):
        target_rows = groups[target]
        warnings = check_consistency(target_rows, target)
        all_warnings.extend(warnings)
        winner = find_winner(target_rows)

        if not args.quiet:
            print(f"\n{'═'*70}")
            print(f"  Target: {target}%   ({len(target_rows)} rows)")
            print(f"{'═'*70}")
            print_table(target_rows, DISPLAY_COLS)
            if winner:
                wm = winner.get("moe_pruning_method") or winner.get("actual_method") or "?"
                print(f"\n  ✓ Best: {wm}  relative_delta_pct={_pct(winner.get('relative_delta_pct',''))}")
            for w in warnings:
                print(f"  ⚠  {w}")

        md_sections.append(f"\n## Target: {target}%\n")
        md_sections.append(make_md_table(target_rows, DISPLAY_COLS))
        if winner:
            wm = winner.get("moe_pruning_method") or winner.get("actual_method") or "?"
            md_sections.append(
                f"\n**Best:** `{wm}` — relative_delta_pct={_pct(winner.get('relative_delta_pct',''))}\n"
            )
        if warnings:
            md_sections.append("\n**Warnings:**\n" + "\n".join(f"- {w}" for w in warnings) + "\n")

    if all_warnings:
        if not args.quiet:
            print(f"\n{'═'*70}")
            print("  WARNINGS")
            print(f"{'═'*70}")
            for w in all_warnings:
                print(f"  ⚠  {w}")

    # Write summary CSV
    all_keys = []
    seen_keys: set = set()
    for r in rows:
        for k in r:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    csv_out = out_dir / "summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in all_keys} for r in rows)
    print(f"\n[summarize] Wrote {csv_out}")

    md_out = out_dir / "summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections))
    print(f"[summarize] Wrote {md_out}")

if __name__ == "__main__":
    main()
