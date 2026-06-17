#!/usr/bin/env python3
"""
summarize_moe_results.py — compact comparison table for MoE pruning CSVs.

Usage:
    python scripts/summarize_moe_results.py --glob "results/moe_target_pruning_*.csv"
    python scripts/summarize_moe_results.py --glob "results/moe_*.csv" --sort delta_ppl
    python scripts/summarize_moe_results.py --glob "results/*.csv" --wide

Columns printed (one row per CSV result entry):
    model  smoke_layers  target_pct  actual_pct  selector  aggregation_mode
    pruning_mode  method  baseline_ppl  compressed_ppl  delta_ppl
    relative_delta_pct  forward_check  status
    residual_stable_experts  residual_skipped_experts  residual_failed_experts
    csv_file

If a CSV uses the old column name (relative_ppl_increase_percent), it is
transparently aliased to relative_delta_pct.
"""
import argparse
import glob as _glob
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed: pip install pandas --break-system-packages")


# ── Column aliases (old → new name) ──────────────────────────────────────────
ALIASES = {
    "relative_ppl_increase_percent": "relative_delta_pct",
    "smoke_layers_used":              "smoke_layers",
    "requested_target_pct":           "target_pct",
    "actual_pruning_percent":         "actual_pct",
    "notes":                          "status",
}

DISPLAY_COLS = [
    "model",
    "smoke_layers",
    "target_pct",
    "actual_pct",
    "selector",
    "aggregation_mode",
    "pruning_mode",
    "method",
    "shape_changed",
    "baseline_ppl",
    "compressed_ppl",
    "delta_ppl",
    "relative_delta_pct",
    "forward_check",
    "status",
    "residual_stable_experts",
    "residual_skipped_experts",
    "residual_failed_experts",
    "csv_file",
]

# Formatting widths
COL_WIDTHS = {
    "model":                     30,
    "smoke_layers":               8,
    "target_pct":                 7,
    "actual_pct":                 8,
    "selector":                  22,
    "aggregation_mode":          10,
    "pruning_mode":              22,
    "method":                    18,
    "shape_changed":              6,
    "baseline_ppl":              10,
    "compressed_ppl":            12,
    "delta_ppl":                  9,
    "relative_delta_pct":         9,
    "forward_check":              5,
    "status":                    18,
    "residual_stable_experts":    7,
    "residual_skipped_experts":   7,
    "residual_failed_experts":    7,
    "csv_file":                  40,
}

FLOAT_COLS = {"baseline_ppl", "compressed_ppl", "delta_ppl", "relative_delta_pct",
              "target_pct", "actual_pct"}


def load_csvs(pattern: str) -> "pd.DataFrame":
    paths = sorted(_glob.glob(pattern, recursive=True))
    if not paths:
        sys.exit(f"No files matched: {pattern}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"  [WARN] Could not read {p}: {e}", file=sys.stderr)
            continue
        df["csv_file"] = os.path.basename(p)
        frames.append(df)
    if not frames:
        sys.exit("No readable CSV files found.")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.rename(columns=ALIASES)
    # Keep only summary rows (those with baseline_ppl)
    if "baseline_ppl" in df.columns:
        df = df[df["baseline_ppl"].notna()]
    return df


def format_val(col: str, val) -> str:
    if pd.isna(val) if not isinstance(val, str) else False:
        return "-"
    if col in ("delta_ppl", "relative_delta_pct"):
        try:
            v = float(val)
            sign = "+" if v >= 0 else ""
            if col == "relative_delta_pct":
                return f"{sign}{v:.2f}%"
            return f"{sign}{v:.4f}"
        except (ValueError, TypeError):
            return str(val)
    if col in FLOAT_COLS:
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            pass
    return str(val) if val is not None else "-"


def print_table(df: "pd.DataFrame", cols: list, wide: bool = False) -> None:
    # Ensure all requested cols exist (fill missing with "-")
    for c in cols:
        if c not in df.columns:
            df[c] = "-"

    rows = df[cols].fillna("-").values.tolist()
    headers = cols

    widths = {c: max(COL_WIDTHS.get(c, len(c)), len(c)) for c in headers}
    if not wide:
        # Compact: truncate csv_file to basename already done
        widths["csv_file"] = min(widths["csv_file"], 40)

    sep = "  "
    hdr_line = sep.join(h.ljust(widths[h]) for h in headers)
    div_line = sep.join("─" * widths[h] for h in headers)
    total_w  = len(hdr_line)

    print("=" * total_w)
    print("MOE RESULTS COMPARISON")
    print("=" * total_w)
    print(hdr_line)
    print(div_line)
    for row in rows:
        parts = []
        for h, v in zip(headers, row):
            s = format_val(h, v)
            w = widths[h]
            parts.append(s[:w].ljust(w) if h not in FLOAT_COLS else s.rjust(w))
        print(sep.join(parts))
    print("=" * total_w)
    print(f"  {len(rows)} row(s) from {df['csv_file'].nunique()} file(s)")
    print("=" * total_w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True,
                    help='Glob pattern for CSV files, e.g. "results/moe_*.csv"')
    ap.add_argument("--sort", default="delta_ppl",
                    help="Column to sort by (default: delta_ppl)")
    ap.add_argument("--wide", action="store_true",
                    help="Print extra-wide table with full csv_file path")
    ap.add_argument("--no-residual", action="store_true",
                    help="Omit residual stats columns for narrower output")
    args = ap.parse_args()

    df = load_csvs(args.glob)
    df = normalize_columns(df)

    cols = [c for c in DISPLAY_COLS if not (args.no_residual and c.startswith("residual_"))]

    sort_col = args.sort
    if sort_col in df.columns:
        df = df.sort_values(sort_col, na_position="last")
    else:
        print(f"  [WARN] Sort column '{sort_col}' not found; using default order.",
              file=sys.stderr)

    print_table(df, cols, wide=args.wide)


if __name__ == "__main__":
    main()
