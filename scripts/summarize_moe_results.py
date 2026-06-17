#!/usr/bin/env python3
"""
summarize_moe_results.py -- compact comparison table for MoE pruning CSVs.

Usage:
    python scripts/summarize_moe_results.py --glob "results/moe_target_pruning_*.csv"
    python scripts/summarize_moe_results.py --glob "results/moe_*.csv" --sort delta_ppl
    python scripts/summarize_moe_results.py --glob "results/*.csv" --wide --no-residual

Columns printed (one row per summary entry):
    file  model  smoke_layers  total_moe_layers  target_pct  actual_pct
    selector  aggregation_mode  pruning_mode  method  physical_pruning
    shape_changed  baseline_ppl  compressed_ppl  delta_ppl  relative_delta_pct
    forward_check  status
    residual_stable_experts  residual_skipped_experts  residual_failed_experts

If a CSV uses old column names they are transparently aliased.
"""
import argparse
import glob as _glob
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed: pip install pandas --break-system-packages")


# -- Column aliases (old name -> canonical name) ------------------------------
ALIASES = {
    "relative_ppl_increase_percent": "relative_delta_pct",
    "smoke_layers_used":              "smoke_layers",
    "requested_target_pct":           "target_pct",
    "actual_pruning_percent":         "actual_pct",
    "notes":                          "status",
    "csv_file":                       "file",
}

# Columns in exact display order (file always first)
DISPLAY_COLS = [
    "file",
    "model",
    "smoke_layers",
    "total_moe_layers",
    "target_pct",
    "actual_pct",
    "selector",
    "aggregation_mode",
    "pruning_mode",
    "method",
    "physical_pruning",
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
]

COL_WIDTHS = {
    "file":                      36,
    "model":                     26,
    "smoke_layers":               8,
    "total_moe_layers":           8,
    "target_pct":                 7,
    "actual_pct":                 8,
    "selector":                  22,
    "aggregation_mode":          10,
    "pruning_mode":              22,
    "method":                    18,
    "physical_pruning":           8,
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
}

FLOAT_COLS = {
    "baseline_ppl", "compressed_ppl", "delta_ppl", "relative_delta_pct",
    "target_pct", "actual_pct",
}


def load_csvs(pattern):
    paths = sorted(_glob.glob(pattern, recursive=True))
    if not paths:
        sys.exit("No files matched: {}".format(pattern))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print("  [WARN] Could not read {}: {}".format(p, e), file=sys.stderr)
            continue
        df["file"] = os.path.basename(p)
        frames.append(df)
    if not frames:
        sys.exit("No readable CSV files found.")
    return pd.concat(frames, ignore_index=True)


def normalize_columns(df):
    rename_map = {old: new for old, new in ALIASES.items()
                  if old in df.columns and new not in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only summary rows (those with baseline_ppl)
    if "baseline_ppl" in df.columns:
        df = df[df["baseline_ppl"].notna()].copy()

    # Derive physical_pruning from pruning_mode if not present
    if "physical_pruning" not in df.columns and "pruning_mode" in df.columns:
        df["physical_pruning"] = df["pruning_mode"].apply(
            lambda m: "yes" if str(m) == "packed_same_channel" else "no"
        )

    return df


def format_val(col, val):
    if not isinstance(val, str) and pd.isna(val):
        return "-"
    if col in ("delta_ppl", "relative_delta_pct"):
        try:
            v = float(val)
            sign = "+" if v >= 0 else ""
            if col == "relative_delta_pct":
                return "{}{:.2f}%".format(sign, v)
            return "{}{:.4f}".format(sign, v)
        except (ValueError, TypeError):
            return str(val)
    if col in FLOAT_COLS:
        try:
            return "{:.4f}".format(float(val))
        except (ValueError, TypeError):
            pass
    return str(val) if val is not None else "-"


def print_table(df, cols, wide=False):
    for c in cols:
        if c not in df.columns:
            df[c] = "-"

    rows = df[cols].fillna("-").values.tolist()
    widths = {c: max(COL_WIDTHS.get(c, len(c)), len(c)) for c in cols}
    if not wide:
        widths["file"] = min(widths.get("file", 36), 36)

    sep = "  "
    hdr_line = sep.join(h.ljust(widths[h]) for h in cols)
    div_line = sep.join("-" * widths[h] for h in cols)
    total_w = len(hdr_line)

    print("=" * total_w)
    print("MOE RESULTS COMPARISON")
    print("=" * total_w)
    print(hdr_line)
    print(div_line)
    for row in rows:
        parts = []
        for h, v in zip(cols, row):
            s = format_val(h, v)
            w = widths[h]
            parts.append(s.rjust(w) if h in FLOAT_COLS else s[:w].ljust(w))
        print(sep.join(parts))
    print("=" * total_w)
    n_files = df["file"].nunique() if "file" in df.columns else "?"
    print("  {} row(s) from {} file(s)".format(len(rows), n_files))
    print("=" * total_w)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--glob", required=True,
                    help="Glob for CSV files, e.g. results/moe_*.csv")
    ap.add_argument("--sort", default="delta_ppl",
                    help="Column to sort by (default: delta_ppl)")
    ap.add_argument("--wide", action="store_true",
                    help="Print full file paths instead of truncating")
    ap.add_argument("--no-residual", action="store_true",
                    help="Omit residual_* columns for narrower output")
    args = ap.parse_args()

    df = load_csvs(args.glob)
    df = normalize_columns(df)

    cols = [c for c in DISPLAY_COLS
            if not (args.no_residual and c.startswith("residual_"))]

    if args.sort in df.columns:
        df = df.sort_values(args.sort, na_position="last")
    else:
        print("  [WARN] Sort column '{}' not found; using default order.".format(args.sort),
              file=sys.stderr)

    print_table(df, cols, wide=args.wide)


if __name__ == "__main__":
    main()
