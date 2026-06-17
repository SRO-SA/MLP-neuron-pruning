#!/usr/bin/env python3
"""
summarize_moe_results.py -- compact comparison table for MoE pruning CSVs.

Usage:
    python scripts/summarize_moe_results.py --glob "results/moe_*.csv"
    python scripts/summarize_moe_results.py --glob "results/moe_*.csv" --sort delta_ppl
    python scripts/summarize_moe_results.py --glob "results/moe_*.csv" --dataset wikitext2
    python scripts/summarize_moe_results.py --glob "results/moe_*.csv" --method pure_delete --target 4 --n-eval 512

Columns printed:
    file  model  layers  target_pct  actual_pct  method  dataset  n_eval
    baseline_ppl  compressed_ppl  delta_ppl  relative_delta_pct
    expert_param_reduction_pct  total_model_param_reduction_pct
    estimated_active_expert_flop_reduction_pct  forward_check  status

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
    "smoke_layers_used":              "processed_moe_layers",
    "requested_target_pct":           "target_pct",
    "actual_pruning_percent":         "actual_pct",
    "notes":                          "status",
    "csv_file":                       "file",
    "reconstruction_eval_samples":    "n_eval",
}

# Columns in exact display order
DISPLAY_COLS = [
    "file",
    "model",
    "layers",               # processed_moe_layers / total_moe_layers
    "target_pct",
    "actual_pct",
    "method",
    "dataset",              # eval_dataset
    "n_eval",
    "baseline_ppl",
    "compressed_ppl",
    "delta_ppl",
    "relative_delta_pct",
    "expert_param_reduction_pct",
    "total_model_param_reduction_pct",
    "estimated_active_expert_flop_reduction_pct",
    "forward_check",
    "status",
]

COL_WIDTHS = {
    "file":                                         36,
    "model":                                        26,
    "layers":                                        7,
    "target_pct":                                    7,
    "actual_pct":                                    8,
    "method":                                       18,
    "dataset":                                      10,
    "n_eval":                                        6,
    "baseline_ppl":                                 10,
    "compressed_ppl":                               12,
    "delta_ppl":                                     9,
    "relative_delta_pct":                            9,
    "expert_param_reduction_pct":                    9,
    "total_model_param_reduction_pct":               9,
    "estimated_active_expert_flop_reduction_pct":    9,
    "forward_check":                                 5,
    "status":                                       18,
}

FLOAT_COLS = {
    "baseline_ppl", "compressed_ppl", "delta_ppl", "relative_delta_pct",
    "target_pct", "actual_pct",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "estimated_active_expert_flop_reduction_pct",
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
    # Rename old -> canonical (skip if target already present)
    rename_map = {old: new for old, new in ALIASES.items()
                  if old in df.columns and new not in df.columns}
    df = df.rename(columns=rename_map)

    # Unify eval_dataset -> dataset
    if "eval_dataset" in df.columns and "dataset" not in df.columns:
        df = df.rename(columns={"eval_dataset": "dataset"})

    # Build layers display: "processed / total" or just total
    if "processed_moe_layers" in df.columns and "total_moe_layers" in df.columns:
        df["layers"] = (
            df["processed_moe_layers"].fillna("?").astype(str)
            + "/"
            + df["total_moe_layers"].fillna("?").astype(str)
        )
    elif "total_moe_layers" in df.columns:
        df["layers"] = df["total_moe_layers"].fillna("?").astype(str)
    # else: "layers" will be filled with "-" by print_table

    # Keep only summary rows (those with baseline_ppl)
    if "baseline_ppl" in df.columns:
        df = df[df["baseline_ppl"].notna()].copy()

    return df


def apply_filters(df, args):
    if args.dataset and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == args.dataset.lower()]
    if args.method and "method" in df.columns:
        df = df[df["method"].astype(str).str.lower() == args.method.lower()]
    if args.target is not None and "target_pct" in df.columns:
        df = df[df["target_pct"].apply(
            lambda v: abs(float(v) - args.target) < 0.01
            if pd.notna(v) else False
        )]
    if args.n_eval is not None and "n_eval" in df.columns:
        df = df[df["n_eval"].apply(
            lambda v: int(float(v)) == args.n_eval
            if pd.notna(v) else False
        )]
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
    if col in ("expert_param_reduction_pct", "total_model_param_reduction_pct",
               "estimated_active_expert_flop_reduction_pct"):
        try:
            return "{:.3f}%".format(float(val))
        except (ValueError, TypeError):
            pass
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
    # Filters
    ap.add_argument("--dataset",
                    help="Filter by dataset name (e.g. wikitext2 or c4)")
    ap.add_argument("--method",
                    help="Filter by method (e.g. pure_delete or residual_full_moe)")
    ap.add_argument("--target", type=float, default=None,
                    help="Filter by target pruning pct (e.g. 4.0)")
    ap.add_argument("--n-eval", dest="n_eval", type=int, default=None,
                    help="Filter by n_eval size (e.g. 512)")
    args = ap.parse_args()

    df = load_csvs(args.glob)
    df = normalize_columns(df)
    df = apply_filters(df, args)

    if df.empty:
        print("No rows match the given filters.")
        return

    if args.sort in df.columns:
        df = df.sort_values(args.sort, na_position="last")
    else:
        print("  [WARN] Sort column '{}' not found; using default order.".format(args.sort),
              file=sys.stderr)

    print_table(df, DISPLAY_COLS, wide=args.wide)


if __name__ == "__main__":
    main()
