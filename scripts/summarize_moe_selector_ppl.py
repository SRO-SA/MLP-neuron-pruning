#!/usr/bin/env python3
"""
summarize_moe_selector_ppl.py
==============================
Post-process a selector-baseline PPL benchmark run.

Data collection strategy (per experiment subfolder):
  1. JSON first: moe_target_pruning_*.json -> data["results"][0]
     JSON has shape fields: new_intermediate, old_intermediate, moe_channel_alignment
  2. CSV merged in for any fields missing from JSON row
  3. Pruning plan path recovered by scanning {subfolder}/pruning_plans/*.json

Selector name normalisation:
  moe_pruning.py writes _sel_str = f"{moe_selector}_{chan_agg}" (e.g.
  "rmsnorm_bound_p95") when pruning_mode=="packed_same_channel".
  This script strips the aggregation suffix so the base name matches the
  known_selectors list.

Outputs (written to --run-dir):
  selector_baseline_summary.csv    -- one row per selector/target
  selector_attribution_summary.csv -- ranking with delta vs rmsnorm_bound

Usage:
  python3 scripts/summarize_moe_selector_ppl.py \\
      --run-dir results/moe_selector_baselines/20260707_194241 \\
      --selectors rmsnorm_bound,down_norm,activation_score,random \\
      --targets 2,4,6,8 \\
      --min-rows 16

  # Rebuild via shell wrapper:
  SUMMARIZE_ONLY=1 RUN_DIR=results/moe_selector_baselines/20260707_194241 \\
      bash scripts/run_moe_selector_baseline_ppl.sh
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ── Output schemas ────────────────────────────────────────────────────────────

BASELINE_SUMMARY_FIELDS = [
    "run_id", "model", "dataset", "selector", "method",
    "target_pct", "actual_pct",
    "old_moe_dim", "moe_dim", "alignment",
    "selected_layer_channels",
    "expert_param_reduction_pct",
    "total_model_param_reduction_pct",
    "estimated_active_expert_flop_reduction_pct",
    "ppl_base", "ppl_pruned", "ppl_delta", "ppl_rel_inc_pct",
    "forward_check", "shape_changed",
    "n_eval", "moe_calib_samples",
    "pruning_plan_path", "loaded_pruning_plan",
    "csv_path", "json_path",
    "status",
]

ATTRIBUTION_FIELDS = [
    "selector", "target_pct", "actual_pct",
    "old_moe_dim", "moe_dim",
    "ppl_rel_inc_pct", "delta_vs_rmsnorm_bound", "rank_at_target", "status",
]

# Aggregation suffixes appended by moe_pruning.py to moe_selector name
_AGG_SUFFIXES = ("_p95", "_p75", "_mean", "_max", "_per_expert")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flt(val: Any, fallback: float = float("nan")) -> float:
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return fallback


def _int_or(val: Any, fallback: int) -> int:
    f = _flt(val)
    if math.isnan(f):
        return fallback
    return int(f)


def _safe_round(val: Any, ndigits: int = 4) -> Any:
    """Round floats; return empty string for NaN/missing."""
    f = _flt(val)
    if math.isnan(f):
        return ""
    return round(f, ndigits)


def _normalize_selector(sel_str: str, known_selectors: List[str]) -> str:
    """
    Strip aggregation suffixes added by moe_pruning.py.

    "rmsnorm_bound_p95" -> "rmsnorm_bound"  (when "rmsnorm_bound" is known)
    "down_norm_p95"     -> "down_norm"
    Returns sel_str unchanged if no known selector prefix matches.
    """
    if sel_str in known_selectors:
        return sel_str
    for suffix in _AGG_SUFFIXES:
        if sel_str.endswith(suffix):
            base = sel_str[: -len(suffix)]
            if base in known_selectors:
                return base
    for known in known_selectors:
        if sel_str.startswith(known + "_"):
            return known
    return sel_str


def infer_moe_dim(actual_pct: float, orig: int, align: int) -> int:
    """
    Fallback: estimate new intermediate dim from actual pruning %.
    Returns orig when actual_pct is 0 or NaN.
    """
    if math.isnan(actual_pct) or actual_pct <= 0:
        return orig
    pruned = round(orig * actual_pct / 100.0 / align) * align
    return max(0, orig - pruned)


def _is_experiment_row(row: Dict) -> bool:
    """
    Return True only for experiment-level summary rows (not per-layer rows).
    Per-layer rows lack selector, target_pct, and PPL values.
    """
    if not str(row.get("selector", "")).strip():
        return False
    # target_pct can be named differently
    tgt = (row.get("target_pct")
           or row.get("requested_target_pct")
           or row.get("target_pruning_percent", ""))
    if not str(tgt).strip():
        return False
    bp = _flt(row.get("baseline_ppl"))
    cp = _flt(row.get("compressed_ppl"))
    return not (math.isnan(bp) or math.isnan(cp))


# ── Per-subfolder data collection ─────────────────────────────────────────────

def collect_experiments(run_dir: str) -> List[Dict]:
    """
    Scan each subdirectory of run_dir for one experiment's data.

    For each subfolder:
      1. Read JSON -> data["results"][0]  (has new_intermediate, old_intermediate,
         moe_channel_alignment, shape_changed, pruning_plan_path, etc.)
      2. Merge CSV row for any fields missing from JSON
      3. Discover pruning_plan_path from {subfolder}/pruning_plans/*.json if not
         already populated in the data files.

    Returns list of merged dicts, one per experiment subfolder.
    """
    rows: List[Dict] = []

    try:
        entries = sorted(os.listdir(run_dir))
    except OSError as exc:
        print(f"[summarize] ERROR: cannot list {run_dir}: {exc}", flush=True)
        return rows

    for entry in entries:
        subfolder = os.path.join(run_dir, entry)
        if not os.path.isdir(subfolder):
            continue

        # ── File discovery ────────────────────────────────────────────────────
        json_files = sorted(glob.glob(
            os.path.join(subfolder, "moe_target_pruning_*.json")
        ))
        csv_files = sorted(glob.glob(
            os.path.join(subfolder, "moe_target_pruning_*.csv")
        ))
        csv_files = [f for f in csv_files if not f.endswith("_per_layer.csv")]

        # Pruning plan: check filesystem even if not recorded in raw files
        plan_files = sorted(glob.glob(
            os.path.join(subfolder, "pruning_plans", "*.json")
        ))
        discovered_plan = plan_files[0] if plan_files else ""

        if not json_files and not csv_files:
            continue

        row: Optional[Dict] = None
        json_path = ""
        csv_path = ""

        # ── 1. JSON (preferred: has shape fields) ─────────────────────────────
        for jf in json_files:
            try:
                with open(jf) as fh:
                    data = json.load(fh)
                # Results are nested under "results", NOT at top level
                results_list = data.get("results", [])
                if not isinstance(results_list, list):
                    continue
                for r in results_list:
                    if isinstance(r, dict) and _is_experiment_row(r):
                        row = dict(r)
                        json_path = jf
                        break
            except Exception as exc:
                print(f"[summarize] WARNING: cannot read JSON {jf}: {exc}", flush=True)
            if row is not None:
                break

        # ── 2. CSV merge (fill in any missing fields) ─────────────────────────
        for cf in csv_files:
            try:
                with open(cf, newline="") as fh:
                    for r in csv.DictReader(fh):
                        if not _is_experiment_row(r):
                            continue
                        if row is None:
                            row = dict(r)
                        else:
                            # Merge: fill in fields that JSON row lacks
                            for k, v in r.items():
                                if k not in row or str(row.get(k, "")).strip() == "":
                                    row[k] = v
                        csv_path = cf
                        break
            except Exception as exc:
                print(f"[summarize] WARNING: cannot read CSV {cf}: {exc}", flush=True)
            if csv_path:
                break

        if row is None:
            continue

        # ── 3. Attach metadata ────────────────────────────────────────────────
        row["_json_path"] = json_path
        row["_csv_path"] = csv_path
        row["_discovered_plan"] = discovered_plan

        rows.append(row)

    return rows


# ── Post-write verification ───────────────────────────────────────────────────

def _verify_csv_written(path: str, expected_rows: int, label: str) -> None:
    """
    Assert file exists, is non-zero, and has the expected number of data rows.
    """
    if not os.path.exists(path):
        print(f"[summarize] ERROR: {label} not found after writing: {path}", flush=True)
        sys.exit(1)

    size = os.path.getsize(path)
    if size == 0:
        print(f"[summarize] ERROR: {label} is 0 bytes after writing", flush=True)
        sys.exit(1)

    try:
        import pandas as pd
        df   = pd.read_csv(path)
        nrow = len(df)
    except ImportError:
        with open(path, newline="") as fh:
            nrow = sum(1 for _ in csv.DictReader(fh))

    if nrow != expected_rows:
        print(
            f"[summarize] ERROR: {label} has {nrow} data rows, "
            f"expected {expected_rows}",
            flush=True,
        )
        sys.exit(1)

    fname = os.path.basename(path)
    print(f"[summarize] {fname} rows: {nrow} size: {size}", flush=True)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_rows(rows: List[Dict], min_rows: int) -> None:
    errors: List[str] = []

    if len(rows) < min_rows:
        errors.append(
            f"Expected >= {min_rows} experiment-level rows, found {len(rows)}."
        )

    for i, row in enumerate(rows):
        bp = _flt(row.get("baseline_ppl"))
        cp = _flt(row.get("compressed_ppl"))
        if math.isnan(bp) or math.isnan(cp):
            errors.append(
                f"Row {i}: selector={row.get('selector')!r} "
                f"target={row.get('target_pct')!r}: PPL missing."
            )

    for i, row in enumerate(rows):
        fwd = str(row.get("forward_check", "True")).strip().lower()
        if fwd in ("false", "0", "fail"):
            errors.append(
                f"Row {i}: selector={row.get('selector')!r} "
                f"target={row.get('target_pct')!r}: forward_check={fwd!r}."
            )

    for i, row in enumerate(rows):
        st = str(row.get("status", "ok")).strip().lower()
        if st not in ("ok", ""):
            errors.append(
                f"Row {i}: selector={row.get('selector')!r} "
                f"target={row.get('target_pct')!r}: status={st!r}."
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
    ap.add_argument("--min-rows",     type=int, default=2)
    args = ap.parse_args()

    run_dir         = args.run_dir
    run_id          = os.path.basename(run_dir.rstrip("/\\"))
    summary_csv     = os.path.join(run_dir, "selector_baseline_summary.csv")
    attribution_csv = os.path.join(run_dir, "selector_attribution_summary.csv")

    known_selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]

    # ── Collect experiment rows ───────────────────────────────────────────────
    print(f"[summarize] Collecting from {run_dir} ...", flush=True)

    raw_rows = collect_experiments(run_dir)
    print(f"[summarize] Found {len(raw_rows)} experiment subfolder(s).", flush=True)

    if not raw_rows:
        print(f"[summarize] ERROR: no experiment-level rows found in {run_dir}", flush=True)
        print("[summarize]   Expected subfolders containing moe_target_pruning_*.json", flush=True)
        sys.exit(1)

    # ── Normalise selector names (_p95 suffix etc.) ───────────────────────────
    for row in raw_rows:
        raw_sel = str(row.get("selector", "")).strip()
        norm    = _normalize_selector(raw_sel, known_selectors)
        if norm != raw_sel:
            print(f"[summarize] Normalised selector {raw_sel!r} -> {norm!r}", flush=True)
        row["selector"] = norm

    # ── Per-selector counts ───────────────────────────────────────────────────
    sel_counts: Dict[str, int] = defaultdict(int)
    for r in raw_rows:
        sel_counts[r["selector"]] += 1
    print(f"[summarize] Rows by selector:", flush=True)
    for sel, cnt in sorted(sel_counts.items()):
        print(f"[summarize]   {sel}: {cnt}", flush=True)

    # Warn (not fail) for expected selectors not found
    for sel in known_selectors:
        if sel not in sel_counts:
            print(f"[summarize] WARNING: expected selector {sel!r} not found "
                  f"(normal for SMOKE/partial runs).", flush=True)

    # ── Validate ──────────────────────────────────────────────────────────────
    validate_rows(raw_rows, args.min_rows)
    print("[summarize] Validation OK.", flush=True)

    # ── Build selector_baseline_summary ──────────────────────────────────────
    summary_rows: List[Dict] = []
    for row in raw_rows:
        # ── Shape fields: prefer JSON (has new_intermediate) ──────────────────
        # new_intermediate / old_intermediate are written by moe_pruning.py
        # into the JSON results dict but NOT into the summary CSV.
        new_int = _flt(row.get("new_intermediate",
                        row.get("new_i",
                        row.get("new_intermediate_size"))))
        old_int = _flt(row.get("old_intermediate",
                        row.get("old_i",
                        row.get("old_intermediate_size"))))
        chan_align = _flt(row.get("moe_channel_alignment",
                           row.get("channel_alignment")))

        if not math.isnan(new_int):
            moe_dim = int(new_int)
        else:
            # Fallback: infer from actual_pct
            actual_pct_raw = _flt(row.get("actual_pct",
                                   row.get("actual_pruning_percent", 0)))
            moe_dim = infer_moe_dim(actual_pct_raw, args.orig_moe_dim, args.moe_align)

        if not math.isnan(old_int):
            old_moe_dim = int(old_int)
        else:
            old_moe_dim = args.orig_moe_dim

        if not math.isnan(chan_align):
            alignment = int(chan_align)
        else:
            alignment = args.moe_align

        # ── PPL fields ────────────────────────────────────────────────────────
        actual_pct = _flt(row.get("actual_pct",
                           row.get("actual_pruning_percent", 0)))
        bp  = _flt(row.get("baseline_ppl"))
        cp  = _flt(row.get("compressed_ppl"))
        dp  = _flt(row.get("delta_ppl"))
        rel = _flt(row.get("relative_delta_pct"))

        # ── target_pct: handle multiple field names ───────────────────────────
        tgt = (row.get("target_pct")
               or row.get("requested_target_pct")
               or row.get("target_pruning_percent", ""))

        # ── Pruning plan path: filesystem discovery first ─────────────────────
        pruning_plan_path = (
            row.get("_discovered_plan")
            or row.get("pruning_plan_path", "")
            or "NONE"
        )
        if not pruning_plan_path:
            pruning_plan_path = "NONE"

        summary_rows.append({
            "run_id":    run_id,
            "model":     row.get("model", args.model),
            "dataset":   row.get("eval_dataset", args.dataset),
            "selector":  row["selector"],
            "method":    row.get("method", row.get("actual_method", "")),
            "target_pct":    tgt,
            "actual_pct":    _safe_round(actual_pct, 4),
            "old_moe_dim":   old_moe_dim,
            "moe_dim":       moe_dim,
            "alignment":     alignment,
            "selected_layer_channels":                row.get("selected_layer_channels", ""),
            "expert_param_reduction_pct":             _safe_round(
                row.get("expert_param_reduction_pct"), 4),
            "total_model_param_reduction_pct":        _safe_round(
                row.get("total_model_param_reduction_pct"), 4),
            "estimated_active_expert_flop_reduction_pct": _safe_round(
                row.get("estimated_active_expert_flop_reduction_pct"), 4),
            "ppl_base":          _safe_round(bp, 6),
            "ppl_pruned":        _safe_round(cp, 6),
            "ppl_delta":         _safe_round(dp, 6),
            "ppl_rel_inc_pct":   _safe_round(rel, 4),
            "forward_check":     row.get("forward_check", ""),
            "shape_changed":     row.get("shape_changed", ""),
            "n_eval":            row.get("n_eval", row.get("reconstruction_eval_samples", "")),
            "moe_calib_samples": row.get("moe_calib_samples", ""),
            "pruning_plan_path": pruning_plan_path,
              "loaded_pruning_plan": row.get("loaded_pruning_plan", ""),
            "csv_path":  row.get("_csv_path", ""),
            "json_path": row.get("_json_path", ""),
            "status":    row.get("status", ""),
        })

    # Write and verify selector_baseline_summary.csv
    os.makedirs(run_dir, exist_ok=True)
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BASELINE_SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    _verify_csv_written(summary_csv, len(summary_rows), "selector_baseline_summary.csv")

    # Build selector_attribution_summary
    best: Dict[Tuple[str, str], Dict] = {}
    for r in summary_rows:
        key      = (str(r["target_pct"]), str(r["selector"]))
        existing = best.get(key)
        if existing is None or _flt(r["ppl_rel_inc_pct"]) < _flt(existing["ppl_rel_inc_pct"]):
            best[key] = r

    attribution_rows: List[Dict] = []
    targets_found = sorted(
        {str(r["target_pct"]) for r in summary_rows},
        key=lambda x: _flt(x),
    )

    for tgt in targets_found:
        rows_at_tgt = [best[(tgt, sel)] for sel in known_selectors if (tgt, sel) in best]
        for key, r in best.items():
            if key[0] == tgt and r["selector"] not in known_selectors:
                rows_at_tgt.append(r)

        rows_at_tgt.sort(key=lambda r: _flt(r["ppl_rel_inc_pct"]))

        ref_rel = next(
            (_flt(r["ppl_rel_inc_pct"]) for r in rows_at_tgt
             if r["selector"] == "rmsnorm_bound"),
            None,
        )

        for rank, r in enumerate(rows_at_tgt, start=1):
            rel       = _flt(r["ppl_rel_inc_pct"])
            delta_ref = round(rel - ref_rel, 4) if ref_rel is not None else "NA"
            attribution_rows.append({
                "selector":               r["selector"],
                "target_pct":             r["target_pct"],
                "actual_pct":             r["actual_pct"],
                "old_moe_dim":            r["old_moe_dim"],
                "moe_dim":                r["moe_dim"],
                "ppl_rel_inc_pct":        _safe_round(rel, 4),
                "delta_vs_rmsnorm_bound": delta_ref,
                "rank_at_target":         rank,
                "status":                 r["status"],
            })

    # Write and verify selector_attribution_summary.csv
    with open(attribution_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ATTRIBUTION_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(attribution_rows)

    _verify_csv_written(attribution_csv, len(attribution_rows), "selector_attribution_summary.csv")

    # Compact ranking table
    print("\n[summarize] -- Selector Attribution: PPL relative increase --")
    for tgt in targets_found:
        rows_at_tgt = sorted(
            [r for r in attribution_rows if str(r["target_pct"]) == tgt],
            key=lambda r: int(r["rank_at_target"]),
        )
        tgt_int = int(float(tgt))
        moe_dim = rows_at_tgt[0]["moe_dim"] if rows_at_tgt else "?"
        old_dim = rows_at_tgt[0]["old_moe_dim"] if rows_at_tgt else "?"
        print(f"\n  Target {tgt_int}%  (old_moe_dim={old_dim} -> moe_dim={moe_dim}):")
        for r in rows_at_tgt:
            sel       = str(r["selector"])
            rel       = _flt(r["ppl_rel_inc_pct"])
            delta     = r["delta_vs_rmsnorm_bound"]
            rank      = r["rank_at_target"]
            delta_str = (
                f"  [{delta:+.3f}% vs rmsnorm_bound]"
                if isinstance(delta, float) else ""
            )
            print(f"    {rank}. {sel:<22s}  {rel:+.3f}%{delta_str}")
    print()

    print(
        f"[summarize] OK: {len(summary_rows)} summary row(s), "
        f"{len(attribution_rows)} attribution row(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
