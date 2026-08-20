#!/usr/bin/env python3
"""Summarize p90/p95/p97.5/p99/max against p95 with paired dNLL."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.summarize_moe_allocation_ranking import (
    _read_paired_rows, _validate_paired_documents,
)
from src.paired_bootstrap import paired_bootstrap_nll_difference

ORDER = {"p90": 0, "p95": 1, "p97.5": 2, "p99": 3, "max": 4}


def build_frontier(rows: list[dict], n_resamples: int) -> tuple[list[dict], list[dict]]:
    expected = set(ORDER)
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["allocation_source"], row["layer_channels"])
        grouped.setdefault(key, {})[row["ranking_aggregation"]] = row
    paired, overlaps = [], []
    for key, by_agg in grouped.items():
        if set(by_agg) != expected:
            raise ValueError(f"aggregation frontier incomplete for {key}: {sorted(by_agg)}")
        p95_docs = _read_paired_rows(by_agg["p95"]["per_example_nll_path"])
        with open(by_agg["p95"]["pruning_plan_path"], encoding="utf-8") as handle:
            p95_plan = json.load(handle)
        p95_ids = {int(r["layer_idx"]): set(map(int, r["prune_idx"]))
                   for r in p95_plan["layers"]}
        for agg in sorted(expected, key=ORDER.get):
            candidate = by_agg[agg]
            docs = _read_paired_rows(candidate["per_example_nll_path"])
            _validate_paired_documents(docs, p95_docs, label=f"{agg} versus p95")
            stats = paired_bootstrap_nll_difference(
                [float(r["pruned_nll_sum"]) for r in docs],
                [float(r["pruned_nll_sum"]) for r in p95_docs],
                [int(r["n_tokens"]) for r in docs], n_resamples=n_resamples,
                seed=42,
            )
            paired.append({
                "dataset": key[0], "allocation_source": key[1],
                "removed_layer_channels": key[2], "aggregation": agg,
                "relative_ppl_change_pct": candidate["relative_ppl_change_pct"],
                "aggregation_minus_p95_mean_nll": stats["mean_nll_difference"],
                "ci95_lower": stats["ci_lower"], "ci95_upper": stats["ci_upper"],
                "significant_95pct": (
                    stats["ci_lower"] > 0 or stats["ci_upper"] < 0
                ),
                "n_documents": stats["n_documents"], "n_tokens": stats["n_tokens"],
                "bootstrap_resamples": stats["n_resamples"],
                "bound_tightness_json_path": candidate["bound_tightness_json_path"],
                "sampled_routed_inputs": candidate.get("sampled_routed_inputs", ""),
                "expert_channel_pairs_evaluated": candidate.get(
                    "expert_channel_pairs_evaluated", ""
                ),
                "ellipsoid_tightness_median": candidate.get(
                    "ellipsoid_tightness_median", ""
                ),
                "ellipsoid_tightness_p95": candidate.get(
                    "ellipsoid_tightness_p95", ""
                ),
                "ellipsoid_tightness_p99": candidate.get(
                    "ellipsoid_tightness_p99", ""
                ),
                "ellipsoid_tightness_max": candidate.get(
                    "ellipsoid_tightness_max", ""
                ),
                "ellipsoid_bound_violations": candidate.get(
                    "ellipsoid_bound_violations", ""
                ),
            })
            with open(candidate["pruning_plan_path"], encoding="utf-8") as handle:
                plan = json.load(handle)
            ids = {int(r["layer_idx"]): set(map(int, r["prune_idx"]))
                   for r in plan["layers"]}
            if {li: len(v) for li, v in ids.items()} != {
                li: len(v) for li, v in p95_ids.items()
            }:
                raise ValueError(f"fixed allocation changed for {agg}")
            overlap = sum(len(ids[li] & p95_ids[li]) for li in ids)
            total = sum(len(v) for v in ids.values())
            overlaps.append({
                "dataset": key[0], "aggregation": agg,
                "selected_channels": total, "overlap_with_p95": overlap,
                "changed_ids_per_ranking": total - overlap,
                "global_jaccard_with_p95": (
                    overlap / (2 * total - overlap) if total else 1.0
                ),
            })
    return paired, overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    summary = os.path.join(args.run_dir, "allocation_ranking_summary.csv")
    with open(summary, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paired, overlaps = build_frontier(rows, args.bootstrap_resamples)
    os.makedirs(args.output_dir)
    for filename, data in (
        ("aggregation_frontier_paired.csv", paired),
        ("aggregation_frontier_selection.csv", overlaps),
    ):
        with open(os.path.join(args.output_dir, filename), "x", newline="",
                  encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    with open(os.path.join(args.output_dir, "aggregation_frontier.json"), "x",
              encoding="utf-8") as handle:
        json.dump({"paired": paired, "selection": overlaps}, handle, indent=2)
    with open(os.path.join(args.output_dir, "aggregation_frontier.md"), "x",
              encoding="utf-8") as handle:
        handle.write("| Dataset | Aggregation | Relative PPL % | dNLL vs p95 | 95% CI | Changed IDs | Jaccard vs p95 |\n")
        handle.write("|---|---|---:|---:|---|---:|---:|\n")
        overlap_by_key = {(r["dataset"], r["aggregation"]): r for r in overlaps}
        for row in paired:
            overlap = overlap_by_key[(row["dataset"], row["aggregation"])]
            handle.write(
                f"| {row['dataset']} | {row['aggregation']} | "
                f"{float(row['relative_ppl_change_pct']):.3f} | "
                f"{float(row['aggregation_minus_p95_mean_nll']):+.6f} | "
                f"[{float(row['ci95_lower']):.6f}, {float(row['ci95_upper']):.6f}] | "
                f"{overlap['changed_ids_per_ranking']} | "
                f"{float(overlap['global_jaccard_with_p95']):.4f} |\n"
            )
    with open(os.path.join(args.output_dir, "aggregation_frontier.tex"), "x",
              encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llrrrr}\n\\toprule\nDataset & Aggregation & Rel. PPL & $\\Delta$NLL & 95\\% CI & Changed IDs \\\\\n\\midrule\n")
        for row in paired:
            overlap = overlap_by_key[(row["dataset"], row["aggregation"])]
            handle.write(
                f"{row['dataset']} & {row['aggregation']} & "
                f"{float(row['relative_ppl_change_pct']):.3f} & "
                f"{float(row['aggregation_minus_p95_mean_nll']):+.6f} & "
                f"[{float(row['ci95_lower']):.6f}, {float(row['ci95_upper']):.6f}] & "
                f"{overlap['changed_ids_per_ranking']} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[aggregation-frontier] OK: {len(paired)} paired rows")


if __name__ == "__main__":
    main()
