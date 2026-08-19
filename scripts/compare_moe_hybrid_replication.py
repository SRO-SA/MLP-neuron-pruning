#!/usr/bin/env python3
"""Compare an independent hybrid replication with its reference run."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def _one(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {pattern!r}; found {matches}")
    return matches[0]


def _result_row(run_dir: str) -> tuple[str, dict]:
    paths = [
        value for value in glob.glob(os.path.join(run_dir, "moe_target_pruning_*.csv"))
        if not value.endswith("_per_layer.csv")
    ]
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one main CSV in {run_dir}: {paths}")
    path = paths[0]
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("eval_dataset") == "wikitext2"]
    if len(rows) != 1:
        raise ValueError(f"expected one wikitext2 result in {path}; found {len(rows)}")
    return path, rows[0]


def _plan_ids(run_dir: str) -> tuple[str, dict[int, list[int]]]:
    path = _one(os.path.join(run_dir, "pruning_plans", "*.json"))
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    return path, {
        int(row["layer_idx"]): sorted(int(value) for value in row.get("prune_idx", []))
        for row in plan["layers"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--replication-dir", required=True)
    parser.add_argument("--relative-ppl-tolerance", type=float, default=0.02)
    parser.add_argument("--baseline-ppl-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    reference_csv, reference = _result_row(args.reference_dir)
    replication_csv, replication = _result_row(args.replication_dir)
    reference_plan, reference_ids = _plan_ids(args.reference_dir)
    replication_plan, replication_ids = _plan_ids(args.replication_dir)
    if reference_ids != replication_ids:
        changed = sorted(
            layer for layer in set(reference_ids) | set(replication_ids)
            if reference_ids.get(layer) != replication_ids.get(layer)
        )
        raise AssertionError(f"replication selected different channel IDs: {changed}")
    baseline_delta = abs(
        float(reference["baseline_ppl"]) - float(replication["baseline_ppl"])
    )
    relative_delta = abs(
        float(reference["relative_delta_pct"])
        - float(replication["relative_delta_pct"])
    )
    if baseline_delta > args.baseline_ppl_tolerance:
        raise AssertionError(
            f"baseline PPL changed by {baseline_delta}, above tolerance "
            f"{args.baseline_ppl_tolerance}"
        )
    if relative_delta > args.relative_ppl_tolerance:
        raise AssertionError(
            f"relative PPL changed by {relative_delta} percentage points, above "
            f"tolerance {args.relative_ppl_tolerance}"
        )
    report = {
        "reference_csv": reference_csv,
        "replication_csv": replication_csv,
        "reference_plan": reference_plan,
        "replication_plan": replication_plan,
        "selected_channel_ids_identical": True,
        "total_layer_channels": sum(len(values) for values in reference_ids.values()),
        "reference_baseline_ppl": float(reference["baseline_ppl"]),
        "replication_baseline_ppl": float(replication["baseline_ppl"]),
        "baseline_ppl_abs_difference": baseline_delta,
        "reference_relative_ppl_pct": float(reference["relative_delta_pct"]),
        "replication_relative_ppl_pct": float(replication["relative_delta_pct"]),
        "relative_ppl_percentage_point_difference": relative_delta,
        "deterministic_within_tolerance": True,
    }
    output = os.path.join(args.replication_dir, "replication_comparison.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        "[replication] OK: identical channel IDs; baseline Δ="
        f"{baseline_delta:.6f}; relative-PPL Δ={relative_delta:.6f} pp"
    )
    print(f"[replication] report: {output}")


if __name__ == "__main__":
    main()
