#!/usr/bin/env python3
"""Select the frozen baseline/4/6/8 checkpoint plans from milestone summary."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment_provenance import file_sha256


def select_checkpoint_specs(
    rows: list[dict], model: str, *, include_target6_comparators: bool = False
) -> list[dict]:
    specs = [{
        "label": "baseline_unpruned", "model": model, "plan_path": "",
        "target_pct": 0.0, "actual_pct": 0.0, "removed_layer_channels": 0,
        "removed_expert_neurons": 0, "allocation_source": "none",
        "ranking_source": "none", "expert_aggregation": "none",
        "plan_sha256": "",
    }]
    for target in (4, 6, 8):
        matches = []
        for row in rows:
            primary_source = (
                row["source_group"] == "frozen_2_4_6"
                if target in (4, 6)
                else "target8_rmsnorm_primary" in row["source_group"]
            )
            if (
                int(round(float(row["requested_pct"]))) == target
                and row["allocation_source"] == "rmsnorm_bound"
                and row["ranking_source"] == "rmsnorm_ellipsoid_bound"
                and row["expert_aggregation"] == "p95"
                and primary_source
            ):
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(
                f"target {target} primary checkpoint selection has {len(matches)} rows"
            )
        row = matches[0]
        plan_path = row["pruning_plan_path"]
        if not os.path.isfile(plan_path):
            raise FileNotFoundError(f"target {target} plan missing: {plan_path}")
        digest = file_sha256(plan_path)
        if row.get("pruning_plan_sha256") and row["pruning_plan_sha256"] != digest:
            raise ValueError(f"target {target} plan hash changed: {plan_path}")
        specs.append({
            "label": f"rmsnorm_alloc__ellipsoid_rank__p95__target{target}",
            "model": model, "plan_path": plan_path,
            "target_pct": float(row["requested_pct"]),
            "actual_pct": float(row["actual_pct"]),
            "removed_layer_channels": int(float(row["removed_layer_channels"])),
            "removed_expert_neurons": int(float(row["removed_expert_neurons"])),
            "allocation_source": row["allocation_source"],
            "ranking_source": row["ranking_source"],
            "expert_aggregation": row["expert_aggregation"],
            "plan_sha256": digest,
        })
    if include_target6_comparators:
        for ranking in ("rmsnorm_bound", "activation_score"):
            matches = [row for row in rows if (
                int(round(float(row["requested_pct"]))) == 6
                and row["allocation_source"] == "rmsnorm_bound"
                and row["ranking_source"] == ranking
                and row["expert_aggregation"] == "p95"
                and row["source_group"] == "frozen_2_4_6"
            )]
            if len(matches) != 1:
                raise ValueError(
                    f"target 6 comparator {ranking} has {len(matches)} rows"
                )
            row = matches[0]
            plan_path = row["pruning_plan_path"]
            digest = file_sha256(plan_path)
            if row.get("pruning_plan_sha256") and row["pruning_plan_sha256"] != digest:
                raise ValueError(f"target 6 comparator plan hash changed: {plan_path}")
            specs.append({
                "label": f"rmsnorm_alloc__{ranking}_rank__p95__target6",
                "model": model, "plan_path": plan_path,
                "target_pct": float(row["requested_pct"]),
                "actual_pct": float(row["actual_pct"]),
                "removed_layer_channels": int(float(row["removed_layer_channels"])),
                "removed_expert_neurons": int(float(row["removed_expert_neurons"])),
                "allocation_source": row["allocation_source"],
                "ranking_source": row["ranking_source"],
                "expert_aggregation": row["expert_aggregation"],
                "plan_sha256": digest,
                "downstream_comparator_only": True,
            })
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--include-target6-comparators", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    with open(args.summary_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    specs = select_checkpoint_specs(
        rows, args.model,
        include_target6_comparators=args.include_target6_comparators,
    )
    for spec in specs:
        spec["checkpoint_dir"] = os.path.join(args.checkpoint_root, spec["label"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(specs, handle, indent=2)
    print(f"[checkpoint-manifest] OK: {len(specs)} checkpoints; {args.output}")


if __name__ == "__main__":
    main()
