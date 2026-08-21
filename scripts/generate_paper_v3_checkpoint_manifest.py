#!/usr/bin/env python3
"""Select the frozen baseline/4/6/8 checkpoint plans from milestone summary."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment_provenance import file_sha256


def _resolve_additional_derived_plan(run_dir: str, rows: list[dict]) -> str:
    """Resolve the validated derived plan even if the summary omitted its path."""
    recorded = {
        row.get("pruning_plan_path", "")
        for row in rows
        if row.get("pruning_plan_path", "")
    }
    existing_recorded = {path for path in recorded if os.path.isfile(path)}
    if len(existing_recorded) > 1:
        raise ValueError(
            f"target-6 down-norm rows use different existing plans: "
            f"{existing_recorded}"
        )
    if len(existing_recorded) == 1:
        plan_path = next(iter(existing_recorded))
    else:
        matches = glob.glob(os.path.join(
            run_dir,
            "rmsnorm_alloc__downnorm_rank",
            "pruning_plans",
            "*.json",
        ))
        if len(matches) != 1:
            raise FileNotFoundError(
                "expected exactly one derived target-6 down-norm plan under "
                f"{run_dir}; found {matches} (summary paths={recorded})"
            )
        plan_path = matches[0]

    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    audit = plan.get("allocation_ranking")
    if not audit:
        raise ValueError(f"derived plan lacks allocation_ranking audit: {plan_path}")
    expected = {
        "experiment_name": "rmsnorm_alloc__downnorm_rank",
        "allocation_source": "rmsnorm_bound",
        "ranking_source": "down_norm",
        "ranking_aggregation_mode": "p95",
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(
                f"derived plan {field}={audit.get(field)!r}; expected {value!r}: "
                f"{plan_path}"
            )
    return plan_path


def select_additional_target6_downnorm_spec(run_dir: str, model: str) -> dict:
    path = os.path.join(run_dir, "allocation_ranking_summary.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if (
            int(round(float(row["requested_pct"]))) == 6
            and row["allocation_source"] == "rmsnorm_bound"
            and row["ranking_source"] == "down_norm"
            and row["ranking_aggregation"] == "p95"
        )]
    if not rows:
        raise ValueError(f"no target-6 RMSNorm-allocation/down-norm-ranking row in {path}")
    for field in ("actual_pct", "layer_channels", "expert_neurons"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"target-6 down-norm rows differ in {field}")
    plan_path = _resolve_additional_derived_plan(run_dir, rows)
    digest = file_sha256(plan_path)
    row = rows[0]
    return {
        "label": "rmsnorm_alloc__downnorm_rank__p95__target6",
        "model": model, "plan_path": plan_path,
        "target_pct": float(row["requested_pct"]),
        "actual_pct": float(row["actual_pct"]),
        "removed_layer_channels": int(float(row["layer_channels"])),
        "removed_expert_neurons": int(float(row["expert_neurons"])),
        "allocation_source": "rmsnorm_bound", "ranking_source": "down_norm",
        "expert_aggregation": "p95", "plan_sha256": digest,
        "downstream_comparator_only": True,
        "source_run_dir": run_dir,
    }


def select_checkpoint_specs(
    rows: list[dict], model: str, *, include_target6_comparators: bool = False,
    include_target6_downnorm_if_available: bool = False,
    include_target2_primary: bool = False,
) -> list[dict]:
    specs = [{
        "label": "baseline_unpruned", "model": model, "plan_path": "",
        "target_pct": 0.0, "actual_pct": 0.0, "removed_layer_channels": 0,
        "removed_expert_neurons": 0, "allocation_source": "none",
        "ranking_source": "none", "expert_aggregation": "none",
        "plan_sha256": "",
    }]
    targets = ((2,) if include_target2_primary else ()) + (4, 6, 8)
    for target in targets:
        matches = []
        for row in rows:
            primary_source = (
                row["source_group"] == "frozen_2_4_6"
                if target in (2, 4, 6)
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
        spec = {
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
        }
        if target == 2:
            spec["additional_operating_point"] = True
        specs.append(spec)
    if include_target6_comparators:
        rankings = ["rmsnorm_bound", "activation_score"]
        if include_target6_downnorm_if_available:
            rankings.append("down_norm")
        for ranking in rankings:
            matches = [row for row in rows if (
                int(round(float(row["requested_pct"]))) == 6
                and row["allocation_source"] == "rmsnorm_bound"
                and row["ranking_source"] == ranking
                and row["expert_aggregation"] == "p95"
                and row["source_group"] == "frozen_2_4_6"
            )]
            if ranking == "down_norm" and not matches:
                continue
            if len(matches) != 1:
                raise ValueError(
                    f"target 6 comparator {ranking} has {len(matches)} rows"
                )
            row = matches[0]
            plan_path = row["pruning_plan_path"]
            digest = file_sha256(plan_path)
            if row.get("pruning_plan_sha256") and row["pruning_plan_sha256"] != digest:
                raise ValueError(f"target 6 comparator plan hash changed: {plan_path}")
            ranking_label = "downnorm" if ranking == "down_norm" else ranking
            specs.append({
                "label": f"rmsnorm_alloc__{ranking_label}_rank__p95__target6",
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
    parser.add_argument("--include-target6-downnorm-if-available",
                        action="store_true")
    parser.add_argument("--include-target2-primary", action="store_true")
    parser.add_argument("--additional-target6-downnorm-run-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    with open(args.summary_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    specs = select_checkpoint_specs(
        rows, args.model,
        include_target6_comparators=args.include_target6_comparators,
        include_target6_downnorm_if_available=(
            args.include_target6_downnorm_if_available
        ),
        include_target2_primary=args.include_target2_primary,
    )
    if args.additional_target6_downnorm_run_dir:
        if any(spec.get("ranking_source") == "down_norm" and
               int(round(float(spec["target_pct"]))) == 6 for spec in specs):
            raise ValueError("target-6 down-norm comparator was selected twice")
        specs.append(select_additional_target6_downnorm_spec(
            args.additional_target6_downnorm_run_dir, args.model
        ))
    for spec in specs:
        spec["checkpoint_dir"] = os.path.join(args.checkpoint_root, spec["label"])
    if args.dry_run:
        for spec in specs:
            print(
                f"[checkpoint-manifest] WOULD INCLUDE {spec['label']}: "
                f"{spec['checkpoint_dir']}"
            )
        print(f"[checkpoint-manifest] DRY RUN: {len(specs)} checkpoints")
        return
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(specs, handle, indent=2)
    print(f"[checkpoint-manifest] OK: {len(specs)} checkpoints; {args.output}")


if __name__ == "__main__":
    main()
