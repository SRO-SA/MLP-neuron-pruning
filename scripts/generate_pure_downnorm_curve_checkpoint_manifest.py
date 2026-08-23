#!/usr/bin/env python3
"""Build an additive checkpoint manifest for the matched 2/4/6/8 down-norm curve."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_provenance import file_sha256
from src.moe_set_certification import selected_by_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-run-dir", required=True)
    parser.add_argument("--existing-checkpoint-manifest", required=True)
    parser.add_argument("--new-checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    curve_root = Path(args.curve_run_dir)
    summary_path = curve_root / "allocation_ranking_summary.csv"
    rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
    existing = json.loads(Path(args.existing_checkpoint_manifest).read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in existing}
    # Include the frozen target-6 ellipsoid reference so the existing paired
    # selector-attribution summary remains well-defined.  The run command uses
    # additional_only, so this checkpoint is not evaluated again.
    specs = [
        dict(by_label["baseline_unpruned"]),
        dict(by_label["rmsnorm_alloc__ellipsoid_rank__p95__target6"]),
    ]
    budget_audit = []
    for target in (2, 4, 6, 8):
        target_rows = [row for row in rows if (
            int(round(float(row["requested_pct"]))) == target
            and row["allocation_source"] == "rmsnorm_bound"
            and row["ranking_source"] == "down_norm"
            and row["ranking_aggregation"] == "p95"
        )]
        if not target_rows:
            raise ValueError(f"curve lacks target {target}")
        totals = {int(float(row["layer_channels"])) for row in target_rows}
        neurons = {int(float(row["expert_neurons"])) for row in target_rows}
        if len(totals) != 1 or len(neurons) != 1:
            raise ValueError(f"target {target}: dataset rows disagree on budget")
        channels = totals.pop(); expert_neurons = neurons.pop()
        ellipsoid_label = f"rmsnorm_alloc__ellipsoid_rank__p95__target{target}"
        ellipsoid = by_label[ellipsoid_label]
        difference = channels - int(ellipsoid["removed_layer_channels"])
        budget_audit.append({
            "target": target, "down_norm_layer_channels": channels,
            "ellipsoid_layer_channels": int(ellipsoid["removed_layer_channels"]),
            "aligned_budget_difference": difference,
        })
        experiment = f"rmsnorm_alloc__downnorm_rank__target{target}"
        plan_matches = glob.glob(str(curve_root / experiment / "pruning_plans" / "*.json"))
        if len(plan_matches) != 1:
            raise FileNotFoundError(f"target {target}: derived plan matches={plan_matches}")
        plan_path = plan_matches[0]
        label = f"rmsnorm_alloc__downnorm_rank__p95__target{target}"
        if target == 6 and label in by_label:
            spec = dict(by_label[label])
            if int(spec["removed_layer_channels"]) != channels:
                raise ValueError("existing target-6 down-norm checkpoint budget differs")
            existing_plan = json.loads(Path(spec["plan_path"]).read_text(encoding="utf-8"))
            curve_plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            if selected_by_layer(existing_plan) != selected_by_layer(curve_plan):
                raise ValueError(
                    "new target-6 down-norm plan identities differ from frozen checkpoint"
                )
        else:
            spec = {
                "label": label, "model": "Qwen/Qwen3-30B-A3B",
                "plan_path": plan_path, "target_pct": float(target),
                "actual_pct": float(target_rows[0]["actual_pct"]),
                "removed_layer_channels": channels,
                "removed_expert_neurons": expert_neurons,
                "allocation_source": "rmsnorm_bound",
                "ranking_source": "down_norm", "expert_aggregation": "p95",
                "plan_sha256": file_sha256(plan_path),
                "checkpoint_dir": str(Path(args.new_checkpoint_root) / label),
            }
        spec["additional_operating_point"] = True
        specs.append(spec)
    payload = {"checkpoints": specs, "matched_budget_audit": budget_audit}
    output = Path(args.output)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print(f"[downnorm-curve-manifest] DRY RUN checkpoints={len(specs)}")
        return
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Export/downstream runners consume a list; preserve the audit beside it.
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    audit_path = output.with_name(output.stem + "_budget_audit.json")
    audit_path.write_text(json.dumps(budget_audit, indent=2), encoding="utf-8")
    print(f"[downnorm-curve-manifest] OK checkpoints={len(specs)} output={output}")


if __name__ == "__main__":
    main()
