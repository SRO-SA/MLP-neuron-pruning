#!/usr/bin/env python3
"""Create an additive checkpoint manifest for distinct Pareto hybrid plans."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_provenance import file_sha256
from src.moe_set_certification import selected_by_layer


def load_plan(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-manifest", required=True)
    parser.add_argument("--existing-checkpoint-manifest", required=True)
    parser.add_argument("--new-checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-intermediate-checkpoints", type=int, default=2)
    parser.add_argument(
        "--minimum-objective-gap-closure", type=float, default=0.10,
        help=(
            "Predeclared minimum fraction of the ellipsoid-to-down-norm "
            "objective gap closed by an intermediate candidate."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_intermediate_checkpoints < 0 or args.max_intermediate_checkpoints > 2:
        raise ValueError("max-intermediate-checkpoints must be between 0 and 2")
    if not 0.0 <= args.minimum_objective_gap_closure <= 1.0:
        raise ValueError("minimum-objective-gap-closure must be in [0, 1]")
    frontier = json.loads(Path(args.frontier_manifest).read_text(encoding="utf-8"))
    existing = json.loads(Path(args.existing_checkpoint_manifest).read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in existing}
    required = [
        "baseline_unpruned",
        "rmsnorm_alloc__ellipsoid_rank__p95__target6",
        "rmsnorm_alloc__downnorm_rank__p95__target6",
    ]
    for label in required:
        if label not in by_label:
            raise ValueError(f"existing checkpoint manifest lacks {label}")
    specs = [dict(by_label[label]) for label in required]
    endpoint_labels = {
        "ellipsoid_slack0": required[1],
        "pure_down_norm": required[2],
    }
    reference_plans = {
        name: load_plan(by_label[label]["plan_path"])
        for name, label in endpoint_labels.items()
    }
    endpoint_signature_to_label = {
        json.dumps(
            {str(k): list(v) for k, v in selected_by_layer(plan).items()},
            sort_keys=True,
        ): endpoint_labels[name]
        for name, plan in reference_plans.items()
    }
    frontier_by_name = {row["plan"]: row for row in frontier["plans"]}
    ellipsoid_row = frontier_by_name["ellipsoid_slack0"]
    down_row = frontier_by_name["pure_down_norm"]
    ellipsoid_objective = float(ellipsoid_row["normalized_down_norm_objective"])
    down_objective = float(down_row["normalized_down_norm_objective"])
    objective_gap = ellipsoid_objective - down_objective
    if objective_gap <= 0:
        raise ValueError("pure down-norm does not improve the recorded objective")
    down_certificate = float(down_row["strict_certificate"])
    eligible_by_selection = {}
    for row in frontier["plans"]:
        if row["plan"] in endpoint_labels:
            continue
        gap_closure = (
            ellipsoid_objective - float(row["normalized_down_norm_objective"])
        ) / objective_gap
        eligible = (
            row.get("certificate_objective_pareto_optimal") is True
            and float(row["strict_certificate"]) < down_certificate
            and gap_closure >= args.minimum_objective_gap_closure
        )
        if not eligible:
            continue
        previous = eligible_by_selection.get(row["selection_sha256"])
        if previous is None or (
            float(row["normalized_down_norm_objective"]),
            float(row["strict_certificate"]),
            float(row["certificate_slack"]),
            row["plan"],
        ) < (
            float(previous["normalized_down_norm_objective"]),
            float(previous["strict_certificate"]),
            float(previous["certificate_slack"]),
            previous["plan"],
        ):
            eligible_by_selection[row["selection_sha256"]] = {
                **row, "objective_gap_closure": gap_closure,
            }
    eligible_rows = list(eligible_by_selection.values())
    selected_intermediates = []
    if eligible_rows and args.max_intermediate_checkpoints:
        # Cover both ends of the eligible certificate/objective frontier without
        # consulting PPL or downstream results: first retain the strongest
        # certificate, then add the best down-norm objective if it is distinct.
        strongest_certificate = min(
            eligible_rows,
            key=lambda row: (
                float(row["strict_certificate"]),
                float(row["normalized_down_norm_objective"]),
                float(row["certificate_slack"]),
                row["plan"],
            ),
        )
        selected_intermediates.append(strongest_certificate)
        if args.max_intermediate_checkpoints > 1:
            remaining = [
                row for row in eligible_rows
                if row["selection_sha256"]
                != strongest_certificate["selection_sha256"]
            ]
            if remaining:
                selected_intermediates.append(min(
                    remaining,
                    key=lambda row: (
                        float(row["normalized_down_norm_objective"]),
                        float(row["strict_certificate"]),
                        float(row["certificate_slack"]),
                        row["plan"],
                    ),
                ))
    selected_intermediate_names = {row["plan"] for row in selected_intermediates}
    seen_selections = set(endpoint_signature_to_label)
    reuse_rows = []
    for row in frontier["plans"]:
        if not row.get("certificate_objective_pareto_optimal"):
            continue
        name = row["plan"]
        plan = load_plan(row["plan_path"])
        signature = json.dumps(
            {str(k): list(v) for k, v in selected_by_layer(plan).items()},
            sort_keys=True,
        )
        if signature in endpoint_signature_to_label:
            reuse_rows.append({
                "frontier_plan": name,
                "reused_checkpoint_label": endpoint_signature_to_label[signature],
                "reason": "identical selected channel identities",
            })
            if name in endpoint_labels and selected_by_layer(plan) != selected_by_layer(reference_plans[name]):
                raise ValueError(f"{name}: endpoint identities differ from frozen checkpoint")
            continue
        if signature in seen_selections:
            reuse_rows.append({
                "frontier_plan": name,
                "reused_checkpoint_label": "previous identical frontier selection",
                "reason": "identical selected channel identities",
            })
            continue
        if name not in selected_intermediate_names:
            reuse_rows.append({
                "frontier_plan": name,
                "reused_checkpoint_label": "",
                "reason": (
                    "not selected by bounded downstream gate: requires Pareto, "
                    "certificate strictly below pure down-norm, material objective "
                    "improvement, and the predeclared strongest-certificate / "
                    "best-objective coverage rule"
                ),
            })
            continue
        seen_selections.add(signature)
        label = f"certified_hybrid__{name}__target6"
        plan_path = str(row["plan_path"])
        spec = {
            "label": label,
            "model": "Qwen/Qwen3-30B-A3B",
            "plan_path": plan_path,
            "target_pct": 6.0,
            "actual_pct": float(plan.get("actual_pct", 6.2066)),
            "removed_layer_channels": 2288,
            "removed_expert_neurons": 292864,
            "allocation_source": "rmsnorm_bound",
            "ranking_source": "certified_hybrid_down_norm",
            "expert_aggregation": "p95",
            "plan_sha256": file_sha256(plan_path),
            "checkpoint_dir": str(Path(args.new_checkpoint_root) / label),
            "additional_operating_point": True,
            "certificate_slack": row["certificate_slack"],
            "selection_sha256": row["selection_sha256"],
            "paired_reference_labels": [required[1], required[2]],
            "objective_gap_closure": next(
                candidate["objective_gap_closure"]
                for candidate in selected_intermediates
                if candidate["plan"] == name
            ),
        }
        specs.append(spec)
    output = Path(args.output)
    if args.dry_run:
        for spec in specs:
            print(f"[hybrid-checkpoint-manifest] WOULD INCLUDE {spec['label']}: {spec['checkpoint_dir']}")
        print(f"[hybrid-checkpoint-manifest] DRY RUN checkpoints={len(specs)} reuse={reuse_rows}")
        print(
            "[hybrid-checkpoint-manifest] bounded intermediates="
            f"{sorted(selected_intermediate_names)}"
        )
        return
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    reuse_path = output.with_name(output.stem + "_selection_reuse.json")
    reuse_path.write_text(json.dumps(reuse_rows, indent=2), encoding="utf-8")
    selection_path = output.with_name(output.stem + "_downstream_selection.json")
    selection_path.write_text(json.dumps({
        "schema_version": 1,
        "maximum_intermediate_checkpoints": args.max_intermediate_checkpoints,
        "minimum_objective_gap_closure": args.minimum_objective_gap_closure,
        "selection_policy": (
            "one eligible strongest-certificate candidate plus one distinct "
            "eligible best-down-norm-objective candidate; no PPL or downstream "
            "outcomes used"
        ),
        "pure_down_norm_strict_certificate": down_certificate,
        "selected_intermediates": selected_intermediates,
        "selection_uses_downstream_results": False,
    }, indent=2), encoding="utf-8")
    print(f"[hybrid-checkpoint-manifest] OK checkpoints={len(specs)} reuse={len(reuse_rows)} output={output}")


if __name__ == "__main__":
    main()
