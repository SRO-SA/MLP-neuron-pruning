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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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
        }
        specs.append(spec)
    output = Path(args.output)
    if args.dry_run:
        for spec in specs:
            print(f"[hybrid-checkpoint-manifest] WOULD INCLUDE {spec['label']}: {spec['checkpoint_dir']}")
        print(f"[hybrid-checkpoint-manifest] DRY RUN checkpoints={len(specs)} reuse={reuse_rows}")
        return
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    reuse_path = output.with_name(output.stem + "_selection_reuse.json")
    reuse_path.write_text(json.dumps(reuse_rows, indent=2), encoding="utf-8")
    print(f"[hybrid-checkpoint-manifest] OK checkpoints={len(specs)} reuse={len(reuse_rows)} output={output}")


if __name__ == "__main__":
    main()
