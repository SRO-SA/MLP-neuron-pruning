#!/usr/bin/env python3
"""Apply the predeclared hybrid stop/go rule and select the target-6 checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ELLIPSOID_LABEL = "rmsnorm_alloc__ellipsoid_rank__p95__target6"
DOWN_LABEL = "rmsnorm_alloc__downnorm_rank__p95__target6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--downstream-table", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--systems-manifest", required=True)
    parser.add_argument("--meaningful-certificate-improvement-pct", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frontier = json.loads(Path(args.frontier).read_text(encoding="utf-8"))["plans"]
    cert_by_plan = {row["plan"]: row for row in frontier}
    downstream = list(csv.DictReader(open(args.downstream_table, newline="", encoding="utf-8")))
    macro = {row["label"]: float(row["accuracy"]) for row in downstream if row["task"] == "macro_average"}
    if ELLIPSOID_LABEL not in macro or DOWN_LABEL not in macro:
        raise ValueError("downstream table lacks pure target-6 endpoints")
    specs = json.loads(Path(args.checkpoint_manifest).read_text(encoding="utf-8"))
    spec_by_label = {row["label"]: row for row in specs}
    label_by_selection = {
        row["selection_sha256"]: row["label"]
        for row in specs if row.get("selection_sha256")
    }
    label_by_selection[cert_by_plan["ellipsoid_slack0"]["selection_sha256"]] = ELLIPSOID_LABEL
    label_by_selection[cert_by_plan["pure_down_norm"]["selection_sha256"]] = DOWN_LABEL
    pure_down_cert = float(cert_by_plan["pure_down_norm"]["strict_certificate"])
    rows = []
    successes = []
    for row in frontier:
        name = row["plan"]
        if name in {"ellipsoid_slack0", "pure_down_norm"}:
            continue
        label = f"certified_hybrid__{name}__target6"
        if label not in macro:
            label = label_by_selection.get(row["selection_sha256"], "")
        if not label or label not in macro:
            # A dominated plan is intentionally not exported/evaluated.
            continue
        accuracy = macro[label]
        cert = float(row["strict_certificate"])
        certificate_improvement_pct = 100.0 * (1.0 - cert / pure_down_cert)
        criterion_a = (
            accuracy >= macro[DOWN_LABEL] - 0.002
            and certificate_improvement_pct >= args.meaningful_certificate_improvement_pct
        )
        criterion_b = (
            accuracy - macro[ELLIPSOID_LABEL] >= 0.005
            and float(row["certificate_slack"]) <= 0.25
        )
        item = {
            "plan": name, "checkpoint_label": label,
            "macro_accuracy": accuracy,
            "difference_vs_ellipsoid": accuracy - macro[ELLIPSOID_LABEL],
            "difference_vs_down_norm": accuracy - macro[DOWN_LABEL],
            "strict_certificate": cert,
            "certificate_improvement_vs_down_norm_pct": certificate_improvement_pct,
            "criterion_a": criterion_a, "criterion_b": criterion_b,
            "successful": criterion_a or criterion_b,
        }
        rows.append(item)
        if item["successful"]:
            successes.append(item)
    if successes:
        outcome = "The certified hybrid successfully recovers practical accuracy."
        chosen = sorted(successes, key=lambda row: (-row["macro_accuracy"], row["strict_certificate"], row["plan"]))[0]
        selected_label = chosen["checkpoint_label"]
    else:
        partial = any(
            min(macro[ELLIPSOID_LABEL], macro[DOWN_LABEL]) < row["macro_accuracy"]
            < max(macro[ELLIPSOID_LABEL], macro[DOWN_LABEL])
            for row in rows
        )
        outcome = (
            "The hybrid shows a partial certificate–accuracy trade-off."
            if partial else
            "The hybrid fails, and the project proceeds as a diagnostic/certification paper."
        )
        selected_label = DOWN_LABEL if macro[DOWN_LABEL] > macro[ELLIPSOID_LABEL] else ELLIPSOID_LABEL
    if selected_label not in spec_by_label:
        raise ValueError(f"selected checkpoint absent from manifest: {selected_label}")
    baseline = spec_by_label["baseline_unpruned"]
    decision = {
        "schema_version": 1,
        "outcome_statement": outcome,
        "stop_method_development": not bool(successes),
        "proceed_to_full_propagation": False,
        "thresholds": {
            "within_down_norm_accuracy_points": 0.2,
            "accuracy_recovery_over_ellipsoid_points": 0.5,
            "maximum_certificate_relaxation_pct": 25.0,
            "operational_meaningful_certificate_improvement_pct": args.meaningful_certificate_improvement_pct,
        },
        "pure_ellipsoid_macro_accuracy": macro[ELLIPSOID_LABEL],
        "pure_down_norm_macro_accuracy": macro[DOWN_LABEL],
        "constrained_plans": rows,
        "selected_target6_checkpoint_label": selected_label,
        "selected_target6_checkpoint_dir": spec_by_label[selected_label]["checkpoint_dir"],
    }
    systems_selected = dict(spec_by_label[selected_label])
    systems_selected.pop("downstream_comparator_only", None)
    systems_selected.pop("additional_operating_point", None)
    systems_selected["systems_selected"] = True
    systems_specs = [baseline, systems_selected]
    if args.dry_run:
        print(json.dumps(decision, indent=2))
        print(f"[hybrid-decision] DRY RUN selected={selected_label}")
        return
    output = Path(args.output); systems_output = Path(args.systems_manifest)
    if output.exists() or systems_output.exists():
        raise FileExistsError("decision or systems manifest already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    systems_output.write_text(json.dumps(systems_specs, indent=2), encoding="utf-8")
    print(f"[hybrid-decision] OK selected={selected_label} outcome={outcome}")


if __name__ == "__main__":
    main()
