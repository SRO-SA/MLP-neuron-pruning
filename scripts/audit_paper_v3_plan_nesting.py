#!/usr/bin/env python3
"""Audit whether independently optimized primary pruning plans are nested."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_by_layer(plan: dict) -> dict[int, set[int]]:
    selected = {}
    for row in plan.get("layers", []):
        layer = int(row["layer_idx"])
        if layer in selected:
            raise ValueError(f"duplicate layer {layer}")
        indices = {int(value) for value in row.get("prune_idx", [])}
        declared = int(row.get("pruned_channels", len(indices)))
        if declared != len(indices):
            raise ValueError(
                f"layer {layer}: declared {declared}, unique indices {len(indices)}"
            )
        selected[layer] = indices
    if not selected:
        raise ValueError("plan has no layer selections")
    counted = sum(map(len, selected.values()))
    declared_total = int(plan.get("total_selected_layer_channels", counted))
    if counted != declared_total:
        raise ValueError(
            f"plan total mismatch: declared {declared_total}, counted {counted}"
        )
    return selected


def audit_pair(
    lower_target: int, lower: dict[int, set[int]],
    upper_target: int, upper: dict[int, set[int]],
) -> tuple[dict, list[dict]]:
    if set(lower) != set(upper):
        raise ValueError("plan layer sets differ")
    per_layer = []
    for layer in sorted(lower):
        retained_from_lower = lower[layer] & upper[layer]
        lost = lower[layer] - upper[layer]
        added = upper[layer] - lower[layer]
        union = lower[layer] | upper[layer]
        per_layer.append({
            "lower_target_pct": lower_target, "upper_target_pct": upper_target,
            "layer_idx": layer,
            "lower_count": len(lower[layer]), "upper_count": len(upper[layer]),
            "intersection_count": len(retained_from_lower),
            "lower_not_in_upper_count": len(lost),
            "upper_not_in_lower_count": len(added),
            "jaccard": len(retained_from_lower) / len(union) if union else 1.0,
            "nested_in_layer": not lost,
            "lower_not_in_upper_indices": sorted(lost),
            "upper_not_in_lower_indices": sorted(added),
        })
    lower_global = {(layer, channel) for layer, values in lower.items()
                    for channel in values}
    upper_global = {(layer, channel) for layer, values in upper.items()
                    for channel in values}
    intersection = lower_global & upper_global
    union = lower_global | upper_global
    summary = {
        "lower_target_pct": lower_target, "upper_target_pct": upper_target,
        "lower_total_layer_channels": len(lower_global),
        "upper_total_layer_channels": len(upper_global),
        "intersection_count": len(intersection),
        "lower_not_in_upper_count": len(lower_global - upper_global),
        "upper_not_in_lower_count": len(upper_global - lower_global),
        "global_jaccard": len(intersection) / len(union) if union else 1.0,
        "fully_nested": lower_global <= upper_global,
        "layers_with_nesting_violations": sum(
            not row["nested_in_layer"] for row in per_layer
        ),
    }
    return summary, per_layer


def audit_manifest(specs: list[dict]) -> dict:
    plans = {}
    provenance = {}
    for spec in specs:
        if (spec.get("allocation_source") != "rmsnorm_bound"
                or spec.get("ranking_source") != "rmsnorm_ellipsoid_bound"
                or spec.get("expert_aggregation") != "p95"):
            continue
        target = int(round(float(spec["target_pct"])))
        if target not in (2, 4, 6, 8):
            continue
        path = spec.get("plan_path", "")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if target in plans:
            raise ValueError(f"multiple primary plans for target {target}")
        plans[target] = selected_by_layer(payload)
        digest = _sha256(path)
        if spec.get("plan_sha256") and spec["plan_sha256"] != digest:
            raise ValueError(f"plan hash mismatch: {path}")
        provenance[target] = {
            "path": path, "sha256": digest,
            "total_layer_channels": sum(map(len, plans[target].values())),
        }
    if not {4, 6, 8} <= set(plans):
        raise ValueError(f"primary target 4/6/8 plans required; found {sorted(plans)}")
    targets = sorted(plans)
    pair_summaries, per_layer = [], []
    for lower, upper in zip(targets, targets[1:]):
        summary, layer_rows = audit_pair(
            lower, plans[lower], upper, plans[upper]
        )
        pair_summaries.append(summary); per_layer.extend(layer_rows)
    return {
        "schema_version": 1,
        "method": {
            "allocation_source": "rmsnorm_bound",
            "ranking_source": "rmsnorm_ellipsoid_bound",
            "expert_aggregation": "p95",
        },
        "plan_provenance": {str(key): value for key, value in provenance.items()},
        "pairwise_nesting": pair_summaries,
        "per_layer": per_layer,
        "interpretation": (
            "Each target was globally allocated and ranked independently under "
            "alignment and per-layer caps. Non-nesting is therefore possible: a "
            "channel removed at a smaller target can be retained at a larger target. "
            "Consequently, non-monotonic downstream task scores need not imply an "
            "evaluation error or a monotone sequence of model perturbations."
        ),
    }


def _csv_safe(row: dict) -> dict:
    return {key: (json.dumps(value) if isinstance(value, list) else value)
            for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.checkpoint_manifest, encoding="utf-8") as handle:
        specs = json.load(handle)
    audit = audit_manifest(specs)
    if args.dry_run:
        print(
            f"[plan-nesting] DRY RUN: targets="
            f"{sorted(map(int, audit['plan_provenance']))} output={args.output_dir}"
        )
        return
    os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "plan_nesting_audit.json"), "x",
              encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    for name, rows in (("plan_nesting_pairs.csv", audit["pairwise_nesting"]),
                       ("plan_nesting_per_layer.csv", audit["per_layer"])):
        with open(os.path.join(args.output_dir, name), "x", newline="",
                  encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(_csv_safe(row) for row in rows)
    with open(os.path.join(args.output_dir, "plan_nesting_audit.md"), "x",
              encoding="utf-8") as handle:
        handle.write("| Lower | Upper | Lower channels | Upper channels | Retained | Lost | Jaccard | Fully nested |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in audit["pairwise_nesting"]:
            handle.write(
                f"| {row['lower_target_pct']} | {row['upper_target_pct']} | "
                f"{row['lower_total_layer_channels']} | {row['upper_total_layer_channels']} | "
                f"{row['intersection_count']} | {row['lower_not_in_upper_count']} | "
                f"{row['global_jaccard']:.4f} | {row['fully_nested']} |\n"
            )
        handle.write("\n" + audit["interpretation"] + "\n")
    print(
        f"[plan-nesting] OK: pairs={len(audit['pairwise_nesting'])} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
