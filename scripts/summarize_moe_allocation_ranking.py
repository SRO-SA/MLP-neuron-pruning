#!/usr/bin/env python3
"""Build and print a compact allocation/ranking matrix summary."""
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
from src.paired_bootstrap import paired_bootstrap_nll_difference


FIELDS = [
    "experiment_name", "allocation_source", "ranking_source", "dataset",
    "allocation_aggregation", "ranking_aggregation", "exact_total_layer_channels",
    "requested_pct", "actual_pct", "layer_channels", "expert_neurons",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "baseline_ppl", "pruned_ppl",
    "relative_ppl_change_pct", "baseline_tokens", "pruned_tokens",
    "token_count_match", "process_id", "model_load_instance_id",
    "pruning_plan_path", "per_layer_csv_path", "score_comparison_json_path",
    "mean_nll_difference", "mean_nll_ci95_lower", "mean_nll_ci95_upper",
    "per_example_nll_path", "bound_tightness_json_path",
    "ellipsoid_tightness_median", "ellipsoid_tightness_p95",
    "ellipsoid_tightness_p99", "ellipsoid_tightness_max",
    "ellipsoid_pruned_tightness_median", "ellipsoid_pruned_tightness_p95",
    "ellipsoid_pruned_tightness_p99", "ellipsoid_pruned_tightness_max",
    "ellipsoid_bound_violations",
    "sphere_tightness_median", "sphere_tightness_p95",
    "sphere_tightness_p99", "sphere_tightness_max",
    "sphere_pruned_tightness_median", "sphere_pruned_tightness_p95",
    "sphere_pruned_tightness_p99", "sphere_pruned_tightness_max",
    "sphere_bound_violations",
    "sampled_routed_inputs", "expert_channel_pairs_evaluated",
    "routed_input_channel_contributions_evaluated",
    "sphere_to_ellipsoid_bound_median", "sphere_to_ellipsoid_bound_p95",
    "sphere_to_ellipsoid_bound_max", "absolute_tolerance", "relative_tolerance",
    "ellipsoid_max_violation_magnitude", "sphere_max_violation_magnitude",
]


def _main_csv(experiment_dir: str) -> str:
    matches = [
        path for path in glob.glob(os.path.join(experiment_dir, "moe_target_pruning_*.csv"))
        if not path.endswith("_per_layer.csv")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one result CSV in {experiment_dir}; found {matches}"
        )
    return matches[0]


def _bound_fields(path: str) -> dict:
    empty = {
        "ellipsoid_tightness_median": "",
        "ellipsoid_tightness_p95": "",
        "ellipsoid_tightness_p99": "",
        "ellipsoid_tightness_max": "",
        "ellipsoid_pruned_tightness_median": "",
        "ellipsoid_pruned_tightness_p95": "",
        "ellipsoid_pruned_tightness_p99": "",
        "ellipsoid_pruned_tightness_max": "",
        "ellipsoid_bound_violations": "",
        "sphere_tightness_median": "",
        "sphere_tightness_p95": "",
        "sphere_tightness_p99": "",
        "sphere_tightness_max": "",
        "sphere_pruned_tightness_median": "",
        "sphere_pruned_tightness_p95": "",
        "sphere_pruned_tightness_p99": "",
        "sphere_pruned_tightness_max": "",
        "sphere_bound_violations": "",
        "sampled_routed_inputs": "",
        "expert_channel_pairs_evaluated": "",
        "routed_input_channel_contributions_evaluated": "",
        "sphere_to_ellipsoid_bound_median": "",
        "sphere_to_ellipsoid_bound_p95": "",
        "sphere_to_ellipsoid_bound_max": "",
        "absolute_tolerance": "",
        "relative_tolerance": "",
        "ellipsoid_max_violation_magnitude": "",
        "sphere_max_violation_magnitude": "",
    }
    if not path:
        return empty
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    stats = payload["global"]["ellipsoid_all"]
    pruned_stats = payload["global"].get("ellipsoid_pruned", {})
    sphere = payload["global"]["sphere_all"]
    sphere_pruned = payload["global"].get("sphere_pruned", {})
    bound_ratio = payload["global"].get("sphere_to_ellipsoid_bound", {})
    return {
        "ellipsoid_tightness_median": stats["median"],
        "ellipsoid_tightness_p95": stats["p95"],
        "ellipsoid_tightness_p99": stats["p99"],
        "ellipsoid_tightness_max": stats["max"],
        "ellipsoid_pruned_tightness_median": pruned_stats.get("median", ""),
        "ellipsoid_pruned_tightness_p95": pruned_stats.get("p95", ""),
        "ellipsoid_pruned_tightness_p99": pruned_stats.get("p99", ""),
        "ellipsoid_pruned_tightness_max": pruned_stats.get("max", ""),
        "ellipsoid_bound_violations": payload["global"][
            "ellipsoid_numerical_violations"
        ],
        "sphere_tightness_median": sphere["median"],
        "sphere_tightness_p95": sphere["p95"],
        "sphere_tightness_p99": sphere["p99"],
        "sphere_tightness_max": sphere["max"],
        "sphere_pruned_tightness_median": sphere_pruned.get("median", ""),
        "sphere_pruned_tightness_p95": sphere_pruned.get("p95", ""),
        "sphere_pruned_tightness_p99": sphere_pruned.get("p99", ""),
        "sphere_pruned_tightness_max": sphere_pruned.get("max", ""),
        "sphere_bound_violations": payload["global"][
            "sphere_numerical_violations"
        ],
        "sampled_routed_inputs": payload.get("sampled_routed_inputs", ""),
        "expert_channel_pairs_evaluated": payload.get(
            "expert_channel_pairs_evaluated", ""
        ),
        "routed_input_channel_contributions_evaluated": payload.get(
            "routed_input_channel_contributions_evaluated", ""
        ),
        "sphere_to_ellipsoid_bound_median": bound_ratio.get("median", ""),
        "sphere_to_ellipsoid_bound_p95": bound_ratio.get("p95", ""),
        "sphere_to_ellipsoid_bound_max": bound_ratio.get("max", ""),
        "absolute_tolerance": payload.get("absolute_tolerance", ""),
        "relative_tolerance": payload.get("relative_tolerance", ""),
        "ellipsoid_max_violation_magnitude": payload["global"].get(
            "ellipsoid_max_violation_magnitude", ""
        ),
        "sphere_max_violation_magnitude": payload["global"].get(
            "sphere_max_violation_magnitude", ""
        ),
    }


def _read_paired_rows(path: str) -> list[dict]:
    if not path:
        raise ValueError("per-example NLL path is empty")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"per-example NLL file is empty: {path}")
    return rows


def _validate_paired_documents(
    left: list[dict], right: list[dict], *, label: str
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label} document counts differ")
    for left_row, right_row in zip(left, right):
        for field in ("dataset", "corpus_sha256", "sample_index", "n_tokens"):
            if left_row[field] != right_row[field]:
                raise ValueError(f"{label} field {field} differs")


def build_paired_allocation_comparisons(
    rows: list[dict], *, bootstrap_resamples: int
) -> list[dict]:
    """Compare exact-budget RMSNorm and down-norm allocations document-wise."""
    groups = {}
    for row in rows:
        if not row.get("exact_total_layer_channels"):
            continue
        key = (
            row["dataset"], row["ranking_source"],
            row["ranking_aggregation"], row["layer_channels"],
        )
        groups.setdefault(key, []).append(row)
    comparisons = []
    for key, group in groups.items():
        by_allocation = {row["allocation_source"]: row for row in group}
        if set(by_allocation) != {"rmsnorm_bound", "down_norm"}:
            continue
        rmsnorm = by_allocation["rmsnorm_bound"]
        downnorm = by_allocation["down_norm"]
        if not rmsnorm["per_example_nll_path"] or not downnorm["per_example_nll_path"]:
            continue
        rmsnorm_docs = _read_paired_rows(rmsnorm["per_example_nll_path"])
        downnorm_docs = _read_paired_rows(downnorm["per_example_nll_path"])
        _validate_paired_documents(
            rmsnorm_docs, downnorm_docs, label="paired allocation"
        )
        stats = paired_bootstrap_nll_difference(
            [float(row["pruned_nll_sum"]) for row in rmsnorm_docs],
            [float(row["pruned_nll_sum"]) for row in downnorm_docs],
            [int(row["n_tokens"]) for row in rmsnorm_docs],
            n_resamples=bootstrap_resamples,
            seed=42,
        )
        comparisons.append({
            "dataset": key[0],
            "ranking_source": key[1],
            "aggregation_mode": key[2],
            "exact_removed_layer_channels": key[3],
            "rmsnorm_experiment": rmsnorm["experiment_name"],
            "downnorm_experiment": downnorm["experiment_name"],
            "rmsnorm_minus_downnorm_mean_nll": stats["mean_nll_difference"],
            "ci95_lower": stats["ci_lower"],
            "ci95_upper": stats["ci_upper"],
            "n_documents": stats["n_documents"],
            "n_tokens": stats["n_tokens"],
            "bootstrap_resamples": stats["n_resamples"],
        })
    return comparisons


def build_paired_aggregation_comparisons(
    rows: list[dict], *, bootstrap_resamples: int
) -> list[dict]:
    """Compare max and p95 ellipsoid aggregation on identical documents."""
    groups = {}
    for row in rows:
        if row.get("ranking_source") != "rmsnorm_ellipsoid_bound":
            continue
        key = (
            row["allocation_source"], row["dataset"], row["layer_channels"],
            row.get("exact_total_layer_channels", ""),
        )
        groups.setdefault(key, []).append(row)
    comparisons = []
    for key, group in groups.items():
        by_aggregation = {row["ranking_aggregation"]: row for row in group}
        if set(by_aggregation) != {"p95", "max"}:
            continue
        p95 = by_aggregation["p95"]
        maximum = by_aggregation["max"]
        if not p95["per_example_nll_path"] or not maximum["per_example_nll_path"]:
            continue
        p95_docs = _read_paired_rows(p95["per_example_nll_path"])
        max_docs = _read_paired_rows(maximum["per_example_nll_path"])
        _validate_paired_documents(p95_docs, max_docs, label="paired aggregation")
        stats = paired_bootstrap_nll_difference(
            [float(row["pruned_nll_sum"]) for row in max_docs],
            [float(row["pruned_nll_sum"]) for row in p95_docs],
            [int(row["n_tokens"]) for row in max_docs],
            n_resamples=bootstrap_resamples,
            seed=42,
        )
        comparisons.append({
            "allocation_source": key[0],
            "dataset": key[1],
            "exact_removed_layer_channels": key[2],
            "requested_exact_total_layer_channels": key[3],
            "p95_experiment": p95["experiment_name"],
            "max_experiment": maximum["experiment_name"],
            "p95_relative_ppl_change_pct": p95["relative_ppl_change_pct"],
            "max_relative_ppl_change_pct": maximum["relative_ppl_change_pct"],
            "max_minus_p95_mean_nll": stats["mean_nll_difference"],
            "ci95_lower": stats["ci_lower"],
            "ci95_upper": stats["ci_upper"],
            "n_documents": stats["n_documents"],
            "n_tokens": stats["n_tokens"],
            "bootstrap_resamples": stats["n_resamples"],
        })
    return comparisons


def build_aggregation_selection_comparisons(rows: list[dict]) -> list[dict]:
    """Compare p95/max selected channel identities under a fixed allocation."""
    unique = {}
    for row in rows:
        plan_path = row.get("pruning_plan_path", "")
        if not plan_path:
            continue
        unique[(row["experiment_name"], plan_path)] = row
    groups = {}
    for row in unique.values():
        if row.get("ranking_source") != "rmsnorm_ellipsoid_bound":
            continue
        key = (
            row["allocation_source"], row.get("exact_total_layer_channels", ""),
            row["layer_channels"], row["ranking_source"],
        )
        groups.setdefault(key, {})[row["ranking_aggregation"]] = row
    comparisons = []
    for key, by_aggregation in groups.items():
        if set(by_aggregation) != {"p95", "max"}:
            continue
        plans = {}
        for aggregation, row in by_aggregation.items():
            with open(row["pruning_plan_path"], encoding="utf-8") as handle:
                plan = json.load(handle)
            plans[aggregation] = {
                int(layer["layer_idx"]): set(int(i) for i in layer["prune_idx"])
                for layer in plan["layers"]
            }
        if set(plans["p95"]) != set(plans["max"]):
            raise ValueError("p95/max aggregation plans contain different layers")
        layer_rows = []
        total_overlap = 0
        total_selected = 0
        for layer_idx in sorted(plans["p95"]):
            p95_ids = plans["p95"][layer_idx]
            max_ids = plans["max"][layer_idx]
            if len(p95_ids) != len(max_ids):
                raise ValueError(
                    f"p95/max fixed allocation differs in layer {layer_idx}"
                )
            overlap = len(p95_ids.intersection(max_ids))
            union = len(p95_ids.union(max_ids))
            total_overlap += overlap
            total_selected += len(p95_ids)
            layer_rows.append({
                "layer_idx": layer_idx,
                "selected_count": len(p95_ids),
                "overlap_count": overlap,
                "changed_ids_per_ranking": len(p95_ids) - overlap,
                "symmetric_difference_count": len(p95_ids.symmetric_difference(max_ids)),
                "jaccard": overlap / union if union else 1.0,
                "p95_selected_ids": sorted(p95_ids),
                "max_selected_ids": sorted(max_ids),
            })
        comparisons.append({
            "allocation_source": key[0],
            "exact_total_layer_channels": key[1],
            "total_selected_layer_channels": total_selected,
            "p95_experiment": by_aggregation["p95"]["experiment_name"],
            "max_experiment": by_aggregation["max"]["experiment_name"],
            "global_overlap_count": total_overlap,
            "global_changed_ids_per_ranking": total_selected - total_overlap,
            "global_symmetric_difference_count": 2 * (total_selected - total_overlap),
            "global_jaccard": (
                total_overlap / (2 * total_selected - total_overlap)
                if total_selected else 1.0
            ),
            "layers": layer_rows,
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = []
    load_ids_by_experiment = {}
    process_ids_by_experiment = {}
    for item in manifest:
        with open(_main_csv(item["output_dir"]), newline="") as handle:
            for result in csv.DictReader(handle):
                if result.get("evaluation_token_count_match", "").lower() not in (
                    "true", "1"
                ):
                    raise ValueError(
                        f"{item['experiment_name']} token-count match failed"
                    )
                if result.get("baseline_eval_tokens") != result.get("pruned_eval_tokens"):
                    raise ValueError(
                        f"{item['experiment_name']} baseline/pruned tokens differ"
                    )
                load_ids_by_experiment.setdefault(
                    item["experiment_name"], set()
                ).add(result.get("model_load_instance_id", ""))
                process_ids_by_experiment.setdefault(
                    item["experiment_name"], set()
                ).add(result.get("process_id", ""))
                tightness_path = result.get("bound_tightness_json_path", "")
                row = {
                    "experiment_name": item["experiment_name"],
                    "allocation_source": result.get("allocation_source", ""),
                    "ranking_source": result.get("ranking_source", ""),
                    "allocation_aggregation": result.get(
                        "allocation_aggregation_mode", "p95"
                    ),
                    "ranking_aggregation": result.get(
                        "ranking_aggregation_mode", result.get("aggregation_mode", "")
                    ),
                    "exact_total_layer_channels": result.get(
                        "exact_total_layer_channels", ""
                    ),
                    "dataset": result.get("eval_dataset", ""),
                    "requested_pct": result.get("target_pct", ""),
                    "actual_pct": result.get("actual_pct", ""),
                    "layer_channels": result.get("selected_layer_channels", ""),
                    "expert_neurons": result.get("removed_expert_neurons", ""),
                    "expert_param_reduction_pct": result.get("expert_param_reduction_pct", ""),
                    "total_model_param_reduction_pct": result.get(
                        "total_model_param_reduction_pct", ""
                    ),
                    "baseline_ppl": result.get("baseline_ppl", ""),
                    "pruned_ppl": result.get("compressed_ppl", ""),
                    "relative_ppl_change_pct": result.get("relative_delta_pct", ""),
                    "baseline_tokens": result.get("baseline_eval_tokens", ""),
                    "pruned_tokens": result.get("pruned_eval_tokens", ""),
                    "token_count_match": result.get("evaluation_token_count_match", ""),
                    "process_id": result.get("process_id", ""),
                    "model_load_instance_id": result.get("model_load_instance_id", ""),
                    "pruning_plan_path": result.get("pruning_plan_path", ""),
                    "per_layer_csv_path": result.get("per_layer_csv_path", ""),
                    "score_comparison_json_path": result.get("score_comparison_json_path", ""),
                    "mean_nll_difference": result.get("mean_nll_difference", ""),
                    "mean_nll_ci95_lower": result.get(
                        "mean_nll_difference_ci95_lower", ""
                    ),
                    "mean_nll_ci95_upper": result.get(
                        "mean_nll_difference_ci95_upper", ""
                    ),
                    "per_example_nll_path": result.get("per_example_nll_path", ""),
                    "bound_tightness_json_path": tightness_path,
                }
                row.update(_bound_fields(tightness_path))
                rows.append(row)
    for experiment, load_ids in load_ids_by_experiment.items():
        if len(load_ids) != 1 or "" in load_ids:
            raise ValueError(
                f"{experiment} does not have exactly one model-load identity: "
                f"{load_ids}"
            )
    all_load_ids = [next(iter(ids)) for ids in load_ids_by_experiment.values()]
    if len(all_load_ids) != len(set(all_load_ids)):
        raise ValueError(
            "two experiment cells reused a model-load identity; fresh-process "
            "isolation failed"
        )
    for experiment, process_ids in process_ids_by_experiment.items():
        if len(process_ids) != 1 or "" in process_ids:
            raise ValueError(
                f"{experiment} does not have exactly one process ID: {process_ids}"
            )
    all_process_ids = [
        next(iter(ids)) for ids in process_ids_by_experiment.values()
    ]
    if len(all_process_ids) != len(set(all_process_ids)):
        raise ValueError("experiment cells did not use distinct process IDs")
    exact_rows = [row for row in rows if row["exact_total_layer_channels"]]
    if exact_rows:
        requested_totals = {
            int(float(row["exact_total_layer_channels"])) for row in exact_rows
        }
        channel_totals = {int(float(row["layer_channels"])) for row in exact_rows}
        expert_neuron_totals = {
            int(float(row["expert_neurons"])) for row in exact_rows
        }
        if len(requested_totals) != 1 or channel_totals != requested_totals:
            raise ValueError(
                "exact-budget experiments did not all match their requested total"
            )
        if len(expert_neuron_totals) != 1:
            raise ValueError(
                "exact-budget experiments removed different expert-neuron totals"
            )
    output = os.path.join(args.run_dir, "allocation_ranking_summary.csv")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    paired_comparisons = []
    groups = {}
    for row in rows:
        key = (
            row["allocation_source"], row["dataset"],
            row["ranking_aggregation"], row["layer_channels"],
        )
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        ellipsoids = [
            row for row in group
            if row["ranking_source"] == "rmsnorm_ellipsoid_bound"
            and row["per_example_nll_path"]
        ]
        if len(ellipsoids) != 1:
            continue
        ellipsoid = ellipsoids[0]
        ellipsoid_docs = _read_paired_rows(ellipsoid["per_example_nll_path"])
        for competitor in group:
            if competitor is ellipsoid or not competitor["per_example_nll_path"]:
                continue
            competitor_docs = _read_paired_rows(competitor["per_example_nll_path"])
            _validate_paired_documents(
                ellipsoid_docs,
                competitor_docs,
                label=(
                    f"paired ranking {ellipsoid['experiment_name']} versus "
                    f"{competitor['experiment_name']}"
                ),
            )
            stats = paired_bootstrap_nll_difference(
                [float(row["pruned_nll_sum"]) for row in ellipsoid_docs],
                [float(row["pruned_nll_sum"]) for row in competitor_docs],
                [int(row["n_tokens"]) for row in ellipsoid_docs],
                n_resamples=args.bootstrap_resamples,
                seed=42,
            )
            paired_comparisons.append({
                "allocation_source": key[0],
                "dataset": key[1],
                "aggregation_mode": key[2],
                "exact_removed_layer_channels": key[3],
                "ellipsoid_experiment": ellipsoid["experiment_name"],
                "competitor_ranking": competitor["ranking_source"],
                "competitor_experiment": competitor["experiment_name"],
                "ellipsoid_minus_competitor_mean_nll": stats[
                    "mean_nll_difference"
                ],
                "ci95_lower": stats["ci_lower"],
                "ci95_upper": stats["ci_upper"],
                "n_documents": stats["n_documents"],
                "n_tokens": stats["n_tokens"],
                "bootstrap_resamples": stats["n_resamples"],
            })
    paired_output = os.path.join(args.run_dir, "paired_ranking_comparisons.csv")
    if paired_comparisons:
        with open(paired_output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(paired_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(paired_comparisons)
    allocation_comparisons = build_paired_allocation_comparisons(
        rows, bootstrap_resamples=args.bootstrap_resamples
    )
    allocation_comparison_output = os.path.join(
        args.run_dir, "paired_allocation_comparisons.csv"
    )
    if allocation_comparisons:
        with open(
            allocation_comparison_output, "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(allocation_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(allocation_comparisons)
    aggregation_comparisons = build_aggregation_selection_comparisons(rows)
    aggregation_comparison_output = os.path.join(
        args.run_dir, "aggregation_selection_comparison.json"
    )
    if aggregation_comparisons:
        with open(aggregation_comparison_output, "w", encoding="utf-8") as handle:
            json.dump(aggregation_comparisons, handle, indent=2)
    paired_aggregation_comparisons = build_paired_aggregation_comparisons(
        rows, bootstrap_resamples=args.bootstrap_resamples
    )
    paired_aggregation_output = os.path.join(
        args.run_dir, "paired_aggregation_comparisons.csv"
    )
    if paired_aggregation_comparisons:
        with open(
            paired_aggregation_output, "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(paired_aggregation_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(paired_aggregation_comparisons)
    count_vectors = []
    seen_plans = set()
    for row in rows:
        plan_path = row["pruning_plan_path"]
        if not plan_path or plan_path in seen_plans:
            continue
        seen_plans.add(plan_path)
        with open(plan_path, encoding="utf-8") as handle:
            plan = json.load(handle)
        audit = plan.get("allocation_ranking", {})
        count_vectors.append({
            "experiment_name": row["experiment_name"],
            "allocation_source": row["allocation_source"],
            "ranking_source": row["ranking_source"],
            "aggregation_mode": row["ranking_aggregation"],
            "source_plan_total_layer_channels": audit.get(
                "source_plan_total_layer_channels"
            ),
            "exact_total_layer_channels": audit.get("exact_total_layer_channels"),
            "total_selected_layer_channels": audit.get(
                "total_selected_layer_channels"
            ),
            "total_removed_expert_neurons": audit.get(
                "total_removed_expert_neurons"
            ),
            "per_layer_counts": {
                str(layer["layer_idx"]): layer["required_pruned_channels"]
                for layer in audit.get("layers", [])
            },
        })
    vectors_output = os.path.join(args.run_dir, "allocation_count_vectors.json")
    with open(vectors_output, "w", encoding="utf-8") as handle:
        json.dump(count_vectors, handle, indent=2)
    print(
        f"{'experiment':36s} {'dataset':12s} {'alloc':14s} {'rank':25s} "
        f"{'act%':>7s} {'ch':>5s} {'base':>9s} {'pruned':>9s} {'rel%':>8s} {'tokens':>9s}"
    )
    print("-" * 155)
    for row in rows:
        print(
            f"{row['experiment_name'][:36]:36s} {row['dataset'][:12]:12s} "
            f"{row['allocation_source'][:14]:14s} {row['ranking_source'][:25]:25s} "
            f"{float(row['actual_pct']):7.3f} {int(float(row['layer_channels'])):5d} "
            f"{float(row['baseline_ppl']):9.4f} {float(row['pruned_ppl']):9.4f} "
            f"{float(row['relative_ppl_change_pct']):8.3f} "
            f"{int(float(row['pruned_tokens'])):9d}"
        )
    print(f"[alloc-rank-summary] CSV: {output}")
    print(f"[alloc-rank-summary] count vectors: {vectors_output}")
    if paired_comparisons:
        print(f"[alloc-rank-summary] paired ranking CIs: {paired_output}")
    if allocation_comparisons:
        print(
            "[alloc-rank-summary] paired allocation CIs: "
            f"{allocation_comparison_output}"
        )
    if aggregation_comparisons:
        print(
            "[alloc-rank-summary] aggregation channel overlap: "
            f"{aggregation_comparison_output}"
        )
    if paired_aggregation_comparisons:
        print(
            "[alloc-rank-summary] paired aggregation CIs: "
            f"{paired_aggregation_output}"
        )


if __name__ == "__main__":
    main()
