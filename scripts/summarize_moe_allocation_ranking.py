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
    "ellipsoid_bound_violations",
    "sphere_tightness_median", "sphere_tightness_p95",
    "sphere_tightness_p99", "sphere_tightness_max",
    "sphere_bound_violations",
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
        "ellipsoid_bound_violations": "",
        "sphere_tightness_median": "",
        "sphere_tightness_p95": "",
        "sphere_tightness_p99": "",
        "sphere_tightness_max": "",
        "sphere_bound_violations": "",
    }
    if not path:
        return empty
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    stats = payload["global"]["ellipsoid_all"]
    sphere = payload["global"]["sphere_all"]
    return {
        "ellipsoid_tightness_median": stats["median"],
        "ellipsoid_tightness_p95": stats["p95"],
        "ellipsoid_tightness_p99": stats["p99"],
        "ellipsoid_tightness_max": stats["max"],
        "ellipsoid_bound_violations": payload["global"][
            "ellipsoid_numerical_violations"
        ],
        "sphere_tightness_median": sphere["median"],
        "sphere_tightness_p95": sphere["p95"],
        "sphere_tightness_p99": sphere["p99"],
        "sphere_tightness_max": sphere["max"],
        "sphere_bound_violations": payload["global"][
            "sphere_numerical_violations"
        ],
    }


def _read_paired_rows(path: str) -> list[dict]:
    if not path:
        raise ValueError("per-example NLL path is empty")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"per-example NLL file is empty: {path}")
    return rows


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
            if len(ellipsoid_docs) != len(competitor_docs):
                raise ValueError("paired ranking document counts differ")
            for left, right in zip(ellipsoid_docs, competitor_docs):
                for field in ("dataset", "corpus_sha256", "sample_index", "n_tokens"):
                    if left[field] != right[field]:
                        raise ValueError(
                            f"paired ranking field {field} differs between "
                            f"{ellipsoid['experiment_name']} and "
                            f"{competitor['experiment_name']}"
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


if __name__ == "__main__":
    main()
