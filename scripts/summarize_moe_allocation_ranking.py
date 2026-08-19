#!/usr/bin/env python3
"""Build and print a compact allocation/ranking matrix summary."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


FIELDS = [
    "experiment_name", "allocation_source", "ranking_source", "dataset",
    "requested_pct", "actual_pct", "layer_channels", "expert_neurons",
    "expert_param_reduction_pct", "baseline_ppl", "pruned_ppl",
    "relative_ppl_change_pct", "baseline_tokens", "pruned_tokens",
    "token_count_match", "process_id", "model_load_instance_id",
    "pruning_plan_path", "per_layer_csv_path", "score_comparison_json_path",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
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
                rows.append({
                    "experiment_name": item["experiment_name"],
                    "allocation_source": result.get("allocation_source", ""),
                    "ranking_source": result.get("ranking_source", ""),
                    "dataset": result.get("eval_dataset", ""),
                    "requested_pct": result.get("target_pct", ""),
                    "actual_pct": result.get("actual_pct", ""),
                    "layer_channels": result.get("selected_layer_channels", ""),
                    "expert_neurons": result.get("removed_expert_neurons", ""),
                    "expert_param_reduction_pct": result.get("expert_param_reduction_pct", ""),
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
                })
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
    output = os.path.join(args.run_dir, "allocation_ranking_summary.csv")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
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


if __name__ == "__main__":
    main()
