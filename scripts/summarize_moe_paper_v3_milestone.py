#!/usr/bin/env python3
"""Combine frozen and follow-up MoE results into one Version 3 milestone table."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.build_moe_paper_v3_evidence import (
    _read_ok_rows,
    _resolve_pruning_plan,
)
from scripts.summarize_moe_allocation_ranking import _bound_fields
from src.experiment_provenance import file_sha256


DATASETS = ("wikitext2", "c4")
OUTPUT_FIELDS = [
    "source_group", "experiment_name", "requested_pct", "allocation_source",
    "ranking_source", "expert_aggregation", "exact_total_layer_channels",
    "actual_pct", "removed_layer_channels", "removed_expert_neurons",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "wikitext2_baseline_ppl", "wikitext2_pruned_ppl",
    "wikitext2_relative_ppl_pct", "wikitext2_mean_nll_difference",
    "wikitext2_nll_ci95_lower", "wikitext2_nll_ci95_upper",
    "wikitext2_tokens", "c4_baseline_ppl", "c4_pruned_ppl",
    "c4_relative_ppl_pct", "c4_mean_nll_difference",
    "c4_nll_ci95_lower", "c4_nll_ci95_upper", "c4_tokens",
    "ellipsoid_tightness_median", "ellipsoid_tightness_p95",
    "ellipsoid_tightness_p99", "ellipsoid_tightness_max",
    "sphere_tightness_median", "sphere_tightness_p95",
    "sphere_tightness_p99", "sphere_tightness_max",
    "sphere_to_ellipsoid_bound_median", "sphere_to_ellipsoid_bound_p95",
    "sphere_to_ellipsoid_bound_max", "ellipsoid_bound_violations",
    "sphere_bound_violations", "sampled_routed_inputs",
    "expert_channel_pairs_evaluated", "pruning_plan_sha256",
    "pruning_plan_path", "result_directories",
]


def _experiment_name(result_directory: str, allocation: str, ranking: str) -> str:
    name = os.path.basename(os.path.normpath(result_directory))
    return name or f"{allocation}_alloc__{ranking}_rank"


def _normalize_evidence_row(row: dict, source_group: str) -> dict:
    bound = {}
    result_csv = row.get("result_csv", "")
    if result_csv and os.path.isfile(result_csv):
        with open(result_csv, newline="", encoding="utf-8") as handle:
            matches = [
                raw for raw in csv.DictReader(handle)
                if raw.get("status") == "ok"
                and raw.get("eval_dataset") == row["dataset"]
            ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {row['dataset']} row in frozen source {result_csv}"
            )
        bound = _bound_fields(matches[0].get("bound_tightness_json_path", ""))
    return {
        "source_group": source_group,
        "experiment_name": _experiment_name(
            row["result_directory"], row["allocation_source"], row["ranking_source"]
        ),
        "dataset": row["dataset"],
        "requested_pct": row["requested_pruning_pct"],
        "allocation_source": row["allocation_source"],
        "ranking_source": row["ranking_source"],
        "expert_aggregation": row["expert_aggregation"],
        "exact_total_layer_channels": "",
        "actual_pct": row["actual_expert_width_pct"],
        "removed_layer_channels": row["removed_layer_channels"],
        "removed_expert_neurons": row["removed_expert_neurons"],
        "expert_param_reduction_pct": row["expert_param_reduction_pct"],
        "total_model_param_reduction_pct": row["total_model_param_reduction_pct"],
        "baseline_ppl": row["baseline_ppl"],
        "pruned_ppl": row["pruned_ppl"],
        "relative_ppl_pct": row["relative_ppl_change_pct"],
        "mean_nll_difference": row["mean_nll_difference"],
        "nll_ci95_lower": row["mean_nll_ci95_lower"],
        "nll_ci95_upper": row["mean_nll_ci95_upper"],
        "tokens": row["evaluation_token_count"],
        "evaluation_sample_set_id": row["evaluation_sample_set_id"],
        "pruning_plan_path": row["pruning_plan_path"],
        "pruning_plan_sha256": row["pruning_plan_sha256"],
        "result_directory": row["result_directory"],
        "bound": bound,
    }


def _normalize_raw_row(run_dir: str, csv_path: str, row: dict) -> dict:
    plan_path, _ = _resolve_pruning_plan(csv_path, row)
    bound = _bound_fields(row.get("bound_tightness_json_path", ""))
    return {
        "source_group": os.path.basename(os.path.normpath(run_dir)),
        "experiment_name": _experiment_name(
            os.path.dirname(csv_path), row["allocation_source"], row["ranking_source"]
        ),
        "dataset": row["eval_dataset"],
        "requested_pct": row["target_pct"],
        "allocation_source": row["allocation_source"],
        "ranking_source": row["ranking_source"],
        "expert_aggregation": row.get(
            "ranking_aggregation_mode", row.get("aggregation_mode", "")
        ),
        "exact_total_layer_channels": row.get("exact_total_layer_channels", ""),
        "actual_pct": row["actual_pct"],
        "removed_layer_channels": row["selected_layer_channels"],
        "removed_expert_neurons": row["removed_expert_neurons"],
        "expert_param_reduction_pct": row["expert_param_reduction_pct"],
        "total_model_param_reduction_pct": row["total_model_param_reduction_pct"],
        "baseline_ppl": row["baseline_ppl"],
        "pruned_ppl": row["compressed_ppl"],
        "relative_ppl_pct": row["relative_delta_pct"],
        "mean_nll_difference": row["mean_nll_difference"],
        "nll_ci95_lower": row["mean_nll_difference_ci95_lower"],
        "nll_ci95_upper": row["mean_nll_difference_ci95_upper"],
        "tokens": row["pruned_eval_tokens"],
        "evaluation_sample_set_id": (
            f"{row['eval_dataset']}:n{row['evaluation_num_texts']}:"
            f"{row['evaluation_corpus_sha256']}"
        ),
        "pruning_plan_path": plan_path,
        "pruning_plan_sha256": file_sha256(plan_path),
        "result_directory": os.path.dirname(csv_path),
        "bound": bound,
    }


def build_milestone_rows(dataset_rows: list[dict]) -> list[dict]:
    """Pivot two dataset rows into one experiment row and validate invariants."""
    sample_sets = defaultdict(set)
    for row in dataset_rows:
        sample_sets[row["dataset"]].add(row["evaluation_sample_set_id"])
    for dataset in DATASETS:
        if len(sample_sets[dataset]) != 1:
            raise ValueError(
                f"{dataset} does not use exactly one sample set: {sample_sets[dataset]}"
            )

    groups = defaultdict(list)
    for row in dataset_rows:
        key = (
            row["source_group"], row["experiment_name"], row["requested_pct"],
            row["allocation_source"], row["ranking_source"],
            row["expert_aggregation"], row["removed_layer_channels"],
        )
        groups[key].append(row)

    output = []
    invariant_fields = (
        "actual_pct", "removed_layer_channels", "removed_expert_neurons",
        "expert_param_reduction_pct", "total_model_param_reduction_pct",
        "pruning_plan_sha256", "pruning_plan_path",
    )
    for key, rows in groups.items():
        by_dataset = {row["dataset"]: row for row in rows}
        if set(by_dataset) != set(DATASETS) or len(rows) != len(DATASETS):
            raise ValueError(f"experiment {key} does not have exactly {DATASETS}")
        first = rows[0]
        for field in invariant_fields:
            if len({str(row[field]) for row in rows}) != 1:
                raise ValueError(f"experiment {key} varies by dataset in {field}")
        result = {
            "source_group": key[0], "experiment_name": key[1],
            "requested_pct": key[2], "allocation_source": key[3],
            "ranking_source": key[4], "expert_aggregation": key[5],
            "exact_total_layer_channels": first["exact_total_layer_channels"],
            "actual_pct": first["actual_pct"],
            "removed_layer_channels": first["removed_layer_channels"],
            "removed_expert_neurons": first["removed_expert_neurons"],
            "expert_param_reduction_pct": first["expert_param_reduction_pct"],
            "total_model_param_reduction_pct": first[
                "total_model_param_reduction_pct"
            ],
            "pruning_plan_sha256": first["pruning_plan_sha256"],
            "pruning_plan_path": first["pruning_plan_path"],
            "result_directories": ";".join(sorted({
                row["result_directory"] for row in rows
            })),
        }
        for dataset, row in by_dataset.items():
            result.update({
                f"{dataset}_baseline_ppl": row["baseline_ppl"],
                f"{dataset}_pruned_ppl": row["pruned_ppl"],
                f"{dataset}_relative_ppl_pct": row["relative_ppl_pct"],
                f"{dataset}_mean_nll_difference": row["mean_nll_difference"],
                f"{dataset}_nll_ci95_lower": row["nll_ci95_lower"],
                f"{dataset}_nll_ci95_upper": row["nll_ci95_upper"],
                f"{dataset}_tokens": row["tokens"],
            })
        bounds = [row["bound"] for row in rows if row["bound"]]
        bound = bounds[0] if bounds else {}
        for field in OUTPUT_FIELDS:
            if field.startswith(("ellipsoid_", "sphere_", "sampled_", "expert_channel_")):
                result.setdefault(field, bound.get(field, ""))
        output.append(result)
    return sorted(output, key=lambda row: (
        float(row["requested_pct"]), row["source_group"], row["experiment_name"],
        row["expert_aggregation"],
    ))


def _fmt(value, digits=3) -> str:
    if value in (None, ""):
        return "--"
    return f"{float(value):.{digits}f}"


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'source':30s} {'experiment':38s} {'tgt':>4s} {'agg':>4s} "
        f"{'ch':>5s} {'act%':>7s} {'W2 rel%':>8s} {'W2 dNLL [95% CI]':>27s} "
        f"{'C4 rel%':>8s} {'C4 dNLL [95% CI]':>27s} {'viol':>5s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        w2_ci = (
            f"{_fmt(row['wikitext2_mean_nll_difference'], 5)} "
            f"[{_fmt(row['wikitext2_nll_ci95_lower'], 5)},"
            f"{_fmt(row['wikitext2_nll_ci95_upper'], 5)}]"
        )
        c4_ci = (
            f"{_fmt(row['c4_mean_nll_difference'], 5)} "
            f"[{_fmt(row['c4_nll_ci95_lower'], 5)},"
            f"{_fmt(row['c4_nll_ci95_upper'], 5)}]"
        )
        violations = row.get("ellipsoid_bound_violations", "")
        print(
            f"{row['source_group'][:30]:30s} {row['experiment_name'][:38]:38s} "
            f"{float(row['requested_pct']):4.0f} {row['expert_aggregation'][:4]:>4s} "
            f"{int(float(row['removed_layer_channels'])):5d} "
            f"{float(row['actual_pct']):7.3f} "
            f"{float(row['wikitext2_relative_ppl_pct']):8.3f} {w2_ci:>27s} "
            f"{float(row['c4_relative_ppl_pct']):8.3f} {c4_ci:>27s} "
            f"{str(violations) if violations != '' else '--':>5s}"
        )


def _write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "| Source | Experiment | Target | Aggregation | Channels | Actual % | "
            "WikiText-2 rel. PPL % | C4 rel. PPL % | Ellipsoid violations |\n"
            "|---|---|---:|---|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['source_group']} | {row['experiment_name']} | "
                f"{_fmt(row['requested_pct'], 0)} | {row['expert_aggregation']} | "
                f"{row['removed_layer_channels']} | {_fmt(row['actual_pct'])} | "
                f"{_fmt(row['wikitext2_relative_ppl_pct'])} | "
                f"{_fmt(row['c4_relative_ppl_pct'])} | "
                f"{row.get('ellipsoid_bound_violations', '') or '--'} |\n"
            )


def _collect_comparisons(source_dirs: list[str]) -> list[dict]:
    names = {
        "paired_ranking_comparisons.csv": "ranking",
        "paired_aggregation_comparisons.csv": "aggregation",
        "paired_allocation_comparisons.csv": "allocation",
    }
    rows = []
    for source_dir in source_dirs:
        for filename, comparison_type in names.items():
            path = os.path.join(source_dir, filename)
            if not os.path.isfile(path):
                continue
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append({
                        "comparison_type": comparison_type,
                        "source_group": os.path.basename(os.path.normpath(source_dir)),
                        **row,
                    })
    return rows


def _collect_aggregation_overlaps(source_dirs: list[str]) -> list[dict]:
    rows = []
    for source_dir in source_dirs:
        path = os.path.join(source_dir, "aggregation_selection_comparison.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"aggregation overlap report is not a list: {path}")
        for row in payload:
            rows.append({
                "source_group": os.path.basename(os.path.normpath(source_dir)),
                **row,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(
            f"refusing to overwrite summary directory: {args.output_dir}"
        )
    evidence_path = os.path.join(args.evidence_dir, "paper_v3_evidence.csv")
    with open(evidence_path, newline="", encoding="utf-8") as handle:
        dataset_rows = [
            _normalize_evidence_row(row, "frozen_2_4_6")
            for row in csv.DictReader(handle)
        ]
    for run_dir, csv_path, row in _read_ok_rows(args.run_dir):
        dataset_rows.append(_normalize_raw_row(run_dir, csv_path, row))
    rows = build_milestone_rows(dataset_rows)

    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "paper_v3_milestone_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(args.output_dir, "paper_v3_milestone_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    markdown_path = os.path.join(args.output_dir, "paper_v3_milestone_summary.md")
    _write_markdown(markdown_path, rows)

    comparisons = _collect_comparisons([args.evidence_dir, *args.run_dir])
    comparison_path = os.path.join(
        args.output_dir, "paper_v3_paired_comparisons.csv"
    )
    if comparisons:
        fields = []
        for row in comparisons:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with open(comparison_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparisons)
    aggregation_overlaps = _collect_aggregation_overlaps(args.run_dir)
    overlap_path = os.path.join(
        args.output_dir, "paper_v3_aggregation_selection_comparisons.json"
    )
    if aggregation_overlaps:
        with open(overlap_path, "w", encoding="utf-8") as handle:
            json.dump(aggregation_overlaps, handle, indent=2)

    _print_table(rows)
    print(f"\n[paper-v3-summary] experiments: {len(rows)}")
    print(f"[paper-v3-summary] CSV : {csv_path}")
    print(f"[paper-v3-summary] JSON: {json_path}")
    print(f"[paper-v3-summary] MD  : {markdown_path}")
    if comparisons:
        print(f"[paper-v3-summary] paired comparisons: {comparison_path}")
    if aggregation_overlaps:
        print(f"[paper-v3-summary] aggregation overlaps: {overlap_path}")


if __name__ == "__main__":
    main()
