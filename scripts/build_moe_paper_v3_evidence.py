#!/usr/bin/env python3
"""Freeze validated allocation/ranking runs into immutable paper evidence."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment_provenance import (
    assert_calibration_evaluation_disjoint,
    build_text_manifest,
    file_sha256,
)


EVIDENCE_FIELDS = [
    "model", "model_revision", "tokenizer_revision", "tokenizer_name_or_path",
    "dataset", "evaluation_sample_set_id", "evaluation_sample_manifest_path",
    "calibration_sample_set_id", "calibration_sample_manifest_path",
    "calibration_eval_disjoint_verified", "n_eval", "evaluation_num_texts",
    "evaluation_token_count", "evaluation_max_seq_len", "evaluation_batch_size",
    "evaluation_preprocessing", "seed", "allocation_source", "ranking_source",
    "expert_aggregation", "requested_pruning_pct", "actual_expert_width_pct",
    "removed_layer_channels", "removed_expert_neurons",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "baseline_ppl", "pruned_ppl", "relative_ppl_change_pct",
    "mean_nll_difference", "mean_nll_ci95_lower", "mean_nll_ci95_upper",
    "paired_bootstrap_resamples", "result_directory", "result_csv",
    "pruning_plan_path", "pruning_plan_sha256", "per_example_nll_path",
    "per_example_nll_sha256", "process_id", "model_load_instance_id",
]


def _main_result_csvs(run_dir: str) -> list[str]:
    paths = []
    for path in glob.glob(
        os.path.join(run_dir, "**", "moe_target_pruning_*.csv"), recursive=True
    ):
        if not path.endswith("_per_layer.csv"):
            paths.append(path)
    return sorted(paths)


def _read_ok_rows(run_dirs: list[str]) -> list[tuple[str, str, dict]]:
    found = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"evidence source directory not found: {run_dir}")
        csv_paths = _main_result_csvs(run_dir)
        if not csv_paths:
            raise FileNotFoundError(f"no result CSVs found under {run_dir}")
        for csv_path in csv_paths:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("status") != "ok":
                        continue
                    if row.get("evaluation_token_count_match", "").lower() not in {
                        "true", "1"
                    }:
                        raise ValueError(f"token-count validation failed: {csv_path}")
                    found.append((run_dir, csv_path, row))
    if not found:
        raise ValueError("no successful evidence rows found")
    return found


def _resolve_identity(
    rows: list[tuple[str, str, dict]], field: str, override: str
) -> str:
    values = {row.get(field, "").strip() for _, _, row in rows}
    values.discard("")
    if override:
        if values and values != {override}:
            raise ValueError(
                f"--{field.replace('_', '-')}={override!r} conflicts with {values}"
            )
        return override
    if len(values) == 1:
        return next(iter(values))
    raise ValueError(
        f"cannot uniquely establish {field}; recorded values={sorted(values)}. "
        f"Pass --{field.replace('_', '-')} from the exact cached checkpoint snapshot."
    )


def _plan_counts(path: str) -> dict[int, int]:
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    return {
        int(layer["layer_idx"]): len(layer.get("prune_idx", []))
        for layer in plan["layers"]
    }


def _copy_comparison_tables(run_dirs: list[str], output_dir: str) -> list[str]:
    names = (
        "paired_ranking_comparisons.csv",
        "paired_allocation_comparisons.csv",
        "paired_aggregation_comparisons.csv",
    )
    outputs = []
    for name in names:
        combined = []
        fields = []
        for run_dir in run_dirs:
            path = os.path.join(run_dir, name)
            if not os.path.isfile(path):
                continue
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    tagged = {"source_run_dir": run_dir, **row}
                    combined.append(tagged)
                    for field in tagged:
                        if field not in fields:
                            fields.append(field)
        if combined:
            output = os.path.join(output_dir, name)
            with open(output, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(combined)
            outputs.append(output)
    return outputs


def build_evidence(
    *, run_dirs: list[str], output_dir: str, model_revision: str = "",
    tokenizer_revision: str = "", require_targets: set[int] | None = None,
    evaluation_corpora: dict[str, list[str]] | None = None,
    calibration_prompts: list[str] | None = None,
) -> dict:
    """Build a new evidence directory without mutating any source result."""
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"refusing to overwrite evidence directory: {output_dir}"
        )
    source_rows = _read_ok_rows(run_dirs)
    identities_by_csv = defaultdict(lambda: {"process": set(), "load": set()})
    run_by_csv = {}
    for run_dir, csv_path, row in source_rows:
        run_by_csv[csv_path] = run_dir
        if not row.get("baseline_eval_tokens") or (
            row.get("baseline_eval_tokens") != row.get("pruned_eval_tokens")
        ):
            raise ValueError(f"baseline/pruned token counts differ: {csv_path}")
        identities_by_csv[csv_path]["process"].add(row.get("process_id", ""))
        identities_by_csv[csv_path]["load"].add(
            row.get("model_load_instance_id", "")
        )
    for csv_path, identities in identities_by_csv.items():
        if any(len(values) != 1 or "" in values for values in identities.values()):
            raise ValueError(f"fresh-process identity is incomplete: {csv_path}")
    process_ids_by_run = defaultdict(list)
    for csv_path in identities_by_csv:
        run_dir = run_by_csv[csv_path]
        process_id = next(iter(identities_by_csv[csv_path]["process"]))
        process_ids_by_run[run_dir].append(process_id)
    load_ids = [next(iter(value["load"])) for value in identities_by_csv.values()]
    for run_dir, process_ids in process_ids_by_run.items():
        if len(process_ids) != len(set(process_ids)):
            raise ValueError(f"two cells reused a process ID within {run_dir}")
    if len(load_ids) != len(set(load_ids)):
        raise ValueError("two evidence cells reused a model-load identity")
    models = {row.get("model", "") for _, _, row in source_rows}
    if len(models) != 1 or "" in models:
        raise ValueError(f"evidence rows do not identify one model: {models}")
    resolved_model_revision = _resolve_identity(
        source_rows, "model_revision", model_revision
    )
    resolved_tokenizer_revision = _resolve_identity(
        source_rows, "tokenizer_revision", tokenizer_revision
    )
    targets = {int(round(float(row["target_pct"]))) for _, _, row in source_rows}
    if require_targets and targets != require_targets:
        raise ValueError(
            f"expected target set {sorted(require_targets)}, found {sorted(targets)}"
        )

    dataset_n_eval = defaultdict(set)
    expected_corpus_hashes = defaultdict(set)
    for _, _, row in source_rows:
        dataset_n_eval[row["eval_dataset"]].add(int(row["n_eval"]))
        expected_corpus_hashes[row["eval_dataset"]].add(
            row["evaluation_corpus_sha256"]
        )
    if any(len(values) != 1 for values in dataset_n_eval.values()):
        raise ValueError(f"datasets use inconsistent n_eval values: {dataset_n_eval}")
    if any(len(values) != 1 for values in expected_corpus_hashes.values()):
        raise ValueError("a dataset has multiple evaluation sample corpora")
    if evaluation_corpora is None:
        from src.evaluation import load_all_eval_datasets

        evaluation_corpora = {}
        for dataset, n_values in dataset_n_eval.items():
            evaluation_corpora.update(load_all_eval_datasets(
                [dataset], max_samples=next(iter(n_values)),
                use_fallback_corpus=False,
            ))
    if calibration_prompts is None:
        from src.merging import RECONSTRUCTION_TRAIN_PROMPTS

        calibration_prompts = list(RECONSTRUCTION_TRAIN_PROMPTS)
    if set(evaluation_corpora) != set(dataset_n_eval):
        raise ValueError("provided evaluation corpora do not match result datasets")
    text_manifest = build_text_manifest(evaluation_corpora, calibration_prompts)
    assert_calibration_evaluation_disjoint(text_manifest)
    for dataset, values in expected_corpus_hashes.items():
        recorded = next(iter(values))
        regenerated = text_manifest["evaluation"][dataset]["corpus_sha256"]
        if recorded != regenerated:
            raise ValueError(
                f"{dataset} sample corpus changed: recorded={recorded}, "
                f"regenerated={regenerated}"
            )

    text_manifest_path = os.path.join(output_dir, "paper_v3_text_provenance.json")

    records = []
    count_vectors = {}
    source_files = {}
    seen = set()
    for run_dir, csv_path, row in source_rows:
        key = (
            run_dir, row.get("allocation_source"), row.get("ranking_source"),
            row.get("ranking_aggregation_mode", row.get("aggregation_mode")),
            row["eval_dataset"], row["target_pct"], row["selected_layer_channels"],
        )
        if key in seen:
            raise ValueError(f"duplicate evidence cell: {key}")
        seen.add(key)
        plan_path = row.get("pruning_plan_path", "")
        nll_path = row.get("per_example_nll_path", "")
        for required, label in ((plan_path, "pruning plan"), (nll_path, "paired NLL")):
            if not required or not os.path.isfile(required):
                raise FileNotFoundError(f"{label} missing for {csv_path}: {required}")
        plan_hash = file_sha256(plan_path)
        recorded_plan_hash = row.get("pruning_plan_sha256", "")
        if recorded_plan_hash and recorded_plan_hash != plan_hash:
            raise ValueError(f"pruning plan hash mismatch: {plan_path}")
        counts = _plan_counts(plan_path)
        if sum(counts.values()) != int(row["selected_layer_channels"]):
            raise ValueError(f"plan count does not match result row: {plan_path}")
        experiment_id = "__".join(
            [f"target{int(round(float(row['target_pct'])))}"]
            + [str(value) for value in key[1:4]]
        )
        count_vectors.setdefault(experiment_id, counts)
        if count_vectors[experiment_id] != counts:
            raise ValueError(f"count vector varies by dataset for {experiment_id}")
        dataset = row["eval_dataset"]
        eval_info = text_manifest["evaluation"][dataset]
        calibration = text_manifest["calibration"]
        record = {
            "model": row["model"],
            "model_revision": resolved_model_revision,
            "tokenizer_revision": resolved_tokenizer_revision,
            "tokenizer_name_or_path": row.get("tokenizer_name_or_path", "") or row["model"],
            "dataset": dataset,
            "evaluation_sample_set_id": (
                f"{dataset}:n{eval_info['num_samples']}:{eval_info['corpus_sha256']}"
            ),
            "evaluation_sample_manifest_path": text_manifest_path,
            "calibration_sample_set_id": (
                f"fixed_prompts:n{calibration['num_samples']}:"
                f"{calibration['corpus_sha256']}"
            ),
            "calibration_sample_manifest_path": text_manifest_path,
            "calibration_eval_disjoint_verified": True,
            "n_eval": row["n_eval"],
            "evaluation_num_texts": row["evaluation_num_texts"],
            "evaluation_token_count": row["pruned_eval_tokens"],
            "evaluation_max_seq_len": row["evaluation_max_seq_len"],
            "evaluation_batch_size": row["evaluation_batch_size"],
            "evaluation_preprocessing": row["evaluation_preprocessing"],
            "seed": row["seed"],
            "allocation_source": row.get("allocation_source", ""),
            "ranking_source": row.get("ranking_source", ""),
            "expert_aggregation": row.get(
                "ranking_aggregation_mode", row.get("aggregation_mode", "")
            ),
            "requested_pruning_pct": row["target_pct"],
            "actual_expert_width_pct": row["actual_pct"],
            "removed_layer_channels": row["selected_layer_channels"],
            "removed_expert_neurons": row["removed_expert_neurons"],
            "expert_param_reduction_pct": row["expert_param_reduction_pct"],
            "total_model_param_reduction_pct": row["total_model_param_reduction_pct"],
            "baseline_ppl": row["baseline_ppl"],
            "pruned_ppl": row["compressed_ppl"],
            "relative_ppl_change_pct": row["relative_delta_pct"],
            "mean_nll_difference": row["mean_nll_difference"],
            "mean_nll_ci95_lower": row["mean_nll_difference_ci95_lower"],
            "mean_nll_ci95_upper": row["mean_nll_difference_ci95_upper"],
            "paired_bootstrap_resamples": row["paired_bootstrap_resamples"],
            "result_directory": os.path.dirname(csv_path),
            "result_csv": csv_path,
            "pruning_plan_path": plan_path,
            "pruning_plan_sha256": plan_hash,
            "per_example_nll_path": nll_path,
            "per_example_nll_sha256": file_sha256(nll_path),
            "process_id": row["process_id"],
            "model_load_instance_id": row["model_load_instance_id"],
        }
        missing = [field for field in EVIDENCE_FIELDS if str(record.get(field, "")) == ""]
        if missing:
            raise ValueError(f"paper evidence fields missing in {csv_path}: {missing}")
        records.append(record)
        for path in (csv_path, plan_path, nll_path):
            source_files[path] = file_sha256(path)

    os.makedirs(output_dir)
    with open(text_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(text_manifest, handle, indent=2)
    csv_output = os.path.join(output_dir, "paper_v3_evidence.csv")
    with open(csv_output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    json_output = os.path.join(output_dir, "paper_v3_evidence.json")
    with open(json_output, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": 1,
            "immutable_source_policy": "source result directories were read only",
            "records": records,
        }, handle, indent=2)
    count_output = os.path.join(output_dir, "paper_v3_allocation_count_vectors.json")
    with open(count_output, "w", encoding="utf-8") as handle:
        json.dump(count_vectors, handle, indent=2, sort_keys=True)
    copied = _copy_comparison_tables(run_dirs, output_dir)
    source_manifest = os.path.join(output_dir, "SOURCE_MANIFEST.json")
    with open(source_manifest, "w", encoding="utf-8") as handle:
        json.dump({
            "source_run_directories": run_dirs,
            "source_file_sha256": source_files,
            "generated_files": [csv_output, json_output, text_manifest_path,
                                count_output, *copied],
        }, handle, indent=2, sort_keys=True)
    return {
        "records": len(records), "targets": sorted(targets),
        "output_dir": output_dir, "csv": csv_output, "json": json_output,
        "text_provenance": text_manifest_path,
        "calibration_eval_disjoint_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--require-targets", default="2,4,6")
    args = parser.parse_args()
    required = {int(value) for value in args.require_targets.split(",") if value}
    result = build_evidence(
        run_dirs=args.run_dir, output_dir=args.output_dir,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        require_targets=required,
    )
    print(
        "[paper-v3-evidence] OK: "
        f"records={result['records']} targets={result['targets']} "
        "calibration/evaluation disjoint=True"
    )
    print(f"[paper-v3-evidence] CSV: {result['csv']}")
    print(f"[paper-v3-evidence] JSON: {result['json']}")
    print(f"[paper-v3-evidence] provenance: {result['text_provenance']}")


if __name__ == "__main__":
    main()
