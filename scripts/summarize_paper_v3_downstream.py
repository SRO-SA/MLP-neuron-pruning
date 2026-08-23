#!/usr/bin/env python3
"""Create paired, paper-ready downstream tables from lm-eval sample logs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.task_config_fingerprint import FINGERPRINT_VERSION, task_config_sha256
from src.experiment_provenance import file_sha256
from src.statistical_audit import (
    apply_multiplicity_adjustments, paired_signflip_statistics,
)


TASK_METRICS = {
    "hellaswag": "acc_norm", "mathqa": "acc_norm",
    "openbookqa": "acc_norm", "piqa": "acc_norm",
    "winogrande": "acc", "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm", "boolq": "acc", "rte": "acc",
}


def _metric_value(metrics: dict, name: str) -> float:
    for key in (f"{name},none", name):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"metric {name!r} absent; available={sorted(metrics)}")


def _sample_value(sample: dict, name: str) -> float:
    for key in (f"{name},none", name):
        value = sample.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)
    metrics = sample.get("metrics", {})
    if isinstance(metrics, dict):
        return _metric_value(metrics, name)
    raise KeyError(f"sample metric {name!r} absent for doc_id={sample.get('doc_id')}")


def _sample_identity(sample: dict) -> tuple:
    return (
        str(sample.get("doc_id", "")), str(sample.get("doc_hash", "")),
        str(sample.get("target_hash", "")),
    )


def paired_bootstrap_accuracy(
    first: dict[str, list[float]], second: dict[str, list[float]],
    *, n_resamples: int, seed: int,
) -> dict:
    if set(first) != set(second):
        raise ValueError("paired task sets differ")
    observed = {task: float(np.mean(first[task]) - np.mean(second[task]))
                for task in first}
    rng = np.random.default_rng(seed)
    draws = {task: np.empty(n_resamples, dtype=np.float64) for task in first}
    for task in first:
        left = np.asarray(first[task], dtype=np.float64)
        right = np.asarray(second[task], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"paired sample counts differ for {task}")
        delta = left - right
        for start in range(0, n_resamples, 1000):
            stop = min(start + 1000, n_resamples)
            indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            draws[task][start:stop] = delta[indices].mean(axis=1)
    randomization = paired_signflip_statistics(
        first, second, n_resamples=n_resamples, seed=seed + 1_000_003,
    )
    result = {}
    for task in first:
        result[task] = {
            "difference": observed[task],
            "ci95_lower": float(np.quantile(draws[task], 0.025)),
            "ci95_upper": float(np.quantile(draws[task], 0.975)),
            "n_examples": len(first[task]),
            "paired_randomization_p_value": randomization[task],
        }
    macro_draw = np.vstack([draws[task] for task in sorted(draws)]).mean(axis=0)
    result["macro_average"] = {
        "difference": float(np.mean(list(observed.values()))),
        "ci95_lower": float(np.quantile(macro_draw, 0.025)),
        "ci95_upper": float(np.quantile(macro_draw, 0.975)),
        "n_examples": sum(len(first[task]) for task in first),
        "paired_randomization_p_value": randomization["macro_average"],
    }
    return result


def flatten_paired_comparison(
    first_label: str, second_label: str, comparison_type: str,
    paired: dict,
) -> list[dict]:
    rows = []
    for task, stat in sorted(paired.items()):
        lower, upper = stat["ci95_lower"], stat["ci95_upper"]
        rows.append({
            "comparison_type": comparison_type,
            "first_label": first_label, "second_label": second_label,
            "difference_definition": "first_label minus second_label accuracy",
            "task": task,
            "metric": (
                "task_macro_average" if task == "macro_average"
                else TASK_METRICS[task]
            ),
            "accuracy_difference": stat["difference"],
            "ci95_lower": lower, "ci95_upper": upper,
            "significant_95pct": bool(lower > 0 or upper < 0),
            "favored_label_if_significant": (
                first_label if lower > 0 else second_label if upper < 0 else ""
            ),
            "n_examples": stat["n_examples"],
            "paired_randomization_p_value": stat.get(
                "paired_randomization_p_value", ""
            ),
        })
    return rows


def _comparison_is_primary(row: dict) -> bool:
    """Pre-declared small confirmatory family for the downstream milestone."""
    if row["task"] != "macro_average":
        return False
    if row["comparison_type"] == "target6_selector_attribution":
        return True
    if row["comparison_type"] == "certified_hybrid_attribution":
        return True
    return (
        row["comparison_type"] == "checkpoint_vs_baseline"
        and row["first_label"] ==
        "rmsnorm_alloc__ellipsoid_rank__p95__target6"
    )


def _annotate_multiplicity(rows: list[dict]) -> None:
    for row in rows:
        primary = _comparison_is_primary(row)
        row["comparison_scope"] = "primary" if primary else "exploratory"
        row["multiplicity_family"] = (
            "primary_macro" if primary else "exploratory_all_other"
        )
    apply_multiplicity_adjustments(rows)


def _identities_manifest(loaded: dict, baseline_label: str) -> dict:
    baseline = loaded[baseline_label]
    result = {}
    for task, identities in sorted(baseline["identities"].items()):
        rows = [
            {"doc_id": value[0], "doc_hash": value[1], "target_hash": value[2]}
            for value in identities
        ]
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        result[task] = {
            "count": len(rows), "sha256": hashlib.sha256(encoded).hexdigest(),
            "identifiers": rows,
        }
    return result


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    protocol = payload.get("paper_v3_protocol")
    if not protocol:
        raise ValueError(f"missing paper_v3_protocol: {path}")
    protocol = dict(protocol)
    stable_config_hash = task_config_sha256(payload.get("configs", {}))
    recorded_version = protocol.get("task_configs_fingerprint")
    if recorded_version == FINGERPRINT_VERSION:
        if protocol.get("task_configs_sha256") != stable_config_hash:
            raise ValueError(f"stable task config hash mismatch: {path}")
    else:
        protocol.setdefault(
            "task_configs_process_raw_sha256",
            protocol.get("task_configs_sha256", ""),
        )
    protocol["task_configs_sha256"] = stable_config_hash
    protocol["task_configs_fingerprint"] = FINGERPRINT_VERSION
    values, samples, identities = {}, {}, {}
    for task, metric in TASK_METRICS.items():
        if task not in payload.get("results", {}):
            continue
        values[task] = _metric_value(payload["results"][task], metric)
        task_samples = payload.get("samples", {}).get(task, [])
        if not task_samples:
            raise ValueError(f"log_samples missing for {task}: {path}")
        samples[task] = [_sample_value(row, metric) for row in task_samples]
        identities[task] = [_sample_identity(row) for row in task_samples]
    if not values:
        raise ValueError(f"no requested tasks found in {path}")
    return {"payload": payload, "protocol": protocol, "values": values,
            "samples": samples, "identities": identities}


def load_tokenizer_audit_registry(paths: list[str]) -> dict[str, dict]:
    """Load passing tokenizer audits keyed by their exact file digest."""
    registry = {}
    for path in paths:
        digest = file_sha256(path)
        if digest in registry:
            raise ValueError(f"duplicate tokenizer audit SHA-256: {path}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        decision = payload.get("decision", {})
        if decision.get("audit_passed_for_downstream") is not True:
            raise ValueError(f"tokenizer audit did not pass: {path}")
        use_fix = decision.get("use_fix_mistral_regex_for_future_evaluation")
        if not isinstance(use_fix, bool):
            raise ValueError(f"tokenizer audit has no resolved policy: {path}")
        registry[digest] = {
            "path": os.path.realpath(path),
            "payload": payload,
            "selected_tokenizer_mode": "fixed" if use_fix else "current",
            "fix_mistral_regex": use_fix,
            "sources": {
                source.get("label"): source
                for source in payload.get("sources", [])
                if source.get("label")
            },
        }
    return registry


def validate_tokenizer_audit_coverage(
    loaded: dict[str, dict], registry: dict[str, dict],
) -> dict:
    """Verify each run against the exact tokenizer audit it recorded."""
    used = {}
    for label, run in loaded.items():
        protocol = run["protocol"]
        digest = protocol.get("tokenizer_audit_sha256", "")
        audit = registry.get(digest)
        if audit is None:
            raise ValueError(
                f"no supplied tokenizer audit matches {label}: sha256={digest!r}"
            )
        if protocol.get("selected_tokenizer_mode") != audit[
            "selected_tokenizer_mode"
        ]:
            raise ValueError(f"tokenizer mode differs from audit for {label}")
        if protocol.get("fix_mistral_regex") != audit["fix_mistral_regex"]:
            raise ValueError(f"tokenizer regex policy differs from audit for {label}")
        source = audit["sources"].get(label)
        if source is None:
            raise ValueError(f"tokenizer audit does not cover checkpoint {label}")
        expected_files = source.get("tokenizer_files_combined_sha256")
        if protocol.get("tokenizer_files_combined_sha256") != expected_files:
            raise ValueError(f"tokenizer file hash differs from audit for {label}")
        used[label] = {
            "tokenizer_audit_sha256": digest,
            "tokenizer_audit_path": audit["path"],
            "selected_tokenizer_mode": audit["selected_tokenizer_mode"],
            "fix_mistral_regex": audit["fix_mistral_regex"],
            "tokenizer_files_combined_sha256": expected_files,
        }
    return used


def _write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Checkpoint | Task | Metric | Accuracy | Delta vs baseline | Paired 95% CI | Significant |\n")
        handle.write("|---|---|---|---:|---:|---|---|\n")
        for row in rows:
            ci = "—" if row["ci95_lower"] == "" else (
                f"[{float(row['ci95_lower']):.4f}, {float(row['ci95_upper']):.4f}]"
            )
            handle.write(
                f"| {row['label']} | {row['task']} | {row['metric']} | "
                f"{float(row['accuracy']):.4f} | {float(row['absolute_change']):+.4f} | "
                f"{ci} | {row['significant_95pct']} |\n"
            )


def _latex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lllrrrl}\n\\toprule\n")
        handle.write("Checkpoint & Task & Metric & Accuracy & $\\Delta$ & 95\\% CI & Sig. \\\\\n\\midrule\n")
        for row in rows:
            ci = "--" if row["ci95_lower"] == "" else (
                f"[{float(row['ci95_lower']):.4f}, {float(row['ci95_upper']):.4f}]"
            )
            vals = [row["label"], row["task"], row["metric"],
                    f"{float(row['accuracy']):.4f}",
                    f"{float(row['absolute_change']):+.4f}", ci,
                    row["significant_95pct"]]
            handle.write(" & ".join(_latex(v) for v in vals) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def _write_paired_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Scope | Type | First | Second | Task | Accuracy difference | Paired 95% CI | Randomization p | Holm p | BH p |\n")
        handle.write("|---|---|---|---|---|---:|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['comparison_scope']} | {row['comparison_type']} | "
                f"{row['first_label']} | "
                f"{row['second_label']} | {row['task']} | "
                f"{float(row['accuracy_difference']):+.4f} | "
                f"[{float(row['ci95_lower']):.4f}, {float(row['ci95_upper']):.4f}] | "
                f"{float(row['paired_randomization_p_value']):.4g} | "
                f"{float(row['holm_adjusted_p_value']):.4g} | "
                f"{float(row['bh_adjusted_p_value']):.4g} |\n"
            )


def _write_paired_latex(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lllllrrrr}\n\\toprule\n")
        handle.write("Scope & Type & First & Second & Task & $\\Delta$acc & 95\\% CI & Holm $p$ & BH $p$ \\\\\n\\midrule\n")
        for row in rows:
            values = (
                row["comparison_scope"], row["comparison_type"],
                row["first_label"], row["second_label"],
                row["task"], f"{float(row['accuracy_difference']):+.4f}",
                f"[{float(row['ci95_lower']):.4f}, {float(row['ci95_upper']):.4f}]",
                f"{float(row['holm_adjusted_p_value']):.4g}",
                f"{float(row['bh_adjusted_p_value']):.4g}",
            )
            handle.write(" & ".join(_latex(value) for value in values) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def summarize(
    specs: list[dict], run_dir: str | list[str], n_resamples: int,
    *, bootstrap_seed: int = 42,
    tokenizer_audits: dict[str, dict] | None = None,
) -> tuple:
    run_dirs = [run_dir] if isinstance(run_dir, str) else run_dir
    loaded = {}
    for spec in specs:
        candidates = [os.path.join(directory, spec["label"], "lm_eval_results.json")
                      for directory in run_dirs]
        matches = [path for path in candidates if os.path.isfile(path)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one result for {spec['label']} across {run_dirs}; "
                f"found {matches}"
            )
        path = matches[0]
        loaded[spec["label"]] = load_run(path)
    audit_hashes = {
        run["protocol"].get("tokenizer_audit_sha256", "")
        for run in loaded.values()
    }
    if tokenizer_audits is None:
        if len(audit_hashes) != 1:
            raise ValueError(
                "multiple tokenizer audits are present; supply each with "
                "--tokenizer-audit"
            )
        tokenizer_coverage = {}
    else:
        tokenizer_coverage = validate_tokenizer_audit_coverage(
            loaded, tokenizer_audits
        )
    baseline_label = "baseline_unpruned"
    baseline = loaded[baseline_label]
    protocol_keys = (
        "harness", "tasks", "task_versions", "num_fewshot", "batch_size", "dtype",
        "seed_python", "seed_numpy", "seed_torch", "seed_fewshot",
        "apply_chat_template", "tokenizer_class", "source_model_revision",
        "tokenizer_revision", "selected_tokenizer_mode", "fix_mistral_regex",
        "trust_dataset_code", "dataset_code_tasks",
        "tokenizer_files_combined_sha256", "task_configs_sha256",
        "task_configs_fingerprint",
    )
    rows, comparisons, paired_rows = [], [], []
    for spec in specs:
        label = spec["label"]
        run = loaded[label]
        for key in protocol_keys:
            if run["protocol"].get(key) != baseline["protocol"].get(key):
                raise ValueError(f"protocol mismatch {label}: {key}")
        common_tasks = set(run["values"])
        if common_tasks != set(baseline["values"]):
            raise ValueError(f"task set mismatch for {label}")
        if label == baseline_label:
            paired = {}
        else:
            for task in common_tasks:
                if run["identities"][task] != baseline["identities"][task]:
                    raise ValueError(f"paired example identities differ: {label} {task}")
            paired = paired_bootstrap_accuracy(
                run["samples"], baseline["samples"],
                n_resamples=n_resamples, seed=bootstrap_seed,
            )
        for task in sorted(common_tasks):
            stat = paired.get(task, {})
            lower, upper = stat.get("ci95_lower", ""), stat.get("ci95_upper", "")
            significant = "N/A" if lower == "" else (
                "yes" if lower > 0 or upper < 0 else "no"
            )
            rows.append({
                "label": label, "target_pct": spec["target_pct"], "task": task,
                "metric": TASK_METRICS[task], "accuracy": run["values"][task],
                "baseline_accuracy": baseline["values"][task],
                "absolute_change": run["values"][task] - baseline["values"][task],
                "ci95_lower": lower, "ci95_upper": upper,
                "significant_95pct": significant,
                "n_examples": len(run["samples"][task]),
            })
        macro = float(np.mean(list(run["values"].values())))
        base_macro = float(np.mean(list(baseline["values"].values())))
        macro_stat = paired.get("macro_average", {})
        rows.append({
            "label": label, "target_pct": spec["target_pct"],
            "task": "macro_average", "metric": "task_macro_average",
            "accuracy": macro, "baseline_accuracy": base_macro,
            "absolute_change": macro - base_macro,
            "ci95_lower": macro_stat.get("ci95_lower", ""),
            "ci95_upper": macro_stat.get("ci95_upper", ""),
            "significant_95pct": (
                "N/A" if not macro_stat else
                ("yes" if macro_stat["ci95_lower"] > 0 or
                 macro_stat["ci95_upper"] < 0 else "no")
            ),
            "n_examples": sum(len(v) for v in run["samples"].values()),
        })
        if paired:
            comparisons.append({"label": label, "versus": baseline_label,
                                "paired": paired})
            paired_rows.extend(flatten_paired_comparison(
                label, baseline_label, "checkpoint_vs_baseline", paired
            ))

    target6 = [spec for spec in specs if int(round(float(spec["target_pct"]))) == 6]
    ellipsoid = [spec for spec in target6 if spec.get("ranking_source") ==
                 "rmsnorm_ellipsoid_bound"]
    competitors = [spec for spec in target6 if spec.get("ranking_source") in
                   ("activation_score", "rmsnorm_bound", "down_norm")]
    if competitors:
        if len(ellipsoid) != 1:
            raise ValueError("target-6 selector attribution requires one ellipsoid checkpoint")
        first_label = ellipsoid[0]["label"]
        first = loaded[first_label]
        for spec in competitors:
            second_label = spec["label"]
            second = loaded[second_label]
            if set(first["samples"]) != set(second["samples"]):
                raise ValueError(f"selector task sets differ: {first_label} {second_label}")
            for task in first["samples"]:
                if first["identities"][task] != second["identities"][task]:
                    raise ValueError(
                        f"selector paired identities differ: {task} "
                        f"{first_label} {second_label}"
                    )
            paired = paired_bootstrap_accuracy(
                first["samples"], second["samples"],
                n_resamples=n_resamples, seed=bootstrap_seed,
            )
            comparisons.append({
                "label": first_label, "versus": second_label,
                "comparison_type": "target6_selector_attribution", "paired": paired,
            })
            paired_rows.extend(flatten_paired_comparison(
                first_label, second_label, "target6_selector_attribution", paired
            ))
    # Explicit, predeclared hybrid references.  The manifest controls this
    # family so the analysis cannot choose comparisons after seeing accuracy.
    for spec in specs:
        references = list(spec.get("paired_reference_labels", []))
        if not references:
            continue
        first_label = spec["label"]
        first = loaded[first_label]
        for second_label in references:
            if second_label not in loaded:
                raise ValueError(
                    f"hybrid paired reference is absent: {first_label} vs {second_label}"
                )
            second = loaded[second_label]
            if set(first["samples"]) != set(second["samples"]):
                raise ValueError(f"hybrid task sets differ: {first_label} {second_label}")
            for task in first["samples"]:
                if first["identities"][task] != second["identities"][task]:
                    raise ValueError(
                        f"hybrid paired identities differ: {task} "
                        f"{first_label} {second_label}"
                    )
            paired = paired_bootstrap_accuracy(
                first["samples"], second["samples"],
                n_resamples=n_resamples, seed=bootstrap_seed,
            )
            comparisons.append({
                "label": first_label, "versus": second_label,
                "comparison_type": "certified_hybrid_attribution", "paired": paired,
            })
            paired_rows.extend(flatten_paired_comparison(
                first_label, second_label, "certified_hybrid_attribution", paired
            ))

    # The compression-curve checkpoints share examples and protocol, so audit
    # adjacent budgets directly. These are exploratory comparisons because the
    # 2%-versus-4% question was raised after observing the point estimates.
    curve_by_target = {}
    for spec in specs:
        if (
            spec.get("allocation_source") == "rmsnorm_bound"
            and spec.get("ranking_source") == "down_norm"
        ):
            target = int(round(float(spec["target_pct"])))
            if target in curve_by_target:
                raise ValueError(f"duplicate pure down-norm target {target}")
            curve_by_target[target] = spec
    if len(curve_by_target) > 1:
        targets = sorted(curve_by_target)
        for lower_target, upper_target in zip(targets, targets[1:]):
            lower_label = curve_by_target[lower_target]["label"]
            upper_label = curve_by_target[upper_target]["label"]
            lower = loaded[lower_label]
            upper = loaded[upper_label]
            if set(lower["samples"]) != set(upper["samples"]):
                raise ValueError(
                    f"adjacent-budget task sets differ: {lower_label} {upper_label}"
                )
            for task in lower["samples"]:
                if lower["identities"][task] != upper["identities"][task]:
                    raise ValueError(
                        f"adjacent-budget paired identities differ: {task} "
                        f"{lower_label} {upper_label}"
                    )
            paired = paired_bootstrap_accuracy(
                upper["samples"], lower["samples"],
                n_resamples=n_resamples, seed=bootstrap_seed,
            )
            comparisons.append({
                "label": upper_label, "versus": lower_label,
                "comparison_type": "compression_curve_adjacent_budget",
                "paired": paired,
            })
            paired_rows.extend(flatten_paired_comparison(
                upper_label, lower_label,
                "compression_curve_adjacent_budget", paired,
            ))
    _annotate_multiplicity(paired_rows)
    audit = {
        "schema_version": 1,
        "macro_interval_verified_task_stratified_paired_bootstrap": True,
        "bootstrap": {
            "seed": bootstrap_seed, "replicates": n_resamples,
            "pairing_unit": "lm-eval example identity within task",
            "stratification": "independent paired resampling within each task",
            "macro_estimand": "equal-weight mean of task accuracy differences",
            "confidence_interval": "percentile 2.5% and 97.5%",
        },
        "paired_randomization": {
            "seed": bootstrap_seed + 1_000_003, "replicates": n_resamples,
            "method": "two-sided paired sign-flip within task",
        },
        "multiple_testing": {
            "primary_definition": (
                "target-6 primary checkpoint versus baseline macro; target-6 "
                "ellipsoid versus matched selector comparator macros; and "
                "predeclared certified-hybrid versus endpoint macros"
            ),
            "exploratory_definition": "all task-level and remaining macro comparisons",
            "adjustments": ["Holm family-wise error", "Benjamini-Hochberg FDR"],
            "adjusted_within": "multiplicity_family",
        },
        "task_versions": baseline["protocol"].get("task_versions", {}),
        "compression_curve_adjacent_budget_comparisons": (
            "exploratory; difference is higher target minus immediately lower "
            "target; the 2%-versus-4% question followed inspection of point estimates"
        ),
        "example_identifiers": _identities_manifest(loaded, baseline_label),
        "tokenizer_audit_coverage": tokenizer_coverage,
        "tokenizer_audit_sha256_values": sorted(audit_hashes),
    }
    return rows, comparisons, paired_rows, baseline["protocol"], audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer-audit", action="append", default=[],
        help="Passing tokenizer audit JSON; repeat when cohorts use different audits",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.checkpoint_manifest, encoding="utf-8") as handle:
        specs = json.load(handle)
    tokenizer_audits = (
        load_tokenizer_audit_registry(args.tokenizer_audit)
        if args.tokenizer_audit else None
    )
    rows, comparisons, paired_rows, protocol, audit = summarize(
        specs, args.run_dir, args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        tokenizer_audits=tokenizer_audits,
    )
    if args.dry_run:
        print(
            f"[downstream-summary] DRY RUN: rows={len(rows)} "
            f"paired_rows={len(paired_rows)} primary="
            f"{sum(row['comparison_scope'] == 'primary' for row in paired_rows)}"
        )
        return
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "downstream_benchmark_table.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "downstream_benchmark_table.json"),
              "x", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "rows": rows,
                   "paired_comparisons": comparisons,
                   "paired_comparison_rows": paired_rows,
                   "statistical_audit": audit}, handle, indent=2)
    with open(os.path.join(args.output_dir, "downstream_statistical_audit.json"),
              "x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    paired_csv = os.path.join(args.output_dir, "downstream_paired_comparisons.csv")
    with open(paired_csv, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)
    _write_markdown(os.path.join(args.output_dir, "downstream_benchmark_table.md"), rows)
    _write_latex(os.path.join(args.output_dir, "downstream_benchmark_table.tex"), rows)
    _write_paired_markdown(
        os.path.join(args.output_dir, "downstream_paired_comparisons.md"), paired_rows
    )
    _write_paired_latex(
        os.path.join(args.output_dir, "downstream_paired_comparisons.tex"), paired_rows
    )
    print(f"[downstream-summary] OK: {len(rows)} rows; {csv_path}")


if __name__ == "__main__":
    main()
