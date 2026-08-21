#!/usr/bin/env python3
"""Paired downstream comparison of HEAPr to the identical baseline samples."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.summarize_paper_v3_downstream import (
    TASK_METRICS, _metric_value, _sample_identity, _sample_value,
    paired_bootstrap_accuracy,
)
from src.statistical_audit import apply_multiplicity_adjustments


def _extract(payload: dict) -> tuple[dict, dict, dict]:
    values, samples, identities = {}, {}, {}
    for task, metric in TASK_METRICS.items():
        if task not in payload.get("results", {}):
            continue
        values[task] = _metric_value(payload["results"][task], metric)
        task_samples = payload.get("samples", {}).get(task, [])
        samples[task] = [_sample_value(row, metric) for row in task_samples]
        identities[task] = [_sample_identity(row) for row in task_samples]
    return values, samples, identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--heapr-results", required=True)
    parser.add_argument("--heapr-protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    baseline = json.load(open(args.baseline_results, encoding="utf-8"))
    heapr = json.load(open(args.heapr_results, encoding="utf-8"))
    protocol = json.load(open(args.heapr_protocol, encoding="utf-8"))
    baseline_identity = baseline["paper_v3_protocol"]["harness"]
    baseline_identity = baseline_identity["git_revision"] or baseline_identity["package_version"]
    if baseline_identity != protocol["lm_eval_identity"]:
        raise ValueError("HEAPr and baseline lm-eval identities differ")
    if baseline.get("versions") != heapr.get("versions"):
        raise ValueError("HEAPr and baseline task versions differ")
    base_protocol = baseline["paper_v3_protocol"]
    heapr_eval_protocol = heapr["heapr_matched_protocol"]
    checks = {
        "num_fewshot": (heapr_eval_protocol["num_fewshot"], base_protocol["num_fewshot"]),
        "batch_size": (heapr_eval_protocol["batch_size"], base_protocol["batch_size"]),
        "seed": (heapr_eval_protocol["seed"], base_protocol["seed_python"]),
        "apply_chat_template": (
            heapr_eval_protocol["apply_chat_template"],
            base_protocol["apply_chat_template"],
        ),
        "tokenizer_class": (
            heapr_eval_protocol.get("tokenizer_class"),
            base_protocol.get("tokenizer_class"),
        ),
        "selected_tokenizer_mode": (
            heapr_eval_protocol.get("selected_tokenizer_mode"),
            base_protocol.get("selected_tokenizer_mode"),
        ),
        "fix_mistral_regex": (
            heapr_eval_protocol.get("fix_mistral_regex"),
            base_protocol.get("fix_mistral_regex"),
        ),
    }
    mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
    expected_dtype = f"torch.{base_protocol['dtype']}"
    if expected_dtype not in heapr_eval_protocol["model_parameter_dtypes"]:
        mismatches["dtype"] = (
            heapr_eval_protocol["model_parameter_dtypes"], expected_dtype
        )
    if mismatches:
        raise ValueError(f"HEAPr/baseline protocol settings differ: {mismatches}")
    base_values, base_samples, base_ids = _extract(baseline)
    heapr_values, heapr_samples, heapr_ids = _extract(heapr)
    if set(base_values) != set(heapr_values):
        raise ValueError("HEAPr and baseline task sets differ")
    for task in base_values:
        if base_ids[task] != heapr_ids[task]:
            raise ValueError(f"HEAPr paired sample identities differ for {task}")
    paired = paired_bootstrap_accuracy(
        heapr_samples, base_samples, n_resamples=args.bootstrap_resamples, seed=42
    )
    rows = []
    for task in sorted(base_values):
        stat = paired[task]
        rows.append({
            "task": task, "metric": TASK_METRICS[task],
            "baseline_accuracy": base_values[task],
            "heapr_accuracy": heapr_values[task],
            "heapr_minus_baseline": stat["difference"],
            "ci95_lower": stat["ci95_lower"], "ci95_upper": stat["ci95_upper"],
            "significant_95pct": stat["ci95_lower"] > 0 or stat["ci95_upper"] < 0,
            "paired_randomization_p_value": stat["paired_randomization_p_value"],
            "multiplicity_family": "heapr_vs_baseline_downstream",
            "n_examples": stat["n_examples"],
        })
    macro = paired["macro_average"]
    rows.append({
        "task": "macro_average", "metric": "task_macro_average",
        "baseline_accuracy": float(np.mean(list(base_values.values()))),
        "heapr_accuracy": float(np.mean(list(heapr_values.values()))),
        "heapr_minus_baseline": macro["difference"],
        "ci95_lower": macro["ci95_lower"], "ci95_upper": macro["ci95_upper"],
        "significant_95pct": macro["ci95_lower"] > 0 or macro["ci95_upper"] < 0,
        "paired_randomization_p_value": macro["paired_randomization_p_value"],
        "multiplicity_family": "heapr_vs_baseline_downstream",
        "n_examples": macro["n_examples"],
    })
    apply_multiplicity_adjustments(rows)
    os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "heapr_downstream_paired.csv"), "x",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "heapr_downstream_paired.json"), "x",
              encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "rows": rows,
                   "macro_average": paired["macro_average"]}, handle, indent=2)
    with open(os.path.join(args.output_dir, "heapr_downstream_paired.md"), "x",
              encoding="utf-8") as handle:
        handle.write("| Task | Metric | Baseline | HEAPr | Delta | Paired 95% CI | Significant |\n")
        handle.write("|---|---|---:|---:|---:|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['task']} | {row['metric']} | {row['baseline_accuracy']:.4f} | "
                f"{row['heapr_accuracy']:.4f} | {row['heapr_minus_baseline']:+.4f} | "
                f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | "
                f"{row['significant_95pct']} |\n"
            )
    with open(os.path.join(args.output_dir, "heapr_downstream_paired.tex"), "x",
              encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llrrrrl}\n\\toprule\nTask & Metric & Baseline & HEAPr & $\\Delta$ & 95\\% CI & Sig. \\\\\n\\midrule\n")
        for row in rows:
            task_tex = row["task"].replace("_", "\\_")
            metric_tex = row["metric"].replace("_", "\\_")
            handle.write(
                f"{task_tex} & {metric_tex} & "
                f"{row['baseline_accuracy']:.4f} & {row['heapr_accuracy']:.4f} & "
                f"{row['heapr_minus_baseline']:+.4f} & "
                f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] & "
                f"{row['significant_95pct']} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[heapr-compare] OK: {len(rows)} tasks; {args.output_dir}")


if __name__ == "__main__":
    main()
