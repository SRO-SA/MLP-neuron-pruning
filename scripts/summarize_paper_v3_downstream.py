#!/usr/bin/env python3
"""Create paired, paper-ready downstream tables from lm-eval sample logs."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


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
    result = {}
    for task in first:
        result[task] = {
            "difference": observed[task],
            "ci95_lower": float(np.quantile(draws[task], 0.025)),
            "ci95_upper": float(np.quantile(draws[task], 0.975)),
            "n_examples": len(first[task]),
        }
    macro_draw = np.vstack([draws[task] for task in sorted(draws)]).mean(axis=0)
    result["macro_average"] = {
        "difference": float(np.mean(list(observed.values()))),
        "ci95_lower": float(np.quantile(macro_draw, 0.025)),
        "ci95_upper": float(np.quantile(macro_draw, 0.975)),
        "n_examples": sum(len(first[task]) for task in first),
    }
    return result


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    protocol = payload.get("paper_v3_protocol")
    if not protocol:
        raise ValueError(f"missing paper_v3_protocol: {path}")
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


def summarize(specs: list[dict], run_dir: str, n_resamples: int) -> tuple:
    loaded = {}
    for spec in specs:
        path = os.path.join(run_dir, spec["label"], "lm_eval_results.json")
        loaded[spec["label"]] = load_run(path)
    baseline_label = "baseline_unpruned"
    baseline = loaded[baseline_label]
    protocol_keys = (
        "harness", "tasks", "num_fewshot", "batch_size", "dtype",
        "seed_python", "seed_numpy", "seed_torch", "seed_fewshot",
        "apply_chat_template", "tokenizer_class", "source_model_revision",
        "tokenizer_revision",
    )
    rows, comparisons = [], []
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
                n_resamples=n_resamples, seed=42,
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
    return rows, comparisons, baseline["protocol"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.checkpoint_manifest, encoding="utf-8") as handle:
        specs = json.load(handle)
    rows, comparisons, protocol = summarize(
        specs, args.run_dir, args.bootstrap_resamples
    )
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "downstream_benchmark_table.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "downstream_benchmark_table.json"),
              "x", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "rows": rows,
                   "paired_comparisons": comparisons}, handle, indent=2)
    _write_markdown(os.path.join(args.output_dir, "downstream_benchmark_table.md"), rows)
    _write_latex(os.path.join(args.output_dir, "downstream_benchmark_table.tex"), rows)
    print(f"[downstream-summary] OK: {len(rows)} rows; {csv_path}")


if __name__ == "__main__":
    main()
