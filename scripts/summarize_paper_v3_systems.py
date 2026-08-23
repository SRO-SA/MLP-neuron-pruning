#!/usr/bin/env python3
"""Collect Version 3 systems JSON files into CSV/JSON/Markdown/LaTeX."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


FIELDS = [
    "label", "batch_size", "prompt_length_tokens", "dtype",
    "checkpoint_storage_bytes", "checkpoint_storage_gb_decimal",
    "checkpoint_storage_gib_binary", "load_time_seconds", "successful_load",
    "after_load_allocated_bytes_total", "peak_inference_allocated_bytes_total",
    "prefill_latency_median_ms", "prefill_latency_stdev_ms",
    "prefill_tokens_per_second_median", "decode_latency_per_token_median_ms",
    "decode_latency_per_token_stdev_ms", "decode_tokens_per_second_median",
    "warmup_repetitions", "timed_repetitions", "nvidia_smi",
    "torch_version", "cuda_runtime_version", "transformers_version",
    "inference_engine", "reduced_intermediate_dimensions_executed",
    "runtime_moe_shapes_confirmed",
    "operator_profiler_enabled", "operator_profiler_widths_confirmed",
    "checkpoint_storage_reduction_vs_baseline_pct",
    "load_time_reduction_vs_baseline_pct",
    "load_hbm_reduction_vs_baseline_pct",
    "peak_hbm_reduction_vs_baseline_pct",
    "prefill_latency_reduction_vs_baseline_pct",
    "prefill_throughput_gain_vs_baseline_pct",
    "decode_latency_reduction_vs_baseline_pct",
    "decode_throughput_gain_vs_baseline_pct",
]


def _pct_reduction(baseline: float, current: float) -> float:
    return 100.0 * (baseline - current) / baseline


def _pct_gain(baseline: float, current: float) -> float:
    return 100.0 * (current - baseline) / baseline


def collect(run_dir: str | list[str]) -> tuple[list[dict], list[dict]]:
    run_dirs = [run_dir] if isinstance(run_dir, str) else run_dir
    rows, raw = [], []
    paths = sorted({path for directory in run_dirs for path in glob.glob(
        os.path.join(directory, "*", "systems.json")
    )})
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        raw.append(payload)
        if not payload.get("reduced_intermediate_dimensions_executed"):
            raise ValueError(f"reduced widths not confirmed: {path}")
        for case in payload["cases"]:
            row = {field: case.get(field, payload.get(field, "")) for field in FIELDS}
            storage = int(payload["checkpoint_storage_bytes"])
            row["checkpoint_storage_gb_decimal"] = storage / 1_000_000_000
            row["checkpoint_storage_gib_binary"] = storage / (1024 ** 3)
            evidence = payload.get("runtime_moe_execution_evidence", {})
            row["runtime_moe_shapes_confirmed"] = bool(
                evidence.get(
                    "all_moe_layers_executed",
                    evidence.get("all_packed_moe_layers_executed", False),
                )
            )
            operator = payload.get("operator_profile_evidence", {})
            row["operator_profiler_enabled"] = bool(operator.get("enabled", False))
            row["operator_profiler_widths_confirmed"] = bool(
                operator.get("enabled", False)
                and operator.get("all_expected_pruned_widths_observed", False)
            )
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no systems.json files beneath {run_dirs}")
    keys = [(row["label"], row["batch_size"], row["prompt_length_tokens"])
            for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate label/batch/prompt systems rows across run directories")
    baseline_rows = {
        (int(row["batch_size"]), int(row["prompt_length_tokens"])): row
        for row in rows if row["label"] == "baseline_unpruned"
    }
    comparison_fields = FIELDS[-8:]
    for row in rows:
        for field in comparison_fields:
            row[field] = ""
        baseline = baseline_rows.get(
            (int(row["batch_size"]), int(row["prompt_length_tokens"]))
        )
        if baseline is None:
            continue
        pairs = {
            "checkpoint_storage_reduction_vs_baseline_pct": (
                "checkpoint_storage_bytes", _pct_reduction
            ),
            "load_time_reduction_vs_baseline_pct": (
                "load_time_seconds", _pct_reduction
            ),
            "load_hbm_reduction_vs_baseline_pct": (
                "after_load_allocated_bytes_total", _pct_reduction
            ),
            "peak_hbm_reduction_vs_baseline_pct": (
                "peak_inference_allocated_bytes_total", _pct_reduction
            ),
            "prefill_latency_reduction_vs_baseline_pct": (
                "prefill_latency_median_ms", _pct_reduction
            ),
            "prefill_throughput_gain_vs_baseline_pct": (
                "prefill_tokens_per_second_median", _pct_gain
            ),
            "decode_latency_reduction_vs_baseline_pct": (
                "decode_latency_per_token_median_ms", _pct_reduction
            ),
            "decode_throughput_gain_vs_baseline_pct": (
                "decode_tokens_per_second_median", _pct_gain
            ),
        }
        for output, (source, function) in pairs.items():
            row[output] = function(float(baseline[source]), float(row[source]))
    return rows, raw


def _md(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Checkpoint | B | Prompt | Prefill ms | Prefill tok/s | Prefill gain | Decode ms/tok | Decode tok/s | Decode gain | Load HBM GiB | Peak HBM GiB |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            prefill_gain = r["prefill_throughput_gain_vs_baseline_pct"]
            decode_gain = r["decode_throughput_gain_vs_baseline_pct"]
            prefill_gain_text = "—" if prefill_gain == "" else f"{float(prefill_gain):+.2f}%"
            decode_gain_text = "—" if decode_gain == "" else f"{float(decode_gain):+.2f}%"
            handle.write(
                f"| {r['label']} | {r['batch_size']} | {r['prompt_length_tokens']} | "
                f"{float(r['prefill_latency_median_ms']):.2f} | "
                f"{float(r['prefill_tokens_per_second_median']):.1f} | "
                f"{prefill_gain_text} | "
                f"{float(r['decode_latency_per_token_median_ms']):.2f} | "
                f"{float(r['decode_tokens_per_second_median']):.1f} | "
                f"{decode_gain_text} | "
                f"{int(r['after_load_allocated_bytes_total']) / 2**30:.2f} | "
                f"{int(r['peak_inference_allocated_bytes_total']) / 2**30:.2f} |\n"
            )


def _tex(path: str, rows: list[dict]) -> None:
    esc = lambda v: str(v).replace("_", r"\_")
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n")
        handle.write("Checkpoint & B & Prompt & Prefill ms & Prefill tok/s & Gain \\% & Decode ms & Decode tok/s & Gain \\% & Load GiB & Peak GiB \\\\\n\\midrule\n")
        for r in rows:
            prefill_gain = r["prefill_throughput_gain_vs_baseline_pct"]
            decode_gain = r["decode_throughput_gain_vs_baseline_pct"]
            values = [esc(r["label"]), r["batch_size"], r["prompt_length_tokens"],
                      f"{float(r['prefill_latency_median_ms']):.2f}",
                      f"{float(r['prefill_tokens_per_second_median']):.1f}",
                      "--" if prefill_gain == "" else f"{float(prefill_gain):+.2f}",
                      f"{float(r['decode_latency_per_token_median_ms']):.2f}",
                      f"{float(r['decode_tokens_per_second_median']):.1f}",
                      "--" if decode_gain == "" else f"{float(decode_gain):+.2f}",
                      f"{int(r['after_load_allocated_bytes_total']) / 2**30:.2f}",
                      f"{int(r['peak_inference_allocated_bytes_total']) / 2**30:.2f}"]
            handle.write(" & ".join(map(str, values)) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows, raw = collect(args.run_dir)
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "systems_benchmark_table.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "systems_benchmark_table.json"),
              "x", encoding="utf-8") as handle:
        json.dump({"rows": rows, "raw_runs": raw}, handle, indent=2)
    _md(os.path.join(args.output_dir, "systems_benchmark_table.md"), rows)
    _tex(os.path.join(args.output_dir, "systems_benchmark_table.tex"), rows)
    print(f"[systems-summary] OK: {len(rows)} rows; {csv_path}")


if __name__ == "__main__":
    main()
