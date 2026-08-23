#!/usr/bin/env python3
"""Collect Version 3 systems JSON files into CSV/JSON/Markdown/LaTeX."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


FIELDS = [
    "label", "batch_size", "prompt_length_tokens", "dtype",
    "checkpoint_storage_bytes", "checkpoint_storage_gb_decimal",
    "checkpoint_storage_gib_binary", "load_time_seconds", "successful_load",
    "after_load_allocated_bytes_total", "peak_inference_allocated_bytes_total",
    "after_load_allocated_gib_total", "peak_inference_allocated_gib_total",
    "prefill_latency_mean_ms",
    "prefill_latency_median_ms", "prefill_latency_stdev_ms",
    "prefill_latency_median_ci95_lower_ms",
    "prefill_latency_median_ci95_upper_ms",
    "prefill_tokens_per_second_median",
    "prefill_tokens_per_second_median_ci95_lower",
    "prefill_tokens_per_second_median_ci95_upper",
    "decode_latency_per_token_mean_ms", "decode_latency_per_token_median_ms",
    "decode_latency_per_token_stdev_ms", "decode_tokens_per_second_median",
    "decode_run_latency_mean_ms", "decode_run_latency_median_ms",
    "decode_run_latency_stdev_ms", "decode_run_tokens_per_second_median",
    "decode_latency_per_token_median_ci95_lower_ms",
    "decode_latency_per_token_median_ci95_upper_ms",
    "decode_tokens_per_second_median_ci95_lower",
    "decode_tokens_per_second_median_ci95_upper",
    "prefill_latency_sample_count", "decode_latency_sample_count",
    "decode_step_sample_count",
    "uncertainty_method", "uncertainty_resamples", "uncertainty_seed",
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
    "prefill_latency_reduction_ci95_lower_pct",
    "prefill_latency_reduction_ci95_upper_pct",
    "prefill_throughput_gain_ci95_lower_pct",
    "prefill_throughput_gain_ci95_upper_pct",
    "decode_latency_reduction_ci95_lower_pct",
    "decode_latency_reduction_ci95_upper_pct",
    "decode_throughput_gain_ci95_lower_pct",
    "decode_throughput_gain_ci95_upper_pct",
]


def _pct_reduction(baseline: float, current: float) -> float:
    return 100.0 * (baseline - current) / baseline


def _pct_gain(baseline: float, current: float) -> float:
    return 100.0 * (current - baseline) / baseline


def _bootstrap_medians(
    values: list[float], *, resamples: int, seed: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("systems uncertainty requires at least two finite samples")
    rng = np.random.default_rng(seed)
    result = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        result[start:stop] = np.median(array[indices], axis=1)
    return result


def _interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _case_uncertainty(case: dict, *, resamples: int, seed: int) -> dict:
    prefill = [float(value) for value in case.get("prefill_latency_samples_ms", [])]
    decode_steps = [
        float(value) for value in case.get("decode_latency_per_token_samples_ms", [])
    ]
    repetitions = int(case["timed_repetitions"])
    tokens_per_repetition = int(case["decode_tokens_per_repetition"])
    expected_steps = repetitions * tokens_per_repetition
    if len(decode_steps) != expected_steps:
        raise ValueError(
            "decode samples cannot be reconstructed into timed repetitions: "
            f"expected={expected_steps} observed={len(decode_steps)}"
        )
    decode = np.median(
        np.asarray(decode_steps, dtype=np.float64).reshape(
            repetitions, tokens_per_repetition
        ),
        axis=1,
    ).tolist()
    prefill_draws = _bootstrap_medians(prefill, resamples=resamples, seed=seed)
    decode_draws = _bootstrap_medians(decode, resamples=resamples, seed=seed + 1)
    prefill_ci = _interval(prefill_draws)
    decode_ci = _interval(decode_draws)
    prefill_factor = int(case["batch_size"]) * int(case["prompt_length_tokens"]) * 1000.0
    decode_factor = int(case["batch_size"]) * 1000.0
    prefill_tps_ci = _interval(prefill_factor / prefill_draws)
    decode_tps_ci = _interval(decode_factor / decode_draws)
    return {
        "prefill_latency_median_ci95_lower_ms": prefill_ci[0],
        "prefill_latency_median_ci95_upper_ms": prefill_ci[1],
        "prefill_tokens_per_second_median_ci95_lower": prefill_tps_ci[0],
        "prefill_tokens_per_second_median_ci95_upper": prefill_tps_ci[1],
        "decode_latency_per_token_median_ci95_lower_ms": decode_ci[0],
        "decode_latency_per_token_median_ci95_upper_ms": decode_ci[1],
        "decode_tokens_per_second_median_ci95_lower": decode_tps_ci[0],
        "decode_tokens_per_second_median_ci95_upper": decode_tps_ci[1],
        "decode_run_latency_mean_ms": float(np.mean(decode)),
        "decode_run_latency_median_ms": float(np.median(decode)),
        "decode_run_latency_stdev_ms": float(np.std(decode, ddof=1)),
        "decode_run_tokens_per_second_median": (
            decode_factor / float(np.median(decode))
        ),
        "prefill_latency_sample_count": len(prefill),
        "decode_latency_sample_count": len(decode),
        "decode_step_sample_count": len(decode_steps),
        "uncertainty_method": (
            "percentile bootstrap over timed repetitions; each decode repetition "
            "is summarized by its median per-token latency; baseline/pruned runs "
            "were sequential and independently resampled"
        ),
        "uncertainty_resamples": resamples,
        "uncertainty_seed": seed,
    }


def _relative_bootstrap_intervals(
    baseline: dict, current: dict, *, resamples: int, seed: int,
) -> dict:
    base_prefill = _bootstrap_medians(
        baseline["prefill"], resamples=resamples, seed=seed
    )
    current_prefill = _bootstrap_medians(
        current["prefill"], resamples=resamples, seed=seed + 1
    )
    base_decode = _bootstrap_medians(
        baseline["decode"], resamples=resamples, seed=seed + 2
    )
    current_decode = _bootstrap_medians(
        current["decode"], resamples=resamples, seed=seed + 3
    )
    prefill_latency = 100.0 * (base_prefill - current_prefill) / base_prefill
    prefill_throughput = 100.0 * (base_prefill / current_prefill - 1.0)
    decode_latency = 100.0 * (base_decode - current_decode) / base_decode
    decode_throughput = 100.0 * (base_decode / current_decode - 1.0)
    result = {}
    for prefix, values in (
        ("prefill_latency_reduction", prefill_latency),
        ("prefill_throughput_gain", prefill_throughput),
        ("decode_latency_reduction", decode_latency),
        ("decode_throughput_gain", decode_throughput),
    ):
        lower, upper = _interval(values)
        result[f"{prefix}_ci95_lower_pct"] = lower
        result[f"{prefix}_ci95_upper_pct"] = upper
    return result


def collect(
    run_dir: str | list[str], *, bootstrap_resamples: int = 0,
    bootstrap_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    run_dirs = [run_dir] if isinstance(run_dir, str) else run_dir
    rows, raw = [], []
    samples_by_key = {}
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
            row["after_load_allocated_gib_total"] = (
                int(payload["after_load_allocated_bytes_total"]) / (1024 ** 3)
            )
            row["peak_inference_allocated_gib_total"] = (
                int(payload["peak_inference_allocated_bytes_total"]) / (1024 ** 3)
            )
            key = (
                payload["label"], int(case["batch_size"]),
                int(case["prompt_length_tokens"]),
            )
            samples_by_key[key] = {
                "prefill": list(case.get("prefill_latency_samples_ms", [])),
            }
            if bootstrap_resamples:
                case_seed = (
                    bootstrap_seed + int(case["batch_size"]) * 100_003
                    + int(case["prompt_length_tokens"]) * 101
                )
                row.update(_case_uncertainty(
                    case, resamples=bootstrap_resamples, seed=case_seed,
                ))
                decode_steps = list(
                    case.get("decode_latency_per_token_samples_ms", [])
                )
                decode_repetitions = int(case["timed_repetitions"])
                decode_tokens = int(case["decode_tokens_per_repetition"])
                samples_by_key[key]["decode"] = np.median(
                    np.asarray(decode_steps, dtype=np.float64).reshape(
                        decode_repetitions, decode_tokens
                    ),
                    axis=1,
                ).tolist()
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
    comparison_fields = [
        field for field in FIELDS
        if field.endswith("_vs_baseline_pct") or field.endswith("_ci95_lower_pct")
        or field.endswith("_ci95_upper_pct")
    ]
    for row in rows:
        for field in comparison_fields:
            row[field] = ""
        baseline_key = (
            int(row["batch_size"]), int(row["prompt_length_tokens"])
        )
        baseline = baseline_rows.get(baseline_key)
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
                "decode_run_latency_median_ms", _pct_reduction
            ),
            "decode_throughput_gain_vs_baseline_pct": (
                "decode_run_tokens_per_second_median", _pct_gain
            ),
        }
        for output, (source, function) in pairs.items():
            row[output] = function(float(baseline[source]), float(row[source]))
        if bootstrap_resamples and row["label"] != "baseline_unpruned":
            key = (row["label"], *baseline_key)
            base_key = ("baseline_unpruned", *baseline_key)
            row.update(_relative_bootstrap_intervals(
                samples_by_key[base_key], samples_by_key[key],
                resamples=bootstrap_resamples,
                seed=(bootstrap_seed + baseline_key[0] * 1_000_003
                      + baseline_key[1] * 1009),
            ))
    return rows, raw


def _md(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Checkpoint | B | Prompt | Load HBM GiB | Peak HBM GiB | Prefill median ms [95% CI] | Prefill mean ± SD ms | Prefill tok/s [95% CI] | Prefill gain [95% CI] | Decode median ms [95% CI] | Decode mean ± SD ms | Decode tok/s [95% CI] | Decode gain [95% CI] |\n")
        handle.write("|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            prefill_gain = r["prefill_throughput_gain_vs_baseline_pct"]
            decode_gain = r["decode_throughput_gain_vs_baseline_pct"]
            prefill_gain_text = "—" if prefill_gain == "" else (
                f"{float(prefill_gain):+.2f}% "
                f"[{float(r['prefill_throughput_gain_ci95_lower_pct']):+.2f}, "
                f"{float(r['prefill_throughput_gain_ci95_upper_pct']):+.2f}]"
            )
            decode_gain_text = "—" if decode_gain == "" else (
                f"{float(decode_gain):+.2f}% "
                f"[{float(r['decode_throughput_gain_ci95_lower_pct']):+.2f}, "
                f"{float(r['decode_throughput_gain_ci95_upper_pct']):+.2f}]"
            )
            handle.write(
                f"| {r['label']} | {r['batch_size']} | {r['prompt_length_tokens']} | "
                f"{float(r['after_load_allocated_gib_total']):.2f} | "
                f"{float(r['peak_inference_allocated_gib_total']):.2f} | "
                f"{float(r['prefill_latency_median_ms']):.2f} "
                f"[{float(r['prefill_latency_median_ci95_lower_ms']):.2f}, "
                f"{float(r['prefill_latency_median_ci95_upper_ms']):.2f}] | "
                f"{float(r['prefill_latency_mean_ms']):.2f} ± "
                f"{float(r['prefill_latency_stdev_ms']):.2f} | "
                f"{float(r['prefill_tokens_per_second_median']):.1f} "
                f"[{float(r['prefill_tokens_per_second_median_ci95_lower']):.1f}, "
                f"{float(r['prefill_tokens_per_second_median_ci95_upper']):.1f}] | "
                f"{prefill_gain_text} | "
                f"{float(r['decode_run_latency_median_ms']):.2f} "
                f"[{float(r['decode_latency_per_token_median_ci95_lower_ms']):.2f}, "
                f"{float(r['decode_latency_per_token_median_ci95_upper_ms']):.2f}] | "
                f"{float(r['decode_run_latency_mean_ms']):.2f} ± "
                f"{float(r['decode_run_latency_stdev_ms']):.2f} | "
                f"{float(r['decode_run_tokens_per_second_median']):.1f} "
                f"[{float(r['decode_tokens_per_second_median_ci95_lower']):.1f}, "
                f"{float(r['decode_tokens_per_second_median_ci95_upper']):.1f}] | "
                f"{decode_gain_text} |\n"
            )


def _tex(path: str, rows: list[dict]) -> None:
    esc = lambda v: str(v).replace("_", r"\_")
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        handle.write("Checkpoint & B & Prompt & Load GiB & Peak GiB & Prefill median [CI] & Decode median [CI] & Prefill gain [CI] & Decode gain [CI] \\\\\n\\midrule\n")
        for r in rows:
            prefill_gain = r["prefill_throughput_gain_vs_baseline_pct"]
            decode_gain = r["decode_throughput_gain_vs_baseline_pct"]
            values = [
                esc(r["label"]), r["batch_size"], r["prompt_length_tokens"],
                f"{float(r['after_load_allocated_gib_total']):.2f}",
                f"{float(r['peak_inference_allocated_gib_total']):.2f}",
                f"{float(r['prefill_latency_median_ms']):.2f} "
                f"[{float(r['prefill_latency_median_ci95_lower_ms']):.2f},"
                f"{float(r['prefill_latency_median_ci95_upper_ms']):.2f}]",
                f"{float(r['decode_run_latency_median_ms']):.2f} "
                f"[{float(r['decode_latency_per_token_median_ci95_lower_ms']):.2f},"
                f"{float(r['decode_latency_per_token_median_ci95_upper_ms']):.2f}]",
                "--" if prefill_gain == "" else
                f"{float(prefill_gain):+.2f} "
                f"[{float(r['prefill_throughput_gain_ci95_lower_pct']):+.2f},"
                f"{float(r['prefill_throughput_gain_ci95_upper_pct']):+.2f}]",
                "--" if decode_gain == "" else
                f"{float(decode_gain):+.2f} "
                f"[{float(r['decode_throughput_gain_ci95_lower_pct']):+.2f},"
                f"{float(r['decode_throughput_gain_ci95_upper_pct']):+.2f}]",
            ]
            handle.write(" & ".join(map(str, values)) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows, raw = collect(
        args.run_dir, bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "systems_benchmark_table.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "systems_benchmark_table.json"),
              "x", encoding="utf-8") as handle:
        json.dump({
            "schema_version": 2,
            "uncertainty": {
                "method": (
                    "within-run percentile bootstrap of medians; baseline and "
                    "pruned runs independently resampled because execution was sequential"
                ),
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "interleaved_trials": False,
            },
            "rows": rows, "raw_runs": raw,
        }, handle, indent=2)
    _md(os.path.join(args.output_dir, "systems_benchmark_table.md"), rows)
    _tex(os.path.join(args.output_dir, "systems_benchmark_table.tex"), rows)
    print(f"[systems-summary] OK: {len(rows)} rows; {csv_path}")


if __name__ == "__main__":
    main()
