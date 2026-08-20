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
    "checkpoint_storage_bytes", "load_time_seconds",
    "after_load_allocated_bytes_total", "peak_inference_allocated_bytes_total",
    "prefill_latency_median_ms", "prefill_latency_stdev_ms",
    "prefill_tokens_per_second_median", "decode_latency_per_token_median_ms",
    "decode_latency_per_token_stdev_ms", "decode_tokens_per_second_median",
    "warmup_repetitions", "timed_repetitions", "nvidia_smi",
    "torch_version", "cuda_runtime_version", "transformers_version",
    "inference_engine", "reduced_intermediate_dimensions_executed",
]


def collect(run_dir: str) -> tuple[list[dict], list[dict]]:
    rows, raw = [], []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "systems.json"))):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        raw.append(payload)
        if not payload.get("reduced_intermediate_dimensions_executed"):
            raise ValueError(f"reduced widths not confirmed: {path}")
        for case in payload["cases"]:
            row = {field: case.get(field, payload.get(field, "")) for field in FIELDS}
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no systems.json files beneath {run_dir}")
    return rows, raw


def _md(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Checkpoint | B | Prompt | Prefill ms | Prefill tok/s | Decode ms/tok | Decode tok/s | Load HBM GiB | Peak HBM GiB |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            handle.write(
                f"| {r['label']} | {r['batch_size']} | {r['prompt_length_tokens']} | "
                f"{float(r['prefill_latency_median_ms']):.2f} | "
                f"{float(r['prefill_tokens_per_second_median']):.1f} | "
                f"{float(r['decode_latency_per_token_median_ms']):.2f} | "
                f"{float(r['decode_tokens_per_second_median']):.1f} | "
                f"{int(r['after_load_allocated_bytes_total']) / 2**30:.2f} | "
                f"{int(r['peak_inference_allocated_bytes_total']) / 2**30:.2f} |\n"
            )


def _tex(path: str, rows: list[dict]) -> None:
    esc = lambda v: str(v).replace("_", r"\_")
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        handle.write("Checkpoint & B & Prompt & Prefill ms & Prefill tok/s & Decode ms & Decode tok/s & Load GiB & Peak GiB \\\\\n\\midrule\n")
        for r in rows:
            values = [esc(r["label"]), r["batch_size"], r["prompt_length_tokens"],
                      f"{float(r['prefill_latency_median_ms']):.2f}",
                      f"{float(r['prefill_tokens_per_second_median']):.1f}",
                      f"{float(r['decode_latency_per_token_median_ms']):.2f}",
                      f"{float(r['decode_tokens_per_second_median']):.1f}",
                      f"{int(r['after_load_allocated_bytes_total']) / 2**30:.2f}",
                      f"{int(r['peak_inference_allocated_bytes_total']) / 2**30:.2f}"]
            handle.write(" & ".join(map(str, values)) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
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
