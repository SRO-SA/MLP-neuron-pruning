#!/usr/bin/env python3
"""Build a paper table comparing construction requirements and measured cost."""
from __future__ import annotations

import argparse
import csv
import json
import os

FIELDS = [
    "method", "calibration_dataset", "calibration_samples_or_tokens",
    "forward_passes", "backward_passes", "requires_gradients",
    "requires_activations", "wall_clock_seconds", "peak_gpu_memory_bytes",
    "comparison_scope",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--our-protocol", required=True)
    parser.add_argument("--heapr-protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    ours = json.load(open(args.our_protocol, encoding="utf-8"))
    heapr = json.load(open(args.heapr_protocol, encoding="utf-8"))
    rows = [{
        "method": ours["method"], "calibration_dataset": "none",
        "calibration_samples_or_tokens": 0,
        "forward_passes": ours["forward_passes_for_score_construction"],
        "backward_passes": ours["backward_passes_for_score_construction"],
        "requires_gradients": ours["requires_gradients"],
        "requires_activations": ours["requires_activations"],
        "wall_clock_seconds": ours[
            "wall_clock_seconds_including_model_load_and_bound_diagnostics"
        ], "peak_gpu_memory_bytes": ours["peak_incremental_gpu_memory_used_bytes_total"],
        "comparison_scope": ours["scope_note"],
    }, {
        "method": heapr["method"],
        "calibration_dataset": heapr["calibration_dataset"],
        "calibration_samples_or_tokens": heapr["calibration_samples"],
        "forward_passes": heapr["forward_passes_declared_by_method"],
        "backward_passes": heapr["backward_passes_declared_by_method"],
        "requires_gradients": heapr["requires_gradients"],
        "requires_activations": heapr["requires_activations"],
        "wall_clock_seconds": heapr["wall_clock_seconds"],
        "peak_gpu_memory_bytes": heapr["peak_incremental_gpu_memory_used_bytes_total"],
        "comparison_scope": heapr["comparability_note"],
    }]
    os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "method_construction_cost.csv"), "x",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "method_construction_cost.json"), "x",
              encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(os.path.join(args.output_dir, "method_construction_cost.md"), "x",
              encoding="utf-8") as handle:
        handle.write("| Method | Calibration | Forward | Backward | Gradients | Activations | Wall s | Peak GPU GiB |\n")
        handle.write("|---|---|---:|---:|---|---|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['method']} | {row['calibration_dataset']} "
                f"({row['calibration_samples_or_tokens']}) | {row['forward_passes']} | "
                f"{row['backward_passes']} | {row['requires_gradients']} | "
                f"{row['requires_activations']} | {float(row['wall_clock_seconds']):.1f} | "
                f"{int(row['peak_gpu_memory_bytes']) / 2**30:.2f} |\n"
            )
    with open(os.path.join(args.output_dir, "method_construction_cost.tex"), "x",
              encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nMethod & Calibration & Fwd & Bwd & Grad. & Wall s & Peak GiB \\\\\n\\midrule\n")
        for row in rows:
            method = row["method"].replace("_", r"\_")
            handle.write(
                f"{method} & {row['calibration_samples_or_tokens']} & "
                f"{row['forward_passes']} & {row['backward_passes']} & "
                f"{row['requires_gradients']} & {float(row['wall_clock_seconds']):.1f} & "
                f"{int(row['peak_gpu_memory_bytes']) / 2**30:.2f} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[method-cost] OK: {args.output_dir}")


if __name__ == "__main__":
    main()
