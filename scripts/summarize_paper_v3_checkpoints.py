#!/usr/bin/env python3
"""Summarize verified physical checkpoints into paper table formats."""
from __future__ import annotations

import argparse
import csv
import json
import os

FIELDS = [
    "label", "target_pct", "actual_pct", "removed_layer_channels",
    "removed_expert_neurons", "total_parameters", "moe_expert_parameters",
    "total_parameter_reduction_pct", "moe_parameter_reduction_pct",
    "serialized_weight_bytes", "checkpoint_payload_bytes",
    "successful_reload", "exact_logits_after_reload", "max_logit_difference",
    "no_hidden_original_width_padding", "plan_sha256", "checkpoint_dir",
]


def build_rows(specs: list[dict]) -> list[dict]:
    rows, baseline = [], None
    for spec in specs:
        path = os.path.join(spec["checkpoint_dir"], "checkpoint_verification.json")
        with open(path, encoding="utf-8") as handle:
            verification = json.load(handle)
        for key in ("successful_reload", "exact_logits_after_reload",
                    "no_hidden_original_width_padding"):
            if not verification.get(key):
                raise ValueError(f"checkpoint failed {key}: {path}")
        counts = verification["parameters_reloaded"]
        if baseline is None and spec["target_pct"] == 0:
            baseline = counts
        rows.append({
            "label": spec["label"], "target_pct": spec["target_pct"],
            "actual_pct": spec["actual_pct"],
            "removed_layer_channels": verification["removed_layer_channels"],
            "removed_expert_neurons": verification["removed_expert_neurons"],
            "total_parameters": counts["total"],
            "moe_expert_parameters": counts["moe_experts"],
            "total_parameter_reduction_pct": "",
            "moe_parameter_reduction_pct": "",
            "serialized_weight_bytes": verification["serialized_weight_bytes"],
            "checkpoint_payload_bytes": verification[
                "checkpoint_payload_bytes_excluding_verification_manifest"
            ],
            "successful_reload": verification["successful_reload"],
            "exact_logits_after_reload": verification["exact_logits_after_reload"],
            "max_logit_difference": verification["max_logit_difference"],
            "no_hidden_original_width_padding": verification[
                "no_hidden_original_width_padding"
            ],
            "plan_sha256": verification["plan_sha256"],
            "checkpoint_dir": spec["checkpoint_dir"],
        })
    if baseline is None:
        raise ValueError("baseline checkpoint missing")
    for row in rows:
        row["total_parameter_reduction_pct"] = 100.0 * (
            1.0 - int(row["total_parameters"]) / baseline["total"]
        )
        row["moe_parameter_reduction_pct"] = 100.0 * (
            1.0 - int(row["moe_expert_parameters"]) / baseline["moe_experts"]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.manifest, encoding="utf-8") as handle:
        rows = build_rows(json.load(handle))
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "checkpoint_table.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "checkpoint_table.json"), "x",
              encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(os.path.join(args.output_dir, "checkpoint_table.md"), "x",
              encoding="utf-8") as handle:
        handle.write("| Checkpoint | Removed channels | Expert neurons | Total params | MoE params | Weights bytes | Reload | Exact logits | No padding |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {row['removed_layer_channels']} | "
                f"{row['removed_expert_neurons']} | {row['total_parameters']} | "
                f"{row['moe_expert_parameters']} | {row['serialized_weight_bytes']} | "
                f"{row['successful_reload']} | {row['exact_logits_after_reload']} | "
                f"{row['no_hidden_original_width_padding']} |\n"
            )
    with open(os.path.join(args.output_dir, "checkpoint_table.tex"), "x",
              encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nCheckpoint & Channels & Expert neurons & Total params & MoE params & Bytes & Reload & No padding \\\\\n\\midrule\n")
        for row in rows:
            label = row["label"].replace("_", r"\_")
            handle.write(
                f"{label} & {row['removed_layer_channels']} & {row['removed_expert_neurons']} & "
                f"{row['total_parameters']} & {row['moe_expert_parameters']} & "
                f"{row['serialized_weight_bytes']} & {row['successful_reload']} & "
                f"{row['no_hidden_original_width_padding']} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[checkpoint-summary] OK: {len(rows)} rows; {csv_path}")


if __name__ == "__main__":
    main()
