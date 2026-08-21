#!/usr/bin/env python3
"""Combine compression, PPL, and downstream evidence into a Pareto table."""
from __future__ import annotations

import argparse
import csv
import json
import os


PRIMARY = {
    "allocation_source": "rmsnorm_bound",
    "ranking_source": "rmsnorm_ellipsoid_bound",
    "expert_aggregation": "p95",
}

FIELDS = [
    "label", "target_pct", "actual_moe_width_reduction_pct",
    "whole_model_parameter_reduction_pct", "serialized_byte_reduction_pct",
    "removed_layer_channels", "removed_expert_neurons",
    "wikitext2_mean_dnll", "c4_mean_dnll", "seven_task_macro_accuracy",
    "macro_accuracy_loss_points", "checkpoint_storage_bytes",
    "checkpoint_storage_gb_decimal", "checkpoint_storage_gib_binary",
    "pareto_optimal", "dominated_by",
]


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _target_label(target: int) -> str:
    return ("baseline_unpruned" if target == 0 else
            f"rmsnorm_alloc__ellipsoid_rank__p95__target{target}")


def _one(rows: list[dict], predicate, label: str) -> dict:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(f"expected one {label}; found {len(matches)}")
    return matches[0]


def _dominates(left: dict, right: dict) -> bool:
    maximize = (
        "whole_model_parameter_reduction_pct", "serialized_byte_reduction_pct",
    )
    minimize = (
        "wikitext2_mean_dnll", "c4_mean_dnll", "macro_accuracy_loss_points",
    )
    no_worse = all(float(left[key]) >= float(right[key]) - 1e-15
                   for key in maximize)
    no_worse &= all(float(left[key]) <= float(right[key]) + 1e-15
                    for key in minimize)
    strictly = any(float(left[key]) > float(right[key]) + 1e-15
                   for key in maximize)
    strictly |= any(float(left[key]) < float(right[key]) - 1e-15
                    for key in minimize)
    return bool(no_worse and strictly)


def build_rows(
    milestone: list[dict], checkpoints: list[dict], downstream: list[dict]
) -> list[dict]:
    checkpoint_by_label = {row["label"]: row for row in checkpoints}
    if len(checkpoint_by_label) != len(checkpoints):
        raise ValueError("checkpoint table contains duplicate labels")
    macros = {row["label"]: row for row in downstream
              if row["task"] == "macro_average"}
    baseline_checkpoint = checkpoint_by_label["baseline_unpruned"]
    baseline_storage = int(float(baseline_checkpoint["serialized_weight_bytes"]))
    baseline_macro = float(macros["baseline_unpruned"]["accuracy"])
    output = []
    for target in (0, 2, 4, 6, 8):
        label = _target_label(target)
        if label not in checkpoint_by_label:
            raise ValueError(f"checkpoint missing from table: {label}")
        if label not in macros:
            raise ValueError(f"downstream macro missing from table: {label}")
        checkpoint = checkpoint_by_label[label]
        macro = float(macros[label]["accuracy"])
        if target == 0:
            wiki_dnll = c4_dnll = 0.0
        else:
            evidence = _one(milestone, lambda row: (
                int(round(float(row["requested_pct"]))) == target
                and all(row[key] == value for key, value in PRIMARY.items())
                and ((target in (2, 4, 6) and row["source_group"] == "frozen_2_4_6")
                     or (target == 8 and "target8_rmsnorm_primary" in row["source_group"]))
            ), f"primary target-{target} PPL evidence")
            wiki_dnll = float(evidence["wikitext2_mean_nll_difference"])
            c4_dnll = float(evidence["c4_mean_nll_difference"])
        storage = int(float(checkpoint["serialized_weight_bytes"]))
        output.append({
            "label": label, "target_pct": target,
            "actual_moe_width_reduction_pct": float(checkpoint["actual_pct"]),
            "whole_model_parameter_reduction_pct": float(
                checkpoint["total_parameter_reduction_pct"]
            ),
            "serialized_byte_reduction_pct": 100.0 * (
                1.0 - storage / baseline_storage
            ),
            "removed_layer_channels": int(float(checkpoint["removed_layer_channels"])),
            "removed_expert_neurons": int(float(checkpoint["removed_expert_neurons"])),
            "wikitext2_mean_dnll": wiki_dnll, "c4_mean_dnll": c4_dnll,
            "seven_task_macro_accuracy": macro,
            "macro_accuracy_loss_points": 100.0 * (baseline_macro - macro),
            "checkpoint_storage_bytes": storage,
            "checkpoint_storage_gb_decimal": storage / 1_000_000_000,
            "checkpoint_storage_gib_binary": storage / (1024 ** 3),
            "pareto_optimal": False, "dominated_by": "",
        })
    for row in output:
        dominators = [candidate["label"] for candidate in output
                      if candidate is not row and _dominates(candidate, row)]
        row["pareto_optimal"] = not dominators
        row["dominated_by"] = ";".join(dominators)
    return output


def _write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Checkpoint | MoE width red. | Model param red. | Byte red. | Wiki dNLL | C4 dNLL | 7-task macro | Loss (points) | Storage GB | Pareto |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {row['actual_moe_width_reduction_pct']:.3f}% | "
                f"{row['whole_model_parameter_reduction_pct']:.3f}% | "
                f"{row['serialized_byte_reduction_pct']:.3f}% | "
                f"{row['wikitext2_mean_dnll']:+.6f} | {row['c4_mean_dnll']:+.6f} | "
                f"{row['seven_task_macro_accuracy']:.4f} | "
                f"{row['macro_accuracy_loss_points']:+.2f} | "
                f"{row['checkpoint_storage_gb_decimal']:.3f} | "
                f"{row['pareto_optimal']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone-csv", required=True)
    parser.add_argument("--checkpoint-table-csv", required=True)
    parser.add_argument("--downstream-table-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = build_rows(_read(args.milestone_csv), _read(args.checkpoint_table_csv),
                      _read(args.downstream_table_csv))
    if args.dry_run:
        print(
            f"[pareto] DRY RUN: checkpoints={len(rows)} pareto="
            f"{[row['label'] for row in rows if row['pareto_optimal']]}"
        )
        return
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "paper_v3_pareto_table.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "paper_v3_pareto_table.json"), "x",
              encoding="utf-8") as handle:
        json.dump({
            "primary_method": PRIMARY,
            "pareto_definition": {
                "maximize": ["whole_model_parameter_reduction_pct",
                             "serialized_byte_reduction_pct"],
                "minimize": ["wikitext2_mean_dnll", "c4_mean_dnll",
                             "macro_accuracy_loss_points"],
                "rule": "no worse on all objectives and strictly better on at least one",
            },
            "rows": rows,
        }, handle, indent=2)
    _write_markdown(os.path.join(args.output_dir, "paper_v3_pareto_table.md"), rows)
    with open(os.path.join(args.output_dir, "paper_v3_pareto_table.tex"), "x",
              encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrrrrl}\n\\toprule\nCheckpoint & MoE red. & Model red. & Byte red. & Wiki dNLL & C4 dNLL & Macro acc. & Loss pts & GB & Pareto \\\\\n\\midrule\n")
        for row in rows:
            label = row["label"].replace("_", r"\_")
            handle.write(
                f"{label} & {row['actual_moe_width_reduction_pct']:.3f} & "
                f"{row['whole_model_parameter_reduction_pct']:.3f} & "
                f"{row['serialized_byte_reduction_pct']:.3f} & "
                f"{row['wikitext2_mean_dnll']:+.6f} & {row['c4_mean_dnll']:+.6f} & "
                f"{row['seven_task_macro_accuracy']:.4f} & "
                f"{row['macro_accuracy_loss_points']:+.2f} & "
                f"{row['checkpoint_storage_gb_decimal']:.3f} & "
                f"{row['pareto_optimal']} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[pareto] OK: {len(rows)} checkpoints; {csv_path}")


if __name__ == "__main__":
    main()
