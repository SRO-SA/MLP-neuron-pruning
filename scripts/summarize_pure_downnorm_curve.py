#!/usr/bin/env python3
"""Join pure down-norm PPL and downstream results into one compression curve."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.moe_set_certification import selected_by_layer
from src.experiment_provenance import file_sha256


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def curve_plan_paths(checkpoint_manifest: str | Path) -> dict[int, str]:
    """Resolve authoritative curve plans from the frozen export manifest."""
    specs = json.loads(Path(checkpoint_manifest).read_text(encoding="utf-8"))
    result = {}
    for target in (2, 4, 6, 8):
        label = f"rmsnorm_alloc__downnorm_rank__p95__target{target}"
        matches = [row for row in specs if row.get("label") == label]
        if len(matches) != 1:
            raise ValueError(
                f"target {target}: expected one checkpoint-manifest plan, "
                f"found {len(matches)}"
            )
        spec = matches[0]
        path = Path(spec["plan_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = spec.get("plan_sha256")
        if expected_hash and file_sha256(str(path)) != expected_hash:
            raise ValueError(f"target {target}: checkpoint-manifest plan hash changed")
        result[target] = str(path)
    return result


def audit_nesting(plan_paths: dict[int, str]) -> list[dict]:
    selected = {}
    for target, path_text in plan_paths.items():
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        selected[target] = {
            layer: set(indices)
            for layer, indices in selected_by_layer(
                json.loads(path.read_text(encoding="utf-8"))
            ).items()
        }
    rows = []
    targets = sorted(selected)
    for lower_target, upper_target in zip(targets, targets[1:]):
        lower, upper = selected[lower_target], selected[upper_target]
        if set(lower) != set(upper):
            raise ValueError("compression-curve plans have different layer sets")
        violating_layers = [
            layer for layer in sorted(lower) if not lower[layer].issubset(upper[layer])
        ]
        intersection = sum(len(lower[layer] & upper[layer]) for layer in lower)
        union = sum(len(lower[layer] | upper[layer]) for layer in lower)
        rows.append({
            "lower_target_pct": lower_target,
            "upper_target_pct": upper_target,
            "lower_removed_layer_channels": sum(map(len, lower.values())),
            "upper_removed_layer_channels": sum(map(len, upper.values())),
            "fully_nested": not violating_layers,
            "lower_channels_missing_from_upper": sum(
                len(lower[layer] - upper[layer]) for layer in lower
            ),
            "channels_added_at_upper": sum(
                len(upper[layer] - lower[layer]) for layer in lower
            ),
            "selected_channel_jaccard": intersection / union if union else 1.0,
            "violating_layer_count": len(violating_layers),
            "violating_layers": violating_layers,
            "plans_independently_optimized": True,
            "interpretation": (
                "nested despite independent fixed-budget optimization"
                if not violating_layers else
                "not nested; each budget was independently optimized and "
                "monotonic set inclusion must not be implied"
            ),
        })
    return rows


def adjacent_macro_rows(path: str) -> list[dict]:
    rows = read_csv(path)
    result = []
    for row in rows:
        if (
            row.get("comparison_type") != "compression_curve_adjacent_budget"
            or row.get("task") != "macro_average"
        ):
            continue
        first_match = re.search(r"target(\d+)$", row["first_label"])
        second_match = re.search(r"target(\d+)$", row["second_label"])
        if not first_match or not second_match:
            raise ValueError("cannot parse adjacent-budget labels")
        result.append({
            "higher_target_pct": int(first_match.group(1)),
            "lower_target_pct": int(second_match.group(1)),
            "difference_definition": "higher target minus lower target macro accuracy",
            "macro_accuracy_difference": row["accuracy_difference"],
            "paired_ci95_lower": row["ci95_lower"],
            "paired_ci95_upper": row["ci95_upper"],
            "paired_randomization_p_value": row["paired_randomization_p_value"],
            "holm_adjusted_p_value": row["holm_adjusted_p_value"],
            "bh_adjusted_p_value": row["bh_adjusted_p_value"],
            "significant_95pct": row["significant_95pct"],
            "comparison_scope": row["comparison_scope"],
        })
    result.sort(key=lambda row: row["lower_target_pct"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl-summary", required=True)
    parser.add_argument("--downstream-table", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--budget-audit", required=True)
    parser.add_argument("--paired-comparisons", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ppl = read_csv(args.ppl_summary)
    downstream = read_csv(args.downstream_table)
    budget = {int(row["target"]): row for row in json.loads(
        Path(args.budget_audit).read_text(encoding="utf-8")
    )}
    rows = []
    plan_paths = curve_plan_paths(args.checkpoint_manifest)
    for target in (2, 4, 6, 8):
        label = f"rmsnorm_alloc__downnorm_rank__p95__target{target}"
        ppl_rows = [row for row in ppl if (
            int(round(float(row["requested_pct"]))) == target
            and row["allocation_source"] == "rmsnorm_bound"
            and row["ranking_source"] == "down_norm"
        )]
        by_dataset = {row["dataset"]: row for row in ppl_rows}
        macro = [row for row in downstream if row["label"] == label and row["task"] == "macro_average"]
        if set(by_dataset) != {"wikitext2", "c4"} or len(macro) != 1:
            raise ValueError(f"target {target}: incomplete PPL/downstream evidence")
        row0 = by_dataset["wikitext2"]
        rows.append({
            "target_pct": target,
            "actual_pct": row0["actual_pct"],
            "removed_layer_channels": row0["layer_channels"],
            "removed_expert_neurons": row0["expert_neurons"],
            "expert_param_reduction_pct": row0["expert_param_reduction_pct"],
            "total_model_param_reduction_pct": row0["total_model_param_reduction_pct"],
            "wikitext2_dnll": by_dataset["wikitext2"]["mean_nll_difference"],
            "c4_dnll": by_dataset["c4"]["mean_nll_difference"],
            "macro_accuracy": macro[0]["accuracy"],
            "macro_accuracy_change_vs_baseline": macro[0]["absolute_change"],
            "macro_ci95_lower": macro[0]["ci95_lower"],
            "macro_ci95_upper": macro[0]["ci95_upper"],
            "aligned_budget_difference_vs_ellipsoid": budget[target]["aligned_budget_difference"],
        })
    nesting = audit_nesting(plan_paths)
    adjacent = adjacent_macro_rows(args.paired_comparisons)
    if len(adjacent) != 3:
        raise ValueError(f"expected three adjacent macro comparisons, got {len(adjacent)}")
    if args.dry_run:
        print(
            f"[downnorm-curve] DRY RUN rows={len(rows)} "
            f"nesting={len(nesting)} adjacent={len(adjacent)}"
        )
        return
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with (output / "pure_downnorm_compression_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "pure_downnorm_compression_curve.json").write_text(
        json.dumps({
            "schema_version": 2, "rows": rows,
            "nesting_audit": nesting,
            "adjacent_paired_macro_comparisons": adjacent,
        }, indent=2), encoding="utf-8"
    )
    for filename, audit_rows in (
        ("pure_downnorm_plan_nesting.csv", nesting),
        ("pure_downnorm_adjacent_paired_macro.csv", adjacent),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
            writer.writeheader(); writer.writerows(audit_rows)
    (output / "pure_downnorm_plan_nesting.json").write_text(
        json.dumps(nesting, indent=2), encoding="utf-8"
    )
    (output / "pure_downnorm_adjacent_paired_macro.json").write_text(
        json.dumps(adjacent, indent=2), encoding="utf-8"
    )
    md = ["| Target | Actual | Channels | WikiText2 dNLL | C4 dNLL | Macro accuracy | Δ vs baseline |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md.append(
            f"| {row['target_pct']}% | {float(row['actual_pct']):.3f}% | {row['removed_layer_channels']} | "
            f"{float(row['wikitext2_dnll']):+.5f} | {float(row['c4_dnll']):+.5f} | "
            f"{float(row['macro_accuracy']):.4f} | {float(row['macro_accuracy_change_vs_baseline']):+.4f} |"
        )
    (output / "pure_downnorm_compression_curve.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = ["\\begin{tabular}{rrrrrrr}", "Target & Actual & Channels & WT2 dNLL & C4 dNLL & Macro acc. & $\\Delta$base \\\\", "\\hline"]
    for row in rows:
        tex.append(
            f"{row['target_pct']}\\% & {float(row['actual_pct']):.3f}\\% & {row['removed_layer_channels']} & "
            f"{float(row['wikitext2_dnll']):+.5f} & {float(row['c4_dnll']):+.5f} & "
            f"{float(row['macro_accuracy']):.4f} & {float(row['macro_accuracy_change_vs_baseline']):+.4f} \\\\"
        )
    tex.append("\\end{tabular}")
    (output / "pure_downnorm_compression_curve.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    nesting_md = [
        "| Lower target | Upper target | Nested | Lost from lower | Added at upper | Jaccard | Violating layers | Interpretation |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in nesting:
        nesting_md.append(
            f"| {row['lower_target_pct']}% | {row['upper_target_pct']}% | "
            f"{row['fully_nested']} | {row['lower_channels_missing_from_upper']} | "
            f"{row['channels_added_at_upper']} | {row['selected_channel_jaccard']:.4f} | "
            f"{row['violating_layer_count']} | {row['interpretation']} |"
        )
    (output / "pure_downnorm_plan_nesting.md").write_text(
        "\n".join(nesting_md) + "\n", encoding="utf-8"
    )
    nesting_tex = [
        "\\begin{tabular}{rrrrrr}",
        "Lower & Upper & Nested & Lost & Added & Jaccard \\\\",
        "\\hline",
    ]
    for row in nesting:
        nesting_tex.append(
            f"{row['lower_target_pct']}\\% & {row['upper_target_pct']}\\% & "
            f"{row['fully_nested']} & {row['lower_channels_missing_from_upper']} & "
            f"{row['channels_added_at_upper']} & "
            f"{row['selected_channel_jaccard']:.4f} \\\\"
        )
    nesting_tex.append("\\end{tabular}")
    (output / "pure_downnorm_plan_nesting.tex").write_text(
        "\n".join(nesting_tex) + "\n", encoding="utf-8"
    )
    adjacent_md = [
        "| Higher target | Lower target | Macro accuracy difference | Paired 95% CI | Randomization p | Holm p | BH p |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in adjacent:
        adjacent_md.append(
            f"| {row['higher_target_pct']}% | {row['lower_target_pct']}% | "
            f"{float(row['macro_accuracy_difference']):+.5f} | "
            f"[{float(row['paired_ci95_lower']):+.5f}, "
            f"{float(row['paired_ci95_upper']):+.5f}] | "
            f"{float(row['paired_randomization_p_value']):.4g} | "
            f"{float(row['holm_adjusted_p_value']):.4g} | "
            f"{float(row['bh_adjusted_p_value']):.4g} |"
        )
    (output / "pure_downnorm_adjacent_paired_macro.md").write_text(
        "\n".join(adjacent_md) + "\n", encoding="utf-8"
    )
    adjacent_tex = [
        "\\begin{tabular}{rrrrrr}",
        "Higher & Lower & $\\Delta$acc & 95\\% CI & Holm $p$ & BH $p$ \\\\",
        "\\hline",
    ]
    for row in adjacent:
        adjacent_tex.append(
            f"{row['higher_target_pct']}\\% & {row['lower_target_pct']}\\% & "
            f"{float(row['macro_accuracy_difference']):+.5f} & "
            f"[{float(row['paired_ci95_lower']):+.5f}, "
            f"{float(row['paired_ci95_upper']):+.5f}] & "
            f"{float(row['holm_adjusted_p_value']):.4g} & "
            f"{float(row['bh_adjusted_p_value']):.4g} \\\\"
        )
    adjacent_tex.append("\\end{tabular}")
    (output / "pure_downnorm_adjacent_paired_macro.tex").write_text(
        "\n".join(adjacent_tex) + "\n", encoding="utf-8"
    )
    print(f"[downnorm-curve] OK rows=4 output={output}")


if __name__ == "__main__":
    main()
