#!/usr/bin/env python3
"""Build paper tables and a compact figure for routed-MoE perturbation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.experiment_provenance import file_sha256
from src.routed_moe_perturbation import (
    paired_bootstrap_mean_difference,
    spearman_correlation,
)


LABELS = (
    "rmsnorm_alloc__ellipsoid_rank__p95__target6",
    "certified_hybrid__downnorm_refinement_slack0p25__target6",
    "certified_hybrid__downnorm_refinement_slack2__target6",
    "rmsnorm_alloc__downnorm_rank__p95__target6",
)
DISPLAY = {
    LABELS[0]: "Pure ellipsoid",
    LABELS[1]: "Hybrid (0.25% slack)",
    LABELS[2]: "Hybrid (2% slack)",
    LABELS[3]: "Pure down-norm",
}
METRICS = (
    "actual_perturbation", "strict_bound_ratio",
    "route_conditioned_bound_ratio", "perturbation_over_base_moe_norm",
    "perturbation_over_residual_norm",
)


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict]) -> str:
    columns = (
        "plan", "corpus", "actual_pruning_percentage", "removed_layer_channels",
        "C_S", "down_norm_objective", "actual_mean", "actual_p95",
        "strict_ratio_median", "strict_ratio_p99", "strict_ratio_max",
        "route_ratio_median", "route_ratio_p99", "route_ratio_max",
        "actual_over_moe_mean", "actual_over_residual_mean", "violations",
    )
    headings = (
        "Plan", "Corpus", "Actual pruning %", "Removed channels",
        "C(S)", "D(S)", "Mean delta", "p95 delta",
        "Median delta/B strict", "p99", "Max", "Median delta/B route",
        "p99", "Max", "Mean delta/||MoE||", "Mean delta/||residual||",
        "Violations",
    )
    lines = ["| " + " | ".join(headings) + " |",
             "|" + "|".join("---" for _ in headings) + "|"]
    for row in rows:
        values = []
        for key in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Plan & $C(S)$ & $D(S)$ & Mean $\delta$ & p95 $\delta$ & "
        r"Max $\delta/B_l$ & Max $\delta/B^{route}$ \\",
        r"\midrule",
    ]
    for row in rows:
        name = str(row["plan"]).replace("%", r"\%").replace("_", r"\_")
        lines.append(
            f"{name} & {row['C_S']:.5g} & {row['down_norm_objective']:.5g} & "
            f"{row['actual_mean']:.5g} & {row['actual_p95']:.5g} & "
            f"{row['strict_ratio_max']:.5g} & {row['route_ratio_max']:.5g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _figure(path_png: Path, path_pdf: Path, aggregate: list[dict], documents: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for color, row in zip(colors, aggregate):
        axes[0].scatter(row["C_S"], row["actual_mean"], s=55, color=color)
        axes[0].annotate(
            row["plan"], (row["C_S"], row["actual_mean"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )
    axes[0].set_xlabel("Strict aggregate certificate C(S)")
    axes[0].set_ylabel("Mean local perturbation")
    axes[0].grid(alpha=0.25)
    data = [
        [float(row["mean_actual_perturbation"]) for row in documents[label]]
        for label in LABELS
    ]
    boxes = axes[1].boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[1].set_xticklabels(["Ellip.", "H-0.25", "H-2", "Down"], rotation=20)
    axes[1].set_ylabel("Document mean local perturbation")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path_png, dpi=220)
    figure.savefig(path_pdf)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_dir)
    results, layers, documents = {}, {}, {}
    for label in LABELS:
        directory = run_root / label
        results[label] = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        layers[label] = _csv_rows(directory / "layer_summary.csv")
        documents[label] = _csv_rows(directory / "document_summary.csv")
        if results[label]["strict_bound_violations"] != 0:
            raise AssertionError(f"{label}: strict theorem violations")
        if results[label]["route_conditioned_bound_violations"] != 0:
            raise AssertionError(f"{label}: route-conditioned theorem violations")
    capture_hashes = {row["capture_manifest_sha256"] for row in results.values()}
    if len(capture_hashes) != 1:
        raise ValueError("plan evaluations did not use one immutable baseline capture")
    doc_hashes = {
        label: [row["text_sha256"] for row in documents[label]] for label in LABELS
    }
    if any(doc_hashes[label] != doc_hashes[LABELS[0]] for label in LABELS[1:]):
        raise ValueError("document identities differ across plan evaluations")

    aggregate = []
    for label in LABELS:
        result = results[label]
        overall = result["overall"]
        aggregate.append({
            "label": label,
            "plan": DISPLAY[label],
            "corpus": result["corpus"]["repo"],
            "actual_pruning_percentage": float(result["actual_pruning_percentage"]),
            "removed_layer_channels": int(result["removed_layer_channels"]),
            "C_S": float(result["strict_global_unpropagated_certificate_C"]),
            "down_norm_objective": float(result["normalized_down_norm_objective"]),
            "actual_mean": float(overall["actual_perturbation"]["mean"]),
            "actual_median": float(overall["actual_perturbation"]["median"]),
            "actual_p95": float(overall["actual_perturbation"]["p95"]),
            "actual_p99": float(overall["actual_perturbation"]["p99"]),
            "actual_max": float(overall["actual_perturbation"]["max"]),
            "strict_ratio_mean": float(overall["strict_bound_ratio"]["mean"]),
            "strict_ratio_median": float(overall["strict_bound_ratio"]["median"]),
            "strict_ratio_p95": float(overall["strict_bound_ratio"]["p95"]),
            "strict_ratio_p99": float(overall["strict_bound_ratio"]["p99"]),
            "strict_ratio_max": float(overall["strict_bound_ratio"]["max"]),
            "route_ratio_mean": float(overall["route_conditioned_bound_ratio"]["mean"]),
            "route_ratio_median": float(overall["route_conditioned_bound_ratio"]["median"]),
            "route_ratio_p95": float(overall["route_conditioned_bound_ratio"]["p95"]),
            "route_ratio_p99": float(overall["route_conditioned_bound_ratio"]["p99"]),
            "route_ratio_max": float(overall["route_conditioned_bound_ratio"]["max"]),
            "actual_over_moe_mean": float(
                overall["perturbation_over_base_moe_norm"]["mean"]
            ),
            "actual_over_residual_mean": float(
                overall["perturbation_over_residual_norm"]["mean"]
            ),
            "violations": int(result["strict_bound_violations"])
                          + int(result["route_conditioned_bound_violations"]),
            "observations": int(result["token_layer_observations"]),
            "plan_sha256": result["source_plan_sha256"],
        })
    layer_observations = []
    for label in LABELS:
        for row in layers[label]:
            layer_observations.append({
                "label": label,
                "plan": DISPLAY[label],
                **row,
            })
    strict_values = np.asarray([
        float(row["strict_set_bound"]) for row in layer_observations
    ])
    objective_values = np.asarray([
        float(row["normalized_down_norm_objective"]) for row in layer_observations
    ])
    empirical_values = np.asarray([
        float(row["actual_perturbation_mean"]) for row in layer_observations
    ])
    correlations = {
        "layer_plan_observations": len(layer_observations),
        "spearman_strict_layer_bound_vs_mean_actual": spearman_correlation(
            strict_values, empirical_values,
        ),
        "spearman_down_norm_objective_vs_mean_actual": spearman_correlation(
            objective_values, empirical_values,
        ),
        "spearman_global_C_vs_overall_mean_actual": spearman_correlation(
            np.asarray([row["C_S"] for row in aggregate]),
            np.asarray([row["actual_mean"] for row in aggregate]),
        ),
    }
    paired = []
    first, second = LABELS[1], LABELS[3]
    for metric in METRICS:
        left = np.asarray([
            float(row[f"mean_{metric}"]) for row in documents[first]
        ])
        right = np.asarray([
            float(row[f"mean_{metric}"]) for row in documents[second]
        ])
        paired.append({
            "first_label": first,
            "second_label": second,
            "sign_convention": "hybrid_0.25_percent_minus_pure_down_norm",
            "metric": metric,
            **paired_bootstrap_mean_difference(
                left, right, resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            ),
        })
    strict_all_below_one = all(row["strict_ratio_max"] <= 1.0 for row in aggregate)
    route_all_below_one = all(row["route_ratio_max"] <= 1.0 for row in aggregate)
    interpretation = {
        "all_strict_bound_ratios_below_one": strict_all_below_one,
        "all_route_conditioned_bound_ratios_below_one": route_all_below_one,
        "objective_orders_actual_perturbation_spearman": correlations[
            "spearman_global_C_vs_overall_mean_actual"
        ],
        "route_conditioning_effect": (
            "The route-conditioned bound is no larger than the uniform strict "
            "layer bound, so its observed/bound ratio is normally higher (tighter)."
        ),
        "summation_effect": (
            "Set-level expert summation is compared in each result against the older "
            "sum of channelwise expert maxima; it is never larger by construction."
        ),
        "channel_level_comparison_caveat": (
            "The prior channel-level tightness audit used a different unit of analysis; "
            "its ratios are not pooled with routed set-level ratios."
        ),
    }
    if args.dry_run:
        print(
            f"[routed-moe-summary] DRY RUN plans={len(aggregate)} "
            f"layers={len(layer_observations)} paired={len(paired)}"
        )
        return

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _write_csv(output / "routed_moe_perturbation_summary.csv", aggregate)
    _write_csv(output / "routed_moe_layer_summary.csv", layer_observations)
    _write_csv(output / "routed_moe_paired_document_bootstrap.csv", paired)
    (output / "routed_moe_perturbation_summary.md").write_text(
        _markdown(aggregate), encoding="utf-8",
    )
    (output / "routed_moe_perturbation_summary.tex").write_text(
        _latex(aggregate), encoding="utf-8",
    )
    _figure(
        output / "routed_moe_perturbation_figure.png",
        output / "routed_moe_perturbation_figure.pdf",
        aggregate, documents,
    )
    payload = {
        "schema_version": 1,
        "comparison_type": "local_same_input_fixed_route_routed_moe",
        "end_to_end_trace_used": False,
        "aggregate": aggregate,
        "correlations": correlations,
        "paired_document_bootstrap": paired,
        "interpretation": interpretation,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "capture_manifest_sha256": next(iter(capture_hashes)),
        "source_result_sha256": {
            label: file_sha256(str(run_root / label / "result.json"))
            for label in LABELS
        },
    }
    (output / "routed_moe_perturbation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    hashes = {
        path.name: file_sha256(str(path))
        for path in output.iterdir() if path.is_file()
    }
    (output / "ARTIFACT_HASHES.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8",
    )
    print(
        f"[routed-moe-summary] OK plans={len(aggregate)} "
        f"strict_below_one={strict_all_below_one} output={output}"
    )


if __name__ == "__main__":
    main()
