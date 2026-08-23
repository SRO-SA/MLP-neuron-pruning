#!/usr/bin/env python3
"""Validate and summarize the five-plan certified-hybrid PPL gate."""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


FIELDS = [
    "plan", "certificate_slack", "strict_certificate",
    "certificate_change_vs_ellipsoid_pct", "normalized_down_norm_objective",
    "dataset", "baseline_ppl", "pruned_ppl", "relative_ppl_change_pct",
    "mean_nll_difference", "mean_nll_ci95_lower", "mean_nll_ci95_upper",
    "removed_layer_channels", "removed_expert_neurons", "actual_pruning_pct",
    "expert_param_reduction_pct", "total_model_param_reduction_pct",
    "forward_check", "token_count_match", "plan_path", "plan_sha256",
    "selection_sha256", "certificate_objective_pareto_optimal",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        matches = [p for p in glob.glob(str(Path(item["output_dir"]) / "moe_target_pruning_*.csv"))
                   if not p.endswith("_per_layer.csv")]
        if len(matches) != 1:
            raise FileNotFoundError(f"{item['plan']}: expected one result CSV; {matches}")
        results = list(csv.DictReader(open(matches[0], newline="", encoding="utf-8")))
        if {row["eval_dataset"] for row in results} != {"wikitext2", "c4"}:
            raise ValueError(f"{item['plan']}: dataset mismatch")
        for result in results:
            if int(float(result["selected_layer_channels"])) != 2288:
                raise ValueError(f"{item['plan']}: fixed budget changed")
            if result["evaluation_token_count_match"].lower() not in {"true", "1"}:
                raise ValueError(f"{item['plan']}: token counts differ")
            rows.append({
                "plan": item["plan"],
                "certificate_slack": item["certificate_slack"],
                "strict_certificate": item["strict_certificate"],
                "certificate_change_vs_ellipsoid_pct": item["certificate_change_vs_ellipsoid_pct"],
                "normalized_down_norm_objective": item["normalized_down_norm_objective"],
                "dataset": result["eval_dataset"],
                "baseline_ppl": result["baseline_ppl"],
                "pruned_ppl": result["compressed_ppl"],
                "relative_ppl_change_pct": result["relative_delta_pct"],
                "mean_nll_difference": result["mean_nll_difference"],
                "mean_nll_ci95_lower": result["mean_nll_difference_ci95_lower"],
                "mean_nll_ci95_upper": result["mean_nll_difference_ci95_upper"],
                "removed_layer_channels": result["selected_layer_channels"],
                "removed_expert_neurons": result["removed_expert_neurons"],
                "actual_pruning_pct": result["actual_pct"],
                "expert_param_reduction_pct": result["expert_param_reduction_pct"],
                "total_model_param_reduction_pct": result["total_model_param_reduction_pct"],
                "forward_check": result["forward_check"],
                "token_count_match": result["evaluation_token_count_match"],
                "plan_path": item["plan_path"],
                "plan_sha256": item["plan_sha256"],
                "selection_sha256": item["selection_sha256"],
                "certificate_objective_pareto_optimal": item["certificate_objective_pareto_optimal"],
            })
    expected_rows = 2 * len(manifest)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} PPL rows, got {len(rows)}")
    if args.dry_run:
        print(
            f"[hybrid-ppl-summary] DRY RUN: {expected_rows} validated rows "
            f"for {len(manifest)} distinct candidates"
        )
        return
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    with (output_dir / "hybrid_ppl_pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "hybrid_ppl_pareto.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2), encoding="utf-8"
    )
    header = "| Plan | Slack | Certificate Δ | Down-norm objective | Dataset | dNLL | PPL Δ |\n|---|---:|---:|---:|---|---:|---:|"
    lines = [header] + [
        f"| {r['plan']} | {r['certificate_slack']} | {float(r['certificate_change_vs_ellipsoid_pct']):.3f}% | "
        f"{float(r['normalized_down_norm_objective']):.6g} | {r['dataset']} | "
        f"{float(r['mean_nll_difference']):.6f} | {float(r['relative_ppl_change_pct']):.3f}% |"
        for r in rows
    ]
    (output_dir / "hybrid_ppl_pareto.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = ["\\begin{tabular}{lrrrrrr}", "Plan & Slack & Cert. $\\Delta$ & $D(S)$ & Dataset & dNLL & PPL $\\Delta$ \\\\", "\\hline"]
    for r in rows:
        plan_tex = r["plan"].replace("_", "\\_")
        tex.append(
            f"{plan_tex} & {r['certificate_slack']} & "
            f"{float(r['certificate_change_vs_ellipsoid_pct']):.3f}\\% & "
            f"{float(r['normalized_down_norm_objective']):.5g} & {r['dataset']} & "
            f"{float(r['mean_nll_difference']):.5f} & {float(r['relative_ppl_change_pct']):.3f}\\% \\\\" 
        )
    tex.append("\\end{tabular}")
    (output_dir / "hybrid_ppl_pareto.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"[hybrid-ppl-summary] OK rows=10 output={output_dir}")


if __name__ == "__main__":
    main()
