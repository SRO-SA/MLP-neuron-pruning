#!/usr/bin/env python3
"""Join pure down-norm PPL and downstream results into one compression curve."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl-summary", required=True)
    parser.add_argument("--downstream-table", required=True)
    parser.add_argument("--budget-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ppl = list(csv.DictReader(open(args.ppl_summary, newline="", encoding="utf-8")))
    downstream = list(csv.DictReader(open(args.downstream_table, newline="", encoding="utf-8")))
    budget = {int(row["target"]): row for row in json.loads(
        Path(args.budget_audit).read_text(encoding="utf-8")
    )}
    rows = []
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
    if args.dry_run:
        print(f"[downnorm-curve] DRY RUN rows={len(rows)}")
        return
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with (output / "pure_downnorm_compression_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "pure_downnorm_compression_curve.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2), encoding="utf-8"
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
    print(f"[downnorm-curve] OK rows=4 output={output}")


if __name__ == "__main__":
    main()
