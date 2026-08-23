#!/usr/bin/env python3
"""Build the deterministic target-6 certificate/down-norm hybrid frontier."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_provenance import file_sha256
from src.moe_set_certification import (
    certificate_for_plan,
    clone_plan_with_selection,
    load_score_bundle,
    matched_plan_validation,
    refine_with_certificate_slack,
    selected_by_layer,
)


SLACKS = (0.00, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.021436)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan_name(rho: float) -> str:
    if rho == 0:
        return "ellipsoid_slack0"
    percent = f"{rho * 100:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"downnorm_refinement_slack{percent}"


def selection_overlap(left: dict[int, set[int]], right: dict[int, set[int]]) -> dict:
    """Summarize matched-budget channel identity overlap."""
    if set(left) != set(right):
        raise ValueError("selection layer sets differ")
    intersection = sum(len(left[layer] & right[layer]) for layer in left)
    union = sum(len(left[layer] | right[layer]) for layer in left)
    left_total = sum(len(values) for values in left.values())
    right_total = sum(len(values) for values in right.values())
    if left_total != right_total:
        raise ValueError("selection budgets differ")
    return {
        "selected_channel_intersection": intersection,
        "selected_channel_replacements": left_total - intersection,
        "selected_channel_jaccard": intersection / union if union else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ellipsoid-plan", required=True)
    parser.add_argument("--down-norm-plan", required=True)
    parser.add_argument("--score-bundle", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--matched-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = [Path(args.ellipsoid_plan), Path(args.down_norm_plan),
             Path(args.score_bundle), Path(args.score_manifest),
             Path(args.matched_validation)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"refusing to overwrite frontier: {output_dir}")

    ellipsoid_plan = json.loads(paths[0].read_text(encoding="utf-8"))
    down_plan = json.loads(paths[1].read_text(encoding="utf-8"))
    validation = json.loads(paths[4].read_text(encoding="utf-8"))
    if not validation.get("strict_gate_passed"):
        raise ValueError("matched-plan validation gate did not pass")
    matched_plan_validation(
        {"ellipsoid": ellipsoid_plan, "down_norm": down_plan},
        expected_total=2288, expected_alignment=16,
    )
    if args.dry_run:
        print(
            f"[hybrid-frontier] DRY RUN plans={len(SLACKS) + 1} "
            f"slacks={SLACKS} "
            f"output={output_dir}"
        )
        return

    scores = load_score_bundle(paths[2])
    ellipsoid_certificate = certificate_for_plan(ellipsoid_plan, scores)
    down_certificate = certificate_for_plan(down_plan, scores)
    output_dir.mkdir(parents=True)
    plans_dir = output_dir / "plans"
    plans_dir.mkdir()

    report_json = {
        "schema_version": 1,
        "interpretation": {
            "expert_channel_ellipsoid": "certified local upper bound",
            "strict_layer": "max_e sum_{j in S_l} b[l,e,j]",
            "strict_global": "unpropagated sum_l strict_layer",
            "p95": "reported heuristic summary, not a uniform certificate",
        },
        "score_bundle": str(paths[2]),
        "score_bundle_sha256": file_sha256(str(paths[2])),
        "source_plans": {
            "ellipsoid": {"path": str(paths[0]), "sha256": file_sha256(str(paths[0]))},
            "down_norm": {"path": str(paths[1]), "sha256": file_sha256(str(paths[1]))},
        },
        "pure_ellipsoid": ellipsoid_certificate,
        "pure_down_norm": down_certificate,
    }
    (output_dir / "set_level_certificates.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8"
    )
    layer_rows = []
    expert_rows = []
    for label, report in (("pure_ellipsoid", ellipsoid_certificate),
                          ("pure_down_norm", down_certificate)):
        layer_rows.extend({"plan": label, **row} for row in report["layers"])
        expert_rows.extend({"plan": label, **row} for row in report["expert_set_bounds"])
    write_csv(output_dir / "set_level_certificate_layers.csv", layer_rows)
    write_csv(output_dir / "set_level_certificate_experts.csv", expert_rows)
    certificate_md = [
        "| Plan | Strict unpropagated certificate | Older channelwise-max expression | Strict / older | Inequality violations |",
        "|---|---:|---:|---:|---:|",
    ]
    certificate_tex = [
        "\\begin{tabular}{lrrrr}",
        "Plan & Strict certificate & Older expression & Ratio & Violations \\\\",
        "\\hline",
    ]
    for label, report in (("pure_ellipsoid", ellipsoid_certificate),
                          ("pure_down_norm", down_certificate)):
        certificate_md.append(
            f"| {label} | {report['strict_global_unpropagated_certificate']:.7g} | "
            f"{report['older_global_channelwise_max_certificate']:.7g} | "
            f"{report['strict_over_older']:.5f} | {report['inequality_violations']} |"
        )
        label_tex = label.replace("_", "\\_")
        certificate_tex.append(
            f"{label_tex} & {report['strict_global_unpropagated_certificate']:.7g} & "
            f"{report['older_global_channelwise_max_certificate']:.7g} & "
            f"{report['strict_over_older']:.5f} & {report['inequality_violations']} \\\\"
        )
    certificate_tex.append("\\end{tabular}")
    (output_dir / "set_level_certificates.md").write_text(
        "\n".join(certificate_md) + "\n", encoding="utf-8"
    )
    (output_dir / "set_level_certificates.tex").write_text(
        "\n".join(certificate_tex) + "\n", encoding="utf-8"
    )

    frontier_rows = []
    plan_manifest = []
    selections = {}
    ellipsoid_selected = {
        key: set(value) for key, value in selected_by_layer(ellipsoid_plan).items()
    }
    down_selected = {
        key: set(value) for key, value in selected_by_layer(down_plan).items()
    }
    for rho in SLACKS:
        name = plan_name(rho)
        if rho == 0.0:
            selected = {k: set(v) for k, v in selected_by_layer(ellipsoid_plan).items()}
            audit = {
                "rho": 0.0,
                "seed": args.seed,
                "randomness_used": False,
                "accepted_swap_count": 0,
                "base_strict_certificate": ellipsoid_certificate[
                    "strict_global_unpropagated_certificate"
                ],
                "strict_certificate_threshold": ellipsoid_certificate[
                    "strict_global_unpropagated_certificate"
                ],
                "final_strict_certificate": ellipsoid_certificate[
                    "strict_global_unpropagated_certificate"
                ],
                "base_down_norm_objective": ellipsoid_certificate[
                    "normalized_down_norm_objective"
                ],
                "final_down_norm_objective": ellipsoid_certificate[
                    "normalized_down_norm_objective"
                ],
                "tie_breaking": "not applicable; exact source plan",
            }
        else:
            selected, audit = refine_with_certificate_slack(
                ellipsoid_plan, down_plan, scores, rho, seed=args.seed
            )
        report = certificate_for_plan(
            clone_plan_with_selection(
                ellipsoid_plan, selected, plan_name=name,
                metadata={"frontier_audit": audit},
            ), scores
        )
        plan = clone_plan_with_selection(
            ellipsoid_plan,
            selected,
            plan_name=name,
            metadata={
                "schema_version": 1,
                "certificate_slack": rho,
                "strict_certificate": report["strict_global_unpropagated_certificate"],
                "strict_certificate_threshold": audit["strict_certificate_threshold"],
                "normalized_down_norm_objective": report["normalized_down_norm_objective"],
                "source_ellipsoid_plan_sha256": file_sha256(str(paths[0])),
                "source_down_norm_plan_sha256": file_sha256(str(paths[1])),
                "score_bundle_sha256": file_sha256(str(paths[2])),
                "frontier_audit": audit,
            },
        )
        plan_path = plans_dir / f"{name}.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        selection_signature = {
            str(layer): sorted(values) for layer, values in sorted(selected.items())
        }
        selection_hash = __import__("hashlib").sha256(
            json.dumps(selection_signature, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        selections[name] = selection_hash
        overlap_ellipsoid = selection_overlap(selected, ellipsoid_selected)
        overlap_down = selection_overlap(selected, down_selected)
        row = {
            "plan": name,
            "certificate_slack": rho,
            "strict_certificate": report["strict_global_unpropagated_certificate"],
            "certificate_change_vs_ellipsoid_pct": 100.0 * (
                report["strict_global_unpropagated_certificate"]
                / ellipsoid_certificate["strict_global_unpropagated_certificate"] - 1.0
            ),
            "normalized_down_norm_objective": report["normalized_down_norm_objective"],
            "removed_layer_channels": 2288,
            "removed_expert_neurons": 292864,
            "actual_pruning_pct": float(plan.get("actual_pct", plan.get("target_pct", 0.0))),
            "accepted_swaps": int(audit["accepted_swap_count"]),
            "selection_sha256": selection_hash,
            "plan_path": str(plan_path),
            "plan_sha256": file_sha256(str(plan_path)),
            "unconstrained_endpoint": False,
            **{f"vs_ellipsoid_{key}": value
               for key, value in overlap_ellipsoid.items()},
            **{f"vs_down_norm_{key}": value
               for key, value in overlap_down.items()},
        }
        frontier_rows.append(row)
        plan_manifest.append(row.copy())

    pure_selected = {k: set(v) for k, v in selected_by_layer(down_plan).items()}
    pure_name = "pure_down_norm"
    pure_plan = clone_plan_with_selection(
        ellipsoid_plan, pure_selected, plan_name=pure_name,
        metadata={
            "schema_version": 1,
            "certificate_slack": None,
            "unconstrained_endpoint": True,
            "strict_certificate": down_certificate["strict_global_unpropagated_certificate"],
            "normalized_down_norm_objective": down_certificate["normalized_down_norm_objective"],
            "source_down_norm_plan_sha256": file_sha256(str(paths[1])),
            "score_bundle_sha256": file_sha256(str(paths[2])),
        },
    )
    pure_path = plans_dir / f"{pure_name}.json"
    pure_path.write_text(json.dumps(pure_plan, indent=2), encoding="utf-8")
    signature = {str(k): sorted(v) for k, v in sorted(pure_selected.items())}
    selection_hash = __import__("hashlib").sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    pure_row = {
        "plan": pure_name,
        "certificate_slack": "unconstrained",
        "strict_certificate": down_certificate["strict_global_unpropagated_certificate"],
        "certificate_change_vs_ellipsoid_pct": 100.0 * (
            down_certificate["strict_global_unpropagated_certificate"]
            / ellipsoid_certificate["strict_global_unpropagated_certificate"] - 1.0
        ),
        "normalized_down_norm_objective": down_certificate["normalized_down_norm_objective"],
        "removed_layer_channels": 2288,
        "removed_expert_neurons": 292864,
        "actual_pruning_pct": float(pure_plan.get("actual_pct", pure_plan.get("target_pct", 0.0))),
        "accepted_swaps": "endpoint",
        "selection_sha256": selection_hash,
        "plan_path": str(pure_path),
        "plan_sha256": file_sha256(str(pure_path)),
        "unconstrained_endpoint": True,
        **{f"vs_ellipsoid_{key}": value for key, value in
           selection_overlap(pure_selected, ellipsoid_selected).items()},
        **{f"vs_down_norm_{key}": value for key, value in
           selection_overlap(pure_selected, down_selected).items()},
    }
    frontier_rows.append(pure_row)
    plan_manifest.append(pure_row.copy())

    for index, row in enumerate(frontier_rows):
        dominated = any(
            other["strict_certificate"] <= row["strict_certificate"]
            and other["normalized_down_norm_objective"] <= row["normalized_down_norm_objective"]
            and (
                other["strict_certificate"] < row["strict_certificate"]
                or other["normalized_down_norm_objective"] < row["normalized_down_norm_objective"]
            )
            for other_index, other in enumerate(frontier_rows)
            if other_index != index
        )
        row["certificate_objective_pareto_optimal"] = not dominated
        plan_manifest[index]["certificate_objective_pareto_optimal"] = not dominated
    distinct = len({row["selection_sha256"] for row in frontier_rows})
    duplicate_groups = {}
    for row in frontier_rows:
        duplicate_groups.setdefault(row["selection_sha256"], []).append(row["plan"])
    duplicates = [values for values in duplicate_groups.values() if len(values) > 1]

    # PPL is run once per genuinely distinct certificate/objective-Pareto
    # selection. Prefer the named endpoints as representatives when a bounded
    # slack reaches exactly the same selection.
    representative_by_selection = {}
    for row in sorted(
        frontier_rows,
        key=lambda value: (
            0 if value["plan"] in {"ellipsoid_slack0", "pure_down_norm"} else 1,
            float(value["certificate_slack"])
            if value["certificate_slack"] != "unconstrained" else float("inf"),
            value["plan"],
        ),
    ):
        if not row["certificate_objective_pareto_optimal"]:
            continue
        representative_by_selection.setdefault(row["selection_sha256"], row["plan"])
    for index, row in enumerate(frontier_rows):
        representative = representative_by_selection.get(row["selection_sha256"], "")
        row["evaluation_representative"] = representative
        row["evaluate_ppl"] = row["plan"] == representative
        plan_manifest[index]["evaluation_representative"] = representative
        plan_manifest[index]["evaluate_ppl"] = row["evaluate_ppl"]

    write_csv(output_dir / "hybrid_frontier.csv", frontier_rows)
    frontier_json = {
        "schema_version": 1,
        "seed": args.seed,
        "predefined_slacks": list(SLACKS),
        "plans": plan_manifest,
        "distinct_selection_count": distinct,
        "identical_selection_groups": duplicates,
        "thresholds_were_not_adapted": True,
        "ppl_evaluation_plan_count": sum(
            bool(row["evaluate_ppl"]) for row in plan_manifest
        ),
        "source_plans": report_json["source_plans"],
        "matched_plan_validation": {
            "path": str(paths[4]),
            "sha256": file_sha256(str(paths[4])),
        },
        "allocation_plan": validation["plans"]["rmsnorm_bound"],
    }
    (output_dir / "hybrid_frontier.json").write_text(
        json.dumps(frontier_json, indent=2), encoding="utf-8"
    )
    frontier_md = [
        "| Plan | Slack | Strict certificate | Certificate change | Down-norm objective | Replacements vs ellipsoid | Jaccard vs ellipsoid | Replacements vs down-norm | Jaccard vs down-norm | Swaps | Selection | Pareto | PPL representative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    frontier_tex = [
        "\\begin{tabular}{lrrrrrr}",
        "Plan & Slack & Certificate & $\\Delta$cert. & $D(S)$ & Swaps & Pareto \\\\",
        "\\hline",
    ]
    for row in frontier_rows:
        frontier_md.append(
            f"| {row['plan']} | {row['certificate_slack']} | {row['strict_certificate']:.7g} | "
            f"{row['certificate_change_vs_ellipsoid_pct']:+.3f}% | "
            f"{row['normalized_down_norm_objective']:.7g} | "
            f"{row['vs_ellipsoid_selected_channel_replacements']} | "
            f"{row['vs_ellipsoid_selected_channel_jaccard']:.5f} | "
            f"{row['vs_down_norm_selected_channel_replacements']} | "
            f"{row['vs_down_norm_selected_channel_jaccard']:.5f} | "
            f"{row['accepted_swaps']} | `{row['selection_sha256'][:12]}` | "
            f"{row['certificate_objective_pareto_optimal']} | "
            f"{row['evaluation_representative']} |"
        )
        plan_tex = row["plan"].replace("_", "\\_")
        frontier_tex.append(
            f"{plan_tex} & {row['certificate_slack']} & {row['strict_certificate']:.7g} & "
            f"{row['certificate_change_vs_ellipsoid_pct']:+.3f}\\% & "
            f"{row['normalized_down_norm_objective']:.7g} & {row['accepted_swaps']} & "
            f"{row['certificate_objective_pareto_optimal']} \\\\"
        )
    frontier_tex.append("\\end{tabular}")
    (output_dir / "hybrid_frontier.md").write_text(
        "\n".join(frontier_md) + "\n", encoding="utf-8"
    )
    (output_dir / "hybrid_frontier.tex").write_text(
        "\n".join(frontier_tex) + "\n", encoding="utf-8"
    )
    print(
        f"[hybrid-frontier] OK plans={len(frontier_rows)} distinct={distinct} "
        f"duplicates={duplicates} output={output_dir}"
    )


if __name__ == "__main__":
    main()
