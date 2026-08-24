#!/usr/bin/env python3
"""Assemble the immutable certified-hybrid closure packet for paper writing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


SOURCE_PATTERNS = {
    "certificate_frontier": (
        "hybrid_frontier.csv", "hybrid_frontier.json",
        "hybrid_frontier.md", "hybrid_frontier.tex",
        "set_level_certificates.json", "set_level_certificates.md",
        "set_level_certificates.tex", "set_level_certificate_layers.csv",
    ),
    "ppl": (
        "hybrid_ppl_pareto.csv", "hybrid_ppl_pareto.json",
        "hybrid_ppl_pareto.md", "hybrid_ppl_pareto.tex",
    ),
    "downnorm_curve": (
        "pure_downnorm_compression_curve.csv",
        "pure_downnorm_compression_curve.json",
        "pure_downnorm_compression_curve.md",
        "pure_downnorm_compression_curve.tex",
        "pure_downnorm_plan_nesting.csv", "pure_downnorm_plan_nesting.json",
        "pure_downnorm_plan_nesting.md", "pure_downnorm_plan_nesting.tex",
        "pure_downnorm_adjacent_paired_macro.csv",
        "pure_downnorm_adjacent_paired_macro.json",
        "pure_downnorm_adjacent_paired_macro.md",
        "pure_downnorm_adjacent_paired_macro.tex",
    ),
    "downstream": (
        "downstream_benchmark_table.csv", "downstream_benchmark_table.json",
        "downstream_benchmark_table.md", "downstream_benchmark_table.tex",
        "downstream_paired_comparisons.csv",
        "downstream_paired_comparisons.md",
        "downstream_paired_comparisons.tex",
        "downstream_statistical_audit.json",
    ),
    "checkpoints": (
        "checkpoint_table.csv", "checkpoint_table.json",
        "checkpoint_table.md", "checkpoint_table.tex",
    ),
    "systems": (
        "systems_benchmark_table.csv", "systems_benchmark_table.json",
        "systems_benchmark_table.md", "systems_benchmark_table.tex",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _certificate_ratio(row: dict) -> float:
    """Return strict-certificate ratio from the persisted frontier schema."""
    if "certificate_ratio_vs_ellipsoid" in row:
        return float(row["certificate_ratio_vs_ellipsoid"])
    return 1.0 + float(row["certificate_change_vs_ellipsoid_pct"]) / 100.0


def copy_group(
    source: Path, output: Path, group: str, names: tuple[str, ...],
) -> list[Path]:
    destination = output / group
    destination.mkdir()
    copied = []
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        target = destination / name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def hybrid_outcome(frontier: dict, downstream_rows: list[dict]) -> dict:
    """Apply the frozen project stop/go rules to intermediate checkpoints."""
    macro = {
        row["label"]: float(row["accuracy"])
        for row in downstream_rows if row.get("task") == "macro_average"
    }
    ellipsoid_label = "rmsnorm_alloc__ellipsoid_rank__p95__target6"
    downnorm_label = "rmsnorm_alloc__downnorm_rank__p95__target6"
    if ellipsoid_label not in macro or downnorm_label not in macro:
        raise ValueError("downstream table lacks target-6 ellipsoid/down-norm endpoints")
    frontier_by_name = {row["plan"]: row for row in frontier["plans"]}
    down_certificate_ratio = _certificate_ratio(frontier_by_name["pure_down_norm"])
    candidates = []
    for label, accuracy in macro.items():
        prefix, suffix = "certified_hybrid__", "__target6"
        if not label.startswith(prefix) or not label.endswith(suffix):
            continue
        plan_name = label[len(prefix):-len(suffix)]
        if plan_name not in frontier_by_name:
            raise ValueError(f"downstream hybrid missing from frontier: {plan_name}")
        row = frontier_by_name[plan_name]
        certificate_ratio = _certificate_ratio(row)
        certificate_improvement_vs_downnorm = (
            down_certificate_ratio - certificate_ratio
        )
        accuracy_gap_vs_downnorm = accuracy - macro[downnorm_label]
        accuracy_recovery_vs_ellipsoid = accuracy - macro[ellipsoid_label]
        within_downnorm_0p2_points = accuracy_gap_vs_downnorm >= -0.002
        meaningfully_stronger_certificate = (
            certificate_improvement_vs_downnorm >= 0.0025 - 1e-12
        )
        recovery_rule = (
            accuracy_recovery_vs_ellipsoid >= 0.005
            and float(row["certificate_slack"]) <= 0.25
        )
        candidates.append({
            "label": label,
            "plan": plan_name,
            "macro_accuracy": accuracy,
            "macro_difference_vs_down_norm": accuracy_gap_vs_downnorm,
            "macro_recovery_vs_ellipsoid": accuracy_recovery_vs_ellipsoid,
            "certificate_ratio_vs_ellipsoid": certificate_ratio,
            "certificate_ratio_improvement_vs_down_norm": (
                certificate_improvement_vs_downnorm
            ),
            "criterion_downnorm_proximity_and_certificate": (
                within_downnorm_0p2_points and meaningfully_stronger_certificate
            ),
            "criterion_accuracy_recovery": recovery_rule,
        })
    successful = [
        row for row in candidates
        if row["criterion_downnorm_proximity_and_certificate"]
        or row["criterion_accuracy_recovery"]
    ]
    if successful:
        category = "The certified hybrid successfully recovers practical accuracy."
        outcome_code = "success"
    elif any(row["macro_recovery_vs_ellipsoid"] > 0 for row in candidates):
        category = "The hybrid shows a partial certificate–accuracy trade-off."
        outcome_code = "partial"
    else:
        category = (
            "The hybrid fails, and the project proceeds as a "
            "diagnostic/certification paper."
        )
        outcome_code = "failure"
    return {
        "outcome": outcome_code,
        "conclusion_sentence": category,
        "success_criterion_met": bool(successful),
        "certificate_improvement_threshold_ratio": 0.0025,
        "accuracy_proximity_threshold": 0.002,
        "accuracy_recovery_threshold": 0.005,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", required=True)
    parser.add_argument("--ppl-dir", required=True)
    parser.add_argument("--downnorm-curve-dir", required=True)
    parser.add_argument("--downstream-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--systems-dir", required=True)
    parser.add_argument("--matched-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = {
        "certificate_frontier": Path(args.frontier_dir),
        "ppl": Path(args.ppl_dir),
        "downnorm_curve": Path(args.downnorm_curve_dir),
        "downstream": Path(args.downstream_dir),
        "checkpoints": Path(args.checkpoint_dir),
        "systems": Path(args.systems_dir),
    }
    validation_path = Path(args.matched_validation)
    for group, directory in sources.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{group}: {directory}")
        for name in SOURCE_PATTERNS[group]:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)

    systems_rows = read_csv(sources["systems"] / "systems_benchmark_table.csv")
    baseline = [row for row in systems_rows if row["label"] == "baseline_unpruned"]
    target = [
        row for row in systems_rows
        if row["label"] == "rmsnorm_alloc__downnorm_rank__p95__target6"
    ]
    if len(baseline) != 5 or len(target) != 5:
        raise ValueError("systems packet requires five matched baseline/target-6 cases")
    if any(not row["prefill_throughput_gain_ci95_lower_pct"] for row in target):
        raise ValueError("systems uncertainty intervals are missing")
    consistent_prefill = all(
        float(row["prefill_throughput_gain_ci95_lower_pct"]) > 0 for row in target
    )
    consistent_decode = all(
        float(row["decode_throughput_gain_ci95_lower_pct"]) > 0 for row in target
    )
    load_hbm_reduction = float(target[0]["load_hbm_reduction_vs_baseline_pct"])
    peak_hbm_reduction_values = sorted({
        round(float(row["peak_hbm_reduction_vs_baseline_pct"]), 10)
        for row in target
    })
    checkpoint_rows = read_csv(sources["checkpoints"] / "checkpoint_table.csv")
    checkpoint_labels = {row["label"] for row in checkpoint_rows}
    required_checkpoints = {
        "baseline_unpruned",
        "rmsnorm_alloc__downnorm_rank__p95__target2",
        "rmsnorm_alloc__downnorm_rank__p95__target4",
        "rmsnorm_alloc__downnorm_rank__p95__target6",
        "rmsnorm_alloc__downnorm_rank__p95__target8",
    }
    if not required_checkpoints.issubset(checkpoint_labels):
        raise ValueError("checkpoint table lacks the final down-norm curve")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    frontier = json.loads(
        (sources["certificate_frontier"] / "hybrid_frontier.json").read_text(
            encoding="utf-8"
        )
    )
    resolved_frontier_plans = []
    seen_plan_filenames = set()
    for row in frontier["plans"]:
        source_plan = Path(row["plan_path"])
        if not source_plan.is_file():
            fallback = sources["certificate_frontier"] / "plans" / source_plan.name
            if not fallback.is_file():
                raise FileNotFoundError(source_plan)
            source_plan = fallback
        if sha256(source_plan) != row["plan_sha256"]:
            raise ValueError(f"frontier plan hash mismatch: {source_plan}")
        if source_plan.name in seen_plan_filenames:
            raise ValueError(f"duplicate frontier plan filename: {source_plan.name}")
        seen_plan_filenames.add(source_plan.name)
        resolved_frontier_plans.append(source_plan)
    downstream_audit = json.loads(
        (sources["downstream"] / "downstream_statistical_audit.json").read_text(
            encoding="utf-8"
        )
    )
    systems_payload = json.loads(
        (sources["systems"] / "systems_benchmark_table.json").read_text(
            encoding="utf-8"
        )
    )
    downstream_rows = read_csv(
        sources["downstream"] / "downstream_benchmark_table.csv"
    )
    hybrid_decision = hybrid_outcome(frontier, downstream_rows)
    settings = {
        "schema_version": 1,
        "frozen_paper_claims": {
            "practical_method": "RMSNorm-bound allocation plus down-norm ranking",
            "conservative_operating_point": (
                "approximately 4.12% whole-model reduction with no detected "
                "significant seven-task macro-accuracy difference"
            ),
            "higher_compression_operating_point": (
                "approximately 5.89% whole-model reduction with a 0.953-point "
                "seven-task macro-accuracy loss"
            ),
            "ellipsoid_contribution": (
                "calibration-free local and strict set-level MoE certificate; "
                "PPL-preserving selector evidence, but lower downstream accuracy "
                "than down-norm at matched target 6"
            ),
            "systems": (
                "physical storage reduction demonstrated; consistent runtime "
                "acceleration not demonstrated"
            ),
        },
        "matched_plan_validation": validation,
        "fine_frontier_protocol": {
            "seed": frontier["seed"],
            "predefined_slacks": frontier["predefined_slacks"],
            "distinct_selection_count": frontier["distinct_selection_count"],
            "ppl_evaluation_plan_count": frontier["ppl_evaluation_plan_count"],
            "thresholds_were_not_adapted": frontier["thresholds_were_not_adapted"],
            "source_plans": frontier["source_plans"],
        },
        "downstream_statistics": downstream_audit,
        "systems_uncertainty": systems_payload["uncertainty"],
        "systems_environment": [
            {
                key: run.get(key) for key in (
                    "label", "dtype", "torch_version", "cuda_runtime_version",
                    "transformers_version", "inference_engine", "nvidia_smi",
                    "attention_implementation", "kernel_environment",
                )
            }
            for run in systems_payload["raw_runs"]
        ],
        "systems_decision": {
            "loaded_hbm_reduction_pct": load_hbm_reduction,
            "peak_hbm_reduction_pct_by_case": peak_hbm_reduction_values,
            "prefill_gain_ci_positive_in_all_cases": consistent_prefill,
            "decode_gain_ci_positive_in_all_cases": consistent_decode,
        },
        "certified_hybrid_decision": hybrid_decision,
    }
    output = Path(args.output_dir)
    if args.dry_run:
        print(
            f"[hybrid-final-packet] DRY RUN groups={len(sources)} "
            f"frontier_plans={len(frontier['plans'])} "
            f"consistent_prefill={consistent_prefill} "
            f"consistent_decode={consistent_decode}"
        )
        return
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    copied = []
    for group, source in sources.items():
        copied.extend(copy_group(source, output, group, SOURCE_PATTERNS[group]))
    plan_output = output / "certificate_frontier" / "plans"
    plan_output.mkdir()
    for source_plan in resolved_frontier_plans:
        target_plan = plan_output / source_plan.name
        shutil.copy2(source_plan, target_plan)
        copied.append(target_plan)
    validation_target = output / "matched_plan_validation.json"
    shutil.copy2(validation_path, validation_target)
    copied.append(validation_target)
    settings_path = output / "exact_experimental_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    copied.append(settings_path)

    hbm_text = (
        f"Loaded HBM decreased by {load_hbm_reduction:.3f}% in the measured run. "
        f"Peak-HBM changes across cases were {peak_hbm_reduction_values}."
    )
    conclusion = (
        "# Final certified-hybrid paper conclusion\n\n"
        "The practical method is **RMSNorm-bound global allocation followed by "
        "down-norm within-layer ranking**, with p95 expert aggregation, aligned "
        "physical tensor pruning, and no weight updates.\n\n"
        "The conservative operating point removes approximately 4.12% of total "
        "model parameters and has no detected significant seven-task macro-accuracy "
        "difference. The higher-compression point removes approximately 5.89% and "
        "has a 0.953-point macro-accuracy loss.\n\n"
        "The ellipsoid theorem contributes a calibration-free expert-channel bound "
        "and strict set-level local MoE certificate. Its ranking preserves perplexity "
        "better at the matched target-6 budget, but down-norm preserves downstream "
        "accuracy better. The reported certificate is local to the MoE calculation "
        "and is not a propagated guarantee for the full Transformer.\n\n"
        f"{hybrid_decision['conclusion_sentence']}\n\n"
        f"Physical checkpoint storage reduction is demonstrated. {hbm_text} "
        "Prefill improvements are configuration-dependent and decode improvement is "
        "not established; consistent runtime acceleration is therefore not "
        "demonstrated.\n\n"
        "Selector experimentation stops after the predefined fine-grid closure. "
        "The project now moves directly to paper writing.\n"
    )
    conclusion_path = output / "final_conclusion.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    copied.append(conclusion_path)
    conclusion_tex = output / "final_conclusion.tex"
    conclusion_tex.write_text(
        "\\section{Final certified-hybrid conclusion}\n"
        "The practical method uses RMSNorm-bound global allocation and down-norm "
        "within-layer ranking with aligned physical pruning. The conservative "
        "operating point reduces whole-model parameters by approximately 4.12\\% "
        "with no detected significant macro-accuracy difference. The 5.89\\% "
        "operating point has a 0.953-point macro-accuracy loss. The ellipsoid "
        "theorem supplies a calibration-free local and strict set-level MoE "
        "certificate. Physical storage reduction is demonstrated, while consistent "
        "runtime acceleration is not demonstrated.\n",
        encoding="utf-8",
    )
    copied.append(conclusion_tex)
    readme = output / "README.md"
    readme.write_text(
        "# Final paper packet\n\n"
        "This directory is the immutable closure packet for the certified-hybrid "
        "milestone. Tables are grouped by source category; all files are indexed "
        "and hashed in `FINAL_PACKET_MANIFEST.json`. No additional selector sweep "
        "is authorized after this packet.\n",
        encoding="utf-8",
    )
    copied.append(readme)
    manifest_path = output / "FINAL_PACKET_MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "stop_experimentation": True,
        "next_phase": "paper writing",
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(copied)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[hybrid-final-packet] OK files={len(copied)} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
