#!/usr/bin/env python3
"""Assemble the immutable certified-hybrid closure packet for paper writing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import shutil
import sys
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

FINAL_HYBRID_LABEL = (
    "certified_hybrid__downnorm_refinement_slack0p25__target6"
)
ELLIPSOID_LABEL = "rmsnorm_alloc__ellipsoid_rank__p95__target6"
DOWNNORM_LABEL = "rmsnorm_alloc__downnorm_rank__p95__target6"
PROPOSED_METHOD = (
    "RMSNorm-bound global allocation + certificate-constrained down-norm "
    "refinement with 0.25% ellipsoid-certificate slack"
)
ENTRY_POINTS = (
    "scripts/build_moe_certified_hybrid_frontier.py",
    "scripts/run_moe_certified_hybrid_ppl.sh",
    "scripts/run_paper_v3_checkpoint_export.sh",
    "scripts/export_verify_moe_checkpoint.py",
    "scripts/run_paper_v3_downstream.sh",
    "scripts/run_paper_v3_lm_eval.py",
    "scripts/summarize_paper_v3_downstream.py",
    "scripts/summarize_paper_v3_checkpoints.py",
    "scripts/summarize_paper_v3_systems.py",
    "scripts/summarize_pure_downnorm_curve.py",
    "scripts/build_moe_certified_hybrid_final_packet.py",
    "src/moe_set_certification.py",
)
CHECKPOINT_EXTRA_FIELDS = (
    "checkpoint_verification_sha256", "source_model_revision",
    "tokenizer_revision", "tokenizer_selected_mode",
    "tokenizer_fix_mistral_regex", "tokenizer_files_combined_sha256",
    "tokenizer_audit_sha256", "tokenizer_audit_passed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _git(args: list[str], *, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), *args],
        text=not binary,
    )


def code_provenance() -> tuple[dict, bytes, str, list[dict]]:
    repo = Path(__file__).resolve().parents[1]
    commit = str(_git(["rev-parse", "HEAD"])).strip()
    status = str(_git([
        "status", "--porcelain", "--untracked-files=no",
    ])).strip()
    patch = bytes(_git(["diff", "--binary", "HEAD", "--"], binary=True))
    if bool(status) != bool(patch):
        raise ValueError(
            "tracked worktree status and reproducibility patch disagree"
        )
    entry_points = []
    for relative in ENTRY_POINTS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entry_points.append({
            "path": relative, "sha256": sha256(path), "bytes": path.stat().st_size,
        })
    dependency_lock = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True,
    )
    provenance = {
        "git_commit_sha": commit,
        "tracked_worktree_clean": not bool(status),
        "tracked_status_porcelain": status.splitlines(),
        "untracked_files_excluded_from_status": True,
        "dirty_patch_sha256": (
            hashlib.sha256(patch).hexdigest() if patch else ""
        ),
        "python_executable": sys.executable,
        "dependency_environment_lock_sha256": hashlib.sha256(
            dependency_lock.encode("utf-8")
        ).hexdigest(),
        "entry_points": entry_points,
    }
    return provenance, patch, dependency_lock, entry_points


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


def _merge_checkpoint_rows(*groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for rows in groups:
        for incoming in rows:
            label = incoming["label"]
            if label not in merged:
                merged[label] = dict(incoming)
                continue
            current = merged[label]
            for key, value in incoming.items():
                previous = current.get(key, "")
                if previous not in ("", None) and value not in ("", None):
                    if str(previous) != str(value):
                        raise ValueError(
                            f"checkpoint table conflict {label}.{key}: "
                            f"{previous!r} != {value!r}"
                        )
                elif value not in ("", None):
                    current[key] = value
    return sorted(
        merged.values(),
        key=lambda row: (float(row.get("target_pct") or 0), row["label"]),
    )


def _paired_macro(
    rows: list[dict], first_label: str, second_label: str,
) -> dict:
    matches = [row for row in rows if (
        row.get("comparison_type") == "certified_hybrid_attribution"
        and row.get("first_label") == first_label
        and row.get("second_label") == second_label
        and row.get("task") == "macro_average"
    )]
    if len(matches) != 1:
        raise ValueError(
            f"expected one paired macro comparison: {first_label} vs "
            f"{second_label}; found {len(matches)}"
        )
    return matches[0]


def _validate_final_hybrid_statistics(paired_rows: list[dict]) -> dict:
    versus_ellipsoid = _paired_macro(
        paired_rows, FINAL_HYBRID_LABEL, ELLIPSOID_LABEL,
    )
    versus_downnorm = _paired_macro(
        paired_rows, FINAL_HYBRID_LABEL, DOWNNORM_LABEL,
    )
    ellipsoid_points = 100.0 * float(versus_ellipsoid["accuracy_difference"])
    downnorm_points = 100.0 * float(versus_downnorm["accuracy_difference"])
    if abs(ellipsoid_points - 0.895) > 0.005:
        raise ValueError(
            f"final hybrid vs ellipsoid changed: {ellipsoid_points:.6f} points"
        )
    if abs(downnorm_points - 0.089) > 0.005:
        raise ValueError(
            f"final hybrid vs down-norm changed: {downnorm_points:.6f} points"
        )
    if not _truth(versus_ellipsoid.get("holm_significant_0_05")):
        raise ValueError("hybrid vs ellipsoid must remain Holm-significant")
    if _truth(versus_downnorm.get("holm_significant_0_05")):
        raise ValueError("hybrid vs down-norm must not be called significant")
    return {
        "hybrid_minus_ellipsoid_macro_accuracy_points": ellipsoid_points,
        "hybrid_minus_ellipsoid_ci95_points": [
            100.0 * float(versus_ellipsoid["ci95_lower"]),
            100.0 * float(versus_ellipsoid["ci95_upper"]),
        ],
        "hybrid_minus_ellipsoid_holm_p": float(
            versus_ellipsoid["holm_adjusted_p_value"]
        ),
        "hybrid_minus_ellipsoid_holm_significant": True,
        "hybrid_minus_downnorm_macro_accuracy_points": downnorm_points,
        "hybrid_minus_downnorm_ci95_points": [
            100.0 * float(versus_downnorm["ci95_lower"]),
            100.0 * float(versus_downnorm["ci95_upper"]),
        ],
        "hybrid_minus_downnorm_holm_p": float(
            versus_downnorm["holm_adjusted_p_value"]
        ),
        "hybrid_minus_downnorm_holm_significant": False,
        "interpretation": (
            "The 0.25%-slack hybrid matches pure down-norm accuracy within "
            "uncertainty while retaining a tighter ellipsoid certificate; it "
            "is not claimed to outperform pure down-norm."
        ),
    }


def _validate_and_enrich_final_checkpoint(
    rows: list[dict], frontier: dict, tokenizer_audit_path: Path,
) -> tuple[dict, dict]:
    matches = [row for row in rows if row["label"] == FINAL_HYBRID_LABEL]
    if len(matches) != 1:
        raise ValueError("combined checkpoint table lacks the final hybrid")
    row = matches[0]
    verification_path = Path(row["checkpoint_dir"]) / "checkpoint_verification.json"
    if not verification_path.is_file():
        raise FileNotFoundError(verification_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    frontier_by_name = {item["plan"]: item for item in frontier["plans"]}
    frontier_row = frontier_by_name["downnorm_refinement_slack0p25"]
    expected = {
        "label": FINAL_HYBRID_LABEL,
        "removed_layer_channels": int(row["removed_layer_channels"]),
        "removed_expert_neurons": int(row["removed_expert_neurons"]),
        "plan_sha256": frontier_row["plan_sha256"],
        "successful_reload": True,
        "exact_logits_after_reload": True,
        "no_hidden_original_width_padding": True,
    }
    for key, value in expected.items():
        if verification.get(key) != value:
            raise ValueError(
                f"final hybrid verification mismatch {key}: "
                f"{verification.get(key)!r} != {value!r}"
            )
    if float(verification["max_logit_difference"]) != 0.0:
        raise ValueError("final hybrid reload logits are not exact")
    counts = verification["parameters_reloaded"]
    table_checks = {
        "total_parameters": counts["total"],
        "moe_expert_parameters": counts["moe_experts"],
        "serialized_weight_bytes": verification["serialized_weight_bytes"],
        "checkpoint_payload_bytes": verification[
            "checkpoint_payload_bytes_excluding_verification_manifest"
        ],
        "max_logit_difference": verification["max_logit_difference"],
    }
    for key, value in table_checks.items():
        if str(row.get(key)) != str(value):
            raise ValueError(
                f"final hybrid table mismatch {key}: {row.get(key)!r} != {value!r}"
            )
    audit = json.loads(tokenizer_audit_path.read_text(encoding="utf-8"))
    decision = audit.get("decision", {})
    if decision.get("audit_passed_for_downstream") is not True:
        raise ValueError("final hybrid tokenizer audit did not pass")
    sources = [
        source for source in audit.get("sources", [])
        if source.get("label") == FINAL_HYBRID_LABEL
    ]
    if len(sources) != 1:
        raise ValueError("final hybrid tokenizer identity is not unique")
    source = sources[0]
    row.update({
        "checkpoint_verification_sha256": sha256(verification_path),
        "source_model_revision": verification.get("source_model_revision", ""),
        "tokenizer_revision": verification.get("tokenizer_revision", ""),
        "tokenizer_selected_mode": decision.get("selected_tokenizer_mode", ""),
        "tokenizer_fix_mistral_regex": decision.get(
            "use_fix_mistral_regex_for_future_evaluation", ""
        ),
        "tokenizer_files_combined_sha256": source[
            "tokenizer_files_combined_sha256"
        ],
        "tokenizer_audit_sha256": sha256(tokenizer_audit_path),
        "tokenizer_audit_passed": True,
    })
    return row, verification


def _write_checkpoint_tables(
    directory: Path, rows: list[dict], verification: dict,
) -> list[Path]:
    directory.mkdir()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in CHECKPOINT_EXTRA_FIELDS:
        if key not in fieldnames:
            fieldnames.append(key)
    for row in rows:
        for key in fieldnames:
            row.setdefault(key, "")
    csv_path = directory / "checkpoint_table.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    json_path = directory / "checkpoint_table.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path = directory / "checkpoint_table.md"
    with md_path.open("x", encoding="utf-8") as handle:
        handle.write(
            "| Checkpoint | Actual % | Removed channels | Expert neurons | "
            "Total params | MoE params | Weight bytes | Plan SHA-256 | Reload | "
            "Exact logits | Max logit diff | No padding | Tokenizer mode | "
            "Tokenizer files SHA-256 | Tokenizer audit SHA-256 |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---|---|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['label']} | {row.get('actual_pct', '')} | "
                f"{row.get('removed_layer_channels', '')} | "
                f"{row.get('removed_expert_neurons', '')} | "
                f"{row.get('total_parameters', '')} | "
                f"{row.get('moe_expert_parameters', '')} | "
                f"{row.get('serialized_weight_bytes', '')} | "
                f"{row.get('plan_sha256', '')} | "
                f"{row.get('successful_reload', '')} | "
                f"{row.get('exact_logits_after_reload', '')} | "
                f"{row.get('max_logit_difference', '')} | "
                f"{row.get('no_hidden_original_width_padding', '')} | "
                f"{row.get('tokenizer_selected_mode', '')} | "
                f"{row.get('tokenizer_files_combined_sha256', '')} | "
                f"{row.get('tokenizer_audit_sha256', '')} |\n"
            )
    tex_path = directory / "checkpoint_table.tex"
    with tex_path.open("x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrrlll}\n\\toprule\n")
        handle.write(
            "Checkpoint & Actual \\% & Channels & Expert neurons & Total params & "
            "MoE params & Bytes & Reload & Exact logits & No padding \\\\\n\\midrule\n"
        )
        for row in rows:
            label = row["label"].replace("_", r"\_")
            handle.write(
                f"{label} & {row.get('actual_pct', '')} & "
                f"{row.get('removed_layer_channels', '')} & "
                f"{row.get('removed_expert_neurons', '')} & "
                f"{row.get('total_parameters', '')} & "
                f"{row.get('moe_expert_parameters', '')} & "
                f"{row.get('serialized_weight_bytes', '')} & "
                f"{row.get('successful_reload', '')} & "
                f"{row.get('exact_logits_after_reload', '')} & "
                f"{row.get('no_hidden_original_width_padding', '')} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")
    verification_path = directory / "final_hybrid_checkpoint_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return [csv_path, json_path, md_path, tex_path, verification_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", required=True)
    parser.add_argument("--ppl-dir", required=True)
    parser.add_argument("--downnorm-curve-dir", required=True)
    parser.add_argument("--downstream-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hybrid-checkpoint-dir", required=True)
    parser.add_argument("--hybrid-tokenizer-audit", required=True)
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
    hybrid_checkpoint_dir = Path(args.hybrid_checkpoint_dir)
    hybrid_tokenizer_audit_path = Path(args.hybrid_tokenizer_audit)
    validation_path = Path(args.matched_validation)
    for group, directory in sources.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{group}: {directory}")
        for name in SOURCE_PATTERNS[group]:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    if not hybrid_checkpoint_dir.is_dir():
        raise FileNotFoundError(hybrid_checkpoint_dir)
    for name in SOURCE_PATTERNS["checkpoints"]:
        if not (hybrid_checkpoint_dir / name).is_file():
            raise FileNotFoundError(hybrid_checkpoint_dir / name)
    if not hybrid_tokenizer_audit_path.is_file():
        raise FileNotFoundError(hybrid_tokenizer_audit_path)

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
    checkpoint_rows = _merge_checkpoint_rows(
        read_csv(sources["checkpoints"] / "checkpoint_table.csv"),
        read_csv(hybrid_checkpoint_dir / "checkpoint_table.csv"),
    )
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
    if FINAL_HYBRID_LABEL not in checkpoint_labels:
        raise ValueError("checkpoint table lacks the final proposed hybrid")

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
    paired_rows = read_csv(
        sources["downstream"] / "downstream_paired_comparisons.csv"
    )
    hybrid_decision = hybrid_outcome(frontier, downstream_rows)
    final_statistics = _validate_final_hybrid_statistics(paired_rows)
    final_checkpoint, final_verification = _validate_and_enrich_final_checkpoint(
        checkpoint_rows, frontier, hybrid_tokenizer_audit_path,
    )
    provenance, dirty_patch, dependency_lock, entry_points = code_provenance()
    settings = {
        "schema_version": 2,
        "frozen_paper_claims": {
            "proposed_method": PROPOSED_METHOD,
            "endpoint_methods": [
                "pure ellipsoid ranking", "pure down-norm ranking",
            ],
            "conservative_operating_point": (
                "approximately 4.12% whole-model reduction with no detected "
                "significant seven-task macro-accuracy difference for the "
                "pure down-norm endpoint"
            ),
            "higher_compression_operating_point": (
                "approximately 5.89% whole-model reduction with a 0.953-point "
                "seven-task macro-accuracy loss for the pure down-norm endpoint"
            ),
            "ellipsoid_contribution": (
                "calibration-free local and strict set-level MoE certificate; "
                "the 0.25%-slack hybrid matches pure down-norm accuracy within "
                "uncertainty while retaining a tighter certificate"
            ),
            "systems": (
                "physical storage reduction demonstrated; consistent runtime "
                "acceleration not demonstrated; the measured pure down-norm "
                "target-6 checkpoint is a shape-equivalent structural benchmark "
                "for the final hybrid"
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
            "benchmark_interpretation": (
                "shape-equivalent structural benchmark using the pure down-norm "
                "target-6 checkpoint; all matched target-6 methods use the same "
                "per-layer widths and physical tensor shapes"
            ),
            "trials_interleaved": False,
            "loaded_hbm_gib": {"baseline": 56.87, "shape_equivalent_target6": 54.86},
            "peak_hbm_gib": {"baseline": 62.32, "shape_equivalent_target6": 60.31},
            "loaded_hbm_reduction_pct": load_hbm_reduction,
            "peak_hbm_reduction_pct_by_case": peak_hbm_reduction_values,
            "prefill_gain_ci_positive_in_all_cases": consistent_prefill,
            "decode_gain_ci_positive_in_all_cases": consistent_decode,
        },
        "certified_hybrid_decision": hybrid_decision,
        "final_hybrid_statistical_interpretation": final_statistics,
        "final_hybrid_checkpoint": final_checkpoint,
        "code_provenance": provenance,
        "table_and_evaluation_entry_points": entry_points,
        "exact_entry_point_commands": {
            "frontier": "scripts/build_moe_certified_hybrid_frontier.py",
            "ppl": "scripts/run_moe_certified_hybrid_ppl.sh",
            "checkpoint_export": "scripts/run_paper_v3_checkpoint_export.sh",
            "downstream_evaluation": "scripts/run_paper_v3_downstream.sh",
            "downstream_tables": "scripts/summarize_paper_v3_downstream.py",
            "checkpoint_tables": "scripts/summarize_paper_v3_checkpoints.py",
            "systems_tables": "scripts/summarize_paper_v3_systems.py",
            "final_packet": "scripts/build_moe_certified_hybrid_final_packet.py",
        },
    }
    output = Path(args.output_dir)
    if args.dry_run:
        print(
            f"[hybrid-final-packet] DRY RUN groups={len(sources)} "
            f"frontier_plans={len(frontier['plans'])} "
            f"final_method={FINAL_HYBRID_LABEL} "
            f"consistent_prefill={consistent_prefill} "
            f"consistent_decode={consistent_decode} "
            f"tracked_worktree_clean={provenance['tracked_worktree_clean']}"
        )
        return
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    copied = []
    for group, source in sources.items():
        if group == "checkpoints":
            continue
        copied.extend(copy_group(source, output, group, SOURCE_PATTERNS[group]))
    copied.extend(_write_checkpoint_tables(
        output / "checkpoints", checkpoint_rows, final_verification,
    ))
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
    provenance_path = output / "code_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    copied.append(provenance_path)
    dependency_path = output / "dependency_environment_lock.txt"
    dependency_path.write_text(dependency_lock, encoding="utf-8")
    copied.append(dependency_path)
    entry_points_path = output / "entry_point_manifest.json"
    entry_points_path.write_text(json.dumps(entry_points, indent=2), encoding="utf-8")
    copied.append(entry_points_path)
    if dirty_patch:
        patch_path = output / "code_provenance.patch"
        patch_path.write_bytes(dirty_patch)
        copied.append(patch_path)

    systems_interpretation = (
        "# Systems evidence interpretation\n\n"
        "The measured checkpoint is the pure down-norm target-6 endpoint and is "
        "reported as a **shape-equivalent structural benchmark** for the final "
        "0.25%-slack hybrid. Both use the same RMSNorm allocation, per-layer "
        "removal counts, total removed channels, alignment, and physical tensor "
        "shapes. Selector identity is therefore not expected to change the executed "
        "GEMM dimensions, but this is not presented as a direct timing run of the "
        "hybrid checkpoint.\n\n"
        "Loaded HBM decreased from 56.87 to 54.86 GiB (approximately 3.53%), and "
        "peak HBM decreased from 62.32 to 60.31 GiB (approximately 3.22%). Prefill "
        "behavior is configuration-dependent, no decode improvement is demonstrated, "
        "and consistent overall inference acceleration is not demonstrated. Trials "
        "were sequential rather than interleaved.\n"
    )
    systems_interpretation_path = output / "systems_interpretation.md"
    systems_interpretation_path.write_text(
        systems_interpretation, encoding="utf-8"
    )
    copied.append(systems_interpretation_path)
    systems_interpretation_tex = output / "systems_interpretation.tex"
    systems_interpretation_tex.write_text(
        "\\section{Systems evidence interpretation}\n"
        "The pure down-norm target-6 checkpoint is reported as a shape-equivalent "
        "structural benchmark for the final 0.25\\%-slack hybrid. Loaded HBM "
        "decreased from 56.87 to 54.86 GiB (approximately 3.53\\%), and peak HBM "
        "decreased from 62.32 to 60.31 GiB (approximately 3.22\\%). Prefill behavior "
        "is configuration-dependent, decode improvement is not demonstrated, and "
        "trials were sequential rather than interleaved. Consistent overall "
        "inference acceleration is not demonstrated.\n",
        encoding="utf-8",
    )
    copied.append(systems_interpretation_tex)

    hbm_text = (
        "Loaded HBM decreased from 56.87 to 54.86 GiB "
        f"(approximately {load_hbm_reduction:.2f}%), and peak HBM decreased "
        "from 62.32 to 60.31 GiB "
        f"(approximately {peak_hbm_reduction_values[0]:.2f}%)."
    )
    ellipsoid_points = final_statistics[
        "hybrid_minus_ellipsoid_macro_accuracy_points"
    ]
    downnorm_points = final_statistics[
        "hybrid_minus_downnorm_macro_accuracy_points"
    ]
    conclusion = (
        "# Final certified-hybrid paper conclusion\n\n"
        f"The final proposed method is **{PROPOSED_METHOD}**, with p95 expert "
        "aggregation, aligned physical tensor pruning, and no weight updates. "
        "Pure ellipsoid and pure down-norm are the two endpoint methods.\n\n"
        f"Against pure ellipsoid, the hybrid gains {ellipsoid_points:.3f} "
        "seven-task macro-accuracy points and the paired comparison remains "
        "significant after Holm correction. Against pure down-norm, its point "
        f"difference is {downnorm_points:+.3f} points and is not statistically "
        "significant. The hybrid therefore matches down-norm accuracy within "
        "uncertainty while retaining a tighter certificate; it is not claimed "
        "to outperform pure down-norm.\n\n"
        "The conservative operating point removes approximately 4.12% of total "
        "model parameters and has no detected significant seven-task macro-accuracy "
        "difference for the pure down-norm endpoint. The higher-compression "
        "pure down-norm point removes approximately 5.89% and has a 0.953-point "
        "macro-accuracy loss.\n\n"
        "The ellipsoid theorem contributes a calibration-free expert-channel bound "
        "and strict set-level local MoE certificate. Pure ellipsoid preserves "
        "perplexity better at the matched target-6 budget, while certificate-"
        "constrained down-norm refinement recovers downstream accuracy. The "
        "reported certificate is local to the MoE calculation and is not a "
        "propagated guarantee for the full Transformer.\n\n"
        f"{hybrid_decision['conclusion_sentence']}\n\n"
        f"Physical checkpoint storage reduction is demonstrated. {hbm_text} "
        "The measured target-6 pure down-norm checkpoint is reported only as a "
        "shape-equivalent structural benchmark for the final hybrid. Trials were "
        "sequential rather than interleaved, prefill behavior is configuration-"
        "dependent, and decode improvement is not demonstrated. Consistent overall "
        "inference acceleration is therefore not demonstrated.\n\n"
        "Selector experimentation stops after the predefined fine-grid closure. "
        "The project now moves directly to paper writing.\n"
    )
    conclusion_path = output / "final_conclusion.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    copied.append(conclusion_path)
    conclusion_tex = output / "final_conclusion.tex"
    conclusion_tex.write_text(
        "\\section{Final certified-hybrid conclusion}\n"
        "The final proposed method uses RMSNorm-bound global allocation followed "
        "by certificate-constrained down-norm refinement with 0.25\\% ellipsoid-"
        "certificate slack and aligned physical pruning. It improves macro accuracy "
        f"over pure ellipsoid by {ellipsoid_points:.3f} points after Holm correction "
        f"and differs from pure down-norm by {downnorm_points:+.3f} points without "
        "statistical significance. It therefore matches down-norm accuracy within "
        "uncertainty while retaining a tighter certificate; no superiority over "
        "down-norm is claimed. Physical storage and HBM reductions are demonstrated. "
        "The systems run is a shape-equivalent structural benchmark with sequential "
        "trials; consistent inference acceleration is not demonstrated.\n",
        encoding="utf-8",
    )
    copied.append(conclusion_tex)
    readme = output / "README.md"
    readme.write_text(
        "# Final paper packet\n\n"
        "This directory is the immutable closure packet for the certified-hybrid "
        "milestone. The final proposed method is RMSNorm-bound global allocation "
        "plus certificate-constrained down-norm refinement with 0.25% ellipsoid-"
        "certificate slack; pure ellipsoid and pure down-norm are endpoints. Tables "
        "are grouped by source category, and all files are indexed and hashed in "
        "`FINAL_PACKET_MANIFEST.json`. No additional selector sweep is authorized "
        "after this packet.\n",
        encoding="utf-8",
    )
    copied.append(readme)
    manifest_path = output / "FINAL_PACKET_MANIFEST.json"
    manifest = {
        "schema_version": 2,
        "proposed_method": PROPOSED_METHOD,
        "stop_experimentation": True,
        "next_phase": "paper writing",
        "hash_scope": "all packet files except FINAL_PACKET_MANIFEST.json itself",
        "code_provenance": provenance,
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
