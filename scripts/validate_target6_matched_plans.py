#!/usr/bin/env python3
"""Fail-fast validation for the four matched target-6 ranking plans."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_provenance import file_sha256
from src.moe_set_certification import matched_plan_validation


def parse_assignments(values: list[str], kind: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must be LABEL=PATH: {value}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if label in result:
            raise ValueError(f"duplicate {kind} label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[label] = path
    return result


def nested_get(value, *keys):
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_file_hash(path_value) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    return file_sha256(str(path)) if path.is_file() else ""


def result_protocol(report: dict) -> dict:
    if isinstance(report.get("results"), list):
        rows = [row for row in report["results"] if row.get("status") == "ok"]
        if not rows:
            raise ValueError("result report has no successful dataset rows")
        common_fields = (
            "model", "resolved_model", "model_revision", "tokenizer_revision",
            "tokenizer_name_or_path", "transformers_version", "torch_version",
            "pruning_mode", "aggregation_mode", "channel_alignment", "seed",
            "calibration_actual_num_prompts", "calibration_corpus_sha256",
            "calibration_eval_disjoint_verified",
        )
        common = {}
        for field in common_fields:
            values = {str(row.get(field, "")) for row in rows}
            if len(values) != 1:
                raise ValueError(f"result rows disagree on {field}: {values}")
            common[field] = next(iter(values))
        calibration_manifest_hashes = {
            _optional_file_hash(row.get("calibration_sample_manifest_path"))
            for row in rows
        }
        if len(calibration_manifest_hashes) != 1:
            raise ValueError("result rows disagree on calibration sample identifiers")
        common["calibration_sample_manifest_sha256"] = next(
            iter(calibration_manifest_hashes)
        )
        datasets = {}
        for row in rows:
            name = str(row.get("eval_dataset", ""))
            if not name or name in datasets:
                raise ValueError(f"invalid/duplicate evaluation dataset {name!r}")
            datasets[name] = {
                "n_eval": row.get("n_eval"),
                "max_seq_len": row.get("evaluation_max_seq_len"),
                "batch_size": row.get("evaluation_batch_size"),
                "num_texts": row.get("evaluation_num_texts"),
                "corpus_sha256": row.get("evaluation_corpus_sha256"),
                "preprocessing": row.get("evaluation_preprocessing"),
                "sample_manifest_sha256": _optional_file_hash(
                    row.get("evaluation_sample_manifest_path")
                ),
                "baseline_tokens": row.get("baseline_eval_tokens"),
                "pruned_tokens": row.get("pruned_eval_tokens"),
                "token_count_match": row.get("evaluation_token_count_match"),
            }
        return {"common": common, "datasets": datasets}
    return {
        "model": report.get("model") or report.get("model_id"),
        "model_revision": report.get("model_revision") or nested_get(report, "provenance", "model_revision"),
        "tokenizer_revision": report.get("tokenizer_revision") or nested_get(report, "provenance", "tokenizer_revision"),
        "tokenizer_name_or_path": report.get("tokenizer_name_or_path") or nested_get(report, "provenance", "tokenizer_name_or_path"),
        "eval_datasets": report.get("eval_datasets") or sorted((report.get("ppl_results") or {}).keys()),
        "n_eval": report.get("n_eval"),
        "max_seq_len": report.get("max_seq_len"),
        "seed": report.get("seed"),
        "pruning_mode": report.get("pruning_mode"),
        "aggregation_mode": report.get("aggregation_mode"),
        "channel_alignment": report.get("channel_alignment"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", default=[], required=True,
                        help="LABEL=PATH; repeat for all four plans")
    parser.add_argument("--result", action="append", default=[],
                        help="LABEL=JSON report; repeat to validate evaluation protocol")
    parser.add_argument("--checkpoint-verification", action="append", default=[],
                        help="LABEL=checkpoint_verification.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-total", type=int, default=2288)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_paths = parse_assignments(args.plan, "plan")
    expected_labels = {"ellipsoid", "down_norm", "activation", "rmsnorm_bound"}
    if set(plan_paths) != expected_labels:
        raise ValueError(f"plan labels must be exactly {sorted(expected_labels)}")
    plans = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in plan_paths.items()
    }
    validation = matched_plan_validation(
        plans, expected_total=args.expected_total, expected_alignment=16
    )
    validation["plans"] = {
        label: {"path": str(path), "sha256": file_sha256(str(path))}
        for label, path in plan_paths.items()
    }

    result_paths = parse_assignments(args.result, "result") if args.result else {}
    if result_paths:
        if set(result_paths) != expected_labels:
            raise ValueError("result labels must match all four plan labels")
        protocols = {
            label: result_protocol(json.loads(path.read_text(encoding="utf-8")))
            for label, path in result_paths.items()
        }
        reference = protocols["ellipsoid"]
        for label, protocol in protocols.items():
            if protocol != reference:
                differences = {
                    key: (reference.get(key), protocol.get(key))
                    for key in sorted(set(reference) | set(protocol))
                    if reference.get(key) != protocol.get(key)
                }
                raise ValueError(f"{label}: evaluation protocol mismatch: {differences}")
        validation["evaluation_protocol"] = reference
        validation["result_reports"] = {
            label: {"path": str(path), "sha256": file_sha256(str(path))}
            for label, path in result_paths.items()
        }

    checkpoint_paths = (
        parse_assignments(args.checkpoint_verification, "checkpoint verification")
        if args.checkpoint_verification else {}
    )
    if checkpoint_paths:
        if set(checkpoint_paths) != expected_labels:
            raise ValueError("checkpoint labels must match all four plan labels")
        shape_signatures = {}
        checkpoint_rows = {}
        for label, path in checkpoint_paths.items():
            row = json.loads(path.read_text(encoding="utf-8"))
            if not row.get("successful_reload") or not row.get("exact_logits_after_reload"):
                raise ValueError(f"{label}: checkpoint reload/logit verification failed")
            shapes = row.get("shape_audit_after_reload")
            shape_signatures[label] = json.dumps(shapes, sort_keys=True)
            checkpoint_rows[label] = {
                "path": str(path), "sha256": file_sha256(str(path)),
                "reload_success": row.get("successful_reload"),
                "exact_logits": row.get("exact_logits_after_reload"),
                "no_hidden_padding": row.get("no_hidden_original_width_padding"),
            }
        if len(set(shape_signatures.values())) != 1:
            raise ValueError("target-6 exported tensor shape paths differ")
        validation["checkpoint_verification"] = checkpoint_rows

    validation["comparison_scope"] = (
        "fixed RMSNorm allocation; only selected channel identities may differ"
    )
    validation["strict_gate_passed"] = True
    output = Path(args.output)
    if args.dry_run:
        print(json.dumps(validation, indent=2))
        print(f"[matched-plans] DRY RUN: would write {output}")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(
        f"[matched-plans] OK labels=4 channels={args.expected_total} "
        f"expert_neurons={validation['total_removed_expert_neurons']} output={output}"
    )


if __name__ == "__main__":
    main()
