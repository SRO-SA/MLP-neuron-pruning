#!/usr/bin/env python3
"""Re-hash and freeze the four provisional physical Version 3 checkpoints."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment_provenance import file_sha256

PLAN_FILENAME = "heterogeneous_pruning_plan.json"


DEFAULT_EVALUATION_CODE = (
    "scripts/audit_paper_v3_tokenizers.py",
    "scripts/run_paper_v3_lm_eval.py",
    "scripts/summarize_paper_v3_downstream.py",
    "scripts/benchmark_paper_v3_inference.py",
    "scripts/summarize_paper_v3_systems.py",
    "src/heterogeneous_moe_checkpoint.py",
    "src/evaluation.py",
)

FILE_FIELDS = (
    "checkpoint_label", "category", "relative_path", "size_bytes",
    "size_gb_decimal", "size_gib_binary", "sha256",
    "matches_export_verification_hash",
)

SUMMARY_FIELDS = (
    "label", "target_pct", "actual_pct", "removed_layer_channels",
    "removed_expert_neurons", "total_parameters", "moe_expert_parameters",
    "serialized_weight_bytes", "serialized_weight_gb_decimal",
    "serialized_weight_gib_binary", "checkpoint_payload_bytes",
    "checkpoint_payload_gb_decimal", "checkpoint_payload_gib_binary",
    "safetensors_shards", "plan_sha256", "successful_reload",
    "exact_logits_after_reload", "no_hidden_original_width_padding",
)


def byte_units(size: int) -> dict[str, float]:
    return {
        "size_gb_decimal": size / 1_000_000_000,
        "size_gib_binary": size / (1024 ** 3),
    }


def file_category(relative_path: str) -> str:
    name = os.path.basename(relative_path)
    if name.endswith(".safetensors.index.json"):
        return "safetensors_index"
    if name.endswith(".safetensors"):
        return "safetensors_shard"
    if name == "config.json":
        return "model_config"
    if name == "generation_config.json":
        return "generation_config"
    if name == PLAN_FILENAME:
        return "pruning_plan"
    if name == "checkpoint_verification.json":
        return "checkpoint_verification"
    if (
        name.startswith("tokenizer") or name.startswith("chat_template")
        or name in {"vocab.json", "merges.txt", "special_tokens_map.json",
                    "added_tokens.json", "sentencepiece.bpe.model"}
    ):
        return "tokenizer"
    return "supporting_file"


def _all_files(root: str) -> list[str]:
    return sorted(
        os.path.join(current, name)
        for current, _, names in os.walk(root) for name in names
    )


def index_checkpoint(spec: dict) -> tuple[list[dict], dict]:
    root = os.path.realpath(spec["checkpoint_dir"])
    verification_path = os.path.join(root, "checkpoint_verification.json")
    with open(verification_path, encoding="utf-8") as handle:
        verification = json.load(handle)
    for key in (
        "successful_reload", "exact_logits_after_reload",
        "no_hidden_original_width_padding",
    ):
        if verification.get(key) is not True:
            raise ValueError(f"{spec['label']} failed frozen condition {key}")
    expected_hashes = verification.get("checkpoint_file_sha256", {})
    rows = []
    seen_relative = set()
    for path in _all_files(root):
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        seen_relative.add(relative)
        digest = file_sha256(path)
        expected = expected_hashes.get(relative)
        match: bool | str = "not_recorded_at_export"
        if expected is not None:
            match = digest == expected
            if not match:
                raise ValueError(f"checkpoint file changed since export: {path}")
        size = os.path.getsize(path)
        rows.append({
            "checkpoint_label": spec["label"], "category": file_category(relative),
            "relative_path": relative, "size_bytes": size, **byte_units(size),
            "sha256": digest, "matches_export_verification_hash": match,
        })
    missing = sorted(set(expected_hashes) - seen_relative)
    if missing:
        raise FileNotFoundError(
            f"{spec['label']} files recorded at export are missing: {missing}"
        )
    categories = Counter(row["category"] for row in rows)
    for required in (
        "safetensors_shard", "safetensors_index", "model_config",
        "generation_config", "tokenizer", "checkpoint_verification",
    ):
        if not categories[required]:
            raise FileNotFoundError(f"{spec['label']} missing category {required}")
    if float(spec["target_pct"]) > 0 and not categories["pruning_plan"]:
        raise FileNotFoundError(f"{spec['label']} missing physical pruning plan")
    current_plan_sha = ""
    if categories["pruning_plan"]:
        plan_path = os.path.join(root, PLAN_FILENAME)
        current_plan_sha = file_sha256(plan_path)
        for declared in (spec.get("plan_sha256", ""), verification.get("plan_sha256", "")):
            if declared and declared != current_plan_sha:
                raise ValueError(f"{spec['label']} plan SHA-256 changed")
    counts = verification["parameters_reloaded"]
    weights = int(verification["serialized_weight_bytes"])
    payload = int(verification[
        "checkpoint_payload_bytes_excluding_verification_manifest"
    ])
    summary = {
        "label": spec["label"], "target_pct": spec["target_pct"],
        "actual_pct": spec["actual_pct"],
        "removed_layer_channels": verification["removed_layer_channels"],
        "removed_expert_neurons": verification["removed_expert_neurons"],
        "total_parameters": counts["total"],
        "moe_expert_parameters": counts["moe_experts"],
        "serialized_weight_bytes": weights,
        "serialized_weight_gb_decimal": weights / 1_000_000_000,
        "serialized_weight_gib_binary": weights / (1024 ** 3),
        "checkpoint_payload_bytes": payload,
        "checkpoint_payload_gb_decimal": payload / 1_000_000_000,
        "checkpoint_payload_gib_binary": payload / (1024 ** 3),
        "safetensors_shards": categories["safetensors_shard"],
        "plan_sha256": current_plan_sha,
        "successful_reload": True, "exact_logits_after_reload": True,
        "no_hidden_original_width_padding": True,
        "checkpoint_dir": root, "category_counts": dict(categories),
        "checkpoint_verification_sha256": file_sha256(verification_path),
    }
    return rows, summary


def git_identity(repo_root: str) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True,
    ).strip()
    origin_commit = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "origin/master"], text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError(f"tracked evaluation code is dirty:\n{status}")
    if commit != origin_commit:
        raise RuntimeError(
            f"evaluation HEAD is not the published origin/master: "
            f"HEAD={commit} origin/master={origin_commit}"
        )
    return {"git_commit": commit, "origin_master_commit": origin_commit,
            "tracked_worktree_clean": True, "head_matches_origin_master": True}


def _write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("| Checkpoint | Actual % | Removed channels | Parameters | Weights GB | Weights GiB | Shards |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {float(row['actual_pct']):.4f} | "
                f"{row['removed_layer_channels']} | {row['total_parameters']} | "
                f"{row['serialized_weight_gb_decimal']:.3f} | "
                f"{row['serialized_weight_gib_binary']:.3f} | "
                f"{row['safetensors_shards']} |\n"
            )


def _escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        handle.write("Checkpoint & Actual \\% & Channels & Parameters & GB & GiB & Shards \\\\\n\\midrule\n")
        for row in rows:
            values = (
                row["label"], f"{float(row['actual_pct']):.4f}",
                row["removed_layer_channels"], row["total_parameters"],
                f"{row['serialized_weight_gb_decimal']:.3f}",
                f"{row['serialized_weight_gib_binary']:.3f}",
                row["safetensors_shards"],
            )
            handle.write(" & ".join(_escape(value) for value in values) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--tokenizer-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-code-file", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.checkpoint_manifest, encoding="utf-8") as handle:
        specs = json.load(handle)
    if len(specs) != 4:
        raise ValueError(f"frozen provisional manifest must contain 4 checkpoints, got {len(specs)}")
    with open(args.tokenizer_audit, encoding="utf-8") as handle:
        tokenizer_audit = json.load(handle)
    if not tokenizer_audit.get("decision", {}).get("audit_passed_for_downstream"):
        raise ValueError("tokenizer audit has not passed downstream gate")

    if args.dry_run:
        total_files = total_bytes = 0
        for spec in specs:
            verification = os.path.join(
                spec["checkpoint_dir"], "checkpoint_verification.json"
            )
            if not os.path.isfile(verification):
                raise FileNotFoundError(verification)
            files = _all_files(spec["checkpoint_dir"])
            total_files += len(files)
            total_bytes += sum(os.path.getsize(path) for path in files)
        print(
            f"[immutable-checkpoints] DRY RUN: checkpoints={len(specs)} "
            f"files={total_files} bytes_to_hash={total_bytes} "
            f"GB={total_bytes / 1_000_000_000:.3f} "
            f"GiB={total_bytes / (1024 ** 3):.3f} output={args.output_dir}"
        )
        return

    file_rows, summaries = [], []
    for spec in specs:
        indexed, summary = index_checkpoint(spec)
        file_rows.extend(indexed); summaries.append(summary)
    code_paths = args.evaluation_code_file or list(DEFAULT_EVALUATION_CODE)
    code_files = []
    for supplied in code_paths:
        path = os.path.realpath(supplied)
        if not os.path.isfile(path):
            raise FileNotFoundError(supplied)
        size = os.path.getsize(path)
        code_files.append({
            "path": os.path.relpath(path, REPO_ROOT).replace(os.sep, "/"),
            "size_bytes": size, **byte_units(size), "sha256": file_sha256(path),
        })
    payload = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "size_units": {
            "GB_decimal": "bytes / 1,000,000,000",
            "GiB_binary": "bytes / 1,073,741,824",
        },
        "source_checkpoint_manifest": os.path.realpath(args.checkpoint_manifest),
        "source_checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
        "tokenizer_audit": os.path.realpath(args.tokenizer_audit),
        "tokenizer_audit_sha256": file_sha256(args.tokenizer_audit),
        "tokenizer_decision": tokenizer_audit["decision"],
        "evaluation_code": {**git_identity(REPO_ROOT), "files": code_files},
        "checkpoints": summaries,
        "checkpoint_files": file_rows,
    }
    os.makedirs(args.output_dir)
    json_path = os.path.join(args.output_dir, "immutable_checkpoint_manifest.json")
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(os.path.join(args.output_dir, "checkpoint_file_hashes.csv"), "x",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILE_FIELDS)
        writer.writeheader(); writer.writerows(file_rows)
    with open(os.path.join(args.output_dir, "immutable_checkpoint_summary.csv"), "x",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(
            {key: row[key] for key in SUMMARY_FIELDS} for row in summaries
        )
    _write_markdown(
        os.path.join(args.output_dir, "immutable_checkpoint_summary.md"), summaries
    )
    _write_latex(
        os.path.join(args.output_dir, "immutable_checkpoint_summary.tex"), summaries
    )
    with open(os.path.join(args.output_dir, "MANIFEST_SHA256.txt"), "x",
              encoding="ascii") as handle:
        handle.write(f"{file_sha256(json_path)}  immutable_checkpoint_manifest.json\n")
    print(
        f"[immutable-checkpoints] OK: checkpoints={len(summaries)} "
        f"files={len(file_rows)} commit={payload['evaluation_code']['git_commit']} "
        f"JSON={json_path}"
    )


if __name__ == "__main__":
    main()
