#!/usr/bin/env python3
"""Record the compatibility limits and construction protocol of a HEAPr run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.experiment_provenance import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requested-ratio", type=float, required=True)
    parser.add_argument("--calibration-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lm-eval-results", required=True)
    parser.add_argument("--lm-eval-identity", required=True)
    parser.add_argument("--reporting-patch", required=True)
    args = parser.parse_args()
    output = os.path.join(args.run_dir, "heapr_protocol.json")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    actual = subprocess.check_output(
        ["git", "-C", args.repo_dir, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != args.commit:
        raise ValueError("HEAPr commit changed")
    cost_path = os.path.join(args.run_dir, "construction_cost.json")
    with open(cost_path, encoding="utf-8") as handle:
        cost = json.load(handle)
    with open(args.lm_eval_results, encoding="utf-8") as handle:
        eval_result = json.load(handle)
    matched = eval_result.get("heapr_matched_protocol", {})
    if not matched.get("_heapr_matched_reporting_patch"):
        raise ValueError("HEAPr matched sample/protocol report missing")
    payload = {
        "schema_version": 1, "method": "HEAPr-G (official implementation)",
        "official_repository": "https://github.com/LLIKKE/HEAPr",
        "git_commit": actual, "model": args.model,
        "requested_compression_ratio": args.requested_ratio,
        "calibration_dataset": "WikiText-2", "calibration_samples": args.calibration_samples,
        "forward_passes_declared_by_method": 2,
        "backward_passes_declared_by_method": 1,
        "requires_gradients": True, "requires_activations": True,
        "wall_clock_seconds": cost["wall_clock_seconds"],
        "peak_gpu_memory_used_bytes_total": cost["peak_gpu_memory_used_bytes_total"],
        "peak_incremental_gpu_memory_used_bytes_total": cost[
            "peak_incremental_gpu_memory_used_bytes_total"
        ],
        "parameters_before": matched["parameters_before"],
        "parameters_after": matched["parameters_after"],
        "model_parameter_dtypes": matched["model_parameter_dtypes"],
        "measured_total_parameter_reduction_pct": 100.0 * (
            1.0 - matched["parameters_after"] / matched["parameters_before"]
        ),
        "lm_eval_results_sha256": file_sha256(args.lm_eval_results),
        "lm_eval_identity": args.lm_eval_identity,
        "reporting_patch_sha256": file_sha256(args.reporting_patch),
        "seed": args.seed, "run_log_sha256": file_sha256(
            os.path.join(args.run_dir, "run.log")
        ),
        "matched_checkpoint": True, "matched_task_names": True,
        "matched_pruning_definition": False,
        "comparability_note": (
            "HEAPr prunes expert-specific atomic experts; the frozen method removes "
            "shared packed channel IDs across all experts. Report equal measured parameter/FLOP "
            "budgets, and do not describe the raw requested ratios as identical structures."
        ),
    }
    with open(output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[heapr] protocol: {output}")


if __name__ == "__main__":
    main()
