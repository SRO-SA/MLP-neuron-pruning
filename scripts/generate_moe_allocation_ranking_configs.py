#!/usr/bin/env python3
"""Generate the bounded 2%/4% allocation-versus-ranking experiment matrix."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

try:
    from generate_moe_selector_baseline_configs import _build_config, _write_yaml
except ImportError:
    from scripts.generate_moe_selector_baseline_configs import _build_config, _write_yaml


PROFILE_EXPERIMENTS = {
    "replicate2": [
        (2, "rmsnorm_bound", "rmsnorm_ellipsoid_bound",
         "rmsnorm_alloc__ellipsoid_rank"),
    ],
    "target2_extended": [
        (2, "rmsnorm_bound", "rmsnorm_ellipsoid_bound",
         "rmsnorm_alloc__ellipsoid_rank"),
        (2, "down_norm", "rmsnorm_ellipsoid_bound",
         "downnorm_alloc__ellipsoid_rank"),
    ],
    "target4": [
        (4, "rmsnorm_bound", "rmsnorm_bound",
         "rmsnorm_alloc__rmsnorm_rank"),
        (4, "rmsnorm_bound", "rmsnorm_ellipsoid_bound",
         "rmsnorm_alloc__ellipsoid_rank"),
        (4, "down_norm", "down_norm",
         "downnorm_alloc__downnorm_rank"),
        (4, "down_norm", "rmsnorm_ellipsoid_bound",
         "downnorm_alloc__ellipsoid_rank"),
    ],
}


def _find_plan(run_dir: str, selector: str, target: int) -> str:
    pattern = os.path.join(
        run_dir, f"{selector}_target{target}", "pruning_plans", "*.json"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {selector} target-{target} plan matching "
            f"{pattern!r}; found {matches}"
        )
    return matches[0]


def _validated_plan_path(
    *,
    selector: str,
    target: int,
    baseline_run_dir: str,
    target2_rmsnorm_run_dir: str,
) -> str:
    run_dir = (
        target2_rmsnorm_run_dir
        if selector == "rmsnorm_bound" and target == 2
        else baseline_run_dir
    )
    path = _find_plan(run_dir, selector, target)
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    checks = {
        "selector": (plan.get("selector"), selector),
        "target_pct": (float(plan.get("target_pct", -1)), float(target)),
        "pruning_mode": (plan.get("pruning_mode"), "packed_same_channel"),
        "aggregation_mode": (plan.get("aggregation_mode"), "p95"),
        "channel_alignment": (int(plan.get("channel_alignment", -1)), 16),
    }
    counted = sum(len(row.get("prune_idx", [])) for row in plan.get("layers", []))
    checks["total_selected_layer_channels"] = (
        int(plan.get("total_selected_layer_channels", -1)), counted
    )
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{path}: {field}={actual!r}, expected {expected!r}"
            )
    return path


def build_matrix_configs(
    *,
    profile: str,
    baseline_run_dir: str,
    target2_rmsnorm_run_dir: str,
    results_dir: str,
    n_eval: int,
    eval_datasets: list[str],
    model: str = "Qwen/Qwen3-30B-A3B",
) -> list[tuple[str, dict]]:
    if profile not in PROFILE_EXPERIMENTS:
        raise ValueError(f"unknown profile {profile!r}")
    configs: list[tuple[str, dict]] = []
    for target, allocation, ranking, name in PROFILE_EXPERIMENTS[profile]:
        plan_path = _validated_plan_path(
            selector=allocation,
            target=target,
            baseline_run_dir=baseline_run_dir,
            target2_rmsnorm_run_dir=target2_rmsnorm_run_dir,
        )
        cfg = _build_config(
            model=model,
            selector=ranking,
            target_pct=float(target),
            dataset=eval_datasets[0],
            n_eval=n_eval,
            channel_alignment=16,
            seed=42,
            max_seq_len=512,
            batch_size=4,
        )
        cfg.update({
            "output_dir": os.path.join(results_dir, name),
            "eval_datasets": eval_datasets,
            "moe_calib_dataset": "wikitext2",
            "allocation_source": allocation,
            "ranking_source": ranking,
            "moe_allocation_plan": plan_path,
            "moe_ranking_plan": None,
            "allocation_ranking_experiment_name": name,
            "moe_fixed_allocation_plan": None,
            "moe_fixed_allocation_selector": None,
            "moe_budget_mode": "global",
            "moe_pruning_mode": "packed_same_channel",
            "moe_same_channel_aggregation": "p95",
            "moe_channel_alignment": 16,
            "max_expert_frac": 0.2,
            "save_pruning_plan": True,
            "load_pruning_plan": None,
            "evaluation_protocol_label": (
                f"{profile};datasets={','.join(eval_datasets)};"
                f"n_eval={n_eval};max_seq_len=512"
            ),
        })
        configs.append((name, cfg))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_EXPERIMENTS), required=True)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--target2-rmsnorm-run-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--n-eval", type=int, required=True)
    parser.add_argument("--eval-datasets", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    datasets = [value.strip() for value in args.eval_datasets.split(",") if value.strip()]
    if not datasets:
        raise SystemExit("--eval-datasets must not be empty")
    try:
        configs = build_matrix_configs(
            profile=args.profile,
            baseline_run_dir=args.baseline_run_dir,
            target2_rmsnorm_run_dir=args.target2_rmsnorm_run_dir,
            results_dir=args.results_dir,
            n_eval=args.n_eval,
            eval_datasets=datasets,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[alloc-rank-config] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    manifest = []
    if not args.dry_run:
        os.makedirs(args.config_dir, exist_ok=True)
    for name, cfg in configs:
        path = os.path.join(
            args.config_dir,
            f"{name}_target{int(cfg['target_pruning_percents'][0])}_n{args.n_eval}.yaml",
        )
        manifest.append({
            "experiment_name": name,
            "config_path": path,
            "output_dir": cfg["output_dir"],
            "allocation_source": cfg["allocation_source"],
            "ranking_source": cfg["ranking_source"],
            "allocation_plan": cfg["moe_allocation_plan"],
            "target_pct": cfg["target_pruning_percents"][0],
        })
        print(
            f"[alloc-rank-config] {name}: allocation={cfg['allocation_source']} "
            f"ranking={cfg['ranking_source']} target={cfg['target_pruning_percents'][0]} "
            f"datasets={datasets} n_eval={args.n_eval}"
        )
        if args.dry_run:
            print(f"[alloc-rank-config] WOULD WRITE {path}")
        elif os.path.exists(path) and not args.overwrite:
            print(f"[alloc-rank-config] EXISTS {path} (use --overwrite)")
        else:
            _write_yaml(path, cfg)
            print(f"[alloc-rank-config] WRITE {path}")
    if not args.dry_run:
        manifest_path = os.path.join(args.config_dir, "matrix_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"[alloc-rank-config] MANIFEST {manifest_path}")


if __name__ == "__main__":
    main()
