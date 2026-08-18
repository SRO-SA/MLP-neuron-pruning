#!/usr/bin/env python3
"""Generate the two bounded target-2 fixed-allocation replay configs."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

try:
    from generate_moe_selector_baseline_configs import _build_config, _write_yaml
except ImportError:  # imported as scripts.generate_moe_plan_replay_configs
    from scripts.generate_moe_selector_baseline_configs import _build_config, _write_yaml


EXPERIMENTS = (
    (
        "original_allocation_ellipsoid_ranking",
        "rmsnorm_bound_target2",
        "rmsnorm_bound",
        "rmsnorm_ellipsoid_bound",
        832,
    ),
    (
        "ellipsoid_allocation_original_ranking",
        "rmsnorm_ellipsoid_bound_target2",
        "rmsnorm_ellipsoid_bound",
        "rmsnorm_bound",
        768,
    ),
)


def find_source_plan(source_run_dir: str, experiment_dir: str) -> str:
    pattern = os.path.join(
        source_run_dir, experiment_dir, "pruning_plans", "*.json"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one source plan matching {pattern!r}; found {matches}"
        )
    return matches[0]


def build_replay_configs(
    *,
    source_run_dir: str,
    results_dir: str,
    n_eval: int,
    model: str = "Qwen/Qwen3-30B-A3B",
    dataset: str = "wikitext2",
) -> list[tuple[str, dict]]:
    configs = []
    for (
        label,
        source_experiment,
        source_selector,
        alternate_selector,
        expected_total,
    ) in EXPERIMENTS:
        source_plan = find_source_plan(source_run_dir, source_experiment)
        with open(source_plan, encoding="utf-8") as handle:
            source_payload = json.load(handle)
        counted_total = sum(
            len(layer.get("prune_idx", []))
            for layer in source_payload.get("layers", [])
        )
        checks = {
            "selector": (source_payload.get("selector"), source_selector),
            "target_pct": (float(source_payload.get("target_pct", -1)), 2.0),
            "pruning_mode": (
                source_payload.get("pruning_mode"), "packed_same_channel"
            ),
            "aggregation_mode": (source_payload.get("aggregation_mode"), "p95"),
            "channel_alignment": (
                int(source_payload.get("channel_alignment", -1)), 16
            ),
            "total_selected_layer_channels": (
                int(source_payload.get("total_selected_layer_channels", -1)),
                expected_total,
            ),
            "counted_layer_channels": (counted_total, expected_total),
        }
        for field, (actual, expected) in checks.items():
            if actual != expected:
                raise ValueError(
                    f"{source_plan}: {field}={actual!r}, expected {expected!r}"
                )
        cfg = _build_config(
            model=model,
            selector=alternate_selector,
            target_pct=2.0,
            dataset=dataset,
            n_eval=n_eval,
            channel_alignment=16,
            seed=42,
            max_seq_len=512,
            batch_size=4,
        )
        cfg.update({
            "output_dir": os.path.join(results_dir, label),
            "moe_fixed_allocation_plan": source_plan,
            "moe_fixed_allocation_selector": source_selector,
            "moe_budget_mode": "global",
            "moe_pruning_mode": "packed_same_channel",
            "moe_same_channel_aggregation": "p95",
            "moe_channel_alignment": 16,
            "max_expert_frac": 0.2,
            "save_pruning_plan": True,
            "load_pruning_plan": None,
        })
        configs.append((label, cfg))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config-dir", default="configs/moe_plan_replay")
    parser.add_argument("--n-eval", type=int, choices=(128, 512), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        configs = build_replay_configs(
            source_run_dir=args.source_run_dir,
            results_dir=args.results_dir,
            n_eval=args.n_eval,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[replay-config] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not args.dry_run:
        os.makedirs(args.config_dir, exist_ok=True)
    for label, cfg in configs:
        path = os.path.join(args.config_dir, f"{label}_n{args.n_eval}.yaml")
        print(
            f"[replay-config] {label}: fixed_allocation="
            f"{cfg['moe_fixed_allocation_selector']} channel_selector="
            f"{cfg['moe_selector']} source={cfg['moe_fixed_allocation_plan']}"
        )
        if args.dry_run:
            print(f"[replay-config] WOULD WRITE {path}")
        elif os.path.exists(path) and not args.overwrite:
            print(f"[replay-config] EXISTS {path} (use --overwrite)")
        else:
            _write_yaml(path, cfg)
            print(f"[replay-config] WRITE {path}")


if __name__ == "__main__":
    main()
