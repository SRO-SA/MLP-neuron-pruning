#!/usr/bin/env python3
"""Generate fresh-process PPL configs for fixed physical pruning plans."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_moe_selector_baseline_configs import _build_config, _write_yaml
from src.experiment_provenance import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--eval-datasets", default="wikitext2,c4")
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.plan_manifest)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    plan_rows = payload.get("plans", payload if isinstance(payload, list) else [])
    allocation_plan = Path(payload["source_plans"]["ellipsoid"]["path"])
    if not allocation_plan.is_file():
        raise FileNotFoundError(allocation_plan)
    if file_sha256(str(allocation_plan)) != payload["source_plans"]["ellipsoid"]["sha256"]:
        raise ValueError("frozen RMSNorm allocation plan hash changed")
    datasets = [value.strip() for value in args.eval_datasets.split(",") if value.strip()]
    if datasets != ["wikitext2", "c4"]:
        raise ValueError("milestone protocol requires eval-datasets=wikitext2,c4")
    config_dir = Path(args.config_dir)
    results_dir = Path(args.results_dir)
    if not args.dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in plan_rows:
        name = str(row["plan"])
        plan_path = Path(row["plan_path"])
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        if file_sha256(str(plan_path)) != row["plan_sha256"]:
            raise ValueError(f"plan hash mismatch: {plan_path}")
        cfg = _build_config(
            model=args.model, selector="rmsnorm_ellipsoid_bound",
            target_pct=6.0, dataset=datasets[0], n_eval=args.n_eval,
            channel_alignment=16, seed=42, max_seq_len=512, batch_size=4,
        )
        output_dir = results_dir / name
        cfg.update({
            "output_dir": str(output_dir),
            "eval_datasets": datasets,
            "moe_calib_dataset": "wikitext2",
            "moe_selector_needs_calib": False,
            "allocation_source": "rmsnorm_bound",
            "ranking_source": "fixed_plan",
            "moe_allocation_plan": str(allocation_plan),
            "moe_ranking_plan": str(plan_path),
            "allocation_ranking_experiment_name": name,
            "exact_total_layer_channels": 2288,
            "save_pruning_plan": True,
            "load_pruning_plan": None,
            "collect_per_example_nll": True,
            "paired_bootstrap_resamples": 10000,
            "save_bound_tightness": False,
            "save_expert_bound_scores": False,
            "evaluation_protocol_label": (
                f"certified_hybrid_frontier;datasets=wikitext2,c4;"
                f"n_eval={args.n_eval};max_seq_len=512"
            ),
        })
        config_path = config_dir / f"{name}_n{args.n_eval}.yaml"
        item = {
            **row,
            "experiment_name": name,
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "allocation_plan": str(allocation_plan),
        }
        manifest.append(item)
        print(f"[fixed-plan-config] {name}: plan={plan_path} output={output_dir}")
        if args.dry_run:
            print(f"[fixed-plan-config] WOULD WRITE {config_path}")
        elif config_path.exists() and not args.overwrite:
            raise FileExistsError(config_path)
        else:
            _write_yaml(str(config_path), cfg)
    if not args.dry_run:
        manifest_path = config_dir / "fixed_plan_eval_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[fixed-plan-config] OK configs={len(manifest)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
