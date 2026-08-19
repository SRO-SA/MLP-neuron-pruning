from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.compare_moe_pruning_plans import compare_plans
from scripts.generate_moe_plan_replay_configs import build_replay_configs
from scripts.generate_moe_allocation_ranking_configs import build_matrix_configs
from src.moe_plan_replay import (
    build_allocation_ranking_selection,
    build_fixed_allocation_selection,
    validate_derived_replay_plan,
    validate_derived_allocation_ranking_plan,
)


def _plan(selector: str, counts=(2, 0)) -> dict:
    layers = []
    for layer_idx, count in enumerate(counts):
        indices = list(range(count))
        layers.append({
            "layer_idx": layer_idx,
            "prune_idx": indices,
            "old_intermediate": 8,
            "new_intermediate": 8 - count,
            "pruned_channels": count,
        })
    return {
        "model_id": "synthetic/model",
        "target_pct": 2.0,
        "actual_pct": 3.125,
        "selector": selector,
        "aggregation_mode": "p95",
        "pruning_mode": "packed_same_channel",
        "channel_alignment": 2,
        "max_layer_frac": 0.5,
        "num_layers": 2,
        "num_experts_per_layer": 4,
        "total_selected_layer_channels": sum(counts),
        "layers": layers,
    }


class FixedAllocationReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.source_path = os.path.join(self.tempdir.name, "source.json")
        self.source = _plan("rmsnorm_bound")
        with open(self.source_path, "w", encoding="utf-8") as handle:
            json.dump(self.source, handle)

    def _build(self, **overrides):
        kwargs = {
            "source_plan_path": self.source_path,
            "expected_source_selector": "rmsnorm_bound",
            "alternate_selector": "rmsnorm_ellipsoid_bound",
            "pruning_mode": "packed_same_channel",
            "channel_alignment": 2,
            "max_layer_frac": 0.5,
            "max_expert_frac": 0.5,
            "target_pct": 2.0,
            "scores_by_layer": {
                0: [5, 4, 3, 2, 1, 0, 6, 7],
                1: [0, 1, 2, 3, 4, 5, 6, 7],
            },
            "layer_sizes": {0: 8, 1: 8},
            "num_experts_by_layer": {0: 4, 1: 4},
            "total_expert_neurons": 64,
        }
        kwargs.update(overrides)
        return build_fixed_allocation_selection(self.source, **kwargs)

    def test_replay_preserves_counts_and_uses_alternate_ranking(self):
        selection, audit = self._build()
        self.assertEqual(selection[(0, -1)], [4, 5])
        self.assertEqual(selection[(1, -1)], [])
        self.assertEqual(audit["total_selected_layer_channels"], 2)
        self.assertEqual(audit["total_removed_expert_neurons"], 8)
        self.assertEqual(audit["source_allocation_selector"], "rmsnorm_bound")
        self.assertEqual(
            audit["alternate_channel_selector"], "rmsnorm_ellipsoid_bound"
        )

    def test_general_interface_same_source_reproduces_plan_ids(self):
        scores = {
            0: [0, 1, 2, 3, 4, 5, 6, 7],
            1: [0, 1, 2, 3, 4, 5, 6, 7],
        }
        selection, audit = build_allocation_ranking_selection(
            self.source,
            allocation_plan_path=self.source_path,
            allocation_source="rmsnorm_bound",
            ranking_source="rmsnorm_bound",
            pruning_mode="packed_same_channel",
            channel_alignment=2,
            max_layer_frac=0.5,
            max_expert_frac=0.5,
            target_pct=2.0,
            scores_by_layer=scores,
            layer_sizes={0: 8, 1: 8},
            num_experts_by_layer={0: 4, 1: 4},
            total_expert_neurons=64,
            experiment_name="rmsnorm_alloc__rmsnorm_rank",
        )
        self.assertEqual(selection[(0, -1)], [0, 1])
        self.assertEqual(audit["allocation_source"], "rmsnorm_bound")
        self.assertEqual(audit["ranking_source"], "rmsnorm_bound")
        self.assertEqual(audit["layers"][0]["jaccard"], 1.0)

    def test_general_interface_changes_only_ids_not_counts(self):
        selection, audit = build_allocation_ranking_selection(
            self.source,
            allocation_plan_path=self.source_path,
            allocation_source="rmsnorm_bound",
            ranking_source="rmsnorm_ellipsoid_bound",
            pruning_mode="packed_same_channel",
            channel_alignment=2,
            max_layer_frac=0.5,
            max_expert_frac=0.5,
            target_pct=2.0,
            scores_by_layer={
                0: [5, 4, 3, 2, 1, 0, 6, 7],
                1: list(range(8)),
            },
            layer_sizes={0: 8, 1: 8},
            num_experts_by_layer={0: 4, 1: 4},
            total_expert_neurons=64,
            experiment_name="rmsnorm_alloc__ellipsoid_rank",
        )
        self.assertEqual(selection[(0, -1)], [4, 5])
        self.assertEqual(audit["total_selected_layer_channels"], 2)
        self.assertEqual(audit["layers"][0]["ranking_score_min"], 0.0)
        self.assertEqual(audit["layers"][0]["ranking_score_max"], 7.0)
        derived = copy.deepcopy(self.source)
        derived["selector"] = "rmsnorm_ellipsoid_bound"
        for row in derived["layers"]:
            row["prune_idx"] = selection[(row["layer_idx"], -1)]
            row["pruned_channels"] = len(row["prune_idx"])
            row["new_intermediate"] = row["old_intermediate"] - len(
                row["prune_idx"]
            )
        derived["allocation_ranking"] = audit
        validate_derived_allocation_ranking_plan(derived, self.source)

    def test_fixed_plan_is_supported_for_allocation_and_ranking(self):
        ranking_plan = copy.deepcopy(self.source)
        ranking_plan["selector"] = "down_norm"
        ranking_plan["layers"][0]["prune_idx"] = [6, 7]
        ranking_path = os.path.join(self.tempdir.name, "ranking.json")
        with open(ranking_path, "w", encoding="utf-8") as handle:
            json.dump(ranking_plan, handle)
        selection, audit = build_allocation_ranking_selection(
            self.source,
            allocation_plan_path=self.source_path,
            allocation_source="fixed_plan",
            ranking_source="fixed_plan",
            pruning_mode="packed_same_channel",
            channel_alignment=2,
            max_layer_frac=0.5,
            max_expert_frac=0.5,
            target_pct=2.0,
            scores_by_layer={0: range(8), 1: range(8)},
            layer_sizes={0: 8, 1: 8},
            num_experts_by_layer={0: 4, 1: 4},
            total_expert_neurons=64,
            experiment_name="fixed_alloc__fixed_rank",
            ranking_plan=ranking_plan,
            ranking_plan_path=ranking_path,
        )
        self.assertEqual(selection[(0, -1)], [6, 7])
        self.assertEqual(audit["allocation_source"], "fixed_plan")
        self.assertEqual(audit["ranking_source"], "fixed_plan")

    def test_missing_source_plan_fails(self):
        with self.assertRaises(FileNotFoundError):
            self._build(source_plan_path=os.path.join(self.tempdir.name, "missing.json"))

    def test_physical_mode_change_fails(self):
        with self.assertRaisesRegex(ValueError, "physical pruning mode"):
            self._build(pruning_mode="per_expert_mask")

    def test_same_selector_fails(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            self._build(alternate_selector="rmsnorm_bound")

    def test_validator_cli_direct_invocation_can_import_src(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        completed = subprocess.run(
            [
                sys.executable,
                os.path.join("scripts", "validate_moe_plan_replay.py"),
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source", completed.stdout)

    def test_allocation_ranking_cli_entrypoints_start_directly(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for script in (
            "generate_moe_allocation_ranking_configs.py",
            "validate_moe_allocation_ranking.py",
            "summarize_moe_allocation_ranking.py",
            "compare_moe_hybrid_replication.py",
        ):
            completed = subprocess.run(
                [sys.executable, os.path.join("scripts", script), "--help"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)

    def test_matrix_runner_defaults_to_full_selector_baseline(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        runner_path = os.path.join(
            repo_root, "scripts", "run_moe_allocation_ranking_matrix.sh"
        )
        with open(runner_path, encoding="utf-8") as handle:
            runner = handle.read()
        self.assertIn(
            "results/moe_selector_baselines/20260818_203025", runner
        )
        self.assertNotIn(
            'BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-results/'
            'moe_selector_baselines/20260818_190239}"',
            runner,
        )

    def test_alignment_violation_fails(self):
        bad = copy.deepcopy(self.source)
        bad["layers"][0]["prune_idx"] = [0]
        bad["layers"][0]["pruned_channels"] = 1
        bad["layers"][0]["new_intermediate"] = 7
        bad["total_selected_layer_channels"] = 1
        with self.assertRaisesRegex(ValueError, "violates alignment"):
            build_fixed_allocation_selection(
                bad,
                source_plan_path=self.source_path,
                expected_source_selector="rmsnorm_bound",
                alternate_selector="rmsnorm_ellipsoid_bound",
                pruning_mode="packed_same_channel",
                channel_alignment=2,
                max_layer_frac=0.5,
                max_expert_frac=0.5,
                target_pct=2.0,
                scores_by_layer={0: range(8), 1: range(8)},
                layer_sizes={0: 8, 1: 8},
                num_experts_by_layer={0: 4, 1: 4},
                total_expert_neurons=64,
            )

    def test_declared_total_mismatch_fails(self):
        bad = copy.deepcopy(self.source)
        bad["total_selected_layer_channels"] = 4
        with self.assertRaisesRegex(ValueError, "total_selected_layer_channels"):
            build_fixed_allocation_selection(
                bad,
                source_plan_path=self.source_path,
                expected_source_selector="rmsnorm_bound",
                alternate_selector="rmsnorm_ellipsoid_bound",
                pruning_mode="packed_same_channel",
                channel_alignment=2,
                max_layer_frac=0.5,
                max_expert_frac=0.5,
                target_pct=2.0,
                scores_by_layer={0: range(8), 1: range(8)},
                layer_sizes={0: 8, 1: 8},
                num_experts_by_layer={0: 4, 1: 4},
                total_expert_neurons=64,
            )

    def test_post_save_validation_detects_changed_layer_count(self):
        selection, audit = self._build()
        derived = _plan("rmsnorm_ellipsoid_bound")
        for row in derived["layers"]:
            row["prune_idx"] = selection[(row["layer_idx"], -1)]
            row["pruned_channels"] = len(row["prune_idx"])
        derived["fixed_allocation_replay"] = audit
        validate_derived_replay_plan(derived, self.source)
        derived["layers"][0]["prune_idx"].append(7)
        with self.assertRaisesRegex(ValueError, "replay count changed"):
            validate_derived_replay_plan(derived, self.source)


class ConfigGeneratorTests(unittest.TestCase):
    def test_generates_only_c_and_d_with_expected_totals_sources(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for experiment, selector, total in (
                ("rmsnorm_bound_target2", "rmsnorm_bound", 832),
                (
                    "rmsnorm_ellipsoid_bound_target2",
                    "rmsnorm_ellipsoid_bound",
                    768,
                ),
            ):
                plan_dir = os.path.join(tempdir, experiment, "pruning_plans")
                os.makedirs(plan_dir)
                plan = {
                    "selector": selector,
                    "target_pct": 2.0,
                    "pruning_mode": "packed_same_channel",
                    "aggregation_mode": "p95",
                    "channel_alignment": 16,
                    "total_selected_layer_channels": total,
                    "layers": [
                        {"layer_idx": 0, "prune_idx": list(range(total))}
                    ],
                }
                with open(os.path.join(plan_dir, "plan.json"), "w") as handle:
                    json.dump(plan, handle)
            configs = build_replay_configs(
                source_run_dir=tempdir,
                results_dir=os.path.join(tempdir, "derived"),
                n_eval=128,
            )
            self.assertEqual(len(configs), 2)
            mapping = {label: cfg for label, cfg in configs}
            cfg_c = mapping["original_allocation_ellipsoid_ranking"]
            self.assertEqual(cfg_c["moe_fixed_allocation_selector"], "rmsnorm_bound")
            self.assertEqual(cfg_c["moe_selector"], "rmsnorm_ellipsoid_bound")
            cfg_d = mapping["ellipsoid_allocation_original_ranking"]
            self.assertEqual(
                cfg_d["moe_fixed_allocation_selector"],
                "rmsnorm_ellipsoid_bound",
            )
            self.assertEqual(cfg_d["moe_selector"], "rmsnorm_bound")

    def test_general_matrix_exposes_independent_allocation_and_ranking(self):
        with tempfile.TemporaryDirectory() as tempdir:
            baseline = os.path.join(tempdir, "baseline")
            trusted = os.path.join(tempdir, "trusted")
            for run_dir, selector, target in (
                (baseline, "rmsnorm_bound", 4),
                (baseline, "down_norm", 4),
                (baseline, "down_norm", 2),
                (trusted, "rmsnorm_bound", 2),
            ):
                plan_dir = os.path.join(
                    run_dir, f"{selector}_target{target}", "pruning_plans"
                )
                os.makedirs(plan_dir, exist_ok=True)
                plan = {
                    "selector": selector,
                    "target_pct": float(target),
                    "actual_pct": float(target),
                    "pruning_mode": "packed_same_channel",
                    "aggregation_mode": "p95",
                    "channel_alignment": 16,
                    "max_layer_frac": 0.1,
                    "total_selected_layer_channels": 16,
                    "layers": [{
                        "layer_idx": 0,
                        "old_intermediate": 768,
                        "new_intermediate": 752,
                        "pruned_channels": 16,
                        "prune_idx": list(range(16)),
                    }],
                }
                with open(os.path.join(plan_dir, "plan.json"), "w") as handle:
                    json.dump(plan, handle)
            configs = build_matrix_configs(
                profile="target4",
                baseline_run_dir=baseline,
                target2_rmsnorm_run_dir=trusted,
                results_dir=os.path.join(tempdir, "results"),
                n_eval=1024,
                eval_datasets=["wikitext2", "c4"],
            )
            self.assertEqual(len(configs), 4)
            mapping = {name: cfg for name, cfg in configs}
            hybrid = mapping["downnorm_alloc__ellipsoid_rank"]
            self.assertEqual(hybrid["allocation_source"], "down_norm")
            self.assertEqual(
                hybrid["ranking_source"], "rmsnorm_ellipsoid_bound"
            )
            self.assertEqual(hybrid["eval_datasets"], ["wikitext2", "c4"])
            self.assertEqual(hybrid["reconstruction_eval_samples"], 1024)


class PlanComparisonTests(unittest.TestCase):
    def test_comparison_contains_requested_allocation_and_score_fields(self):
        with tempfile.TemporaryDirectory() as tempdir:
            old = _plan("rmsnorm_bound", counts=(2, 0))
            ellipsoid = _plan("rmsnorm_ellipsoid_bound", counts=(0, 2))
            old_path = os.path.join(tempdir, "old.json")
            ellipsoid_path = os.path.join(tempdir, "ellipsoid.json")
            score_path = os.path.join(tempdir, "scores.json")
            with open(old_path, "w") as handle:
                json.dump(old, handle)
            with open(ellipsoid_path, "w") as handle:
                json.dump(ellipsoid, handle)
            metrics = {
                "legacy_min": 1.0, "legacy_median": 2.0,
                "legacy_p95": 3.0, "legacy_max": 4.0,
                "ellipsoid_min": 0.5, "ellipsoid_median": 1.0,
                "ellipsoid_p95": 1.5, "ellipsoid_max": 2.0,
                "ellipsoid_to_legacy_ratio_min": 0.5,
                "ellipsoid_to_legacy_ratio_median": 0.5,
                "ellipsoid_to_legacy_ratio_p95": 0.5,
                "ellipsoid_to_legacy_ratio_max": 0.5,
                "spearman_ellipsoid_vs_legacy": 0.75,
            }
            score_payload = {
                "legacy_selector": "rmsnorm_bound",
                "new_selector": "rmsnorm_ellipsoid_bound",
                "global": {"metrics": metrics},
                "layers": [
                    {"layer_idx": 0, "num_experts": 4, "metrics": metrics},
                    {"layer_idx": 1, "num_experts": 4, "metrics": metrics},
                ],
            }
            with open(score_path, "w") as handle:
                json.dump(score_payload, handle)
            report = compare_plans(old_path, ellipsoid_path, score_path)
            self.assertEqual(report["layers_selected_only_by_old"], [0])
            self.assertEqual(report["layers_selected_only_by_ellipsoid"], [1])
            self.assertEqual(report["global_selected_channel_jaccard"], 0.0)
            self.assertEqual(report["global_spearman_old_vs_ellipsoid"], 0.75)
            self.assertIn("ellipsoid_to_old_ratio_median", report["per_layer"][0])
            self.assertTrue(report["old_plan"]["overshot_requested_target"])


if __name__ == "__main__":
    unittest.main()
