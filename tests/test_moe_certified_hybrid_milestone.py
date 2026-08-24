from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.build_moe_certified_hybrid_frontier import main as build_frontier
from scripts.build_moe_certified_hybrid_final_packet import (
    SOURCE_PATTERNS, hybrid_outcome, main as build_final_packet,
)
from scripts.generate_moe_hybrid_checkpoint_manifest import main as checkpoint_manifest
from scripts.generate_moe_fixed_plan_eval_configs import resolve_rmsnorm_allocation_plan
from scripts.select_moe_certified_hybrid_outcome import successful_plan_sort_key
from scripts.summarize_pure_downnorm_curve import audit_nesting
from scripts.validate_target6_matched_plans import merge_compatible_protocols
from src.experiment_provenance import file_sha256


def target6_plan(selector: str, *, down: bool = False) -> dict:
    layers = []
    for layer_idx in range(48):
        count = 48 if layer_idx < 47 else 32
        start = 32 if down else 0
        selected = list(range(start, start + count))
        layers.append({
            "layer_idx": layer_idx, "prune_idx": selected,
            "old_intermediate": 768, "new_intermediate": 768 - count,
            "pruned_channels": count,
        })
    return {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "transformers_version": "test", "torch_version": "test",
        "target_pct": 6.0, "actual_pct": 6.2066, "selector": selector,
        "aggregation_mode": "p95", "pruning_mode": "packed_same_channel",
        "channel_alignment": 16, "max_layer_frac": 0.2,
        "num_layers": 48, "num_experts_per_layer": 128,
        "total_selected_layer_channels": 2288, "layers": layers,
    }


class CertifiedHybridMilestoneTests(unittest.TestCase):
    def test_final_packet_validates_and_hashes_complete_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = {}
            for group, names in SOURCE_PATTERNS.items():
                directory = root / group
                directory.mkdir()
                directories[group] = directory
                for name in names:
                    (directory / name).write_text("placeholder", encoding="utf-8")
            frontier_plans = directories["certificate_frontier"] / "plans"
            frontier_plans.mkdir()
            ellipsoid_plan = frontier_plans / "ellipsoid_slack0.json"
            downnorm_plan = frontier_plans / "pure_down_norm.json"
            ellipsoid_plan.write_text("{}", encoding="utf-8")
            downnorm_plan.write_text("{}", encoding="utf-8")
            (directories["certificate_frontier"] / "hybrid_frontier.json").write_text(
                json.dumps({
                    "seed": 42, "predefined_slacks": [0.0, 0.0025],
                    "distinct_selection_count": 2, "ppl_evaluation_plan_count": 2,
                    "thresholds_were_not_adapted": True, "source_plans": {},
                    "plans": [
                        {
                            "plan": "ellipsoid_slack0",
                            "certificate_change_vs_ellipsoid_pct": 0.0,
                            "certificate_slack": 0.0,
                            "plan_path": str(ellipsoid_plan),
                            "plan_sha256": file_sha256(str(ellipsoid_plan)),
                        },
                        {
                            "plan": "pure_down_norm",
                            "certificate_change_vs_ellipsoid_pct": 2.1436,
                            "certificate_slack": "unconstrained",
                            "plan_path": str(downnorm_plan),
                            "plan_sha256": file_sha256(str(downnorm_plan)),
                        },
                    ],
                })
            )
            (directories["downstream"] / "downstream_statistical_audit.json").write_text("{}")
            with (
                directories["downstream"] / "downstream_benchmark_table.csv"
            ).open("w", newline="", encoding="utf-8") as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=["label", "task", "accuracy"])
                writer.writeheader()
                writer.writerows([
                    {
                        "label": "rmsnorm_alloc__ellipsoid_rank__p95__target6",
                        "task": "macro_average", "accuracy": 0.64990,
                    },
                    {
                        "label": "rmsnorm_alloc__downnorm_rank__p95__target6",
                        "task": "macro_average", "accuracy": 0.65796,
                    },
                ])
            checkpoint_labels = [
                "baseline_unpruned",
                "rmsnorm_alloc__downnorm_rank__p95__target2",
                "rmsnorm_alloc__downnorm_rank__p95__target4",
                "rmsnorm_alloc__downnorm_rank__p95__target6",
                "rmsnorm_alloc__downnorm_rank__p95__target8",
            ]
            with (directories["checkpoints"] / "checkpoint_table.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=["label"])
                writer.writeheader()
                writer.writerows({"label": label} for label in checkpoint_labels)
            system_rows = []
            for label in (
                "baseline_unpruned",
                "rmsnorm_alloc__downnorm_rank__p95__target6",
            ):
                for prompt in (128, 512, 2048, 4096, 8192):
                    system_rows.append({
                        "label": label,
                        "prefill_throughput_gain_ci95_lower_pct": (
                            "" if label == "baseline_unpruned" else "-1"
                        ),
                        "decode_throughput_gain_ci95_lower_pct": (
                            "" if label == "baseline_unpruned" else "-1"
                        ),
                        "load_hbm_reduction_vs_baseline_pct": (
                            "" if label == "baseline_unpruned" else "5"
                        ),
                        "peak_hbm_reduction_vs_baseline_pct": (
                            "" if label == "baseline_unpruned" else "4"
                        ),
                        "prompt_length_tokens": prompt,
                    })
            with (directories["systems"] / "systems_benchmark_table.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=list(system_rows[0]))
                writer.writeheader(); writer.writerows(system_rows)
            (directories["systems"] / "systems_benchmark_table.json").write_text(
                json.dumps({
                    "uncertainty": {"resamples": 10000, "seed": 42},
                    "raw_runs": [{"label": label} for label in checkpoint_labels[:2]],
                })
            )
            validation = root / "matched.json"
            validation.write_text("{}")
            argv = ["packet"]
            for option, group in (
                ("--frontier-dir", "certificate_frontier"),
                ("--ppl-dir", "ppl"),
                ("--downnorm-curve-dir", "downnorm_curve"),
                ("--downstream-dir", "downstream"),
                ("--checkpoint-dir", "checkpoints"),
                ("--systems-dir", "systems"),
            ):
                argv.extend((option, str(directories[group])))
            argv.extend((
                "--matched-validation", str(validation),
                "--output-dir", str(root / "packet"),
            ))
            with mock.patch.object(sys, "argv", argv):
                build_final_packet()
            manifest = json.loads(
                (root / "packet" / "FINAL_PACKET_MANIFEST.json").read_text()
            )
            self.assertTrue(manifest["stop_experimentation"])
            self.assertEqual(manifest["next_phase"], "paper writing")
            settings = json.loads(
                (root / "packet" / "exact_experimental_settings.json").read_text()
            )
            self.assertEqual(
                settings["certified_hybrid_decision"]["outcome"], "failure"
            )

    def test_hybrid_success_rule_is_frozen_and_machine_readable(self) -> None:
        frontier = {
            "plans": [
                {"plan": "ellipsoid_slack0", "certificate_change_vs_ellipsoid_pct": 0.0,
                 "certificate_slack": 0.0},
                {"plan": "downnorm_refinement_slack1",
                 "certificate_change_vs_ellipsoid_pct": 1.0,
                 "certificate_slack": 0.01},
                {"plan": "pure_down_norm", "certificate_change_vs_ellipsoid_pct": 2.1436,
                 "certificate_slack": "unconstrained"},
            ]
        }
        downstream = [
            {"label": "rmsnorm_alloc__ellipsoid_rank__p95__target6",
             "task": "macro_average", "accuracy": "0.6500"},
            {"label": "rmsnorm_alloc__downnorm_rank__p95__target6",
             "task": "macro_average", "accuracy": "0.6580"},
            {"label": "certified_hybrid__downnorm_refinement_slack1__target6",
             "task": "macro_average", "accuracy": "0.6565"},
        ]
        decision = hybrid_outcome(frontier, downstream)
        self.assertEqual(decision["outcome"], "success")
        self.assertTrue(decision["success_criterion_met"])

    def test_downnorm_nesting_audit_does_not_assume_monotonicity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for target, selections in {
                2: [[0, 1]],
                4: [[1, 2, 3, 4]],
            }.items():
                plan = {
                    "layers": [{
                        "layer_idx": 0, "old_intermediate": 8,
                        "new_intermediate": 8 - len(selections[0]),
                        "pruned_channels": len(selections[0]),
                        "prune_idx": selections[0],
                    }]
                }
                path = root / f"target{target}.json"
                path.write_text(json.dumps(plan))
                paths[target] = str(path)
            rows = audit_nesting(paths)
            self.assertFalse(rows[0]["fully_nested"])
            self.assertEqual(rows[0]["lower_channels_missing_from_upper"], 1)
            self.assertIn("independently optimized", rows[0]["interpretation"])

    def test_checkpoint_gate_exports_at_most_two_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ellipsoid = target6_plan("rmsnorm_ellipsoid_bound")
            down = target6_plan("down_norm", down=True)
            plan_paths = {}
            for name, plan in (("ellipsoid_slack0", ellipsoid),
                               ("pure_down_norm", down)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(plan))
                plan_paths[name] = path
            rows = [
                {
                    "plan": "ellipsoid_slack0", "plan_path": str(plan_paths["ellipsoid_slack0"]),
                    "certificate_slack": 0.0, "strict_certificate": 100.0,
                    "normalized_down_norm_objective": 20.0,
                    "certificate_objective_pareto_optimal": True,
                    "selection_sha256": "ellipsoid",
                },
                {
                    "plan": "pure_down_norm", "plan_path": str(plan_paths["pure_down_norm"]),
                    "certificate_slack": "unconstrained", "strict_certificate": 102.0,
                    "normalized_down_norm_objective": 10.0,
                    "certificate_objective_pareto_optimal": True,
                    "selection_sha256": "down",
                },
            ]
            for index, (slack, certificate, objective) in enumerate((
                (0.0025, 100.2, 18.0), (0.005, 100.5, 15.0),
                (0.01, 101.0, 12.0),
            )):
                plan = target6_plan("rmsnorm_ellipsoid_bound")
                for layer in plan["layers"]:
                    count = len(layer["prune_idx"])
                    start = 4 * (index + 1)
                    layer["prune_idx"] = list(range(start, start + count))
                name = f"intermediate_{index}"
                path = root / f"{name}.json"
                path.write_text(json.dumps(plan))
                rows.append({
                    "plan": name, "plan_path": str(path),
                    "certificate_slack": slack,
                    "strict_certificate": certificate,
                    "normalized_down_norm_objective": objective,
                    "certificate_objective_pareto_optimal": True,
                    "selection_sha256": name,
                })
            frontier = root / "frontier.json"
            frontier.write_text(json.dumps({"plans": rows}))
            existing = root / "existing.json"
            existing.write_text(json.dumps([
                {"label": "baseline_unpruned", "checkpoint_dir": "/baseline",
                 "target_pct": 0.0},
                {"label": "rmsnorm_alloc__ellipsoid_rank__p95__target6",
                 "checkpoint_dir": "/ellipsoid", "target_pct": 6.0,
                 "plan_path": str(plan_paths["ellipsoid_slack0"])},
                {"label": "rmsnorm_alloc__downnorm_rank__p95__target6",
                 "checkpoint_dir": "/down", "target_pct": 6.0,
                 "plan_path": str(plan_paths["pure_down_norm"])},
            ]))
            output = root / "manifest.json"
            argv = [
                "manifest", "--frontier-manifest", str(frontier),
                "--existing-checkpoint-manifest", str(existing),
                "--new-checkpoint-root", str(root / "checkpoints"),
                "--output", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                checkpoint_manifest()
            manifest = json.loads(output.read_text())
            intermediate_specs = [
                row for row in manifest if row["label"].startswith("certified_hybrid__")
            ]
            self.assertEqual(len(intermediate_specs), 2)
            self.assertEqual(
                {row["label"] for row in intermediate_specs},
                {
                    "certified_hybrid__intermediate_0__target6",
                    "certified_hybrid__intermediate_2__target6",
                },
            )
            selection = json.loads(
                (root / "manifest_downstream_selection.json").read_text()
            )
            self.assertFalse(selection["selection_uses_downstream_results"])
            self.assertIn("strongest-certificate", selection["selection_policy"])

    def test_success_tie_break_prefers_smallest_certificate_slack(self) -> None:
        common = {
            "macro_accuracy": 0.65796,
            "strict_certificate": 102.1436,
        }
        rows = [
            {**common, "plan": "downnorm_refinement_slack25",
             "certificate_slack": 0.25},
            {**common, "plan": "downnorm_refinement_slack10",
             "certificate_slack": 0.10},
            {**common, "plan": "downnorm_refinement_slack5",
             "certificate_slack": 0.05},
        ]

        selected = sorted(rows, key=successful_plan_sort_key)[0]

        self.assertEqual(selected["plan"], "downnorm_refinement_slack5")

    def test_fixed_plan_eval_uses_true_rmsnorm_allocation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier_dir = root / "certification_frontier"
            frontier_dir.mkdir()
            ellipsoid_path = root / "ellipsoid.json"
            rmsnorm_path = root / "rmsnorm.json"
            ellipsoid_path.write_text(json.dumps(
                target6_plan("rmsnorm_ellipsoid_bound")
            ))
            rmsnorm_path.write_text(json.dumps(target6_plan("rmsnorm_bound")))
            validation_path = root / "matched_plan_validation.json"
            validation_path.write_text(json.dumps({
                "strict_gate_passed": True,
                "plans": {
                    "rmsnorm_bound": {
                        "path": str(rmsnorm_path),
                        "sha256": file_sha256(str(rmsnorm_path)),
                    }
                },
            }))
            frontier_path = frontier_dir / "hybrid_frontier.json"
            payload = {
                "source_plans": {
                    "ellipsoid": {
                        "path": str(ellipsoid_path),
                        "sha256": file_sha256(str(ellipsoid_path)),
                    }
                }
            }
            frontier_path.write_text(json.dumps(payload))

            resolved = resolve_rmsnorm_allocation_plan(payload, frontier_path)

            self.assertEqual(resolved, rmsnorm_path)
            self.assertNotEqual(resolved, ellipsoid_path)

    def test_legacy_missing_protocol_metadata_is_not_a_mismatch(self) -> None:
        merged, coverage = merge_compatible_protocols({
            "legacy": {
                "common": {"model": "model", "revision": ""},
                "datasets": {"c4": {"corpus": "abc", "tokens": 10}},
            },
            "new": {
                "common": {"model": "model", "revision": "revision-1"},
                "datasets": {"c4": {"corpus": "abc", "tokens": 10}},
            },
        })
        self.assertEqual(merged["common"]["revision"], "revision-1")
        self.assertEqual(coverage["common.revision"], ["new"])
        with self.assertRaisesRegex(ValueError, "conflicting populated"):
            merge_compatible_protocols({
                "left": {"corpus": "abc"}, "right": {"corpus": "xyz"},
            })

    def test_frontier_dry_math_writes_fine_fixed_budget_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ellipsoid_path = root / "ellipsoid.json"
            down_path = root / "down.json"
            rmsnorm_path = root / "rmsnorm.json"
            ellipsoid_path.write_text(json.dumps(target6_plan("rmsnorm_ellipsoid_bound")))
            down_path.write_text(json.dumps(target6_plan("down_norm", down=True)))
            rmsnorm_path.write_text(json.dumps(target6_plan("rmsnorm_bound")))
            arrays = {}
            rng = np.random.default_rng(42)
            for layer in range(48):
                base = rng.uniform(1.0, 2.0, size=(3, 768)).astype(np.float32)
                down = np.tile(np.linspace(2.0, 0.1, 768), (3, 1)).astype(np.float32)
                arrays[f"layer_{layer}__ellipsoid"] = base
                arrays[f"layer_{layer}__down_norm"] = down
            bundle = root / "scores.npz"
            np.savez_compressed(bundle, **arrays)
            score_manifest = root / "scores.json"
            score_manifest.write_text("{}")
            validation = root / "validation.json"
            validation.write_text(json.dumps({
                "strict_gate_passed": True,
                "plans": {
                    "rmsnorm_bound": {
                        "path": str(rmsnorm_path),
                        "sha256": file_sha256(str(rmsnorm_path)),
                    }
                },
            }))
            output = root / "frontier"
            argv = [
                "build", "--ellipsoid-plan", str(ellipsoid_path),
                "--down-norm-plan", str(down_path), "--score-bundle", str(bundle),
                "--score-manifest", str(score_manifest),
                "--matched-validation", str(validation), "--output-dir", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                build_frontier()
            frontier = json.loads((output / "hybrid_frontier.json").read_text())
            self.assertEqual(len(frontier["plans"]), 8)
            self.assertEqual(
                frontier["predefined_slacks"],
                [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.021436],
            )
            for row in frontier["plans"]:
                plan = json.loads(Path(row["plan_path"]).read_text())
                self.assertEqual(plan["total_selected_layer_channels"], 2288)
                self.assertTrue(all(
                    len(layer["prune_idx"]) % 16 == 0 for layer in plan["layers"]
                ))
            self.assertEqual(
                json.loads((output / "set_level_certificates.json").read_text())
                ["pure_ellipsoid"]["inequality_violations"], 0,
            )
            evaluated = [row for row in frontier["plans"] if row["evaluate_ppl"]]
            self.assertEqual(
                len({row["selection_sha256"] for row in evaluated}),
                len(evaluated),
            )
            self.assertTrue(all(
                row["certificate_objective_pareto_optimal"] for row in evaluated
            ))


if __name__ == "__main__":
    unittest.main()
