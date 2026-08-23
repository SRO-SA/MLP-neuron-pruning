#!/usr/bin/env python3
"""Verify required milestone artifacts and write the immutable conclusion/index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone-root", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--systems-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.milestone_root)
    decision_path = Path(args.decision)
    systems = Path(args.systems_dir)
    required = [
        root / "matched_plan_validation.json",
        root / "certification_frontier" / "set_level_certificates.json",
        root / "certification_frontier" / "set_level_certificate_layers.csv",
        root / "certification_frontier" / "hybrid_frontier.json",
        root / "hybrid_ppl" / "paper_tables" / "hybrid_ppl_pareto.csv",
        root / "hybrid_downstream" / "paper_tables" / "downstream_benchmark_table.csv",
        root / "hybrid_downstream" / "paper_tables" / "downstream_paired_comparisons.csv",
        root / "pure_downnorm_curve" / "paper_tables" / "pure_downnorm_compression_curve.csv",
        decision_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not systems.is_dir() or not list(systems.rglob("*.json")):
        raise FileNotFoundError(f"systems benchmark JSON not found under {systems}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    allowed = {
        "The certified hybrid successfully recovers practical accuracy.",
        "The hybrid shows a partial certificate–accuracy trade-off.",
        "The hybrid fails, and the project proceeds as a diagnostic/certification paper.",
    }
    if decision["outcome_statement"] not in allowed:
        raise ValueError("conclusion is not one of the three allowed statements")
    conclusion = (
        "# Certified hybrid milestone conclusion\n\n"
        f"**{decision['outcome_statement']}**\n\n"
        f"Selected target-6 checkpoint: `{decision['selected_target6_checkpoint_label']}`.\n\n"
        "The ellipsoid theorem is used as an expert-channel and strict set-level "
        "local MoE certificate. The reported global value is an unpropagated sum "
        "of layer certificates, not an end-to-end Transformer guarantee. Down-norm "
        "is the calibration-free practical selection objective. No weights were "
        "updated. Full propagation-aware optimization remains out of scope.\n\n"
        + ("Per the stop/go rule, method development stops here and work returns "
           "to systems evidence and paper writing.\n" if decision["stop_method_development"]
           else "The bounded hybrid passed the stop/go rule; no additional sweep is authorized by this milestone.\n")
    )
    conclusion_path = root / "milestone_conclusion.md"
    manifest_path = root / "MILESTONE_MANIFEST.json"
    if args.dry_run:
        print(f"[hybrid-finalize] DRY RUN required={len(required)} outcome={decision['outcome_statement']}")
        return
    if conclusion_path.exists() or manifest_path.exists():
        raise FileExistsError("milestone conclusion/manifest already exists")
    conclusion_path.write_text(conclusion, encoding="utf-8")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest_path)
    manifest = {
        "schema_version": 1,
        "outcome_statement": decision["outcome_statement"],
        "selected_target6_checkpoint_label": decision["selected_target6_checkpoint_label"],
        "files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[hybrid-finalize] OK files={len(files)} conclusion={conclusion_path}")


if __name__ == "__main__":
    main()
