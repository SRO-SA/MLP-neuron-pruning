#!/usr/bin/env python3
"""Freeze environment, license, timing, hash, and anonymous-code provenance."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_provenance import file_sha256


DATASET_LICENSES = {
    "allenai/c4": "ODC-BY",
    "Salesforce/wikitext": "CC-BY-SA-4.0",
    "Rowan/hellaswag": "MIT",
    "math_qa": "Apache-2.0",
    "allenai/openbookqa": "Apache-2.0",
    "baber/piqa": "unknown/not pinned in historical protocol",
    "allenai/winogrande": "unknown/not pinned in historical protocol",
    "allenai/ai2_arc": "CC-BY-SA-4.0",
}


def _command(args: list[str], *, required: bool = False) -> str:
    try:
        return subprocess.check_output(
            args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        if required:
            raise
        return ""


def _timings(run_root: Path) -> dict:
    rows = []
    for path in sorted(run_root.rglob("stage_timing.tsv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            stage, start, end, seconds, gpus, gpu_hours = line.split("\t")
            rows.append({
                "source": str(path), "stage": stage,
                "start_epoch": int(start), "end_epoch": int(end),
                "wall_seconds": int(seconds), "gpu_count": int(gpus),
                "gpu_hours": float(gpu_hours), "recovery_method": "exact_stage_timing",
            })
    progress_seconds = 0.0
    progress_matches = 0
    pattern = re.compile(r"\[(?:(\d+):)?(\d+):(\d+)<")
    for path in sorted(run_root.rglob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for hours, minutes, seconds in pattern.findall(text):
            progress_seconds += int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
            progress_matches += 1
    return {
        "exact_stage_rows": rows,
        "exact_wall_clock_hours_sum": sum(row["wall_seconds"] for row in rows) / 3600,
        "exact_gpu_hours_sum": sum(row["gpu_hours"] for row in rows),
        "historical_progress_bar_duration_hours_sum": progress_seconds / 3600,
        "historical_progress_bar_matches": progress_matches,
        "warning": (
            "Progress-bar durations can overlap exact stage timings and are not added "
            "to the exact total. Historical runs without stage timing do not support "
            "an exact total GPU-hour claim."
        ),
    }


def _anonymous_archive(output: Path) -> dict:
    tracked = _command(["git", "ls-files"], required=True).splitlines()
    allowed_prefixes = ("src/", "scripts/", "tests/", "configs/", "docs/")
    allowed_names = {
        "run_experiment.py", "requirements.txt", "environment.yml",
        "pyproject.toml", "setup.py", "LICENSE", "README.md",
    }
    identity_pattern = re.compile(
        rb"(?i)(sro-sa|c:\\users\\srosa|[a-z0-9._%+-]+@(?!example\.com)[a-z0-9.-]+\.[a-z]{2,})"
    )
    included, excluded_identity = [], []
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        if not (normalized.startswith(allowed_prefixes) or normalized in allowed_names):
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if identity_pattern.search(data):
            excluded_identity.append(normalized)
            continue
        included.append((normalized, path, data))
    manifest = {
        "schema_version": 1,
        "archive_purpose": "anonymous AXIOM reproducibility code",
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "included_files": [name for name, _, _ in included],
        "excluded_for_identity_review": excluded_identity,
        "weights_included": False,
        "result_corpora_included": False,
    }
    with tarfile.open(output, "w:gz", compresslevel=9) as archive:
        for name, _path, data in included:
            info = tarfile.TarInfo(f"axiom_anonymous_code/{name}")
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
        encoded = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("axiom_anonymous_code/ARCHIVE_MANIFEST.json")
        info.size = len(encoded)
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        archive.addfile(info, io.BytesIO(encoded))
    return {
        **manifest,
        "archive_path": str(output),
        "archive_sha256": file_sha256(str(output)),
        "archive_bytes": output.stat().st_size,
        "public_link_status": "ready for upload; no external link created by this script",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-run-dir", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--downstream-result", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.experiment_run_dir)
    capture_path = run_root / "baseline_capture" / "capture_manifest.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    result_paths = sorted((run_root / "plan_results").glob("*/result.json"))
    if len(result_paths) != 4:
        raise ValueError(f"expected four plan results, found {len(result_paths)}")
    downstream = {}
    if args.downstream_result:
        downstream = json.loads(Path(args.downstream_result).read_text(encoding="utf-8"))
    if args.dry_run:
        print(
            f"[submission-audit] DRY RUN results={len(result_paths)} "
            f"documents={capture['dataset']['document_count']}"
        )
        return

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    freeze = _command([sys.executable, "-m", "pip", "freeze"], required=True)
    (output / "environment_lock.txt").write_text(freeze + "\n", encoding="utf-8")
    hardware = _command([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    (output / "hardware.txt").write_text(hardware + "\n", encoding="utf-8")
    archive = _anonymous_archive(output / "axiom_anonymous_code.tar.gz")
    protocol = downstream.get("paper_v3_protocol", {})
    licenses = {
        "model": {
            "id": capture["model"], "revision": capture["model_revision"],
            "license": "Apache-2.0",
        },
        "local_validation_dataset": capture["dataset"],
        "historical_evaluation_datasets": [
            {
                "id": name, "license": license,
                "revision": "not pinned in historical evaluation protocol",
            }
            for name, license in DATASET_LICENSES.items()
        ],
        "evaluation_framework": {
            "name": "EleutherAI lm-evaluation-harness",
            "license": "MIT",
            "version_or_revision": protocol.get("harness", {}),
            "task_versions": protocol.get("task_versions", {}),
        },
    }
    artifact_hashes = {
        str(path.relative_to(run_root)): file_sha256(str(path))
        for path in [capture_path, *result_paths]
    }
    entrypoint_paths = {
        "orchestration": REPO_ROOT / "scripts/run_routed_moe_perturbation.sh",
        "capture": REPO_ROOT / "scripts/capture_routed_moe_baseline.py",
        "checkpoint_evaluation": REPO_ROOT / "scripts/evaluate_routed_moe_perturbation.py",
        "tables_and_figure": REPO_ROOT / "scripts/summarize_routed_moe_perturbation.py",
        "artifact_audit": REPO_ROOT / "scripts/build_axiom_submission_artifact_audit.py",
        "mathematical_runtime": REPO_ROOT / "src/routed_moe_perturbation.py",
    }
    audit = {
        "schema_version": 1,
        "experiment": "set-level routed-MoE local same-input perturbation",
        "git": {
            "commit": _command(["git", "rev-parse", "HEAD"]),
            "worktree_status": _command(["git", "status", "--porcelain"]),
            "diff_sha256": hashlib.sha256(
                _command(["git", "diff", "--binary"]).encode("utf-8")
            ).hexdigest(),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timing": _timings(run_root),
        "licenses_and_revisions": licenses,
        "entry_points": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": file_sha256(str(path)),
            }
            for name, path in entrypoint_paths.items()
        },
        "exact_command_documentation": "docs/ROUTED_MOE_PERTURBATION_RUNBOOK.md",
        "exact_command_documentation_sha256": file_sha256(str(
            REPO_ROOT / "docs/ROUTED_MOE_PERTURBATION_RUNBOOK.md"
        )),
        "artifact_hashes": artifact_hashes,
        "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
        "environment_lock_sha256": file_sha256(str(output / "environment_lock.txt")),
        "anonymous_code_archive": archive,
        "scope_guards": {
            "end_to_end_propagation": False,
            "plan_retuning": False,
            "weight_updates": False,
            "target8_run_included": False,
        },
    }
    (output / "submission_artifact_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8",
    )
    print(
        f"[submission-audit] OK exact_gpu_hours="
        f"{audit['timing']['exact_gpu_hours_sum']:.4f} archive={archive['archive_path']}"
    )


if __name__ == "__main__":
    main()
