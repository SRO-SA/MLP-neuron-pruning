#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:?Set MANIFEST to the generated checkpoint manifest JSON}"
DRY_RUN="${DRY_RUN:-0}"
DTYPE="${DTYPE:-bfloat16}"
SKIP_VERIFIED_EXISTING="${SKIP_VERIFIED_EXISTING:-0}"
CHECKPOINT_TABLE_DIR="${CHECKPOINT_TABLE_DIR:-$(dirname "${MANIFEST}")/checkpoint_paper_tables}"
ESTIMATED_CHECKPOINT_GIB="${ESTIMATED_CHECKPOINT_GIB:-65}"
CHECKPOINT_HEADROOM_GIB="${CHECKPOINT_HEADROOM_GIB:-20}"

python3 - "${MANIFEST}" "${DRY_RUN}" "${SKIP_VERIFIED_EXISTING}" \
    "${ESTIMATED_CHECKPOINT_GIB}" "${CHECKPOINT_HEADROOM_GIB}" <<'PY'
import json
import os
import shutil
import sys

manifest = sys.argv[1]
dry_run = sys.argv[2] == "1"
skip_existing = sys.argv[3] == "1"
estimated_checkpoint_gib = float(sys.argv[4])
headroom_gib = float(sys.argv[5])

with open(manifest, encoding="utf-8") as handle:
    specs = json.load(handle)
if not isinstance(specs, list) or not specs:
    raise ValueError(f"checkpoint manifest must be a non-empty list: {manifest}")

pending = []
for spec in specs:
    verification_path = os.path.join(
        spec["checkpoint_dir"], "checkpoint_verification.json"
    )
    if skip_existing and os.path.isfile(verification_path):
        continue
    pending.append(spec)

checkpoint_paths = [os.path.abspath(spec["checkpoint_dir"]) for spec in specs]
common_root = os.path.commonpath(checkpoint_paths)
probe_path = common_root
while not os.path.exists(probe_path):
    parent = os.path.dirname(probe_path)
    if parent == probe_path:
        raise FileNotFoundError(
            f"could not find an existing ancestor for checkpoint root {common_root}"
        )
    probe_path = parent

free_gib = shutil.disk_usage(probe_path).free / (1024 ** 3)
required_gib = len(pending) * estimated_checkpoint_gib + headroom_gib
print(
    "[checkpoint-preflight] "
    f"root={common_root} filesystem_probe={probe_path} "
    f"pending={len(pending)} free_gib={free_gib:.1f} "
    f"estimated_required_gib={required_gib:.1f} "
    f"({estimated_checkpoint_gib:.1f} GiB/checkpoint + "
    f"{headroom_gib:.1f} GiB headroom)",
    flush=True,
)
if free_gib < required_gib:
    message = (
        "insufficient free space for the pending physical checkpoints: "
        f"need approximately {required_gib:.1f} GiB, found {free_gib:.1f} GiB. "
        "Choose a checkpoint root on a larger filesystem or free space, then "
        "regenerate the manifest."
    )
    if dry_run:
        print(f"[checkpoint-preflight] WARNING: {message}", flush=True)
    else:
        raise SystemExit(f"[checkpoint-preflight] ERROR: {message}")
PY

python3 - "${MANIFEST}" "${DRY_RUN}" "${DTYPE}" "${SKIP_VERIFIED_EXISTING}" <<'PY'
import json, os, shlex, subprocess, sys

manifest, dry_run, dtype = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
skip_existing = sys.argv[4] == "1"
with open(manifest, encoding="utf-8") as handle:
    specs = json.load(handle)
for spec in specs:
    verification_path = os.path.join(
        spec["checkpoint_dir"], "checkpoint_verification.json"
    )
    if os.path.isfile(verification_path) and skip_existing:
        verification = json.load(open(verification_path, encoding="utf-8"))
        expected = {
            "label": spec["label"], "plan_sha256": spec["plan_sha256"],
            "removed_layer_channels": spec["removed_layer_channels"],
            "removed_expert_neurons": spec["removed_expert_neurons"],
            "successful_reload": True, "exact_logits_after_reload": True,
            "no_hidden_original_width_padding": True,
        }
        mismatches = {key: (verification.get(key), value)
                      for key, value in expected.items()
                      if verification.get(key) != value}
        if mismatches:
            raise ValueError(
                f"existing checkpoint verification mismatch for {spec['label']}: {mismatches}"
            )
        print(f"[checkpoint-run] VERIFIED EXISTING {spec['label']}", flush=True)
        continue
    command = [
        sys.executable, "scripts/export_verify_moe_checkpoint.py",
        "--model", spec["model"], "--checkpoint-dir", spec["checkpoint_dir"],
        "--label", spec["label"], "--dtype", dtype,
        "--expected-layer-channels", str(spec["removed_layer_channels"]),
        "--expected-expert-neurons", str(spec["removed_expert_neurons"]),
    ]
    if spec["plan_path"]:
        command += [
            "--plan", spec["plan_path"],
            "--expected-plan-sha256", spec["plan_sha256"],
        ]
    print("[checkpoint-run] " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)
PY
if [ "${DRY_RUN}" != "1" ]; then
    python3 scripts/summarize_paper_v3_checkpoints.py \
        --manifest "${MANIFEST}" \
        --output-dir "${CHECKPOINT_TABLE_DIR}"
fi
