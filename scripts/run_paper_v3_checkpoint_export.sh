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
