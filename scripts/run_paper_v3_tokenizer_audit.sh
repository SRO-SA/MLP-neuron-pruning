#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:?set MANIFEST to the frozen four-checkpoint manifest}"
OUTPUT_DIR="${OUTPUT_DIR:?set a new OUTPUT_DIR}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-100}"
DRY_RUN="${DRY_RUN:-0}"

args=(--checkpoint-manifest "${MANIFEST}" --output-dir "${OUTPUT_DIR}"
      --samples-per-dataset "${SAMPLES_PER_DATASET}")
if [ "${DRY_RUN}" = "1" ]; then args+=(--dry-run); fi

python3 scripts/audit_paper_v3_tokenizers.py "${args[@]}"
