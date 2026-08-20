#!/usr/bin/env bash
set -euo pipefail
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/moe_selector_baselines/20260818_222729}"
RUN_DIR="${RUN_DIR:?set a new RUN_DIR}"
DRY_RUN="${DRY_RUN:-0}"
if [ -e "${RUN_DIR}" ]; then echo "[method-cost] ERROR: refusing ${RUN_DIR}"; exit 1; fi
if [ "${DRY_RUN}" = "1" ]; then
  echo "[method-cost] WOULD measure weight-only score construction; no calibration/PPL/pruning"
  exit 0
fi
mkdir -p "${RUN_DIR}"
python3 scripts/run_with_gpu_resource_monitor.py \
  --output "${RUN_DIR}/construction_cost.json" -- \
  env SOURCE_RUN_DIR="${SOURCE_RUN_DIR}" OUTPUT_DIR="${RUN_DIR}/score_only" \
  bash scripts/run_moe_plan_comparison.sh
python3 - "${RUN_DIR}" <<'PY'
import json, os, sys
root = sys.argv[1]
cost = json.load(open(os.path.join(root, "construction_cost.json")))
payload = {
  "schema_version": 1,
  "method": "RMSNorm allocation + RMSNorm ellipsoid within-layer ranking",
  "calibration_dataset": None,
  "calibration_tokens": 0,
  "forward_passes_for_score_construction": 0,
  "backward_passes_for_score_construction": 0,
  "requires_gradients": False,
  "requires_activations": False,
  "wall_clock_seconds_including_model_load_and_bound_diagnostics": cost["wall_clock_seconds"],
  "peak_gpu_memory_used_bytes_total": cost["peak_gpu_memory_used_bytes_total"],
  "peak_incremental_gpu_memory_used_bytes_total": cost["peak_incremental_gpu_memory_used_bytes_total"],
  "scope_note": "Measures model load plus both weight-only bound scores; plan comparison runs no pruning or PPL."
}
json.dump(payload, open(os.path.join(root, "method_protocol.json"), "x"), indent=2)
print("[method-cost] protocol:", os.path.join(root, "method_protocol.json"))
PY
