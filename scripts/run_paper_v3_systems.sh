#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:?set MANIFEST to checkpoint_specs.json}"
RUN_DIR="${RUN_DIR:?set a new RUN_DIR}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
INCLUDE8="${INCLUDE8:-0}"
DRY_RUN="${DRY_RUN:-0}"
CASES="${CASES:-1x128,1x512,1x2048,2x512,4x512}"
ONLY_TARGETS="${ONLY_TARGETS:-}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi
if [ "${DRY_RUN}" != "1" ] && [ -e "${RUN_DIR}" ]; then
  echo "[systems] ERROR: refusing to overwrite ${RUN_DIR}"; exit 1
fi

while IFS=$'\t' read -r label checkpoint target comparator; do
  if [ "${comparator}" = "True" ]; then continue; fi
  if [ "${target}" = "8.0" ] && [ "${INCLUDE8}" != "1" ]; then continue; fi
  if [ -n "${ONLY_TARGETS}" ]; then
    target_int="${target%%.*}"
    case ",${ONLY_TARGETS}," in
      *",${target_int},"*) ;;
      *) continue ;;
    esac
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[systems] WOULD RUN ${label}: ${checkpoint}"
  else
    mkdir -p "${RUN_DIR}/${label}"
    python3 scripts/benchmark_paper_v3_inference.py \
      --checkpoint "${checkpoint}" --label "${label}" \
      --output "${RUN_DIR}/${label}/systems.json" --cases "${CASES}" \
      2>&1 | tee "${RUN_DIR}/${label}/run.log"
  fi
done < <(python3 - "${MANIFEST}" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    print(f"{r['label']}\t{r['checkpoint_dir']}\t{r['target_pct']}\t{r.get('downstream_comparator_only', False)}")
PY
)
if [ "${DRY_RUN}" = "1" ]; then exit 0; fi
python3 scripts/summarize_paper_v3_systems.py \
  --run-dir "${RUN_DIR}" --output-dir "${RUN_DIR}/paper_tables"
