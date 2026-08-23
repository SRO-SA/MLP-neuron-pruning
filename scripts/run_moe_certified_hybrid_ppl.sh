#!/usr/bin/env bash
# Five independent fresh model processes; no checkpoint or result is overwritten.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PLAN_MANIFEST="${PLAN_MANIFEST:?set PLAN_MANIFEST to hybrid_frontier.json}"
RUN_DIR="${RUN_DIR:?set a new RUN_DIR}"
VENV="${VENV:-/workspace/venvs/qwen-pruning}"
N_EVAL="${N_EVAL:-1024}"
DRY_RUN="${DRY_RUN:-0}"
CONFIG_DIR="${RUN_DIR}/configs"
MANIFEST="${CONFIG_DIR}/fixed_plan_eval_manifest.json"

if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi
if [ "${DRY_RUN}" != "1" ] && [ -e "${RUN_DIR}" ]; then
  echo "[hybrid-ppl] ERROR: refusing to overwrite ${RUN_DIR}"
  exit 1
fi

args=(--plan-manifest "${PLAN_MANIFEST}" --results-dir "${RUN_DIR}" \
  --config-dir "${CONFIG_DIR}" --eval-datasets wikitext2,c4 \
  --n-eval "${N_EVAL}" --overwrite)
if [ "${DRY_RUN}" = "1" ]; then
  python3 scripts/generate_moe_fixed_plan_eval_configs.py "${args[@]}" --dry-run
  echo "[hybrid-ppl] DRY RUN complete; no model loaded"
  exit 0
fi

python3 -c "import torch; assert torch.cuda.is_available(); print('[hybrid-ppl] CUDA OK', torch.__version__)"
mkdir -p "${RUN_DIR}"
python3 scripts/generate_moe_fixed_plan_eval_configs.py "${args[@]}"

while IFS=$'\t' read -r name config output plan plan_hash allocation_plan; do
  mkdir -p "${output}"
  echo "[hybrid-ppl] START fresh process ${name}"
  python3 run_experiment.py --config "${config}" --moe-target-pruning \
    2>&1 | tee "${RUN_DIR}/${name}.log"
  derived_plan="$(find "${output}/pruning_plans" -maxdepth 1 -name '*.json' -print -quit)"
  [ -n "${derived_plan}" ] || { echo "[hybrid-ppl] ERROR: derived plan missing"; exit 1; }
  python3 scripts/validate_moe_allocation_ranking.py \
    --allocation-plan "${allocation_plan}" --derived-plan "${derived_plan}" \
    --allocation-source rmsnorm_bound --ranking-source fixed_plan \
    --experiment-name "${name}"
  python3 - "${output}" "${plan}" "${plan_hash}" "${derived_plan}" <<'PY'
import csv, glob, hashlib, json, os, sys
output, plan, expected_hash, derived_plan = sys.argv[1:]
matches = [p for p in glob.glob(os.path.join(output, "moe_target_pruning_*.csv"))
           if not p.endswith("_per_layer.csv")]
assert len(matches) == 1, matches
rows = list(csv.DictReader(open(matches[0], newline="", encoding="utf-8")))
assert len(rows) == 2, len(rows)
assert {row["eval_dataset"] for row in rows} == {"wikitext2", "c4"}
assert all(int(float(row["selected_layer_channels"])) == 2288 for row in rows)
assert all(int(float(row["removed_expert_neurons"])) == 292864 for row in rows)
assert all(row["ranking_source"] == "fixed_plan" for row in rows)
assert all(row["allocation_source"] == "rmsnorm_bound" for row in rows)
assert all(row["evaluation_token_count_match"].lower() in {"true", "1"} for row in rows)
assert all(row["forward_check"] == "OK" and row["status"] == "ok" for row in rows)
digest = hashlib.sha256(open(plan, "rb").read()).hexdigest()
assert digest == expected_hash
source = json.load(open(plan, encoding="utf-8"))
derived = json.load(open(derived_plan, encoding="utf-8"))
source_ids = {int(r["layer_idx"]): sorted(r["prune_idx"]) for r in source["layers"]}
derived_ids = {int(r["layer_idx"]): sorted(r["prune_idx"]) for r in derived["layers"]}
assert source_ids == derived_ids, "derived plan changed fixed-plan identities"
print(f"[hybrid-ppl-validate] OK plan={plan} datasets=2 channels=2288")
PY
  echo "[hybrid-ppl] FINISH ${name}"
done < <(python3 - "${MANIFEST}" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8")):
    print("\t".join([row["experiment_name"], row["config_path"],
                     row["output_dir"], row["plan_path"], row["plan_sha256"],
                     row["allocation_plan"]]))
PY
)

python3 scripts/summarize_moe_certified_hybrid_ppl.py \
  --run-dir "${RUN_DIR}" --manifest "${MANIFEST}" \
  --output-dir "${RUN_DIR}/paper_tables"
echo "[hybrid-ppl] COMPLETE ${RUN_DIR}"
