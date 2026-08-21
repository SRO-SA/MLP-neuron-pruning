# Version 3 next-phase runbook

All commands are run from the repository root on the GPU server. Every output
directory below is new; frozen directories are read-only inputs.

## 0. Session variables

```bash
cd ~/workspace/MLP-neuron-pruning
source /workspace/venvs/qwen-pruning/bin/activate

RUN_ID="$(date +%Y%m%d_%H%M%S)"
NEXT_ROOT="results/paper_v3_next_phase/${RUN_ID}"
MILESTONE_DIR="results/paper_v3_summary/qwen3_30b_milestone_20260820_052129"
MILESTONE_CSV="${MILESTONE_DIR}/paper_v3_milestone_summary.csv"
PAIRED_DNLL_SOURCE="${MILESTONE_DIR}/paper_v3_paired_comparisons.csv"
PRIMARY_MANIFEST="results/paper_v3_post_milestone/20260820_060635/checkpoint_specs_paths_v2.json"
CHECKPOINT_ROOT="/paper_v3_checkpoints/20260820_060635"
TOKENIZER_AUDIT="results/paper_v3_post_freeze/20260820_postfreeze_v1/tokenizer_audit_v3/tokenizer_audit.json"
PRIMARY_DOWNSTREAM="results/paper_v3_post_freeze/20260820_postfreeze_v1/downstream_primary_batch8_v1"
LM_EVAL_IDENTITY="$(python3 - <<'PY'
from scripts.run_paper_v3_lm_eval import harness_identity
x = harness_identity()
print(x['git_revision'] or x['package_version'])
PY
)"

mkdir -p "${NEXT_ROOT}"
```

## 1. Statistical and nesting audit (CPU-only)

First perform no-write validation:

```bash
python3 scripts/build_paper_v3_paired_table.py \
  --input "${PAIRED_DNLL_SOURCE}" \
  --source-root results/moe_allocation_ranking \
  --profile complete \
  --output-dir "${NEXT_ROOT}/paired_dnll_audit" \
  --dry-run

python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "${PRIMARY_MANIFEST}" \
  --run-dir "${PRIMARY_DOWNSTREAM}" \
  --output-dir "${NEXT_ROOT}/downstream_statistical_audit" \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --dry-run

python3 scripts/audit_paper_v3_plan_nesting.py \
  --checkpoint-manifest "${PRIMARY_MANIFEST}" \
  --output-dir "${NEXT_ROOT}/plan_nesting_audit" \
  --dry-run
```

Then remove only `--dry-run` from each command. Expected outputs include paired
randomization p-values, Holm and BH adjusted p-values, the task-stratified macro
bootstrap audit, all example identifiers and task versions, and plan nesting
tables for 4→6 and 6→8.

## 2. Generate the optional target-6 down-norm-ranking plan

This is one fresh model process, not a broad sweep:

```bash
T6_DOWN_RUN="results/moe_allocation_ranking/target6_rmsnorm_downnorm_rank_n1024_${RUN_ID}"

PROFILE=target6_rmsnorm_downnorm_ranking_only \
RUN_DIR="${T6_DOWN_RUN}" \
N_EVAL=1024 \
EVAL_DATASETS=wikitext2,c4 \
BASELINE_RUN_DIR=results/moe_selector_baselines/20260818_203025 \
TARGET2_RMSNORM_RUN_DIR=results/moe_selector_baselines/20260818_222729 \
DRY_RUN=1 \
bash scripts/run_moe_allocation_ranking_matrix.sh
```

Repeat without `DRY_RUN=1`. Validation must report the same RMSNorm target-6
allocation vector and exactly 2288 removed layer-channels.

## 3. Add target 2 and target-6 comparator checkpoints

```bash
EXPANDED_MANIFEST="${NEXT_ROOT}/checkpoint_specs_target2_target6_attribution.json"

python3 scripts/generate_paper_v3_checkpoint_manifest.py \
  --summary-csv "${MILESTONE_CSV}" \
  --output "${EXPANDED_MANIFEST}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --include-target2-primary \
  --include-target6-comparators \
  --additional-target6-downnorm-run-dir "${T6_DOWN_RUN}" \
  --dry-run
```

Repeat without `--dry-run`, then check storage. Four new checkpoints require
roughly 240–280 GiB of free space including safety headroom.

```bash
df -hT "${CHECKPOINT_ROOT}"

DRY_RUN=1 \
MANIFEST="${EXPANDED_MANIFEST}" \
CHECKPOINT_TABLE_DIR="${NEXT_ROOT}/checkpoint_tables" \
bash scripts/run_paper_v3_checkpoint_export.sh

MANIFEST="${EXPANDED_MANIFEST}" \
CHECKPOINT_TABLE_DIR="${NEXT_ROOT}/checkpoint_tables" \
SKIP_VERIFIED_EXISTING=1 \
bash scripts/run_paper_v3_checkpoint_export.sh
```

Existing baseline/4/6/8 directories must print `VERIFIED EXISTING`; only target
2 and the target-6 selector comparators are added.

## 4. Incremental downstream evaluation at batch 8

```bash
T6_ATTR_RUN="${NEXT_ROOT}/downstream_target6_attribution_batch8"
TARGET2_RUN="${NEXT_ROOT}/downstream_target2_batch8"

MANIFEST="${EXPANDED_MANIFEST}" TOKENIZER_AUDIT="${TOKENIZER_AUDIT}" \
RUN_DIR="${T6_ATTR_RUN}" LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" \
RUN_KIND=comparator_only SKIP_SUMMARY=1 BATCH_SIZE=8 DTYPE=bfloat16 \
SEED=42 INCLUDE_OPTIONAL=0 TRUST_DATASET_CODE=1 DRY_RUN=1 \
bash scripts/run_paper_v3_downstream.sh

MANIFEST="${EXPANDED_MANIFEST}" TOKENIZER_AUDIT="${TOKENIZER_AUDIT}" \
RUN_DIR="${TARGET2_RUN}" LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" \
RUN_KIND=additional_only SKIP_SUMMARY=1 BATCH_SIZE=8 DTYPE=bfloat16 \
SEED=42 INCLUDE_OPTIONAL=0 TRUST_DATASET_CODE=1 DRY_RUN=1 \
bash scripts/run_paper_v3_downstream.sh
```

Repeat both commands without `DRY_RUN=1`. Then combine existing and additive
raw results without rerunning baseline/4/6/8:

```bash
COMBINED_DOWNSTREAM="${NEXT_ROOT}/downstream_combined_tables"

python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "${EXPANDED_MANIFEST}" \
  --run-dir "${PRIMARY_DOWNSTREAM}" \
  --run-dir "${T6_ATTR_RUN}" \
  --run-dir "${TARGET2_RUN}" \
  --output-dir "${COMBINED_DOWNSTREAM}" \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --dry-run
```

Repeat without `--dry-run`. The target-6 selector attribution compares
ellipsoid with activation, RMSNorm, and down-norm rankings on identical examples.

## 5. Pareto table

```bash
python3 scripts/build_paper_v3_pareto_table.py \
  --milestone-csv "${MILESTONE_CSV}" \
  --checkpoint-table-csv "${NEXT_ROOT}/checkpoint_tables/checkpoint_table.csv" \
  --downstream-table-csv "${COMBINED_DOWNSTREAM}/downstream_benchmark_table.csv" \
  --output-dir "${NEXT_ROOT}/pareto_tables" \
  --dry-run
```

Repeat without `--dry-run`. Pareto status is computed from declared objectives;
no target is manually preferred.

## 6. Systems benchmark: baseline versus target 6 first

```bash
SYSTEMS_T6="${NEXT_ROOT}/systems_baseline_target6"

MANIFEST="${EXPANDED_MANIFEST}" TOKENIZER_AUDIT="${TOKENIZER_AUDIT}" \
RUN_DIR="${SYSTEMS_T6}" ONLY_TARGETS="0,6" INCLUDE8=0 \
CASES="1x128,1x1024,2x128,2x1024" DECODE_TOKENS=32 \
WARMUPS=3 REPETITIONS=10 PROFILE_GEMM_SHAPES=1 DRY_RUN=1 \
bash scripts/run_paper_v3_systems.sh
```

Repeat without `DRY_RUN=1`. The JSON distinguishes executed packed-module
shape evidence from CUDA operator-profiler shape evidence. If the operator trace
does not expose every width, the run remains valid for latency/HBM, but the paper
must not claim kernel-profiler confirmation.

Only after target 6 shows a measurable benefit, run 4 and 8 in a new directory:

```bash
SYSTEMS_4_8="${NEXT_ROOT}/systems_target4_target8"
MANIFEST="${EXPANDED_MANIFEST}" TOKENIZER_AUDIT="${TOKENIZER_AUDIT}" \
RUN_DIR="${SYSTEMS_4_8}" ONLY_TARGETS="4,8" INCLUDE8=1 \
CASES="1x128,1x1024,2x128,2x1024" DECODE_TOKENS=32 \
WARMUPS=3 REPETITIONS=10 PROFILE_GEMM_SHAPES=1 \
bash scripts/run_paper_v3_systems.sh

python3 scripts/summarize_paper_v3_systems.py \
  --run-dir "${SYSTEMS_T6}" --run-dir "${SYSTEMS_4_8}" \
  --output-dir "${NEXT_ROOT}/systems_combined_tables"
```

## 7. HEAPr external baseline (after systems)

HEAPr is the closest official baseline because it supports Qwen3-30B-A3B and
atomic-expert structured pruning. Its atomic-expert pruning structure is not the
same as shared packed-channel pruning. Match and report measured MoE parameters,
whole-model parameters, serialized bytes, and eventually active FLOPs separately.

```bash
HEAPR_COMMIT="$(git ls-remote https://github.com/LLIKKE/HEAPr.git refs/heads/master | awk '{print $1}')"
HEAPR_WORK="/workspace/external/HEAPr_${HEAPR_COMMIT:0:12}"
HEAPR_RUN="${NEXT_ROOT}/heapr_qwen3_ratio006207"

HEAPR_COMMIT="${HEAPR_COMMIT}" WORK_DIR="${HEAPR_WORK}" \
RUN_DIR="${HEAPR_RUN}" RATIO=0.06207 CALI_NSAMPLES=128 BATCH_SIZE=8 \
TRUST_DATASET_CODE=1 EXPORT_CHECKPOINT=1 \
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" DRY_RUN=1 \
bash scripts/run_heapr_matched_baseline.sh
```

Before the real run, provision a dedicated HEAPr environment and enough storage.
The following real command provisions the environment once and runs the pinned
method:

```bash
mkdir -p /workspace/external

HEAPR_COMMIT="${HEAPR_COMMIT}" WORK_DIR="${HEAPR_WORK}" \
RUN_DIR="${HEAPR_RUN}" RATIO=0.06207 CALI_NSAMPLES=128 BATCH_SIZE=8 \
TRUST_DATASET_CODE=1 EXPORT_CHECKPOINT=1 \
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" \
HEAPR_VENV="/workspace/venvs/heapr_paper_v3_${HEAPR_COMMIT:0:12}" \
INSTALL_HEAPR_DEPS=1 LM_EVAL_INSTALL_SPEC="lm_eval==0.4.12" \
bash scripts/run_heapr_matched_baseline.sh
```

Then compare it with the already completed batch-8 baseline:

```bash
python3 scripts/compare_heapr_downstream.py \
  --baseline-results "${PRIMARY_DOWNSTREAM}/baseline_unpruned/lm_eval_results.json" \
  --heapr-results "${HEAPR_RUN}/lm_eval_results.json" \
  --heapr-protocol "${HEAPR_RUN}/heapr_protocol.json" \
  --output-dir "${HEAPR_RUN}/paired_tables" \
  --bootstrap-resamples 10000
```

The run records the pinned commit, two-forward/one-backward construction
protocol, wall time, peak HBM, measured parameter denominators, serialized bytes,
and the seven-task sample logs. Treat the first requested ratio as a budget probe;
do not call it matched unless its measured MoE and whole-model reductions are
reported alongside the primary target-6 checkpoint.
