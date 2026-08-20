# Paper Version 3 post-milestone runbook

All commands start in `~/workspace/MLP-neuron-pruning`. Every output directory
must be new. The scripts refuse to overwrite frozen results or verified
checkpoints.

## 0. Environment and CPU/dry-run checks

```bash
cd ~/workspace/MLP-neuron-pruning
source /workspace/venvs/qwen-pruning/bin/activate

python3 -m unittest \
  tests.test_paper_v3_paired_table \
  tests.test_paper_v3_checkpoint_manifest \
  tests.test_paper_v3_post_milestone \
  tests.test_heterogeneous_moe_checkpoint \
  tests.test_rmsnorm_ellipsoid_bound \
  tests.test_moe_plan_replay \
  tests.test_allocation_ranking_summary
```

## 1. Paired dNLL paper table

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
POST_ROOT="results/paper_v3_post_milestone/${RUN_ID}"
mkdir -p "${POST_ROOT}"

python3 scripts/build_paper_v3_paired_table.py \
  --input results/paper_v3_summary/qwen3_30b_milestone_20260820_052129/paper_v3_paired_comparisons.csv \
  --output-dir "${POST_ROOT}/paired_dnll"
```

The table contains 26 rows: 13 preregistered comparisons times WikiText-2 and
C4. Its dNLL convention is always `first_method - second_method`; significance
is true only when the paired bootstrap interval excludes zero.

## 2. Physical checkpoint freeze

Check free disk first. Four BF16 30B-class checkpoints require roughly 240 GB;
the two optional target-6 comparator checkpoints raise this to roughly 360 GB.
Exact sizes are measured, not estimated, in the output table.

```bash
df -hT . /workspace /tmp

CHECKPOINT_ROOT="/workspace/paper_v3_checkpoints/${RUN_ID}"
CHECKPOINT_MANIFEST="${POST_ROOT}/checkpoint_specs.json"

python3 scripts/generate_paper_v3_checkpoint_manifest.py \
  --summary-csv results/paper_v3_summary/qwen3_30b_milestone_20260820_052129/paper_v3_milestone_summary.csv \
  --output "${CHECKPOINT_MANIFEST}" \
  --checkpoint-root "${CHECKPOINT_ROOT}"

DRY_RUN=1 MANIFEST="${CHECKPOINT_MANIFEST}" \
  bash scripts/run_paper_v3_checkpoint_export.sh

MANIFEST="${CHECKPOINT_MANIFEST}" \
CHECKPOINT_TABLE_DIR="${POST_ROOT}/checkpoint_tables" \
  bash scripts/run_paper_v3_checkpoint_export.sh
```

The exporter saves the complete plan and selected IDs, calls the trusted
physical pruning functions, verifies every heterogeneous tensor width, reloads
the checkpoint, requires exact fixed-prompt logits, counts parameters, checks
that no original-width padding exists, hashes every payload file, and records
exact byte sizes.

Optional target-6 RMSNorm- and activation-ranking comparators can be appended
without rewriting verified checkpoints:

```bash
COMPARATOR_MANIFEST="${POST_ROOT}/checkpoint_specs_with_target6_comparators.json"
python3 scripts/generate_paper_v3_checkpoint_manifest.py \
  --summary-csv results/paper_v3_summary/qwen3_30b_milestone_20260820_052129/paper_v3_milestone_summary.csv \
  --output "${COMPARATOR_MANIFEST}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --include-target6-comparators

SKIP_VERIFIED_EXISTING=1 MANIFEST="${COMPARATOR_MANIFEST}" \
CHECKPOINT_TABLE_DIR="${POST_ROOT}/checkpoint_tables_with_comparators" \
  bash scripts/run_paper_v3_checkpoint_export.sh
```

## 3. Downstream evaluation

Pin one LM Evaluation Harness release in both the primary and HEAPr
environments. This runbook uses the official `v0.4.12` release.

```bash
python3 -m pip install --upgrade \
  "git+https://github.com/EleutherAI/lm-evaluation-harness.git@v0.4.12"
LM_EVAL_IDENTITY=$(python3 -c \
  'import importlib.metadata; print(importlib.metadata.version("lm_eval"))')
echo "LM_EVAL_IDENTITY=${LM_EVAL_IDENTITY}"

DOWNSTREAM_MANIFEST="${CHECKPOINT_MANIFEST}"
# Use COMPARATOR_MANIFEST here instead if the optional target-6 checkpoints exist.

DRY_RUN=1 MANIFEST="${DOWNSTREAM_MANIFEST}" \
RUN_DIR="${POST_ROOT}/downstream" \
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" \
  bash scripts/run_paper_v3_downstream.sh

MANIFEST="${DOWNSTREAM_MANIFEST}" \
RUN_DIR="${POST_ROOT}/downstream" \
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" \
BATCH_SIZE=4 SEED=42 INCLUDE_OPTIONAL=1 \
  bash scripts/run_paper_v3_downstream.sh
```

Required tasks are HellaSwag, MathQA, OpenBookQA, PIQA, WinoGrande, ARC-Easy,
and ARC-Challenge. `INCLUDE_OPTIONAL=1` adds BoolQ and RTE. All are zero-shot,
without a chat template, and every result includes task versions and logged
per-example decisions for paired bootstrap intervals.

For a plumbing-only check before the full run, use a separate directory and
set `SMOKE_LIMIT=10`. Limited results are never paper results.

## 4. Systems measurements

Run baseline, 4%, and 6% first:

```bash
DRY_RUN=1 MANIFEST="${CHECKPOINT_MANIFEST}" \
RUN_DIR="${POST_ROOT}/systems_primary" \
  bash scripts/run_paper_v3_systems.sh

MANIFEST="${CHECKPOINT_MANIFEST}" \
RUN_DIR="${POST_ROOT}/systems_primary" \
CASES="1x128,1x512,1x2048,2x512,4x512" \
  bash scripts/run_paper_v3_systems.sh
```

After inspecting those results, run a new directory including 8%:

```bash
MANIFEST="${CHECKPOINT_MANIFEST}" \
RUN_DIR="${POST_ROOT}/systems_target8" \
CASES="1x128,1x512,1x2048,2x512,4x512" INCLUDE8=1 ONLY_TARGETS=8 \
  bash scripts/run_paper_v3_systems.sh
```

Prefill is a timed full-context forward pass. Decode is timed cached one-token
forward steps. Reports include warm-up/repetition counts, mean/median/standard
deviation, load and peak HBM, software/GPU configuration, exact checkpoint
bytes, and an assertion that reduced intermediate widths are executed.

## 5. Method-construction cost and HEAPr gate

Measure the weight-only method without pruning or PPL:

```bash
DRY_RUN=1 RUN_DIR="${POST_ROOT}/method_cost_ours" \
  bash scripts/run_paper_v3_method_cost.sh

RUN_DIR="${POST_ROOT}/method_cost_ours" \
SOURCE_RUN_DIR=results/moe_selector_baselines/20260818_222729 \
  bash scripts/run_paper_v3_method_cost.sh
```

HEAPr has an official Qwen3-30B-A3B implementation, but its expert-specific
atomic pruning is not the same structure as shared packed-channel width
removal. Pin the current official commit before running and compare measured
parameter/FLOP budgets, not raw requested percentages.

```bash
HEAPR_COMMIT=$(git ls-remote https://github.com/LLIKKE/HEAPr.git refs/heads/master | awk '{print $1}')
echo "Pinned HEAPr commit: ${HEAPR_COMMIT}"

DRY_RUN=1 HEAPR_COMMIT="${HEAPR_COMMIT}" \
WORK_DIR="/workspace/external/heapr_${RUN_ID}" \
RUN_DIR="${POST_ROOT}/heapr_target6" \
HEAPR_VENV="/workspace/venvs/heapr-paper-v3_${RUN_ID}" \
  bash scripts/run_heapr_matched_baseline.sh

HEAPR_COMMIT="${HEAPR_COMMIT}" \
WORK_DIR="/workspace/external/heapr_${RUN_ID}" \
RUN_DIR="${POST_ROOT}/heapr_target6" \
HEAPR_VENV="/workspace/venvs/heapr-paper-v3_${RUN_ID}" \
RATIO=0.06207 CALI_NSAMPLES=128 SEED=42 \
INSTALL_HEAPR_DEPS=1 \
LM_EVAL_INSTALL_SPEC="git+https://github.com/EleutherAI/lm-evaluation-harness.git@v0.4.12" \
LM_EVAL_IDENTITY="${LM_EVAL_IDENTITY}" \
  bash scripts/run_heapr_matched_baseline.sh
```

The runtime patch changes only reporting and fixed evaluation settings. It
fails if the pinned upstream source no longer matches, and records pre/post
parameter counts plus its own hash. Compare paired downstream samples after
both runs exist:

```bash
python3 scripts/compare_heapr_downstream.py \
  --baseline-results "${POST_ROOT}/downstream/baseline_unpruned/lm_eval_results.json" \
  --heapr-results "${POST_ROOT}/heapr_target6/lm_eval_results.json" \
  --heapr-protocol "${POST_ROOT}/heapr_target6/heapr_protocol.json" \
  --output-dir "${POST_ROOT}/heapr_target6/paired_tables"

python3 scripts/build_method_cost_table.py \
  --our-protocol "${POST_ROOT}/method_cost_ours/method_protocol.json" \
  --heapr-protocol "${POST_ROOT}/heapr_target6/heapr_protocol.json" \
  --output-dir "${POST_ROOT}/method_cost_table"
```

Do not promote HEAPr as a matched structural baseline unless its measured
parameter/FLOP budget and physical execution audit are comparable.

## 6. Limited target-6 aggregation study

Run only after required checkpoint, downstream, and primary systems work:

```bash
DRY_RUN=1 PROFILE=target6_aggregation_limited \
RUN_DIR="${POST_ROOT}/target6_aggregation_limited" \
BASELINE_RUN_DIR=results/moe_selector_baselines/20260818_203025 \
TARGET2_RMSNORM_RUN_DIR=results/moe_selector_baselines/20260818_222729 \
  bash scripts/run_moe_allocation_ranking_matrix.sh

PROFILE=target6_aggregation_limited \
RUN_DIR="${POST_ROOT}/target6_aggregation_limited" \
N_EVAL=1024 EVAL_DATASETS=wikitext2,c4 \
BASELINE_RUN_DIR=results/moe_selector_baselines/20260818_203025 \
TARGET2_RMSNORM_RUN_DIR=results/moe_selector_baselines/20260818_222729 \
  bash scripts/run_moe_allocation_ranking_matrix.sh
```

All five cells use the frozen RMSNorm allocation and exactly 2288 removed
layer-channels. The generated frontier table reports p90, p95, p97.5, p99,
and max quality, paired dNLL versus p95, channel-ID changes, Jaccard overlap,
and sampled expert-level bound tightness/violations.

## 7. Machine-readable release manifest

Add only directories that actually completed:

```bash
python3 scripts/build_paper_v3_release_manifest.py \
  --artifact "${POST_ROOT}/paired_dnll" \
  --artifact "${POST_ROOT}/checkpoint_tables" \
  --artifact "${POST_ROOT}/downstream/paper_tables" \
  --artifact "${POST_ROOT}/systems_primary/paper_tables" \
  --artifact "${POST_ROOT}/target6_aggregation_limited/aggregation_frontier_tables" \
  --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
  --output-dir "${POST_ROOT}/release_manifest" \
  --release-label qwen3_30b_paper_v3_post_milestone
```

The release manifest contains absolute paths, sizes, SHA-256 hashes, checkpoint
verification payloads, and the frozen primary-method definition.
