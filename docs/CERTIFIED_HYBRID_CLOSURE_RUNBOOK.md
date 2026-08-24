# Certified-hybrid final closure

This is the only authorized selector closure after milestone
`20260823_015217`. It reads frozen artifacts and writes new subdirectories. It
does not alter the pruning pipeline, weights, allocation, or earlier results.

## 1. Update and run CPU tests

```bash
cd ~/workspace/MLP-neuron-pruning
git pull --rebase
source /workspace/venvs/qwen-pruning/bin/activate

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_moe_set_certification \
  tests.test_moe_certified_hybrid_milestone \
  tests.test_paper_v3_post_milestone \
  tests.test_paper_v3_statistical_audit

MILESTONE_ROOT="results/moe_certified_hybrid_milestone/20260823_015217"
FINE_FRONTIER="$MILESTONE_ROOT/certification_frontier_fine_v1"
FINE_PPL="$MILESTONE_ROOT/hybrid_ppl_fine_v2"
FINE_CHECKPOINT_MANIFEST="$MILESTONE_ROOT/hybrid_checkpoint_manifest_fine_v1.json"
FINE_CHECKPOINT_ROOT="/paper_v3_checkpoints/20260823_015217_certified_hybrid_fine_v1"
FINE_TOKENIZER_AUDIT_DIR="$MILESTONE_ROOT/hybrid_tokenizer_audit_fine_v1"
FINE_RAW="$MILESTONE_ROOT/hybrid_downstream_fine_v1/raw"
FINE_DOWNSTREAM_TABLES="$MILESTONE_ROOT/hybrid_downstream_fine_v1/paper_tables"
DOWNSTREAM_CURVE_V2="$MILESTONE_ROOT/pure_downnorm_curve/downstream_paper_tables_v2"
DOWNNORM_TABLES_V2="$MILESTONE_ROOT/pure_downnorm_curve/paper_tables_v2"
SYSTEMS_TABLES_V2="$MILESTONE_ROOT/systems_v2/paper_tables_with_uncertainty"
FINE_CHECKPOINT_TABLES="$MILESTONE_ROOT/hybrid_checkpoint_tables_fine_v1"
FINE_TOKENIZER_AUDIT="$FINE_TOKENIZER_AUDIT_DIR/tokenizer_audit.json"
FINAL_PACKET="$MILESTONE_ROOT/final_paper_packet_v2"

EXISTING_HYBRID_MANIFEST="$MILESTONE_ROOT/hybrid_checkpoint_manifest.json"
PRIMARY_DOWNSTREAM="results/paper_v3_post_freeze/20260820_postfreeze_v1/downstream_primary_batch8_v1"
COMPARATOR_DOWNSTREAM="results/paper_v3_next_phase/20260821_221413/downstream_target2_target6_selector_attribution_batch8_v2/target6_comparators"
ORIGINAL_TOKENIZER_AUDIT="results/paper_v3_post_freeze/20260820_postfreeze_v1/tokenizer_audit_v3/tokenizer_audit.json"
EXPANDED_TOKENIZER_AUDIT="results/paper_v3_next_phase/20260821_221413/tokenizer_audit_expanded8_v1/tokenizer_audit.json"
DOWN_CURVE_MANIFEST="$MILESTONE_ROOT/pure_downnorm_curve/checkpoint_manifest.json"
DOWN_CURVE_AUDIT="$MILESTONE_ROOT/pure_downnorm_curve/tokenizer_audit/tokenizer_audit.json"
DOWN_CURVE_RAW="$MILESTONE_ROOT/pure_downnorm_curve/downstream_raw"
LM_EVAL_IDENTITY="0.4.12"
```

Every output variable above must name a nonexistent path before its producing
step. Never remove or replace an earlier milestone directory.

## 2. Generate the fine frontier without model evaluation

Resolve the frozen endpoints from the original frontier:

```bash
readarray -t SOURCE_PLANS < <(python3 - "$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["source_plans"]["ellipsoid"]["path"])
print(payload["source_plans"]["down_norm"]["path"])
PY
)
ELLIPSOID_PLAN="${SOURCE_PLANS[0]}"
DOWNNORM_PLAN="${SOURCE_PLANS[1]}"

python3 scripts/build_moe_certified_hybrid_frontier.py \
  --ellipsoid-plan "$ELLIPSOID_PLAN" \
  --down-norm-plan "$DOWNNORM_PLAN" \
  --score-bundle "$MILESTONE_ROOT/scores/all_expert_certificate_scores.npz" \
  --score-manifest "$MILESTONE_ROOT/scores/all_expert_certificate_scores.json" \
  --matched-validation "$MILESTONE_ROOT/matched_plan_validation.json" \
  --output-dir "$FINE_FRONTIER" --seed 42 --dry-run

python3 scripts/build_moe_certified_hybrid_frontier.py \
  --ellipsoid-plan "$ELLIPSOID_PLAN" \
  --down-norm-plan "$DOWNNORM_PLAN" \
  --score-bundle "$MILESTONE_ROOT/scores/all_expert_certificate_scores.npz" \
  --score-manifest "$MILESTONE_ROOT/scores/all_expert_certificate_scores.json" \
  --matched-validation "$MILESTONE_ROOT/matched_plan_validation.json" \
  --output-dir "$FINE_FRONTIER" --seed 42
```

The manifest contains the predefined slacks `0, 0.0025, 0.005, 0.01, 0.015,
0.02, 0.021436`, selection hashes, strict certificates, objectives, overlap,
Pareto flags, duplicate groups, and one PPL representative per distinct Pareto
selection.

## 3. Evaluate only distinct certificate/objective-Pareto selections

```bash
PLAN_MANIFEST="$FINE_FRONTIER/hybrid_frontier.json" \
RUN_DIR="$FINE_PPL" N_EVAL=1024 DRY_RUN=1 \
  bash scripts/run_moe_certified_hybrid_ppl.sh

PLAN_MANIFEST="$FINE_FRONTIER/hybrid_frontier.json" \
RUN_DIR="$FINE_PPL" N_EVAL=1024 \
  bash scripts/run_moe_certified_hybrid_ppl.sh
```

Duplicate or dominated selections are skipped before any model is loaded.

## 4. Select and evaluate at most two intermediate checkpoints

Selection is based only on the certificate/objective frontier. A candidate
must be Pareto-optimal, close at least 10% of the ellipsoid-to-down-norm
objective gap, and retain a strict certificate below pure down-norm. Of the
eligible selections, the bounded gate chooses the strongest-certificate point
and the distinct best-down-norm-objective point. It never consults PPL or
downstream outcomes.

```bash
python3 scripts/generate_moe_hybrid_checkpoint_manifest.py \
  --frontier-manifest "$FINE_FRONTIER/hybrid_frontier.json" \
  --existing-checkpoint-manifest "$EXISTING_HYBRID_MANIFEST" \
  --new-checkpoint-root "$FINE_CHECKPOINT_ROOT" \
  --output "$FINE_CHECKPOINT_MANIFEST" \
  --max-intermediate-checkpoints 2 \
  --minimum-objective-gap-closure 0.10 --dry-run

python3 scripts/generate_moe_hybrid_checkpoint_manifest.py \
  --frontier-manifest "$FINE_FRONTIER/hybrid_frontier.json" \
  --existing-checkpoint-manifest "$EXISTING_HYBRID_MANIFEST" \
  --new-checkpoint-root "$FINE_CHECKPOINT_ROOT" \
  --output "$FINE_CHECKPOINT_MANIFEST" \
  --max-intermediate-checkpoints 2 \
  --minimum-objective-gap-closure 0.10
```

Count new intermediates:

```bash
NEW_INTERMEDIATES="$(python3 - "$FINE_CHECKPOINT_MANIFEST" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(row["label"].startswith("certified_hybrid__") for row in rows))
PY
)"
echo "new intermediate checkpoints: $NEW_INTERMEDIATES"
test "$NEW_INTERMEDIATES" -le 2
```

If this is nonzero, check storage, export, audit, and evaluate:

```bash
if [ "$NEW_INTERMEDIATES" -gt 0 ]; then
  df -hT /paper_v3_checkpoints
  MANIFEST="$FINE_CHECKPOINT_MANIFEST" \
  CHECKPOINT_TABLE_DIR="$MILESTONE_ROOT/hybrid_checkpoint_tables_fine_v1" \
  SKIP_VERIFIED_EXISTING=1 DRY_RUN=1 \
    bash scripts/run_paper_v3_checkpoint_export.sh

  MANIFEST="$FINE_CHECKPOINT_MANIFEST" \
  CHECKPOINT_TABLE_DIR="$MILESTONE_ROOT/hybrid_checkpoint_tables_fine_v1" \
  SKIP_VERIFIED_EXISTING=1 \
    bash scripts/run_paper_v3_checkpoint_export.sh

  MANIFEST="$FINE_CHECKPOINT_MANIFEST" OUTPUT_DIR="$FINE_TOKENIZER_AUDIT_DIR" \
  SAMPLES_PER_DATASET=1024 bash scripts/run_paper_v3_tokenizer_audit.sh

  MANIFEST="$FINE_CHECKPOINT_MANIFEST" \
  TOKENIZER_AUDIT="$FINE_TOKENIZER_AUDIT_DIR/tokenizer_audit.json" \
  RUN_DIR="$FINE_RAW" LM_EVAL_IDENTITY="$LM_EVAL_IDENTITY" \
  RUN_KIND=additional_only SKIP_SUMMARY=1 BATCH_SIZE=8 DTYPE=bfloat16 \
  SEED=42 TRUST_DATASET_CODE=1 INCLUDE_OPTIONAL=0 \
    bash scripts/run_paper_v3_downstream.sh
fi
```

Do not delete frozen checkpoints to make room. Use additional storage if the
checkpoint-export preflight fails.

Build fine downstream tables. Add the new audit only when intermediates were
actually evaluated:

```bash
SUMMARY_AUDITS=(
  --tokenizer-audit "$ORIGINAL_TOKENIZER_AUDIT"
  --tokenizer-audit "$EXPANDED_TOKENIZER_AUDIT"
)
SUMMARY_RUNS=(
  --run-dir "$PRIMARY_DOWNSTREAM"
  --run-dir "$COMPARATOR_DOWNSTREAM"
)
if [ "$NEW_INTERMEDIATES" -gt 0 ]; then
  SUMMARY_AUDITS+=(--tokenizer-audit "$FINE_TOKENIZER_AUDIT_DIR/tokenizer_audit.json")
  SUMMARY_RUNS+=(--run-dir "$FINE_RAW")
fi

python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "$FINE_CHECKPOINT_MANIFEST" \
  "${SUMMARY_RUNS[@]}" "${SUMMARY_AUDITS[@]}" \
  --output-dir "$FINE_DOWNSTREAM_TABLES" \
  --bootstrap-resamples 10000 --bootstrap-seed 42
```

This is the final selector evaluation. No further slack values are added.

## 5. Audit compression-curve nesting and adjacent budgets

Rebuild the downstream summary to add the exploratory paired 4%-minus-2%,
6%-minus-4%, and 8%-minus-6% comparisons:

```bash
python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "$DOWN_CURVE_MANIFEST" \
  --run-dir "$PRIMARY_DOWNSTREAM" \
  --run-dir "$COMPARATOR_DOWNSTREAM" \
  --run-dir "$DOWN_CURVE_RAW" \
  --tokenizer-audit "$ORIGINAL_TOKENIZER_AUDIT" \
  --tokenizer-audit "$EXPANDED_TOKENIZER_AUDIT" \
  --tokenizer-audit "$DOWN_CURVE_AUDIT" \
  --output-dir "$DOWNSTREAM_CURVE_V2" \
  --bootstrap-resamples 10000 --bootstrap-seed 42

python3 scripts/summarize_pure_downnorm_curve.py \
  --ppl-summary "$MILESTONE_ROOT/pure_downnorm_curve/ppl/allocation_ranking_summary.csv" \
  --downstream-table "$DOWNSTREAM_CURVE_V2/downstream_benchmark_table.csv" \
  --checkpoint-manifest "$DOWN_CURVE_MANIFEST" \
  --paired-comparisons "$DOWNSTREAM_CURVE_V2/downstream_paired_comparisons.csv" \
  --budget-audit "${DOWN_CURVE_MANIFEST%.json}_budget_audit.json" \
  --output-dir "$DOWNNORM_TABLES_V2"
```

The nesting report never infers set inclusion from budget order. Adjacent
budget comparisons are task-stratified, paired by example identity, and marked
exploratory.

## 6. Add systems uncertainty without rerunning inference

The frozen systems JSON contains all raw repetitions. Re-summarize it:

```bash
python3 scripts/summarize_paper_v3_systems.py \
  --run-dir "$MILESTONE_ROOT/systems_v2" \
  --output-dir "$SYSTEMS_TABLES_V2" \
  --bootstrap-resamples 10000 --bootstrap-seed 42
```

The output reports exact loaded/peak HBM, mean and standard deviation, median
bootstrap intervals, and independently bootstrapped baseline-versus-pruned
relative intervals. The report records that the original runs were sequential,
not interleaved.

## 7. Build the immutable final paper packet

```bash
python3 scripts/build_moe_certified_hybrid_final_packet.py \
  --frontier-dir "$FINE_FRONTIER" \
  --ppl-dir "$FINE_PPL/paper_tables" \
  --downnorm-curve-dir "$DOWNNORM_TABLES_V2" \
  --downstream-dir "$FINE_DOWNSTREAM_TABLES" \
  --checkpoint-dir "$MILESTONE_ROOT/pure_downnorm_curve/checkpoint_tables" \
  --hybrid-checkpoint-dir "$FINE_CHECKPOINT_TABLES" \
  --hybrid-tokenizer-audit "$FINE_TOKENIZER_AUDIT" \
  --systems-dir "$SYSTEMS_TABLES_V2" \
  --matched-validation "$MILESTONE_ROOT/matched_plan_validation.json" \
  --output-dir "$FINAL_PACKET" --dry-run

python3 scripts/build_moe_certified_hybrid_final_packet.py \
  --frontier-dir "$FINE_FRONTIER" \
  --ppl-dir "$FINE_PPL/paper_tables" \
  --downnorm-curve-dir "$DOWNNORM_TABLES_V2" \
  --downstream-dir "$FINE_DOWNSTREAM_TABLES" \
  --checkpoint-dir "$MILESTONE_ROOT/pure_downnorm_curve/checkpoint_tables" \
  --hybrid-checkpoint-dir "$FINE_CHECKPOINT_TABLES" \
  --hybrid-tokenizer-audit "$FINE_TOKENIZER_AUDIT" \
  --systems-dir "$SYSTEMS_TABLES_V2" \
  --matched-validation "$MILESTONE_ROOT/matched_plan_validation.json" \
  --output-dir "$FINAL_PACKET"
```

`FINAL_PACKET_MANIFEST.json` hashes every copied or generated artifact and
declares that experimentation has stopped. The packet designates RMSNorm-bound
global allocation plus certificate-constrained down-norm refinement with 0.25%
ellipsoid-certificate slack as the final proposed method. Pure ellipsoid and
pure down-norm remain the endpoint methods. Continue directly with paper
writing.
