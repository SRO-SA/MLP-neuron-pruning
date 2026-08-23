# Certified hybrid milestone: exact server sequence

This is the final bounded experiment before paper writing. Every output path is
new. Frozen plans, checkpoints, evaluations, and reports are read-only inputs.

## 0. Update, test, and define one immutable milestone root

Run from the repository root in the existing GPU environment:

```bash
cd ~/workspace/MLP-neuron-pruning
git pull --rebase
source /workspace/venvs/qwen-pruning/bin/activate

python3 -m unittest \
  tests.test_moe_set_certification \
  tests.test_moe_certified_hybrid_milestone \
  tests.test_rmsnorm_ellipsoid_bound \
  tests.test_paper_v3_statistical_audit \
  tests.test_paper_v3_checkpoint_manifest

MILESTONE_ID="$(date +%Y%m%d_%H%M%S)"
MILESTONE_ROOT="results/moe_certified_hybrid_milestone/${MILESTONE_ID}"
EXISTING_CHECKPOINT_MANIFEST="results/paper_v3_next_phase/20260821_221413/checkpoint_specs_target2_target6_attribution.json"
PRIMARY_DOWNSTREAM="results/paper_v3_post_freeze/20260820_postfreeze_v1/downstream_primary_batch8_v1"
COMPARATOR_DOWNSTREAM="results/paper_v3_next_phase/20260821_221413/downstream_target2_target6_selector_attribution_batch8_v2/target6_comparators"
ORIGINAL_TOKENIZER_AUDIT="results/paper_v3_post_freeze/20260820_postfreeze_v1/tokenizer_audit_v3/tokenizer_audit.json"
EXPANDED_TOKENIZER_AUDIT="results/paper_v3_next_phase/20260821_221413/tokenizer_audit_expanded8_v1/tokenizer_audit.json"
LM_EVAL_IDENTITY="0.4.12"

test ! -e "$MILESTONE_ROOT"
```

Keep these variables in the same shell. If reconnecting, restore the exact
`MILESTONE_ID` instead of creating a new one.

## 1. Matched-plan gate, all-expert scores, and hybrid plans

First run the no-write preflight:

```bash
MILESTONE_ROOT="$MILESTONE_ROOT" DRY_RUN=1 \
  bash scripts/prepare_moe_certified_hybrid_milestone.sh
```

Then run the real preparation. It stops before score collection if the four
plans are not matched at exactly 2,288 layer-channels:

```bash
MILESTONE_ROOT="$MILESTONE_ROOT" \
  bash scripts/prepare_moe_certified_hybrid_milestone.sh
```

Inspect the compact frontier:

```bash
python3 - "$MILESTONE_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
validation = json.loads((root / "matched_plan_validation.json").read_text())
frontier = json.loads((root / "certification_frontier/hybrid_frontier.json").read_text())
assert validation["strict_gate_passed"] is True
assert validation["total_removed_layer_channels"] == 2288
assert validation["total_removed_expert_neurons"] == 292864
print("matched plan validation: PASS")
for row in frontier["plans"]:
    print(row["plan"], "slack=", row["certificate_slack"],
          "certificate=", row["strict_certificate"],
          "D=", row["normalized_down_norm_objective"],
          "pareto=", row["certificate_objective_pareto_optimal"],
          "selection=", row["selection_sha256"][:12])
print("identical selections:", frontier["identical_selection_groups"])
PY
```

Adjacent slack points are allowed to be identical; the script records this and
does not invent replacement thresholds.

## 2. Five-plan WikiText2/C4 PPL gate

```bash
PLAN_MANIFEST="$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" \
RUN_DIR="$MILESTONE_ROOT/hybrid_ppl" \
N_EVAL=1024 DRY_RUN=1 \
  bash scripts/run_moe_certified_hybrid_ppl.sh

PLAN_MANIFEST="$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" \
RUN_DIR="$MILESTONE_ROOT/hybrid_ppl" \
N_EVAL=1024 \
  bash scripts/run_moe_certified_hybrid_ppl.sh
```

Expected: five fresh model processes, ten dataset rows, 2,288 channels and
292,864 expert neurons for every row. The table is:

```bash
column -s, -t < "$MILESTONE_ROOT/hybrid_ppl/paper_tables/hybrid_ppl_pareto.csv" | less -S
```

## 3. Export only distinct certificate/objective-Pareto hybrid checkpoints

The two endpoint checkpoints are reused only after selected-channel identities
are verified against their frozen plans. New constrained checkpoints go to a
new external root.

```bash
HYBRID_CHECKPOINT_ROOT="/paper_v3_checkpoints/${MILESTONE_ID}_certified_hybrid"
HYBRID_CHECKPOINT_MANIFEST="$MILESTONE_ROOT/hybrid_checkpoint_manifest.json"

python3 scripts/generate_moe_hybrid_checkpoint_manifest.py \
  --frontier-manifest "$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" \
  --existing-checkpoint-manifest "$EXISTING_CHECKPOINT_MANIFEST" \
  --new-checkpoint-root "$HYBRID_CHECKPOINT_ROOT" \
  --output "$HYBRID_CHECKPOINT_MANIFEST" \
  --dry-run

python3 scripts/generate_moe_hybrid_checkpoint_manifest.py \
  --frontier-manifest "$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" \
  --existing-checkpoint-manifest "$EXISTING_CHECKPOINT_MANIFEST" \
  --new-checkpoint-root "$HYBRID_CHECKPOINT_ROOT" \
  --output "$HYBRID_CHECKPOINT_MANIFEST"

df -hT /paper_v3_checkpoints
MANIFEST="$HYBRID_CHECKPOINT_MANIFEST" \
CHECKPOINT_TABLE_DIR="$MILESTONE_ROOT/hybrid_checkpoint_tables" \
SKIP_VERIFIED_EXISTING=1 DRY_RUN=1 \
  bash scripts/run_paper_v3_checkpoint_export.sh
```

Read the preflight estimate. Free enough storage or use a larger checkpoint
root before the real export. Then:

```bash
MANIFEST="$HYBRID_CHECKPOINT_MANIFEST" \
CHECKPOINT_TABLE_DIR="$MILESTONE_ROOT/hybrid_checkpoint_tables" \
SKIP_VERIFIED_EXISTING=1 \
  bash scripts/run_paper_v3_checkpoint_export.sh
```

## 4. Audit tokenizers and evaluate the hybrid downstream

```bash
HYBRID_TOKENIZER_AUDIT_DIR="$MILESTONE_ROOT/hybrid_tokenizer_audit"

MANIFEST="$HYBRID_CHECKPOINT_MANIFEST" \
OUTPUT_DIR="$HYBRID_TOKENIZER_AUDIT_DIR" \
SAMPLES_PER_DATASET=1024 \
  bash scripts/run_paper_v3_tokenizer_audit.sh

HYBRID_TOKENIZER_AUDIT="$HYBRID_TOKENIZER_AUDIT_DIR/tokenizer_audit.json"
HYBRID_RAW="$MILESTONE_ROOT/hybrid_downstream/raw"

MANIFEST="$HYBRID_CHECKPOINT_MANIFEST" \
TOKENIZER_AUDIT="$HYBRID_TOKENIZER_AUDIT" \
RUN_DIR="$HYBRID_RAW" \
LM_EVAL_IDENTITY="$LM_EVAL_IDENTITY" \
RUN_KIND=additional_only SKIP_SUMMARY=1 \
BATCH_SIZE=8 DTYPE=bfloat16 SEED=42 \
TRUST_DATASET_CODE=1 INCLUDE_OPTIONAL=0 \
  bash scripts/run_paper_v3_downstream.sh
```

Build the paired table using existing endpoint evaluations plus new hybrids:

```bash
python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "$HYBRID_CHECKPOINT_MANIFEST" \
  --run-dir "$PRIMARY_DOWNSTREAM" \
  --run-dir "$COMPARATOR_DOWNSTREAM" \
  --run-dir "$HYBRID_RAW" \
  --tokenizer-audit "$ORIGINAL_TOKENIZER_AUDIT" \
  --tokenizer-audit "$EXPANDED_TOKENIZER_AUDIT" \
  --tokenizer-audit "$HYBRID_TOKENIZER_AUDIT" \
  --output-dir "$MILESTONE_ROOT/hybrid_downstream/paper_tables" \
  --bootstrap-resamples 10000 --bootstrap-seed 42
```

This produces task-stratified paired bootstrap intervals, sign-flip
randomization p-values, and Holm/BH adjustments for comparisons against the
baseline, pure ellipsoid, and pure down-norm endpoints.

## 5. Complete the pure down-norm 2/4/6/8 curve

First run matched PPL experiments under the existing RMSNorm allocations:

```bash
PROFILE=pure_downnorm_rmsnorm_allocation_curve \
RUN_DIR="$MILESTONE_ROOT/pure_downnorm_curve/ppl" \
N_EVAL=1024 EVAL_DATASETS=wikitext2,c4 \
BASELINE_RUN_DIR=results/moe_selector_baselines/20260818_203025 \
TARGET2_RMSNORM_RUN_DIR=results/moe_selector_baselines/20260818_222729 \
DRY_RUN=1 \
  bash scripts/run_moe_allocation_ranking_matrix.sh

PROFILE=pure_downnorm_rmsnorm_allocation_curve \
RUN_DIR="$MILESTONE_ROOT/pure_downnorm_curve/ppl" \
N_EVAL=1024 EVAL_DATASETS=wikitext2,c4 \
BASELINE_RUN_DIR=results/moe_selector_baselines/20260818_203025 \
TARGET2_RMSNORM_RUN_DIR=results/moe_selector_baselines/20260818_222729 \
  bash scripts/run_moe_allocation_ranking_matrix.sh
```

Create and export the missing 2%, 4%, and 8% checkpoints. Target 6 reuses the
already verified checkpoint:

```bash
DOWN_CURVE_CHECKPOINT_ROOT="/paper_v3_checkpoints/${MILESTONE_ID}_downnorm_curve"
DOWN_CURVE_MANIFEST="$MILESTONE_ROOT/pure_downnorm_curve/checkpoint_manifest.json"

python3 scripts/generate_pure_downnorm_curve_checkpoint_manifest.py \
  --curve-run-dir "$MILESTONE_ROOT/pure_downnorm_curve/ppl" \
  --existing-checkpoint-manifest "$EXISTING_CHECKPOINT_MANIFEST" \
  --new-checkpoint-root "$DOWN_CURVE_CHECKPOINT_ROOT" \
  --output "$DOWN_CURVE_MANIFEST" --dry-run

python3 scripts/generate_pure_downnorm_curve_checkpoint_manifest.py \
  --curve-run-dir "$MILESTONE_ROOT/pure_downnorm_curve/ppl" \
  --existing-checkpoint-manifest "$EXISTING_CHECKPOINT_MANIFEST" \
  --new-checkpoint-root "$DOWN_CURVE_CHECKPOINT_ROOT" \
  --output "$DOWN_CURVE_MANIFEST"

MANIFEST="$DOWN_CURVE_MANIFEST" \
CHECKPOINT_TABLE_DIR="$MILESTONE_ROOT/pure_downnorm_curve/checkpoint_tables" \
SKIP_VERIFIED_EXISTING=1 DRY_RUN=1 \
  bash scripts/run_paper_v3_checkpoint_export.sh

MANIFEST="$DOWN_CURVE_MANIFEST" \
CHECKPOINT_TABLE_DIR="$MILESTONE_ROOT/pure_downnorm_curve/checkpoint_tables" \
SKIP_VERIFIED_EXISTING=1 \
  bash scripts/run_paper_v3_checkpoint_export.sh
```

Audit and evaluate only the new 2%, 4%, and 8% checkpoints:

```bash
DOWN_CURVE_AUDIT_DIR="$MILESTONE_ROOT/pure_downnorm_curve/tokenizer_audit"
MANIFEST="$DOWN_CURVE_MANIFEST" OUTPUT_DIR="$DOWN_CURVE_AUDIT_DIR" \
SAMPLES_PER_DATASET=1024 bash scripts/run_paper_v3_tokenizer_audit.sh
DOWN_CURVE_AUDIT="$DOWN_CURVE_AUDIT_DIR/tokenizer_audit.json"
DOWN_CURVE_RAW="$MILESTONE_ROOT/pure_downnorm_curve/downstream_raw"

MANIFEST="$DOWN_CURVE_MANIFEST" TOKENIZER_AUDIT="$DOWN_CURVE_AUDIT" \
RUN_DIR="$DOWN_CURVE_RAW" LM_EVAL_IDENTITY="$LM_EVAL_IDENTITY" \
RUN_KIND=additional_only ONLY_TARGETS=2,4,8 SKIP_SUMMARY=1 \
BATCH_SIZE=8 DTYPE=bfloat16 SEED=42 TRUST_DATASET_CODE=1 INCLUDE_OPTIONAL=0 \
  bash scripts/run_paper_v3_downstream.sh

python3 scripts/summarize_paper_v3_downstream.py \
  --checkpoint-manifest "$DOWN_CURVE_MANIFEST" \
  --run-dir "$PRIMARY_DOWNSTREAM" \
  --run-dir "$COMPARATOR_DOWNSTREAM" \
  --run-dir "$DOWN_CURVE_RAW" \
  --tokenizer-audit "$ORIGINAL_TOKENIZER_AUDIT" \
  --tokenizer-audit "$EXPANDED_TOKENIZER_AUDIT" \
  --tokenizer-audit "$DOWN_CURVE_AUDIT" \
  --output-dir "$MILESTONE_ROOT/pure_downnorm_curve/downstream_paper_tables" \
  --bootstrap-resamples 10000 --bootstrap-seed 42

python3 scripts/summarize_pure_downnorm_curve.py \
  --ppl-summary "$MILESTONE_ROOT/pure_downnorm_curve/ppl/allocation_ranking_summary.csv" \
  --downstream-table "$MILESTONE_ROOT/pure_downnorm_curve/downstream_paper_tables/downstream_benchmark_table.csv" \
  --budget-audit "${DOWN_CURVE_MANIFEST%.json}_budget_audit.json" \
  --output-dir "$MILESTONE_ROOT/pure_downnorm_curve/paper_tables"
```

## 6. Apply the stop/go rule and benchmark systems behavior

```bash
DECISION="$MILESTONE_ROOT/hybrid_decision.json"
SYSTEMS_MANIFEST="$MILESTONE_ROOT/systems_checkpoint_manifest.json"

python3 scripts/select_moe_certified_hybrid_outcome.py \
  --frontier "$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" \
  --downstream-table "$MILESTONE_ROOT/hybrid_downstream/paper_tables/downstream_benchmark_table.csv" \
  --checkpoint-manifest "$HYBRID_CHECKPOINT_MANIFEST" \
  --output "$DECISION" --systems-manifest "$SYSTEMS_MANIFEST" --dry-run

python3 scripts/select_moe_certified_hybrid_outcome.py \
  --frontier "$MILESTONE_ROOT/certification_frontier/hybrid_frontier.json" \
  --downstream-table "$MILESTONE_ROOT/hybrid_downstream/paper_tables/downstream_benchmark_table.csv" \
  --checkpoint-manifest "$HYBRID_CHECKPOINT_MANIFEST" \
  --output "$DECISION" --systems-manifest "$SYSTEMS_MANIFEST"
```

The 1% certificate-improvement threshold is a predeclared operational meaning
of “meaningfully stronger” for stop/go criterion A. It can be changed only
before looking at the new downstream results, using the corresponding CLI
argument, and the chosen value is saved in the decision JSON.

Use the hybrid tokenizer audit, which covers baseline and every possible final
hybrid endpoint:

```bash
MANIFEST="$SYSTEMS_MANIFEST" TOKENIZER_AUDIT="$HYBRID_TOKENIZER_AUDIT" \
RUN_DIR="$MILESTONE_ROOT/systems" ONLY_TARGETS=0,6 \
CASES=1x128,1x512,1x2048,4x512,4x2048 \
DECODE_TOKENS=32 WARMUPS=3 REPETITIONS=10 PROFILE_GEMM_SHAPES=1 DRY_RUN=1 \
  bash scripts/run_paper_v3_systems.sh

MANIFEST="$SYSTEMS_MANIFEST" TOKENIZER_AUDIT="$HYBRID_TOKENIZER_AUDIT" \
RUN_DIR="$MILESTONE_ROOT/systems" ONLY_TARGETS=0,6 \
CASES=1x128,1x512,1x2048,4x512,4x2048 \
DECODE_TOKENS=32 WARMUPS=3 REPETITIONS=10 PROFILE_GEMM_SHAPES=1 \
  bash scripts/run_paper_v3_systems.sh
```

The operator traces are required evidence for reduced executed GEMM widths;
checkpoint byte/FLOP reductions alone are not a speed claim.

## 7. Freeze the conclusion and return to paper writing

```bash
python3 scripts/finalize_moe_certified_hybrid_milestone.py \
  --milestone-root "$MILESTONE_ROOT" \
  --decision "$DECISION" \
  --systems-dir "$MILESTONE_ROOT/systems" \
  --dry-run

python3 scripts/finalize_moe_certified_hybrid_milestone.py \
  --milestone-root "$MILESTONE_ROOT" \
  --decision "$DECISION" \
  --systems-dir "$MILESTONE_ROOT/systems"
```

After this succeeds, `milestone_conclusion.md` contains exactly one allowed
outcome and `MILESTONE_MANIFEST.json` hashes every in-directory artifact. Stop
experimentation and return to paper writing unless a concrete implementation
error is found.
