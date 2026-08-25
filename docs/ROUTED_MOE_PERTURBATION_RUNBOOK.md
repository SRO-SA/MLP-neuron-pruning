# AXIOM routed-MoE local perturbation audit

This is the final target-6 validation. It never changes a pruning plan and it
does not propagate pruned hidden states between layers. The baseline
post-RMSNorm input, top-k identities, and normalized routing weights are
captured once and replayed against four physical checkpoints in separate
processes.

## 1. Update and run CPU tests

```bash
cd ~/workspace/MLP-neuron-pruning
git pull --rebase
source /workspace/venvs/qwen-pruning/bin/activate

PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  tests.test_routed_moe_perturbation \
  tests.test_moe_set_certification \
  tests.test_moe_certified_hybrid_milestone
```

## 2. Define and verify frozen inputs

```bash
MILESTONE_ROOT="results/moe_certified_hybrid_milestone/20260823_015217"
MANIFEST="$MILESTONE_ROOT/hybrid_checkpoint_manifest_fine_v1.json"
TOKENIZER_AUDIT="$MILESTONE_ROOT/hybrid_tokenizer_audit_fine_v1/tokenizer_audit.json"
SCORE_BUNDLE="$MILESTONE_ROOT/scores/all_expert_certificate_scores.npz"
SCORE_MANIFEST="$MILESTONE_ROOT/scores/all_expert_certificate_scores.json"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
ROUTED_RUN="$MILESTONE_ROOT/routed_moe_perturbation_${RUN_ID}"

for path in "$MANIFEST" "$TOKENIZER_AUDIT" "$SCORE_BUNDLE" "$SCORE_MANIFEST"; do
  test -f "$path" || { echo "ERROR missing: $path"; exit 1; }
done

python3 - "$MANIFEST" <<'PY'
import json, os, sys
required = {
    "baseline_unpruned",
    "rmsnorm_alloc__ellipsoid_rank__p95__target6",
    "certified_hybrid__downnorm_refinement_slack0p25__target6",
    "certified_hybrid__downnorm_refinement_slack2__target6",
    "rmsnorm_alloc__downnorm_rank__p95__target6",
}
rows = json.load(open(sys.argv[1], encoding="utf-8"))
by_label = {row["label"]: row for row in rows}
assert required <= set(by_label), required - set(by_label)
for label in sorted(required):
    path = by_label[label]["checkpoint_dir"]
    assert os.path.isdir(path), path
    print(f"OK {label}: {path}")
PY
```

The held-out sample is 32 C4 validation documents after eligible-document
offset 4,096, truncated to 512 tokens. The resolved C4 commit, source indices,
text hashes, token hashes, preprocessing, and model revision are frozen in the
capture manifest. Earlier operating-point evaluation used eligible indices
0--1,023, so the declared ranges are disjoint.

## 3. Dry run

```bash
MANIFEST="$MANIFEST" \
TOKENIZER_AUDIT="$TOKENIZER_AUDIT" \
SCORE_BUNDLE="$SCORE_BUNDLE" \
SCORE_MANIFEST="$SCORE_MANIFEST" \
RUN_DIR="$ROUTED_RUN" \
NUM_DOCUMENTS=32 SKIP_DOCUMENTS=4096 MAX_SEQ_LEN=512 \
DRY_RUN=1 \
bash scripts/run_routed_moe_perturbation.sh
```

## 4. Launch the GPU run safely across SSH disconnects

```bash
LAUNCH_LOG="${ROUTED_RUN}.launch.log"

nohup env \
  MANIFEST="$MANIFEST" \
  TOKENIZER_AUDIT="$TOKENIZER_AUDIT" \
  SCORE_BUNDLE="$SCORE_BUNDLE" \
  SCORE_MANIFEST="$SCORE_MANIFEST" \
  RUN_DIR="$ROUTED_RUN" \
  NUM_DOCUMENTS=32 SKIP_DOCUMENTS=4096 MAX_SEQ_LEN=512 \
  SEED=42 DTYPE=bfloat16 \
  bash scripts/run_routed_moe_perturbation.sh \
  >"$LAUNCH_LOG" 2>&1 < /dev/null &

echo $! > "${ROUTED_RUN}.pid"
echo "PID=$(cat "${ROUTED_RUN}.pid")"
echo "LOG=$LAUNCH_LOG"
```

Monitor without starting a second process:

```bash
tail -80 "$LAUNCH_LOG"
nvidia-smi
```

Completion is marked by `[routed-moe] COMPLETE`. A theorem violation retains
the result directory and terminates the wrapper before it can be mistaken for
a successful paper result.

## 5. Verify and print the paper table

```bash
python3 - "$ROUTED_RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
capture = json.load(open(root / "baseline_capture/capture_manifest.json", encoding="utf-8"))
assert capture["end_to_end_trace_used"] is False
assert capture["dataset"]["disjoint_from_operating_point_evaluation"] is True
assert capture["fixed_route_replay_validation"]["compared_against_native_unpruned_moe"] is True
labels = capture["evaluated_plan_labels"]
for label in labels:
    result = json.load(open(root / "plan_results" / label / "result.json", encoding="utf-8"))
    assert result["strict_bound_violations"] == 0
    assert result["route_conditioned_bound_violations"] == 0
    assert result["end_to_end_trace_used"] is False
    print(label, result["overall"]["strict_bound_ratio"]["max"],
          result["overall"]["route_conditioned_bound_ratio"]["max"])
print("[routed-moe] VERIFIED")
PY

cat "$ROUTED_RUN/paper_tables/routed_moe_perturbation_summary.md"
cat "$ROUTED_RUN/paper_tables/routed_moe_paired_document_bootstrap.csv"
```

## 6. Build the submission-artifact audit and anonymous code archive

```bash
SUBMISSION_AUDIT="$ROUTED_RUN/submission_artifact_audit"
DOWNSTREAM_BASELINE="results/paper_v3_post_freeze/20260820_postfreeze_v1/downstream_primary_batch8_v1/baseline_unpruned/lm_eval_results.json"

python3 scripts/build_axiom_submission_artifact_audit.py \
  --experiment-run-dir "$ROUTED_RUN" \
  --checkpoint-manifest "$MANIFEST" \
  --downstream-result "$DOWNSTREAM_BASELINE" \
  --output-dir "$SUBMISSION_AUDIT" \
  --dry-run

python3 scripts/build_axiom_submission_artifact_audit.py \
  --experiment-run-dir "$ROUTED_RUN" \
  --checkpoint-manifest "$MANIFEST" \
  --downstream-result "$DOWNSTREAM_BASELINE" \
  --output-dir "$SUBMISSION_AUDIT"
```

The archive is
`$SUBMISSION_AUDIT/axiom_anonymous_code.tar.gz`. Uploading it to an anonymous
artifact host is a separate external action; the audit records its SHA-256 and
does not claim that a public link already exists.

Do not start the optional target-8 extension until these target-6 tables have
been inspected. No new plan, selector, or propagation method is authorized by
this runbook.
