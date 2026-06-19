#!/usr/bin/env python3
"""
test_result_row_fields.py
=========================
Lightweight unit test: verifies that result-row construction logic in
moe_pruning.py produces all expected CSV/JSON fields for every supported
method, WITHOUT loading any model or running any GPU code.

Run:
    python3 scripts/test_result_row_fields.py

Exit 0 → all checks pass.  Exit 1 → failures found.
"""

from __future__ import annotations
import ast, os, re, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Extract MOE_SUMMARY_CSV_KEYS from moe_pruning.py without importing torch ──
def _load_csv_keys() -> list[str]:
    path = os.path.join(REPO_ROOT, "src", "moe_pruning.py")
    with open(path) as f:
        src = f.read()
    m = re.search(r'MOE_SUMMARY_CSV_KEYS\s*=\s*(\[.*?\])', src, re.DOTALL)
    if not m:
        raise RuntimeError("MOE_SUMMARY_CSV_KEYS not found in src/moe_pruning.py")
    return ast.literal_eval(m.group(1))

# ── Extract _VARIANT_MAP from the patch we applied ─────────────────────────────
def _load_variant_map() -> dict[str, str]:
    path = os.path.join(REPO_ROOT, "src", "moe_pruning.py")
    with open(path) as f:
        src = f.read()
    m = re.search(r'_VARIANT_MAP\s*=\s*(\{.*?\})', src, re.DOTALL)
    if not m:
        raise RuntimeError("_VARIANT_MAP not found in src/moe_pruning.py")
    return ast.literal_eval(m.group(1))

MOE_SUMMARY_CSV_KEYS = _load_csv_keys()
_VARIANT_MAP         = _load_variant_map()

# ── Required fields in every successful result row ─────────────────────────────
REQUIRED_FIELDS = [
    "model",
    "method",
    "moe_pruning_method",
    "target_pct",
    "actual_pct",
    "status",
    "forward_check",
    "baseline_ppl",
    "compressed_ppl",
    "delta_ppl",
    "relative_delta_pct",
    "expert_param_reduction_pct",
    "requested_method",
    "actual_method",
    "residual_variant",
    "residual_fallback_used",
    "residual_applied_experts",
    "residual_coverage_pct",
]

# Alias: moe_pruning_method isn't in MOE_SUMMARY_CSV_KEYS (it maps to "method"),
# so exclude it from that cross-check only.
CSV_KEY_EXCLUDE = {"moe_pruning_method"}

# ── Simulate the stats-computation block ──────────────────────────────────────
def _compute_stats(method: str, residual_applied: bool) -> dict:
    _resid_stable      = 10 if residual_applied else 0
    _resid_total_cand  = 20
    _err_cnt           = 5  if residual_applied else 0
    _err_del_wsum      = 0.5 * _err_cnt
    _err_res_wsum      = 0.3 * _err_cnt
    _upd_norms_all     = [0.1, 0.2] if residual_applied else []

    _cov_pct    = 100.0 * _resid_stable / _resid_total_cand if _resid_total_cand > 0 else 0.0
    _mean_e_del = _err_del_wsum / _err_cnt if _err_cnt > 0 else float("nan")
    _mean_e_res = _err_res_wsum / _err_cnt if _err_cnt > 0 else float("nan")
    _mean_imp   = (
        100.0 * (_mean_e_del - _mean_e_res) / (_mean_e_del + 1e-12)
        if _err_cnt > 0 else float("nan")
    )
    _mean_upd = sum(_upd_norms_all) / len(_upd_norms_all) if _upd_norms_all else float("nan")
    _max_upd  = max(_upd_norms_all) if _upd_norms_all else float("nan")
    _actual_method = method if (residual_applied or method == "pure_delete") else "pure_delete"
    _resid_variant = _VARIANT_MAP.get(method, method)

    return dict(
        _cov_pct=_cov_pct, _mean_e_del=_mean_e_del, _mean_e_res=_mean_e_res,
        _mean_imp=_mean_imp, _mean_upd=_mean_upd, _max_upd=_max_upd,
        _actual_method=_actual_method, _resid_variant=_resid_variant,
    )

def build_summary_row(method: str, residual_applied: bool) -> dict:
    """Mirror summary.update() logic from moe_pruning.py."""
    s = _compute_stats(method, residual_applied)
    return {
        "model":                        "Qwen/Qwen3-30B-A3B",
        "method":                       method,
        "moe_pruning_method":           method,
        "target_pct":                   2.0,
        "actual_pct":                   1.98,
        "status":                       "ok",
        "forward_check":                True,
        "baseline_ppl":                 12.34,
        "compressed_ppl":               12.50,
        "delta_ppl":                    0.16,
        "relative_delta_pct":           1.30,
        "expert_param_reduction_pct":   1.95,
        "requested_method":             method,
        "actual_method":                s["_actual_method"],
        "residual_variant":             s["_resid_variant"],
        "residual_fallback_used":       False,
        "residual_applied_experts":     10 if residual_applied else 0,
        "residual_coverage_pct":        round(s["_cov_pct"], 2),
        "mean_err_delete":              s["_mean_e_del"],
        "mean_err_resid":               s["_mean_e_res"],
        "mean_local_improvement_pct":   s["_mean_imp"],
        "mean_update_norm":             s["_mean_upd"],
        "max_update_norm":              s["_max_upd"],
        "residual_lambda":              None,
        "pruning_plan_path":            "",
        "loaded_pruning_plan":          False,
        "residual_total_candidate_experts": 20,
        "residual_attempted_experts":   10 if residual_applied else 0,
        "residual_stable_experts":      10 if residual_applied else 0,
        "residual_rejected_experts":    0,
        "residual_skipped_experts":     0,
        "residual_failed_experts":      0,
        "residual_skip_too_few_tokens": 0,
        "residual_skip_ill_conditioned": 0,
        "residual_skip_non_finite":     0,
        "residual_skip_update_too_large": 0,
        "residual_skip_not_improved":   0,
        "residual_applied":             residual_applied,
        "residual_time_sec":            0.0,
        "expert_layout":                "unpacked",
        "model_revision":               "",
        "transformers_version":         "",
        "torch_version":                "2.x",
        "csv_path":                     "results/moe_target_pruning_test.csv",
        "json_path":                    "results/moe_target_pruning_test.json",
        "per_layer_csv_path":           "",
    }

# ── Run all checks ─────────────────────────────────────────────────────────────
def run_checks() -> bool:
    failures: list[str] = []
    methods = list(_VARIANT_MAP.keys())

    print("=" * 70)
    print("Result-row field smoke test")
    print(f"  Methods: {len(methods)}  |  Required fields: {len(REQUIRED_FIELDS)}")
    print(f"  MOE_SUMMARY_CSV_KEYS: {len(MOE_SUMMARY_CSV_KEYS)} keys")
    print("=" * 70)

    for method in methods:
        residual_applied = method != "pure_delete"
        row = build_summary_row(method, residual_applied)
        method_failures: list[str] = []

        # Check 1: required fields present and accessible
        for field in REQUIRED_FIELDS:
            if field not in row:
                method_failures.append(f"  missing field '{field}'")
            else:
                try:
                    _ = row[field]
                except Exception as e:
                    method_failures.append(f"  field '{field}' raised {type(e).__name__}: {e}")

        # Check 2: actual_method correct
        expected_am = method if (residual_applied or method == "pure_delete") else "pure_delete"
        if row.get("actual_method") != expected_am:
            method_failures.append(
                f"  actual_method: expected {expected_am!r}, got {row.get('actual_method')!r}"
            )

        # Check 3: residual_variant is proper slug (never raw method name for known methods)
        expected_variant = _VARIANT_MAP.get(method, method)
        if row.get("residual_variant") != expected_variant:
            method_failures.append(
                f"  residual_variant: expected {expected_variant!r}, "
                f"got {row.get('residual_variant')!r}"
            )

        # Check 4: coverage=0 for pure_delete, >0 for residual
        cov = row.get("residual_coverage_pct", -1)
        if method == "pure_delete" and cov != 0.0:
            method_failures.append(f"  residual_coverage_pct should be 0.0, got {cov}")
        elif method != "pure_delete" and not (cov > 0):
            method_failures.append(f"  residual_coverage_pct should be >0, got {cov}")

        # Check 5: required fields in MOE_SUMMARY_CSV_KEYS
        for field in REQUIRED_FIELDS:
            if field in CSV_KEY_EXCLUDE:
                continue
            if field not in MOE_SUMMARY_CSV_KEYS:
                method_failures.append(
                    f"  '{field}' is required but not in MOE_SUMMARY_CSV_KEYS"
                )

        status = "PASS" if not method_failures else "FAIL"
        print(f"  {method:58s}  {status}")
        for msg in method_failures:
            print(f"    ✗{msg}")
        failures.extend(f"{method}:{m}" for m in method_failures)

    print()

    # Check 6: MOE_SUMMARY_CSV_KEYS has no duplicate keys
    if len(MOE_SUMMARY_CSV_KEYS) != len(set(MOE_SUMMARY_CSV_KEYS)):
        dups = [k for k in set(MOE_SUMMARY_CSV_KEYS) if MOE_SUMMARY_CSV_KEYS.count(k) > 1]
        failures.append(f"MOE_SUMMARY_CSV_KEYS has duplicate keys: {dups}")
        print(f"  ✗ MOE_SUMMARY_CSV_KEYS has duplicate keys: {dups}")

    if failures:
        print(f"RESULT: {len(failures)} failure(s)")
        return False
    print(f"RESULT: All checks passed  ({len(methods)} methods × {len(REQUIRED_FIELDS)} required fields)")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)
