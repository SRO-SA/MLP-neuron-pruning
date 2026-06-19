#!/usr/bin/env python3
"""
check_config_matrix.py — verify the full benchmark config matrix is complete.

Expected matrix:
  5 targets  × 2 methods × 2 datasets × 2 eval sizes = 40 configs

Usage:
    python scripts/check_config_matrix.py
    python scripts/check_config_matrix.py --fix   # print missing names only
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS   = os.path.join(REPO_ROOT, "configs")

TARGETS  = [1, 2, 4, 6, 8]
METHODS  = ["pure_delete", "residual_full_moe"]
DATASETS = ["wikitext2", "c4"]
N_EVALS  = [64, 512]

EXPECTED_TOTAL = len(TARGETS) * len(METHODS) * len(DATASETS) * len(N_EVALS)


def expected_configs():
    for t in TARGETS:
        for m in METHODS:
            for d in DATASETS:
                for n in N_EVALS:
                    yield f"moe_full48_packed_p95_{t}pct_{m}_{d}_n{n}.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="Print missing filenames only (for scripting)")
    args = parser.parse_args()

    existing = set(os.listdir(CONFIGS))
    expected = list(expected_configs())
    missing  = [f for f in expected if f not in existing]
    present  = [f for f in expected if f in existing]

    if args.fix:
        for f in missing:
            print(f)
        sys.exit(0 if not missing else 1)

    # ── Full report ────────────────────────────────────────────────────────────
    print()
    print("── Benchmark config matrix check ──────────────────────────────────────")
    print(f"  Targets  : {TARGETS}")
    print(f"  Methods  : {METHODS}")
    print(f"  Datasets : {DATASETS}")
    print(f"  N_evals  : {N_EVALS}")
    print(f"  Expected : {EXPECTED_TOTAL} configs")
    print(f"  Found    : {len(present)}")
    print(f"  Missing  : {len(missing)}")
    print()

    if missing:
        print("  Missing files:")
        for f in sorted(missing):
            print(f"    ✗  {f}")
        print()
        print("  To create them, run:")
        print("    python scripts/generate_benchmark_configs.py")
        print()
        print("  CONFIG MATRIX INCOMPLETE")
        sys.exit(1)
    else:
        print("  ✓  All 40 benchmark configs present")

        # Quick content sanity check: each config should have its target_pct set
        bad = []
        for fname in expected:
            fpath = os.path.join(CONFIGS, fname)
            with open(fpath) as fh:
                content = fh.read()
            # Extract expected target from filename e.g. 6pct -> 6.0
            pct_str = fname.split("_")[4]          # e.g. "6pct"
            pct_val = pct_str.replace("pct", "")   # "6"
            if f"- {pct_val}.0" not in content:
                bad.append((fname, pct_val))

        if bad:
            print()
            print("  Target % mismatch in these files:")
            for fn, pv in bad:
                print(f"    ✗  {fn}  (expected target {pv}.0)")
            sys.exit(1)
        else:
            print("  ✓  All configs have correct target_pruning_percent")
        print()
        print("  CONFIG MATRIX COMPLETE")


if __name__ == "__main__":
    main()
