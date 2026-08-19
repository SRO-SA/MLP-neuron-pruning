#!/usr/bin/env python3
"""Validate one saved allocation/ranking experiment plan."""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.moe_plan_replay import validate_derived_allocation_ranking_plan


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-plan", required=True)
    parser.add_argument("--derived-plan", required=True)
    parser.add_argument("--allocation-source", required=True)
    parser.add_argument("--ranking-source", required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    allocation = _load(args.allocation_plan)
    derived = _load(args.derived_plan)
    validate_derived_allocation_ranking_plan(derived, allocation)
    audit = derived["allocation_ranking"]
    expected = {
        "allocation_source": args.allocation_source,
        "ranking_source": args.ranking_source,
        "experiment_name": args.experiment_name,
        "pruning_mode": "packed_same_channel",
        "channel_alignment": 16,
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(
                f"audit {field}={audit.get(field)!r}, expected {value!r}"
            )
    print(
        "[alloc-rank-validate] OK: "
        f"{args.experiment_name} layer_channels="
        f"{audit['total_selected_layer_channels']} expert_neurons="
        f"{audit['total_removed_expert_neurons']} actual_pct={audit['actual_pct']}"
    )


if __name__ == "__main__":
    main()
