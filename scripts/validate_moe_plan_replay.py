#!/usr/bin/env python3
"""Validate a saved fixed-allocation replay plan against its source plan."""
from __future__ import annotations

import argparse
import json
import os
import sys

# When invoked as ``python3 scripts/validate_moe_plan_replay.py``, Python puts
# ``scripts/`` rather than the repository root on sys.path.  Add the root
# explicitly so the CLI behaves the same way as module-based test imports.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.moe_plan_replay import validate_derived_replay_plan


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--source-selector", required=True)
    parser.add_argument("--alternate-selector", required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    args = parser.parse_args()

    source = _load(args.source)
    derived = _load(args.derived)
    validate_derived_replay_plan(derived, source)
    replay = derived["fixed_allocation_replay"]
    expected_pairs = {
        "source_allocation_selector": args.source_selector,
        "alternate_channel_selector": args.alternate_selector,
        "pruning_mode": "packed_same_channel",
        "channel_alignment": 16,
        "total_selected_layer_channels": args.expected_total,
    }
    for field, expected in expected_pairs.items():
        actual = replay.get(field)
        if actual != expected:
            raise ValueError(
                f"derived replay field {field}={actual!r}, expected {expected!r}"
            )
    if derived.get("selector") != args.alternate_selector:
        raise ValueError("derived plan selector is not the alternate selector")
    if source.get("selector") != args.source_selector:
        raise ValueError("source plan selector does not match the expected selector")
    print(
        "[replay-validate] OK: fixed_allocation="
        f"{args.source_selector} channel_selector={args.alternate_selector} "
        f"layer_channels={args.expected_total} "
        f"expert_neurons={replay['total_removed_expert_neurons']} "
        f"actual_pct={replay['actual_pct']}"
    )


if __name__ == "__main__":
    main()
