#!/usr/bin/env python3
"""No-CUDA/no-Torch dry-run tests for ellipsoid selector config generation."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "generate_moe_selector_baseline_configs.py"
SPEC = importlib.util.spec_from_file_location("selector_config_generator", GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


def main() -> None:
    cfg = GENERATOR._build_config(
        model="Qwen/Qwen3-30B-A3B",
        selector="rmsnorm_ellipsoid_bound",
        target_pct=2.0,
        dataset="wikitext2",
        n_eval=128,
        channel_alignment=16,
        seed=42,
        max_seq_len=512,
        batch_size=4,
    )
    expected = {
        "moe_selector": "rmsnorm_ellipsoid_bound",
        "moe_pruning_mode": "packed_same_channel",
        "moe_same_channel_aggregation": "p95",
        "moe_budget_mode": "global",
        "moe_channel_alignment": 16,
        "max_expert_frac": 0.2,
        "scaling_methods": ["pure_delete"],
        "reconstruction_eval_samples": 128,
    }
    for key, value in expected.items():
        assert cfg[key] == value, f"{key}: {cfg[key]!r} != {value!r}"
    assert cfg["moe_selector_needs_calib"] is False
    assert "rmsnorm_ellipsoid_bound" in GENERATOR._SUPPORTED_SELECTORS
    GENERATOR._validate_selector_names(
        ["rmsnorm_bound", "rmsnorm_ellipsoid_bound"]
    )
    try:
        GENERATOR._validate_selector_names(["not_a_selector"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown selector was not rejected")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ellipsoid.yaml"
        GENERATOR._write_yaml(str(out), cfg)
        text = out.read_text()
        assert "moe_selector: rmsnorm_ellipsoid_bound" in text
        assert "moe_same_channel_aggregation: p95" in text
        assert "moe_budget_mode: global" in text

    print("ellipsoid config dry-run: PASS")


if __name__ == "__main__":
    main()
