#!/usr/bin/env python3
"""Collect all-expert ellipsoid and down-norm scores without calibration data."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_provenance import file_sha256
from src.moe_pruning import (
    discover_moe_architecture,
    get_expert_weights,
    get_moe_input_rmsnorm_weight,
)
from src.rmsnorm_geometry import (
    compute_rmsnorm_ellipsoid_and_down_norm_from_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    npz_path = output_dir / "all_expert_certificate_scores.npz"
    manifest_path = output_dir / "all_expert_certificate_scores.json"
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"refusing to overwrite score directory: {output_dir}")
    if args.dry_run:
        print(
            f"[certificate-scores] DRY RUN model={args.model} dtype={args.dtype} "
            f"device_map={args.device_map} output={output_dir} calibration=False"
        )
        return

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    infos, arch = discover_moe_architecture(model)
    arrays = {}
    layer_manifest = []
    for info in infos:
        if not info.is_moe:
            continue
        gamma = get_moe_input_rmsnorm_weight(info)
        ellipsoid_rows = []
        down_rows = []
        for expert_idx, expert in enumerate(info.expert_modules):
            weights = get_expert_weights(expert)
            ellipsoid, down_norm = compute_rmsnorm_ellipsoid_and_down_norm_from_weights(
                weights["gate_proj"], weights["up_proj"], weights["down_proj"], gamma
            )
            ellipsoid_rows.append(ellipsoid.numpy())
            down_rows.append(down_norm.numpy())
            del weights, ellipsoid, down_norm
        ellipsoid_array = np.stack(ellipsoid_rows).astype(np.float32, copy=False)
        down_array = np.stack(down_rows).astype(np.float32, copy=False)
        layer_idx = int(info.layer_idx)
        arrays[f"layer_{layer_idx}__ellipsoid"] = ellipsoid_array
        arrays[f"layer_{layer_idx}__down_norm"] = down_array
        layer_manifest.append({
            "layer_idx": layer_idx,
            "packed_experts": bool(info.experts_packed),
            "num_experts": int(ellipsoid_array.shape[0]),
            "intermediate_width": int(ellipsoid_array.shape[1]),
            "gamma_shape": list(gamma.shape),
        })
        print(
            f"[certificate-scores] layer={layer_idx} "
            f"experts={ellipsoid_array.shape[0]} width={ellipsoid_array.shape[1]}"
        )

    output_dir.mkdir(parents=True)
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": str(getattr(model.config, "_commit_hash", "") or ""),
        "model_class": type(model).__name__,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "calibration_required": False,
        "gradients_required": False,
        "weight_updates": False,
        "score_definition": {
            "ellipsoid": (
                "d_model/2*(||gamma*g||*||gamma*u||+"
                "abs((gamma*g) dot (gamma*u)))*||d||"
            ),
            "down_norm": "||down_column||_2",
        },
        "accumulation_dtype": "float32",
        "stored_dtype": "float32",
        "memory_policy": "one expert FP32 weight triplet at a time",
        "architecture": arch,
        "layers": layer_manifest,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "score_npz": str(npz_path),
        "score_npz_sha256": file_sha256(str(npz_path)),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"[certificate-scores] OK layers={len(layer_manifest)} "
        f"NPZ={npz_path} JSON={manifest_path}"
    )


if __name__ == "__main__":
    main()
