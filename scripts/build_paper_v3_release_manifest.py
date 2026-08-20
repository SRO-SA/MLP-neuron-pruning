#!/usr/bin/env python3
"""Index immutable Version 3 artifacts and their SHA-256 hashes."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_paths(paths: list[str]) -> list[dict]:
    rows, seen = [], set()
    for supplied in paths:
        absolute = os.path.realpath(supplied)
        if not os.path.exists(absolute):
            raise FileNotFoundError(supplied)
        files = [absolute] if os.path.isfile(absolute) else [
            os.path.join(root, name)
            for root, _, names in os.walk(absolute) for name in names
        ]
        for path in sorted(files):
            if path in seen:
                continue
            seen.add(path)
            rows.append({
                "path": path, "size_bytes": os.path.getsize(path),
                "sha256": sha256(path),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True,
                        help="File or directory; repeat for every frozen artifact group")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--release-label", default="paper_v3_post_milestone")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    artifacts = index_paths(args.artifact)
    checkpoints = []
    if args.checkpoint_manifest:
        with open(args.checkpoint_manifest, encoding="utf-8") as handle:
            specs = json.load(handle)
        for spec in specs:
            verification_path = os.path.join(
                spec["checkpoint_dir"], "checkpoint_verification.json"
            )
            with open(verification_path, encoding="utf-8") as handle:
                verification = json.load(handle)
            checkpoints.append({"spec": spec, "verification": verification,
                                "verification_sha256": sha256(verification_path)})
    manifest = {
        "schema_version": 1, "release_label": args.release_label,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "immutability_policy": (
            "Inputs were read only. This exporter refuses an existing output directory."
        ),
        "primary_method": {
            "allocation_source": "rmsnorm_bound",
            "ranking_source": "rmsnorm_ellipsoid_bound",
            "expert_aggregation": "p95",
            "physical_pruning": "hardware-aligned packed same-channel tensor repacking",
        },
        "artifact_files": artifacts, "checkpoints": checkpoints,
    }
    os.makedirs(args.output_dir)
    json_path = os.path.join(args.output_dir, "paper_v3_release_manifest.json")
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(args.output_dir, "artifact_index.csv"), "x", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
        writer.writeheader(); writer.writerows(artifacts)
    with open(os.path.join(args.output_dir, "README.md"), "x", encoding="utf-8") as handle:
        handle.write(f"# {args.release_label}\n\n")
        handle.write(f"Indexed files: {len(artifacts)}  \n")
        handle.write(f"Verified checkpoints: {len(checkpoints)}  \n\n")
        handle.write("Primary method: global `rmsnorm_bound` allocation followed by "
                     "within-layer `rmsnorm_ellipsoid_bound` ranking with p95 expert aggregation.\n")
    print(f"[release-manifest] OK: {len(artifacts)} files; {json_path}")


if __name__ == "__main__":
    main()
