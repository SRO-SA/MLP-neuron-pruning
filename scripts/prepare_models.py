#!/usr/bin/env python3
"""
prepare_models.py — download and cache models/datasets for qwen_swiglu_pruning.

Usage:
    python scripts/prepare_models.py \\
        --models Qwen/Qwen2.5-0.5B Qwen/Qwen3-30B-A3B \\
        --datasets wikitext2 c4 \\
        --cache-dir /workspace/hf_cache \\
        --skip-existing

Arguments:
    --models        One or more HuggingFace model IDs to download.
    --datasets      One or more dataset names: wikitext2, c4, wikitext103.
    --cache-dir     Local directory for HF cache (default: /workspace/hf_cache).
    --skip-existing Skip download if the model/dataset is already cached.
    --token         HuggingFace access token (or set HF_TOKEN env var).
    --dry-run       Print what would be downloaded without downloading.
"""

import argparse
import os
import shutil
import sys
import time

# ── Imports ──────────────────────────────────────────────────────────────────

try:
    from huggingface_hub import snapshot_download, repo_info
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
except ImportError:
    sys.exit("huggingface_hub not installed: pip install huggingface_hub>=0.24.0")

try:
    from datasets import load_dataset
    import datasets as ds_lib
except ImportError:
    sys.exit("datasets not installed: pip install datasets>=2.20.0")


# ── Dataset name → HF identifier mapping ─────────────────────────────────────

DATASET_MAP = {
    "wikitext2":   ("wikitext", "wikitext-2-raw-v1"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
    "c4":          ("c4",       "en"),
}


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def is_model_cached(model_id: str, cache_dir: str) -> bool:
    """Check whether the model snapshot directory exists and is non-empty."""
    # HF stores snapshots under <cache_dir>/models--<org>--<name>/snapshots/
    slug = "models--" + model_id.replace("/", "--")
    snap_dir = os.path.join(cache_dir, slug, "snapshots")
    if not os.path.isdir(snap_dir):
        return False
    for sub in os.listdir(snap_dir):
        if os.path.isdir(os.path.join(snap_dir, sub)):
            return True
    return False


def download_model(model_id: str, cache_dir: str, token: str | None,
                   skip_existing: bool, dry_run: bool) -> bool:
    """Download a model snapshot. Returns True on success."""
    print(f"\n  Model: {model_id}")

    if skip_existing and is_model_cached(model_id, cache_dir):
        slug = "models--" + model_id.replace("/", "--")
        size = dir_size(os.path.join(cache_dir, slug))
        print(f"  → already cached ({fmt_bytes(size)}) — skipping")
        return True

    if dry_run:
        print("  → [dry-run] would download")
        return True

    try:
        # Check accessibility first
        repo_info(model_id, token=token)
    except GatedRepoError:
        print(
            f"  ✗ Model is gated. Provide a token with access:\n"
            f"      --token YOUR_HF_TOKEN\n"
            f"    or: huggingface-cli login",
            file=sys.stderr,
        )
        return False
    except RepositoryNotFoundError:
        print(f"  ✗ Repository not found: {model_id}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ✗ Could not reach Hub: {e}", file=sys.stderr)
        return False

    print("  → downloading snapshot (this may take a while) ...")
    t0 = time.time()
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        elapsed = time.time() - t0
        size = dir_size(local_dir)
        print(f"  ✓ downloaded to {local_dir}")
        print(f"    size: {fmt_bytes(size)},  time: {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}", file=sys.stderr)
        return False


def download_dataset(name: str, cache_dir: str, skip_existing: bool,
                     dry_run: bool) -> bool:
    """Download a dataset. Returns True on success."""
    print(f"\n  Dataset: {name}")

    if name not in DATASET_MAP:
        print(
            f"  ✗ Unknown dataset '{name}'. Supported: {', '.join(DATASET_MAP)}",
            file=sys.stderr,
        )
        return False

    ds_name, config = DATASET_MAP[name]

    if dry_run:
        print(f"  → [dry-run] would load {ds_name} ({config})")
        return True

    print(f"  → loading {ds_name}/{config} ...")
    try:
        ds_env = os.environ.copy()
        ds_env["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

        # Use streaming=False to actually download; just load validation split
        _ = load_dataset(
            ds_name,
            config,
            split="validation",
            cache_dir=os.path.join(cache_dir, "datasets"),
            trust_remote_code=False,
        )
        print(f"  ✓ {name} validation split cached")
        return True
    except Exception as e:
        if "train" in str(e).lower() or "split" in str(e).lower():
            # Some datasets have no validation split; try train
            try:
                _ = load_dataset(
                    ds_name, config, split="train[:100]",
                    cache_dir=os.path.join(cache_dir, "datasets"),
                    trust_remote_code=False,
                )
                print(f"  ✓ {name} (train[:100]) cached")
                return True
            except Exception as e2:
                print(f"  ✗ Dataset load failed: {e2}", file=sys.stderr)
                return False
        print(f"  ✗ Dataset load failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--models", nargs="*", default=[],
        metavar="MODEL_ID",
        help="HuggingFace model IDs to download",
    )
    ap.add_argument(
        "--datasets", nargs="*", default=[],
        metavar="DATASET",
        help="Dataset names: wikitext2, c4, wikitext103",
    )
    ap.add_argument(
        "--cache-dir", default="/workspace/hf_cache",
        help="HuggingFace cache directory (default: /workspace/hf_cache)",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="Skip download if already cached",
    )
    ap.add_argument(
        "--token", default=None,
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading",
    )
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or None
    cache_dir = os.path.expanduser(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "datasets"), exist_ok=True)

    # Set env vars so HF libraries find the cache
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

    print()
    print("━" * 60)
    print("  prepare_models.py")
    print(f"  cache_dir   : {cache_dir}")
    print(f"  skip_existing: {args.skip_existing}")
    print(f"  dry_run     : {args.dry_run}")
    print("━" * 60)

    if not args.models and not args.datasets:
        print(
            "\nNo models or datasets specified.\n"
            "Example:\n"
            "  python scripts/prepare_models.py \\\n"
            "    --models Qwen/Qwen2.5-0.5B Qwen/Qwen3-30B-A3B \\\n"
            "    --datasets wikitext2 c4 \\\n"
            "    --skip-existing\n"
        )
        sys.exit(0)

    failures = []

    # ── Models ────────────────────────────────────────────────────────────────
    if args.models:
        print(f"\n── Models ({len(args.models)}) ──────────────────────────────────────────")
        for mid in args.models:
            ok = download_model(
                mid, cache_dir, token,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            if not ok:
                failures.append(f"model:{mid}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    if args.datasets:
        print(f"\n── Datasets ({len(args.datasets)}) ─────────────────────────────────────────")
        for dname in args.datasets:
            ok = download_dataset(
                dname, cache_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            if not ok:
                failures.append(f"dataset:{dname}")

    # ── Disk usage summary ────────────────────────────────────────────────────
    print()
    print("── Disk usage ───────────────────────────────────────────────")
    total = dir_size(cache_dir)
    print(f"  {cache_dir}: {fmt_bytes(total)} total")
    free = shutil.disk_usage(cache_dir).free
    print(f"  Free on device: {fmt_bytes(free)}")

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("━" * 60)
    if failures:
        print(f"  FAILED ({len(failures)} item(s)):")
        for f in failures:
            print(f"    ✗  {f}")
        print("━" * 60)
        sys.exit(1)
    else:
        print("  PREPARE COMPLETE")
        print("━" * 60)


if __name__ == "__main__":
    main()
