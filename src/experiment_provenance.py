"""Reproducibility manifests for pruning calibration and PPL evaluation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
from typing import Mapping, Sequence


def normalize_document(text: str) -> str:
    """Normalize whitespace only; preserve case and punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip()


def document_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalized_document_sha256(text: str) -> str:
    return document_sha256(normalize_document(text))


def corpus_sha256(texts: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_text_manifest(
    evaluation_corpora: Mapping[str, Sequence[str]],
    calibration_prompts: Sequence[str],
) -> dict:
    calibration = [
        {
            "index": index,
            "sample_id": f"calibration:{index}:{document_sha256(text)}",
            "text_sha256": document_sha256(text),
            "normalized_text_sha256": normalized_document_sha256(text),
        }
        for index, text in enumerate(calibration_prompts)
    ]
    calibration_raw = {row["text_sha256"] for row in calibration}
    calibration_normalized = {
        row["normalized_text_sha256"] for row in calibration
    }
    evaluation = {}
    total_raw_overlap = 0
    total_normalized_overlap = 0
    for dataset, texts in evaluation_corpora.items():
        samples = [
            {
                "index": index,
                "sample_id": (
                    f"{dataset}:{index}:{document_sha256(text)}"
                ),
                "text_sha256": document_sha256(text),
                "normalized_text_sha256": normalized_document_sha256(text),
            }
            for index, text in enumerate(texts)
        ]
        raw_overlap = sorted(
            calibration_raw.intersection(row["text_sha256"] for row in samples)
        )
        normalized_overlap = sorted(
            calibration_normalized.intersection(
                row["normalized_text_sha256"] for row in samples
            )
        )
        total_raw_overlap += len(raw_overlap)
        total_normalized_overlap += len(normalized_overlap)
        evaluation[dataset] = {
            "num_samples": len(texts),
            "corpus_sha256": corpus_sha256(texts),
            "samples": samples,
            "calibration_raw_overlap_count": len(raw_overlap),
            "calibration_normalized_overlap_count": len(normalized_overlap),
            "calibration_raw_overlap_hashes": raw_overlap,
            "calibration_normalized_overlap_hashes": normalized_overlap,
        }
    return {
        "schema_version": 1,
        "calibration": {
            "source": "src.merging.RECONSTRUCTION_TRAIN_PROMPTS",
            "num_samples": len(calibration_prompts),
            "corpus_sha256": corpus_sha256(calibration_prompts),
            "samples": calibration,
        },
        "evaluation": evaluation,
        "disjointness": {
            "exact_text_overlap_count": total_raw_overlap,
            "normalized_text_overlap_count": total_normalized_overlap,
            "verified_disjoint": (
                total_raw_overlap == 0 and total_normalized_overlap == 0
            ),
        },
    }


def assert_calibration_evaluation_disjoint(manifest: Mapping[str, object]) -> None:
    disjointness = manifest.get("disjointness", {})
    if not isinstance(disjointness, Mapping) or not bool(
        disjointness.get("verified_disjoint", False)
    ):
        raise AssertionError(
            "activation calibration prompts overlap evaluation examples: "
            f"{disjointness}"
        )


def save_text_manifest(path: str, manifest: Mapping[str, object]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def extract_loaded_revision_metadata(model, tokenizer, resolved_model: str) -> dict:
    config = getattr(model, "config", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    model_revision = str(
        getattr(config, "_commit_hash", "")
        or getattr(config, "revision", "")
        or ""
    )
    tokenizer_revision = str(
        tokenizer_kwargs.get("_commit_hash", "")
        or tokenizer_kwargs.get("revision", "")
        or ""
    )
    return {
        "resolved_model": resolved_model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "transformers_version": importlib.metadata.version("transformers"),
    }
