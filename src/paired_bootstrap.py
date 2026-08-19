"""Paired document-bootstrap utilities for language-model NLL comparisons."""
from __future__ import annotations

import csv
import math
from typing import Mapping, Sequence

import numpy as np


def paired_bootstrap_nll_difference(
    candidate_nll_sums: Sequence[float],
    reference_nll_sums: Sequence[float],
    token_counts: Sequence[int],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Return candidate-minus-reference token NLL and a paired document CI.

    Documents are resampled as paired units.  Within each bootstrap replicate,
    the reported statistic is the corpus token-weighted mean NLL difference.
    """
    candidate = np.asarray(candidate_nll_sums, dtype=np.float64)
    reference = np.asarray(reference_nll_sums, dtype=np.float64)
    tokens = np.asarray(token_counts, dtype=np.int64)
    if candidate.ndim != 1 or reference.ndim != 1 or tokens.ndim != 1:
        raise ValueError("paired bootstrap inputs must be one-dimensional")
    if not (len(candidate) == len(reference) == len(tokens)) or not len(tokens):
        raise ValueError("paired bootstrap inputs must have equal non-zero length")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("paired bootstrap NLL sums must be finite")
    if np.any(tokens <= 0):
        raise ValueError("paired bootstrap token counts must all be positive")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    differences = candidate - reference
    point = float(differences.sum() / tokens.sum())
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    # Chunking bounds temporary index memory for large evaluation corpora.
    chunk = 256
    for start in range(0, n_resamples, chunk):
        stop = min(start + chunk, n_resamples)
        indices = rng.integers(0, len(tokens), size=(stop - start, len(tokens)))
        numerator = differences[indices].sum(axis=1)
        denominator = tokens[indices].sum(axis=1)
        estimates[start:stop] = numerator / denominator
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "mean_nll_difference": point,
        "ci_lower": float(low),
        "ci_upper": float(high),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "n_documents": int(len(tokens)),
        "n_tokens": int(tokens.sum()),
        "seed": int(seed),
    }


def write_paired_nll_csv(
    path: str,
    *,
    dataset: str,
    corpus_sha256: str,
    baseline_examples: Sequence[Mapping[str, object]],
    pruned_examples: Sequence[Mapping[str, object]],
) -> None:
    """Write aligned baseline/pruned per-document sufficient statistics."""
    if len(baseline_examples) != len(pruned_examples):
        raise ValueError("baseline/pruned per-document lengths differ")
    fields = [
        "dataset", "corpus_sha256", "sample_index", "n_tokens",
        "baseline_nll_sum", "baseline_nll_mean",
        "pruned_nll_sum", "pruned_nll_mean",
    ]
    rows = []
    for index, (baseline, pruned) in enumerate(
        zip(baseline_examples, pruned_examples)
    ):
        baseline_index = int(baseline["sample_index"])
        pruned_index = int(pruned["sample_index"])
        baseline_tokens = int(baseline["n_tokens"])
        pruned_tokens = int(pruned["n_tokens"])
        if baseline_index != index or pruned_index != index:
            raise ValueError(f"per-document sample index mismatch at row {index}")
        if baseline_tokens != pruned_tokens or baseline_tokens <= 0:
            raise ValueError(f"per-document token mismatch at row {index}")
        baseline_sum = float(baseline["nll_sum"])
        pruned_sum = float(pruned["nll_sum"])
        if not math.isfinite(baseline_sum) or not math.isfinite(pruned_sum):
            raise ValueError(f"non-finite per-document NLL at row {index}")
        rows.append({
            "dataset": dataset,
            "corpus_sha256": corpus_sha256,
            "sample_index": index,
            "n_tokens": baseline_tokens,
            "baseline_nll_sum": baseline_sum,
            "baseline_nll_mean": baseline_sum / baseline_tokens,
            "pruned_nll_sum": pruned_sum,
            "pruned_nll_mean": pruned_sum / pruned_tokens,
        })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
