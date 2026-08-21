"""Multiple-testing and paired-randomization helpers for paper audits."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Benjamini-Hochberg FDR adjusted p-values in input order."""
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 1.0
    total = len(values)
    for reverse_rank in range(total - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, total * float(values[index]) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_signflip_statistics(
    first: dict[str, list[float]],
    second: dict[str, list[float]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """Paired randomization p-values, stratified by task for the macro test.

    Each task's paired example differences are independently sign-flipped under
    the sharp null.  The macro statistic is the equal-weight mean of the task
    means, preserving the paper's task-macro estimand.
    """
    if set(first) != set(second):
        raise ValueError("paired task sets differ")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    deltas: dict[str, np.ndarray] = {}
    observed: dict[str, float] = {}
    for task in sorted(first):
        left = np.asarray(first[task], dtype=np.float64)
        right = np.asarray(second[task], dtype=np.float64)
        if left.shape != right.shape or left.ndim != 1 or left.size == 0:
            raise ValueError(f"invalid paired samples for {task}")
        delta = left - right
        deltas[task] = delta
        observed[task] = float(delta.mean())

    rng = np.random.default_rng(seed)
    draws = {task: np.empty(n_resamples, dtype=np.float64) for task in deltas}
    for task, delta in deltas.items():
        for start in range(0, n_resamples, 1000):
            stop = min(start + 1000, n_resamples)
            signs = rng.integers(
                0, 2, size=(stop - start, len(delta)), dtype=np.int8
            ).astype(np.float64)
            signs = signs * 2.0 - 1.0
            draws[task][start:stop] = (signs * delta).mean(axis=1)

    result = {}
    for task in sorted(deltas):
        exceed = int(np.count_nonzero(
            np.abs(draws[task]) >= abs(observed[task]) - 1e-15
        ))
        result[task] = (exceed + 1.0) / (n_resamples + 1.0)
    macro_observed = float(np.mean(list(observed.values())))
    macro_draws = np.vstack([draws[task] for task in sorted(draws)]).mean(axis=0)
    macro_exceed = int(np.count_nonzero(
        np.abs(macro_draws) >= abs(macro_observed) - 1e-15
    ))
    result["macro_average"] = (macro_exceed + 1.0) / (n_resamples + 1.0)
    return result


def apply_multiplicity_adjustments(
    rows: list[dict], *, family_key: str = "multiplicity_family"
) -> None:
    """Mutate rows with Holm/BH values, correcting within declared families."""
    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        family = str(row[family_key])
        families.setdefault(family, []).append(index)
    for family, indices in families.items():
        if not family:
            raise ValueError("multiplicity family cannot be empty")
        p_values = [float(rows[index]["paired_randomization_p_value"])
                    for index in indices]
        holm = holm_adjust(p_values)
        bh = benjamini_hochberg_adjust(p_values)
        for position, index in enumerate(indices):
            rows[index]["holm_adjusted_p_value"] = holm[position]
            rows[index]["bh_adjusted_p_value"] = bh[position]
            rows[index]["holm_significant_0_05"] = holm[position] < 0.05
            rows[index]["bh_significant_0_05"] = bh[position] < 0.05
            rows[index]["multiplicity_family_size"] = len(indices)
