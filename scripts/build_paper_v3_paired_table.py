#!/usr/bin/env python3
"""Extract the preregistered Version 3 paired dNLL comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re


COMPLETE_REQUESTS = [
    (2, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
    (4, "ranking", "rmsnorm_bound", "activation_score"),
    (4, "ranking", "rmsnorm_bound", "down_norm"),
    (4, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
    (4, "ranking", "down_norm", "activation_score"),
    (4, "ranking", "down_norm", "down_norm"),
    (4, "aggregation", "rmsnorm_bound", "max"),
    (6, "ranking", "rmsnorm_bound", "activation_score"),
    (6, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
    (6, "allocation", "exact2256", "down_norm"),
    (6, "aggregation", "rmsnorm_bound", "max"),
    (8, "ranking", "rmsnorm_bound", "activation_score"),
    (8, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
]

PRIMARY_REQUESTS = [
    (4, "ranking", "rmsnorm_bound", "activation_score"),
    (4, "ranking", "rmsnorm_bound", "down_norm"),
    (4, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
    (6, "ranking", "rmsnorm_bound", "activation_score"),
    (6, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
    (8, "ranking", "rmsnorm_bound", "activation_score"),
    (8, "ranking", "rmsnorm_bound", "rmsnorm_bound"),
]

# Backward-compatible public name used by earlier callers.
REQUESTS = COMPLETE_REQUESTS

FIELDS = [
    "target_pct", "comparison_type", "allocation_context", "dataset",
    "first_method", "second_method", "difference_definition",
    "mean_dnll_first_minus_second", "ci95_lower", "ci95_upper",
    "significant_95pct", "favored_method_if_significant", "n_documents",
    "n_tokens", "bootstrap_resamples", "source_group", "source_run_dir",
]


def _target(row: dict) -> int:
    haystack = " ".join(str(row.get(field, "")) for field in (
        "source_run_dir", "source_group", "ellipsoid_experiment",
        "competitor_experiment", "p95_experiment", "max_experiment",
    ))
    matches = {int(value) for value in re.findall(r"target(\d+)", haystack)}
    if len(matches) != 1:
        raise ValueError(f"cannot infer one target from paired row: {row}")
    return next(iter(matches))


def _float(row: dict, field: str) -> float:
    value = float(row[field])
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {field}: {row[field]!r}")
    return value


def normalize_comparison(row: dict) -> dict:
    kind = row["comparison_type"]
    target = _target(row)
    dataset = row["dataset"]
    if kind == "ranking":
        allocation = row["allocation_source"]
        competitor = row["competitor_ranking"]
        first = "rmsnorm_ellipsoid_bound"
        second = competitor
        mean = _float(row, "ellipsoid_minus_competitor_mean_nll")
        lower = _float(row, "ci95_lower")
        upper = _float(row, "ci95_upper")
        request_key = (target, kind, allocation, competitor)
        context = allocation
    elif kind == "aggregation":
        allocation = row["allocation_source"]
        # Source statistic is max-minus-p95. Paper convention is p95-minus-max.
        source_mean = _float(row, "max_minus_p95_mean_nll")
        source_lower = _float(row, "ci95_lower")
        source_upper = _float(row, "ci95_upper")
        first, second = "p95", "max"
        mean, lower, upper = -source_mean, -source_upper, -source_lower
        request_key = (target, kind, allocation, "max")
        context = allocation
    elif kind == "allocation":
        if int(float(row["exact_removed_layer_channels"])) != 2256:
            raise ValueError("target-6 allocation comparison is not exact budget 2256")
        if row.get("ranking_source") != "rmsnorm_ellipsoid_bound":
            raise ValueError("exact allocation comparison does not use ellipsoid ranking")
        first, second = "rmsnorm_bound allocation", "down_norm allocation"
        mean = _float(row, "rmsnorm_minus_downnorm_mean_nll")
        lower = _float(row, "ci95_lower")
        upper = _float(row, "ci95_upper")
        request_key = (target, kind, "exact2256", "down_norm")
        context = "exact 2256; ellipsoid ranking"
    else:
        raise ValueError(f"unknown comparison_type={kind!r}")
    if lower > upper:
        raise ValueError(f"reversed confidence interval: {lower}, {upper}")
    significant = lower > 0.0 or upper < 0.0
    favored = ""
    if significant:
        favored = first if upper < 0.0 else second
    return {
        "request_key": request_key,
        "target_pct": target,
        "comparison_type": kind,
        "allocation_context": context,
        "dataset": dataset,
        "first_method": first,
        "second_method": second,
        "difference_definition": "first_method minus second_method token dNLL",
        "mean_dnll_first_minus_second": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "significant_95pct": significant,
        "favored_method_if_significant": favored,
        "n_documents": int(float(row["n_documents"])),
        "n_tokens": int(float(row["n_tokens"])),
        "bootstrap_resamples": int(float(row["bootstrap_resamples"])),
        "source_group": row.get("source_group", ""),
        "source_run_dir": row.get("source_run_dir", ""),
    }


def build_requested_table(
    source_rows: list[dict], requests: list[tuple] | None = None,
) -> list[dict]:
    requests = COMPLETE_REQUESTS if requests is None else requests
    normalized = [normalize_comparison(row) for row in source_rows]
    by_key = {}
    for row in normalized:
        key = (*row["request_key"], row["dataset"])
        by_key.setdefault(key, []).append(row)
    output = []
    for request in requests:
        for dataset in ("wikitext2", "c4"):
            matches = by_key.get((*request, dataset), [])
            if len(matches) != 1:
                raise ValueError(
                    f"requested paired comparison {request} dataset={dataset} "
                    f"has {len(matches)} row(s)"
                )
            row = dict(matches[0])
            row.pop("request_key", None)
            output.append(row)
    return output


def _fmt(value: object) -> str:
    return f"{float(value):.6f}"


def _write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "| Target | Dataset | Context | First | Second | dNLL | 95% CI | Significant | Favored |\n"
            "|---:|---|---|---|---|---:|---|---|---|\n"
        )
        for row in rows:
            ci = f"[{_fmt(row['ci95_lower'])}, {_fmt(row['ci95_upper'])}]"
            handle.write(
                f"| {row['target_pct']} | {row['dataset']} | "
                f"{row['allocation_context']} | {row['first_method']} | "
                f"{row['second_method']} | {_fmt(row['mean_dnll_first_minus_second'])} | "
                f"{ci} | {'Yes' if row['significant_95pct'] else 'No'} | "
                f"{row['favored_method_if_significant'] or '—'} |\n"
            )


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{rllllrrll}\n\\toprule\n")
        handle.write(
            "Target & Dataset & Context & First & Second & $\\Delta$NLL & 95\\% CI & Sig. & Favored \\\\\n\\midrule\n"
        )
        for row in rows:
            values = [
                row["target_pct"], row["dataset"], row["allocation_context"],
                row["first_method"], row["second_method"],
                _fmt(row["mean_dnll_first_minus_second"]),
                f"[{_fmt(row['ci95_lower'])}, {_fmt(row['ci95_upper'])}]",
                "Yes" if row["significant_95pct"] else "No",
                row["favored_method_if_significant"] or "--",
            ]
            handle.write(" & ".join(_latex_escape(value) for value in values) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("primary", "complete"),
                        default="complete")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.input, newline="", encoding="utf-8") as handle:
        requests = PRIMARY_REQUESTS if args.profile == "primary" else COMPLETE_REQUESTS
        rows = build_requested_table(list(csv.DictReader(handle)), requests=requests)
    os.makedirs(args.output_dir)
    csv_path = os.path.join(args.output_dir, "paper_v3_paired_dnll_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(args.output_dir, "paper_v3_paired_dnll_table.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    md_path = os.path.join(args.output_dir, "paper_v3_paired_dnll_table.md")
    tex_path = os.path.join(args.output_dir, "paper_v3_paired_dnll_table.tex")
    _write_markdown(md_path, rows)
    _write_latex(tex_path, rows)
    for row in rows:
        print(
            f"T{row['target_pct']} {row['dataset']:10s} "
            f"{row['first_method']} vs {row['second_method']}: "
            f"dNLL={_fmt(row['mean_dnll_first_minus_second'])} "
            f"CI=[{_fmt(row['ci95_lower'])}, {_fmt(row['ci95_upper'])}] "
            f"sig={'YES' if row['significant_95pct'] else 'NO'}"
        )
    print(f"[paired-table] OK: {len(rows)} rows; CSV={csv_path}")


if __name__ == "__main__":
    main()
