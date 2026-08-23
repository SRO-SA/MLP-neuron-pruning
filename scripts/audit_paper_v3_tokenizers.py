#!/usr/bin/env python3
"""Audit original and physically exported tokenizers with/without regex repair."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import warnings
from collections import defaultdict
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment_provenance import corpus_sha256, document_sha256, file_sha256


CANARIES = (
    "A plain English sentence with ordinary punctuation.",
    "don't won't can't I'm we're they'd you've isn't shouldn't",
    "1 12 123 1,234.56 2026-08-20 +3.14e-10 $99.95 50%",
    "hello...world?! -- em—dash – en–dash _underscore_ /slash\\backslash",
    "def f(x):\n\treturn x**2  # code\nprint(f(7))",
    " leading space\nmultiple\n\nlines\tand\t tabs  trailing space ",
    "Unicode: café naïve résumé Ελληνικά Русский العربية 中文 日本語 한국어",
    "Emoji: 😀 🚀 ❤️‍🔥 👨‍👩‍👧‍👦 👍🏽 © ™ € £ ¥",
    "<|im_start|>user\nTokenizer canary<|im_end|>\n<|im_start|>assistant",
    "Regex boundaries: 'quoted' “curly” [brackets] {braces} (parentheses) a/b:c;d",
)

TOKENIZER_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
    "sentencepiece.bpe.model", "chat_template.jinja",
}

CSV_FIELDS = (
    "relation", "left_source", "left_mode", "right_source", "right_mode",
    "collection", "n_examples", "token_id_mismatches", "decoded_mismatches",
)

PAPER_V3_FROZEN_LABELS = frozenset({
    "baseline_unpruned",
    "rmsnorm_alloc__ellipsoid_rank__p95__target4",
    "rmsnorm_alloc__ellipsoid_rank__p95__target6",
    "rmsnorm_alloc__ellipsoid_rank__p95__target8",
})

PURE_DOWNNORM_CURVE_LABELS = frozenset({
    "baseline_unpruned",
    "rmsnorm_alloc__ellipsoid_rank__p95__target6",
    "rmsnorm_alloc__downnorm_rank__p95__target2",
    "rmsnorm_alloc__downnorm_rank__p95__target4",
    "rmsnorm_alloc__downnorm_rank__p95__target6",
    "rmsnorm_alloc__downnorm_rank__p95__target8",
})

CERTIFIED_HYBRID_FINE_LABELS = frozenset({
    "baseline_unpruned",
    "rmsnorm_alloc__ellipsoid_rank__p95__target6",
    "rmsnorm_alloc__downnorm_rank__p95__target6",
})

CHECKPOINT_COHORTS = (
    ("pure_downnorm_curve", PURE_DOWNNORM_CURVE_LABELS),
    ("paper_v3_frozen", PAPER_V3_FROZEN_LABELS),
    ("certified_hybrid_fine", CERTIFIED_HYBRID_FINE_LABELS),
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def validate_checkpoint_cohort(specs: list[dict]) -> str:
    available = {spec["label"] for spec in specs}
    for cohort, expected_labels in CHECKPOINT_COHORTS:
        if expected_labels.issubset(available):
            return cohort
    missing = {
        cohort: sorted(expected_labels - available)
        for cohort, expected_labels in CHECKPOINT_COHORTS
    }
    raise ValueError(f"frozen checkpoint labels missing for supported cohorts: {missing}")


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _tokenizer_file(path: str) -> bool:
    name = os.path.basename(path)
    return (
        name in TOKENIZER_NAMES
        or name.startswith("tokenizer.")
        or name.startswith("tokenizer_")
        or name.startswith("chat_template")
    )


def _combined_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["filename"]):
        digest.update(row["filename"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def tokenizer_file_hashes(source: str, tokenizer: Any) -> list[dict]:
    paths: dict[str, str] = {}
    if os.path.isdir(source):
        for name in os.listdir(source):
            path = os.path.join(source, name)
            if os.path.isfile(path) and _tokenizer_file(path):
                paths[name] = path
    for value in (getattr(tokenizer, "init_kwargs", {}) or {}).values():
        if isinstance(value, str) and os.path.isfile(value) and _tokenizer_file(value):
            paths[os.path.basename(value)] = value
    if not os.path.isdir(source):
        from transformers.utils.hub import cached_file
        names = TOKENIZER_NAMES.union(
            set(getattr(tokenizer, "vocab_files_names", {}).values())
        )
        for name in sorted(names):
            try:
                path = cached_file(source, name, local_files_only=True)
            except (EnvironmentError, OSError, ValueError):
                path = None
            if path and os.path.isfile(path):
                paths[name] = path
    rows = []
    for filename, path in sorted(paths.items()):
        rows.append({
            "filename": filename,
            "size_bytes": os.path.getsize(path),
            "sha256": file_sha256(path),
        })
    if not rows:
        raise FileNotFoundError(f"no tokenizer files resolved for {source}")
    return rows


def load_tokenizer(source: str, *, fix: bool) -> tuple[Any, list[str]]:
    from transformers import AutoTokenizer
    handler = _Capture()
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tokenizer = AutoTokenizer.from_pretrained(
                source, trust_remote_code=True, use_fast=True,
                fix_mistral_regex=fix,
            )
        messages = handler.messages + [str(item.message) for item in caught]
    finally:
        logger.removeHandler(handler)
    return tokenizer, list(dict.fromkeys(messages))


def encode_record(tokenizer: Any, text: str) -> dict:
    ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=True)]
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    return {"token_ids": ids, "decoded": decoded}


def compare_records(
    left: list[dict], right: list[dict], examples: list[dict],
    *, representative_limit: int = 12,
) -> tuple[list[dict], list[dict]]:
    if len(left) != len(right) or len(left) != len(examples):
        raise ValueError("tokenizer comparison lengths differ")
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_examples": 0, "token_id_mismatches": 0,
                 "decoded_mismatches": 0}
    )
    representatives = []
    for example, first, second in zip(examples, left, right):
        collection = example["collection"]
        stat = grouped[collection]
        stat["n_examples"] += 1
        ids_differ = first["token_ids"] != second["token_ids"]
        decoded_differ = first["decoded"] != second["decoded"]
        stat["token_id_mismatches"] += int(ids_differ)
        stat["decoded_mismatches"] += int(decoded_differ)
        if (ids_differ or decoded_differ) and len(representatives) < representative_limit:
            limit = min(len(first["token_ids"]), len(second["token_ids"]))
            first_difference = next(
                (index for index in range(limit)
                 if first["token_ids"][index] != second["token_ids"][index]),
                limit if len(first["token_ids"]) != len(second["token_ids"]) else -1,
            )
            representatives.append({
                "collection": collection,
                "sample_index": example["sample_index"],
                "text_sha256": example["text_sha256"],
                "text_excerpt": example["text"][:240],
                "first_difference_index": first_difference,
                "left_token_count": len(first["token_ids"]),
                "right_token_count": len(second["token_ids"]),
                "left_token_ids_prefix": first["token_ids"][:128],
                "right_token_ids_prefix": second["token_ids"][:128],
                "left_decoded_excerpt": first["decoded"][:240],
                "right_decoded_excerpt": second["decoded"][:240],
            })
    return [dict(collection=key, **value) for key, value in sorted(grouped.items())], representatives


def build_decision(rows: list[dict]) -> dict:
    fix_rows = [row for row in rows if row["relation"] == "current_vs_fixed"]
    export_rows = [row for row in rows if row["relation"] == "original_vs_export"]
    def exports_match(mode: str) -> bool:
        mode_rows = [row for row in export_rows if row["right_mode"] == mode]
        return bool(mode_rows) and all(
            row["token_id_mismatches"] == 0 and row["decoded_mismatches"] == 0
            for row in mode_rows
        )

    current_matches = exports_match("current")
    fixed_matches = exports_match("fixed")
    # Preserve the tokenizer behavior used by the validated PPL experiments when
    # it is identical across all exported checkpoints.  The Mistral repair is a
    # fallback only when it is the mode that restores that equivalence.
    if current_matches:
        selected_mode = "current"
        use_fix = False
        selection_reason = (
            "All exported checkpoint tokenizers exactly match the pinned hub "
            "tokenizer without the Mistral regex repair. The historical/current "
            "loading path is preserved."
        )
    elif fixed_matches:
        selected_mode = "fixed"
        use_fix = True
        selection_reason = (
            "The exported checkpoint tokenizers match the pinned hub tokenizer "
            "only when the Mistral regex repair is enabled."
        )
    else:
        selected_mode = None
        use_fix = None
        selection_reason = (
            "Neither tokenizer loading mode makes every exported checkpoint "
            "exactly match the pinned hub tokenizer."
        )
    original_eval_fix_mismatches = sum(
        row["token_id_mismatches"] for row in fix_rows
        if row["left_source"] == "hub_original"
        if row["collection"] in ("wikitext2", "c4")
    )
    exported_eval_fix_mismatches = sum(
        row["token_id_mismatches"] for row in fix_rows
        if row["left_source"] != "hub_original"
        if row["collection"] in ("wikitext2", "c4")
    )
    ppl_rerun_required = bool(use_fix and original_eval_fix_mismatches > 0)
    return {
        "audit_passed_for_downstream": selected_mode is not None,
        "selected_tokenizer_mode": selected_mode,
        "all_exports_match_original_current": current_matches,
        "all_exports_match_original_fixed": fixed_matches,
        "fix_changes_any_audited_token_ids": any(
            row["token_id_mismatches"] > 0 for row in fix_rows
        ),
        "fix_changes_original_wikitext2_or_c4_token_ids": (
            original_eval_fix_mismatches > 0
        ),
        "fix_changes_exported_wikitext2_or_c4_token_ids": (
            exported_eval_fix_mismatches > 0
        ),
        "use_fix_mistral_regex_for_future_evaluation": use_fix,
        "selection_reason": selection_reason,
        "local_mistral_warning_consistent_with_false_positive": bool(
            current_matches and not fixed_matches
        ),
        "previous_ppl_rerun_required": ppl_rerun_required,
        "previous_ppl_rerun_reason": (
            "Required: the hub tokenizer used by prior PPL runs changes token IDs "
            "on audited WikiText2/C4 passages when the regex repair is enabled."
            if ppl_rerun_required else
            "Not required: the selected tokenizer policy preserves the tokenization "
            "used by the validated PPL experiments."
        ),
    }


def _write_markdown(path: str, rows: list[dict], decision: dict) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("# Tokenizer audit\n\n")
        handle.write(f"Audit passed for downstream: `{decision['audit_passed_for_downstream']}`  \n")
        handle.write(f"Selected tokenizer mode: `{decision['selected_tokenizer_mode']}`  \n")
        handle.write(
            "Use `fix_mistral_regex=True`: "
            f"`{decision['use_fix_mistral_regex_for_future_evaluation']}`  \n"
        )
        handle.write(f"Selection reason: {decision['selection_reason']}  \n")
        handle.write(f"Previous PPL rerun required: `{decision['previous_ppl_rerun_required']}`\n\n")
        handle.write("| Relation | Left | Right | Collection | Examples | ID mismatches | Decode mismatches |\n")
        handle.write("|---|---|---|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['relation']} | {row['left_source']} ({row['left_mode']}) | "
                f"{row['right_source']} ({row['right_mode']}) | {row['collection']} | "
                f"{row['n_examples']} | {row['token_id_mismatches']} | "
                f"{row['decoded_mismatches']} |\n"
            )


def _escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(path: str, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llllrrr}\n\\toprule\n")
        handle.write("Relation & Left & Right & Corpus & $n$ & ID diff. & Decode diff. \\\\\n\\midrule\n")
        for row in rows:
            values = (
                row["relation"], f"{row['left_source']}:{row['left_mode']}",
                f"{row['right_source']}:{row['right_mode']}", row["collection"],
                row["n_examples"], row["token_id_mismatches"],
                row["decoded_mismatches"],
            )
            handle.write(" & ".join(_escape(value) for value in values) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples_per_dataset < 100:
        raise ValueError("tokenizer audit requires at least 100 samples per dataset")
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    with open(args.checkpoint_manifest, encoding="utf-8") as handle:
        specs = json.load(handle)
    checkpoint_cohort = validate_checkpoint_cohort(specs)
    for spec in specs:
        if not os.path.isdir(spec["checkpoint_dir"]):
            raise FileNotFoundError(spec["checkpoint_dir"])
    if args.dry_run:
        print(
            f"[tokenizer-audit] DRY RUN: model={args.model} "
            f"cohort={checkpoint_cohort} checkpoints={len(specs)} "
            f"datasets=wikitext2,c4 samples_each={args.samples_per_dataset} "
            f"output={args.output_dir}"
        )
        return

    from src.evaluation import load_all_eval_datasets

    corpora = load_all_eval_datasets(
        ["wikitext2", "c4"], max_samples=args.samples_per_dataset,
        use_fallback_corpus=False,
    )
    examples = [
        {"collection": "canary", "sample_index": index,
         "text": text, "text_sha256": document_sha256(text)}
        for index, text in enumerate(CANARIES)
    ]
    for dataset, texts in corpora.items():
        if len(texts) < args.samples_per_dataset:
            raise ValueError(f"{dataset} returned only {len(texts)} samples")
        examples.extend({
            "collection": dataset, "sample_index": index, "text": text,
            "text_sha256": document_sha256(text),
        } for index, text in enumerate(texts))

    sources = [("hub_original", args.model)] + [
        (spec["label"], spec["checkpoint_dir"]) for spec in specs
    ]
    loaded: dict[str, dict[str, list[dict]]] = {}
    source_reports = []
    for label, source in sources:
        loaded[label] = {}
        mode_reports = {}
        file_rows = None
        for mode, fix in (("current", False), ("fixed", True)):
            tokenizer, messages = load_tokenizer(source, fix=fix)
            if file_rows is None:
                file_rows = tokenizer_file_hashes(source, tokenizer)
            records = [encode_record(tokenizer, example["text"]) for example in examples]
            loaded[label][mode] = records
            mode_reports[mode] = {
                "fix_mistral_regex": fix,
                "tokenizer_class": type(tokenizer).__name__,
                "name_or_path": str(tokenizer.name_or_path),
                "vocab_size": len(tokenizer),
                "warnings": messages,
            }
        source_reports.append({
            "label": label, "source": os.path.realpath(source) if os.path.exists(source) else source,
            "tokenizer_files": file_rows,
            "tokenizer_files_combined_sha256": _combined_hash(file_rows or []),
            "modes": mode_reports,
        })

    rows, detailed = [], []
    for label, _ in sources:
        stats, representatives = compare_records(
            loaded[label]["current"], loaded[label]["fixed"], examples,
        )
        for stat in stats:
            rows.append({
                "relation": "current_vs_fixed", "left_source": label,
                "left_mode": "current", "right_source": label,
                "right_mode": "fixed", **stat,
            })
        detailed.append({
            "relation": "current_vs_fixed", "left_source": label,
            "right_source": label, "representative_mismatches": representatives,
        })
    for label, _ in sources[1:]:
        for mode in ("current", "fixed"):
            stats, representatives = compare_records(
                loaded["hub_original"][mode], loaded[label][mode], examples,
            )
            for stat in stats:
                rows.append({
                    "relation": "original_vs_export", "left_source": "hub_original",
                    "left_mode": mode, "right_source": label,
                    "right_mode": mode, **stat,
                })
            detailed.append({
                "relation": "original_vs_export", "mode": mode,
                "left_source": "hub_original", "right_source": label,
                "representative_mismatches": representatives,
            })

    decision = build_decision(rows)
    payload = {
        "schema_version": 2,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint_manifest": os.path.realpath(args.checkpoint_manifest),
        "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
        "libraries": {
            "python": platform.python_version(), "transformers": _version("transformers"),
            "tokenizers": _version("tokenizers"), "datasets": _version("datasets"),
            "huggingface_hub": _version("huggingface-hub"),
        },
        "protocol": {
            "samples_per_dataset": args.samples_per_dataset,
            "canary_count": len(CANARIES), "add_special_tokens": True,
            "decode_skip_special_tokens": False,
            "decode_clean_up_tokenization_spaces": False,
            "dataset_loader": "src.evaluation.load_all_eval_datasets",
        },
        "corpora": {
            dataset: {"num_samples": len(texts), "corpus_sha256": corpus_sha256(texts),
                      "sample_sha256": [document_sha256(text) for text in texts]}
            for dataset, texts in corpora.items()
        },
        "sources": source_reports, "comparison_rows": rows,
        "comparison_details": detailed, "decision": decision,
    }
    os.makedirs(args.output_dir)
    json_path = os.path.join(args.output_dir, "tokenizer_audit.json")
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    csv_path = os.path.join(args.output_dir, "tokenizer_audit.csv")
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    _write_markdown(os.path.join(args.output_dir, "tokenizer_audit.md"), rows, decision)
    _write_latex(os.path.join(args.output_dir, "tokenizer_audit.tex"), rows)
    if not decision["audit_passed_for_downstream"]:
        raise RuntimeError(
            "tokenizer audit completed but neither loading mode makes every "
            f"exported tokenizer match the pinned hub tokenizer; inspect {json_path}"
        )
    print(
        f"[tokenizer-audit] OK: comparisons={len(rows)} "
        f"selected_mode={decision['selected_tokenizer_mode']} "
        f"fix_mistral_regex="
        f"{decision['use_fix_mistral_regex_for_future_evaluation']} "
        f"fix_changes_original_eval="
        f"{decision['fix_changes_original_wikitext2_or_c4_token_ids']} "
        f"ppl_rerun_required={decision['previous_ppl_rerun_required']} JSON={json_path}"
    )


if __name__ == "__main__":
    main()
