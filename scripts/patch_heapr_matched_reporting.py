#!/usr/bin/env python3
"""Patch only HEAPr reporting/eval arguments; fail if pinned source differs."""
from __future__ import annotations

import argparse
import hashlib
import os
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--patch-record", required=True)
    args = parser.parse_args()
    path = os.path.join(args.repo_dir, "main.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    original_hash = hashlib.sha256(text.encode()).hexdigest()
    if "_heapr_matched_reporting_patch" in text:
        raise RuntimeError("HEAPr reporting patch already applied")
    text, n_import = re.subn(r"^import os\s*$", "import os\nimport json", text,
                             count=1, flags=re.MULTILINE)
    text, n_before = re.subn(
        r"(?m)^(\s*)model\.eval\(\)\s*$",
        r"\1model.eval()\n\1_heapr_parameters_before = sum(p.numel() for p in model.parameters())",
        text, count=1,
    )
    text, n_after = re.subn(
        r"(?m)^(\s*)pruning_global\(model, cali_data, config, args, logger\)\s*$",
        r"\1pruning_global(model, cali_data, config, args, logger)\n"
        r"\1_heapr_parameters_after = sum(p.numel() for p in model.parameters())",
        text, count=1,
    )
    text, n_utils = re.subn(
        r"from lm_eval\.utils import make_table",
        "from lm_eval.utils import make_table, handle_non_serializable",
        text, count=1,
    )
    pattern = re.compile(
        r"(?P<indent>\s*)results\s*=\s*lm_eval\.simple_evaluate\(\s*"
        r"hflm,\s*tasks=tasks,\s*batch_size=\"auto\",\s*max_batch_size=256\s*\)",
        flags=re.MULTILINE,
    )

    def replacement(match: re.Match) -> str:
        indent = match.group("indent")
        body = f'''{indent}results = lm_eval.simple_evaluate(
{indent}    hflm, tasks=tasks, num_fewshot=0,
{indent}    batch_size=int(os.environ["HEAPR_MATCHED_BATCH_SIZE"]),
{indent}    log_samples=True, apply_chat_template=False,
{indent}    fewshot_as_multiturn=False, bootstrap_iters=10000,
{indent}    random_seed=int(os.environ["HEAPR_MATCHED_SEED"]),
{indent}    numpy_random_seed=int(os.environ["HEAPR_MATCHED_SEED"]),
{indent}    torch_random_seed=int(os.environ["HEAPR_MATCHED_SEED"]),
{indent}    fewshot_random_seed=int(os.environ["HEAPR_MATCHED_SEED"]),
{indent})
{indent}results["heapr_matched_protocol"] = {{
{indent}    "_heapr_matched_reporting_patch": True,
{indent}    "parameters_before": _heapr_parameters_before,
{indent}    "parameters_after": _heapr_parameters_after,
{indent}    "num_fewshot": 0, "batch_size": int(os.environ["HEAPR_MATCHED_BATCH_SIZE"]),
{indent}    "seed": int(os.environ["HEAPR_MATCHED_SEED"]),
{indent}    "apply_chat_template": False,
{indent}    "model_parameter_dtypes": sorted({{str(p.dtype) for p in model.parameters()}}),
{indent}    "task_versions": results.get("versions", {{}}),
{indent}}}
{indent}with open(os.environ["HEAPR_MATCHED_RESULTS_JSON"], "x", encoding="utf-8") as _fh:
{indent}    json.dump(results, _fh, indent=2, default=handle_non_serializable)'''
        return body

    text, n_eval = pattern.subn(replacement, text, count=1)
    counts = {"import": n_import, "before": n_before, "after": n_after,
              "utils": n_utils, "eval": n_eval}
    if set(counts.values()) != {1}:
        raise RuntimeError(f"pinned HEAPr source did not match reporting patch: {counts}")
    patched_hash = hashlib.sha256(text.encode()).hexdigest()
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.makedirs(os.path.dirname(args.patch_record) or ".", exist_ok=True)
    with open(args.patch_record, "x", encoding="utf-8") as handle:
        import json
        json.dump({"source_file": path, "original_sha256": original_hash,
                   "patched_sha256": patched_hash,
                   "scope": "reporting, fixed lm-eval settings, sample logging, parameter counts; pruning code unchanged"},
                  handle, indent=2)
    print(f"[heapr-patch] reporting-only patch: {args.patch_record}")


if __name__ == "__main__":
    main()
