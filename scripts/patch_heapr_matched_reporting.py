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
    text, n_import = re.subn(r"^import os\s*$", "import os\nimport json\nimport time", text,
                             count=1, flags=re.MULTILINE)
    text, n_before = re.subn(
        r"(?m)^(\s*)model\.eval\(\)\s*$",
        r"\1model.eval()\n"
        r"\1_heapr_parameters_before = sum(p.numel() for p in model.parameters())\n"
        r"\1_heapr_moe_parameters_before = sum(p.numel() for n, p in model.named_parameters() if '.mlp.experts.' in n)",
        text, count=1,
    )
    text, n_after = re.subn(
        r"(?m)^(\s*)pruning_global\(model, cali_data, config, args, logger\)\s*$",
        r"\1for _heapr_gpu in range(torch.cuda.device_count()):\n"
        r"\1    torch.cuda.synchronize(_heapr_gpu)\n"
        r"\1    torch.cuda.reset_peak_memory_stats(_heapr_gpu)\n"
        r"\1_heapr_selection_start_allocated = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))\n"
        r"\1_heapr_selection_started = time.perf_counter()\n"
        r"\1pruning_global(model, cali_data, config, args, logger)\n"
        r"\1for _heapr_gpu in range(torch.cuda.device_count()):\n"
        r"\1    torch.cuda.synchronize(_heapr_gpu)\n"
        r"\1_heapr_selection_wall_seconds = time.perf_counter() - _heapr_selection_started\n"
        r"\1_heapr_selection_peak_allocated = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))\n"
        r"\1_heapr_parameters_after = sum(p.numel() for p in model.parameters())\n"
        r"\1_heapr_moe_parameters_after = sum(p.numel() for n, p in model.named_parameters() if '.mlp.experts.' in n)\n"
        r"\1if os.environ.get('HEAPR_MATCHED_EXPORT_CHECKPOINT') == '1':\n"
        r"\1    _heapr_checkpoint_dir = os.environ['HEAPR_MATCHED_CHECKPOINT_DIR']\n"
        r"\1    if os.path.exists(_heapr_checkpoint_dir):\n"
        r"\1        raise FileExistsError(_heapr_checkpoint_dir)\n"
        r"\1    model.save_pretrained(_heapr_checkpoint_dir, safe_serialization=True)\n"
        r"\1    tokenizer.save_pretrained(_heapr_checkpoint_dir)",
        text, count=1,
    )
    text, n_utils = re.subn(
        r"from lm_eval\.utils import make_table",
        "from lm_eval.utils import make_table, handle_non_serializable",
        text, count=1,
    )
    text, n_trust = re.subn(
        r"(?m)^(\s*)from lm_eval\.tasks import TaskManager\s*$",
        r"\1from lm_eval.tasks import TaskManager\n"
        r"\1import datasets\n"
        r"\1if os.environ.get('HEAPR_MATCHED_TRUST_DATASET_CODE') != '1':\n"
        r"\1    raise RuntimeError('MathQA dataset-code authorization is required')\n"
        r"\1datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True",
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
{indent}    "moe_expert_parameters_before": _heapr_moe_parameters_before,
{indent}    "moe_expert_parameters_after": _heapr_moe_parameters_after,
{indent}    "num_fewshot": 0, "batch_size": int(os.environ["HEAPR_MATCHED_BATCH_SIZE"]),
{indent}    "seed": int(os.environ["HEAPR_MATCHED_SEED"]),
{indent}    "apply_chat_template": False,
{indent}    "model_parameter_dtypes": sorted({{str(p.dtype) for p in model.parameters()}}),
{indent}    "task_versions": results.get("versions", {{}}),
{indent}    "trust_dataset_code": os.environ.get("HEAPR_MATCHED_TRUST_DATASET_CODE") == "1",
{indent}    "tokenizer_class": type(tokenizer).__name__,
{indent}    "tokenizer_name_or_path": tokenizer.name_or_path,
{indent}    "selected_tokenizer_mode": "current",
{indent}    "fix_mistral_regex": False,
{indent}    "calibration_token_count": int(cali_data["input_ids"].numel()),
{indent}    "selection_wall_clock_seconds": _heapr_selection_wall_seconds,
{indent}    "selection_start_allocated_bytes_total": _heapr_selection_start_allocated,
{indent}    "selection_peak_allocated_bytes_total": _heapr_selection_peak_allocated,
{indent}    "selection_peak_incremental_allocated_bytes_total": max(0, _heapr_selection_peak_allocated - _heapr_selection_start_allocated),
{indent}}}
{indent}with open(os.environ["HEAPR_MATCHED_RESULTS_JSON"], "x", encoding="utf-8") as _fh:
{indent}    json.dump(results, _fh, indent=2, default=handle_non_serializable)'''
        return body

    text, n_eval = pattern.subn(replacement, text, count=1)
    counts = {"import": n_import, "before": n_before, "after": n_after,
              "utils": n_utils, "trust": n_trust, "eval": n_eval}
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
                   "scope": "reporting, fixed lm-eval settings, sample logging, parameter counts, physical checkpoint export; pruning code unchanged"},
                  handle, indent=2)
    print(f"[heapr-patch] reporting-only patch: {args.patch_record}")


if __name__ == "__main__":
    main()
