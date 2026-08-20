"""Load and enforce the frozen tokenizer-audit decision."""
from __future__ import annotations

import json
import os

from .experiment_provenance import file_sha256


def resolve_tokenizer_policy(
    audit_path: str, checkpoint_path: str, *, label: str = "",
) -> dict:
    with open(audit_path, encoding="utf-8") as handle:
        audit = json.load(handle)
    decision = audit.get("decision", {})
    if decision.get("audit_passed_for_downstream") is not True:
        raise ValueError("tokenizer audit did not pass the downstream gate")
    if decision.get("use_fix_mistral_regex_for_future_evaluation") is not True:
        raise ValueError("tokenizer audit does not authorize the corrected regex setting")
    checkpoint_real = os.path.realpath(checkpoint_path)
    matches = [source for source in audit.get("sources", []) if (
        source.get("label") == label
        or (
            os.path.isabs(str(source.get("source", "")))
            and os.path.realpath(str(source["source"])) == checkpoint_real
        )
    )]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint tokenizer is not uniquely covered by audit: "
            f"label={label!r} checkpoint={checkpoint_real!r} matches={len(matches)}"
        )
    return {
        "fix_mistral_regex": True,
        "tokenizer_audit_path": os.path.realpath(audit_path),
        "tokenizer_audit_sha256": file_sha256(audit_path),
        "audited_source_label": matches[0]["label"],
        "tokenizer_files_combined_sha256": matches[0][
            "tokenizer_files_combined_sha256"
        ],
        "previous_ppl_rerun_required": bool(
            decision.get("previous_ppl_rerun_required", False)
        ),
    }
