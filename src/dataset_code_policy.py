"""Fail-closed authorization for benchmark dataset loading scripts."""
from __future__ import annotations


DATASET_CODE_TASKS = frozenset({"mathqa"})


def configure_dataset_code_trust(
    tasks: list[str], *, allow: bool, datasets_module=None,
) -> dict:
    """Authorize dataset loading scripts only for the fixed requested task set."""
    required_tasks = sorted(DATASET_CODE_TASKS.intersection(tasks))
    if required_tasks and not allow:
        raise ValueError(
            "dataset loading code authorization is required for tasks: "
            f"{required_tasks}; rerun with --trust-dataset-code"
        )
    if required_tasks:
        if datasets_module is None:
            import datasets as datasets_module
        datasets_module.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    return {
        "trust_dataset_code": bool(required_tasks and allow),
        "dataset_code_tasks": required_tasks,
    }
