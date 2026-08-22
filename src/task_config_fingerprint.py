"""Stable fingerprints for lm-eval task configurations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from typing import Any


FINGERPRINT_VERSION = "stable-json-v2"
_REPR_ADDRESS = re.compile(r"(?<= at )0x[0-9a-fA-F]+(?=>)")


def _normalize_string(value: str) -> str:
    """Remove only process-specific addresses from Python object repr strings."""
    return _REPR_ADDRESS.sub("0xADDR", value)


def stable_task_config(value: Any) -> Any:
    """Convert task config objects or saved JSON into a stable JSON value."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__float__": repr(value)}
        return value
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(key): stable_task_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [stable_task_config(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [stable_task_config(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if callable(value):
        # lm-eval persists callables with its JSON fallback as their repr.
        # Normalize the live object to that same representation so hashing a
        # runtime config and hashing the saved JSON config produce one digest.
        return _normalize_string(repr(value))
    # Match the persisted JSON fallback for opaque objects for the same reason.
    return _normalize_string(repr(value))


def task_config_sha256(value: Any) -> str:
    normalized = stable_task_config(value)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
