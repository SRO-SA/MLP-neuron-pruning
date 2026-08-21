"""Stable fingerprints for lm-eval task configurations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from typing import Any


FINGERPRINT_VERSION = "stable-json-v1"
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
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return {"__callable__": f"{module}.{name}"}
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return {"__type__": kind, "__repr__": _normalize_string(repr(value))}


def task_config_sha256(value: Any) -> str:
    normalized = stable_task_config(value)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
