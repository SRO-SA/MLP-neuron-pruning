"""CPU-only helpers for interpreting systems profiler evidence."""
from __future__ import annotations


def shape_integers(value) -> set[int]:
    """Collect all integer dimensions from a nested profiler-shape value."""
    found = set()
    if isinstance(value, (list, tuple)):
        for item in value:
            found.update(shape_integers(item))
    elif isinstance(value, int):
        found.add(value)
    return found
