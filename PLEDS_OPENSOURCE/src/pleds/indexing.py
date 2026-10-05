"""Shared software form of target CRC-based indexing."""

from __future__ import annotations
from typing import Callable, Tuple
from .hash_spec import direct_hash_value_fields, hash_index

SketchKey = Tuple[int, int, int, int, int]
IndexFunction = Callable[[SketchKey, int, int], int]


def target_index(key: SketchKey, hash_id: int, width: int) -> int:
    """Return the index used by the validated P4 hash profile."""

    if width <= 0:
        raise ValueError("width must be positive")
    if isinstance(key, tuple) and len(key) == 5:
        value = direct_hash_value_fields(key, hash_id)  # type: ignore[arg-type]
        return value % width
    return hash_index(str(key), salt=hash_id, table_size=width)
