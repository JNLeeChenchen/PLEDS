"""Binary features from the normalized IPv4 five-tuple."""

from __future__ import annotations

from typing import Sequence

import numpy as np


FIELD_WIDTHS: tuple[tuple[str, int], ...] = (
    ("src_addr", 32),
    ("dst_addr", 32),
    ("src_port", 16),
    ("dst_port", 16),
    ("protocol", 8),
)
FEATURE_COUNT = sum(width for _name, width in FIELD_WIDTHS)


def normalize_flow_key(
    key: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int, int]:
    """Apply the generated P4 parser's normalization to an integer five-tuple."""

    src_addr, dst_addr, src_port, dst_port, protocol = (int(value) for value in key)
    if protocol not in (6, 17):
        src_port = 0
        dst_port = 0
    return src_addr, dst_addr, src_port, dst_port, protocol


def five_tuple_bit_matrix(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Return the 104 network-order bits of each normalized five-tuple."""

    if len(columns) != len(FIELD_WIDTHS):
        raise ValueError(f"expected {len(FIELD_WIDTHS)} five-tuple columns")
    row_count = len(columns[0])
    if any(len(column) != row_count for column in columns):
        raise ValueError("five-tuple columns have different lengths")
    normalized = [np.asarray(column, dtype=np.uint64) for column in columns]
    protocol = normalized[4]
    l4_valid = (protocol == 6) | (protocol == 17)
    normalized[2] = np.where(l4_valid, normalized[2], 0)
    normalized[3] = np.where(l4_valid, normalized[3], 0)
    result = np.empty((row_count, FEATURE_COUNT), dtype=np.uint8)
    offset = 0
    for values, (_name, width) in zip(normalized, FIELD_WIDTHS):
        if values.size and int(values.max()) >= 1 << width:
            raise ValueError(f"field value exceeds bit<{width}>")
        shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
        result[:, offset : offset + width] = ((values[:, None] >> shifts) & 1).astype(
            np.uint8
        )
        offset += width
    return result


def feature_names() -> list[str]:
    names: list[str] = []
    for field, width in FIELD_WIDTHS:
        names.extend(f"{field}_bit_{bit}" for bit in range(width - 1, -1, -1))
    return names
