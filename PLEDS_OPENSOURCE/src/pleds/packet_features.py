"""Packet-derived features used by training and P4 inference."""

from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
from .key_bits import five_tuple_bit_matrix

FlowKey = Tuple[int, int, int, int, int]


def feature_matrix(keys: Sequence[FlowKey]) -> np.ndarray:
    """Vectorized software form of the 12 key-invariant P4 features."""

    values = np.asarray(keys, dtype=np.uint64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("keys must be five-tuples")
    src, dst, sport, dport, protocol = values.T
    l4_valid = (protocol == 6) | (protocol == 17)
    sport = np.where(l4_valid, sport, 0)
    dport = np.where(l4_valid, dport, 0)
    service_ports = np.asarray([22, 80, 443, 8080], dtype=np.uint64)
    return np.column_stack(
        [
            (src >> 31) & 1,
            (src >> 27) & 1,
            (dst >> 31) & 1,
            (dst >> 23) & 1,
            np.isin(sport, service_ports),
            np.isin(dport, service_ports),
            dport >= 1024,
            protocol == 6,
            protocol == 17,
            (src >> 23) & 1,
            (dst >> 27) & 1,
            (src >> 15) & 1,
        ]
    ).astype(np.uint32)


def deployment_feature_matrix(
    keys: Sequence[FlowKey],
    *,
    feature_count: int,
) -> np.ndarray:
    """Construct one of the feature schemas supported by generated P4."""

    if feature_count == 12:
        return feature_matrix(keys)
    if feature_count not in {16, 32, 64, 104}:
        raise ValueError(
            "packet features must use the 12-feature compact schema or a "
            "16/32/64/104-bit five-tuple prefix"
        )
    values = np.asarray(keys, dtype=np.uint64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("keys must be five-tuples")
    return five_tuple_bit_matrix(
        tuple(values[:, index] for index in range(values.shape[1]))
    )[:, :feature_count]
