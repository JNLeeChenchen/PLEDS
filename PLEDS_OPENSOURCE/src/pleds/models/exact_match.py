"""Vectorized execution for exact-match lowered model rules."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence, Tuple

import numpy as np


def predict_exact_patterns(
    features: np.ndarray,
    pattern_actions: Iterable[Tuple[Sequence[int], int]],
    *,
    feature_count: int,
) -> np.ndarray:
    """Apply first-match exact rules with a default action of zero."""

    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] != feature_count:
        raise ValueError(
            f"exact-match model requires a two-dimensional {feature_count}-feature matrix"
        )
    first_actions = {}
    for pattern, action in pattern_actions:
        key = tuple(int(value) for value in pattern)
        if len(key) != feature_count:
            raise ValueError("exact-match rule has the wrong feature width")
        first_actions.setdefault(key, int(action))

    predictions = np.zeros(matrix.shape[0], dtype=np.int64)
    if not first_actions or matrix.shape[0] == 0:
        return predictions

    contiguous = np.ascontiguousarray(matrix)
    row_type = np.dtype((np.void, contiguous.dtype.itemsize * feature_count))
    row_keys = contiguous.view(row_type).reshape(-1)
    patterns_by_action = defaultdict(list)
    for pattern, action in first_actions.items():
        if action != 0:
            patterns_by_action[action].append(pattern)
    for action, patterns in patterns_by_action.items():
        pattern_matrix = np.ascontiguousarray(
            np.asarray(patterns, dtype=contiguous.dtype)
        )
        pattern_keys = pattern_matrix.view(row_type).reshape(-1)
        predictions[np.isin(row_keys, pattern_keys)] = action
    return predictions
