"""Train hotness predictors using counts from training windows only."""

from __future__ import annotations
import numpy as np
from .packet_features import deployment_feature_matrix

FLOW_KEY_DTYPE = np.dtype(
    [
        ("src_addr", "<u4"),
        ("dst_addr", "<u4"),
        ("src_port", "<u2"),
        ("dst_port", "<u2"),
        ("protocol", "u1"),
    ],
    align=False,
)


def structured_feature_matrix(
    keys: np.ndarray,
    *,
    feature_count: int = 104,
) -> np.ndarray:
    fields = tuple(keys[name] for name in FLOW_KEY_DTYPE.names or ())
    return deployment_feature_matrix(
        np.column_stack(fields), feature_count=feature_count
    )


def sample_hotness_training_rows(
    keys: np.ndarray,
    counts: np.ndarray,
    *,
    hot_fraction: float,
    max_per_class: int,
    seed: int,
    feature_count: int = 104,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    if not 0.0 < hot_fraction < 1.0:
        raise ValueError("hot_fraction must lie in (0, 1)")
    if keys.size != counts.size or keys.size < 2:
        raise ValueError("hotness training requires at least two flows")
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    order = np.argsort(-counts.astype(np.int64), kind="stable")
    hot_count = max(1, min(keys.size - 1, int(round(keys.size * hot_fraction))))
    positive = order[:hot_count]
    negative = order[hot_count:]
    rng = np.random.default_rng(seed)
    if positive.size > max_per_class:
        positive = np.sort(rng.choice(positive, size=max_per_class, replace=False))
    if negative.size > max_per_class:
        negative = np.sort(rng.choice(negative, size=max_per_class, replace=False))
    selected = np.concatenate([positive, negative])
    labels = np.concatenate(
        [
            np.ones(positive.size, dtype=np.uint32),
            np.zeros(negative.size, dtype=np.uint32),
        ]
    )
    return (
        structured_feature_matrix(keys[selected], feature_count=feature_count),
        labels,
        {
            "hot_fraction": hot_fraction,
            "hot_population": hot_count,
            "positive_training_rows": int(positive.size),
            "negative_training_rows": int(negative.size),
            "minimum_hot_count": int(counts[order[hot_count - 1]]),
            "feature_count": feature_count,
        },
    )
