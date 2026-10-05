"""Piecewise/range frontend for bucket-style PLEDS decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class PiecewiseRangeReport:
    train_accuracy: float
    lowered_fidelity: float
    feature_count: int
    bucket_count: int
    segment_count: int
    score_feature_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "feature_count": self.feature_count,
            "bucket_count": self.bucket_count,
            "segment_count": self.segment_count,
            "score_feature_count": self.score_feature_count,
        }


class LoweredPiecewiseRange:
    """Range table over a simple integer score.

    The first version uses `sum(features)` as the score. This is deliberately
    simple but captures the P4 shape needed by learned indexes and partitioned
    structures: range/LPM table -> segment/bucket/action.
    """

    def __init__(self, segments: list[tuple[int, int, int]], *, feature_count: int):
        if not segments:
            raise ValueError("piecewise range model needs at least one segment")
        self.segments = segments
        self.feature_count = feature_count

    @property
    def rule_count(self) -> int:
        return len(self.segments)

    def score_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        return int(sum(int(value) for value in features))

    def predict_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        score = self.score_one(features)
        for lower, upper, action in self.segments:
            if lower <= score <= upper:
                return action
        return 0

    def partition_one(
        self, features: list[int] | tuple[int, ...] | np.ndarray, partition_count: int
    ) -> int:
        if partition_count <= 0:
            raise ValueError("partition_count must be positive")
        score = self.score_one(features)
        for segment_id, (lower, upper, _action) in enumerate(self.segments):
            if lower <= score <= upper:
                return segment_id % partition_count
        return score % partition_count

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError(
                f"piecewise-range model requires a two-dimensional {self.feature_count}-feature matrix"
            )
        scores = np.sum(matrix, axis=1).astype(np.int64)
        predictions = np.zeros(matrix.shape[0], dtype=np.int64)
        unresolved = np.ones(matrix.shape[0], dtype=bool)
        for lower, upper, action in self.segments:
            selected = unresolved & (scores >= lower) & (scores <= upper)
            predictions[selected] = action
            unresolved[selected] = False
        return predictions

    def to_model_plan(self) -> dict[str, object]:
        return {
            "format": "pleds_piecewise_range_ir_v1",
            "model_type": "piecewise_range",
            "lowering": "integer_score_range_table",
            "feature_count": self.feature_count,
            "output_type": "binary_or_bucket",
            "score": "sum_binary_features",
            "rule_count": self.rule_count,
            "segments": [
                {"lower_inclusive": lower, "upper_inclusive": upper, "action": action}
                for lower, upper, action in self.segments
            ],
            "p4_codegen_status": "ir_only",
        }


def train_and_lower_piecewise_range(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    bucket_count: int = 4,
) -> tuple[None, LoweredPiecewiseRange, PiecewiseRangeReport]:
    scores = np.sum(x_train, axis=1).astype(np.int64)
    min_score = int(np.min(scores))
    max_score = int(np.max(scores))
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    boundaries = np.linspace(min_score, max_score + 1, bucket_count + 1, dtype=int)
    segments: list[tuple[int, int, int]] = []
    for idx in range(bucket_count):
        lower = int(boundaries[idx])
        upper = int(boundaries[idx + 1] - 1)
        mask = (scores >= lower) & (scores <= upper)
        if not np.any(mask):
            action = 0
        else:
            action = int(np.mean(y_train[mask]) >= 0.5)
        segments.append((lower, upper, action))
    lowered = LoweredPiecewiseRange(segments, feature_count=int(x_train.shape[1]))
    lowered_pred = lowered.predict(x_train)
    report = PiecewiseRangeReport(
        train_accuracy=float(accuracy_score(y_train, lowered_pred)),
        lowered_fidelity=1.0,
        feature_count=int(x_train.shape[1]),
        bucket_count=bucket_count,
        segment_count=lowered.rule_count,
        score_feature_count=int(x_train.shape[1]),
    )
    return None, lowered, report
