"""Additive lookup-score model frontends for PLEDS."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB


@dataclass(frozen=True)
class AdditiveScoreReport:
    train_accuracy: float
    lowered_fidelity: float
    feature_count: int
    contribution_count: int
    score_bits: int
    scale: int
    model_type: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "feature_count": self.feature_count,
            "contribution_count": self.contribution_count,
            "score_bits": self.score_bits,
            "scale": self.scale,
            "model_type": self.model_type,
        }


class LoweredAdditiveScore:
    def __init__(
        self,
        *,
        model_type: str,
        contribution_tables: list[dict[int, int]],
        bias: int,
        threshold: int = 0,
        scale: int = 1024,
    ):
        if not contribution_tables:
            raise ValueError("additive score model needs at least one feature table")
        self.model_type = model_type
        self.contribution_tables = contribution_tables
        self.bias = bias
        self.threshold = threshold
        self.scale = scale

    @property
    def feature_count(self) -> int:
        return len(self.contribution_tables)

    @property
    def contribution_count(self) -> int:
        return sum(len(table) for table in self.contribution_tables)

    @property
    def max_abs_score(self) -> int:
        total = abs(self.bias) + abs(self.threshold)
        for table in self.contribution_tables:
            total += max(abs(value) for value in table.values())
        return max(1, total)

    @property
    def score_bits(self) -> int:
        return int(math.ceil(math.log2(self.max_abs_score + 1))) + 1

    def score_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        score = self.bias
        for idx, table in enumerate(self.contribution_tables):
            score += table.get(int(features[idx]), 0)
        return score

    def predict_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        return int(self.score_one(features) >= self.threshold)

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError(
                f"additive-score model requires a two-dimensional {self.feature_count}-feature matrix"
            )
        scores = np.full(matrix.shape[0], self.bias, dtype=np.int64)
        for feature, table in enumerate(self.contribution_tables):
            values = matrix[:, feature]
            for value, contribution in table.items():
                scores[values == value] += contribution
        return (scores >= self.threshold).astype(np.int64)

    def to_model_plan(self) -> dict[str, object]:
        return {
            "format": "pleds_additive_score_ir_v1",
            "model_type": self.model_type,
            "lowering": "lookup_contribution_then_integer_threshold",
            "feature_count": self.feature_count,
            "output_type": "binary",
            "bias": self.bias,
            "threshold": self.threshold,
            "scale": self.scale,
            "score_bits": self.score_bits,
            "contribution_count": self.contribution_count,
            "feature_tables": [
                {
                    "feature": idx,
                    "entries": [
                        {"value": value, "contribution": contribution}
                        for value, contribution in sorted(table.items())
                    ],
                }
                for idx, table in enumerate(self.contribution_tables)
            ],
            "p4_codegen_status": "ir_only",
        }


def train_and_lower_bernoulli_nb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    scale: int = 1024,
) -> tuple[BernoulliNB, LoweredAdditiveScore, AdditiveScoreReport]:
    clf = BernoulliNB()
    clf.fit(x_train, y_train)
    classes = list(clf.classes_)
    if classes != [0, 1]:
        raise ValueError(
            f"BernoulliNB lowering expects binary classes [0, 1], got {classes}"
        )

    log_prob0 = clf.feature_log_prob_[0]
    log_prob1 = clf.feature_log_prob_[1]
    bias = int(round(float(clf.class_log_prior_[1] - clf.class_log_prior_[0]) * scale))
    tables: list[dict[int, int]] = []
    for p0_log, p1_log in zip(log_prob0, log_prob1):
        p0 = math.exp(float(p0_log))
        p1 = math.exp(float(p1_log))
        contribution_if_1 = float(p1_log - p0_log)
        contribution_if_0 = math.log(max(1e-12, 1.0 - p1)) - math.log(
            max(1e-12, 1.0 - p0)
        )
        tables.append(
            {
                0: int(round(contribution_if_0 * scale)),
                1: int(round(contribution_if_1 * scale)),
            }
        )
    lowered = LoweredAdditiveScore(
        model_type="naive_bayes_lookup",
        contribution_tables=tables,
        bias=bias,
        threshold=0,
        scale=scale,
    )
    teacher_pred = clf.predict(x_train)
    lowered_pred = lowered.predict(x_train)
    report = AdditiveScoreReport(
        train_accuracy=float(accuracy_score(y_train, teacher_pred)),
        lowered_fidelity=float(accuracy_score(teacher_pred, lowered_pred)),
        feature_count=int(x_train.shape[1]),
        contribution_count=lowered.contribution_count,
        score_bits=lowered.score_bits,
        scale=scale,
        model_type="naive_bayes_lookup",
    )
    return clf, lowered, report
