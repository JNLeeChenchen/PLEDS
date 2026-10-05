"""Isolation-forest frontend lowered to a TCAM positive selector."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from .exact_match import predict_exact_patterns


@dataclass(frozen=True)
class IsolationForestReport:
    train_accuracy: float
    lowered_fidelity: float
    rule_count: int
    feature_count: int
    tree_count: int
    threshold_quantile: float
    score_threshold: float
    positive_train_coverage: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "rule_count": self.rule_count,
            "feature_count": self.feature_count,
            "tree_count": self.tree_count,
            "threshold_quantile": self.threshold_quantile,
            "score_threshold": self.score_threshold,
            "positive_train_coverage": self.positive_train_coverage,
        }


class LoweredIsolationForest:
    """Accepted one-class patterns represented as exact TCAM entries."""

    def __init__(
        self,
        positive_patterns: list[tuple[int, ...]],
        *,
        feature_count: int,
        source: dict[str, object],
    ):
        self.positive_patterns = positive_patterns
        self.positive_set = set(positive_patterns)
        self.feature_count = feature_count
        self.source = source

    @property
    def rule_count(self) -> int:
        return len(self.positive_patterns)

    def predict_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        pattern = tuple(int(value) for value in features)
        return int(pattern in self.positive_set)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return predict_exact_patterns(
            x,
            ((pattern, 1) for pattern in self.positive_patterns),
            feature_count=self.feature_count,
        )

    def to_model_plan(self) -> dict[str, object]:
        return {
            "format": "pleds_rule_ir_v1",
            "model_type": "isolation_forest",
            "lowering": "one_class_forest_positive_space_to_tcam",
            "feature_count": self.feature_count,
            "output_type": "binary",
            "rule_count": self.rule_count,
            "rules": [
                {
                    "rule_id": rule_id,
                    "action": 1,
                    "value": "".join(str(bit) for bit in pattern),
                    "mask": "1" * self.feature_count,
                }
                for rule_id, pattern in enumerate(self.positive_patterns)
            ],
            "source_model": self.source,
            "p4_codegen_status": "rule_ir_compile_ready",
            "notes": [
                "IsolationForest is trained as a one-class selector over positive/member examples.",
                "Accepted binary feature assignments are enumerated offline and lowered to RuleIR exact TCAM entries.",
                "This is a bounded feature-space lowering; wider schemas require direct tree-path lowering or resource-guided feature compression.",
            ],
        }


def _binary_feature_space(feature_count: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=feature_count)), dtype=np.int64)


def train_and_lower_isolation_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 75,
    threshold_quantile: float = 0.8,
    random_state: int = 7,
) -> tuple[IsolationForest, LoweredIsolationForest, IsolationForestReport]:
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1")
    positives = x_train[y_train == 1]
    if positives.size == 0:
        raise ValueError("isolation_forest lowering needs positive/member examples")
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(256, int(positives.shape[0])),
        contamination="auto",
        random_state=random_state,
    )
    clf.fit(positives)
    positive_scores = clf.score_samples(positives)
    score_threshold = float(np.quantile(positive_scores, threshold_quantile))

    feature_count = int(x_train.shape[1])
    feature_space = _binary_feature_space(feature_count)
    accepted = clf.score_samples(feature_space) >= score_threshold
    positive_patterns = [
        tuple(int(value) for value in row)
        for row, label in zip(feature_space, accepted)
        if bool(label)
    ]
    source = {
        "trainer": "sklearn.ensemble.IsolationForest",
        "tree_count": n_estimators,
        "threshold_quantile": threshold_quantile,
        "score_threshold": score_threshold,
        "training_mode": "one_class_positive_members",
    }
    lowered = LoweredIsolationForest(
        positive_patterns, feature_count=feature_count, source=source
    )
    teacher_pred = (clf.score_samples(x_train) >= score_threshold).astype(np.int64)
    lowered_pred = lowered.predict(x_train)
    report = IsolationForestReport(
        train_accuracy=float(accuracy_score(y_train, teacher_pred)),
        lowered_fidelity=float(accuracy_score(teacher_pred, lowered_pred)),
        rule_count=lowered.rule_count,
        feature_count=feature_count,
        tree_count=n_estimators,
        threshold_quantile=threshold_quantile,
        score_threshold=score_threshold,
        positive_train_coverage=int(np.sum((lowered_pred == 1) & (y_train == 1))),
    )
    return clf, lowered, report
