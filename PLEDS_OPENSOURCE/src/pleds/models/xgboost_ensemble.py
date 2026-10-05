"""Boosted-tree frontend lowered to a TCAM positive selector."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.metrics import accuracy_score

from .exact_match import predict_exact_patterns


@dataclass(frozen=True)
class XGBoostEnsembleReport:
    train_accuracy: float
    lowered_fidelity: float
    rule_count: int
    feature_count: int
    tree_count: int
    max_depth: int
    learning_rate: float
    trainer: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "rule_count": self.rule_count,
            "feature_count": self.feature_count,
            "tree_count": self.tree_count,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "trainer": self.trainer,
        }


class LoweredXGBoostEnsemble:
    """Positive boosted-tree decisions represented as exact TCAM entries."""

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
            "model_type": "xgboost_ensemble",
            "lowering": "boosted_tree_positive_space_to_tcam",
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
                "The deployed model is not distilled into a decision tree; boosted-tree decisions are enumerated over the bounded binary feature space and lowered to RuleIR.",
                "For wider feature schemas, this lowering must be replaced by a direct score-table backend or resource-guided feature compression.",
            ],
        }


def _binary_feature_space(feature_count: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=feature_count)), dtype=np.int64)


def train_and_lower_xgboost_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 7,
    max_depth: int = 2,
    learning_rate: float = 0.25,
    random_state: int = 7,
) -> tuple[XGBClassifier, LoweredXGBoostEnsemble, XGBoostEnsembleReport]:
    try:
        from xgboost import XGBClassifier, __version__ as xgboost_version
    except ImportError as exc:
        raise ImportError(
            "XGBoost training requires pip install 'pleds[xgboost]'"
        ) from exc
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
    )
    clf.fit(x_train, y_train)
    feature_count = int(x_train.shape[1])
    feature_space = _binary_feature_space(feature_count)
    teacher_space_pred = clf.predict(feature_space)
    positive_patterns = [
        tuple(int(value) for value in row)
        for row, label in zip(feature_space, teacher_space_pred)
        if int(label) == 1
    ]
    source = {
        "trainer": "xgboost.XGBClassifier",
        "xgboost_version": xgboost_version,
        "intended_family": "xgboost_ensemble",
        "tree_count": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }
    lowered = LoweredXGBoostEnsemble(
        positive_patterns, feature_count=feature_count, source=source
    )
    teacher_pred = clf.predict(x_train)
    lowered_pred = lowered.predict(x_train)
    report = XGBoostEnsembleReport(
        train_accuracy=float(accuracy_score(y_train, teacher_pred)),
        lowered_fidelity=float(accuracy_score(teacher_pred, lowered_pred)),
        rule_count=lowered.rule_count,
        feature_count=feature_count,
        tree_count=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        trainer=f"xgboost_{xgboost_version}",
    )
    return clf, lowered, report
