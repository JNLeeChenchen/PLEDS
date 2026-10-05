"""Rule-list frontend for TCAM-friendly Boolean models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score

from .exact_match import predict_exact_patterns


@dataclass(frozen=True)
class RuleListReport:
    train_accuracy: float
    lowered_fidelity: float
    rule_count: int
    feature_count: int
    min_precision: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "rule_count": self.rule_count,
            "feature_count": self.feature_count,
            "min_precision": self.min_precision,
        }


class LoweredRuleList:
    def __init__(self, rules: list[tuple[tuple[int, ...], int]], *, feature_count: int):
        self.rules = rules
        self.feature_count = feature_count

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    def predict_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        row = tuple(int(value) for value in features)
        for pattern, action in self.rules:
            if row == pattern:
                return action
        return 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return predict_exact_patterns(
            x,
            self.rules,
            feature_count=self.feature_count,
        )

    def to_model_plan(self) -> dict[str, object]:
        return {
            "format": "pleds_rule_ir_v1",
            "model_type": "rule_list",
            "lowering": "exact_binary_patterns_to_tcam",
            "feature_count": self.feature_count,
            "output_type": "binary",
            "rule_count": self.rule_count,
            "rules": [
                {
                    "rule_id": rule_id,
                    "action": action,
                    "value": "".join(str(bit) for bit in pattern),
                    "mask": "1" * self.feature_count,
                }
                for rule_id, (pattern, action) in enumerate(self.rules)
            ],
            "p4_codegen_status": "ir_only",
            "notes": [
                "This is a generic Boolean rule-list frontend.",
                "A Tsetlin Machine frontend can emit the same RuleIR using ternary masks.",
            ],
        }


def train_and_lower_rule_list(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_rules: int = 128,
    min_precision: float = 0.8,
) -> tuple[None, LoweredRuleList, RuleListReport]:
    positives: Counter[tuple[int, ...]] = Counter()
    totals: Counter[tuple[int, ...]] = Counter()
    for row, label in zip(x_train, y_train):
        pattern = tuple(int(value) for value in row)
        totals[pattern] += 1
        if int(label) == 1:
            positives[pattern] += 1

    candidates: list[tuple[tuple[int, ...], int, float]] = []
    for pattern, pos_count in positives.items():
        precision = pos_count / totals[pattern]
        if precision >= min_precision:
            candidates.append((pattern, pos_count, precision))
    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    rules = [(pattern, 1) for pattern, _count, _precision in candidates[:max_rules]]
    lowered = LoweredRuleList(rules, feature_count=int(x_train.shape[1]))
    lowered_pred = lowered.predict(x_train)
    report = RuleListReport(
        train_accuracy=float(accuracy_score(y_train, lowered_pred)),
        lowered_fidelity=1.0,
        rule_count=lowered.rule_count,
        feature_count=int(x_train.shape[1]),
        min_precision=min_precision,
    )
    return None, lowered, report
