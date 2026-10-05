"""Decision-tree frontend and rule lowering for the PLEDS MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree


@dataclass(frozen=True)
class IntervalPredicate:
    feature: int
    lower_exclusive: float | None
    upper_inclusive: float | None

    def matches(self, value: float) -> bool:
        if self.lower_exclusive is not None and not value > self.lower_exclusive:
            return False
        if self.upper_inclusive is not None and not value <= self.upper_inclusive:
            return False
        return True

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "feature": self.feature,
            "lower_exclusive": self.lower_exclusive,
            "upper_inclusive": self.upper_inclusive,
        }


@dataclass(frozen=True)
class TreeRule:
    predicates: tuple[IntervalPredicate, ...]
    action: int

    def matches(self, features: list[int] | tuple[int, ...] | np.ndarray) -> bool:
        return all(
            pred.matches(float(features[pred.feature])) for pred in self.predicates
        )

    def as_dict(self, rule_id: int) -> dict[str, object]:
        return {
            "rule_id": rule_id,
            "action": self.action,
            "predicates": [predicate.as_dict() for predicate in self.predicates],
        }


class LoweredDecisionTree:
    def __init__(self, rules: list[TreeRule]):
        if not rules:
            raise ValueError("lowered decision tree needs at least one rule")
        self.rules = rules

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    @property
    def predicate_count(self) -> int:
        return sum(len(rule.predicates) for rule in self.rules)

    def predict_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        for rule in self.rules:
            if rule.matches(features):
                return rule.action
        return 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x)
        if matrix.ndim != 2:
            raise ValueError("decision-tree features must be a two-dimensional matrix")
        predictions = np.zeros(matrix.shape[0], dtype=np.int64)
        unresolved = np.ones(matrix.shape[0], dtype=bool)
        for rule in self.rules:
            selected = unresolved.copy()
            for predicate in rule.predicates:
                values = matrix[:, predicate.feature]
                if predicate.lower_exclusive is not None:
                    selected &= values > predicate.lower_exclusive
                if predicate.upper_inclusive is not None:
                    selected &= values <= predicate.upper_inclusive
            predictions[selected] = rule.action
            unresolved[selected] = False
            if not np.any(unresolved):
                break
        return predictions

    def to_rule_plan(self, *, feature_count: int) -> dict[str, object]:
        return {
            "format": "pleds_lowered_decision_tree_v1",
            "model_type": "decision_tree",
            "feature_count": feature_count,
            "output_type": "binary",
            "rule_count": self.rule_count,
            "predicate_count": self.predicate_count,
            "rules": [rule.as_dict(rule_id) for rule_id, rule in enumerate(self.rules)],
        }


@dataclass(frozen=True)
class DecisionTreeReport:
    train_accuracy: float
    lowered_fidelity: float
    rule_count: int
    predicate_count: int
    max_depth: int
    feature_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "rule_count": self.rule_count,
            "predicate_count": self.predicate_count,
            "max_depth": self.max_depth,
            "feature_count": self.feature_count,
        }


def train_decision_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_depth: int = 5,
    max_leaf_nodes: int | None = None,
    min_samples_leaf: int = 10,
    random_state: int = 7,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)
    return clf


def lower_tree(clf: DecisionTreeClassifier) -> LoweredDecisionTree:
    tree = clf.tree_
    rules: list[TreeRule] = []

    def walk(node_id: int, predicates: list[IntervalPredicate]) -> None:
        feature = tree.feature[node_id]
        if feature == _tree.TREE_UNDEFINED:
            counts = tree.value[node_id][0]
            action = int(np.argmax(counts))
            rules.append(TreeRule(predicates=tuple(predicates), action=action))
            return

        threshold = float(tree.threshold[node_id])
        left_predicates = predicates + [
            IntervalPredicate(
                feature=int(feature),
                lower_exclusive=None,
                upper_inclusive=threshold,
            )
        ]
        right_predicates = predicates + [
            IntervalPredicate(
                feature=int(feature),
                lower_exclusive=threshold,
                upper_inclusive=None,
            )
        ]
        walk(tree.children_left[node_id], left_predicates)
        walk(tree.children_right[node_id], right_predicates)

    walk(0, [])
    return LoweredDecisionTree(rules)


def train_and_lower_decision_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_depth: int = 5,
    max_leaf_nodes: int | None = None,
    min_samples_leaf: int = 10,
    random_state: int = 7,
) -> tuple[DecisionTreeClassifier, LoweredDecisionTree, DecisionTreeReport]:
    clf = train_decision_tree(
        x_train,
        y_train,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    lowered = lower_tree(clf)
    teacher_pred = clf.predict(x_train)
    lowered_pred = lowered.predict(x_train)
    report = DecisionTreeReport(
        train_accuracy=float(accuracy_score(y_train, teacher_pred)),
        lowered_fidelity=float(accuracy_score(teacher_pred, lowered_pred)),
        rule_count=lowered.rule_count,
        predicate_count=lowered.predicate_count,
        max_depth=int(clf.get_depth()),
        feature_count=int(x_train.shape[1]),
    )
    return clf, lowered, report
