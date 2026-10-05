"""Tree-ensemble frontend lowering for PLEDS.

This module implements a direct, Planter-style RF frontend at the IR/simulator
level. P4 code generation is intentionally separate: the lowered artifact records
feature-code and per-tree leaf-table structure so a later generator can emit the
TNA tables without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .decision_tree import LoweredDecisionTree, lower_tree


@dataclass(frozen=True)
class TreeEnsembleReport:
    train_accuracy: float
    lowered_fidelity: float
    tree_count: int
    total_rule_count: int
    total_predicate_count: int
    max_depth: int
    max_leaf_nodes: int | None
    feature_count: int
    vote_threshold: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "train_accuracy": self.train_accuracy,
            "lowered_fidelity": self.lowered_fidelity,
            "tree_count": self.tree_count,
            "total_rule_count": self.total_rule_count,
            "total_predicate_count": self.total_predicate_count,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "feature_count": self.feature_count,
            "vote_threshold": self.vote_threshold,
        }


class LoweredTreeEnsemble:
    """Bounded binary tree ensemble with majority-vote output."""

    def __init__(self, trees: list[LoweredDecisionTree], *, vote_threshold: int):
        if not trees:
            raise ValueError("lowered tree ensemble needs at least one tree")
        if vote_threshold < 1 or vote_threshold > len(trees):
            raise ValueError("vote_threshold must be within the tree count")
        self.trees = trees
        self.vote_threshold = vote_threshold

    @property
    def tree_count(self) -> int:
        return len(self.trees)

    @property
    def rule_count(self) -> int:
        return sum(tree.rule_count for tree in self.trees)

    @property
    def predicate_count(self) -> int:
        return sum(tree.predicate_count for tree in self.trees)

    def predict_one(self, features: list[int] | tuple[int, ...] | np.ndarray) -> int:
        votes = sum(tree.predict_one(features) for tree in self.trees)
        return int(votes >= self.vote_threshold)

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x)
        if matrix.ndim != 2:
            raise ValueError("tree-ensemble features must be a two-dimensional matrix")
        votes = np.zeros(matrix.shape[0], dtype=np.int64)
        for tree in self.trees:
            votes += tree.predict(matrix)
        return (votes >= self.vote_threshold).astype(np.int64)

    def to_model_plan(self, *, feature_count: int) -> dict[str, object]:
        return {
            "format": "pleds_tree_ensemble_ir_v1",
            "model_type": "random_forest_ensemble",
            "lowering": "direct_tree_ensemble",
            "p4_mapping": "feature_code_tables_then_per_tree_leaf_tables_then_vote_threshold",
            "feature_count": feature_count,
            "output_type": "binary",
            "tree_count": self.tree_count,
            "vote_threshold": self.vote_threshold,
            "total_rule_count": self.rule_count,
            "total_predicate_count": self.predicate_count,
            "trees": [
                {
                    **tree.to_rule_plan(feature_count=feature_count),
                    "tree_id": tree_id,
                }
                for tree_id, tree in enumerate(self.trees)
            ],
            "p4_codegen_status": "ir_only",
            "notes": [
                "This is a direct ensemble artifact, not a distilled decision tree.",
                "A Tofino generator should lower feature predicates into bounded feature-code tables.",
            ],
        }


def train_and_lower_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 5,
    max_depth: int = 3,
    max_leaf_nodes: int | None = 8,
    min_samples_leaf: int = 10,
    random_state: int = 7,
) -> tuple[RandomForestClassifier, LoweredTreeEnsemble, TreeEnsembleReport]:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)
    lowered_trees = [lower_tree(tree) for tree in clf.estimators_]
    vote_threshold = (n_estimators // 2) + 1
    lowered = LoweredTreeEnsemble(lowered_trees, vote_threshold=vote_threshold)
    teacher_pred = clf.predict(x_train)
    lowered_pred = lowered.predict(x_train)
    report = TreeEnsembleReport(
        train_accuracy=float(accuracy_score(y_train, teacher_pred)),
        lowered_fidelity=float(accuracy_score(teacher_pred, lowered_pred)),
        tree_count=n_estimators,
        total_rule_count=lowered.rule_count,
        total_predicate_count=lowered.predicate_count,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        feature_count=int(x_train.shape[1]),
        vote_threshold=vote_threshold,
    )
    return clf, lowered, report
