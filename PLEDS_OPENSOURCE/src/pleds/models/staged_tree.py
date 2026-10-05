"""Level-by-level match-action representation for decision trees."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from .decision_tree import LoweredDecisionTree


@dataclass(frozen=True)
class StagedTreeNode:
    node_id: int
    level: int
    feature: int | None
    threshold: float | None
    left: int | None
    right: int | None
    action: int | None

    @property
    def is_leaf(self) -> bool:
        return self.action is not None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left,
            "right": self.right,
            "action": self.action,
        }


@dataclass(frozen=True)
class TreeRepresentationCost:
    representation: str
    entries: int
    key_bits: int
    metadata_bits: int
    stages: int

    def score(
        self,
        *,
        entry_weight: int = 1,
        key_bit_weight: int = 1,
        metadata_weight: int = 2,
        stage_weight: int = 512,
    ) -> int:
        return (
            self.entries * entry_weight
            + self.key_bits * key_bit_weight
            + self.metadata_bits * metadata_weight
            + self.stages * stage_weight
        )


class LoweredStagedDecisionTree:
    def __init__(self, nodes: tuple[StagedTreeNode, ...], *, feature_count: int):
        if not nodes or nodes[0].node_id != 0:
            raise ValueError("staged decision tree requires root node zero")
        self.nodes = nodes
        self.feature_count = feature_count
        self._by_id = {node.node_id: node for node in nodes}
        if len(self._by_id) != len(nodes):
            raise ValueError("staged decision-tree node ids must be unique")

    @property
    def max_depth(self) -> int:
        return max(node.level for node in self.nodes)

    @property
    def table_count(self) -> int:
        return max(1, self.max_depth)

    @property
    def entry_count(self) -> int:
        internal = sum(not node.is_leaf for node in self.nodes)
        leaves = sum(node.is_leaf for node in self.nodes)
        return internal * 2 + leaves

    @property
    def node_id_bits(self) -> int:
        return max(1, math.ceil(math.log2(len(self.nodes) + 1)))

    def predict_one(self, features: np.ndarray | list[int] | tuple[int, ...]) -> int:
        node = self._by_id[0]
        while not node.is_leaf:
            assert node.feature is not None
            assert node.threshold is not None
            next_id = (
                node.left
                if float(features[node.feature]) <= node.threshold
                else node.right
            )
            if next_id is None:
                raise ValueError(f"internal tree node {node.node_id} has no successor")
            node = self._by_id[next_id]
        assert node.action is not None
        return node.action

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x)
        if matrix.ndim != 2:
            raise ValueError("decision-tree features must be a two-dimensional matrix")
        return np.asarray([self.predict_one(row) for row in matrix], dtype=np.int64)

    def to_model_plan(self) -> dict[str, object]:
        levels = []
        for level in range(self.max_depth + 1):
            level_nodes = [node for node in self.nodes if node.level == level]
            levels.append(
                {
                    "level": level,
                    "features": sorted(
                        {
                            node.feature
                            for node in level_nodes
                            if node.feature is not None
                        }
                    ),
                    "nodes": [node.as_dict() for node in level_nodes],
                }
            )
        return {
            "format": "pleds_staged_tree_ir_v1",
            "model_type": "decision_tree",
            "selected_representation": "staged_nodes",
            "output_type": "binary",
            "feature_count": self.feature_count,
            "node_count": len(self.nodes),
            "entry_count": self.entry_count,
            "table_count": self.table_count,
            "node_id_bits": self.node_id_bits,
            "levels": levels,
        }


def lower_tree_to_stages(
    classifier: DecisionTreeClassifier, *, feature_count: int
) -> LoweredStagedDecisionTree:
    tree = classifier.tree_
    nodes: list[StagedTreeNode] = []

    def visit(node_id: int, level: int) -> None:
        feature = int(tree.feature[node_id])
        if feature == _tree.TREE_UNDEFINED:
            nodes.append(
                StagedTreeNode(
                    node_id=node_id,
                    level=level,
                    feature=None,
                    threshold=None,
                    left=None,
                    right=None,
                    action=int(np.argmax(tree.value[node_id][0])),
                )
            )
            return
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        nodes.append(
            StagedTreeNode(
                node_id=node_id,
                level=level,
                feature=feature,
                threshold=float(tree.threshold[node_id]),
                left=left,
                right=right,
                action=None,
            )
        )
        visit(left, level + 1)
        visit(right, level + 1)

    visit(0, 0)
    nodes.sort(key=lambda node: node.node_id)
    return LoweredStagedDecisionTree(tuple(nodes), feature_count=feature_count)


def representation_costs(
    rules: LoweredDecisionTree,
    staged: LoweredStagedDecisionTree,
) -> tuple[TreeRepresentationCost, TreeRepresentationCost]:
    rule_cost = TreeRepresentationCost(
        representation="root_to_leaf_rules",
        entries=rules.rule_count,
        key_bits=staged.feature_count,
        metadata_bits=1,
        stages=1,
    )
    level_key_bits = []
    for level in range(staged.max_depth):
        level_features = {
            node.feature
            for node in staged.nodes
            if node.level == level and node.feature is not None
        }
        level_key_bits.append(staged.node_id_bits + len(level_features))
    staged_cost = TreeRepresentationCost(
        representation="staged_nodes",
        entries=staged.entry_count,
        key_bits=max(level_key_bits, default=staged.node_id_bits),
        metadata_bits=staged.node_id_bits + 2,
        stages=staged.table_count,
    )
    return rule_cost, staged_cost


def select_tree_representation(
    rules: LoweredDecisionTree,
    staged: LoweredStagedDecisionTree,
    *,
    max_stages: int,
    max_entries: int,
    stage_weight: int = 512,
) -> str:
    feasible = [
        cost
        for cost in representation_costs(rules, staged)
        if cost.stages <= max_stages and cost.entries <= max_entries
    ]
    if not feasible:
        raise ValueError(
            "neither decision-tree representation fits the supplied limits"
        )
    return min(
        feasible,
        key=lambda cost: (
            cost.score(stage_weight=stage_weight),
            cost.stages,
            cost.entries,
        ),
    ).representation


__all__ = [
    "LoweredStagedDecisionTree",
    "StagedTreeNode",
    "TreeRepresentationCost",
    "lower_tree_to_stages",
    "representation_costs",
    "select_tree_representation",
]
