"""TM-guided distillation into PLEDS binary RuleIR."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score


def _entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    positive_fraction = float(np.mean(labels))
    if positive_fraction <= 0.0 or positive_fraction >= 1.0:
        return 0.0
    return -(
        positive_fraction * np.log2(positive_fraction)
        + (1.0 - positive_fraction) * np.log2(1.0 - positive_fraction)
    )


def _condition_mask(x: np.ndarray, conditions: dict[int, int]) -> np.ndarray:
    mask = np.ones(x.shape[0], dtype=bool)
    for feature, value in conditions.items():
        mask &= x[:, feature] == value
        if not np.any(mask):
            break
    return mask


@dataclass
class _Node:
    node_id: int
    conditions: dict[int, int]
    depth: int
    samples: np.ndarray
    labels: np.ndarray
    matching_indices: np.ndarray
    reach: float
    impurity: float
    best_split: int | None = None
    best_gain: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def priority(self) -> float:
        return self.reach * self.impurity


@dataclass
class _TeacherQueryTree:
    seed_rows: np.ndarray
    teacher_predict: Callable[[np.ndarray], np.ndarray]
    max_leaf_nodes: int
    node_sample_size: int
    min_samples_leaf: int
    random_state: int
    max_depth: int
    purity_threshold: float
    min_reach: float
    max_candidate_features: int | None
    rng: np.random.Generator = field(init=False)
    feature_probabilities: np.ndarray = field(init=False)
    next_node_id: int = 0
    teacher_queries: int = 0
    synthetic_rows: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        count = max(1, self.seed_rows.shape[0])
        self.feature_probabilities = (
            self.seed_rows.sum(axis=0, dtype=np.float64) + 0.5
        ) / (count + 1.0)
        self.feature_probabilities = np.clip(self.feature_probabilities, 0.01, 0.99)

    def _sample(
        self,
        conditions: dict[int, int],
        matching: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        if matching is None:
            matching = np.flatnonzero(_condition_mask(self.seed_rows, conditions))
        reach = float(matching.size) / float(max(1, self.seed_rows.shape[0]))
        parts: list[np.ndarray] = []
        if matching.size:
            take = min(self.node_sample_size, int(matching.size))
            parts.append(
                self.seed_rows[self.rng.choice(matching, size=take, replace=False)]
            )
        need = self.node_sample_size - sum(part.shape[0] for part in parts)
        if need > 0:
            base = self.rng.choice(self.seed_rows.shape[0], size=need, replace=True)
            synthetic = self.seed_rows[base].astype(np.uint8, copy=True)
            for feature, value in conditions.items():
                synthetic[:, feature] = value
            parts.append(synthetic)
            self.synthetic_rows += need
        return np.concatenate(parts).astype(np.uint8, copy=False), reach, matching

    def _candidate_features(self, node: _Node) -> np.ndarray:
        candidates = np.asarray(
            [
                feature
                for feature in range(self.seed_rows.shape[1])
                if feature not in node.conditions
            ],
            dtype=np.int64,
        )
        if (
            self.max_candidate_features
            and candidates.size > self.max_candidate_features
        ):
            candidates = self.rng.choice(
                candidates, size=self.max_candidate_features, replace=False
            )
            candidates.sort()
        return candidates

    def _terminal(self, node: _Node) -> bool:
        if node.reach <= self.min_reach or node.depth >= self.max_depth:
            return True
        if node.labels.size < 2 * self.min_samples_leaf:
            return True
        if not self._candidate_features(node).size:
            return True
        positive_fraction = float(np.mean(node.labels))
        return max(positive_fraction, 1.0 - positive_fraction) >= self.purity_threshold

    def _assign_split(self, node: _Node) -> None:
        if self._terminal(node):
            return
        best_feature = None
        best_gain = 0.0
        for feature in self._candidate_features(node):
            left = node.samples[:, feature] == 0
            right = ~left
            left_count = int(np.sum(left))
            right_count = int(np.sum(right))
            if (
                left_count < self.min_samples_leaf
                or right_count < self.min_samples_leaf
            ):
                continue
            weighted = left_count / node.labels.size * _entropy(node.labels[left])
            weighted += right_count / node.labels.size * _entropy(node.labels[right])
            gain = node.impurity - weighted
            if gain > best_gain + 1e-12:
                best_feature = int(feature)
                best_gain = float(gain)
        node.best_split = best_feature
        node.best_gain = best_gain

    def _make_node(
        self,
        conditions: dict[int, int],
        depth: int,
        matching: np.ndarray | None = None,
    ) -> _Node:
        samples, reach, matching = self._sample(conditions, matching)
        labels = np.asarray(self.teacher_predict(samples), dtype=np.uint32)
        self.teacher_queries += samples.shape[0]
        node = _Node(
            node_id=self.next_node_id,
            conditions=dict(conditions),
            depth=depth,
            samples=samples,
            labels=labels,
            matching_indices=matching,
            reach=reach,
            impurity=_entropy(labels),
        )
        self.next_node_id += 1
        self._assign_split(node)
        return node

    def fit(self) -> _Node:
        root = self._make_node({}, 0)
        queue: list[tuple[float, int, _Node]] = []
        if root.best_split is not None:
            heapq.heappush(queue, (-root.priority, root.node_id, root))
        leaf_count = 1
        while queue and leaf_count < self.max_leaf_nodes:
            _priority, _node_id, node = heapq.heappop(queue)
            if not node.is_leaf or node.best_split is None:
                continue
            feature = node.best_split
            left_conditions = dict(node.conditions)
            left_conditions[feature] = 0
            right_conditions = dict(node.conditions)
            right_conditions[feature] = 1
            matching_values = self.seed_rows[node.matching_indices, feature]
            node.left = self._make_node(
                left_conditions,
                node.depth + 1,
                node.matching_indices[matching_values == 0],
            )
            node.right = self._make_node(
                right_conditions,
                node.depth + 1,
                node.matching_indices[matching_values == 1],
            )
            leaf_count += 1
            for child in (node.left, node.right):
                if child.best_split is not None:
                    heapq.heappush(queue, (-child.priority, child.node_id, child))
        return root


def _leaves(node: _Node) -> list[_Node]:
    if node.is_leaf:
        return [node]
    result: list[_Node] = []
    if node.left is not None:
        result.extend(_leaves(node.left))
    if node.right is not None:
        result.extend(_leaves(node.right))
    return result


@dataclass(frozen=True)
class TmGuidedReport:
    teacher_train_accuracy: float
    distilled_train_accuracy: float
    fidelity: float
    rule_count: int
    feature_count: int
    tm_clauses: int
    tm_threshold: int
    tm_specificity: float
    tm_epochs: int
    member_score_threshold: int
    teacher_queries: int
    synthetic_rows: int
    tree_depth: int

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


class LoweredTmGuided:
    """Positive teacher-query leaves represented as ternary binary-feature rules."""

    def __init__(self, rules: list[dict[str, object]], *, feature_count: int):
        self.rules = rules
        self.feature_count = feature_count

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint8)
        predictions = np.zeros(x.shape[0], dtype=np.int64)
        for rule in self.rules:
            mask = np.fromiter((int(bit) for bit in str(rule["mask"])), dtype=np.uint8)
            value = np.fromiter(
                (int(bit) for bit in str(rule["value"])), dtype=np.uint8
            )
            selected = mask.astype(bool)
            if np.any(selected):
                predictions[np.all(x[:, selected] == value[selected], axis=1)] = 1
            else:
                predictions[:] = 1
        return predictions

    def predict_one(self, row: list[int] | tuple[int, ...] | np.ndarray) -> int:
        return int(self.predict(np.asarray(row, dtype=np.uint8).reshape(1, -1))[0])

    def to_model_plan(self) -> dict[str, object]:
        return {
            "format": "pleds_rule_ir_v1",
            "model_type": "tm_guided",
            "lowering": "tm_teacher_query_distillation",
            "feature_count": self.feature_count,
            "output_type": "binary",
            "rule_count": self.rule_count,
            "rules": self.rules,
            "p4_codegen_status": "rule_ir_compile_ready",
        }


def _rules_from_tree(
    root: _Node,
    *,
    feature_count: int,
    hot_leaf_threshold: float,
    min_reach: float,
) -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    for leaf in _leaves(root):
        positive_fraction = float(np.mean(leaf.labels)) if leaf.labels.size else 0.0
        if positive_fraction <= hot_leaf_threshold or leaf.reach <= min_reach:
            continue
        value = ["0"] * feature_count
        mask = ["0"] * feature_count
        for feature, bit in sorted(leaf.conditions.items()):
            value[feature] = str(bit)
            mask[feature] = "1"
        rules.append(
            {
                "rule_id": len(rules),
                "action": 1,
                "value": "".join(value),
                "mask": "".join(mask),
                "leaf_reach": leaf.reach,
                "leaf_hot_fraction": positive_fraction,
                "literal_count": len(leaf.conditions),
            }
        )
    return rules


def train_tm_teacher(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 7,
    number_of_clauses: int = 128,
    threshold: int = 32,
    specificity: float = 5.0,
    epochs: int = 5,
) -> TMClassifier:
    """Train the TM once so deployment operating points can share one teacher."""

    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError as exc:
        raise ImportError("TM training requires pip install 'pleds[tm]'") from exc

    x_train = np.ascontiguousarray(x_train, dtype=np.uint32)
    y_train = np.ascontiguousarray(y_train, dtype=np.uint32)
    if x_train.ndim != 2 or y_train.ndim != 1 or x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train have incompatible shapes")
    if x_train.shape[0] == 0 or not np.all((x_train == 0) | (x_train == 1)):
        raise ValueError(
            "TM-guided distillation requires a non-empty binary feature matrix"
        )
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise ValueError("TM-guided distillation requires both binary classes")

    teacher = TMClassifier(
        number_of_clauses=number_of_clauses,
        T=threshold,
        s=specificity,
        platform="CPU",
        weighted_clauses=True,
        feature_negation=True,
    )
    for epoch in range(epochs):
        np.random.seed(random_state + epoch)
        teacher.fit(x_train, y_train, shuffle=True)
    return teacher


def tm_member_scores(
    teacher: TMClassifier,
    x: np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Return the clipped member-class margin used to select an LBF threshold."""

    x = np.ascontiguousarray(x, dtype=np.uint32)
    if teacher.number_of_classes != 2:
        raise ValueError("binary scoring requires a two-class TM")
    for clause_bank in teacher.clause_banks:
        # TMU pickles NumPy state but not every CFFI pointer used by incremental
        # prediction. Rebuild those pointers when a persisted teacher is loaded.
        if clause_bank.incremental_clause_evaluation_initialized and not hasattr(
            clause_bank, "lcm_p"
        ):
            clause_bank.incremental_clause_evaluation_initialized = False
        clause_bank._cffi_init()
    weights = np.stack(
        [teacher.weight_banks[index].get_weights() for index in range(2)], axis=0
    ).astype(np.int64, copy=False)
    scores = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        transformed = teacher.transform(x[start:stop]).reshape(
            stop - start, 2, teacher.number_of_clauses
        )
        class_sums = np.sum(transformed * weights[None, :, :], axis=2)
        class_sums = np.clip(class_sums, -teacher.T, teacher.T)
        scores[start:stop] = class_sums[:, 1] - class_sums[:, 0]
    return scores


def lower_tm_guided_teacher(
    teacher: TMClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    member_score_threshold: int = 0,
    random_state: int = 7,
    epochs: int = 5,
    max_rules: int = 128,
    node_sample_size: int = 256,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
    purity_threshold: float = 0.995,
    hot_leaf_threshold: float = 0.5,
    min_reach: float = 0.0,
    max_candidate_features: int | None = None,
) -> tuple[LoweredTmGuided, TmGuidedReport]:
    """Extract one thresholded, hardware-bounded operating point from a TM."""

    x_train = np.ascontiguousarray(x_train, dtype=np.uint32)
    y_train = np.ascontiguousarray(y_train, dtype=np.uint32)

    def teacher_predict(rows: np.ndarray) -> np.ndarray:
        return (tm_member_scores(teacher, rows) > member_score_threshold).astype(
            np.uint32
        )

    tree = _TeacherQueryTree(
        seed_rows=x_train.astype(np.uint8),
        teacher_predict=teacher_predict,
        max_leaf_nodes=max(1, max_rules),
        node_sample_size=max(2, node_sample_size),
        min_samples_leaf=max(1, min_samples_leaf),
        random_state=random_state + 1009,
        max_depth=x_train.shape[1] if max_depth is None else max_depth,
        purity_threshold=purity_threshold,
        min_reach=min_reach,
        max_candidate_features=max_candidate_features,
    )
    root = tree.fit()
    rules = _rules_from_tree(
        root,
        feature_count=x_train.shape[1],
        hot_leaf_threshold=hot_leaf_threshold,
        min_reach=min_reach,
    )
    lowered = LoweredTmGuided(rules, feature_count=x_train.shape[1])
    teacher_predictions = teacher_predict(x_train).astype(np.int64)
    distilled_predictions = lowered.predict(x_train)
    report = TmGuidedReport(
        teacher_train_accuracy=float(accuracy_score(y_train, teacher_predictions)),
        distilled_train_accuracy=float(accuracy_score(y_train, distilled_predictions)),
        fidelity=float(accuracy_score(teacher_predictions, distilled_predictions)),
        rule_count=lowered.rule_count,
        feature_count=x_train.shape[1],
        tm_clauses=teacher.number_of_clauses,
        tm_threshold=teacher.T,
        tm_specificity=float(teacher.s),
        tm_epochs=epochs,
        member_score_threshold=int(member_score_threshold),
        teacher_queries=tree.teacher_queries,
        synthetic_rows=tree.synthetic_rows,
        tree_depth=max(leaf.depth for leaf in _leaves(root)),
    )
    return lowered, report


def train_and_lower_tm_guided(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 7,
    number_of_clauses: int = 128,
    threshold: int = 32,
    specificity: float = 5.0,
    epochs: int = 5,
    max_rules: int = 128,
    node_sample_size: int = 256,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
    purity_threshold: float = 0.995,
    hot_leaf_threshold: float = 0.5,
    min_reach: float = 0.0,
    max_candidate_features: int | None = None,
) -> tuple[TMClassifier, LoweredTmGuided, TmGuidedReport]:
    teacher = train_tm_teacher(
        x_train,
        y_train,
        random_state=random_state,
        number_of_clauses=number_of_clauses,
        threshold=threshold,
        specificity=specificity,
        epochs=epochs,
    )
    lowered, report = lower_tm_guided_teacher(
        teacher,
        x_train,
        y_train,
        member_score_threshold=0,
        random_state=random_state,
        epochs=epochs,
        max_rules=max_rules,
        node_sample_size=node_sample_size,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        purity_threshold=purity_threshold,
        hot_leaf_threshold=hot_leaf_threshold,
        min_reach=min_reach,
        max_candidate_features=max_candidate_features,
    )
    return teacher, lowered, report
