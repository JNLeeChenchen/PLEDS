"""Execute persisted PLEDS lowered-model plans without retraining."""

from __future__ import annotations

from typing import Mapping

import numpy as np


class PersistedModelRuntime:
    """Prediction interface backed only by a persisted deployable model plan."""

    def __init__(self, plan: Mapping[str, object]) -> None:
        self.plan = dict(plan)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return predict_model_plan(self.plan, features)

    def predict_one(self, features: np.ndarray) -> int:
        matrix = np.asarray(features).reshape(1, -1)
        return int(self.predict(matrix)[0])


def _feature_matrix(plan: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("model-plan features must be a two-dimensional matrix")
    expected = int(plan["feature_count"])
    if matrix.shape[1] != expected:
        raise ValueError(
            f"model plan requires {expected} features, received {matrix.shape[1]}"
        )
    return matrix


def _rule_matches(rule: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    if "mask" in rule:
        mask = np.fromiter((int(bit) for bit in str(rule["mask"])), dtype=np.uint8)
        value = np.fromiter((int(bit) for bit in str(rule["value"])), dtype=np.uint8)
        selected = mask.astype(bool)
        if not np.any(selected):
            return np.ones(features.shape[0], dtype=bool)
        return np.all(features[:, selected] == value[selected], axis=1)

    matches = np.ones(features.shape[0], dtype=bool)
    for predicate in rule.get("predicates", []):
        feature = int(predicate["feature"])
        values = features[:, feature]
        lower = predicate.get("lower_exclusive")
        upper = predicate.get("upper_inclusive")
        if lower is not None:
            matches &= values > float(lower)
        if upper is not None:
            matches &= values <= float(upper)
    return matches


def _predict_rule_plan(plan: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    predictions = np.zeros(features.shape[0], dtype=np.int64)
    unresolved = np.ones(features.shape[0], dtype=bool)
    for rule in plan.get("rules", []):
        selected = unresolved & _rule_matches(rule, features)
        predictions[selected] = int(rule["action"])
        unresolved[selected] = False
    return predictions


def _predict_tree_ensemble(
    plan: Mapping[str, object], features: np.ndarray
) -> np.ndarray:
    trees = list(plan.get("trees", []))
    if not trees:
        raise ValueError("tree-ensemble plan contains no trees")
    votes = np.zeros(features.shape[0], dtype=np.int64)
    for tree in trees:
        votes += _predict_rule_plan(tree, features)
    return (votes >= int(plan["vote_threshold"])).astype(np.int64)


def _predict_additive_score(
    plan: Mapping[str, object], features: np.ndarray
) -> np.ndarray:
    scores = np.full(features.shape[0], int(plan["bias"]), dtype=np.int64)
    tables = list(plan.get("feature_tables", []))
    if len(tables) != features.shape[1]:
        raise ValueError("additive-score plan has an incomplete feature table set")
    for table in tables:
        feature = int(table["feature"])
        values = features[:, feature]
        for entry in table.get("entries", []):
            scores[values == int(entry["value"])] += int(entry["contribution"])
    return (scores >= int(plan["threshold"])).astype(np.int64)


def _predict_piecewise_range(
    plan: Mapping[str, object], features: np.ndarray
) -> np.ndarray:
    scores = np.sum(features, axis=1).astype(np.int64)
    predictions = np.zeros(features.shape[0], dtype=np.int64)
    unresolved = np.ones(features.shape[0], dtype=bool)
    for segment in plan.get("segments", []):
        selected = (
            unresolved
            & (scores >= int(segment["lower_inclusive"]))
            & (scores <= int(segment["upper_inclusive"]))
        )
        predictions[selected] = int(segment["action"])
        unresolved[selected] = False
    return predictions


def predict_model_plan(plan: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    """Return exact lowered-model decisions for a persisted model plan."""

    matrix = _feature_matrix(plan, features)
    plan_format = str(plan.get("format"))
    if plan_format in {"pleds_rule_ir_v1", "pleds_lowered_decision_tree_v1"}:
        return _predict_rule_plan(plan, matrix)
    if plan_format == "pleds_tree_ensemble_ir_v1":
        return _predict_tree_ensemble(plan, matrix)
    if plan_format == "pleds_additive_score_ir_v1":
        return _predict_additive_score(plan, matrix)
    if plan_format == "pleds_piecewise_range_ir_v1":
        return _predict_piecewise_range(plan, matrix)
    raise ValueError(f"unsupported persisted model-plan format: {plan_format}")
