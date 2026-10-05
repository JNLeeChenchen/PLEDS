import pickle

import numpy as np
import pytest

pytest.importorskip("tmu")

from pleds.models.tm_guided import (
    _TeacherQueryTree,
    _condition_mask,
    tm_member_scores,
    train_and_lower_tm_guided,
    train_tm_teacher,
)
from pleds.p4gen.ternary import lower_binary_rule_plan_to_ternary


def test_tm_guided_trains_real_teacher_and_emits_rule_ir() -> None:
    patterns = np.array(
        [[(value >> bit) & 1 for bit in range(3, -1, -1)] for value in range(16)],
        dtype=np.uint32,
    )
    x_train = np.repeat(patterns, 8, axis=0)
    y_train = ((x_train[:, 0] & x_train[:, 1]) | x_train[:, 2]).astype(np.uint32)

    teacher, lowered, report = train_and_lower_tm_guided(
        x_train,
        y_train,
        random_state=11,
        number_of_clauses=32,
        threshold=12,
        specificity=3.0,
        epochs=3,
        max_rules=16,
        node_sample_size=64,
    )
    teacher_prediction = teacher.predict(x_train).astype(np.int64)
    plan = lowered.to_model_plan()
    entries = lower_binary_rule_plan_to_ternary(plan)

    assert plan["model_type"] == "tm_guided"
    assert plan["rule_count"] == len(entries) == report.rule_count
    assert np.mean(teacher_prediction == lowered.predict(x_train)) == report.fidelity
    assert 0.0 <= report.teacher_train_accuracy <= 1.0
    assert np.array_equal(
        teacher_prediction, (tm_member_scores(teacher, x_train) > 0).astype(int)
    )


def test_tm_teacher_scores_survive_pickle_round_trip() -> None:
    x_train = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]] * 8,
        dtype=np.uint32,
    )
    y_train = (x_train[:, 0] | x_train[:, 1]).astype(np.uint32)
    teacher = train_tm_teacher(
        x_train,
        y_train,
        random_state=13,
        number_of_clauses=16,
        threshold=8,
        specificity=3.0,
        epochs=2,
    )
    before = tm_member_scores(teacher, x_train)
    restored = pickle.loads(pickle.dumps(teacher))
    after = tm_member_scores(restored, x_train)
    np.testing.assert_array_equal(after, before)


def test_teacher_query_tree_reuses_exact_parent_matches() -> None:
    rows = np.array(
        [[(value >> bit) & 1 for bit in range(5, -1, -1)] for value in range(64)],
        dtype=np.uint8,
    )
    tree = _TeacherQueryTree(
        seed_rows=rows,
        teacher_predict=lambda values: (values[:, 0] | values[:, 3]).astype(np.uint32),
        max_leaf_nodes=8,
        node_sample_size=16,
        min_samples_leaf=2,
        random_state=17,
        max_depth=6,
        purity_threshold=0.995,
        min_reach=0.0,
        max_candidate_features=None,
    )
    root = tree.fit()
    pending = [root]
    while pending:
        node = pending.pop()
        expected = np.flatnonzero(_condition_mask(rows, node.conditions))
        np.testing.assert_array_equal(node.matching_indices, expected)
        pending.extend(child for child in (node.left, node.right) if child is not None)
