import numpy as np
import pytest

from pleds.models.plan_runtime import predict_model_plan
from pleds.models.registry import train_and_lower_candidate


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        ("decision_tree", {"max_depth": 3, "min_samples_leaf": 2}),
        (
            "random_forest_ensemble",
            {
                "n_estimators": 3,
                "max_depth": 2,
                "max_leaf_nodes": 4,
                "min_samples_leaf": 2,
            },
        ),
        (
            "xgboost_ensemble",
            {"n_estimators": 2, "max_depth": 1, "learning_rate": 0.25},
        ),
        ("rule_list", {"max_rules": 16}),
        ("naive_bayes_lookup", {"scale": 256}),
        ("piecewise_range", {"bucket_count": 3}),
        ("isolation_forest", {"n_estimators": 4, "threshold_quantile": 0.5}),
        (
            "tm_guided",
            {
                "number_of_clauses": 10,
                "threshold": 5,
                "specificity": 3.0,
                "epochs": 1,
                "max_rules": 8,
                "node_sample_size": 32,
                "max_depth": 2,
            },
        ),
    ],
)
def test_persisted_plan_matches_training_time_lowering(family, parameters) -> None:
    if family == "tm_guided":
        pytest.importorskip("tmu")
    if family == "xgboost_ensemble":
        pytest.importorskip("xgboost")
    rng = np.random.default_rng(17)
    x_train = rng.integers(0, 2, size=(96, 12), dtype=np.uint8)
    y_train = (x_train[:, 0] | (x_train[:, 1] & x_train[:, 2])).astype(np.uint8)
    x_eval = rng.integers(0, 2, size=(128, 12), dtype=np.uint8)
    candidate = train_and_lower_candidate(
        family,
        x_train,
        y_train,
        seed=17,
        parameters=parameters,
    )
    expected = candidate.lowered.predict(x_eval)
    scalar = np.asarray(
        [candidate.lowered.predict_one(row) for row in x_eval], dtype=np.int64
    )
    actual = predict_model_plan(candidate.model_plan, x_eval)
    np.testing.assert_array_equal(expected, scalar)
    np.testing.assert_array_equal(actual, expected)
