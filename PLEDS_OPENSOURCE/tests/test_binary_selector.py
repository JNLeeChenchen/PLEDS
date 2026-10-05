import numpy as np
import pytest

from pleds.models.plan_runtime import PersistedModelRuntime
from pleds.p4gen.binary_selector import compact_binary_selector


def rule(action=1):
    return {"value": "100000000000", "mask": "100000000000", "action": action}


@pytest.mark.parametrize(
    "plan",
    [
        {"format": "pleds_rule_ir_v1", "rules": []},
        {
            "format": "pleds_rule_ir_v1",
            "rules": [{"value": "0" * 12, "mask": "0" * 12, "action": 1}],
        },
        {"format": "pleds_rule_ir_v1", "rules": [rule()]},
        {
            "format": "pleds_lowered_decision_tree_v1",
            "rules": [
                {"predicates": [{"feature": 2, "lower_exclusive": 0.5}], "action": 1}
            ],
        },
        {
            "format": "pleds_tree_ensemble_ir_v1",
            "trees": [{"rules": [rule()]}, {"rules": []}],
            "vote_threshold": 1,
        },
        {
            "format": "pleds_additive_score_ir_v1",
            "bias": -3,
            "threshold": 2,
            "feature_tables": [
                {"feature": i, "entries": [{"value": 1, "contribution": i - 4}]}
                for i in range(12)
            ],
        },
        {
            "format": "pleds_piecewise_range_ir_v1",
            "segments": [{"lower_inclusive": 3, "upper_inclusive": 7, "action": 1}],
        },
    ],
)
def test_exact_mapping_preserves_all_binary_inputs_and_disjointness(plan):
    plan = {**plan, "feature_count": 12}
    if "rules" in plan:
        plan["rules"] = [
            {"rule_id": index, **item} for index, item in enumerate(plan["rules"])
        ]
    mapped = compact_binary_selector(plan)
    inputs = ((np.arange(4096)[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8)
    assert np.array_equal(
        PersistedModelRuntime(plan).predict(inputs),
        PersistedModelRuntime(mapped).predict(inputs),
    )
    matches = np.zeros(4096, dtype=np.uint16)
    for item in mapped["rules"]:
        mask = np.array([int(bit) for bit in item["mask"]], dtype=bool)
        value = np.array([int(bit) for bit in item["value"]], dtype=np.uint8)
        matches += np.all(inputs[:, mask] == value[mask], axis=1)
    assert np.max(matches) <= 1


def test_mapping_keeps_compact_ordered_rules_when_cubes_expand():
    rules = [
        {
            "rule_id": i,
            "value": "0" * i + "1" + "0" * (11 - i),
            "mask": "0" * i + "1" + "0" * (11 - i),
            "action": i % 2,
        }
        for i in range(12)
    ]
    original = {"format": "pleds_rule_ir_v1", "feature_count": 12, "rules": rules}
    mapped = compact_binary_selector(original)
    assert len(mapped["rules"]) <= len(rules)
