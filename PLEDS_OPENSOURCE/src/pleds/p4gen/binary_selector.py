"""Exact bounded binary-model mapping for application selector tables."""

from functools import lru_cache

import numpy as np

from pleds.models.plan_runtime import PersistedModelRuntime
from pleds.p4gen.ternary import lower_binary_rule_plan_to_ternary


def compact_binary_selector(plan: dict) -> dict:
    """Factor the complete truth table into disjoint cubes, without fitting."""
    count = int(plan["feature_count"])
    if count != 12:
        raise ValueError("compact selector mapping requires the 12-feature schema")
    for rule in plan.get("rules", []):
        if "mask" in rule and (
            len(rule["mask"]) != count or len(rule["value"]) != count
        ):
            raise ValueError("rule width differs from compact feature count")
    assignments = (
        (np.arange(1 << count)[:, None] >> np.arange(count - 1, -1, -1)) & 1
    ).astype(np.uint8)
    predictions = PersistedModelRuntime(plan).predict(assignments)
    if not np.all((predictions == 0) | (predictions == 1)):
        raise ValueError("selector outputs must be binary")

    @lru_cache(maxsize=None)
    def factor(values: bytes, feature: int):
        if not any(values):
            return ()
        if all(values):
            return (("0" * (count - feature), "0" * (count - feature)),)
        half = len(values) // 2
        left, right = values[:half], values[half:]
        if left == right:
            return tuple(
                ("0" + value, "0" + mask) for value, mask in factor(left, feature + 1)
            )
        return tuple(
            (bit + value, "1" + mask)
            for bit, child in (("0", left), ("1", right))
            for value, mask in factor(child, feature + 1)
        )

    cubes = factor(predictions.astype(np.uint8).tobytes(), 0)
    rules = [
        {"rule_id": i, "value": value, "mask": mask, "action": 1}
        for i, (value, mask) in enumerate(cubes)
    ]
    if not rules:
        rules = [{"rule_id": 0, "value": "0" * count, "mask": "0" * count, "action": 0}]
    mapping = "complete_binary_function_disjoint_cubes"
    if plan["format"] in {
        "pleds_rule_ir_v1",
        "pleds_lowered_decision_tree_v1",
    } and 0 < len(plan["rules"]) < len(rules):
        rules = [
            {
                "rule_id": entry.rule_id,
                "value": entry.value,
                "mask": entry.mask,
                "action": int(entry.action == "set_model_positive"),
            }
            for entry in lower_binary_rule_plan_to_ternary(plan)
        ]
        mapping = "exact_existing_priority_rules"
    mapped = {
        "format": "pleds_rule_ir_v1",
        "model_type": plan.get("model_type"),
        "source_format": plan["format"],
        "feature_count": count,
        "lowering": mapping,
        "rule_count": len(rules),
        "rules": rules,
    }
    if not np.array_equal(
        predictions, PersistedModelRuntime(mapped).predict(assignments)
    ):
        raise ValueError("exact binary mapping changed predictions")
    return mapped
