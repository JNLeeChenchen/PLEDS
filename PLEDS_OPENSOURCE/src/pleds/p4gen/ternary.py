"""Ternary lowering utilities for selector tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TernaryEntry:
    rule_id: int
    value: str
    mask: str
    action: str
    priority: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "rule_id": self.rule_id,
            "value": self.value,
            "mask": self.mask,
            "action": self.action,
            "priority": self.priority,
        }


def active_rule_features(rule_plan: dict[str, object]) -> tuple[int, ...]:
    """Return feature indices referenced by at least one deployed rule."""

    feature_count = int(rule_plan["feature_count"])
    active: set[int] = set()
    for raw_rule in rule_plan.get("rules", []):
        if not isinstance(raw_rule, dict):
            raise TypeError("rule entry must be a mapping")
        if "mask" in raw_rule:
            mask = str(raw_rule["mask"])
            if len(mask) != feature_count:
                raise ValueError(f"rule mask width must be {feature_count}: {raw_rule}")
            active.update(index for index, bit in enumerate(mask) if bit == "1")
        else:
            for predicate in raw_rule.get("predicates", []):
                if not isinstance(predicate, dict):
                    raise TypeError("predicate must be a mapping")
                active.add(int(predicate["feature"]))
    if any(index < 0 or index >= feature_count for index in active):
        raise ValueError("active feature index is outside the model feature width")
    return tuple(sorted(active))


def _predicate_to_bit(predicate: dict[str, object]) -> int | None:
    """Convert a binary-feature interval predicate into a concrete bit.

    sklearn trees split binary 0/1 features at threshold 0.5. The left branch is
    <= 0.5 and means bit 0; the right branch is > 0.5 and means bit 1.
    """

    lower = predicate.get("lower_exclusive")
    upper = predicate.get("upper_inclusive")
    if lower is None and upper is not None and float(upper) >= 0.5:
        return 0
    if lower is not None and upper is None and float(lower) <= 0.5:
        return 1
    return None


def _entries_overlap(left: TernaryEntry, right: TernaryEntry) -> bool:
    return not any(
        lmask == rmask == "1" and lvalue != rvalue
        for lvalue, lmask, rvalue, rmask in zip(
            left.value, left.mask, right.value, right.mask
        )
    )


def _merge_pair(left: TernaryEntry, right: TernaryEntry) -> TernaryEntry | None:
    if left.action != right.action or left.mask != right.mask:
        return None
    differing = [
        index
        for index, (lvalue, rvalue, mask) in enumerate(
            zip(left.value, right.value, left.mask)
        )
        if mask == "1" and lvalue != rvalue
    ]
    if len(differing) != 1:
        return None
    index = differing[0]
    value = list(left.value)
    mask = list(left.mask)
    value[index] = "0"
    mask[index] = "0"
    return TernaryEntry(
        rule_id=min(left.rule_id, right.rule_id),
        value="".join(value),
        mask="".join(mask),
        action=left.action,
        priority=max(left.priority, right.priority),
    )


def optimize_ternary_entries(entries: list[TernaryEntry]) -> list[TernaryEntry]:
    """Merge same-action cubes when priority-dependent behavior is unchanged."""

    optimized = list(entries)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(optimized):
            for right_index in range(left_index + 1, len(optimized)):
                right = optimized[right_index]
                merged = _merge_pair(left, right)
                if merged is None:
                    continue
                opposite = [
                    entry for entry in optimized if entry.action != merged.action
                ]
                if any(_entries_overlap(merged, entry) for entry in opposite):
                    continue
                optimized = [
                    entry
                    for index, entry in enumerate(optimized)
                    if index not in {left_index, right_index}
                ]
                optimized.append(merged)
                optimized.sort(key=lambda entry: (-entry.priority, entry.rule_id))
                changed = True
                break
            if changed:
                break

    unique: dict[tuple[str, str, str], TernaryEntry] = {}
    for entry in optimized:
        key = (entry.value, entry.mask, entry.action)
        current = unique.get(key)
        if current is None or entry.priority > current.priority:
            unique[key] = entry
    return sorted(unique.values(), key=lambda entry: (-entry.priority, entry.rule_id))


def optimized_rule_plan(rule_plan: dict[str, object]) -> dict[str, object]:
    """Return a canonical RuleIR containing the safely merged ternary entries."""

    entries = lower_binary_rule_plan_to_ternary(rule_plan, optimize=True)
    payload = {
        key: value
        for key, value in rule_plan.items()
        if key not in {"format", "rules", "rule_count", "predicate_count"}
    }
    payload.update(
        {
            "format": "pleds_rule_ir_v1",
            "source_format": rule_plan.get("format"),
            "feature_count": int(rule_plan["feature_count"]),
            "rule_count": len(entries),
            "rules": [
                {
                    "rule_id": index,
                    "value": entry.value,
                    "mask": entry.mask,
                    "action": 1 if entry.action == "set_model_positive" else 0,
                }
                for index, entry in enumerate(entries)
            ],
        }
    )
    return payload


def lower_binary_rule_plan_to_ternary(
    rule_plan: dict[str, object], *, optimize: bool = False
) -> list[TernaryEntry]:
    feature_count = int(rule_plan["feature_count"])
    entries: list[TernaryEntry] = []
    rules = rule_plan.get("rules", [])
    if not isinstance(rules, list):
        raise TypeError("rule_plan.rules must be a list")

    if rule_plan.get("format") == "pleds_rule_ir_v1":
        for priority, raw_rule in enumerate(rules):
            if not isinstance(raw_rule, dict):
                raise TypeError("rule entry must be a mapping")
            value = str(raw_rule["value"])
            mask = str(raw_rule["mask"])
            if len(value) != feature_count or len(mask) != feature_count:
                raise ValueError(
                    f"rule value/mask width must be {feature_count}: {raw_rule}"
                )
            entries.append(
                TernaryEntry(
                    rule_id=int(raw_rule["rule_id"]),
                    value=value,
                    mask=mask,
                    action=(
                        "set_model_positive"
                        if int(raw_rule["action"]) == 1
                        else "set_model_negative"
                    ),
                    priority=len(rules) - priority,
                )
            )
        return optimize_ternary_entries(entries) if optimize else entries

    for priority, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, dict):
            raise TypeError("rule entry must be a mapping")
        value = ["0"] * feature_count
        mask = ["0"] * feature_count
        rule_id = int(raw_rule["rule_id"])
        action_id = int(raw_rule["action"])
        predicates = raw_rule.get("predicates", [])
        if not isinstance(predicates, list):
            raise TypeError("rule.predicates must be a list")

        for raw_predicate in predicates:
            if not isinstance(raw_predicate, dict):
                raise TypeError("predicate must be a mapping")
            feature = int(raw_predicate["feature"])
            bit = _predicate_to_bit(raw_predicate)
            if bit is None:
                raise ValueError(
                    f"cannot lower non-binary predicate in rule {rule_id}: {raw_predicate}"
                )
            if feature < 0 or feature >= feature_count:
                raise ValueError(
                    f"feature index out of range in rule {rule_id}: {feature}"
                )
            if mask[feature] == "1" and value[feature] != str(bit):
                raise ValueError(
                    f"conflicting predicates for feature {feature} in rule {rule_id}"
                )
            value[feature] = str(bit)
            mask[feature] = "1"

        entries.append(
            TernaryEntry(
                rule_id=rule_id,
                value="".join(value),
                mask="".join(mask),
                action="set_model_positive" if action_id == 1 else "set_model_negative",
                priority=len(rules) - priority,
            )
        )

    return optimize_ternary_entries(entries) if optimize else entries


__all__ = [
    "TernaryEntry",
    "active_rule_features",
    "lower_binary_rule_plan_to_ternary",
    "optimized_rule_plan",
    "optimize_ternary_entries",
]
