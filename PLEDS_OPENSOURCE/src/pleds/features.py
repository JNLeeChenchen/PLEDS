"""Feature definitions shared by software and generated P4 pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class FeatureConditionKind(str, Enum):
    FIELD_BIT = "field_bit"
    PREFIX = "prefix"
    EQUALITY = "equality"
    RANGE = "range"
    CLASS = "class"


class FeatureMatchKind(str, Enum):
    DIRECT = "direct_match"
    MAT_ASSISTED = "mat_assisted_match"


@dataclass(frozen=True)
class FeatureCondition:
    name: str
    kind: FeatureConditionKind
    field: str
    field_width: int
    direct_entry_expansion: int | None = 1
    mat_entries: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.field:
            raise ValueError("feature condition name and field must not be empty")
        if self.field_width <= 0 or self.mat_entries <= 0:
            raise ValueError("feature condition widths and MAT size must be positive")
        if self.direct_entry_expansion is not None and self.direct_entry_expansion <= 0:
            raise ValueError("direct-entry expansion must be positive")


@dataclass(frozen=True)
class FeatureLoweringCost:
    added_entries: int
    key_bits: int
    metadata_bits: int
    dependency_stages: int

    def __post_init__(self) -> None:
        if (
            min(
                self.added_entries,
                self.key_bits,
                self.metadata_bits,
                self.dependency_stages,
            )
            < 0
        ):
            raise ValueError("feature lowering costs must be nonnegative")


@dataclass(frozen=True)
class FeatureCostWeights:
    entry: int = 1
    key_bit: int = 1
    metadata_bit: int = 2
    dependency_stage: int = 512


@dataclass(frozen=True)
class FeatureLoweringDecision:
    condition: FeatureCondition
    implementation: FeatureMatchKind
    use_count: int
    cost: FeatureLoweringCost

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.condition.name,
            "condition_kind": self.condition.kind.value,
            "field": self.condition.field,
            "implementation": self.implementation.value,
            "use_count": self.use_count,
            "cost": {
                "added_entries": self.cost.added_entries,
                "key_bits": self.cost.key_bits,
                "metadata_bits": self.cost.metadata_bits,
                "dependency_stages": self.cost.dependency_stages,
            },
        }


def feature_lowering_alternatives(
    condition: FeatureCondition, *, use_count: int
) -> dict[FeatureMatchKind, FeatureLoweringCost]:
    if use_count <= 0:
        raise ValueError("feature use count must be positive")
    alternatives = {
        FeatureMatchKind.MAT_ASSISTED: FeatureLoweringCost(
            added_entries=condition.mat_entries,
            key_bits=use_count,
            metadata_bits=1,
            dependency_stages=1,
        )
    }
    if condition.direct_entry_expansion is not None:
        alternatives[FeatureMatchKind.DIRECT] = FeatureLoweringCost(
            added_entries=(condition.direct_entry_expansion - 1) * use_count,
            key_bits=condition.field_width * use_count,
            metadata_bits=0,
            dependency_stages=0,
        )
    return alternatives


def _weighted_feature_cost(
    cost: FeatureLoweringCost, weights: FeatureCostWeights
) -> tuple[int, int, int, int, int]:
    score = (
        cost.added_entries * weights.entry
        + cost.key_bits * weights.key_bit
        + cost.metadata_bits * weights.metadata_bit
        + cost.dependency_stages * weights.dependency_stage
    )
    return (
        score,
        cost.dependency_stages,
        cost.added_entries,
        cost.metadata_bits,
        cost.key_bits,
    )


def select_feature_lowering(
    condition: FeatureCondition,
    *,
    use_count: int,
    weights: FeatureCostWeights | None = None,
) -> FeatureLoweringDecision:
    """Choose a target-aware implementation for one selected feature."""

    selected_weights = weights or FeatureCostWeights()
    alternatives = feature_lowering_alternatives(condition, use_count=use_count)
    implementation, cost = min(
        alternatives.items(),
        key=lambda item: (
            _weighted_feature_cost(item[1], selected_weights),
            0 if item[0] is FeatureMatchKind.DIRECT else 1,
        ),
    )
    return FeatureLoweringDecision(condition, implementation, use_count, cost)


def build_feature_lowering_plan(
    conditions: Iterable[tuple[FeatureCondition, int]],
    *,
    weights: FeatureCostWeights | None = None,
) -> tuple[FeatureLoweringDecision, ...]:
    return tuple(
        select_feature_lowering(condition, use_count=use_count, weights=weights)
        for condition, use_count in conditions
    )


def model_feature_use_counts(model_plan: dict[str, object]) -> dict[int, int]:
    """Count feature references in supported lowered model representations."""

    counts: dict[int, int] = {}

    def add(feature: int) -> None:
        counts[feature] = counts.get(feature, 0) + 1

    def visit(plan: dict[str, object]) -> None:
        rules = plan.get("rules", [])
        if isinstance(rules, list):
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                mask = rule.get("mask")
                if isinstance(mask, str):
                    for feature, bit in enumerate(mask):
                        if bit == "1":
                            add(feature)
                predicates = rule.get("predicates", [])
                if isinstance(predicates, list):
                    for predicate in predicates:
                        if isinstance(predicate, dict) and "feature" in predicate:
                            add(int(predicate["feature"]))
        trees = plan.get("trees", [])
        if isinstance(trees, list):
            for tree in trees:
                if isinstance(tree, dict):
                    visit(tree)
        levels = plan.get("levels", [])
        if isinstance(levels, list):
            for level in levels:
                if not isinstance(level, dict):
                    continue
                for node in level.get("nodes", []):
                    if isinstance(node, dict) and node.get("feature") is not None:
                        add(int(node["feature"]))
        tables = plan.get("feature_tables", [])
        if isinstance(tables, list):
            for table in tables:
                if isinstance(table, dict) and table.get("feature") is not None:
                    add(int(table["feature"]))

    visit(model_plan)
    return counts


__all__ = [
    "FeatureCondition",
    "FeatureConditionKind",
    "FeatureCostWeights",
    "FeatureDef",
    "FeatureLoweringCost",
    "FeatureLoweringDecision",
    "FeatureMatchKind",
    "PacketFields",
    "COMPACT_PACKET_FEATURES",
    "build_feature_lowering_plan",
    "extract_compact_packet_features",
    "feature_lowering_alternatives",
    "model_feature_use_counts",
    "select_feature_lowering",
    "compact_feature_plan",
]


@dataclass(frozen=True)
class FeatureDef:
    index: int
    name: str
    description: str
    p4_expr: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "name": self.name,
            "description": self.description,
            "p4_expr": self.p4_expr,
        }


@dataclass(frozen=True)
class PacketFields:
    """Normalized packet fields consumed by the current binary feature plan."""

    src_addr: int
    dst_addr: int
    src_port: int
    dst_port: int
    protocol: int
    total_len: int

    def __post_init__(self) -> None:
        _check_unsigned("src_addr", self.src_addr, 32)
        _check_unsigned("dst_addr", self.dst_addr, 32)
        _check_unsigned("src_port", self.src_port, 16)
        _check_unsigned("dst_port", self.dst_port, 16)
        _check_unsigned("protocol", self.protocol, 8)
        _check_unsigned("total_len", self.total_len, 16)


def _check_unsigned(name: str, value: int, width: int) -> None:
    if value < 0 or value >= 1 << width:
        raise ValueError(f"{name} must fit in bit<{width}>: {value}")


def _bit(value: int, index: int) -> int:
    return (value >> index) & 1


def extract_compact_packet_features(packet: PacketFields) -> list[int]:
    """Evaluate the software equivalent of the current P4 feature plan."""

    l4_valid = packet.protocol in {6, 17}
    src_port = packet.src_port if l4_valid else 0
    dst_port = packet.dst_port if l4_valid else 0
    return [
        _bit(packet.src_addr, 31),
        _bit(packet.src_addr, 27),
        _bit(packet.dst_addr, 31),
        _bit(packet.dst_addr, 23),
        int(src_port in {22, 80, 443, 8080}),
        int(dst_port in {22, 80, 443, 8080}),
        int(dst_port >= 1024),
        int(packet.protocol == 6),
        int(packet.protocol == 17),
        _bit(packet.src_addr, 23),
        _bit(packet.dst_addr, 27),
        _bit(packet.src_addr, 15),
    ]


COMPACT_PACKET_FEATURES: tuple[FeatureDef, ...] = (
    FeatureDef(0, "src_bit_31", "top source-IP bit", "hdr.ipv4.src_addr[31:31]"),
    FeatureDef(1, "src_bit_27", "source-IP bit 27", "hdr.ipv4.src_addr[27:27]"),
    FeatureDef(2, "dst_bit_31", "top destination-IP bit", "hdr.ipv4.dst_addr[31:31]"),
    FeatureDef(3, "dst_bit_23", "destination-IP bit 23", "hdr.ipv4.dst_addr[23:23]"),
    FeatureDef(
        4,
        "src_service_port",
        "source port in common service set",
        "ig_md.src_service_port",
    ),
    FeatureDef(
        5,
        "dst_service_port",
        "destination port in common service set",
        "ig_md.dst_service_port",
    ),
    FeatureDef(
        6, "dst_ephemeral_port", "destination port >= 1024", "ig_md.dst_ephemeral_port"
    ),
    FeatureDef(7, "is_tcp", "IPv4 protocol is TCP", "ig_md.proto_tcp"),
    FeatureDef(8, "is_udp", "IPv4 protocol is UDP", "ig_md.proto_udp"),
    FeatureDef(9, "src_bit_23", "source-IP bit 23", "hdr.ipv4.src_addr[23:23]"),
    FeatureDef(10, "dst_bit_27", "destination-IP bit 27", "hdr.ipv4.dst_addr[27:27]"),
    FeatureDef(11, "src_bit_15", "source-IP bit 15", "hdr.ipv4.src_addr[15:15]"),
)


def compact_feature_plan() -> dict[str, object]:
    return {
        "format": "pleds_feature_plan_v1",
        "schema": "compact_packet_features_v1",
        "feature_count": len(COMPACT_PACKET_FEATURES),
        "features": [feature.as_dict() for feature in COMPACT_PACKET_FEATURES],
        "notes": [
            "Features 4-6 depend on TCP/UDP parser extraction of L4 ports.",
            "The generated P4 initializes L4 ports to zero when no TCP/UDP header is valid.",
            "Every feature is a deterministic function of the normalized five-tuple key.",
        ],
    }
