"""Intermediate representations shared by the PLEDS compiler stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .spec import HardwareConstraints, ModelOutputSpec, PledsSpec


class OperationKind(str, Enum):
    FIELD_ACCESS = "field_access"
    FEATURE_TABLE = "feature_table"
    INFERENCE_TABLE = "inference_table"
    CONTROL = "control"
    HASH = "hash"
    STATE_ACCESS = "state_access"
    OUTPUT = "output"


@dataclass(frozen=True)
class OperationResources:
    tcam_entries: int = 0
    tcam_units: int = 0
    sram_units: int = 0
    metadata_bits: int = 0
    hash_units: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.tcam_entries,
                self.tcam_units,
                self.sram_units,
                self.metadata_bits,
                self.hash_units,
            )
            < 0
        ):
            raise ValueError("operation resource costs must be nonnegative")

    def as_dict(self) -> dict[str, int]:
        return {
            "tcam_entries": self.tcam_entries,
            "tcam_units": self.tcam_units,
            "sram_units": self.sram_units,
            "metadata_bits": self.metadata_bits,
            "hash_units": self.hash_units,
        }


@dataclass(frozen=True)
class OperationIR:
    name: str
    kind: OperationKind
    dependencies: tuple[str, ...] = ()
    stage_span: int = 1
    resources: OperationResources = field(default_factory=OperationResources)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operation name must not be empty")
        if self.stage_span < 0:
            raise ValueError("operation stage span must be nonnegative")
        if self.name in self.dependencies:
            raise ValueError(f"operation {self.name!r} cannot depend on itself")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "dependencies": list(self.dependencies),
            "stage_span": self.stage_span,
            "resources": self.resources.as_dict(),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class LogicalLdsIR:
    name: str
    task: str
    key_schema: str
    feature_schema: tuple[str, ...]
    model_candidates: tuple[str, ...]
    model_output_type: str
    backend_type: str
    semantic_guards: tuple[str, ...]
    objective: str
    objective_direction: str = "minimize"
    backend_candidates: tuple[str, ...] = ()
    mechanisms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelIR:
    candidate_id: str
    family: str
    output: ModelOutputSpec
    feature_count: int
    representation: str
    plan: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.feature_count <= 0:
            raise ValueError("model feature count must be positive")


@dataclass(frozen=True)
class CompositionIR:
    logical: LogicalLdsIR
    mechanism: str
    backend_type: str
    model: ModelIR | None
    operations: tuple[OperationIR, ...]
    semantic_guards: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "application": self.logical.name,
            "mechanism": self.mechanism,
            "backend_type": self.backend_type,
            "model": (
                {
                    "candidate_id": self.model.candidate_id,
                    "family": self.model.family,
                    "output_type": self.model.output.kind.value,
                    "output_width": self.model.output.width,
                    "feature_count": self.model.feature_count,
                    "representation": self.model.representation,
                }
                if self.model is not None
                else None
            ),
            "semantic_guards": list(self.semantic_guards),
            "operations": [operation.as_dict() for operation in self.operations],
        }


@dataclass(frozen=True)
class ResourceEstimate:
    model_rules: int
    model_key_bits: int
    metadata_bits: int
    backend_bits: int
    hash_count: int
    estimated_tcam_entries: int
    estimated_sram_units: int
    estimated_tcam_units: int
    estimated_stages: int
    estimated_phv_bits: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.model_rules,
            self.model_key_bits,
            self.metadata_bits,
            self.backend_bits,
            self.hash_count,
            self.estimated_tcam_entries,
            self.estimated_sram_units,
            self.estimated_tcam_units,
            self.estimated_stages,
        )
        if min(values) < 0:
            raise ValueError("resource estimates must be nonnegative")
        if self.estimated_phv_bits is not None and self.estimated_phv_bits < 0:
            raise ValueError("estimated PHV width must be nonnegative")

    def as_dict(self) -> dict[str, int | None]:
        return {
            "model_rules": self.model_rules,
            "model_key_bits": self.model_key_bits,
            "metadata_bits": self.metadata_bits,
            "backend_bits": self.backend_bits,
            "hash_count": self.hash_count,
            "estimated_tcam_entries": self.estimated_tcam_entries,
            "estimated_sram_units": self.estimated_sram_units,
            "estimated_tcam_units": self.estimated_tcam_units,
            "estimated_stages": self.estimated_stages,
            "estimated_phv_bits": self.estimated_phv_bits,
        }


@dataclass(frozen=True)
class HardwareLdsIR:
    logical: LogicalLdsIR
    selected_model: str
    backend_type: str
    resource_estimate: ResourceEstimate
    constraints: HardwareConstraints

    @property
    def likely_feasible(self) -> bool:
        estimate = self.resource_estimate
        constraints = self.constraints
        return (
            estimate.estimated_stages <= constraints.max_stages
            and estimate.estimated_tcam_entries <= constraints.max_tcam_entries
            and estimate.estimated_tcam_units <= constraints.max_tcam_units
            and estimate.estimated_sram_units <= constraints.max_sram_units
            and estimate.metadata_bits <= constraints.max_metadata_bits
            and estimate.hash_count <= constraints.max_hash_units
        )


def build_logical_ir(spec: PledsSpec) -> LogicalLdsIR:
    guards: list[str] = []
    if spec.constraints.no_false_negative:
        guards.append("no_false_negative")
    if spec.constraints.no_lost_update:
        guards.append("no_lost_update")

    return LogicalLdsIR(
        name=spec.name,
        task=spec.task,
        key_schema=spec.key,
        feature_schema=spec.features,
        model_candidates=spec.models,
        model_output_type=spec.model_output_type,
        backend_type=spec.backend.type,
        semantic_guards=tuple(guards),
        objective=spec.objective.metric,
        objective_direction=spec.objective.direction,
        backend_candidates=tuple(item.type for item in spec.candidate_backends),
        mechanisms=spec.mechanisms,
    )


__all__ = [
    "CompositionIR",
    "HardwareLdsIR",
    "LogicalLdsIR",
    "ModelIR",
    "OperationIR",
    "OperationKind",
    "OperationResources",
    "ResourceEstimate",
    "build_logical_ir",
]
