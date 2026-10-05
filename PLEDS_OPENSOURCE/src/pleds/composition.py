"""Composition of learned-model outputs with stateful data-plane operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .ir import (
    CompositionIR,
    LogicalLdsIR,
    ModelIR,
    OperationIR,
    OperationKind,
    OperationResources,
    ResourceEstimate,
)
from .spec import ModelOutputKind, ModelOutputSpec


class CompositionMechanism(str, Enum):
    CONVENTIONAL = "conventional"
    TIERING = "tiering"
    PARTITIONING = "partitioning"


_BACKEND_MECHANISMS = {
    "partitioned_flowradar": CompositionMechanism.PARTITIONING,
    "tiered_flowradar": CompositionMechanism.TIERING,
    "flowradar": CompositionMechanism.CONVENTIONAL,
}


_OUTPUT_BY_MECHANISM = {
    CompositionMechanism.TIERING: ModelOutputKind.HOT_COLD,
    CompositionMechanism.PARTITIONING: ModelOutputKind.PARTITION_ID,
}


@dataclass(frozen=True)
class CompositionCosts:
    feature_table_entries: int = 0
    feature_table_units: int = 0
    feature_sram_units: int = 0
    feature_stages: int = 0
    inference_entries: int = 0
    inference_tcam_units: int = 0
    inference_sram_units: int = 0
    inference_stages: int = 0
    backend_sram_units: int = 0
    backend_stages: int = 1
    metadata_bits: int = 0
    hash_units: int = 0

    @classmethod
    def from_estimate(cls, estimate: ResourceEstimate) -> "CompositionCosts":
        return cls(
            inference_entries=estimate.estimated_tcam_entries,
            inference_tcam_units=estimate.estimated_tcam_units,
            inference_stages=1 if estimate.model_rules else 0,
            backend_sram_units=estimate.estimated_sram_units,
            backend_stages=max(
                1, estimate.estimated_stages - (1 if estimate.model_rules else 0)
            ),
            metadata_bits=estimate.metadata_bits,
            hash_units=estimate.hash_count,
        )


def infer_mechanism(backend_type: str) -> CompositionMechanism:
    try:
        return _BACKEND_MECHANISMS[backend_type]
    except KeyError as exc:
        raise ValueError(
            f"no composition mechanism registered for {backend_type!r}"
        ) from exc


def output_for_mechanism(
    mechanism: CompositionMechanism, *, partition_count: int = 2
) -> ModelOutputSpec:
    if mechanism is CompositionMechanism.CONVENTIONAL:
        return ModelOutputSpec()
    kind = _OUTPUT_BY_MECHANISM[mechanism]
    width = (
        max(1, math.ceil(math.log2(partition_count)))
        if kind is ModelOutputKind.PARTITION_ID
        else 1
    )
    return ModelOutputSpec(kind=kind, width=width)


def validate_model_output(
    mechanism: CompositionMechanism, output: ModelOutputSpec
) -> None:
    if mechanism is CompositionMechanism.CONVENTIONAL:
        return
    required = _OUTPUT_BY_MECHANISM[mechanism]
    compatible = output.kind is required
    # Existing binary model plans encode the one-bit hot/cold interface directly.
    if mechanism is CompositionMechanism.TIERING:
        compatible = compatible or output.kind is ModelOutputKind.BINARY_DECISION
    if not compatible:
        raise ValueError(
            f"{mechanism.value} requires {required.value}, got {output.kind.value}"
        )


def build_composition_ir(
    logical: LogicalLdsIR,
    *,
    backend_type: str,
    model: ModelIR | None,
    costs: CompositionCosts,
    mechanism: CompositionMechanism | str | None = None,
) -> CompositionIR:
    """Connect a lowered model to its backend as data-plane operations."""

    selected = (
        infer_mechanism(backend_type)
        if mechanism is None
        else CompositionMechanism(mechanism)
    )
    if model is None and selected is not CompositionMechanism.CONVENTIONAL:
        selected = CompositionMechanism.CONVENTIONAL
    if model is not None:
        validate_model_output(selected, model.output)

    operations: list[OperationIR] = [
        OperationIR(
            name="packet_fields",
            kind=OperationKind.FIELD_ACCESS,
            stage_span=0,
            attributes={"key_schema": logical.key_schema},
        )
    ]
    model_dependency = "packet_fields"
    if model is not None and costs.feature_stages:
        operations.append(
            OperationIR(
                name="feature_tables",
                kind=OperationKind.FEATURE_TABLE,
                dependencies=("packet_fields",),
                stage_span=costs.feature_stages,
                resources=OperationResources(
                    tcam_entries=costs.feature_table_entries,
                    tcam_units=costs.feature_table_units,
                    sram_units=costs.feature_sram_units,
                    metadata_bits=costs.metadata_bits,
                ),
            )
        )
        model_dependency = "feature_tables"

    route_dependency = "packet_fields"
    if model is not None:
        operations.append(
            OperationIR(
                name="inference",
                kind=OperationKind.INFERENCE_TABLE,
                dependencies=(model_dependency,),
                stage_span=max(1, costs.inference_stages),
                resources=OperationResources(
                    tcam_entries=costs.inference_entries,
                    tcam_units=costs.inference_tcam_units,
                    sram_units=costs.inference_sram_units,
                    metadata_bits=model.output.width,
                ),
                attributes={"representation": model.representation},
            )
        )
        operations.append(
            OperationIR(
                name="state_selection",
                kind=OperationKind.CONTROL,
                dependencies=("inference",),
                stage_span=0,
            )
        )
        route_dependency = "state_selection"

    operations.append(
        OperationIR(
            name="backend_hashes",
            kind=OperationKind.HASH,
            dependencies=("packet_fields",),
            stage_span=0,
            resources=OperationResources(hash_units=costs.hash_units),
        )
    )
    operations.append(
        OperationIR(
            name="backend_state",
            kind=OperationKind.STATE_ACCESS,
            dependencies=(route_dependency, "backend_hashes"),
            stage_span=max(1, costs.backend_stages),
            resources=OperationResources(
                sram_units=costs.backend_sram_units,
                metadata_bits=costs.metadata_bits,
            ),
            attributes={"backend": backend_type, "mechanism": selected.value},
        )
    )
    operations.append(
        OperationIR(
            name="application_result",
            kind=OperationKind.OUTPUT,
            dependencies=("backend_state",),
            stage_span=0,
        )
    )
    return CompositionIR(
        logical=logical,
        mechanism=selected.value,
        backend_type=backend_type,
        model=model,
        operations=tuple(operations),
        semantic_guards=logical.semantic_guards,
    )


__all__ = [
    "CompositionCosts",
    "CompositionMechanism",
    "build_composition_ir",
    "infer_mechanism",
    "output_for_mechanism",
    "validate_model_output",
]
