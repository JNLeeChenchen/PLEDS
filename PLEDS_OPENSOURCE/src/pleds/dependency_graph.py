"""Dependency analysis for composed data-plane operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .ir import CompositionIR, OperationIR, OperationResources, ResourceEstimate


@dataclass(frozen=True)
class ScheduledOperation:
    operation: OperationIR
    first_stage: int
    stage_end: int

    def as_dict(self) -> dict[str, object]:
        payload = self.operation.as_dict()
        payload.update({"first_stage": self.first_stage, "stage_end": self.stage_end})
        return payload


@dataclass(frozen=True)
class StageResourceUse:
    stage: int
    resources: OperationResources
    operations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "operations": list(self.operations),
            **self.resources.as_dict(),
        }


class DependencyGraph:
    """A DAG whose edges encode data and control dependencies."""

    def __init__(self, operations: Iterable[OperationIR]):
        ordered = tuple(operations)
        self._operations = ordered
        self._by_name = {operation.name: operation for operation in ordered}
        if len(self._by_name) != len(ordered):
            raise ValueError("dependency graph operation names must be unique")
        for operation in ordered:
            unknown = set(operation.dependencies) - self._by_name.keys()
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"operation {operation.name!r} has unknown dependencies: {names}"
                )
        self._topological = self._topological_order()
        self._schedule = self._schedule_operations()

    @classmethod
    def from_composition(cls, composition: CompositionIR) -> "DependencyGraph":
        return cls(composition.operations)

    @property
    def topological_order(self) -> tuple[str, ...]:
        return self._topological

    @property
    def schedule(self) -> tuple[ScheduledOperation, ...]:
        return self._schedule

    @property
    def critical_path_stages(self) -> int:
        return max((item.stage_end for item in self._schedule), default=0)

    @property
    def peak_metadata_bits(self) -> int:
        return max(
            (item.operation.resources.metadata_bits for item in self._schedule),
            default=0,
        )

    @property
    def stage_resources(self) -> tuple[StageResourceUse, ...]:
        grouped: dict[int, list[ScheduledOperation]] = {}
        scheduled_by_name = {item.operation.name: item for item in self._schedule}
        for item in self._schedule:
            resources = item.operation.resources
            has_resources = any(resources.as_dict().values())
            if item.operation.stage_span == 0 and not has_resources:
                continue
            stage = item.first_stage
            if item.operation.stage_span == 0:
                consumers = [
                    scheduled_by_name[operation.name]
                    for operation in self._operations
                    if item.operation.name in operation.dependencies
                ]
                if consumers:
                    stage = min(consumer.first_stage for consumer in consumers)
            grouped.setdefault(stage, []).append(item)
        result = []
        for stage, items in sorted(grouped.items()):
            result.append(
                StageResourceUse(
                    stage=stage,
                    operations=tuple(item.operation.name for item in items),
                    resources=OperationResources(
                        tcam_entries=sum(
                            item.operation.resources.tcam_entries for item in items
                        ),
                        tcam_units=sum(
                            item.operation.resources.tcam_units for item in items
                        ),
                        sram_units=sum(
                            item.operation.resources.sram_units for item in items
                        ),
                        metadata_bits=max(
                            (item.operation.resources.metadata_bits for item in items),
                            default=0,
                        ),
                        hash_units=sum(
                            item.operation.resources.hash_units for item in items
                        ),
                    ),
                )
            )
        return tuple(result)

    def _topological_order(self) -> tuple[str, ...]:
        indegree = {name: 0 for name in self._by_name}
        successors = {name: [] for name in self._by_name}
        for operation in self._operations:
            indegree[operation.name] = len(operation.dependencies)
            for dependency in operation.dependencies:
                successors[dependency].append(operation.name)
        ready = [
            operation.name
            for operation in self._operations
            if indegree[operation.name] == 0
        ]
        ordered: list[str] = []
        while ready:
            name = ready.pop(0)
            ordered.append(name)
            for successor in successors[name]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(ordered) != len(self._operations):
            raise ValueError("data-plane operation dependencies contain a cycle")
        return tuple(ordered)

    def _schedule_operations(self) -> tuple[ScheduledOperation, ...]:
        stage_end: dict[str, int] = {}
        scheduled = []
        for name in self._topological:
            operation = self._by_name[name]
            first_stage = max(
                (stage_end[dependency] for dependency in operation.dependencies),
                default=0,
            )
            end = first_stage + operation.stage_span
            stage_end[name] = end
            scheduled.append(ScheduledOperation(operation, first_stage, end))
        return tuple(scheduled)

    def as_dict(self) -> dict[str, object]:
        return {
            "format": "pleds_dependency_graph_v1",
            "topological_order": list(self.topological_order),
            "critical_path_stages": self.critical_path_stages,
            "peak_metadata_bits": self.peak_metadata_bits,
            "operations": [item.as_dict() for item in self.schedule],
            "stage_resources": [item.as_dict() for item in self.stage_resources],
        }


def apply_dependency_lower_bound(
    estimate: ResourceEstimate, graph: DependencyGraph
) -> ResourceEstimate:
    """Add structural lower bounds without replacing target-calibrated estimates."""

    return replace(
        estimate,
        metadata_bits=max(estimate.metadata_bits, graph.peak_metadata_bits),
        estimated_stages=max(estimate.estimated_stages, graph.critical_path_stages),
    )


__all__ = [
    "DependencyGraph",
    "ScheduledOperation",
    "StageResourceUse",
    "apply_dependency_lower_bound",
]
