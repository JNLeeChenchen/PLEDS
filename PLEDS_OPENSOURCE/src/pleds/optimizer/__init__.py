"""Hardware-aware candidate selection for PLEDS.

The experiment drivers produce validation metrics and resource plans.  This
module applies the common semantic and hardware gates so that selection logic
does not remain duplicated across individual experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Hashable, Iterable, Mapping

from pleds.ir import ResourceEstimate
from pleds.spec import HardwareConstraints


@dataclass(frozen=True)
class CandidateEvaluation:
    """One validation-evaluated implementation candidate."""

    name: str
    objective_value: float
    resource_estimate: ResourceEstimate
    semantic_checks: Mapping[str, bool] = field(default_factory=dict)
    compile_success: bool | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateDecision:
    candidate: CandidateEvaluation
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.name,
            "objective_value": self.candidate.objective_value,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "compile_success": self.candidate.compile_success,
            "resource_estimate": self.candidate.resource_estimate.as_dict(),
        }


@dataclass(frozen=True)
class SelectionResult:
    selected: CandidateEvaluation | None
    decisions: tuple[CandidateDecision, ...]
    objective_direction: str

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_candidate": self.selected.name if self.selected else None,
            "objective_direction": self.objective_direction,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


def _objective_cost(candidate: CandidateEvaluation, direction: str) -> float:
    if direction == "minimize":
        return candidate.objective_value
    if direction == "maximize":
        return -candidate.objective_value
    raise ValueError("objective_direction must be 'minimize' or 'maximize'")


def _cost_vector(
    candidate: CandidateEvaluation, objective_direction: str
) -> tuple[float, int, int, int, int, int, int]:
    estimate = candidate.resource_estimate
    return (
        _objective_cost(candidate, objective_direction),
        estimate.estimated_sram_units,
        estimate.estimated_tcam_units,
        estimate.estimated_tcam_entries,
        estimate.estimated_stages,
        estimate.metadata_bits,
        estimate.hash_count,
    )


def dominates(
    left: CandidateEvaluation,
    right: CandidateEvaluation,
    *,
    objective_direction: str,
) -> bool:
    """Return whether ``left`` is no worse in every constrained dimension."""

    left_values = _cost_vector(left, objective_direction)
    right_values = _cost_vector(right, objective_direction)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def pareto_front(
    candidates: Iterable[CandidateEvaluation],
    *,
    objective_direction: str,
) -> tuple[CandidateEvaluation, ...]:
    """Return the nondominated candidates in deterministic order."""

    rows = tuple(candidates)
    frontier = [
        candidate
        for candidate in rows
        if not any(
            dominates(other, candidate, objective_direction=objective_direction)
            for other in rows
            if other is not candidate
        )
    ]
    return tuple(
        sorted(
            frontier,
            key=lambda item: (*_cost_vector(item, objective_direction), item.name),
        )
    )


def _default_identity(candidate: CandidateEvaluation) -> Hashable:
    explicit = candidate.payload.get("deployment_identity")
    if explicit is not None:
        if isinstance(explicit, (str, int, float, tuple)):
            return explicit
        return json.dumps(explicit, sort_keys=True, separators=(",", ":"))
    output = candidate.payload.get("model_output_sha256")
    backend = candidate.payload.get("backend")
    parameters = candidate.payload.get("backend_parameters")
    if output is not None and backend is not None:
        return (
            str(output),
            str(backend),
            json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        )
    return candidate.name


def deduplicate_candidates(
    candidates: Iterable[CandidateEvaluation],
    *,
    objective_direction: str,
    identity: Callable[[CandidateEvaluation], Hashable] | None = None,
) -> tuple[CandidateEvaluation, ...]:
    """Keep the least-cost representative of each equivalent deployment."""

    identity_fn = identity or _default_identity
    selected: dict[Hashable, CandidateEvaluation] = {}
    for candidate in candidates:
        key = identity_fn(candidate)
        current = selected.get(key)
        if current is None or (
            *_cost_vector(candidate, objective_direction),
            candidate.name,
        ) < (*_cost_vector(current, objective_direction), current.name):
            selected[key] = candidate
    return tuple(selected.values())


def resource_violations(
    estimate: ResourceEstimate,
    constraints: HardwareConstraints,
    *,
    physical: bool = True,
) -> tuple[str, ...]:
    """Return every static hardware-budget violation."""

    checks = (
        ("stages", estimate.estimated_stages, constraints.max_stages),
        ("tcam_entries", estimate.estimated_tcam_entries, constraints.max_tcam_entries),
        ("tcam_units", estimate.estimated_tcam_units, constraints.max_tcam_units),
        ("sram_units", estimate.estimated_sram_units, constraints.max_sram_units),
        ("metadata_bits", estimate.metadata_bits, constraints.max_metadata_bits),
        ("hash_units", estimate.hash_count, constraints.max_hash_units),
    )
    return tuple(
        f"{name}={actual} exceeds {limit}"
        for name, actual, limit in checks
        if actual > limit
        and (physical or name in {"tcam_entries", "metadata_bits", "hash_units"})
    )


def evaluate_candidate(
    candidate: CandidateEvaluation,
    constraints: HardwareConstraints,
    *,
    require_compile: bool = True,
    check_physical: bool = True,
) -> CandidateDecision:
    """Apply semantic guards, static budgets, and optional compiler feedback."""

    reasons = list(
        resource_violations(
            candidate.resource_estimate, constraints, physical=check_physical
        )
    )
    if constraints.no_false_negative and not candidate.semantic_checks.get(
        "no_false_negative", False
    ):
        reasons.append("no_false_negative guard failed")
    if constraints.no_lost_update and not candidate.semantic_checks.get(
        "no_lost_update", False
    ):
        reasons.append("no_lost_update guard failed")
    if require_compile and candidate.compile_success is not True:
        reasons.append(
            "bf-p4c compile not run"
            if candidate.compile_success is None
            else "bf-p4c compile failed"
        )
    return CandidateDecision(candidate, not reasons, tuple(reasons))


def select_candidate(
    candidates: Iterable[CandidateEvaluation],
    constraints: HardwareConstraints,
    *,
    objective_direction: str,
    require_compile: bool = True,
) -> SelectionResult:
    """Select the best feasible candidate using deterministic resource ties."""

    if objective_direction not in {"minimize", "maximize"}:
        raise ValueError("objective_direction must be 'minimize' or 'maximize'")
    decisions = tuple(
        evaluate_candidate(candidate, constraints, require_compile=require_compile)
        for candidate in candidates
    )
    feasible = [decision.candidate for decision in decisions if decision.accepted]
    if not feasible:
        return SelectionResult(None, decisions, objective_direction)

    def key(
        candidate: CandidateEvaluation,
    ) -> tuple[float, int, int, int, int, int, str]:
        estimate = candidate.resource_estimate
        return (
            _objective_cost(candidate, objective_direction),
            estimate.estimated_stages,
            estimate.estimated_tcam_units,
            estimate.estimated_sram_units,
            (
                estimate.estimated_phv_bits
                if estimate.estimated_phv_bits is not None
                else estimate.metadata_bits
            ),
            estimate.hash_count,
            candidate.name,
        )

    return SelectionResult(min(feasible, key=key), decisions, objective_direction)


__all__ = [
    "CandidateDecision",
    "CandidateEvaluation",
    "SelectionResult",
    "deduplicate_candidates",
    "dominates",
    "evaluate_candidate",
    "pareto_front",
    "resource_violations",
    "select_candidate",
]
