"""Error-ordered target search without irreversible estimated-resource pruning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass
class TargetSearchResult(Generic[T]):
    selected: T | None
    examined: list[T]
    rejected: list[T]
    skipped: list[T]


def ordered_target_search(
    candidates: Iterable[T],
    *,
    error: Callable[[T], float],
    estimated_resources: Callable[[T], tuple],
    legal: Callable[[T], bool],
    compile_candidate: Callable[[T], tuple | None],
    identity: Callable[[T], str],
    conventional: Callable[[T], bool] = lambda candidate: False,
) -> TargetSearchResult[T]:
    """Compile until all candidates at the best feasible error are resolved.

    ``legal`` checks semantics and exact logical limits, never physical resource
    predictions. ``compile_candidate`` returns actual resource usage only for a
    target-feasible candidate. It may reuse code, but must bind each candidate's
    entries separately. Conventional candidates initialize a feasible incumbent.
    """
    rows = list(candidates)
    for row in rows:
        if not math.isfinite(error(row)):
            raise ValueError("candidate application error must be finite")
    rejected = [row for row in rows if not legal(row)]
    ordered = sorted(
        (row for row in rows if legal(row)),
        key=lambda row: (
            not conventional(row),
            error(row),
            estimated_resources(row),
            identity(row),
        ),
    )
    examined, skipped = [], []
    best, best_key = None, None
    for row in ordered:
        if best_key is not None and error(row) > best_key[0] and not conventional(row):
            skipped.append(row)
            continue
        examined.append(row)
        resources = compile_candidate(row)
        if resources is None:
            continue
        key = (error(row), resources, identity(row))
        if best_key is None or key < best_key:
            best, best_key = row, key
    return TargetSearchResult(best, examined, rejected, skipped)


def logical_constraints_pass(row: dict, constraints: dict) -> bool:
    """Check declared counts/widths, separately from target allocation estimates."""
    checks = (
        ("estimated_metadata_bits", "max_metadata_bits"),
        ("estimated_tcam_entries", "max_tcam_entries"),
        ("hash_count", "max_hash_units"),
    )
    return bool(row.get("rule_budget_feasible", True)) and all(
        int(row.get(field, 0)) <= int(constraints[limit])
        for field, limit in checks
        if limit in constraints
    )
