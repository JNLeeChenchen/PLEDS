import pytest

from pleds.optimizer.target_search import ordered_target_search


def search(rows):
    visited = []

    def compile_candidate(row):
        visited.append(row["name"])
        return row.get("actual")

    result = ordered_target_search(
        rows,
        error=lambda r: r["error"],
        estimated_resources=lambda r: (r["estimate"],),
        legal=lambda r: r.get("legal", True),
        identity=lambda r: r["name"],
        compile_candidate=compile_candidate,
    )
    return result, visited


def test_estimated_overrun_does_not_discard_target_feasible_candidate():
    result, visited = search(
        [
            dict(name="accurate", error=0.001, estimate=9, actual=(8,)),
            dict(name="fallback", error=0.03, estimate=7, actual=(7,)),
        ]
    )
    assert result.selected["name"] == "accurate"
    assert visited == ["accurate"]


def test_estimated_dominator_failure_does_not_discard_alternative():
    result, visited = search(
        [
            dict(name="optimistic", error=0.01, estimate=1),
            dict(name="alternative", error=0.02, estimate=2, actual=(8,)),
            dict(name="worse", error=0.03, estimate=3, actual=(7,)),
        ]
    )
    assert result.selected["name"] == "alternative"
    assert visited == ["optimistic", "alternative"]


def test_all_equal_error_candidates_use_actual_resource_tie_break():
    result, visited = search(
        [
            dict(name="estimate_winner", error=0.01, estimate=1, actual=(9,)),
            dict(name="actual_winner", error=0.01, estimate=2, actual=(8,)),
        ]
    )
    assert result.selected["name"] == "actual_winner"
    assert len(visited) == 2


def test_illegal_candidate_never_reaches_compiler():
    result, visited = search([dict(name="illegal", error=0, estimate=1, legal=False)])
    assert result.selected is None
    assert visited == []
    assert len(result.rejected) == 1


def test_failed_compilations_do_not_terminate_search():
    result, visited = search(
        [dict(name="a", error=0.1, estimate=1), dict(name="b", error=0.2, estimate=2)]
    )
    assert result.selected is None
    assert visited == ["a", "b"]


def test_nonfinite_error_rejected():
    with pytest.raises(ValueError, match="finite"):
        search([dict(name="bad", error=float("nan"), estimate=1)])
