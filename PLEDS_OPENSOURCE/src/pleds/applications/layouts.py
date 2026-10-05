"""Existing equal-memory FlowRadar layouts and recovery scoring."""

from __future__ import annotations
from collections import Counter


def powers_of_two(limit: int) -> list[int]:
    values = []
    value = 16
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def baseline_layout(
    budget_kib: int, *, filter_hashes: int, counting_hashes: int, cell_bits: int
) -> dict[str, int]:
    budget_bits = budget_kib * 8192
    candidates = []
    for filter_bank_entries in powers_of_two(budget_bits):
        for counting_row_entries in powers_of_two(budget_bits):
            actual = (
                filter_hashes * filter_bank_entries
                + counting_hashes * cell_bits * counting_row_entries
            )
            if actual == budget_bits:
                candidates.append((filter_bank_entries, counting_row_entries))
    if len(candidates) != 1:
        raise ValueError(f"expected one baseline layout for {budget_kib} KiB")
    filter_bank_entries, counting_row_entries = candidates[0]
    return {
        "actual_memory_bits": budget_bits,
        "filter_bank_entries": filter_bank_entries,
        "counting_row_entries": counting_row_entries,
    }


def hash_layout(
    budget_kib: int, *, filter_hashes: int, counting_hashes: int, cell_bits: int
) -> dict[str, int]:
    budget_bits = budget_kib * 8192
    candidates = []
    for filter_bank_entries in powers_of_two(budget_bits):
        for counting_row_entries in powers_of_two(budget_bits):
            actual = 2 * (
                filter_hashes * filter_bank_entries
                + counting_hashes * cell_bits * counting_row_entries
            )
            if actual == budget_bits:
                candidates.append((filter_bank_entries, counting_row_entries))
    if len(candidates) != 1:
        raise ValueError(f"expected one hash layout for {budget_kib} KiB")
    filter_bank_entries, counting_row_entries = candidates[0]
    return {
        "actual_memory_bits": budget_bits,
        "partition_count": 2,
        "filter_bank_entries": filter_bank_entries,
        "counting_row_entries": counting_row_entries,
    }


def tiered_layouts(
    budget_kib: int,
    *,
    exact_bank_counts: list[int],
    filter_hashes: int,
    counting_hashes: int,
    cell_bits: int,
    exact_entry_bits: int,
) -> list[dict[str, int]]:
    budget_bits = budget_kib * 8192
    powers = powers_of_two(budget_bits)
    layouts = []
    for exact_bank_count in exact_bank_counts:
        for exact_bank_entries in powers:
            for filter_bank_entries in powers:
                for counting_row_entries in powers:
                    actual = (
                        exact_bank_count * exact_bank_entries * exact_entry_bits
                        + filter_hashes * filter_bank_entries
                        + counting_hashes * cell_bits * counting_row_entries
                    )
                    if actual != budget_bits:
                        continue
                    exact_fraction = (
                        exact_bank_count
                        * exact_bank_entries
                        * exact_entry_bits
                        / budget_bits
                    )
                    if exact_fraction > 0.5:
                        continue
                    layouts.append(
                        {
                            "actual_memory_bits": actual,
                            "exact_bank_count": exact_bank_count,
                            "exact_bank_entries": exact_bank_entries,
                            "exact_capacity": exact_bank_count * exact_bank_entries,
                            "exact_memory_fraction": exact_fraction,
                            "filter_bank_entries": filter_bank_entries,
                            "counting_row_entries": counting_row_entries,
                        }
                    )
    if not layouts:
        raise ValueError(f"no Tiered FlowRadar layout for {budget_kib} KiB")
    return layouts


def score_backend(
    backend: object, packets: list[tuple[int, int, int, int, int]]
) -> dict[str, object]:
    true_counts = Counter(packets)
    true_keys = set(true_counts)
    backend.update_many(packets)
    decoded = backend.decode()
    recovered = true_keys & decoded.recovered_counts.keys()
    exact_counts = sum(
        decoded.recovered_counts[key] == true_counts[key] for key in recovered
    )
    return {
        "packet_count": len(packets),
        "flow_count": len(true_counts),
        "recovered_flow_count": len(recovered),
        "recovered_flow_ratio": len(recovered) / len(true_counts),
        "exact_count_flow_count": exact_counts,
        "exact_count_ratio": exact_counts / len(true_counts),
        "residual_flow_cells": decoded.residual_flow_cells,
        "residual_packet_cells": decoded.residual_packet_cells,
        "trustworthy_counts": decoded.trustworthy_counts,
    }
