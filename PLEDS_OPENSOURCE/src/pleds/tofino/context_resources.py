"""Parse physical resource allocations from a Tofino ``context.json`` file."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator
import json


NESTED_STAGE_TABLE_KEYS = (
    "ternary_indirection_stage_table",
    "action_stage_table",
    "selection_stage_table",
    "statistics_stage_table",
    "meter_stage_table",
    "stateful_stage_table",
)


def _stage_table_roots(table: dict[str, object]) -> list[dict[str, object]]:
    roots = list(table.get("stage_tables") or [])
    match_attributes = table.get("match_attributes") or {}
    if isinstance(match_attributes, dict):
        roots.extend(match_attributes.get("stage_tables") or [])
    return [item for item in roots if isinstance(item, dict)]


def _walk_stage_tables(stage_table: dict[str, object]) -> Iterator[dict[str, object]]:
    yield stage_table
    for key in NESTED_STAGE_TABLE_KEYS:
        nested = stage_table.get(key)
        if isinstance(nested, dict):
            yield from _walk_stage_tables(nested)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    yield from _walk_stage_tables(item)


def _memory_units(allocation: dict[str, object]) -> list[int]:
    units: list[int] = []
    for item in allocation.get("memory_units_and_vpns") or []:
        if isinstance(item, dict):
            units.extend(int(value) for value in item.get("memory_units") or [])
    if not units and allocation.get("memory_unit") is not None:
        units.append(int(allocation["memory_unit"]))
    return units


def _component(table: dict[str, object]) -> str:
    name = str(table.get("name", ""))
    lowered = name.lower()
    refs = table.get("stateful_table_refs") or []
    referenced_names = " ".join(
        str(item.get("name", "")) for item in refs if isinstance(item, dict)
    ).lower()
    combined = f"{lowered} {referenced_names}"
    if "partition_shard" in combined:
        return "partition_mat"
    if "flow_filter" in combined and "compute_flow_hash" not in combined:
        return "flow_filter_state"
    if any(
        marker in combined
        for marker in (
            "_key_src",
            "_key_dst",
            "_key_ports",
            "_key_protocol",
            "_flow_count",
            "_packet_count",
        )
    ):
        return "flow_record_state"
    if "pleds_" in combined and "_packets" in combined:
        return "instrumentation"
    if any(
        marker in combined
        for marker in (
            "feature_table",
            "src_service_port_table",
            "dst_service_port_table",
            "dst_ephemeral_port_table",
            "protocol_feature_table",
            "length_feature_table",
        )
    ):
        return "feature_mat"
    if any(
        marker in combined
        for marker in ("learned_selector", "partition_selector", "inferencemat")
    ):
        return "inference_mat"
    if any(marker in combined for marker in ("compute_flow_hash",)):
        return "hash_logic"
    if "exact" in combined:
        return "exact_path"
    return "other"


def _phv_summary(context: dict[str, object]) -> dict[str, object]:
    by_gress: dict[str, dict[str, int]] = {}
    for gress in ("ingress", "egress"):
        peak_containers = 0
        peak_bits = 0
        for stage in context.get("phv_allocation") or []:
            if not isinstance(stage, dict):
                continue
            containers = [
                item for item in stage.get(gress) or [] if isinstance(item, dict)
            ]
            peak_containers = max(peak_containers, len(containers))
            peak_bits = max(
                peak_bits,
                sum(int(item.get("word_bit_width", 0)) for item in containers),
            )
        by_gress[gress] = {
            "peak_allocated_containers": peak_containers,
            "peak_allocated_container_bits": peak_bits,
        }
    return by_gress


def summarize_context(context: dict[str, object]) -> dict[str, object]:
    by_stage: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    by_component: dict[str, dict[str, set[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    table_rows: list[dict[str, object]] = []
    used_stages: set[int] = set()

    for raw_table in context.get("tables") or []:
        if not isinstance(raw_table, dict):
            continue
        component = _component(raw_table)
        table_units: dict[str, set[tuple[int, int]]] = defaultdict(set)
        stages: set[int] = set()
        for root in _stage_table_roots(raw_table):
            for stage_table in _walk_stage_tables(root):
                raw_stage = stage_table.get("stage_number")
                if raw_stage is None or int(raw_stage) < 0:
                    continue
                stage = int(raw_stage)
                stages.add(stage)
                used_stages.add(stage)
                allocation = stage_table.get("memory_resource_allocation") or {}
                if not isinstance(allocation, dict):
                    continue
                memory_type = str(allocation.get("memory_type", ""))
                if not memory_type:
                    continue
                for unit in _memory_units(allocation):
                    by_stage[stage][memory_type].add(unit)
                    by_component[component][memory_type].add((stage, unit))
                    table_units[memory_type].add((stage, unit))
        if stages or table_units:
            table_rows.append(
                {
                    "name": raw_table.get("name"),
                    "table_type": raw_table.get("table_type"),
                    "logical_size": raw_table.get("size"),
                    "component": component,
                    "stages": sorted(stages),
                    "memory_units": {
                        key: len(value) for key, value in sorted(table_units.items())
                    },
                }
            )

    totals: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for stage, resources in by_stage.items():
        for memory_type, units in resources.items():
            totals[memory_type].update((stage, unit) for unit in units)

    stateful = {
        str(table.get("name")): {
            "logical_entries": int(table.get("size", 0)),
            "entry_width_bits": int(table.get("alu_width", 0))
            * (2 if table.get("dual_width_mode") else 1),
            "allocated_entries": sum(
                int(stage.get("size", 0))
                for stage in table.get("stage_tables") or []
                if isinstance(stage, dict)
            ),
            "component": _component(table),
        }
        for table in context.get("tables") or []
        if isinstance(table, dict) and table.get("table_type") == "stateful"
    }
    logical_stateful_bits: dict[str, int] = defaultdict(int)
    allocated_stateful_bits: dict[str, int] = defaultdict(int)
    for item in stateful.values():
        component = str(item["component"])
        width = int(item["entry_width_bits"])
        logical_stateful_bits[component] += int(item["logical_entries"]) * width
        allocated_stateful_bits[component] += int(item["allocated_entries"]) * width

    return {
        "format": "pleds_tofino_compiler_resources_v1",
        "program_name": context.get("program_name"),
        "compiler_version": context.get("compiler_version"),
        "target": context.get("target"),
        "used_stage_numbers": sorted(used_stages),
        "stage_count": len(used_stages),
        "pipeline_stage_span": max(used_stages) + 1 if used_stages else 0,
        "last_stage": max(used_stages) if used_stages else None,
        "memory_units": {key: len(value) for key, value in sorted(totals.items())},
        "memory_units_by_stage": {
            str(stage): {key: len(value) for key, value in sorted(resources.items())}
            for stage, resources in sorted(by_stage.items())
        },
        "memory_units_by_component": {
            component: {key: len(value) for key, value in sorted(resources.items())}
            for component, resources in sorted(by_component.items())
        },
        "stateful_tables": stateful,
        "logical_stateful_bits_by_component": dict(
            sorted(logical_stateful_bits.items())
        ),
        "allocated_stateful_bits_by_component": dict(
            sorted(allocated_stateful_bits.items())
        ),
        "dynamic_hash_calculations": len(
            context.get("dynamic_hash_calculations") or []
        ),
        "phv": _phv_summary(context),
        "tables": table_rows,
        "notes": [
            "Memory-unit counts are derived from compiler allocations in context.json.",
            "Logical stateful entries and compiler-allocated entries are reported separately.",
            "Component counts need not sum to the program total when the compiler shares a physical unit.",
        ],
    }


def summarize_context_file(path: str | Path) -> dict[str, object]:
    context_path = Path(path)
    return summarize_context(json.loads(context_path.read_text(encoding="utf-8")))
