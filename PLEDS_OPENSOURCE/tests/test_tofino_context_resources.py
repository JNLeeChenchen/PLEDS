from __future__ import annotations

from pleds.tofino.context_resources import summarize_context


def _allocation(memory_type: str, unit: int) -> dict[str, object]:
    return {
        "stage_number": 3,
        "size": 131_072,
        "memory_resource_allocation": {
            "memory_type": memory_type,
            "memory_units_and_vpns": [{"memory_units": [unit], "vpns": [0]}],
        },
    }


def test_context_summary_uses_compiler_allocations() -> None:
    context = {
        "program_name": "test",
        "compiler_version": "9.7.0",
        "target": "tofino",
        "tables": [
            {
                "name": "SwitchIngress.flow_filter0",
                "table_type": "stateful",
                "size": 4096,
                "alu_width": 1,
                "stage_tables": [_allocation("sram", 10)],
            },
            {
                "name": "SwitchIngress.flow_filter1",
                "table_type": "stateful",
                "size": 4096,
                "alu_width": 1,
                "stage_tables": [_allocation("sram", 11)],
            },
            {
                "name": "SwitchIngress.learned_selector",
                "table_type": "match",
                "size": 64,
                "match_attributes": {"stage_tables": [_allocation("tcam", 2)]},
            },
        ],
        "dynamic_hash_calculations": [{}, {}],
        "phv_allocation": [
            {
                "stage_number": 3,
                "ingress": [
                    {"phv_number": 1, "word_bit_width": 32},
                    {"phv_number": 64, "word_bit_width": 8},
                ],
                "egress": [],
            }
        ],
    }
    summary = summarize_context(context)
    assert summary["memory_units"] == {"sram": 2, "tcam": 1}
    assert summary["memory_units_by_component"]["flow_filter_state"]["sram"] == 2
    assert summary["memory_units_by_component"]["inference_mat"]["tcam"] == 1
    assert summary["stateful_tables"]["SwitchIngress.flow_filter0"] == {
        "logical_entries": 4096,
        "entry_width_bits": 1,
        "allocated_entries": 131_072,
        "component": "flow_filter_state",
    }
    assert summary["stage_count"] == 1
    assert summary["pipeline_stage_span"] == 4
    assert summary["last_stage"] == 3
    assert summary["logical_stateful_bits_by_component"]["flow_filter_state"] == 8192
    assert (
        summary["allocated_stateful_bits_by_component"]["flow_filter_state"] == 262_144
    )
    assert summary["dynamic_hash_calculations"] == 2
    assert summary["phv"]["ingress"]["peak_allocated_container_bits"] == 40


def test_counting_rows_are_reported_as_record_state() -> None:
    context = {
        "tables": [
            {
                "name": "SwitchIngress.row0_packet_count",
                "table_type": "stateful",
                "size": 256,
                "alu_width": 32,
                "stage_tables": [_allocation("sram", 0)],
            }
        ]
    }
    summary = summarize_context(context)
    assert summary["memory_units_by_component"]["flow_record_state"]["sram"] == 1
    assert summary["logical_stateful_bits_by_component"]["flow_record_state"] == 8192


def test_flowradar_state_is_split_into_filter_and_record_components() -> None:
    context = {
        "tables": [
            {
                "name": "SwitchIngress.p0_flow_filter0",
                "table_type": "stateful",
                "size": 2048,
                "alu_width": 1,
                "stage_tables": [_allocation("sram", 0)],
            },
            {
                "name": "SwitchIngress.p0_row0_key_src",
                "table_type": "stateful",
                "size": 128,
                "alu_width": 32,
                "stage_tables": [_allocation("sram", 1)],
            },
        ]
    }
    summary = summarize_context(context)
    assert summary["memory_units_by_component"]["flow_filter_state"]["sram"] == 1
    assert summary["memory_units_by_component"]["flow_record_state"]["sram"] == 1
