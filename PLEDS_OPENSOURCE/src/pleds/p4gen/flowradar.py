"""Compile-ready Tofino generation for model-partitioned FlowRadar."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from pleds.hash_spec import p4_hash_compute_action, p4_hash_externs, register_hash_plan
from pleds.ir import ResourceEstimate
from pleds.key_bits import feature_names
from pleds.p4gen.ternary import lower_binary_rule_plan_to_ternary
from pleds.tofino.bfrt import (
    direct_prefix_selector_bfrt_plan,
    direct_prefix_selector_fields,
    no_selector_bfrt_plan,
    runtime_control_py,
)


@dataclass(frozen=True)
class FlowRadarP4Artifact:
    p4_path: Path
    runtime_path: Path
    resource_path: Path
    feature_path: Path
    bfrt_path: Path
    runtime_control_path: Path


def _power_of_two(value: int, label: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{label} must be a power of two")


def _selector_entries(rule_plan: dict[str, object]) -> list[dict[str, object]]:
    entries = []
    for lowered in lower_binary_rule_plan_to_ternary(rule_plan):
        entry = lowered.as_dict()
        entry["action"] = (
            "set_partition0"
            if entry["action"] == "set_model_positive"
            else "set_partition1"
        )
        entries.append(entry)
    return entries


def _selector_declaration(feature_count: int, selector_size: int) -> str:
    keys = "\n".join(
        f"            {name}: ternary;"
        for name, _width in direct_prefix_selector_fields(feature_count)
    )
    return f"""
    action set_partition0() {{
        ig_md.partition_id = 1w0;
    }}

    action set_partition1() {{
        ig_md.partition_id = 1w1;
    }}

    table partition_selector {{
        key = {{
{keys}
        }}
        actions = {{
            set_partition0;
            set_partition1;
        }}
        default_action = set_partition1();
        size = {selector_size};
    }}
"""


def _packed_registers(
    *,
    partition_count: int,
    filter_bank_entries: int,
    counting_row_entries: int,
    filter_hashes: int,
    counting_hashes: int,
) -> str:
    total_filter_entries = partition_count * filter_bank_entries
    total_counting_entries = partition_count * counting_row_entries
    filter_index_width = int(math.log2(total_filter_entries))
    counting_index_width = int(math.log2(total_counting_entries))
    filters = []
    for bank in range(filter_hashes):
        filters.append(
            f"""
    Register<bit<1>, bit<{filter_index_width}>>({total_filter_entries}, 1w0) flow_filter{bank};
    RegisterAction<bit<1>, bit<{filter_index_width}>, bit<1>>(flow_filter{bank}) flow_filter{bank}_set = {{
        void apply(inout bit<1> value, out bit<1> output) {{
            output = value;
            value = 1w1;
        }}
    }};
"""
        )
    rows = []
    for row in range(counting_hashes):
        rows.append(
            f"""
    Register<bit<32>, bit<{counting_index_width}>>({total_counting_entries}, 32w0) row{row}_key_src;
    RegisterAction<bit<32>, bit<{counting_index_width}>, bit<32>>(row{row}_key_src) row{row}_key_src_update = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            if (ig_md.new_flow == 1w1) {{
                value = value ^ ig_md.flow_src;
            }}
            output = value;
        }}
    }};

    Register<bit<32>, bit<{counting_index_width}>>({total_counting_entries}, 32w0) row{row}_key_dst;
    RegisterAction<bit<32>, bit<{counting_index_width}>, bit<32>>(row{row}_key_dst) row{row}_key_dst_update = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            if (ig_md.new_flow == 1w1) {{
                value = value ^ ig_md.flow_dst;
            }}
            output = value;
        }}
    }};

    Register<bit<32>, bit<{counting_index_width}>>({total_counting_entries}, 32w0) row{row}_key_ports;
    RegisterAction<bit<32>, bit<{counting_index_width}>, bit<32>>(row{row}_key_ports) row{row}_key_ports_update = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            if (ig_md.new_flow == 1w1) {{
                value = value ^ ig_md.flow_ports;
            }}
            output = value;
        }}
    }};

    Register<bit<32>, bit<{counting_index_width}>>({total_counting_entries}, 32w0) row{row}_key_protocol;
    RegisterAction<bit<32>, bit<{counting_index_width}>, bit<32>>(row{row}_key_protocol) row{row}_key_protocol_update = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            if (ig_md.new_flow == 1w1) {{
                value = value ^ ig_md.flow_protocol;
            }}
            output = value;
        }}
    }};

    Register<bit<32>, bit<{counting_index_width}>>({total_counting_entries}, 32w0) row{row}_flow_count;
    RegisterAction<bit<32>, bit<{counting_index_width}>, bit<32>>(row{row}_flow_count) row{row}_flow_count_update = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            if (ig_md.new_flow == 1w1) {{
                value = value + 1;
            }}
            output = value;
        }}
    }};

    Register<bit<32>, bit<{counting_index_width}>>({total_counting_entries}, 32w0) row{row}_packet_count;
    RegisterAction<bit<32>, bit<{counting_index_width}>, bit<32>>(row{row}_packet_count) row{row}_packet_count_update = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            value = value + 1;
            output = value;
        }}
    }};
"""
        )
    return "".join(filters + rows)


def _packed_apply(
    *,
    partition_count: int,
    filter_bank_entries: int,
    counting_row_entries: int,
    filter_hashes: int,
    counting_hashes: int,
) -> str:
    filter_local_width = int(math.log2(filter_bank_entries))
    counting_local_width = int(math.log2(counting_row_entries))
    filter_index = (
        f"ig_md.partition_id ++ ig_md.flow_hash{{hash_id}}_value[{filter_local_width - 1}:0]"
        if partition_count == 2
        else f"ig_md.flow_hash{{hash_id}}_value[{filter_local_width - 1}:0]"
    )
    counting_index = (
        f"ig_md.partition_id ++ ig_md.flow_hash{{hash_id}}_value[{counting_local_width - 1}:0]"
        if partition_count == 2
        else f"ig_md.flow_hash{{hash_id}}_value[{counting_local_width - 1}:0]"
    )
    filter_reads = "\n".join(
        f"            ig_md.filter_old{bank} = flow_filter{bank}_set.execute({filter_index.format(hash_id=counting_hashes + bank)});"
        for bank in range(filter_hashes)
    )
    miss = " || ".join(
        f"ig_md.filter_old{bank} == 1w0" for bank in range(filter_hashes)
    )
    updates = "\n".join(
        f"            ig_md.stateful_out = row{row}_key_src_update.execute({counting_index.format(hash_id=row)});\n"
        f"            ig_md.stateful_out = row{row}_key_dst_update.execute({counting_index.format(hash_id=row)});\n"
        f"            ig_md.stateful_out = row{row}_key_ports_update.execute({counting_index.format(hash_id=row)});\n"
        f"            ig_md.stateful_out = row{row}_key_protocol_update.execute({counting_index.format(hash_id=row)});\n"
        f"            ig_md.stateful_out = row{row}_flow_count_update.execute({counting_index.format(hash_id=row)});\n"
        f"            ig_md.stateful_out = row{row}_packet_count_update.execute({counting_index.format(hash_id=row)});"
        for row in range(counting_hashes)
    )
    return f"""{filter_reads}
            if ({miss}) {{
                ig_md.new_flow = 1w1;
            }}
{updates}"""


def _p4_text(
    *,
    feature_count: int,
    selector_size: int,
    filter_bank_entries: Sequence[int],
    counting_row_entries: Sequence[int],
    filter_hashes: int,
    counting_hashes: int,
) -> str:
    if feature_count != 104:
        raise ValueError(
            "FlowRadar lowering currently requires 104 direct five-tuple bits"
        )
    partition_count = len(filter_bank_entries)
    if partition_count not in {1, 2} or len(counting_row_entries) != partition_count:
        raise ValueError(
            "FlowRadar lowering requires one or two equal-dimension partitions"
        )
    if filter_hashes <= 0 or counting_hashes <= 0:
        raise ValueError("FlowRadar hash counts must be positive")
    if filter_hashes + counting_hashes > 8:
        raise ValueError("the target hash profile provides at most eight hashes")
    for partition, value in enumerate(filter_bank_entries):
        _power_of_two(int(value), f"partition {partition} filter bank size")
    for partition, value in enumerate(counting_row_entries):
        _power_of_two(int(value), f"partition {partition} counting row size")
    if partition_count == 2 and len(set(filter_bank_entries)) != 1:
        raise ValueError(
            "packed FlowRadar partitions require equal filter-bank dimensions"
        )
    if partition_count == 2 and len(set(counting_row_entries)) != 1:
        raise ValueError(
            "packed FlowRadar partitions require equal counting-row dimensions"
        )

    hash_count = filter_hashes + counting_hashes
    metadata_hashes = "\n".join(
        f"        bit<32> flow_hash{index}_value;" for index in range(hash_count)
    )
    filter_old = "\n".join(
        f"        bit<1> filter_old{bank};" for bank in range(filter_hashes)
    )
    hash_actions = "\n".join(
        p4_hash_compute_action("flow", "flow_hash", index)
        for index in range(hash_count)
    )
    hash_apply = "\n".join(
        f"            compute_flow_hash{index}_table.apply();"
        for index in range(hash_count)
    )
    registers = _packed_registers(
        partition_count=partition_count,
        filter_bank_entries=int(filter_bank_entries[0]),
        counting_row_entries=int(counting_row_entries[0]),
        filter_hashes=filter_hashes,
        counting_hashes=counting_hashes,
    )
    packed_apply = _packed_apply(
        partition_count=partition_count,
        filter_bank_entries=int(filter_bank_entries[0]),
        counting_row_entries=int(counting_row_entries[0]),
        filter_hashes=filter_hashes,
        counting_hashes=counting_hashes,
    )
    selector_declaration = (
        _selector_declaration(feature_count, selector_size)
        if partition_count == 2
        else ""
    )
    selector_apply = (
        "            partition_selector.apply();" if partition_count == 2 else ""
    )
    reset_filter = "\n".join(
        f"        ig_md.filter_old{bank} = 1w0;" for bank in range(filter_hashes)
    )

    return f"""/* Auto-generated PLEDS model-partitioned FlowRadar. */

#include <core.p4>
#include <tna.p4>

const bit<32> SELECTOR_SIZE = {selector_size};

header ethernet_h {{
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}}

header ipv4_h {{
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3> flags;
    bit<13> frag_offset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}}

header tcp_h {{
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4> data_offset;
    bit<4> res;
    bit<8> flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}}

header udp_h {{
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> len;
    bit<16> checksum;
}}

struct headers_t {{
    ethernet_h ethernet;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
}}

struct metadata_t {{
        bit<1> partition_id;
        bit<1> new_flow;
        bit<16> l4_sport;
        bit<16> l4_dport;
        bit<32> flow_src;
        bit<32> flow_dst;
        bit<32> flow_ports;
        bit<32> flow_protocol;
        bit<32> stateful_out;
{filter_old}
{metadata_hashes}
}}

struct egress_metadata_t {{
}}

parser SwitchIngressParser(
    packet_in pkt,
    out headers_t hdr,
    out metadata_t ig_md,
    out ingress_intrinsic_metadata_t ig_intr_md) {{
    state start {{
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }}

    state parse_ethernet {{
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {{
            16w0x0800: parse_ipv4;
            default: accept;
        }}
    }}

    state parse_ipv4 {{
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.ihl, hdr.ipv4.flags[0:0], hdr.ipv4.frag_offset, hdr.ipv4.protocol) {{
            (4w5, 1w0, 13w0, 8w6): parse_tcp;
            (4w5, 1w0, 13w0, 8w17): parse_udp;
            default: accept;
        }}
    }}

    state parse_tcp {{
        pkt.extract(hdr.tcp);
        transition accept;
    }}

    state parse_udp {{
        pkt.extract(hdr.udp);
        transition accept;
    }}
}}

control SwitchIngress(
    inout headers_t hdr,
    inout metadata_t ig_md,
    in ingress_intrinsic_metadata_t ig_intr_md,
    in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {{

{p4_hash_externs("flow", hash_count)}
{hash_actions}
{selector_declaration}
{registers}

    apply {{
        ig_md.partition_id = 1w1;
        ig_md.new_flow = 1w0;
        ig_md.l4_sport = 16w0;
        ig_md.l4_dport = 16w0;
        ig_md.flow_src = 32w0;
        ig_md.flow_dst = 32w0;
        ig_md.flow_ports = 32w0;
        ig_md.flow_protocol = 32w0;
        ig_md.stateful_out = 32w0;
{reset_filter}
        if (hdr.tcp.isValid()) {{
            ig_md.l4_sport = hdr.tcp.src_port;
            ig_md.l4_dport = hdr.tcp.dst_port;
        }} else if (hdr.udp.isValid()) {{
            ig_md.l4_sport = hdr.udp.src_port;
            ig_md.l4_dport = hdr.udp.dst_port;
        }}
        if (hdr.ipv4.isValid() && hdr.ipv4.ihl == 4w5 &&
            hdr.ipv4.flags[0:0] == 1w0 && hdr.ipv4.frag_offset == 13w0) {{
            ig_md.flow_src = hdr.ipv4.src_addr;
            ig_md.flow_dst = hdr.ipv4.dst_addr;
            ig_md.flow_ports = ig_md.l4_sport ++ ig_md.l4_dport;
            ig_md.flow_protocol = 24w0 ++ hdr.ipv4.protocol;
{selector_apply}
{hash_apply}
{packed_apply}
        }}
    }}
}}

control SwitchIngressDeparser(
    packet_out pkt,
    inout headers_t hdr,
    in metadata_t ig_md,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {{
    apply {{
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
    }}
}}

parser SwitchEgressParser(
    packet_in pkt,
    out headers_t hdr,
    out egress_metadata_t eg_md,
    out egress_intrinsic_metadata_t eg_intr_md) {{
    state start {{
        pkt.extract(eg_intr_md);
        transition accept;
    }}
}}

control SwitchEgress(
    inout headers_t hdr,
    inout egress_metadata_t eg_md,
    in egress_intrinsic_metadata_t eg_intr_md,
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {{
    apply {{ }}
}}

control SwitchEgressDeparser(
    packet_out pkt,
    inout headers_t hdr,
    in egress_metadata_t eg_md,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {{
    apply {{ }}
}}

Pipeline(
    SwitchIngressParser(),
    SwitchIngress(),
    SwitchIngressDeparser(),
    SwitchEgressParser(),
    SwitchEgress(),
    SwitchEgressDeparser()) pipe;

Switch(pipe) main;
"""


def write_flowradar_artifacts(
    *,
    out_dir: Path,
    rule_plan: dict[str, object],
    resource_estimate: ResourceEstimate,
    filter_bank_entries: Sequence[int],
    counting_row_entries: Sequence[int],
    filter_hashes: int,
    counting_hashes: int,
    backend_type: str = "partitioned_flowradar",
) -> FlowRadarP4Artifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_count = int(rule_plan["feature_count"])
    entries = _selector_entries(rule_plan)
    selector_size = max(1, len(entries))
    p4_path = out_dir / "pleds_partitioned_flowradar_tna.p4"
    runtime_path = out_dir / "runtime_entries.json"
    resource_path = out_dir / "resource_plan.json"
    feature_path = out_dir / "feature_plan.json"
    bfrt_path = out_dir / "bfrt_plan.json"
    runtime_control_path = out_dir / "runtime_control.py"

    p4_path.write_text(
        _p4_text(
            feature_count=feature_count,
            selector_size=selector_size,
            filter_bank_entries=filter_bank_entries,
            counting_row_entries=counting_row_entries,
            filter_hashes=filter_hashes,
            counting_hashes=counting_hashes,
        ),
        encoding="utf-8",
    )
    feature_payload = {
        "format": "pleds_feature_plan_v1",
        "schema": "normalized_ipv4_five_tuple_prefix_bits_v1",
        "feature_count": feature_count,
        "fields": [
            {"name": name, "width": width, "implementation": "direct_match"}
            for name, width in direct_prefix_selector_fields(feature_count)
        ],
        "features": [
            {"index": index, "name": name, "implementation": "direct_match"}
            for index, name in enumerate(feature_names()[:feature_count])
        ],
    }
    partition_count = len(filter_bank_entries)
    if partition_count == 1 and entries:
        raise ValueError("unpartitioned FlowRadar must not contain selector rules")
    if partition_count == 2 and not entries:
        raise ValueError("partitioned FlowRadar requires selector rules")
    all_filter_hashes = register_hash_plan(
        table_size=int(filter_bank_entries[0]),
        count=filter_hashes + counting_hashes,
    )
    counting_hash_plan = register_hash_plan(
        table_size=int(counting_row_entries[0]),
        count=counting_hashes,
    )
    hash_plan = {
        "profile": all_filter_hashes["profile"],
        "flow_filter": {
            "hash_ids": list(range(counting_hashes, counting_hashes + filter_hashes)),
            "table_size": int(filter_bank_entries[0]),
            "specs": all_filter_hashes["specs"][counting_hashes:],
        },
        "counting_table": {
            "hash_ids": list(range(counting_hashes)),
            "table_size": int(counting_row_entries[0]),
            "specs": counting_hash_plan["specs"],
        },
        "packed_index": "partition_prefix_then_low_local_hash_bits",
    }
    selector = (
        direct_prefix_selector_bfrt_plan(
            "pipe.SwitchIngress.partition_selector", entries, feature_count
        )
        if partition_count == 2
        else no_selector_bfrt_plan(model_type="none")
    )
    register_names = [
        f"SwitchIngress.flow_filter{bank}" for bank in range(filter_hashes)
    ]
    for row in range(counting_hashes):
        register_names.extend(
            [
                f"SwitchIngress.row{row}_key_src",
                f"SwitchIngress.row{row}_key_dst",
                f"SwitchIngress.row{row}_key_ports",
                f"SwitchIngress.row{row}_key_protocol",
                f"SwitchIngress.row{row}_flow_count",
                f"SwitchIngress.row{row}_packet_count",
            ]
        )
    total_filter_entries = partition_count * int(filter_bank_entries[0])
    total_counting_entries = partition_count * int(counting_row_entries[0])
    state_reset_registers = [
        {
            "name": f"pipe.SwitchIngress.flow_filter{bank}",
            "entry_count": total_filter_entries,
            "write_value": 0,
        }
        for bank in range(filter_hashes)
    ]
    for row in range(counting_hashes):
        state_reset_registers.extend(
            {
                "name": f"pipe.SwitchIngress.row{row}_{suffix}",
                "entry_count": total_counting_entries,
                "write_value": 0,
            }
            for suffix in (
                "key_src",
                "key_dst",
                "key_ports",
                "key_protocol",
                "flow_count",
                "packet_count",
            )
        )
    bfrt_payload = {
        "format": "pleds_bfrt_plan_v1",
        "program_name": "pleds_partitioned_flowradar_tna",
        "feature_tables": {
            "format": "pleds_bfrt_feature_plan_v1",
            "entries": [],
            "entry_count": 0,
        },
        "selector": selector,
        "register_initialization": {
            "format": "pleds_register_init_v1",
            "registers": [],
        },
        "state_reset": {
            "format": "pleds_register_reset_v1",
            "scope": "before_each_evaluation_slot",
            "registers": state_reset_registers,
        },
        "flowradar": {
            "filter_bank_entries": list(filter_bank_entries),
            "counting_row_entries": list(counting_row_entries),
            "filter_hashes": filter_hashes,
            "counting_hashes": counting_hashes,
            "registers": register_names,
            "counting_cell_physical_bits": 192,
            "hash_plan": hash_plan,
            "state_reset": "control_plane_zeroes_all_registers_before_each_evaluation_slot",
        },
        "summary": {
            "feature_entry_count": 0,
            "selector_entry_count": len(entries),
            "register_write_count": 0,
        },
        "live_safe": False,
    }
    runtime_payload = {
        "format": "pleds_runtime_entries_v1",
        "selector_entries": entries,
        "register_initialization": [],
        "state_reset": bfrt_payload["state_reset"],
        "flowradar": bfrt_payload["flowradar"],
    }
    resource_payload = {
        "format": "pleds_resource_plan_v1",
        "backend": backend_type,
        "resource_estimate": resource_estimate.as_dict(),
        "layout": bfrt_payload["flowradar"],
        "semantic_properties": {
            "single_partition_update_per_packet": True,
            "flow_filter_mark_before_counting_update": True,
            "offline_decode_required": True,
        },
    }
    for path, payload in (
        (runtime_path, runtime_payload),
        (resource_path, resource_payload),
        (feature_path, feature_payload),
        (bfrt_path, bfrt_payload),
    ):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    runtime_control_path.write_text(runtime_control_py(), encoding="utf-8")
    return FlowRadarP4Artifact(
        p4_path=p4_path,
        runtime_path=runtime_path,
        resource_path=resource_path,
        feature_path=feature_path,
        bfrt_path=bfrt_path,
        runtime_control_path=runtime_control_path,
    )
