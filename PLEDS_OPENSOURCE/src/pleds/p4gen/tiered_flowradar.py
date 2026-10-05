"""Compile-ready Tofino generation for model-guided tiered FlowRadar."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from pleds.hash_spec import p4_hash_compute_action, p4_hash_extern, register_hash_plan
from pleds.ir import ResourceEstimate
from pleds.features import COMPACT_PACKET_FEATURES, compact_feature_plan
from pleds.key_bits import feature_names
from pleds.p4gen.flowradar import _packed_apply, _packed_registers
from pleds.p4gen.ternary import lower_binary_rule_plan_to_ternary
from pleds.tofino.bfrt import (
    direct_prefix_selector_bfrt_plan,
    direct_prefix_selector_fields,
    feature_bfrt_plan,
    selector_bfrt_plan,
    runtime_control_py,
)


@dataclass(frozen=True)
class TieredFlowRadarP4Artifact:
    p4_path: Path
    runtime_path: Path
    resource_path: Path
    feature_path: Path
    bfrt_path: Path
    runtime_control_path: Path


def _power_of_two_width(value: int, label: str) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{label} must be a positive power of two")
    return max(1, int(math.log2(value)))


def _selector_declaration(feature_count: int, selector_size: int) -> str:
    fields = (
        tuple((f"ig_md.f{i}", 1) for i in range(12))
        if feature_count == 12
        else direct_prefix_selector_fields(feature_count)
    )
    keys = "\n".join(f"            {name}: ternary;" for name, _width in fields)
    return f"""
    action set_model_positive() {{
        ig_md.model_positive = 1w1;
    }}

    action set_model_negative() {{
        ig_md.model_positive = 1w0;
    }}

    table learned_selector {{
        key = {{
{keys}
        }}
        actions = {{
            set_model_positive;
            set_model_negative;
        }}
        default_action = set_model_negative();
        size = {selector_size};
    }}
"""


def _compact_parser_features() -> tuple[str, dict[str, str], str]:
    initialization = "\n".join(
        f"        ig_md.f{feature.index} = {feature.p4_expr};"
        for feature in COMPACT_PACKET_FEATURES
        if feature.index not in {4, 5, 6, 7, 8}
    )
    initialization += "\n" + "\n".join(
        f"        ig_md.f{i} = 1w0;" for i in (4, 5, 6, 7, 8)
    )
    entries = feature_bfrt_plan()["entries"]
    bodies, states = {}, []
    for protocol, index in (("tcp", 7), ("udp", 8)):
        bodies[protocol] = (
            f"        ig_md.f{index} = 1w1;\n        transition {protocol}_src_service;"
        )
        for label, port_field, feature, next_state, table in (
            (
                "src_service",
                "src_port",
                4,
                f"{protocol}_dst_service",
                "src_service_port_table",
            ),
            (
                "dst_service",
                "dst_port",
                5,
                f"{protocol}_dst_ephemeral",
                "dst_service_port_table",
            ),
            ("dst_ephemeral", "dst_port", 6, "accept", "dst_ephemeral_port_table"),
        ):
            matches = []
            for entry in entries:
                if entry["table"].endswith("." + table):
                    key = next(iter(entry["key"].values()))
                    matches.append(
                        f"            (16w{key['value']} &&& 16w{key['mask']}): {protocol}_{label}_yes;"
                    )
            states.append(
                f"""
    state {protocol}_{label} {{
        transition select(hdr.{protocol}.{port_field}) {{
{chr(10).join(matches)}
            default: {next_state};
        }}
    }}
    state {protocol}_{label}_yes {{
        ig_md.f{feature} = 1w1;
        transition {next_state};
    }}
"""
            )
    return initialization, bodies, "".join(states)


def _exact_metadata(exact_bank_entries: Sequence[int]) -> str:
    fields: list[str] = []
    for bank, entries in enumerate(exact_bank_entries):
        width = _power_of_two_width(int(entries), f"exact bank {bank} size")
        fields.extend(
            [
                f"        bit<2> exact_state{bank};",
                f"        bit<1> exact_accept{bank};",
                f"        bit<{width}> exact_index{bank};",
                f"        bit<32> exact_tag{bank};",
                f"        bit<32> exact_count_out{bank};",
            ]
        )
    return "\n".join(fields)


def _exact_registers(exact_bank_entries: Sequence[int]) -> str:
    declarations: list[str] = []
    for bank, entries_raw in enumerate(exact_bank_entries):
        entries = int(entries_raw)
        width = _power_of_two_width(entries, f"exact bank {bank} size")
        declarations.append(
            f"""
    Register<bit<32>, bit<{width}>>({entries}, 32w0) exact{bank}_tags;
    RegisterAction<bit<32>, bit<{width}>, bit<2>>(exact{bank}_tags) exact{bank}_tag_claim = {{
        void apply(inout bit<32> value, out bit<2> output) {{
            if (value == 32w0) {{
                value = ig_md.exact_tag{bank};
                output = 2w1;
            }} else if (value == ig_md.exact_tag{bank}) {{
                output = 2w2;
            }} else {{
                output = 2w0;
            }}
        }}
    }};
"""
        )
        for suffix, source in (
            ("key_src", "flow_src"),
            ("key_dst", "flow_dst"),
            ("key_ports", "flow_ports"),
            ("key_protocol", "flow_protocol"),
        ):
            declarations.append(
                f"""
    Register<bit<32>, bit<{width}>>({entries}, 32w0) exact{bank}_{suffix};
    RegisterAction<bit<32>, bit<{width}>, bit<32>>(exact{bank}_{suffix}) exact{bank}_{suffix}_store = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            if (ig_md.exact_state{bank} == 2w1) {{
                value = ig_md.{source};
            }}
            output = value;
        }}
    }};
"""
            )
        declarations.append(
            f"""
    Register<bit<32>, bit<{width}>>({entries}, 32w0) exact{bank}_counts;
    RegisterAction<bit<32>, bit<{width}>, bit<32>>(exact{bank}_counts) exact{bank}_count_inc = {{
        void apply(inout bit<32> value, out bit<32> output) {{
            value = value + 1;
            output = value;
        }}
    }};
"""
        )
    return "".join(declarations)


def _exact_initialization(exact_bank_entries: Sequence[int]) -> str:
    lines: list[str] = []
    for bank, entries in enumerate(exact_bank_entries):
        width = _power_of_two_width(int(entries), f"exact bank {bank} size")
        lines.extend(
            [
                f"        ig_md.exact_state{bank} = 2w0;",
                f"        ig_md.exact_accept{bank} = 1w0;",
                f"        ig_md.exact_index{bank} = {width}w0;",
                f"        ig_md.exact_tag{bank} = 32w0;",
                f"        ig_md.exact_count_out{bank} = 32w0;",
            ]
        )
    return "\n".join(lines)


def _exact_probe(exact_bank_entries: Sequence[int]) -> str:
    blocks: list[str] = []
    for bank, entries in enumerate(exact_bank_entries):
        width = _power_of_two_width(int(entries), f"exact bank {bank} size")
        guard = (
            "ig_md.model_positive == 1w1"
            if bank == 0
            else f"ig_md.exact_accept{bank - 1} == 1w0"
        )
        index_hash = 6 + bank
        blocks.append(
            f"""
            if ({guard}) {{
                compute_flow_hash{index_hash}_table.apply();
                ig_md.exact_tag{bank} = ig_md.flow_hash{index_hash}_value | 32w0x80000000;
                ig_md.exact_index{bank} = ig_md.flow_hash{index_hash}_value[{width - 1}:0];
                ig_md.exact_state{bank} = exact{bank}_tag_claim.execute(ig_md.exact_index{bank});
                if (ig_md.exact_state{bank} != 2w0) {{
                    ig_md.exact_accept{bank} = 1w1;
                    ig_md.stateful_out = exact{bank}_key_src_store.execute(ig_md.exact_index{bank});
                    ig_md.stateful_out = exact{bank}_key_dst_store.execute(ig_md.exact_index{bank});
                    ig_md.stateful_out = exact{bank}_key_ports_store.execute(ig_md.exact_index{bank});
                    ig_md.stateful_out = exact{bank}_key_protocol_store.execute(ig_md.exact_index{bank});
                }}
            }}
"""
        )
    return "".join(blocks)


def _exact_or_fallback_apply(
    *,
    exact_bank_entries: Sequence[int],
    fallback_filter_bank_entries: int,
    fallback_counting_row_entries: int,
    flow_filter_hashes: int,
    counting_hashes: int,
) -> str:
    exact_probe = _exact_probe(exact_bank_entries)
    accepted = " || ".join(
        f"ig_md.exact_accept{bank} == 1w1" for bank in range(len(exact_bank_entries))
    )
    count_updates = " else ".join(
        f"if (ig_md.exact_accept{bank} == 1w1) {{\n"
        f"                ig_md.exact_count_out{bank} = exact{bank}_count_inc.execute(ig_md.exact_index{bank});\n"
        f"            }}"
        for bank in range(len(exact_bank_entries))
    )
    fallback_hashes = "\n".join(
        f"                compute_flow_hash{hash_id}_table.apply();"
        for hash_id in range(flow_filter_hashes + counting_hashes)
    )
    fallback_update = _packed_apply(
        partition_count=1,
        filter_bank_entries=fallback_filter_bank_entries,
        counting_row_entries=fallback_counting_row_entries,
        filter_hashes=flow_filter_hashes,
        counting_hashes=counting_hashes,
    )
    return f"""            learned_selector.apply();
{exact_probe}
            if ({accepted}) {{
                {count_updates}
            }} else {{
{fallback_hashes}
{fallback_update}
            }}"""


def _p4_text(
    *,
    feature_count: int,
    selector_size: int,
    exact_bank_entries: Sequence[int],
    fallback_filter_bank_entries: int,
    fallback_counting_row_entries: int,
    flow_filter_hashes: int,
    counting_hashes: int,
) -> str:
    if feature_count not in {12, 104}:
        raise ValueError(
            "tiered_flowradar requires 12 compact features or 104 five-tuple bits"
        )
    if len(exact_bank_entries) not in {1, 2}:
        raise ValueError("tiered_flowradar requires one or two exact banks")
    for bank, entries in enumerate(exact_bank_entries):
        _power_of_two_width(int(entries), f"exact bank {bank} size")
    filter_width = _power_of_two_width(
        fallback_filter_bank_entries, "fallback filter bank size"
    )
    counting_width = _power_of_two_width(
        fallback_counting_row_entries, "fallback counting row size"
    )
    if flow_filter_hashes <= 0 or counting_hashes <= 0:
        raise ValueError("FlowRadar hash counts must be positive")
    if flow_filter_hashes + counting_hashes > 4:
        raise ValueError("tiered_flowradar reserves hash ids 4--7 for exact banks")

    hash_ids = list(range(flow_filter_hashes + counting_hashes))
    for bank in range(len(exact_bank_entries)):
        hash_ids.append(6 + bank)
    hash_ids = sorted(set(hash_ids))
    hash_metadata = "\n".join(
        f"        bit<32> flow_hash{hash_id}_value;" for hash_id in hash_ids
    )
    hash_externs = "\n".join(p4_hash_extern("flow", hash_id) for hash_id in hash_ids)
    hash_actions = "\n".join(
        p4_hash_compute_action("flow", "flow_hash", hash_id) for hash_id in hash_ids
    )
    filter_old = "\n".join(
        f"        bit<1> filter_old{bank};" for bank in range(flow_filter_hashes)
    )
    reset_filter = "\n".join(
        f"        ig_md.filter_old{bank} = 1w0;" for bank in range(flow_filter_hashes)
    )
    selector = _selector_declaration(feature_count, selector_size)
    feature_fields = feature_declarations = feature_apply = ""
    parser_features = parser_feature_states = ""
    parser_l4 = {
        "tcp": "        transition accept;",
        "udp": "        transition accept;",
    }
    if feature_count == 12:
        feature_fields = "\n".join(f"        bit<1> f{i};" for i in range(12))
        parser_features, parser_l4, parser_feature_states = _compact_parser_features()
    exact_metadata = _exact_metadata(exact_bank_entries)
    exact_registers = _exact_registers(exact_bank_entries)
    fallback_registers = _packed_registers(
        partition_count=1,
        filter_bank_entries=fallback_filter_bank_entries,
        counting_row_entries=fallback_counting_row_entries,
        filter_hashes=flow_filter_hashes,
        counting_hashes=counting_hashes,
    )
    exact_initialization = _exact_initialization(exact_bank_entries)
    update_path = _exact_or_fallback_apply(
        exact_bank_entries=exact_bank_entries,
        fallback_filter_bank_entries=fallback_filter_bank_entries,
        fallback_counting_row_entries=fallback_counting_row_entries,
        flow_filter_hashes=flow_filter_hashes,
        counting_hashes=counting_hashes,
    )

    return f"""/* Auto-generated PLEDS model-guided tiered FlowRadar. */

#include <core.p4>
#include <tna.p4>

const bit<32> SELECTOR_SIZE = {selector_size};
const bit<32> FALLBACK_FILTER_BANK_ENTRIES = {fallback_filter_bank_entries};
const bit<32> FALLBACK_COUNTING_ROW_ENTRIES = {fallback_counting_row_entries};

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
        bit<1> model_positive;
        bit<1> new_flow;
        bit<16> l4_sport;
        bit<16> l4_dport;
        bit<32> flow_src;
        bit<32> flow_dst;
        bit<32> flow_ports;
        bit<32> flow_protocol;
        bit<32> stateful_out;
{filter_old}
{hash_metadata}
{exact_metadata}
{feature_fields}
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
{parser_features}
        transition select(hdr.ipv4.ihl, hdr.ipv4.flags[0:0], hdr.ipv4.frag_offset, hdr.ipv4.protocol) {{
            (4w5, 1w0, 13w0, 8w6): parse_tcp;
            (4w5, 1w0, 13w0, 8w17): parse_udp;
            default: accept;
        }}
    }}

    state parse_tcp {{
        pkt.extract(hdr.tcp);
{parser_l4['tcp']}
    }}

    state parse_udp {{
        pkt.extract(hdr.udp);
{parser_l4['udp']}
    }}
{parser_feature_states}
}}

control SwitchIngress(
    inout headers_t hdr,
    inout metadata_t ig_md,
    in ingress_intrinsic_metadata_t ig_intr_md,
    in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {{

{hash_externs}
{hash_actions}
{selector}
{feature_declarations}
{exact_registers}
{fallback_registers}

    apply {{
        ig_md.partition_id = 1w0;
        ig_md.model_positive = 1w0;
        ig_md.new_flow = 1w0;
        ig_md.l4_sport = 16w0;
        ig_md.l4_dport = 16w0;
        ig_md.flow_src = 32w0;
        ig_md.flow_dst = 32w0;
        ig_md.flow_ports = 32w0;
        ig_md.flow_protocol = 32w0;
        ig_md.stateful_out = 32w0;
{reset_filter}
{exact_initialization}
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
{feature_apply}
{update_path}
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


def write_tiered_flowradar_artifacts(
    *,
    out_dir: Path,
    rule_plan: dict[str, object],
    resource_estimate: ResourceEstimate,
    exact_bank_entries: Sequence[int],
    fallback_filter_bank_entries: int,
    fallback_counting_row_entries: int,
    flow_filter_hashes: int,
    counting_hashes: int,
) -> TieredFlowRadarP4Artifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_count = int(rule_plan["feature_count"])
    ternary_entries = [
        entry.as_dict() for entry in lower_binary_rule_plan_to_ternary(rule_plan)
    ]
    if not ternary_entries:
        raise ValueError("tiered_flowradar requires selector rules")
    selector_size = len(ternary_entries)

    p4_path = out_dir / "pleds_tiered_flowradar_tna.p4"
    runtime_path = out_dir / "runtime_entries.json"
    resource_path = out_dir / "resource_plan.json"
    feature_path = out_dir / "feature_plan.json"
    bfrt_path = out_dir / "bfrt_plan.json"
    runtime_control_path = out_dir / "runtime_control.py"

    p4_path.write_text(
        _p4_text(
            feature_count=feature_count,
            selector_size=selector_size,
            exact_bank_entries=exact_bank_entries,
            fallback_filter_bank_entries=fallback_filter_bank_entries,
            fallback_counting_row_entries=fallback_counting_row_entries,
            flow_filter_hashes=flow_filter_hashes,
            counting_hashes=counting_hashes,
        ),
        encoding="utf-8",
    )

    feature_payload = (
        compact_feature_plan()
        if feature_count == 12
        else {
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
    )
    if feature_count == 12:
        feature_payload["placement"] = "ingress_parser"
        for feature in feature_payload["features"]:
            feature["p4_expr"] = f"ig_md.f{feature['index']}"
            feature["implementation"] = "parser_extract_and_transition"
    exact_register_suffixes = (
        "tags",
        "key_src",
        "key_dst",
        "key_ports",
        "key_protocol",
        "counts",
    )
    exact_registers = [
        f"SwitchIngress.exact{bank}_{suffix}"
        for bank in range(len(exact_bank_entries))
        for suffix in exact_register_suffixes
    ]
    fallback_registers = [
        f"SwitchIngress.flow_filter{bank}" for bank in range(flow_filter_hashes)
    ]
    for row in range(counting_hashes):
        fallback_registers.extend(
            f"SwitchIngress.row{row}_{suffix}"
            for suffix in (
                "key_src",
                "key_dst",
                "key_ports",
                "key_protocol",
                "flow_count",
                "packet_count",
            )
        )
    state_reset = []
    for bank, entries in enumerate(exact_bank_entries):
        state_reset.extend(
            {
                "name": f"pipe.SwitchIngress.exact{bank}_{suffix}",
                "entry_count": int(entries),
                "write_value": 0,
            }
            for suffix in exact_register_suffixes
        )
    state_reset.extend(
        {
            "name": f"pipe.SwitchIngress.flow_filter{bank}",
            "entry_count": fallback_filter_bank_entries,
            "write_value": 0,
        }
        for bank in range(flow_filter_hashes)
    )
    for row in range(counting_hashes):
        state_reset.extend(
            {
                "name": f"pipe.SwitchIngress.row{row}_{suffix}",
                "entry_count": fallback_counting_row_entries,
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

    fallback_hash_plan = register_hash_plan(
        table_size=fallback_filter_bank_entries,
        count=flow_filter_hashes + counting_hashes,
    )
    exact_hash_specs = register_hash_plan(table_size=1 << 32, count=8)["specs"]
    exact_plan = {
        "entry_physical_bits": 192,
        "bank_entries": [int(value) for value in exact_bank_entries],
        "registers": exact_registers,
        "tag_hash_ids": [6 + bank for bank in range(len(exact_bank_entries))],
        "index_hash_ids": [6 + bank for bank in range(len(exact_bank_entries))],
        "tag_hash_specs": [
            exact_hash_specs[6 + bank] for bank in range(len(exact_bank_entries))
        ],
        "index_hash_specs": [
            register_hash_plan(table_size=int(entries), count=8)["specs"][6 + bank]
            for bank, entries in enumerate(exact_bank_entries)
        ],
        "fingerprint_bits": 31,
        "full_key_storage": True,
    }
    fallback_plan = {
        "filter_bank_entries": fallback_filter_bank_entries,
        "counting_row_entries": fallback_counting_row_entries,
        "filter_hashes": flow_filter_hashes,
        "counting_hashes": counting_hashes,
        "registers": fallback_registers,
        "counting_cell_physical_bits": 192,
        "hash_plan": {
            "profile": fallback_hash_plan["profile"],
            "counting_table": {
                "hash_ids": list(range(counting_hashes)),
                "table_size": fallback_counting_row_entries,
                "specs": register_hash_plan(
                    table_size=fallback_counting_row_entries, count=counting_hashes
                )["specs"],
            },
            "flow_filter": {
                "hash_ids": list(
                    range(counting_hashes, counting_hashes + flow_filter_hashes)
                ),
                "table_size": fallback_filter_bank_entries,
                "specs": fallback_hash_plan["specs"][counting_hashes:],
            },
        },
    }
    selector = (
        selector_bfrt_plan("pipe.SwitchIngress.learned_selector", ternary_entries)
        if feature_count == 12
        else direct_prefix_selector_bfrt_plan(
            "pipe.SwitchIngress.learned_selector", ternary_entries, feature_count
        )
    )
    features = {
        "format": "pleds_bfrt_feature_plan_v1",
        "entries": [],
        "entry_count": 0,
        "placement": (
            "ingress_parser" if feature_count == 12 else "direct_packet_fields"
        ),
    }
    bfrt_payload = {
        "format": "pleds_bfrt_plan_v1",
        "program_name": "pleds_tiered_flowradar_tna",
        "feature_tables": features,
        "selector": selector,
        "register_initialization": {
            "format": "pleds_register_init_v1",
            "registers": [],
        },
        "state_reset": {
            "format": "pleds_register_reset_v1",
            "scope": "before_each_evaluation_slot",
            "registers": state_reset,
        },
        "tiered_flowradar": {"exact_tier": exact_plan, "fallback": fallback_plan},
        "end_window_query": {
            "exact_registers": [f"pipe.{name}" for name in exact_registers],
            "fallback_registers": [f"pipe.{name}" for name in fallback_registers],
            "decode": "merge_full_key_exact_records_with_flowradar_peeling_output",
        },
        "summary": {
            "feature_entry_count": features["entry_count"],
            "selector_entry_count": len(ternary_entries),
            "register_write_count": 0,
            "state_register_count": len(state_reset),
        },
        "live_safe": False,
    }
    runtime_payload = {
        "format": "pleds_runtime_entries_v1",
        "program": p4_path.name,
        "model_rules": rule_plan,
        "ternary_entries": ternary_entries,
        "state_reset": bfrt_payload["state_reset"],
        "tiered_flowradar": bfrt_payload["tiered_flowradar"],
        "end_window_query": bfrt_payload["end_window_query"],
    }
    resource_payload = {
        "format": "pleds_resource_plan_v1",
        "backend": "tiered_flowradar",
        "resource_estimate": resource_estimate.as_dict(),
        "layout": bfrt_payload["tiered_flowradar"],
        "semantic_properties": {
            "single_record_path_per_packet": True,
            "exact_tier_stores_full_five_tuple": True,
            "exact_tier_fingerprint_bits": 31,
            "exact_collision_falls_back_to_flowradar": True,
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
    runtime_control_path.chmod(0o755)
    return TieredFlowRadarP4Artifact(
        p4_path=p4_path,
        runtime_path=runtime_path,
        resource_path=resource_path,
        feature_path=feature_path,
        bfrt_path=bfrt_path,
        runtime_control_path=runtime_control_path,
    )
