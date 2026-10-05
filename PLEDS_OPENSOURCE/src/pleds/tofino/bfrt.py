"""BFRT-like runtime artifact formatting.

These artifacts are intentionally JSON plans, not live BFRT commands. They make
the table/register programming intent explicit while keeping hardware execution
behind a separate safety gate.
"""

from __future__ import annotations


FIVE_TUPLE_FIELDS: tuple[tuple[str, int], ...] = (
    ("hdr.ipv4.src_addr", 32),
    ("hdr.ipv4.dst_addr", 32),
    ("ig_md.l4_sport", 16),
    ("ig_md.l4_dport", 16),
    ("hdr.ipv4.protocol", 8),
)


def direct_prefix_selector_fields(feature_count: int) -> tuple[tuple[str, int], ...]:
    """Return P4 key fields for a network-order prefix of the five tuple."""

    if feature_count <= 0 or feature_count > sum(
        width for _, width in FIVE_TUPLE_FIELDS
    ):
        raise ValueError(f"unsupported direct feature count: {feature_count}")
    remaining = feature_count
    fields: list[tuple[str, int]] = []
    for name, width in FIVE_TUPLE_FIELDS:
        if remaining <= 0:
            break
        selected_width = min(width, remaining)
        selected_name = (
            name
            if selected_width == width
            else f"{name}[{width - 1}:{width - selected_width}]"
        )
        fields.append((selected_name, selected_width))
        remaining -= selected_width
    return tuple(fields)


def selector_bfrt_plan(
    selector_table: str,
    ternary_entries: list[dict[str, object]],
    active_features: tuple[int, ...] | None = None,
) -> dict[str, object]:
    formatted_entries = []
    for entry in ternary_entries:
        value = str(entry["value"])
        mask = str(entry["mask"])
        if len(value) != len(mask):
            raise ValueError(f"value/mask width mismatch: {entry}")
        selected = (
            active_features if active_features is not None else tuple(range(len(value)))
        )
        key = {
            f"ig_md.f{idx}": {
                "match_type": "ternary",
                "value": int(value[idx]),
                "mask": int(mask[idx]),
            }
            for idx in selected
        }
        formatted_entries.append(
            {
                "table": selector_table,
                "priority": int(entry["priority"]),
                "action": f"SwitchIngress.{entry['action']}",
                "key": key,
                "data": {},
                "source_rule_id": int(entry["rule_id"]),
            }
        )

    return {
        "format": "pleds_bfrt_selector_plan_v1",
        "table": selector_table,
        "entries": formatted_entries,
        "notes": [
            "Each PLEDS feature bit is mapped to one ternary BFRT key field.",
            "Field names follow generated P4 metadata names and may need target context validation.",
        ],
    }


def five_tuple_selector_bfrt_plan(
    selector_table: str, ternary_entries: list[dict[str, object]]
) -> dict[str, object]:
    """Map a 104-bit network-order RuleIR key to five P4 ternary fields."""

    return direct_prefix_selector_bfrt_plan(selector_table, ternary_entries, 104)


def direct_prefix_selector_bfrt_plan(
    selector_table: str,
    ternary_entries: list[dict[str, object]],
    feature_count: int,
) -> dict[str, object]:
    """Map a network-order RuleIR prefix to direct packet-field keys."""

    fields = direct_prefix_selector_fields(feature_count)
    formatted_entries = []
    for entry in ternary_entries:
        value = str(entry["value"])
        mask = str(entry["mask"])
        if len(value) != feature_count or len(mask) != feature_count:
            raise ValueError(
                f"direct RuleIR value/mask must be {feature_count} bits: {entry}"
            )
        key = {}
        offset = 0
        for field, width in fields:
            key[field] = {
                "match_type": "ternary",
                "value": int(value[offset : offset + width], 2),
                "mask": int(mask[offset : offset + width], 2),
            }
            offset += width
        formatted_entries.append(
            {
                "table": selector_table,
                "priority": int(entry["priority"]),
                "action": f"SwitchIngress.{entry['action']}",
                "key": key,
                "data": {},
                "source_rule_id": int(entry["rule_id"]),
            }
        )
    return {
        "format": "pleds_bfrt_direct_prefix_selector_plan_v1",
        "table": selector_table,
        "entries": formatted_entries,
        "notes": [
            f"The {feature_count} RuleIR bits are grouped into a network-order prefix of the normalized IPv4 five tuple.",
            "This direct-match representation requires no derived feature MAT.",
        ],
    }


def no_selector_bfrt_plan(*, model_type: str) -> dict[str, object]:
    return {
        "format": "pleds_bfrt_no_selector_plan_v1",
        "model_type": model_type,
        "entries": [],
        "tables": [],
        "notes": [
            "This model is compiled into direct P4 logic and has no model table entries.",
            "Feature tables and Bloom registers are still programmed through the common plan.",
        ],
    }


def _ternary_range(start: int, end: int, width: int) -> list[tuple[int, int]]:
    if start < 0 or end < start or end >= (1 << width):
        raise ValueError(f"bad range {start}..{end} for width {width}")
    out: list[tuple[int, int]] = []
    cur = start
    while cur <= end:
        size = cur & -cur if cur else 1 << width
        remaining = end - cur + 1
        while size > remaining:
            size >>= 1
        mask = ((1 << width) - 1) ^ (size - 1)
        out.append((cur & mask, mask))
        cur += size
    return out


def _ternary_entry(
    table: str,
    field: str,
    value: int,
    mask: int,
    action: str,
    data: dict[str, int],
    priority: int,
) -> dict[str, object]:
    return {
        "table": table,
        "priority": priority,
        "action": action,
        "key": {field: {"match_type": "ternary", "value": value, "mask": mask}},
        "data": data,
    }


def feature_bfrt_plan(
    active_features: tuple[int, ...] | None = None
) -> dict[str, object]:
    active = set(range(12) if active_features is None else active_features)
    entries: list[dict[str, object]] = []
    priority = 10000
    for port in [22, 80, 443, 8080]:
        if 4 in active:
            entries.append(
                _ternary_entry(
                    "pipe.SwitchIngress.src_service_port_table",
                    "ig_md.l4_sport",
                    port,
                    0xFFFF,
                    "SwitchIngress.set_src_service_port",
                    {"value": 1},
                    priority,
                )
            )
            priority -= 1
        if 5 in active:
            entries.append(
                _ternary_entry(
                    "pipe.SwitchIngress.dst_service_port_table",
                    "ig_md.l4_dport",
                    port,
                    0xFFFF,
                    "SwitchIngress.set_dst_service_port",
                    {"value": 1},
                    priority,
                )
            )
            priority -= 1
    if 6 in active:
        for value, mask in _ternary_range(1024, 65535, 16):
            entries.append(
                _ternary_entry(
                    "pipe.SwitchIngress.dst_ephemeral_port_table",
                    "ig_md.l4_dport",
                    value,
                    mask,
                    "SwitchIngress.set_dst_ephemeral_port",
                    {"value": 1},
                    priority,
                )
            )
            priority -= 1
    if active & {7, 8}:
        entries.append(
            _ternary_entry(
                "pipe.SwitchIngress.protocol_feature_table",
                "hdr.ipv4.protocol",
                6,
                0xFF,
                "SwitchIngress.set_protocol_features",
                {"tcp": 1, "udp": 0},
                priority,
            )
        )
        priority -= 1
        entries.append(
            _ternary_entry(
                "pipe.SwitchIngress.protocol_feature_table",
                "hdr.ipv4.protocol",
                17,
                0xFF,
                "SwitchIngress.set_protocol_features",
                {"tcp": 0, "udp": 1},
                priority,
            )
        )
        priority -= 1
    return {
        "format": "pleds_bfrt_feature_plan_v1",
        "entries": entries,
        "entry_count": len(entries),
    }


def validate_bfrt_plan(
    plan: dict[str, object], bfrt: dict[str, object]
) -> dict[str, object]:
    tables = {table["name"]: table for table in bfrt.get("tables", [])}
    errors: list[str] = []
    warnings: list[str] = []

    def validate_match_entries(
        group: str, table_name: str, entries: list[dict[str, object]]
    ) -> None:
        if table_name not in tables:
            errors.append(f"missing {group} table: {table_name}")
            return
        table = tables[table_name]
        key_names = {key["name"] for key in table.get("key", [])}
        action_names = {action["name"] for action in table.get("action_specs", [])}
        action_data = {
            action["name"]: {item["name"] for item in action.get("data", [])}
            for action in table.get("action_specs", [])
        }
        for entry in entries:
            if entry["action"] not in action_names:
                errors.append(f"missing action {entry['action']} in {table_name}")
            for field in entry["key"]:
                if field not in key_names:
                    errors.append(f"missing key field {field} in {table_name}")
            for field in entry.get("data", {}):
                if field not in action_data.get(entry["action"], set()):
                    errors.append(
                        f"missing action data field {field} for {entry['action']} in {table_name}"
                    )
            if "priority" in entry and "$MATCH_PRIORITY" not in key_names:
                warnings.append(f"{table_name} has no $MATCH_PRIORITY key")

    selector = plan["selector"]
    if selector.get("format") == "pleds_bfrt_no_selector_plan_v1":
        pass
    elif "table" in selector:
        validate_match_entries("selector", selector["table"], selector["entries"])
    else:
        by_selector_table: dict[str, list[dict[str, object]]] = {}
        for entry in selector["entries"]:
            by_selector_table.setdefault(entry["table"], []).append(entry)
        for table_name, entries in by_selector_table.items():
            validate_match_entries("selector", table_name, entries)
    by_feature_table: dict[str, list[dict[str, object]]] = {}
    for entry in plan.get("feature_tables", {}).get("entries", []):
        by_feature_table.setdefault(entry["table"], []).append(entry)
    for table_name, entries in by_feature_table.items():
        validate_match_entries("feature", table_name, entries)
    exact_table = plan.get("exact_table", {})
    if isinstance(exact_table, dict) and exact_table.get("entries"):
        validate_match_entries(
            "exact", str(exact_table["table"]), list(exact_table["entries"])
        )
    by_auxiliary_table: dict[str, list[dict[str, object]]] = {}
    for entry in plan.get("auxiliary_tables", {}).get("entries", []):
        by_auxiliary_table.setdefault(entry["table"], []).append(entry)
    for table_name, entries in by_auxiliary_table.items():
        validate_match_entries("auxiliary", table_name, entries)

    for register in plan["register_initialization"]["registers"]:
        name = register["name"]
        if name not in tables:
            errors.append(f"missing register table: {name}")
            continue
        table = tables[name]
        key_names = {key["name"] for key in table.get("key", [])}
        data_names = {
            item.get("singleton", {}).get("name")
            for item in table.get("data", [])
            if item.get("singleton", {}).get("name")
        }
        if "$REGISTER_INDEX" not in key_names:
            errors.append(f"missing $REGISTER_INDEX in {name}")
        canonical_name = name[len("pipe.") :] if name.startswith("pipe.") else name
        expected_data = f"{canonical_name}.f1"
        if expected_data not in data_names:
            errors.append(f"missing register data field {expected_data} in {name}")

    for register in plan.get("state_reset", {}).get("registers", []):
        name = register["name"]
        if name not in tables:
            errors.append(f"missing resettable register table: {name}")
            continue
        table = tables[name]
        key_names = {key["name"] for key in table.get("key", [])}
        data_names = {
            item.get("singleton", {}).get("name")
            for item in table.get("data", [])
            if item.get("singleton", {}).get("name")
        }
        if "$REGISTER_INDEX" not in key_names:
            errors.append(f"missing $REGISTER_INDEX in resettable register {name}")
        canonical_name = name[len("pipe.") :] if name.startswith("pipe.") else name
        expected_data = f"{canonical_name}.f1"
        if expected_data not in data_names:
            errors.append(f"missing register data field {expected_data} in {name}")

    return {
        "format": "pleds_bfrt_validation_v1",
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def runtime_control_py() -> str:
    return '''#!/usr/bin/env python3
"""Install generated PLEDS BFRT plan.

This script is generated for the switch host. It is inert unless called with
--install and a valid BFRT gRPC environment.
"""

import argparse
import json
import time
from pathlib import Path


def _data_tuple(name, value):
    import bfrt_grpc.client as gc

    return gc.DataTuple(name, int(value))


def _table_get_compat(bfrt_info, table_name):
    try:
        return bfrt_info.table_get(table_name)
    except KeyError:
        if table_name.startswith("pipe."):
            return bfrt_info.table_get(table_name[len("pipe."):])
        raise


def _register_data_field(table_name):
    if table_name.startswith("pipe."):
        table_name = table_name[len("pipe."):]
    return "{}.f1".format(table_name)


def _install_match_entries(bfrt_info, target, entries):
    import bfrt_grpc.client as gc

    by_table = {}
    for entry in entries:
        by_table.setdefault(entry["table"], []).append(entry)
    for table_name, table_entries in by_table.items():
        table = _table_get_compat(bfrt_info, table_name)
        keys = []
        data = []
        for entry in table_entries:
            tuples = []
            for field, spec in entry["key"].items():
                if spec.get("match_type") == "exact":
                    tuples.append(gc.KeyTuple(field, int(spec["value"])))
                elif spec.get("match_type") == "range":
                    tuples.append(gc.KeyTuple(field, low=int(spec["low"]), high=int(spec["high"])))
                else:
                    tuples.append(gc.KeyTuple(field, int(spec["value"]), int(spec["mask"])))
            if "priority" in entry:
                tuples.append(gc.KeyTuple("$MATCH_PRIORITY", int(entry["priority"])))
            keys.append(table.make_key(tuples))
            data.append(table.make_data([_data_tuple(k, v) for k, v in entry.get("data", {}).items()], entry["action"]))
        if keys:
            table.entry_add(target, keys, data)
        print("installed {} entries into {}".format(len(keys), table_name))


def _install_registers(bfrt_info, target, register_plan):
    import bfrt_grpc.client as gc

    for reg in register_plan["registers"]:
        table = _table_get_compat(bfrt_info, reg["name"])
        keys = []
        data = []
        for index in reg["indexes"]:
            keys.append(table.make_key([gc.KeyTuple("$REGISTER_INDEX", int(index))]))
            field = _register_data_field(reg["name"])
            data.append(table.make_data([gc.DataTuple(field, int(reg["write_value"]))]))
        if keys:
            try:
                table.entry_mod(target, keys, data)
            except Exception:
                table.entry_add(target, keys, data)
        print("set {} indexes in {}".format(len(keys), reg["name"]))


def _reset_registers(bfrt_info, target, reset_plan):
    import bfrt_grpc.client as gc

    for reg in reset_plan.get("registers", []):
        table = _table_get_compat(bfrt_info, reg["name"])
        field = _register_data_field(reg["name"])
        keys = [
            table.make_key([gc.KeyTuple("$REGISTER_INDEX", index)])
            for index in range(int(reg["entry_count"]))
        ]
        data = [
            table.make_data([gc.DataTuple(field, int(reg.get("write_value", 0)))])
            for _ in keys
        ]
        if keys:
            table.entry_mod(target, keys, data)
        print("reset {} indexes in {}".format(len(keys), reg["name"]))


def _install_mirror_session(bfrt_info, target, mirror_plan, egress_port):
    if not mirror_plan:
        return
    if egress_port is None:
        raise ValueError("--mirror-egress-port is required by this BFRT plan")

    import bfrt_grpc.client as gc

    table = _table_get_compat(bfrt_info, mirror_plan["table"])
    key = table.make_key([gc.KeyTuple("$sid", int(mirror_plan["session_id"]))])
    data = table.make_data(
        [
            gc.DataTuple("$session_enable", bool_val=True),
            gc.DataTuple("$direction", str_val=mirror_plan["direction"]),
            gc.DataTuple("$ucast_egress_port", int(egress_port)),
            gc.DataTuple("$ucast_egress_port_valid", bool_val=True),
            gc.DataTuple("$egress_port_queue", 0),
            gc.DataTuple("$ingress_cos", 0),
            gc.DataTuple("$packet_color", str_val="GREEN"),
            gc.DataTuple(
                "$max_pkt_len",
                int(mirror_plan["maximum_mirrored_packet_length"]),
            ),
        ],
        "$normal",
    )
    try:
        table.entry_mod(target, [key], [data])
    except Exception:
        table.entry_add(target, [key], [data])
    print("installed mirror session {}".format(mirror_plan["session_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--grpc-addr", default="localhost:50052")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pipe-id", type=lambda x: int(x, 0), default=0xFFFF)
    parser.add_argument("--mirror-egress-port", type=int)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))
    if not args.install and not args.reset_state:
        print("dry run only; pass --install or --reset-state to program BFRT")
        return

    import bfrt_grpc.client as gc

    started = time.time()
    interface = None
    try:
        interface = gc.ClientInterface(args.grpc_addr, client_id=args.client_id, device_id=args.device_id, is_master=True)
    except TypeError:
        interface = gc.ClientInterface(args.grpc_addr, client_id=args.client_id, device_id=args.device_id)
    try:
        interface.bind_pipeline_config(plan["program_name"])
        bfrt_info = interface.bfrt_info_get(plan["program_name"])
        target = gc.Target(device_id=args.device_id, pipe_id=args.pipe_id)
        connected = time.time()

        if args.install:
            _install_match_entries(bfrt_info, target, plan["feature_tables"]["entries"])
            _install_match_entries(bfrt_info, target, plan["selector"]["entries"])
            if plan.get("exact_table", {}).get("entries"):
                _install_match_entries(bfrt_info, target, plan["exact_table"]["entries"])
            if plan.get("auxiliary_tables", {}).get("entries"):
                _install_match_entries(bfrt_info, target, plan["auxiliary_tables"]["entries"])
            _install_mirror_session(
                bfrt_info,
                target,
                plan.get("mirror_session"),
                args.mirror_egress_port,
            )
        tables_installed = time.time()
        if args.install or args.reset_state:
            _reset_registers(bfrt_info, target, plan.get("state_reset", {}))
        if args.install:
            _install_registers(bfrt_info, target, plan["register_initialization"])
        finished = time.time()
        timing = {
            "bfrt_connection_seconds": connected - started,
            "match_table_install_seconds": tables_installed - connected,
            "state_initialization_seconds": finished - tables_installed,
            "total_install_seconds": finished - started,
        }
        print("PLEDS_INSTALL_TIMING_JSON_BEGIN")
        print(json.dumps(timing, indent=2, sort_keys=True))
        print("PLEDS_INSTALL_TIMING_JSON_END")
    finally:
        if interface is not None and hasattr(interface, "tear_down_stream"):
            interface.tear_down_stream()


if __name__ == "__main__":
    main()
'''
