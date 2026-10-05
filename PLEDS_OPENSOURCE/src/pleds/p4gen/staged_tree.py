"""Match-action table plans for level-by-level decision-tree execution."""

from __future__ import annotations

import math
from typing import Any, Mapping


def staged_tree_table_plan(model_plan: Mapping[str, Any]) -> dict[str, object]:
    if model_plan.get("format") != "pleds_staged_tree_ir_v1":
        raise ValueError("staged-tree lowering requires pleds_staged_tree_ir_v1")
    node_id_bits = int(model_plan["node_id_bits"])
    tables = []
    total_entries = 0
    maximum_key_bits = 0
    for raw_level in model_plan["levels"]:
        level = int(raw_level["level"])
        features = tuple(int(feature) for feature in raw_level.get("features", []))
        feature_offsets = {feature: index for index, feature in enumerate(features)}
        entries = []
        for raw_node in raw_level["nodes"]:
            node_id = int(raw_node["node_id"])
            action = raw_node.get("action")
            if action is not None:
                entries.append(
                    {
                        "node_id": node_id,
                        "feature_value": "0" * len(features),
                        "feature_mask": "0" * len(features),
                        "action": "set_model_output",
                        "parameters": {"value": int(action)},
                    }
                )
                continue
            feature = int(raw_node["feature"])
            threshold = float(raw_node["threshold"])
            if threshold != 0.5:
                raise ValueError(
                    "the staged binary-feature representation requires threshold 0.5"
                )
            for bit, successor in ((0, raw_node["left"]), (1, raw_node["right"])):
                value = ["0"] * len(features)
                mask = ["0"] * len(features)
                offset = feature_offsets[feature]
                value[offset] = str(bit)
                mask[offset] = "1"
                entries.append(
                    {
                        "node_id": node_id,
                        "feature_value": "".join(value),
                        "feature_mask": "".join(mask),
                        "action": "set_next_node",
                        "parameters": {"node_id": int(successor)},
                    }
                )
        key_bits = node_id_bits + len(features)
        maximum_key_bits = max(maximum_key_bits, key_bits)
        total_entries += len(entries)
        tables.append(
            {
                "name": f"tree_level_{level}",
                "level": level,
                "match_kind": "ternary",
                "key": {
                    "node_id_bits": node_id_bits,
                    "feature_indices": list(features),
                    "key_bits": key_bits,
                },
                "entries": entries,
                "entry_count": len(entries),
            }
        )
    return {
        "format": "pleds_staged_tree_mat_plan_v1",
        "output_width": 1,
        "node_id_bits": node_id_bits,
        "table_count": len(tables),
        "entry_count": total_entries,
        "maximum_key_bits": maximum_key_bits,
        "estimated_tcam_units": sum(
            math.ceil(table["key"]["key_bits"] / 44)
            * math.ceil(max(1, table["entry_count"]) / 512)
            for table in tables
        ),
        "tables": tables,
    }


__all__ = ["staged_tree_table_plan"]
