from __future__ import annotations

import json
from pathlib import Path

from pleds.ir import ResourceEstimate
from pleds.p4gen.tiered_flowradar import write_tiered_flowradar_artifacts


def _plan() -> dict[str, object]:
    return {
        "format": "pleds_rule_ir_v1",
        "model_type": "tm_guided",
        "output_type": "binary",
        "feature_count": 104,
        "rule_count": 1,
        "rules": [
            {
                "rule_id": 0,
                "value": "0" * 104,
                "mask": "1" + "0" * 103,
                "action": 1,
            }
        ],
    }


def _estimate() -> ResourceEstimate:
    return ResourceEstimate(
        model_rules=1,
        model_key_bits=104,
        metadata_bits=739,
        backend_bits=524288,
        hash_count=8,
        estimated_tcam_entries=1,
        estimated_sram_units=26,
        estimated_tcam_units=3,
        estimated_stages=12,
        estimated_phv_bits=739,
    )


def test_tiered_flowradar_emits_fingerprint_tiers_and_fallback(tmp_path: Path) -> None:
    artifact = write_tiered_flowradar_artifacts(
        out_dir=tmp_path,
        rule_plan=_plan(),
        resource_estimate=_estimate(),
        exact_bank_entries=[256, 256],
        fallback_filter_bank_entries=16384,
        fallback_counting_row_entries=1024,
        flow_filter_hashes=2,
        counting_hashes=2,
    )
    source = artifact.p4_path.read_text(encoding="utf-8")
    assert "table learned_selector" in source
    assert "exact0_tag_claim.execute" in source
    assert "exact1_tag_claim.execute" in source
    assert "exact0_key_src_store.execute" in source
    assert "exact1_key_protocol_store.execute" in source
    assert "if (ig_md.exact_accept0 == 1w1 || ig_md.exact_accept1 == 1w1)" in source
    assert "flow_filter0_set.execute" in source
    assert "row1_packet_count_update.execute" in source

    bfrt = json.loads(artifact.bfrt_path.read_text(encoding="utf-8"))
    exact = bfrt["tiered_flowradar"]["exact_tier"]
    assert exact["fingerprint_bits"] == 31
    assert exact["full_key_storage"] is True
    assert exact["tag_hash_ids"] == [6, 7]
    assert exact["index_hash_ids"] == [6, 7]
    assert bfrt["summary"]["state_register_count"] == 26
    assert len(bfrt["state_reset"]["registers"]) == 26


def test_tiered_flowradar_supports_one_exact_bank(tmp_path: Path) -> None:
    artifact = write_tiered_flowradar_artifacts(
        out_dir=tmp_path,
        rule_plan=_plan(),
        resource_estimate=_estimate(),
        exact_bank_entries=[512],
        fallback_filter_bank_entries=8192,
        fallback_counting_row_entries=512,
        flow_filter_hashes=2,
        counting_hashes=2,
    )
    source = artifact.p4_path.read_text(encoding="utf-8")
    assert "exact0_tag_claim.execute" in source
    assert "exact1_tag_claim.execute" not in source
    assert "compute_flow_hash6_table.apply" in source
    assert "compute_flow_hash7_table.apply" not in source


def test_tiered_flowradar_computes_compact_features_from_packet(tmp_path: Path) -> None:
    plan = {
        **_plan(),
        "feature_count": 12,
        "rules": [
            {"rule_id": 0, "value": "0" * 12, "mask": "1" + "0" * 11, "action": 1}
        ],
    }
    artifact = write_tiered_flowradar_artifacts(
        out_dir=tmp_path,
        rule_plan=plan,
        resource_estimate=_estimate(),
        exact_bank_entries=[512],
        fallback_filter_bank_entries=16384,
        fallback_counting_row_entries=1024,
        flow_filter_hashes=2,
        counting_hashes=2,
    )
    source = artifact.p4_path.read_text()
    assert "ig_md.f11: ternary" in source
    assert "state tcp_src_service" in source
    assert "state udp_dst_ephemeral" in source
    assert "ig_md.f7 = 1w1;" in source
    assert "ig_md.f8 = 1w1;" in source
    assert "src_service_port_table.apply();" not in source
    runtime = json.loads(artifact.bfrt_path.read_text())
    assert runtime["feature_tables"]["entry_count"] == 0
    assert runtime["feature_tables"]["placement"] == "ingress_parser"
    assert len(runtime["selector"]["entries"][0]["key"]) == 12
    features = json.loads(artifact.feature_path.read_text())
    assert features["placement"] == "ingress_parser"
    assert features["features"][4]["p4_expr"] == "ig_md.f4"
