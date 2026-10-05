from __future__ import annotations

import json
from pathlib import Path

import pytest

from pleds.ir import ResourceEstimate
from pleds.p4gen.flowradar import write_flowradar_artifacts


def plan() -> dict[str, object]:
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


def estimate() -> ResourceEstimate:
    return ResourceEstimate(
        model_rules=1,
        model_key_bits=104,
        metadata_bits=356,
        backend_bits=221184,
        hash_count=4,
        estimated_tcam_entries=1,
        estimated_sram_units=29,
        estimated_tcam_units=3,
        estimated_stages=9,
        estimated_phv_bits=356,
    )


def test_flowradar_artifact_has_independent_partition_state(tmp_path: Path) -> None:
    artifact = write_flowradar_artifacts(
        out_dir=tmp_path,
        rule_plan=plan(),
        resource_estimate=estimate(),
        filter_bank_entries=[2048, 2048],
        counting_row_entries=[128, 128],
        filter_hashes=2,
        counting_hashes=2,
    )
    source = artifact.p4_path.read_text(encoding="utf-8")
    assert "Register<bit<1>, bit<12>>(4096, 1w0) flow_filter0" in source
    assert "Register<bit<32>, bit<8>>(256, 32w0) row0_key_src" in source
    assert source.count("_flow_count_update.execute") == 2
    assert source.count("_packet_count_update.execute") == 2
    bfrt = json.loads(artifact.bfrt_path.read_text(encoding="utf-8"))
    assert bfrt["summary"]["selector_entry_count"] == 1
    assert len(bfrt["flowradar"]["registers"]) == 14
    assert len(bfrt["state_reset"]["registers"]) == 14
    assert {entry["entry_count"] for entry in bfrt["state_reset"]["registers"]} == {
        256,
        4096,
    }
    assert bfrt["flowradar"]["hash_plan"]["flow_filter"]["hash_ids"] == [2, 3]
    assert bfrt["flowradar"]["hash_plan"]["counting_table"]["hash_ids"] == [0, 1]
    runtime = json.loads(artifact.runtime_path.read_text(encoding="utf-8"))
    assert runtime["flowradar"] == bfrt["flowradar"]
    assert runtime["state_reset"] == bfrt["state_reset"]


def test_flowradar_artifact_rejects_non_power_of_two_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="power of two"):
        write_flowradar_artifacts(
            out_dir=tmp_path,
            rule_plan=plan(),
            resource_estimate=estimate(),
            filter_bank_entries=[2048, 3000],
            counting_row_entries=[128, 128],
            filter_hashes=2,
            counting_hashes=2,
        )
