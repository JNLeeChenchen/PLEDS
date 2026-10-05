from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from pleds import workflow
from pleds.compiler import available_backends, load_compilation_request
from pleds.cli import main
from pleds.flow_workload import load_workload, write_demo_trace
from pleds.models.plan_runtime import PersistedModelRuntime
from pleds.models.registry import available_model_frontends, train_and_lower_candidate
from pleds.spec import HardwareConstraints


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "configs/run_flowradar.yaml").read_text())


def run(tmp_path, config, **kwargs):
    source = tmp_path / "run.yaml"
    source.write_text(yaml.safe_dump(config))
    return workflow.run_workflow(source, tmp_path / "output", **kwargs)


def test_demo_generates_all_three_backends_and_portable_request(tmp_path, config):
    result = run(tmp_path, config)
    assert result["success"]
    assert result["mode"] == "offline_estimates"
    assert result["target_compiled"] is False
    rows = json.loads((tmp_path / "output/candidates.json").read_text())
    assert {row["backend"] for row in rows} == set(available_backends())
    assert all(row["resource_estimate"]["backend_bits"] == 16 * 8192 for row in rows)
    assert all(0 <= row["validation_recovered_flow_ratio"] <= 1 for row in rows)
    assert all(row["selection_status"] for row in rows)
    selected = Path(result["selected_package"])
    assert list(selected.glob("*.p4"))
    for name in [
        "resource_plan.json",
        "bfrt_plan.json",
        "composition_ir.json",
        "dependency_graph.json",
    ]:
        assert (selected / name).is_file()
    request = load_compilation_request(selected.parent / "request.yaml")
    assert request.output_dir == selected
    assert request.spec_path.parent == selected.parent
    runtime = subprocess.run(
        [
            sys.executable,
            str(selected / "runtime_control.py"),
            "--plan",
            str(selected / "bfrt_plan.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert runtime.returncode == 0, runtime.stderr
    assert "dry run only" in runtime.stdout


def test_nonempty_output_is_preserved(tmp_path, config):
    out = tmp_path / "output"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        run(tmp_path, config)
    assert marker.read_text() == "original"


def test_no_feasible_candidate_is_reported(tmp_path, config):
    config["constraints"]["max_sram_units"] = 1
    result = run(tmp_path, config)
    assert result["success"] is False
    assert result["selected"] is None
    assert not (tmp_path / "output/selected").exists()
    assert (tmp_path / "output/candidates.json").is_file()


def test_target_feedback_uses_actual_resources_and_rejects_failures(
    tmp_path, config, monkeypatch
):
    compiler = tmp_path / "bf-p4c"
    compiler.touch()
    original = workflow.compile_request
    calls = []

    def fake_compile(request, *, compiler=None, **kwargs):
        if compiler is None:
            return original(request, **kwargs)
        calls.append(request.backend_type)
        status = original(request, overwrite=True, allow_constraint_overrun=True)
        if request.backend_type == "tiered_flowradar":
            status.update(success=False, stage="bf_p4c")
            return status
        build = request.output_dir / "tofino_build"
        build.mkdir(exist_ok=True)
        resources = {"pipeline_stage_span": 8, "memory_units": {"sram": 14}}
        compiled = {
            "compile_success": True,
            "build_dir": str(build),
            "compiler_resources": resources,
        }
        workflow.write_json(request.output_dir / "tofino_compile_status.json", compiled)
        return {
            **status,
            "success": True,
            "stage": "complete",
            "compile_status": compiled,
        }

    monkeypatch.setattr(workflow, "compile_request", fake_compile)
    # All static physical estimates exceed this; target feedback must still run.
    config["constraints"]["max_stages"] = 8
    result = run(tmp_path, config, compiler=compiler)
    assert result["success"] and result["target_compiled"]
    assert result["mode"] == "target_feedback"
    assert calls[0] == "flowradar"
    assert "tiered_flowradar" in calls
    assert result["selected"]["backend"] != "tiered_flowradar"
    assert result["selected"]["compiler_resources"]["pipeline_stage_span"] == 8
    assert (Path(result["selected_package"]) / "tofino_build").is_dir()


@pytest.mark.parametrize(
    "train,validation", [([0], [0]), ([1], [0]), ([0, 0], [1]), ([], [1])]
)
def test_invalid_window_splits_are_rejected(tmp_path, train, validation):
    trace = tmp_path / "trace.parquet"
    write_demo_trace(trace)
    with pytest.raises(ValueError):
        load_workload(
            trace,
            training_windows=train,
            validation_windows=validation,
            slot_duration_ms=1000,
            slots_per_window=1,
        )


def test_validation_packets_do_not_change_training_counts(tmp_path):
    trace = tmp_path / "trace.parquet"
    write_demo_trace(trace)
    options = dict(
        training_windows=[0],
        validation_windows=[1],
        slot_duration_ms=1000,
        slots_per_window=1,
    )
    keys, counts, _, _ = load_workload(trace, **options)
    table = pq.read_table(trace)
    values = table["dst_port"].to_pylist()
    windows = table["window_id"].to_pylist()
    values = [65000 if window == 1 else value for value, window in zip(values, windows)]
    table = table.set_column(
        table.schema.get_field_index("dst_port"),
        "dst_port",
        pa.array(values, type=pa.uint16()),
    )
    pq.write_table(table, trace)
    after_keys, after_counts, _, _ = load_workload(trace, **options)
    np.testing.assert_array_equal(keys, after_keys)
    np.testing.assert_array_equal(counts, after_counts)


def test_bad_prepared_trace_is_rejected(tmp_path):
    trace = tmp_path / "trace.parquet"
    write_demo_trace(trace)
    table = pq.read_table(trace)
    table = table.set_column(
        0, "packet_ordinal", pa.array([0] * len(table), type=pa.uint64())
    )
    pq.write_table(table, trace)
    with pytest.raises(ValueError, match="ordinals"):
        load_workload(
            trace,
            training_windows=[0],
            validation_windows=[1],
            slot_duration_ms=1000,
            slots_per_window=1,
        )


def test_other_applications_are_not_registered(tmp_path, config):
    assert set(available_backends()) == {
        "flowradar",
        "tiered_flowradar",
        "partitioned_flowradar",
    }
    assert len(available_model_frontends()) == 8
    config["application"] = "unsupported_application"
    with pytest.raises(ValueError, match="flow_record_collection"):
        run(tmp_path, config)


def test_pcap_input_and_cli_paths_from_another_directory(tmp_path, config, monkeypatch):
    import dpkt

    folder = tmp_path / "config"
    folder.mkdir()
    capture = folder / "trace.pcap"
    with capture.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle)
        for window in range(2):
            for index in range(32):
                flow = index % 8
                udp = dpkt.udp.UDP(
                    sport=10000 + flow,
                    dport=443 if flow < 2 else 20000,
                    data=b"test",
                    ulen=12,
                )
                ip = dpkt.ip.IP(
                    src=(0xC0000201 + flow).to_bytes(4, "big"),
                    dst=(0xC6336401).to_bytes(4, "big"),
                    p=17,
                    data=udp,
                )
                ip.len = len(ip)
                ethernet = dpkt.ethernet.Ethernet(
                    src=b"\x02\x00\x00\x00\x00\x01",
                    dst=b"\x02\x00\x00\x00\x00\x02",
                    type=0x800,
                    data=ip,
                )
                writer.writepkt(bytes(ethernet), ts=window + index / 1000)
    config["input"] = {"format": "pcap", "path": "trace.pcap", "window_seconds": 1}
    config_path = folder / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(config_path), "--output-dir", "result"]) == 0
    report = json.loads((tmp_path / "result/workload.json").read_text())
    assert report["training_packet_count"] == 32
    assert report["validation_packet_count"] == 32


def test_generation_only_request_runs_from_exported_selection(tmp_path, config):
    result = run(tmp_path, config)
    request = Path(result["selected_package"]).parent / "request.yaml"
    output = tmp_path / "recompiled"
    assert main(["compile", str(request), "--output-dir", str(output)]) == 0
    assert list(output.glob("*.p4"))


def test_partitioning_cannot_silently_drop_all_requested_models(tmp_path, config):
    config["models"] = [{"type": "decision_tree", "feature_counts": [12]}]
    with pytest.raises(ValueError, match="104-bit"):
        run(tmp_path, config)


@pytest.mark.parametrize(
    "family,parameters",
    [
        ("decision_tree", {"max_depth": 2, "min_samples_leaf": 2}),
        ("random_forest_ensemble", {"n_estimators": 2, "max_depth": 2}),
        ("rule_list", {"max_rules": 8}),
        ("naive_bayes_lookup", {"scale": 256}),
        ("piecewise_range", {"bucket_count": 3}),
        ("isolation_forest", {"n_estimators": 2, "threshold_quantile": 0.5}),
        ("xgboost_ensemble", {"n_estimators": 2, "max_depth": 1}),
        (
            "tm_guided",
            {
                "number_of_clauses": 10,
                "epochs": 1,
                "threshold": 5,
                "max_rules": 8,
                "max_depth": 2,
                "node_sample_size": 32,
            },
        ),
    ],
)
def test_all_frontends_reach_tiered_p4_without_other_applications(
    tmp_path, family, parameters
):
    if family == "tm_guided":
        pytest.importorskip("tmu")
    if family == "xgboost_ensemble":
        pytest.importorskip("xgboost")
    rng = np.random.default_rng(3)
    features = rng.integers(0, 2, (64, 12), dtype=np.uint32)
    labels = (features[:, 0] | features[:, 1]).astype(np.uint32)
    trained = train_and_lower_candidate(
        family, features, labels, seed=3, parameters=parameters
    )
    mapped = workflow.normalize_plan(trained.model_plan)
    space = ((np.arange(4096)[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint32)
    np.testing.assert_array_equal(
        PersistedModelRuntime(trained.model_plan).predict(space),
        PersistedModelRuntime(mapped).predict(space),
    )
    model = {"plan": mapped, "family": family, "feature_count": 12}
    layout = next(
        row
        for row in workflow.candidate_layouts({})
        if row["backend"] == "tiered_flowradar"
    )
    row = {"candidate_id": family, "backend": layout["backend"], "layout": layout}
    exported = workflow.export_candidate(
        row, model, tmp_path, HardwareConstraints(max_metadata_bits=1024)
    )
    package = Path(exported["package"])
    assert (package / "pleds_tiered_flowradar_tna.p4").is_file()
    assert (package / "runtime_entries.json").is_file()
    assert exported["resource_estimate"]["backend_bits"] == layout["actual_memory_bits"]
