"""Standalone FlowRadar training, layout selection, and P4 generation."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
import yaml

from .applications.flowradar import (
    FlowRadarCore,
    PackedPartitionedFlowRadar,
    TieredFlowRadar,
)
from .applications.layouts import (
    baseline_layout,
    hash_layout,
    tiered_layouts,
    score_backend,
)
from .compiler import compile_request, load_compilation_request
from .flow_workload import load_workload, write_demo_trace
from .indexing import target_index
from .models.plan_runtime import PersistedModelRuntime
from .models.registry import (
    available_model_frontends,
    supports_application_feature_count,
    train_and_lower_candidate,
)
from .optimizer import resource_violations
from .optimizer.target_search import ordered_target_search
from .packet_features import deployment_feature_matrix
from .p4gen.binary_selector import compact_binary_selector
from .p4gen.ternary import optimized_rule_plan
from .spec import HardwareConstraints
from .traces import preprocess_pcap
from .training import sample_hotness_training_rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def identity(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def normalize_plan(plan: dict) -> dict:
    if int(plan["feature_count"]) == 12:
        return compact_binary_selector(plan)
    if int(plan["feature_count"]) != 104 or plan["format"] not in {
        "pleds_rule_ir_v1",
        "pleds_lowered_decision_tree_v1",
    }:
        raise ValueError("104-bit FlowRadar selectors require a rule representation")
    plan = optimized_rule_plan(plan)
    if not plan["rules"]:
        plan = {
            **plan,
            "rule_count": 1,
            "rules": [
                {
                    "rule_id": 0,
                    "value": "0" * 104,
                    "mask": "0" * 104,
                    "action": 0,
                }
            ],
        }
    return plan


def candidate_layouts(search: dict) -> list[dict]:
    budgets = search.get("memory_kib", [16])
    mechanisms = search.get("mechanisms", ["partitioning", "tiering"])
    banks = search.get("exact_bank_counts", [1])
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(type(x) is not int or x <= 0 for x in budgets)
    ):
        raise ValueError("memory_kib must be a non-empty list of positive integers")
    if not isinstance(mechanisms, list) or not set(mechanisms) <= {
        "partitioning",
        "tiering",
    }:
        raise ValueError("FlowRadar mechanisms must be partitioning or tiering")
    if not isinstance(banks, list) or not banks or any(x not in {1, 2} for x in banks):
        raise ValueError("exact_bank_counts must contain one or two")
    rows = []
    params = {"filter_hashes": 2, "counting_hashes": 2, "cell_bits": 192}
    for budget in sorted(set(budgets)):
        layouts = []
        if search.get("include_conventional", True):
            layouts.append(
                {"backend": "flowradar", **baseline_layout(budget, **params)}
            )
        if "partitioning" in mechanisms:
            layouts.append(
                {"backend": "partitioned_flowradar", **hash_layout(budget, **params)}
            )
        if "tiering" in mechanisms:
            layouts.extend(
                {"backend": "tiered_flowradar", **layout}
                for layout in tiered_layouts(
                    budget, exact_bank_counts=banks, exact_entry_bits=192, **params
                )
            )
        rows.extend({"budget_kib": budget, **layout} for layout in layouts)
    if not rows:
        raise ValueError("search must include at least one layout")
    return rows


def train_models(
    config: dict, keys: np.ndarray, counts: np.ndarray, out: Path
) -> list[dict]:
    training = config.get("training", {})
    seed = int(training.get("seed", 7))
    rows = []
    seen = set()
    for model in config.get("models", []):
        family = model["type"]
        if family not in available_model_frontends():
            raise ValueError(f"unsupported model family: {family}")
        variants = model.get("parameters", [{}])
        if isinstance(variants, dict):
            variants = [variants]
        if (
            not isinstance(variants, list)
            or not variants
            or not all(isinstance(x, dict) for x in variants)
        ):
            raise ValueError(
                "model parameters must be a mapping or non-empty list of mappings"
            )
        for width in model.get("feature_counts", [104]):
            if not supports_application_feature_count(
                family, "flow_record_collection", width
            ):
                raise ValueError(
                    f"unsupported FlowRadar feature count {width} for {family}"
                )
            x, y, sample_report = sample_hotness_training_rows(
                keys,
                counts,
                feature_count=width,
                seed=seed,
                hot_fraction=float(training.get("hot_fraction", 0.1)),
                max_per_class=int(training.get("max_per_class", 4096)),
            )
            for parameters in variants:
                trained = train_and_lower_candidate(
                    family, x, y, seed=seed, parameters=parameters
                )
                plan = normalize_plan(trained.model_plan)
                model_id = (
                    family
                    + "-"
                    + identity(
                        {
                            "width": width,
                            "plan": plan,
                            "parameters": parameters,
                            "seed": seed,
                        }
                    )
                )
                if model_id in seen:
                    continue
                seen.add(model_id)
                folder = out / "models" / model_id
                write_json(folder / "trained_plan.json", trained.model_plan)
                write_json(folder / "model_plan.json", plan)
                write_json(
                    folder / "training_report.json",
                    {
                        **sample_report,
                        **trained.report,
                        "family": family,
                        "parameters": parameters,
                        "seed": seed,
                    },
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "family": family,
                        "feature_count": width,
                        "plan": plan,
                        "original_plan": trained.model_plan,
                        "model_plan": str(folder / "model_plan.json"),
                    }
                )
    return rows


@lru_cache(maxsize=262144)
def _key_hash(key: tuple, hash_id: int) -> int:
    return target_index(key, hash_id, 1 << 32)


def _index(key: tuple, hash_id: int, width: int) -> int:
    return _key_hash(key, hash_id) % width


def evaluate_layout(layout: dict, slots: list[dict], labels: dict | None) -> dict:
    metrics = []
    for slot in slots:
        if layout["backend"] == "flowradar":
            backend = FlowRadarCore(
                flow_filter_bits=2 * layout["filter_bank_entries"],
                counting_cells=2 * layout["counting_row_entries"],
                flow_filter_hashes=2,
                counting_hashes=2,
                index_fn=_index,
            )
        elif layout["backend"] == "partitioned_flowradar":
            backend = PackedPartitionedFlowRadar(
                partition_count=2,
                flow_filter_bank_bits=layout["filter_bank_entries"],
                counting_row_cells=layout["counting_row_entries"],
                selector=lambda key: int(labels[key]),
                index_fn=_index,
            )
        else:
            backend = TieredFlowRadar(
                exact_bank_entries=[layout["exact_bank_entries"]]
                * layout["exact_bank_count"],
                selector=lambda key: bool(labels[key]),
                fallback_flow_filter_bits=2 * layout["filter_bank_entries"],
                fallback_counting_cells=2 * layout["counting_row_entries"],
                flow_filter_hashes=2,
                counting_hashes=2,
                exact_hash_start=6,
                exact_fingerprint_bits=31,
                index_fn=_index,
            )
        metrics.append(
            {
                "window": slot["window"],
                "slot": slot["slot"],
                **score_backend(backend, slot["packets"]),
            }
        )
    recovery = statistics.mean(row["recovered_flow_ratio"] for row in metrics)
    return {
        "validation_recovered_flow_ratio": recovery,
        "validation_recovery_loss": 1 - recovery,
        "validation_slots": metrics,
    }


def backend_parameters(layout: dict) -> dict:
    if layout["backend"] == "tiered_flowradar":
        return {
            "exact_bank_entries": [layout["exact_bank_entries"]]
            * layout["exact_bank_count"],
            "fallback_filter_bank_entries": layout["filter_bank_entries"],
            "fallback_counting_row_entries": layout["counting_row_entries"],
            "flow_filter_hashes": 2,
            "counting_hashes": 2,
        }
    count = 2 if layout["backend"] == "partitioned_flowradar" else 1
    return {
        "filter_bank_entries": [layout["filter_bank_entries"]] * count,
        "counting_row_entries": [layout["counting_row_entries"]] * count,
        "filter_hashes": 2,
        "counting_hashes": 2,
    }


def export_candidate(
    row: dict, model: dict | None, out: Path, constraints: HardwareConstraints
) -> dict:
    folder = out / "candidates" / row["candidate_id"]
    folder.mkdir(parents=True)
    parameters = backend_parameters(row["layout"])
    spec_raw = {
        "name": row["candidate_id"],
        "task": "flow_record_collection",
        "key": "five_tuple",
        "features": [
            (
                "compact_packet_features"
                if model and model["feature_count"] == 12
                else "five_tuple_bits"
            )
        ],
        "models": [model["family"]] if model else [],
        "backend": {"type": row["backend"], "parameters": parameters},
        "constraints": asdict(constraints),
        "objective": {"maximize": "recovered_flow_ratio"},
    }
    spec_path = folder / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_raw, sort_keys=False), encoding="utf-8")
    model_path = None
    if model:
        model_path = folder / "model_plan.json"
        write_json(model_path, model["plan"])
    request_path = folder / "request.yaml"
    request_raw = {"spec": "spec.yaml", "output": "package"}
    if model:
        request_raw["model_plan"] = "model_plan.json"
    request_path.write_text(
        yaml.safe_dump(request_raw, sort_keys=False), encoding="utf-8"
    )
    request = load_compilation_request(request_path)
    status = compile_request(request, allow_constraint_overrun=True)
    estimate = status["resource_estimate"]
    if estimate["backend_bits"] != row["layout"]["actual_memory_bits"]:
        raise AssertionError("replayed and generated state sizes differ")
    return {
        **row,
        "request": str(request_path),
        "package": str(request.output_dir),
        "resource_estimate": estimate,
        "logical_violations": list(
            resource_violations(_estimate(estimate), constraints, physical=False)
        ),
    }


def _estimate(values: dict):
    from .ir import ResourceEstimate

    return ResourceEstimate(**values)


def resource_key(values: dict) -> tuple:
    return (
        values["estimated_stages"],
        values["estimated_sram_units"],
        values["estimated_tcam_units"],
        values["metadata_bits"],
    )


def run_workflow(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    compiler: str | Path | None = None,
) -> dict:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("application") != "flow_record_collection"
    ):
        raise ValueError("this release requires application: flow_record_collection")
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    constraints = HardwareConstraints(**config.get("constraints", {}))
    if any(
        value <= 0
        for name, value in asdict(constraints).items()
        if name.startswith("max_")
    ):
        raise ValueError("hardware limits must be positive")
    layouts = candidate_layouts(config.get("search", {}))
    compiler_value = compiler or config.get("target", {}).get("compiler")
    compiler_path = None
    if compiler_value:
        compiler_path = resolve(
            Path.cwd() if compiler else config_path.parent, str(compiler_value)
        )
        if not compiler_path.is_file():
            raise FileNotFoundError(compiler_path)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", config)
    input_spec = config.get("input", {})
    kind = input_spec.get("format", "pcap")
    if kind == "demo":
        trace = out / "input" / "demo.parquet"
        write_demo_trace(trace)
    elif kind == "pcap":
        trace = out / "input" / "trace.parquet"
        preprocess_pcap(
            resolve(config_path.parent, input_spec["path"]),
            trace,
            window_seconds=float(input_spec.get("window_seconds", 1)),
        )
    elif kind == "parquet":
        trace = resolve(config_path.parent, input_spec["path"])
    else:
        raise ValueError("input format must be pcap, parquet, or demo")
    training, validation = config.get("training", {}), config.get("validation", {})
    keys, counts, slots, workload_report = load_workload(
        trace,
        training_windows=training.get("windows", [0]),
        validation_windows=validation.get("windows", [1]),
        slot_duration_ms=int(validation.get("slot_duration_ms", 1000)),
        slots_per_window=int(validation.get("slots_per_window", 1)),
    )
    write_json(out / "workload.json", workload_report)
    models = train_models(config, keys, counts, out)
    if any(layout["backend"] != "flowradar" for layout in layouts) and not models:
        raise ValueError("learned layouts require at least one model")
    if any(
        layout["backend"] == "partitioned_flowradar" for layout in layouts
    ) and not any(model["feature_count"] == 104 for model in models):
        raise ValueError("partitioned_flowradar requires a 104-bit rule model")
    validation_keys = sorted({key for slot in slots for key in slot["packets"]})
    predictions = {}
    for model in models:
        features = deployment_feature_matrix(
            validation_keys, feature_count=model["feature_count"]
        )
        mapped = PersistedModelRuntime(model["plan"]).predict(features)
        original = PersistedModelRuntime(model["original_plan"]).predict(features)
        if not np.array_equal(mapped, original):
            raise AssertionError("selector normalization changed mapped predictions")
        predictions[model["model_id"]] = dict(zip(validation_keys, map(int, mapped)))
    rows = []
    for layout in layouts:
        compatible = (
            [None]
            if layout["backend"] == "flowradar"
            else [
                model
                for model in models
                if layout["backend"] != "partitioned_flowradar"
                or model["feature_count"] == 104
            ]
        )
        for model in compatible:
            model_id = model["model_id"] if model else "none"
            row = {
                "candidate_id": identity({"layout": layout, "model": model_id}),
                "backend": layout["backend"],
                "model_id": model_id,
                "layout": layout,
                "model_family": model["family"] if model else "none",
                "feature_count": model["feature_count"] if model else 0,
                **evaluate_layout(layout, slots, predictions.get(model_id)),
            }
            rows.append(export_candidate(row, model, out, constraints))
    cache = {}

    def compile_candidate(row):
        request = load_compilation_request(row["request"])
        status = compile_request(
            request, compiler=compiler_path, overwrite=True, compile_cache=cache
        )
        row["selection_status"] = status["stage"]
        row["target_compiled"] = bool(compiler_path and status["success"])
        if not status["success"]:
            return None
        if compiler_path:
            resources = status["compile_status"].get("compiler_resources", {})
            units = resources.get("memory_units", {})
            actual = (
                resources.get("pipeline_stage_span"),
                units.get("sram", 0),
                units.get("tcam", 0),
            )
            if not units or any(
                type(value) is not int or value < 0 for value in actual
            ):
                row["selection_status"] = "missing_target_resource_report"
                row["target_compiled"] = False
                return None
            row["compiler_resources"] = resources
            return actual
        return resource_key(row["resource_estimate"])

    result = ordered_target_search(
        rows,
        error=lambda row: row["validation_recovery_loss"],
        estimated_resources=lambda row: resource_key(row["resource_estimate"]),
        legal=lambda row: not row["logical_violations"],
        compile_candidate=compile_candidate,
        identity=lambda row: row["candidate_id"],
        conventional=lambda row: row["backend"] == "flowradar",
    )
    for row in result.rejected:
        row["selection_status"] = "logical_constraint_rejection"
    for row in result.skipped:
        row["selection_status"] = "worse_validation_error"
    write_json(out / "candidates.json", rows)
    selected = result.selected
    summary = {
        "format": "pleds_flowradar_run_v1",
        "success": selected is not None,
        "mode": "target_feedback" if compiler_path else "offline_estimates",
        "evaluation": "software_reference_replay",
        "candidate_count": len(rows),
        "examined_count": len(result.examined),
        "selected": selected,
        "target_compiled": bool(selected and selected.get("target_compiled")),
        "output_dir": str(out),
    }
    if selected:
        selected_dir = out / "selected"
        shutil.copytree(Path(selected["request"]).parent, selected_dir)
        # Cache hits may refer to another candidate's build; export it with the winner.
        if compiler_path:
            status = json.loads(
                (selected_dir / "package" / "tofino_compile_status.json").read_text()
            )
            build = Path(status["build_dir"])
            local_build = selected_dir / "package" / "tofino_build"
            if not local_build.exists():
                shutil.copytree(build, local_build)
        summary["selected_package"] = str(selected_dir / "package")
    write_json(out / "summary.json", summary)
    return summary
