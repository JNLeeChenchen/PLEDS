"""Unified PLEDS lowering and Tofino compilation pipeline."""

from __future__ import annotations


from dataclasses import dataclass


import json


import math


from pathlib import Path


from typing import Any, Callable, Mapping


import yaml


from pleds.composition import (
    CompositionCosts,
    CompositionMechanism,
    build_composition_ir,
    infer_mechanism,
    output_for_mechanism,
)


from pleds.dependency_graph import DependencyGraph, apply_dependency_lower_bound


from pleds.deployment_contract import sha256_file, validate_compilation_contract


from pleds.ir import ModelIR, ResourceEstimate, build_logical_ir


from pleds.optimizer import resource_violations


from pleds.p4gen.flowradar import write_flowradar_artifacts


from pleds.p4gen.tiered_flowradar import write_tiered_flowradar_artifacts


from pleds.p4gen.binary_selector import compact_binary_selector


from pleds.p4gen.ternary import (
    lower_binary_rule_plan_to_ternary,
    optimized_rule_plan,
)


from pleds.spec import PledsSpec, load_spec


from pleds.tofino.compile import compile_with_cache, run_local_compile


BackendLowerer = Callable[..., Any]


_BACKENDS: dict[str, BackendLowerer] = {}


@dataclass(frozen=True)
class CompilationRequest:
    """Post-selection compiler input.

    Training and validation select ``model_plan`` and the backend parameters.
    Compilation consumes that frozen choice and never consults test metrics.
    """

    request_path: Path
    spec_path: Path
    spec: PledsSpec
    backend_type: str
    model_plan: Path | None
    inputs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    output_dir: Path
    deployment_contract: Path | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "request_path": str(self.request_path),
            "spec_path": str(self.spec_path),
            "backend_type": self.backend_type,
            "model_plan": str(self.model_plan) if self.model_plan else None,
            "inputs": dict(self.inputs),
            "parameters": dict(self.parameters),
            "output_dir": str(self.output_dir),
            "deployment_contract": (
                str(self.deployment_contract) if self.deployment_contract else None
            ),
        }


def register_backend(name: str) -> Callable[[BackendLowerer], BackendLowerer]:
    def decorate(function: BackendLowerer) -> BackendLowerer:
        if name in _BACKENDS:
            raise ValueError(f"backend already registered: {name}")
        _BACKENDS[name] = function
        return function

    return decorate


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_compilation_request(
    path: str | Path, *, output_dir: str | Path | None = None
) -> CompilationRequest:
    request_path = Path(path).resolve()
    raw = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("compile request must be a mapping")
    base = request_path.parent
    spec_path = _resolve(base, raw["spec"])
    spec = load_spec(spec_path)
    backend_raw = raw.get("backend", {}) or {}
    if not isinstance(backend_raw, dict):
        raise TypeError("request.backend must be a mapping")
    backend_type = str(backend_raw.get("type", spec.backend.type))
    matching_backends = tuple(
        backend for backend in spec.candidate_backends if backend.type == backend_type
    )
    if not matching_backends:
        raise ValueError(
            f"request backend {backend_type!r} is not declared by the specification"
        )
    model_value = raw.get("model_plan")
    model_plan = _resolve(base, model_value) if model_value else None
    inputs_raw = raw.get("inputs", {}) or {}
    if not isinstance(inputs_raw, dict):
        raise TypeError("request.inputs must be a mapping")
    inputs: dict[str, Any] = {}
    for key, value in inputs_raw.items():
        inputs[key] = str(_resolve(base, value)) if isinstance(value, str) else value
    selected_output = output_dir if output_dir is not None else raw.get("output")
    if selected_output is None:
        raise ValueError("compile request output is required")
    out = _resolve(base, selected_output)
    parameters = dict(matching_backends[0].params)
    parameters.update(backend_raw.get("parameters", {}) or {})
    contract_value = raw.get("deployment_contract")
    deployment_contract = _resolve(base, contract_value) if contract_value else None
    return CompilationRequest(
        request_path=request_path,
        spec_path=spec_path,
        spec=spec,
        backend_type=backend_type,
        model_plan=model_plan,
        inputs=inputs,
        parameters=parameters,
        output_dir=out,
        deployment_contract=deployment_contract,
    )


def _load_model_plan(path: Path | None) -> dict[str, object]:
    if path is None:
        raise ValueError("this backend requires model_plan")
    payload = json.loads(path.read_text(encoding="utf-8"))
    nested = payload.get("model_plan")
    plan = nested if isinstance(nested, dict) else payload
    if not isinstance(plan, dict) or "feature_count" not in plan or "rules" not in plan:
        raise ValueError(f"not a deployable RuleIR model plan: {path}")
    return optimized_rule_plan(plan)


def _resource_from_dict(payload: Mapping[str, Any]) -> ResourceEstimate:
    fields = {
        "model_rules",
        "model_key_bits",
        "metadata_bits",
        "backend_bits",
        "hash_count",
        "estimated_tcam_entries",
        "estimated_sram_units",
        "estimated_tcam_units",
        "estimated_stages",
        "estimated_phv_bits",
    }
    values = {key: payload[key] for key in fields if key in payload}
    return ResourceEstimate(**values)


def _write_composition_analysis(
    request: CompilationRequest,
    resource_payload: dict[str, Any],
) -> tuple[ResourceEstimate, Path, Path]:
    estimate = _resource_from_dict(resource_payload["resource_estimate"])
    logical = build_logical_ir(request.spec)
    model = None
    mechanism = infer_mechanism(request.backend_type)
    if request.model_plan is not None:
        raw = json.loads(request.model_plan.read_text(encoding="utf-8"))
        plan = raw.get("model_plan") if isinstance(raw.get("model_plan"), dict) else raw
        partition_count = int(
            request.parameters.get(
                "partition_count", request.parameters.get("partitions", 2)
            )
        )
        output = output_for_mechanism(mechanism, partition_count=partition_count)
        model = ModelIR(
            candidate_id=sha256_file(request.model_plan)[:16],
            family=str(plan.get("model_type", "lowered_model")),
            output=output,
            feature_count=int(plan.get("feature_count", estimate.model_key_bits or 1)),
            representation=str(
                plan.get(
                    "selected_representation",
                    plan.get("lowering", plan.get("format", "match_action")),
                )
            ),
            plan=plan,
        )
    else:
        mechanism = CompositionMechanism.CONVENTIONAL

    bfrt_path = request.output_dir / "bfrt_plan.json"
    feature_entries: list[Mapping[str, Any]] = []
    if bfrt_path.is_file():
        bfrt = json.loads(bfrt_path.read_text(encoding="utf-8"))
        raw_feature_entries = bfrt.get("feature_tables", {}).get("entries", [])
        if isinstance(raw_feature_entries, list):
            feature_entries = [
                entry for entry in raw_feature_entries if isinstance(entry, dict)
            ]
    feature_table_count = len(
        {
            str(entry.get("table", entry.get("table_name", "feature_table")))
            for entry in feature_entries
        }
    )
    inference_stages = 1 if model is not None else 0
    feature_stages = 1 if feature_table_count else 0
    backend_stages = max(
        1, estimate.estimated_stages - inference_stages - feature_stages
    )
    costs = CompositionCosts(
        feature_table_entries=len(feature_entries),
        feature_stages=feature_stages,
        inference_entries=estimate.model_rules,
        inference_tcam_units=estimate.estimated_tcam_units,
        inference_stages=inference_stages,
        backend_sram_units=estimate.estimated_sram_units,
        backend_stages=backend_stages,
        metadata_bits=estimate.metadata_bits,
        hash_units=estimate.hash_count,
    )
    composition = build_composition_ir(
        logical,
        backend_type=request.backend_type,
        model=model,
        costs=costs,
        mechanism=mechanism,
    )
    graph = DependencyGraph.from_composition(composition)
    estimate = apply_dependency_lower_bound(estimate, graph)
    resource_payload["resource_estimate"] = estimate.as_dict()
    resource_payload["dependency_analysis"] = {
        "critical_path_stages": graph.critical_path_stages,
        "peak_metadata_bits": graph.peak_metadata_bits,
    }
    composition_path = request.output_dir / "composition_ir.json"
    graph_path = request.output_dir / "dependency_graph.json"
    composition_path.write_text(
        json.dumps(composition.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    graph_path.write_text(
        json.dumps(graph.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (request.output_dir / "resource_plan.json").write_text(
        json.dumps(resource_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return estimate, composition_path, graph_path


def _package_manifest(
    *,
    request: CompilationRequest,
    program: Path,
    files: list[Path],
    details: Mapping[str, Any],
) -> dict[str, object]:
    logical = build_logical_ir(request.spec)
    return {
        "format": "pleds_compile_package_manifest_v2",
        "program": program.name,
        "compile_ready": True,
        "backend": request.backend_type,
        "files": [path.name for path in files],
        "source_request": str(request.request_path),
        "source_spec": str(request.spec_path),
        "source_model": str(request.model_plan) if request.model_plan else None,
        "source_deployment_contract": (
            str(request.deployment_contract) if request.deployment_contract else None
        ),
        "source_deployment_contract_sha256": (
            sha256_file(request.deployment_contract)
            if request.deployment_contract
            else None
        ),
        "logical_ir": {
            "name": logical.name,
            "task": logical.task,
            "key_schema": logical.key_schema,
            "feature_schema": list(logical.feature_schema),
            "model_output_type": logical.model_output_type,
            "backend_type": logical.backend_type,
            "semantic_guards": list(logical.semantic_guards),
            "objective": logical.objective,
        },
        **dict(details),
    }


def _lower_flowradar_common(
    request: CompilationRequest, out_dir: Path, *, partitioned: bool
) -> dict[str, object]:
    model_plan = (
        _load_model_plan(request.model_plan)
        if partitioned
        else {
            "format": "pleds_rule_ir_v1",
            "model_type": "none",
            "output_type": "binary",
            "feature_count": 104,
            "rule_count": 0,
            "rules": [],
        }
    )
    feature_count = int(model_plan["feature_count"])
    if feature_count != 104:
        raise ValueError("partitioned_flowradar requires 104 direct five-tuple bits")
    filter_bank_entries = [
        int(value) for value in request.parameters["filter_bank_entries"]
    ]
    counting_row_entries = [
        int(value) for value in request.parameters["counting_row_entries"]
    ]
    partition_count = 2 if partitioned else 1
    if (
        len(filter_bank_entries) != partition_count
        or len(counting_row_entries) != partition_count
    ):
        raise ValueError(
            f"{request.backend_type} requires {partition_count} partition dimensions"
        )
    if partitioned and (
        len(set(filter_bank_entries)) != 1 or len(set(counting_row_entries)) != 1
    ):
        raise ValueError("packed FlowRadar partitions require equal dimensions")
    filter_hashes = int(request.parameters.get("filter_hashes", 2))
    counting_hashes = int(request.parameters.get("counting_hashes", 2))
    if filter_hashes + counting_hashes > 8:
        raise ValueError("partitioned_flowradar requests too many target hashes")
    if any(value <= 0 or value & (value - 1) for value in filter_bank_entries):
        raise ValueError("FlowRadar filter bank dimensions must be powers of two")
    if any(value <= 0 or value & (value - 1) for value in counting_row_entries):
        raise ValueError("FlowRadar counting row dimensions must be powers of two")

    entries = [
        entry.as_dict() for entry in lower_binary_rule_plan_to_ternary(model_plan)
    ]
    backend_bits = sum(
        filter_hashes * filter_bank_entries[partition]
        + counting_hashes * counting_row_entries[partition] * 192
        for partition in range(partition_count)
    )
    sram_units = filter_hashes * max(
        1, math.ceil(sum(filter_bank_entries) / 131072)
    ) + counting_hashes * 6 * max(1, math.ceil(sum(counting_row_entries) / 4096))
    metadata_bits = (
        1
        + 1
        + 32
        + 4 * 32
        + 32
        + filter_hashes
        + 32 * (filter_hashes + counting_hashes)
    )
    estimate = ResourceEstimate(
        model_rules=len(entries),
        model_key_bits=feature_count if entries else 0,
        metadata_bits=metadata_bits,
        backend_bits=backend_bits,
        hash_count=filter_hashes + counting_hashes,
        estimated_tcam_entries=len(entries),
        estimated_sram_units=sram_units + (1 if partitioned else 0),
        estimated_tcam_units=(
            max(1, math.ceil(feature_count / 44))
            * max(1, math.ceil(len(entries) / 512))
            if entries
            else 0
        ),
        estimated_stages=(1 if partitioned else 0)
        + filter_hashes
        + 3 * counting_hashes,
        estimated_phv_bits=metadata_bits,
    )
    artifact = write_flowradar_artifacts(
        out_dir=out_dir,
        rule_plan=model_plan,
        resource_estimate=estimate,
        filter_bank_entries=filter_bank_entries,
        counting_row_entries=counting_row_entries,
        filter_hashes=filter_hashes,
        counting_hashes=counting_hashes,
        backend_type=request.backend_type,
    )
    files = [
        artifact.p4_path,
        artifact.bfrt_path,
        artifact.feature_path,
        artifact.resource_path,
        artifact.runtime_control_path,
        artifact.runtime_path,
    ]
    return _package_manifest(
        request=request,
        program=artifact.p4_path,
        files=files,
        details={
            "partition_count": partition_count,
            "filter_bank_entries": filter_bank_entries,
            "counting_row_entries": counting_row_entries,
            "filter_hashes": filter_hashes,
            "counting_hashes": counting_hashes,
            "selector_entry_count": len(entries),
            "semantic_checks": {
                "single_partition_update_per_packet": True,
                "flow_filter_mark_before_counting_update": True,
                "offline_decode_required": True,
            },
        },
    )


@register_backend("flowradar")
def _lower_flowradar(request: CompilationRequest, out_dir: Path) -> dict[str, object]:
    return _lower_flowradar_common(request, out_dir, partitioned=False)


@register_backend("partitioned_flowradar")
def _lower_partitioned_flowradar(
    request: CompilationRequest, out_dir: Path
) -> dict[str, object]:
    return _lower_flowradar_common(request, out_dir, partitioned=True)


@register_backend("tiered_flowradar")
def _lower_tiered_flowradar(
    request: CompilationRequest, out_dir: Path
) -> dict[str, object]:
    if request.model_plan is None:
        raise ValueError("tiered_flowradar requires model_plan")
    payload = json.loads(request.model_plan.read_text(encoding="utf-8"))
    model_plan = payload.get("model_plan", payload)
    feature_count = int(model_plan["feature_count"])
    if feature_count not in {12, 104}:
        raise ValueError(
            "tiered_flowradar requires 12 compact features or 104 five-tuple bits"
        )
    if feature_count == 12:
        model_plan = compact_binary_selector(model_plan)
    else:
        model_plan = _load_model_plan(request.model_plan)

    exact_bank_entries = [
        int(value) for value in request.parameters["exact_bank_entries"]
    ]
    if len(exact_bank_entries) not in {1, 2}:
        raise ValueError("tiered_flowradar requires one or two exact banks")
    fallback_filter_bank_entries = int(
        request.parameters["fallback_filter_bank_entries"]
    )
    fallback_counting_row_entries = int(
        request.parameters["fallback_counting_row_entries"]
    )
    flow_filter_hashes = int(request.parameters.get("flow_filter_hashes", 2))
    counting_hashes = int(request.parameters.get("counting_hashes", 2))
    dimensions = [
        *exact_bank_entries,
        fallback_filter_bank_entries,
        fallback_counting_row_entries,
    ]
    if any(value <= 0 or value & (value - 1) for value in dimensions):
        raise ValueError("tiered_flowradar dimensions must be powers of two")
    if flow_filter_hashes <= 0 or counting_hashes <= 0:
        raise ValueError("tiered_flowradar hash counts must be positive")
    if flow_filter_hashes + counting_hashes > 4:
        raise ValueError("tiered_flowradar reserves hash ids 4--7 for exact banks")

    entries = [
        entry.as_dict() for entry in lower_binary_rule_plan_to_ternary(model_plan)
    ]
    if not entries:
        raise ValueError("tiered_flowradar requires a non-empty model plan")
    exact_bits = sum(exact_bank_entries) * 192
    fallback_bits = (
        flow_filter_hashes * fallback_filter_bank_entries
        + counting_hashes * fallback_counting_row_entries * 192
    )
    exact_sram_units = sum(
        6 * max(1, math.ceil(entries_per_bank / 4096))
        for entries_per_bank in exact_bank_entries
    )
    fallback_sram_units = flow_filter_hashes * max(
        1, math.ceil(fallback_filter_bank_entries / 131072)
    ) + counting_hashes * 6 * max(1, math.ceil(fallback_counting_row_entries / 4096))
    exact_metadata_bits = sum(
        2 + 1 + max(1, int(math.log2(entries_per_bank))) + 2 * 32
        for entries_per_bank in exact_bank_entries
    )
    hash_count = flow_filter_hashes + counting_hashes + len(exact_bank_entries)
    metadata_bits = (
        3 + 2 * 16 + 5 * 32 + flow_filter_hashes + hash_count * 32 + exact_metadata_bits
    )
    metadata_bits += 12 if feature_count == 12 else 0
    estimate = ResourceEstimate(
        model_rules=len(entries),
        model_key_bits=feature_count,
        metadata_bits=metadata_bits,
        backend_bits=exact_bits + fallback_bits,
        hash_count=hash_count,
        estimated_tcam_entries=len(entries),
        estimated_sram_units=exact_sram_units + fallback_sram_units,
        estimated_tcam_units=max(1, math.ceil(feature_count / 44))
        * max(1, math.ceil(len(entries) / 512)),
        estimated_stages=(
            1 + 3 * len(exact_bank_entries) + flow_filter_hashes + 3 * counting_hashes
        ),
        estimated_phv_bits=metadata_bits,
    )
    artifact = write_tiered_flowradar_artifacts(
        out_dir=out_dir,
        rule_plan=model_plan,
        resource_estimate=estimate,
        exact_bank_entries=exact_bank_entries,
        fallback_filter_bank_entries=fallback_filter_bank_entries,
        fallback_counting_row_entries=fallback_counting_row_entries,
        flow_filter_hashes=flow_filter_hashes,
        counting_hashes=counting_hashes,
    )
    files = [
        artifact.p4_path,
        artifact.bfrt_path,
        artifact.feature_path,
        artifact.resource_path,
        artifact.runtime_control_path,
        artifact.runtime_path,
    ]
    return _package_manifest(
        request=request,
        program=artifact.p4_path,
        files=files,
        details={
            "exact_bank_entries": exact_bank_entries,
            "fallback_filter_bank_entries": fallback_filter_bank_entries,
            "fallback_counting_row_entries": fallback_counting_row_entries,
            "flow_filter_hashes": flow_filter_hashes,
            "counting_hashes": counting_hashes,
            "selector_entry_count": len(entries),
            "semantic_checks": {
                "single_record_path_per_packet": True,
                "exact_tier_stores_full_five_tuple": True,
                "exact_tier_fingerprint_bits": 31,
                "exact_collision_falls_back_to_flowradar": True,
                "offline_decode_required": True,
            },
        },
    )


def compile_request(
    request: CompilationRequest,
    *,
    compiler: str | Path | None = None,
    overwrite: bool = False,
    allow_constraint_overrun: bool = False,
    compile_cache: dict | None = None,
) -> dict[str, object]:
    """Lower one frozen design, enforce constraints, and optionally run bf-p4c."""

    if request.backend_type not in _BACKENDS:
        raise ValueError(
            f"unsupported backend {request.backend_type!r}; available: {', '.join(available_backends())}"
        )
    if request.deployment_contract is not None:
        validate_compilation_contract(
            contract_path=request.deployment_contract,
            backend_type=request.backend_type,
            model_plan=request.model_plan,
            parameters=request.parameters,
        )
    out = request.output_dir
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = _BACKENDS[request.backend_type](request, out)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resource_payload = json.loads(
        (out / "resource_plan.json").read_text(encoding="utf-8")
    )
    estimate, composition_path, graph_path = _write_composition_analysis(
        request, resource_payload
    )
    manifest["files"] = list(
        dict.fromkeys(
            [*manifest.get("files", []), composition_path.name, graph_path.name]
        )
    )
    manifest["composition_ir"] = composition_path.name
    manifest["dependency_graph"] = graph_path.name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    violations = list(
        resource_violations(
            estimate, request.spec.constraints, physical=compiler is None
        )
    )
    if manifest.get("compile_ready") is not True:
        violations.append("backend lowering is not compile-ready")
    semantic = resource_payload.get("semantic_properties", {})
    if (
        request.spec.constraints.no_false_negative
        and semantic.get("no_false_negative") is not True
    ):
        violations.append("no_false_negative semantic guard is not satisfied")
    if (
        request.spec.constraints.no_lost_update
        and semantic.get("no_lost_update") is not True
    ):
        violations.append("no_lost_update semantic guard is not satisfied")
    if violations and not allow_constraint_overrun:
        status = {
            "format": "pleds_compilation_result_v1",
            "success": False,
            "stage": "static_constraint_check",
            "violations": violations,
            "manifest": str(manifest_path),
            "resource_estimate": estimate.as_dict(),
        }
        (out / "compilation_result.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return status
    compile_status = None
    compiler_violations: list[str] = []
    if compiler is not None:
        compile_status = (
            run_local_compile(out, compiler=compiler, overwrite=overwrite)
            if compile_cache is None
            else compile_with_cache(
                out,
                compiler=compiler,
                overwrite=overwrite,
                cache=compile_cache,
                runner=run_local_compile,
            )
        )
        resources = compile_status.get("compiler_resources", {})
        units = resources.get("memory_units", {}) if isinstance(resources, dict) else {}
        physical_checks = (
            (
                "compiled_stages",
                resources.get("pipeline_stage_span"),
                request.spec.constraints.max_stages,
            ),
            (
                "compiled_tcam_units",
                units.get("tcam"),
                request.spec.constraints.max_tcam_units,
            ),
            (
                "compiled_sram_units",
                units.get("sram"),
                request.spec.constraints.max_sram_units,
            ),
        )
        compiler_violations = [
            f"{name}={actual} exceeds {limit}"
            for name, actual, limit in physical_checks
            if isinstance(actual, int) and actual > limit
        ]
    constraints_passed = not violations and not compiler_violations
    bfrt_valid = compile_status is None or all(
        not isinstance(compile_status.get(field), dict)
        or bool(compile_status[field].get("valid"))
        for field in ("bfrt_validation", "runtime_plan_validation")
    )
    if not bfrt_valid:
        compiler_violations.append(
            "generated BFRT plan does not match the compiler schema"
        )
        constraints_passed = False
    success = (constraints_passed or allow_constraint_overrun) and (
        compile_status is None
        or (bool(compile_status["compile_success"]) and bfrt_valid)
    )
    if success:
        stage = "complete"
    elif compile_status is not None and not compile_status["compile_success"]:
        stage = "bf_p4c"
    elif not bfrt_valid:
        stage = "bfrt_validation"
    else:
        stage = "constraint_check_after_compile"
    status = {
        "format": "pleds_compilation_result_v1",
        "success": success,
        "stage": stage,
        "backend": request.backend_type,
        "manifest": str(manifest_path),
        "static_constraint_violations": violations,
        "compiler_constraint_violations": compiler_violations,
        "constraint_overrun_allowed": allow_constraint_overrun,
        "compile_status": compile_status,
        "resource_estimate": estimate.as_dict(),
    }
    (out / "compilation_result.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


__all__ = [
    "CompilationRequest",
    "available_backends",
    "compile_request",
    "load_compilation_request",
    "register_backend",
]
