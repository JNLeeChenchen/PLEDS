"""Application-level PLEDS specification parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml


class ModelOutputKind(str, Enum):
    """Typed interface between a learned model and a data structure."""

    BINARY_DECISION = "binary_decision"
    HOT_COLD = "hot_cold"
    PARTITION_ID = "partition_id"


_OUTPUT_ALIASES = {
    "binary": ModelOutputKind.BINARY_DECISION,
    "binary_decision": ModelOutputKind.BINARY_DECISION,
    "hot_cold": ModelOutputKind.HOT_COLD,
    "hot/cold": ModelOutputKind.HOT_COLD,
    "partition": ModelOutputKind.PARTITION_ID,
    "partition_id": ModelOutputKind.PARTITION_ID,
    "score_bucket": ModelOutputKind.PARTITION_ID,
}


@dataclass(frozen=True)
class ModelOutputSpec:
    kind: ModelOutputKind = ModelOutputKind.BINARY_DECISION
    width: int = 1

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("model output width must be positive")
        if (
            self.kind
            in {
                ModelOutputKind.BINARY_DECISION,
                ModelOutputKind.HOT_COLD,
            }
            and self.width != 1
        ):
            raise ValueError(f"{self.kind.value} requires a one-bit output")

    @property
    def value_count(self) -> int:
        return 1 << self.width


@dataclass(frozen=True)
class HardwareConstraints:
    no_false_negative: bool = False
    no_lost_update: bool = False
    max_stages: int = 8
    max_tcam_entries: int = 1024
    max_tcam_units: int = 8
    max_sram_units: int = 12
    max_metadata_bits: int = 256
    max_hash_units: int = 16


@dataclass(frozen=True)
class BackendSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    mechanism: str | None = None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    field: str
    field_width: int
    direct_entry_expansion: int | None = 1
    mat_entries: int = 1


@dataclass(frozen=True)
class ObjectiveSpec:
    minimize: str | None = None
    maximize: str | None = None

    @property
    def direction(self) -> str:
        if self.minimize is not None:
            return "minimize"
        if self.maximize is not None:
            return "maximize"
        return "unspecified"

    @property
    def metric(self) -> str:
        return self.minimize or self.maximize or "unspecified"


@dataclass(frozen=True)
class PledsSpec:
    name: str
    task: str
    key: str
    features: tuple[str, ...]
    models: tuple[str, ...]
    backend: BackendSpec
    constraints: HardwareConstraints
    objective: ObjectiveSpec
    model_output: ModelOutputSpec = field(default_factory=ModelOutputSpec)
    backend_candidates: tuple[BackendSpec, ...] = ()
    mechanisms: tuple[str, ...] = ()
    model_parameters: Mapping[str, tuple[Mapping[str, Any], ...]] = field(
        default_factory=dict
    )
    include_conventional: bool = True
    feature_specs: tuple[FeatureSpec, ...] = ()

    @property
    def model_output_type(self) -> str:
        """Compatibility property for callers that previously consumed a string."""

        return self.model_output.kind.value

    @property
    def candidate_backends(self) -> tuple[BackendSpec, ...]:
        return self.backend_candidates or (self.backend,)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"expected list, got {type(value).__name__}")
    return tuple(str(item) for item in value)


def _parse_features(value: Any) -> tuple[tuple[str, ...], tuple[FeatureSpec, ...]]:
    if value is None:
        return (), ()
    if not isinstance(value, list):
        raise TypeError("spec.features must be a list")
    names: list[str] = []
    detailed: list[FeatureSpec] = []
    for item in value:
        if isinstance(item, str):
            names.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError("feature entries must be strings or mappings")
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("detailed feature entry requires a name")
        direct = item.get("direct_entry_expansion", 1)
        detailed.append(
            FeatureSpec(
                name=name,
                kind=str(item["kind"]),
                field=str(item["field"]),
                field_width=int(item["field_width"]),
                direct_entry_expansion=None if direct is None else int(direct),
                mat_entries=int(item.get("mat_entries", 1)),
            )
        )
        names.append(name)
    return tuple(names), tuple(detailed)


def _parse_backend(raw: Any) -> BackendSpec:
    if not isinstance(raw, dict) or "type" not in raw:
        raise ValueError("each backend requires a type")
    params = raw.get("parameters", raw.get("params", {})) or {}
    if not isinstance(params, dict):
        raise TypeError("backend parameters must be a mapping")
    inline = {
        key: value
        for key, value in raw.items()
        if key not in {"type", "mechanism", "parameters", "params"}
    }
    return BackendSpec(
        type=str(raw["type"]),
        params={**inline, **params},
        mechanism=(str(raw["mechanism"]) if raw.get("mechanism") else None),
    )


def _parse_backends(raw: Mapping[str, Any]) -> tuple[BackendSpec, ...]:
    candidates_raw = raw.get("backends")
    if candidates_raw is None:
        backend_raw = raw.get("backend")
        if backend_raw is None:
            raise ValueError("spec.backend or spec.backends is required")
        return (_parse_backend(backend_raw),)
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("spec.backends must be a non-empty list")
    candidates = tuple(_parse_backend(item) for item in candidates_raw)
    names = [candidate.type for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("spec.backends contains duplicate backend types")
    return candidates


def _parse_models(
    value: Any,
) -> tuple[tuple[str, ...], Mapping[str, tuple[Mapping[str, Any], ...]]]:
    if value is None:
        return (), {}
    if not isinstance(value, list):
        raise TypeError("spec.models must be a list")
    names: list[str] = []
    parameter_grid: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for item in value:
        if isinstance(item, str):
            name = item
            variants: tuple[Mapping[str, Any], ...] = ({},)
        elif isinstance(item, dict):
            name = str(item.get("type", item.get("name", "")))
            if not name:
                raise ValueError("model entry requires a type")
            raw_variants = item.get("parameters", item.get("parameter_grid", [{}]))
            if isinstance(raw_variants, dict):
                raw_variants = [raw_variants]
            if not isinstance(raw_variants, list) or not raw_variants:
                raise ValueError(f"model {name!r} requires a non-empty parameter grid")
            if not all(isinstance(variant, dict) for variant in raw_variants):
                raise TypeError(f"model {name!r} parameters must be mappings")
            variants = tuple(dict(variant) for variant in raw_variants)
        else:
            raise TypeError("model entries must be strings or mappings")
        if name in parameter_grid:
            raise ValueError(f"duplicate model family: {name}")
        names.append(name)
        parameter_grid[name] = variants
    return tuple(names), parameter_grid


def _parse_output(raw: Any, backends: tuple[BackendSpec, ...]) -> ModelOutputSpec:
    if raw is None:
        mechanisms = {backend.mechanism for backend in backends if backend.mechanism}
        types = {backend.type for backend in backends}
        tiered = {
            "tiered_flowradar",
        }
        partitioned = {
            "partitioned_flowradar",
        }
        if mechanisms == {"tiering"} or (not mechanisms and types <= tiered):
            return ModelOutputSpec(ModelOutputKind.HOT_COLD, 1)
        if mechanisms == {"partitioning"} or (not mechanisms and types <= partitioned):
            counts = []
            for backend in backends:
                value = backend.params.get(
                    "partition_count", backend.params.get("partitions", 2)
                )
                if isinstance(value, int):
                    counts.append(value)
            width = max(1, max(counts, default=2).bit_length() - 1)
            if counts and (1 << width) < max(counts):
                width += 1
            return ModelOutputSpec(ModelOutputKind.PARTITION_ID, width)
        return ModelOutputSpec()
    if isinstance(raw, str):
        output_type = raw
        width = 1
    elif isinstance(raw, dict):
        output_type = str(raw.get("type", "binary"))
        width = int(raw.get("width", 1))
    else:
        raise TypeError("spec.model_output must be a string or mapping")
    try:
        kind = _OUTPUT_ALIASES[output_type.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model output type: {output_type}") from exc
    return ModelOutputSpec(kind=kind, width=width)


def load_spec(path: str | Path) -> PledsSpec:
    spec_path = Path(path)
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"spec must be a mapping: {spec_path}")

    constraints_raw = raw.get("constraints", {}) or {}
    objective_raw = raw.get("objective", {}) or {}
    if not isinstance(constraints_raw, dict):
        raise TypeError("spec.constraints must be a mapping")
    if not isinstance(objective_raw, dict):
        raise TypeError("spec.objective must be a mapping")

    backends = _parse_backends(raw)
    if str(raw.get("task")) != "flow_record_collection":
        raise ValueError("this release supports flow_record_collection only")
    supported_backends = {"flowradar", "partitioned_flowradar", "tiered_flowradar"}
    if any(backend.type not in supported_backends for backend in backends):
        raise ValueError("this release supports FlowRadar backends only")
    models, model_parameters = _parse_models(raw.get("models"))
    features, feature_specs = _parse_features(raw.get("features"))
    mechanisms_raw = raw.get("mechanisms", raw.get("mechanism"))
    if isinstance(mechanisms_raw, str):
        mechanisms = (mechanisms_raw,)
    else:
        mechanisms = _as_tuple(mechanisms_raw)

    return PledsSpec(
        name=str(raw["name"]),
        task=str(raw["task"]),
        key=str(raw["key"]),
        features=features,
        models=models,
        backend=backends[0],
        backend_candidates=backends,
        mechanisms=mechanisms,
        model_parameters=model_parameters,
        include_conventional=bool(raw.get("include_conventional", True)),
        feature_specs=feature_specs,
        constraints=HardwareConstraints(
            no_false_negative=bool(constraints_raw.get("no_false_negative", False)),
            no_lost_update=bool(constraints_raw.get("no_lost_update", False)),
            max_stages=int(constraints_raw.get("max_stages", 8)),
            max_tcam_entries=int(constraints_raw.get("max_tcam_entries", 1024)),
            max_tcam_units=int(constraints_raw.get("max_tcam_units", 8)),
            max_sram_units=int(constraints_raw.get("max_sram_units", 12)),
            max_metadata_bits=int(constraints_raw.get("max_metadata_bits", 256)),
            max_hash_units=int(constraints_raw.get("max_hash_units", 16)),
        ),
        objective=ObjectiveSpec(
            minimize=objective_raw.get("minimize"),
            maximize=objective_raw.get("maximize"),
        ),
        model_output=_parse_output(raw.get("model_output"), backends),
    )


__all__ = [
    "BackendSpec",
    "FeatureSpec",
    "HardwareConstraints",
    "ModelOutputKind",
    "ModelOutputSpec",
    "ObjectiveSpec",
    "PledsSpec",
    "load_spec",
]
