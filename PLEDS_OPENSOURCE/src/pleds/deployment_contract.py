"""Validation for experiment-bound deployment contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_FORMAT = "pleds_deployment_contract_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_deployment_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != CONTRACT_FORMAT:
        raise ValueError(f"unsupported deployment contract: {contract_path}")
    return payload


def validate_compilation_contract(
    *,
    contract_path: str | Path,
    backend_type: str,
    model_plan: Path | None,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a compile request differs from its evaluated realization."""

    path = Path(contract_path).resolve()
    contract = load_deployment_contract(path)
    backend = contract.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("deployment contract is missing backend")
    if backend.get("type") != backend_type:
        raise ValueError(
            f"deployment contract backend {backend.get('type')!r} does not match "
            f"compile request {backend_type!r}"
        )

    expected_parameters = backend.get("parameters")
    if not isinstance(expected_parameters, dict):
        raise ValueError("deployment contract is missing backend parameters")
    for name, expected in expected_parameters.items():
        actual = parameters.get(name)
        if actual != expected:
            raise ValueError(
                f"deployment contract parameter mismatch for {name}: "
                f"evaluated={expected!r}, compile_request={actual!r}"
            )

    model = contract.get("model_plan")
    if model is None:
        if model_plan is not None:
            raise ValueError(
                "deployment contract requires the original backend without a model"
            )
    else:
        if not isinstance(model, dict):
            raise ValueError("deployment contract model_plan must be an object or null")
        if model_plan is None:
            raise ValueError("deployment contract requires a model plan")
        expected_model_hash = str(model.get("sha256", ""))
        actual_model_hash = sha256_file(model_plan)
        if not expected_model_hash or actual_model_hash != expected_model_hash:
            raise ValueError(
                "deployment contract model hash does not match the compile request"
            )

    artifacts = contract.get("bound_artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("deployment contract bound_artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("invalid deployment contract artifact record")
        artifact_path = Path(str(artifact["path"]))
        expected_hash = str(artifact["sha256"])
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"deployment contract input is missing: {artifact_path}"
            )
        if sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"deployment contract input changed: {artifact_path}")
    return contract


__all__ = [
    "CONTRACT_FORMAT",
    "load_deployment_contract",
    "sha256_file",
    "validate_compilation_contract",
]
