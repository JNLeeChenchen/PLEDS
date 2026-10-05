"""Local Tofino compilation and artifact caching for PLEDS packages."""

from __future__ import annotations


import json


import hashlib


import subprocess


import time


from pathlib import Path


from pleds.tofino.bfrt import validate_bfrt_plan


from pleds.tofino.context_resources import summarize_context_file


def _validate_p4_14_runtime_plan(
    plan: dict[str, object], context: dict[str, object]
) -> dict[str, object]:
    tables = {
        str(item.get("name")): item
        for item in context.get("tables") or []
        if isinstance(item, dict)
    }
    required = {
        str(item.get("name"))
        for item in (plan.get("state_reset") or {}).get("registers") or []
        if isinstance(item, dict)
    }
    required.update(
        str(name)
        for name in (plan.get("end_window_query") or {}).get("candidate_source") or []
    )
    selector = plan.get("selector") or {}
    selector_entries = (
        (selector.get("entries") or []) if isinstance(selector, dict) else []
    )
    selector_table = (
        str(selector.get("table", "")) if isinstance(selector, dict) else ""
    )
    if selector_entries and selector_table:
        required.add(selector_table)
    missing_objects = sorted(name for name in required if name and name not in tables)

    missing_fields: list[str] = []
    missing_actions: list[str] = []
    if selector_entries and selector_table in tables:
        compiled = tables[selector_table]
        compiled_fields = {
            str(item.get("name"))
            for item in compiled.get("match_key_fields") or []
            if isinstance(item, dict)
        }
        compiled_actions = {
            str(item.get("name"))
            for item in compiled.get("actions") or []
            if isinstance(item, dict)
        }
        requested_fields = {
            str(field)
            for entry in selector_entries
            if isinstance(entry, dict)
            for field in (entry.get("key") or {})
        }
        requested_actions = {
            str(entry.get("action"))
            for entry in selector_entries
            if isinstance(entry, dict)
        }
        missing_fields = sorted(requested_fields - compiled_fields)
        missing_actions = sorted(requested_actions - compiled_actions)
    return {
        "format": "pleds_p4_14_runtime_plan_validation_v1",
        "valid": not (missing_objects or missing_fields or missing_actions),
        "missing_objects": missing_objects,
        "missing_selector_fields": missing_fields,
        "missing_selector_actions": missing_actions,
    }


def _write_p4_14_runtime_conf(build: Path, program_name: str) -> Path:
    """Emit a bf_switchd config for P4_14 artifacts exposed through BFRT."""

    native_conf = build / f"{program_name}.conf"
    native = json.loads(native_conf.read_text(encoding="utf-8"))
    payload = {
        "chip_list": native.get("chip_list", []),
        "instance": int(native.get("instance", 0)),
        "p4_devices": [
            {
                "device-id": 0,
                "p4_programs": [
                    {
                        "program-name": program_name,
                        "bfrt-config": "bfrt.json",
                        "p4_pipelines": [
                            {
                                "p4_pipeline_name": "pipe",
                                "context": "context.json",
                                "config": "tofino.bin",
                                "pipe_scope": [0, 1, 2, 3],
                                "path": ".",
                            }
                        ],
                    }
                ],
                "agent0": "lib/libpltfm_mgr.so",
            }
        ],
    }
    output = build / f"{program_name}.runtime.conf"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def run_local_compile(
    package_dir: str | Path,
    *,
    compiler: str | Path,
    build_name: str = "tofino_build",
    overwrite: bool = False,
    timeout: int = 1200,
) -> dict[str, object]:
    """Compile and validate one generated package with a local ``bf-p4c``.

    The returned resources come from compiler-produced ``context.json`` rather
    than the pre-compile estimator.  Existing build directories are preserved
    unless the caller explicitly opts into replacement.
    """

    package = Path(package_dir).resolve()
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program = package / str(manifest.get("program", ""))
    if not program.is_file():
        raise FileNotFoundError(program)
    compiler_path = Path(compiler).resolve()
    if not compiler_path.is_file():
        raise FileNotFoundError(compiler_path)
    build = package / build_name
    if build.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing build: {build}")
        import shutil

        shutil.rmtree(build)

    started = time.perf_counter()
    p4_version = str(manifest.get("p4_version", "p4-16"))
    if p4_version == "p4-14":
        command = [
            str(compiler_path),
            "--target",
            "tofino",
            "--std",
            "p4-14",
            "--bf-rt-schema",
            str(build / "bfrt.json"),
            "--output",
            str(build),
            str(program),
        ]
        architecture = "tofino_native"
    else:
        command = [
            str(compiler_path),
            "--target",
            "tofino",
            "--arch",
            "tna",
            "--output",
            str(build),
            str(program),
        ]
        architecture = "tna"
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path = package / "tofino_compiler.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    result: dict[str, object] = {
        "format": "pleds_tofino_compile_status_v2",
        "compile_success": process.returncode == 0,
        "returncode": process.returncode,
        "compiler": str(compiler_path),
        "target": "tofino",
        "architecture": architecture,
        "p4_version": p4_version,
        "program": str(program),
        "build_dir": str(build),
        "elapsed_seconds": elapsed,
        "compiler_log": str(log_path),
    }
    if process.returncode == 0:
        if p4_version == "p4-14":
            runtime_conf = _write_p4_14_runtime_conf(build, program.stem)
            result["runtime_conf"] = str(runtime_conf)
            result["runtime_state_initialization"] = (
                "cold_load_required_for_p4_14_stateful_alu_registers"
            )
        context_path = (
            build / "context.json"
            if p4_version == "p4-14"
            else build / "pipe" / "context.json"
        )
        resources = summarize_context_file(context_path)
        result["compiler_resources"] = resources
        plan_path = package / "bfrt_plan.json"
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if p4_version == "p4-14":
                context = json.loads(context_path.read_text(encoding="utf-8"))
                result["runtime_plan_validation"] = _validate_p4_14_runtime_plan(
                    plan, context
                )
            else:
                bfrt = json.loads((build / "bfrt.json").read_text(encoding="utf-8"))
                result["bfrt_validation"] = validate_bfrt_plan(plan, bfrt)
    status_path = package / "tofino_compile_status.json"
    status_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def compile_with_cache(package, *, compiler, cache, overwrite=False, runner=None):
    """Reuse target code, revalidating each package's installed state plan."""
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text())
    program = package / manifest["program"]
    version = manifest.get("p4_version", "p4-16")
    key = (
        str(Path(compiler).resolve()),
        version,
        hashlib.sha256(program.read_bytes()).hexdigest(),
    )
    runner = runner or run_local_compile
    if key not in cache:
        status = runner(package, compiler=compiler, overwrite=overwrite)
        cache[key] = (package, status)
        return status
    source, original = cache[key]
    status = {
        **original,
        "program": str(program),
        "elapsed_seconds": 0.0,
        "compile_reused_from": str(source),
        "shared_build_artifact": True,
    }
    plan_path = package / "bfrt_plan.json"
    if status.get("compile_success") and plan_path.is_file():
        build = Path(status["build_dir"])
        plan = json.loads(plan_path.read_text())
        if version == "p4-14":
            status["runtime_plan_validation"] = _validate_p4_14_runtime_plan(
                plan, json.loads((build / "context.json").read_text())
            )
        else:
            status["bfrt_validation"] = validate_bfrt_plan(
                plan, json.loads((build / "bfrt.json").read_text())
            )
    (package / "tofino_compile_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    return status
