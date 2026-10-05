import json

from pleds.tofino.compile import compile_with_cache


def test_identical_code_revalidates_each_package_entries(tmp_path, monkeypatch):
    packages = [tmp_path / "first", tmp_path / "second"]
    build = tmp_path / "build"
    build.mkdir()
    (build / "bfrt.json").write_text("{}")
    for package in packages:
        package.mkdir()
        (package / "manifest.json").write_text(json.dumps({"program": "program.p4"}))
        (package / "program.p4").write_text("identical target program")
        (package / "bfrt_plan.json").write_text(json.dumps({"owner": package.name}))
    compiled, rebound = [], []

    def runner(package, **kwargs):
        compiled.append(package.name)
        return {
            "compile_success": True,
            "build_dir": str(build),
            "elapsed_seconds": 10,
            "bfrt_validation": {"valid": True},
        }

    def binding(plan, schema):
        rebound.append(plan["owner"])
        return {"valid": False, "errors": ["invalid second plan"]}

    monkeypatch.setattr("pleds.tofino.compile.validate_bfrt_plan", binding)
    cache = {}
    first = compile_with_cache(packages[0], compiler="mock", cache=cache, runner=runner)
    second = compile_with_cache(
        packages[1], compiler="mock", cache=cache, runner=runner
    )
    assert compiled == ["first"]
    assert rebound == ["second"]
    assert first["bfrt_validation"]["valid"]
    assert not second["bfrt_validation"]["valid"]
    assert second["elapsed_seconds"] == 0
