"""Command-line interface for the FlowRadar review release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .compiler import available_backends, compile_request, load_compilation_request
from .models.registry import available_model_frontends
from .workflow import run_workflow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pleds", description="PLEDS FlowRadar compiler"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="train, select, and generate a FlowRadar deployment"
    )
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--compiler", type=Path, help="local bf-p4c executable")
    compile_parser = commands.add_parser(
        "compile", help="generate or compile a selected request"
    )
    compile_parser.add_argument("request", type=Path)
    compile_parser.add_argument("--output-dir", type=Path)
    compile_parser.add_argument("--compiler", type=Path)
    compile_parser.add_argument("--overwrite", action="store_true")
    commands.add_parser(
        "inspect", help="list included application backends and model frontends"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(
                json.dumps(
                    {
                        "application": "flow_record_collection",
                        "backends": available_backends(),
                        "models": available_model_frontends(),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            result = run_workflow(args.config, args.output_dir, compiler=args.compiler)
            selected = result.get("selected")
            print(
                json.dumps(
                    {
                        "success": result["success"],
                        "mode": result["mode"],
                        "candidate_count": result["candidate_count"],
                        "selected_backend": selected["backend"] if selected else None,
                        "selected_model": (
                            selected["model_family"] if selected else None
                        ),
                        "selected_package": result.get("selected_package"),
                        "summary": str(args.output_dir.resolve() / "summary.json"),
                    },
                    indent=2,
                )
            )
        else:
            output = args.output_dir.resolve() if args.output_dir else None
            request = load_compilation_request(args.request, output_dir=output)
            result = compile_request(
                request, compiler=args.compiler, overwrite=args.overwrite
            )
            print(json.dumps(result, indent=2))
        return 0 if result["success"] else 2
    except (ValueError, TypeError, OSError, ImportError, KeyError) as exc:
        print(f"pleds: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
