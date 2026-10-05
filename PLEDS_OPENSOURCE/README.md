# PLEDS

Programmable LEarnable Data Structures

## Review Release

This repository provides a limited-scope release of PLEDS for peer review. It
includes the FlowRadar application and the compiler components required for its
training, optimization, and P4 generation.

Upon acceptance of the paper, we will release the complete PLEDS codebase,
including the remaining application implementations and evaluation scripts.

## Included Components

- Conventional FlowRadar, learned partitioning, and learned tiering.
- Eight learned-model frontends and their deployment mappings.
- Packet-derived features, shared CRC32 indexing, and ordered trace replay.
- Composition IR, operation dependency graphs, resource estimates, and
  target-feedback candidate selection.
- Tofino P4 generation, BFRT plans, initialization, and runtime installation code.

This repository contains source code, tests, and a self-contained synthetic
demonstration.

## Quick Start

Use Python 3.8 or newer. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
pleds run configs/run_flowradar.yaml --output-dir build/flowradar
```

The example runs locally, creates a synthetic trace, trains models, evaluates
FlowRadar layouts, and generates the selected deployment package. Outputs are
written to the requested directory. Use a new output directory for another run.

The command prints the selected backend, model family, package path, and summary
path. By default, selection uses resource estimates and the summary reports
`offline_estimates`. Add `--compiler` to enable Tofino compilation and
target-feedback selection.

## Input and Output

The YAML input specifies a trace, separate training and validation windows,
model families and parameters, FlowRadar memory layouts, and hardware limits.
Input paths are relative to the YAML file; `--output-dir` is relative to the
current working directory.

For a user-provided capture, edit `input.path` and window ranges in
`configs/run_flowradar_pcap.yaml`, then run:

```bash
pleds run configs/run_flowradar_pcap.yaml --output-dir build/my-flowradar
```

PCAP and PCAPNG are supported. A prepared Parquet trace is also accepted using
`input.format: parquet`. See [Configuration](docs/configuration.md) for its
schema and model options.

| Output | Purpose |
| --- | --- |
| `summary.json` | Selected candidate, validation recovery, and compilation status |
| `candidates.json` | Every evaluated candidate, resources, and selection outcome |
| `workload.json` | Training and validation window counts |
| `models/` | Trained and mapped model plans with training reports |
| `candidates/<id>/` | Candidate specification, compile request, model, and generated package |
| `selected/` | Self-contained copy of the winning specification and package |

Each package contains P4 source, a composition IR, an operation dependency
graph, a resource plan, a feature plan, BFRT/runtime entries, and
`runtime_control.py`.

## Tofino Compilation

To use target feedback during selection, supply a local `bf-p4c` executable in
an environment with the matching Tofino SDE:

```bash
pleds run configs/run_flowradar.yaml \
  --output-dir build/flowradar-target \
  --compiler "$SDE_INSTALL/bin/bf-p4c"
```

The compiler checks the generated BFRT plan against the target schema and uses
reported stages and memory units to select a feasible candidate. Load the
resulting program with the SDE tooling, then install its entries using the
generated `runtime_control.py` script.

An exported request can be compiled separately:

```bash
pleds compile build/flowradar/selected/request.yaml \
  --output-dir build/recompiled \
  --compiler "$SDE_INSTALL/bin/bf-p4c"
```

[Tofino Deployment](docs/tofino.md) describes the generated runtime files.

## Learned Models

| Frontend | Tiered FlowRadar input schema |
| --- | --- |
| `decision_tree` | 12 binary predicates or 104 five-tuple bits |
| `rule_list` | 12 binary predicates or 104 five-tuple bits |
| `tm_guided` | 12 binary predicates or 104 five-tuple bits |
| `random_forest_ensemble` | 12 binary predicates |
| `xgboost_ensemble` | 12 binary predicates |
| `naive_bayes_lookup` | 12 binary predicates |
| `piecewise_range` | 12 binary predicates |
| `isolation_forest` | 12 binary predicates |

Partitioned FlowRadar uses the 104-bit five-tuple schema with `decision_tree`,
`rule_list`, and `tm_guided`.

The compact schema contains 12 predicates derived from packet keys. The
five-tuple schema encodes two IPv4 addresses (32 bits each), two transport ports
(16 bits each), and the protocol field (8 bits), totaling 104 bits. Set
`feature_counts` to `[12]`, `[104]`, or `[12, 104]` according to the combination
above.

TM trains a Tsetlin machine and extracts bounded rules for deployment. The
12-feature selector mapping preserves the persisted model's binary decisions
over the complete 12-bit input space.

Install optional frontends before using them:

```bash
python -m pip install -e '.[tm,xgboost]'
```

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest tests
```

Tests cover feature extraction, model mapping, FlowRadar operations, and P4
generation.
Optional-model tests run when their dependencies are installed.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
