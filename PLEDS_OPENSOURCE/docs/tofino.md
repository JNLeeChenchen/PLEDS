# Tofino Deployment

The generated programs target Tofino 1 using P4-16 and TNA. A matching SDE,
`bf-p4c`, and BFRT Python bindings must be supplied by the user.

## Package

| File | Role |
| --- | --- |
| `*.p4` | Generated packet parser, inference, hashing, and FlowRadar state updates |
| `manifest.json` | Program name and package members |
| `model_plan.json` (parent directory) | Selected learned-model decisions |
| `feature_plan.json` | Packet fields and feature mapping |
| `bfrt_plan.json` | Table entries and register initialization/reset plan |
| `runtime_entries.json` | Runtime-facing layout and hash configuration |
| `runtime_control.py` | BFRT installation and reset commands |
| `resource_plan.json` | Logical state capacity and resource estimates |
| `composition_ir.json` | Model/data-structure composition |
| `dependency_graph.json` | Operation dependencies and stage lower bounds |
| `compilation_result.json` | Generation, constraint, and target-compilation status |
| `tofino_build/` | Target artifacts, when target compilation was requested |

CRC definitions, input field order, and index reduction are shared through
`hash_spec.py`. The runtime plan records the hash profile and register
dimensions. Tiered FlowRadar uses 31-bit fingerprints, matching the reference
replay and the generated program.

## Runtime

The runtime script defaults to a dry run:

```bash
python build/flowradar/selected/package/runtime_control.py \
  --plan build/flowradar/selected/package/bfrt_plan.json
```

Load the compiled program using the switch's SDE tooling and configure ports
for the generated forwarding behavior. Then use the matching BFRT environment
to install entries:

```bash
python build/flowradar-target/selected/package/runtime_control.py \
  --plan build/flowradar-target/selected/package/bfrt_plan.json \
  --grpc-addr localhost:50052 --install
```

Installation resets the package's registers and starts an empty collection
interval. `--reset-state` resets those registers again without reinstalling
model entries. Both commands change the selected program's state. Choose
`--device-id` and `--pipe-id` for the intended deployment.
