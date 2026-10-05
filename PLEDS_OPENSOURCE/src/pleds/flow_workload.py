"""Ordered FlowRadar workloads and a small, self-contained demonstration trace."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .traces import TRACE_SCHEMA
from .training import FLOW_KEY_DTYPE


KEY_FIELDS = tuple(FLOW_KEY_DTYPE.names)


def write_demo_trace(path: Path) -> None:
    """Write synthetic packets with documentation-only IPv4 addresses."""
    rows = []
    ordinal = 0
    for window in range(2):
        packets = []
        for flow in range(512):
            hot = flow < 64
            key = (
                0xC0000200 + flow % 256,
                0xC6336400 + flow // 256,
                10000 + flow,
                443 if hot else 20000 + flow,
                6 if hot else 17,
            )
            packets.extend([key] * (24 if hot else 1 + flow % 3))
        rng = np.random.default_rng(23 + window)
        rng.shuffle(packets)
        for index, key in enumerate(packets):
            rows.append(
                (ordinal, window * 1_000_000_000 + index * 100_000, window, *key, 64)
            )
            ordinal += 1
    columns = list(zip(*rows))
    table = pa.Table.from_arrays(
        [
            pa.array(column, type=field.type)
            for column, field in zip(columns, TRACE_SCHEMA)
        ],
        schema=TRACE_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def load_workload(
    path: Path,
    *,
    training_windows: list[int],
    validation_windows: list[int],
    slot_duration_ms: int,
    slots_per_window: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Aggregate training flows and keep validation packets in capture order."""
    if not training_windows or not validation_windows:
        raise ValueError("training and validation windows must both be non-empty")
    if len(set(training_windows)) != len(training_windows) or len(
        set(validation_windows)
    ) != len(validation_windows):
        raise ValueError("window lists must not contain duplicates")
    if any(
        type(window) is not int or window < 0
        for window in training_windows + validation_windows
    ):
        raise ValueError("window ids must be nonnegative integers")
    if max(training_windows) >= min(validation_windows):
        raise ValueError("training windows must precede validation windows")
    if slot_duration_ms <= 0 or slots_per_window <= 0:
        raise ValueError("slot duration and slot count must be positive")

    required = ["packet_ordinal", "timestamp_ns", "window_id", *KEY_FIELDS]
    schema = pq.read_schema(path)
    for name in required:
        if (
            name not in schema.names
            or schema.field(name).type != TRACE_SCHEMA.field(name).type
        ):
            raise ValueError(f"invalid prepared trace field: {name}")
    ids = sorted(set(training_windows + validation_windows))
    table = pq.read_table(path, columns=required, filters=[("window_id", "in", ids)])
    if table.num_rows == 0 or any(table[name].null_count for name in required):
        raise ValueError("selected trace windows must contain non-null packets")
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in required}
    ordinals = columns["packet_ordinal"]
    if np.any(ordinals[1:] <= ordinals[:-1]):
        raise ValueError("packet ordinals must be strictly increasing")
    timestamps = columns["timestamp_ns"]
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("trace timestamps must be nondecreasing")
    windows = columns["window_id"]
    if np.any(windows[1:] < windows[:-1]):
        raise ValueError("trace window ids must be nondecreasing")
    if set(ids) != set(int(value) for value in np.unique(windows)):
        raise ValueError("one or more requested windows are empty or missing")
    packets = list(zip(*(columns[name].tolist() for name in KEY_FIELDS)))
    for key in packets:
        if key[4] not in {6, 17} and (key[2] or key[3]):
            raise ValueError("non-TCP/UDP packets must have zero normalized ports")

    training_set = set(training_windows)
    counts = Counter(
        key for key, window in zip(packets, windows) if window in training_set
    )
    sorted_keys = sorted(counts)
    keys = np.array(sorted_keys, dtype=FLOW_KEY_DTYPE)
    values = np.asarray([counts[key] for key in sorted_keys], dtype=np.uint64)
    slots = []
    for window in validation_windows:
        indices = np.flatnonzero(windows == window)
        bins = (timestamps[indices] - timestamps[indices[0]]) // (
            slot_duration_ms * 1_000_000
        )
        populated = np.unique(bins)
        # Reuse the campaign's evenly spaced non-empty slot selection.
        chosen = populated[
            np.unique(
                np.linspace(
                    0, len(populated) - 1, min(slots_per_window, len(populated))
                ).astype(int)
            )
        ]
        for slot in chosen:
            positions = indices[bins == slot]
            if len(positions) >= 2**32:
                raise ValueError("a validation slot exceeds the 32-bit packet counter")
            slots.append(
                {
                    "window": window,
                    "slot": int(slot),
                    "packets": [packets[index] for index in positions],
                }
            )
    report = {
        "training_windows": training_windows,
        "validation_windows": validation_windows,
        "training_flow_count": len(keys),
        "training_packet_count": int(values.sum()),
        "validation_slot_count": len(slots),
        "validation_packet_count": sum(len(slot["packets"]) for slot in slots),
        "slot_duration_ms": slot_duration_ms,
        "packet_order_preserved": True,
    }
    return keys, values, slots, report
