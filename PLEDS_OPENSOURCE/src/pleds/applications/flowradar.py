"""Packet-wise FlowRadar core and partitioned reference semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from pleds.indexing import IndexFunction, target_index


FlowKey = Tuple[int, int, int, int, int]
PartitionSelector = Callable[[FlowKey], int]
TierSelector = Callable[[FlowKey], bool]


def encode_flow_key(key: FlowKey) -> int:
    src, dst, sport, dport, protocol = (int(value) for value in key)
    if not 0 <= src < 1 << 32 or not 0 <= dst < 1 << 32:
        raise ValueError("IPv4 fields must fit in 32 bits")
    if not 0 <= sport < 1 << 16 or not 0 <= dport < 1 << 16:
        raise ValueError("port fields must fit in 16 bits")
    if not 0 <= protocol < 1 << 8:
        raise ValueError("protocol must fit in 8 bits")
    return (((((src << 32) | dst) << 16 | sport) << 16 | dport) << 8) | protocol


def decode_flow_key(value: int) -> FlowKey:
    protocol = value & 0xFF
    value >>= 8
    dport = value & 0xFFFF
    value >>= 16
    sport = value & 0xFFFF
    value >>= 16
    dst = value & 0xFFFFFFFF
    value >>= 32
    src = value & 0xFFFFFFFF
    return src, dst, sport, dport, protocol


@dataclass
class FlowRadarCell:
    flow_xor: int = 0
    flow_count: int = 0
    packet_count: int = 0


@dataclass(frozen=True)
class FlowRadarDecodeResult:
    recovered_counts: dict[FlowKey, int]
    residual_flow_cells: int
    residual_packet_cells: int
    trustworthy_counts: bool


class FlowRadarCore:
    """FlowRadar's flow-filter and invertible counting-table core."""

    def __init__(
        self,
        *,
        flow_filter_bits: int,
        flow_filter_hashes: int,
        counting_cells: int,
        counting_hashes: int = 4,
        index_fn: IndexFunction = target_index,
    ) -> None:
        if flow_filter_bits <= 0 or counting_cells < counting_hashes:
            raise ValueError("FlowRadar structures must be non-empty")
        if flow_filter_hashes <= 0 or counting_hashes <= 0:
            raise ValueError("FlowRadar hash counts must be positive")
        if flow_filter_bits % flow_filter_hashes:
            raise ValueError("flow-filter bits must divide evenly among hash banks")
        if flow_filter_hashes + counting_hashes > 8:
            raise ValueError("the validated P4 hash profile exposes eight hashes")
        if counting_cells % counting_hashes:
            raise ValueError("counting cells must divide evenly among hash rows")
        self.flow_filter_bits = flow_filter_bits
        self.flow_filter_hashes = flow_filter_hashes
        self.flow_filter_bank_bits = flow_filter_bits // flow_filter_hashes
        self.counting_cells = counting_cells
        self.counting_hashes = counting_hashes
        self.cells_per_row = counting_cells // counting_hashes
        self.index_fn = index_fn
        self.flow_filter = [
            bytearray(self.flow_filter_bank_bits) for _ in range(flow_filter_hashes)
        ]
        self.cells = [FlowRadarCell() for _ in range(counting_cells)]
        self.packet_count = 0
        self.new_flow_count = 0
        self.flow_filter_false_positive_count = 0
        self._observed_flows: set[FlowKey] = set()

    def _filter_indexes(self, key: FlowKey) -> tuple[int, ...]:
        return tuple(
            self.index_fn(
                key,
                self.counting_hashes + index,
                self.flow_filter_bank_bits,
            )
            for index in range(self.flow_filter_hashes)
        )

    def _counting_indexes(self, key: FlowKey) -> tuple[int, ...]:
        return tuple(
            row * self.cells_per_row + self.index_fn(key, row, self.cells_per_row)
            for row in range(self.counting_hashes)
        )

    def update(self, key: FlowKey) -> None:
        self.packet_count += 1
        filter_indexes = self._filter_indexes(key)
        filter_hit = all(
            self.flow_filter[bank][index] for bank, index in enumerate(filter_indexes)
        )
        first_observation = key not in self._observed_flows
        if first_observation:
            self._observed_flows.add(key)
            if filter_hit:
                self.flow_filter_false_positive_count += 1
        new_flow = not filter_hit
        if new_flow:
            self.new_flow_count += 1
            for bank, index in enumerate(filter_indexes):
                self.flow_filter[bank][index] = 1
        encoded = encode_flow_key(key)
        for index in self._counting_indexes(key):
            cell = self.cells[index]
            if new_flow:
                cell.flow_xor ^= encoded
                cell.flow_count += 1
            cell.packet_count += 1

    def update_many(self, packets: Sequence[FlowKey]) -> None:
        for key in packets:
            self.update(key)

    def decode(self) -> FlowRadarDecodeResult:
        cells = [
            FlowRadarCell(cell.flow_xor, cell.flow_count, cell.packet_count)
            for cell in self.cells
        ]
        queue = deque(index for index, cell in enumerate(cells) if cell.flow_count == 1)
        recovered: dict[FlowKey, int] = {}
        while queue:
            pure_index = queue.popleft()
            pure = cells[pure_index]
            if pure.flow_count != 1:
                continue
            key = decode_flow_key(pure.flow_xor)
            if key in recovered:
                continue
            indexes = self._counting_indexes(key)
            if pure_index not in indexes:
                continue
            packet_count = pure.packet_count
            if any(
                cells[index].flow_count <= 0 or cells[index].packet_count < packet_count
                for index in indexes
            ):
                continue
            recovered[key] = packet_count
            encoded = encode_flow_key(key)
            for index in indexes:
                cell = cells[index]
                cell.flow_xor ^= encoded
                cell.flow_count -= 1
                cell.packet_count -= packet_count
                if cell.flow_count == 1:
                    queue.append(index)
        residual_flow_cells = sum(
            cell.flow_count != 0 or cell.flow_xor != 0 for cell in cells
        )
        residual_packet_cells = sum(cell.packet_count != 0 for cell in cells)
        return FlowRadarDecodeResult(
            recovered_counts=recovered,
            residual_flow_cells=residual_flow_cells,
            residual_packet_cells=residual_packet_cells,
            trustworthy_counts=residual_packet_cells == 0,
        )


class PartitionedFlowRadar:
    """Route each flow to one independent FlowRadar core."""

    def __init__(
        self,
        *,
        flow_filter_bits: Sequence[int],
        counting_cells: Sequence[int],
        selector: PartitionSelector,
        flow_filter_hashes: int = 4,
        counting_hashes: int = 4,
        index_fn: IndexFunction = target_index,
    ) -> None:
        if len(flow_filter_bits) != len(counting_cells) or not flow_filter_bits:
            raise ValueError("partition dimensions must have equal non-zero length")
        self.selector = selector
        self.partitions = [
            FlowRadarCore(
                flow_filter_bits=int(filter_bits),
                flow_filter_hashes=flow_filter_hashes,
                counting_cells=int(cell_count),
                counting_hashes=counting_hashes,
                index_fn=index_fn,
            )
            for filter_bits, cell_count in zip(flow_filter_bits, counting_cells)
        ]

    def partition_for(self, key: FlowKey) -> int:
        partition = int(self.selector(key))
        if not 0 <= partition < len(self.partitions):
            raise ValueError(f"partition id out of range: {partition}")
        return partition

    def update(self, key: FlowKey) -> None:
        self.partitions[self.partition_for(key)].update(key)

    def update_many(self, packets: Sequence[FlowKey]) -> None:
        for key in packets:
            self.update(key)

    def decode(self) -> FlowRadarDecodeResult:
        recovered: dict[FlowKey, int] = {}
        residual_flow_cells = 0
        residual_packet_cells = 0
        trustworthy = True
        for partition in self.partitions:
            result = partition.decode()
            overlap = recovered.keys() & result.recovered_counts.keys()
            if overlap:
                raise AssertionError("a flow was decoded from multiple partitions")
            recovered.update(result.recovered_counts)
            residual_flow_cells += result.residual_flow_cells
            residual_packet_cells += result.residual_packet_cells
            trustworthy = trustworthy and result.trustworthy_counts
        return FlowRadarDecodeResult(
            recovered_counts=recovered,
            residual_flow_cells=residual_flow_cells,
            residual_packet_cells=residual_packet_cells,
            trustworthy_counts=trustworthy,
        )


class TieredFlowRadar:
    """Store selected flows exactly and send all remaining packets to FlowRadar."""

    def __init__(
        self,
        *,
        exact_bank_entries: Sequence[int],
        selector: TierSelector,
        fallback_flow_filter_bits: int,
        fallback_counting_cells: int,
        flow_filter_hashes: int = 2,
        counting_hashes: int = 2,
        exact_hash_start: int = 6,
        exact_fingerprint_bits: int | None = None,
        exact_fingerprint_hash_start: int | None = None,
        index_fn: IndexFunction = target_index,
    ) -> None:
        if not exact_bank_entries or any(
            int(value) <= 0 for value in exact_bank_entries
        ):
            raise ValueError("exact tier banks must be non-empty")
        if exact_hash_start < 0:
            raise ValueError("exact hash start must be non-negative")
        if exact_hash_start + len(exact_bank_entries) > 8:
            raise ValueError("the validated P4 hash profile exposes eight hashes")
        if exact_fingerprint_bits is not None and not 1 <= exact_fingerprint_bits <= 31:
            raise ValueError("exact_fingerprint_bits must be in [1, 31]")
        resolved_fingerprint_hash_start = (
            exact_hash_start
            if exact_fingerprint_hash_start is None
            else exact_fingerprint_hash_start
        )
        if resolved_fingerprint_hash_start < 0:
            raise ValueError("exact fingerprint hash start must be non-negative")
        if (
            exact_fingerprint_bits is not None
            and resolved_fingerprint_hash_start + len(exact_bank_entries) > 8
        ):
            raise ValueError("the validated P4 hash profile exposes eight hashes")
        self.selector = selector
        self.index_fn = index_fn
        self.exact_hash_start = exact_hash_start
        self.exact_fingerprint_bits = exact_fingerprint_bits
        self.exact_fingerprint_hash_start = resolved_fingerprint_hash_start
        self.exact_keys: list[list[FlowKey | None]] = [
            [None] * int(entries) for entries in exact_bank_entries
        ]
        self.exact_tags: list[list[int]] = [
            [0] * int(entries) for entries in exact_bank_entries
        ]
        self.exact_counts: list[list[int]] = [
            [0] * int(entries) for entries in exact_bank_entries
        ]
        self.exact_overflow_packets = 0
        self.exact_fingerprint_alias_packets = 0
        self.fallback = FlowRadarCore(
            flow_filter_bits=fallback_flow_filter_bits,
            flow_filter_hashes=flow_filter_hashes,
            counting_cells=fallback_counting_cells,
            counting_hashes=counting_hashes,
            index_fn=index_fn,
        )

    def _update_exact(self, key: FlowKey) -> bool:
        for bank, keys in enumerate(self.exact_keys):
            index = self.index_fn(key, self.exact_hash_start + bank, len(keys))
            stored = keys[index]
            tag = 0
            if self.exact_fingerprint_bits is not None:
                tag = (1 << self.exact_fingerprint_bits) | self.index_fn(
                    key,
                    self.exact_fingerprint_hash_start + bank,
                    1 << self.exact_fingerprint_bits,
                )
            if stored is None:
                keys[index] = key
                self.exact_tags[bank][index] = tag
                self.exact_counts[bank][index] = 1
                return True
            same_record = (
                stored == key
                if self.exact_fingerprint_bits is None
                else self.exact_tags[bank][index] == tag
            )
            if same_record:
                if self.exact_fingerprint_bits is not None and stored != key:
                    self.exact_fingerprint_alias_packets += 1
                self.exact_counts[bank][index] += 1
                return True
        return False

    def update(self, key: FlowKey) -> None:
        selected = self.selector(key)
        if selected and self._update_exact(key):
            return
        if selected:
            self.exact_overflow_packets += 1
        self.fallback.update(key)

    def update_many(self, packets: Sequence[FlowKey]) -> None:
        for key in packets:
            self.update(key)

    def decode(self) -> FlowRadarDecodeResult:
        fallback = self.fallback.decode()
        recovered = dict(fallback.recovered_counts)
        for keys, counts in zip(self.exact_keys, self.exact_counts):
            for key, count in zip(keys, counts):
                if key is None:
                    continue
                if key in recovered:
                    raise AssertionError("a flow was recorded in both FlowRadar tiers")
                recovered[key] = count
        return FlowRadarDecodeResult(
            recovered_counts=recovered,
            residual_flow_cells=fallback.residual_flow_cells,
            residual_packet_cells=fallback.residual_packet_cells,
            trustworthy_counts=fallback.trustworthy_counts,
        )


class PackedPartitionedFlowRadar:
    """Reference the packed-register layout emitted by the Tofino backend."""

    def __init__(
        self,
        *,
        partition_count: int,
        flow_filter_bank_bits: int,
        counting_row_cells: int,
        selector: PartitionSelector,
        flow_filter_hashes: int = 2,
        counting_hashes: int = 2,
        index_fn: IndexFunction = target_index,
    ) -> None:
        if partition_count <= 0:
            raise ValueError("partition count must be positive")
        if flow_filter_bank_bits <= 0 or counting_row_cells <= 0:
            raise ValueError("packed FlowRadar dimensions must be positive")
        self.partition_count = partition_count
        self.flow_filter_bank_bits = flow_filter_bank_bits
        self.counting_row_cells = counting_row_cells
        self.selector = selector
        self.flow_filter_hashes = flow_filter_hashes
        self.counting_hashes = counting_hashes
        self.index_fn = index_fn
        self.flow_filter = [
            bytearray(partition_count * flow_filter_bank_bits)
            for _ in range(flow_filter_hashes)
        ]
        self.cells = [
            FlowRadarCell()
            for _ in range(counting_hashes * partition_count * counting_row_cells)
        ]

    def partition_for(self, key: FlowKey) -> int:
        partition = int(self.selector(key))
        if not 0 <= partition < self.partition_count:
            raise ValueError(f"partition id out of range: {partition}")
        return partition

    def _filter_indexes(self, key: FlowKey, partition: int) -> tuple[int, ...]:
        base = partition * self.flow_filter_bank_bits
        return tuple(
            base
            + self.index_fn(
                key,
                self.counting_hashes + bank,
                self.flow_filter_bank_bits,
            )
            for bank in range(self.flow_filter_hashes)
        )

    def _counting_indexes(self, key: FlowKey, partition: int) -> tuple[int, ...]:
        row_span = self.partition_count * self.counting_row_cells
        partition_base = partition * self.counting_row_cells
        return tuple(
            row * row_span
            + partition_base
            + self.index_fn(key, row, self.counting_row_cells)
            for row in range(self.counting_hashes)
        )

    def update(self, key: FlowKey) -> None:
        partition = self.partition_for(key)
        filter_indexes = self._filter_indexes(key, partition)
        new_flow = any(
            self.flow_filter[bank][index] == 0
            for bank, index in enumerate(filter_indexes)
        )
        for bank, index in enumerate(filter_indexes):
            self.flow_filter[bank][index] = 1
        encoded = encode_flow_key(key)
        for index in self._counting_indexes(key, partition):
            cell = self.cells[index]
            if new_flow:
                cell.flow_xor ^= encoded
                cell.flow_count += 1
            cell.packet_count += 1

    def update_many(self, packets: Sequence[FlowKey]) -> None:
        for key in packets:
            self.update(key)

    def decode(self) -> FlowRadarDecodeResult:
        recovered: dict[FlowKey, int] = {}
        residual_flow_cells = 0
        residual_packet_cells = 0
        trustworthy = True
        row_span = self.partition_count * self.counting_row_cells
        for partition in range(self.partition_count):
            core = FlowRadarCore(
                flow_filter_bits=self.flow_filter_hashes * self.flow_filter_bank_bits,
                flow_filter_hashes=self.flow_filter_hashes,
                counting_cells=self.counting_hashes * self.counting_row_cells,
                counting_hashes=self.counting_hashes,
                index_fn=self.index_fn,
            )
            for bank in range(self.flow_filter_hashes):
                start = partition * self.flow_filter_bank_bits
                end = start + self.flow_filter_bank_bits
                core.flow_filter[bank][:] = self.flow_filter[bank][start:end]
            for row in range(self.counting_hashes):
                source = row * row_span + partition * self.counting_row_cells
                target = row * self.counting_row_cells
                for offset in range(self.counting_row_cells):
                    cell = self.cells[source + offset]
                    core.cells[target + offset] = FlowRadarCell(
                        cell.flow_xor, cell.flow_count, cell.packet_count
                    )
            result = core.decode()
            overlap = recovered.keys() & result.recovered_counts.keys()
            if overlap:
                raise AssertionError("a flow was decoded from multiple partitions")
            recovered.update(result.recovered_counts)
            residual_flow_cells += result.residual_flow_cells
            residual_packet_cells += result.residual_packet_cells
            trustworthy = trustworthy and result.trustworthy_counts
        return FlowRadarDecodeResult(
            recovered_counts=recovered,
            residual_flow_cells=residual_flow_cells,
            residual_packet_cells=residual_packet_cells,
            trustworthy_counts=trustworthy,
        )
