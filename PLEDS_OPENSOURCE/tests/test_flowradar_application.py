from __future__ import annotations

from collections import Counter

from pleds.applications.flowradar import (
    FlowRadarCore,
    PackedPartitionedFlowRadar,
    PartitionedFlowRadar,
    TieredFlowRadar,
    decode_flow_key,
    encode_flow_key,
)


def index_fn(key, hash_id, width):
    return (7 * int(key[0]) + 11 * hash_id) % width


def flow(value):
    return value, value + 1, 1000 + value, 2000 + value, 6


def test_flow_key_encoding_round_trip():
    key = (0x0A000001, 0xC0A80101, 12345, 443, 6)
    assert decode_flow_key(encode_flow_key(key)) == key


def test_flowradar_decodes_flow_ids_and_packet_counts():
    packets = [flow(1)] * 5 + [flow(2)] * 3 + [flow(3)]
    core = FlowRadarCore(
        flow_filter_bits=512,
        flow_filter_hashes=4,
        counting_cells=64,
        counting_hashes=4,
        index_fn=index_fn,
    )
    core.update_many(packets)
    result = core.decode()

    assert result.recovered_counts == dict(Counter(packets))
    assert result.residual_flow_cells == 0
    assert result.residual_packet_cells == 0
    assert result.trustworthy_counts


def test_flow_filter_uses_disjoint_banks_with_the_declared_total_budget():
    core = FlowRadarCore(
        flow_filter_bits=512,
        flow_filter_hashes=4,
        counting_cells=64,
        counting_hashes=4,
        index_fn=index_fn,
    )
    assert core.flow_filter_bank_bits == 128
    assert len(core.flow_filter) == 4
    assert sum(len(bank) for bank in core.flow_filter) == 512


def test_flow_filter_requires_equal_hash_banks():
    import pytest

    with pytest.raises(ValueError, match="divide evenly"):
        FlowRadarCore(
            flow_filter_bits=510,
            flow_filter_hashes=4,
            counting_cells=64,
            counting_hashes=4,
            index_fn=index_fn,
        )


def test_partitioned_flowradar_updates_and_decodes_one_partition_per_flow():
    packets = [flow(1)] * 2 + [flow(2)] * 4 + [flow(3)] * 3 + [flow(4)]
    partitioned = PartitionedFlowRadar(
        flow_filter_bits=[256, 256],
        counting_cells=[32, 32],
        selector=lambda key: key[0] % 2,
        index_fn=index_fn,
    )
    partitioned.update_many(packets)
    result = partitioned.decode()

    assert result.recovered_counts == dict(Counter(packets))
    assert sum(partition.packet_count for partition in partitioned.partitions) == len(
        packets
    )
    assert all(
        key in partitioned.partitions[partitioned.partition_for(key)]._observed_flows
        for key in set(packets)
    )


def test_packed_register_layout_matches_independent_partitions():
    packets = [flow(1)] * 2 + [flow(2)] * 4 + [flow(3)] * 3 + [flow(4)] + [flow(5)] * 2
    selector = lambda key: key[0] % 2
    independent = PartitionedFlowRadar(
        flow_filter_bits=[256, 256],
        counting_cells=[32, 32],
        selector=selector,
        flow_filter_hashes=2,
        counting_hashes=2,
        index_fn=index_fn,
    )
    packed = PackedPartitionedFlowRadar(
        partition_count=2,
        flow_filter_bank_bits=128,
        counting_row_cells=16,
        selector=selector,
        flow_filter_hashes=2,
        counting_hashes=2,
        index_fn=index_fn,
    )
    independent.update_many(packets)
    packed.update_many(packets)
    assert packed.decode() == independent.decode()


def test_tiered_flowradar_merges_exact_and_fallback_records():
    packets = [flow(1)] * 5 + [flow(2)] * 3 + [flow(3)] * 2 + [flow(4)]
    tiered = TieredFlowRadar(
        exact_bank_entries=[1, 1],
        selector=lambda key: key[0] <= 3,
        fallback_flow_filter_bits=512,
        fallback_counting_cells=64,
        flow_filter_hashes=2,
        counting_hashes=2,
        exact_hash_start=6,
        index_fn=lambda key, hash_id, width: (
            0 if hash_id >= 6 else (7 * int(key[0]) + 11 * hash_id) % width
        ),
    )

    tiered.update_many(packets)
    result = tiered.decode()

    assert result.recovered_counts == dict(Counter(packets))
    assert tiered.exact_overflow_packets == 2
    assert result.trustworthy_counts


def test_tiered_flowradar_fingerprint_mode_matches_p4_alias_semantics():
    first = flow(1)
    alias = flow(5)
    tiered = TieredFlowRadar(
        exact_bank_entries=[1],
        selector=lambda _key: True,
        fallback_flow_filter_bits=64,
        fallback_counting_cells=16,
        exact_fingerprint_bits=2,
        exact_hash_start=6,
        index_fn=lambda key, _hash_id, width: int(key[0]) % width,
    )

    tiered.update_many([first, alias])
    result = tiered.decode()

    assert result.recovered_counts == {first: 2}
    assert tiered.exact_fingerprint_alias_packets == 1
    assert tiered.exact_overflow_packets == 0
