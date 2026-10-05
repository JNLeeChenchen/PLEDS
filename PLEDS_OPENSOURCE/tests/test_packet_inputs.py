import numpy as np
import zlib

from pleds.features import PacketFields, extract_compact_packet_features
from pleds.hash_spec import CRC32C, CRC32_IEEE, crc32_value, direct_hash_value_fields
from pleds.indexing import target_index
from pleds.packet_features import deployment_feature_matrix


def test_standard_crc_vectors():
    payload = b"123456789"
    assert crc32_value(payload, CRC32_IEEE) == 0xCBF43926
    assert crc32_value(payload, CRC32_IEEE) == zlib.crc32(payload) & 0xFFFFFFFF
    assert crc32_value(payload, CRC32C) == 0xE3069283


def test_features_match_packet_fields():
    keys = [
        (0xC0000201, 0xC6336402, 12345, 443, 6),
        (0xC6336403, 0xC0000204, 53, 65000, 17),
        (0xC0000205, 0xC6336406, 0, 0, 1),
    ]
    actual = deployment_feature_matrix(keys, feature_count=12)
    expected = [
        extract_compact_packet_features(
            PacketFields(
                src_addr=src,
                dst_addr=dst,
                src_port=sport,
                dst_port=dport,
                protocol=protocol,
                total_len=64,
            )
        )
        for src, dst, sport, dport, protocol in keys
    ]
    np.testing.assert_array_equal(actual, expected)
    bits = deployment_feature_matrix(keys, feature_count=104)
    assert bits.shape == (3, 104)
    expected_bits = [
        list(map(int, f"{src:032b}{dst:032b}{sport:016b}{dport:016b}{proto:08b}"))
        for src, dst, sport, dport, proto in keys
    ]
    np.testing.assert_array_equal(bits, expected_bits)


def test_index_reduction_uses_shared_crc_profile():
    key = (0xC0000201, 0xC6336402, 12345, 443, 6)
    for hash_id in range(8):
        for width in (16, 64, 1024, 65536):
            assert (
                target_index(key, hash_id, width)
                == direct_hash_value_fields(key, hash_id) % width
            )
