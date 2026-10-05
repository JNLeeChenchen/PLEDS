from pathlib import Path

import dpkt
import pyarrow.parquet as pq

from pleds.traces import TraceCounters, iter_ipv4_records, preprocess_pcap


def _ipv4_packet(
    src: bytes, dst: bytes, sport: int, dport: int, *, protocol: int = 6
) -> bytes:
    if protocol == 6:
        transport = dpkt.tcp.TCP(sport=sport, dport=dport)
    else:
        transport = dpkt.udp.UDP(sport=sport, dport=dport, ulen=8)
    packet = dpkt.ip.IP(src=src, dst=dst, p=protocol, ttl=64, data=transport)
    packet.len = len(packet)
    return bytes(packet)


def _write_raw_pcap(path: Path) -> None:
    with path.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=101, nano=True)
        writer.writepkt(
            _ipv4_packet(b"\x0a\x00\x00\x01", b"\x0a\x00\x00\x02", 1000, 80), ts=1.0
        )
        writer.writepkt(
            _ipv4_packet(
                b"\x0a\x00\x00\x03", b"\x0a\x00\x00\x04", 53, 2000, protocol=17
            ),
            ts=2.25,
        )
        writer.writepkt(
            _ipv4_packet(b"\x0a\x00\x00\x05", b"\x0a\x00\x00\x06", 2000, 443), ts=3.1
        )
        writer.close()


def _ipv4_packet_with_options() -> bytes:
    packet = dpkt.ip.IP(
        src=b"\x0a\x00\x00\x07",
        dst=b"\x0a\x00\x00\x08",
        p=6,
        ttl=64,
        data=dpkt.tcp.TCP(sport=1234, dport=443),
    )
    packet.opts = b"\x01\x01\x01\x01"
    packet.hl = 6
    packet.len = len(packet)
    return bytes(packet)


def test_iter_ipv4_records_preserves_order_and_windows(tmp_path: Path) -> None:
    pcap = tmp_path / "trace.pcap"
    _write_raw_pcap(pcap)
    counters = TraceCounters()
    rows = list(iter_ipv4_records(pcap, window_seconds=2.0, counters=counters))

    assert [row.packet_ordinal for row in rows] == [0, 1, 2]
    assert [row.window_id for row in rows] == [0, 0, 1]
    assert [row.src_port for row in rows] == [1000, 53, 2000]
    assert counters.source_packets == 3
    assert counters.emitted_ipv4_packets == 3


def test_preprocess_pcap_writes_parquet_and_manifest(tmp_path: Path) -> None:
    pcap = tmp_path / "trace.pcap"
    output = tmp_path / "trace.parquet"
    _write_raw_pcap(pcap)

    manifest = preprocess_pcap(pcap, output, window_seconds=1.0, batch_size=2)
    table = pq.read_table(output)

    assert table.num_rows == 3
    assert table.column("packet_ordinal").to_pylist() == [0, 1, 2]
    assert table.column("window_id").to_pylist() == [0, 1, 2]
    assert manifest["packet_order_preserved"] is True
    assert manifest["counters"]["emitted_ipv4_packets"] == 3
    assert output.with_suffix(".parquet.manifest.json").is_file()


def test_iter_ipv4_records_excludes_ipv4_options_to_match_p4_domain(
    tmp_path: Path,
) -> None:
    pcap = tmp_path / "trace_with_options.pcap"
    with pcap.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=101, nano=True)
        writer.writepkt(_ipv4_packet_with_options(), ts=1.0)
        writer.writepkt(
            _ipv4_packet(b"\x0a\x00\x00\x01", b"\x0a\x00\x00\x02", 1000, 80), ts=2.0
        )
        writer.close()

    counters = TraceCounters()
    rows = list(iter_ipv4_records(pcap, window_seconds=1.0, counters=counters))

    assert len(rows) == 1
    assert counters.skipped_ipv4_options == 1
    assert counters.emitted_ipv4_packets == 1
