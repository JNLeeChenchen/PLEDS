"""Ordered packet-trace ingestion for FlowRadar workloads."""

from __future__ import annotations

import hashlib
import json
import socket
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple

import dpkt
import pyarrow as pa
import pyarrow.parquet as pq


TRACE_SCHEMA = pa.schema(
    [
        ("packet_ordinal", pa.uint64()),
        ("timestamp_ns", pa.uint64()),
        ("window_id", pa.uint32()),
        ("src_addr", pa.uint32()),
        ("dst_addr", pa.uint32()),
        ("src_port", pa.uint16()),
        ("dst_port", pa.uint16()),
        ("protocol", pa.uint8()),
        ("ipv4_total_len", pa.uint16()),
    ]
)


@dataclass(frozen=True)
class PacketRecord:
    packet_ordinal: int
    timestamp_ns: int
    window_id: int
    src_addr: int
    dst_addr: int
    src_port: int
    dst_port: int
    protocol: int
    ipv4_total_len: int

    def as_tuple(self) -> Tuple[int, ...]:
        return (
            self.packet_ordinal,
            self.timestamp_ns,
            self.window_id,
            self.src_addr,
            self.dst_addr,
            self.src_port,
            self.dst_port,
            self.protocol,
            self.ipv4_total_len,
        )


@dataclass
class TraceCounters:
    source_packets: int = 0
    emitted_ipv4_packets: int = 0
    skipped_non_ipv4: int = 0
    skipped_malformed: int = 0
    skipped_ipv4_fragment: int = 0
    skipped_ipv4_options: int = 0

    def as_dict(self) -> dict:
        return {
            "source_packets": self.source_packets,
            "emitted_ipv4_packets": self.emitted_ipv4_packets,
            "skipped_non_ipv4": self.skipped_non_ipv4,
            "skipped_malformed": self.skipped_malformed,
            "skipped_ipv4_fragment": self.skipped_ipv4_fragment,
            "skipped_ipv4_options": self.skipped_ipv4_options,
        }


def _sampled_file_fingerprint(path: Path, sample_bytes: int = 1 << 20) -> str:
    """Hash file metadata plus its first and last sample without reading all PCAP bytes."""

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _open_capture(path: Path) -> Tuple[BinaryIO, object, int]:
    handle = path.open("rb")
    try:
        try:
            reader = dpkt.pcap.Reader(handle)
            return handle, reader, int(reader.datalink())
        except (ValueError, dpkt.dpkt.NeedData):
            handle.seek(0)
            reader = dpkt.pcapng.Reader(handle)
            return handle, reader, int(reader.datalink())
    except Exception:
        handle.close()
        raise


def _unwrap_ethernet_ipv4(payload: bytes) -> Optional[dpkt.ip.IP]:
    ethernet = dpkt.ethernet.Ethernet(payload)
    body = ethernet.data
    while isinstance(body, dpkt.ethernet.VLANtag8021Q):
        body = body.data
    if isinstance(body, dpkt.ip.IP):
        return body
    return None


def _decode_ipv4(payload: bytes, linktype: int) -> Optional[dpkt.ip.IP]:
    if linktype in {getattr(dpkt.pcap, "DLT_RAW", 12), 101}:
        if not payload or payload[0] >> 4 != 4:
            return None
        packet = dpkt.ip.IP(payload)
        return packet
    if linktype == getattr(dpkt.pcap, "DLT_EN10MB", 1):
        return _unwrap_ethernet_ipv4(payload)
    if linktype == getattr(dpkt.pcap, "DLT_LINUX_SLL", 113):
        cooked = dpkt.sll.SLL(payload)
        return cooked.data if isinstance(cooked.data, dpkt.ip.IP) else None
    raise ValueError("unsupported PCAP link type: {}".format(linktype))


def _ports(ip_packet: dpkt.ip.IP) -> Tuple[int, int]:
    if isinstance(ip_packet.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        return int(ip_packet.data.sport), int(ip_packet.data.dport)
    return 0, 0


def _timestamp_ns(timestamp: object, divisor: object) -> int:
    if isinstance(timestamp, Decimal):
        return int(timestamp * Decimal(1_000_000_000))
    if float(divisor or 0) == 1_000_000:
        # Multiplying the epoch timestamp directly by 1e9 loses the original
        # microsecond resolution in a binary float.
        return int(round(float(timestamp) * 1_000_000)) * 1000
    return int(round(float(timestamp) * 1_000_000_000))


def iter_ipv4_records(
    pcap_path: Path,
    *,
    window_seconds: float,
    max_source_packets: Optional[int] = None,
    counters: Optional[TraceCounters] = None,
) -> Iterator[PacketRecord]:
    """Yield normalized IPv4 packets in the exact order stored in the capture."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if max_source_packets is not None and max_source_packets <= 0:
        raise ValueError("max_source_packets must be positive")

    state = counters if counters is not None else TraceCounters()
    handle, reader, linktype = _open_capture(pcap_path)
    timestamp_divisor = getattr(reader, "_divisor", None)
    first_timestamp_ns: Optional[int] = None
    window_ns = int(round(window_seconds * 1_000_000_000))
    try:
        for ordinal, (timestamp, payload) in enumerate(reader):
            if max_source_packets is not None and ordinal >= max_source_packets:
                break
            state.source_packets += 1
            try:
                ip_packet = _decode_ipv4(payload, linktype)
                if ip_packet is None:
                    state.skipped_non_ipv4 += 1
                    continue
                # Later fragments do not carry the transport header needed by the
                # normalized five-tuple, so exclude all IPv4 fragments explicitly.
                if int(ip_packet.mf) or int(ip_packet.offset):
                    state.skipped_ipv4_fragment += 1
                    continue
                # Generated P4 parsers process only the fixed 20-byte IPv4 header.
                # Keep the offline workload domain identical to the data plane.
                if int(ip_packet.hl) != 5:
                    state.skipped_ipv4_options += 1
                    continue
                timestamp_ns = _timestamp_ns(timestamp, timestamp_divisor)
                if first_timestamp_ns is None:
                    first_timestamp_ns = timestamp_ns
                src_port, dst_port = _ports(ip_packet)
                src_addr = int.from_bytes(ip_packet.src, byteorder="big", signed=False)
                dst_addr = int.from_bytes(ip_packet.dst, byteorder="big", signed=False)
                total_len = int(ip_packet.len)
                if total_len <= 0 or total_len >= 1 << 16:
                    raise ValueError("invalid IPv4 total length")
                state.emitted_ipv4_packets += 1
                yield PacketRecord(
                    packet_ordinal=ordinal,
                    timestamp_ns=timestamp_ns,
                    window_id=(timestamp_ns - first_timestamp_ns) // window_ns,
                    src_addr=src_addr,
                    dst_addr=dst_addr,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=int(ip_packet.p),
                    ipv4_total_len=total_len,
                )
            except (ValueError, IndexError, TypeError, dpkt.dpkt.Error, socket.error):
                state.skipped_malformed += 1
    finally:
        handle.close()


def _write_batch(writer: pq.ParquetWriter, columns: list) -> None:
    arrays = [
        pa.array(column, type=field.type)
        for column, field in zip(columns, TRACE_SCHEMA)
    ]
    writer.write_batch(pa.record_batch(arrays, schema=TRACE_SCHEMA))


def preprocess_pcap(
    pcap_path: Path,
    output_path: Path,
    *,
    window_seconds: float,
    batch_size: int = 250_000,
    max_source_packets: Optional[int] = None,
    compression: str = "zstd",
) -> dict:
    """Convert a PCAP to an ordered Parquet trace and return its manifest."""

    pcap_path = pcap_path.resolve()
    output_path = output_path.resolve()
    if not pcap_path.is_file():
        raise FileNotFoundError(str(pcap_path))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    counters = TraceCounters()
    columns = [[] for _ in TRACE_SCHEMA]
    first_record: Optional[PacketRecord] = None
    last_record: Optional[PacketRecord] = None
    start_monotonic = time.monotonic()
    writer = pq.ParquetWriter(str(output_path), TRACE_SCHEMA, compression=compression)
    try:
        for record in iter_ipv4_records(
            pcap_path,
            window_seconds=window_seconds,
            max_source_packets=max_source_packets,
            counters=counters,
        ):
            if first_record is None:
                first_record = record
            last_record = record
            for column, value in zip(columns, record.as_tuple()):
                column.append(value)
            if len(columns[0]) >= batch_size:
                _write_batch(writer, columns)
                columns = [[] for _ in TRACE_SCHEMA]
        if columns[0]:
            _write_batch(writer, columns)
    finally:
        writer.close()

    stat = pcap_path.stat()
    manifest = {
        "format": "pleds_ordered_ipv4_trace_v1",
        "source": {
            "path": str(pcap_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sampled_sha256": _sampled_file_fingerprint(pcap_path),
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "schema": str(TRACE_SCHEMA),
            "compression": compression,
        },
        "window_seconds": window_seconds,
        "max_source_packets": max_source_packets,
        "counters": counters.as_dict(),
        "first_timestamp_ns": first_record.timestamp_ns if first_record else None,
        "last_timestamp_ns": last_record.timestamp_ns if last_record else None,
        "max_window_id": last_record.window_id if last_record else None,
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "packet_order_preserved": True,
        "fragment_policy": "all IPv4 fragments excluded because later fragments lack L4 ports",
        "ipv4_options_policy": "IPv4 packets with IHL other than 5 excluded to match generated P4 parsers",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
