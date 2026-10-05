"""Shared, hardware-equivalent hash specifications for PLEDS backends."""

from __future__ import annotations

import hashlib
import ipaddress
import struct
from dataclasses import dataclass
from functools import lru_cache

from pleds.key_bits import normalize_flow_key


DEFAULT_HASH_PROFILE = "tofino_custom_crc32_permuted_ipv4_5tuple_v2"
STANDARD_HASH_FIELDS = (
    "ipv4.src_addr",
    "ipv4.dst_addr",
    "ipv4.protocol",
    "l4.src_port",
    "l4.dst_port",
)
DEFAULT_HASH_FIELDS = STANDARD_HASH_FIELDS


@dataclass(frozen=True)
class Crc32Profile:
    name: str
    polynomial: int
    reversed: bool = True
    msb: bool = False
    extended: bool = False
    init: int = 0xFFFFFFFF
    xor_out: int = 0xFFFFFFFF

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "polynomial": f"0x{self.polynomial:08X}",
            "reversed": self.reversed,
            "msb": self.msb,
            "extended": self.extended,
            "init": f"0x{self.init:08X}",
            "xor_out": f"0x{self.xor_out:08X}",
        }


CRC32_IEEE = Crc32Profile("crc32_ieee", 0x04C11DB7)
CRC32C = Crc32Profile("crc32c_castagnoli", 0x1EDC6F41)


@dataclass(frozen=True)
class HashVariant:
    profile: Crc32Profile
    fields: tuple[str, ...]


# The target SDE contains examples for both polynomials. Field permutations
# create distinct hash inputs without introducing synthetic salt bytes.
HASH_VARIANTS = (
    HashVariant(CRC32_IEEE, STANDARD_HASH_FIELDS),
    HashVariant(
        CRC32C,
        (
            "ipv4.dst_addr",
            "ipv4.src_addr",
            "ipv4.protocol",
            "l4.dst_port",
            "l4.src_port",
        ),
    ),
    HashVariant(
        CRC32_IEEE,
        (
            "l4.src_port",
            "l4.dst_port",
            "ipv4.src_addr",
            "ipv4.dst_addr",
            "ipv4.protocol",
        ),
    ),
    HashVariant(
        CRC32C,
        (
            "l4.dst_port",
            "l4.src_port",
            "ipv4.dst_addr",
            "ipv4.src_addr",
            "ipv4.protocol",
        ),
    ),
    HashVariant(
        CRC32_IEEE,
        (
            "ipv4.protocol",
            "ipv4.src_addr",
            "l4.src_port",
            "ipv4.dst_addr",
            "l4.dst_port",
        ),
    ),
    HashVariant(
        CRC32C,
        (
            "ipv4.protocol",
            "ipv4.dst_addr",
            "l4.dst_port",
            "ipv4.src_addr",
            "l4.src_port",
        ),
    ),
    HashVariant(
        CRC32_IEEE,
        (
            "ipv4.src_addr",
            "l4.dst_port",
            "ipv4.dst_addr",
            "l4.src_port",
            "ipv4.protocol",
        ),
    ),
    HashVariant(
        CRC32C,
        (
            "ipv4.dst_addr",
            "l4.src_port",
            "ipv4.src_addr",
            "l4.dst_port",
            "ipv4.protocol",
        ),
    ),
)


@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    @classmethod
    def parse(cls, value: str) -> "FlowKey":
        parts = value.split(",")
        if len(parts) != 5:
            raise ValueError(
                f"expected five-tuple key 'src,dst,sport,dport,proto', got: {value!r}"
            )
        src_ip, dst_ip, src_port, dst_port, protocol = parts
        result = cls(src_ip, dst_ip, int(src_port), int(dst_port), int(protocol))
        if not 0 <= result.src_port <= 0xFFFF or not 0 <= result.dst_port <= 0xFFFF:
            raise ValueError(f"port outside bit<16>: {value!r}")
        if not 0 <= result.protocol <= 0xFF:
            raise ValueError(f"protocol outside bit<8>: {value!r}")
        return result

    def field_bytes(self, field: str) -> bytes:
        src_port = self.src_port if self.protocol in (6, 17) else 0
        dst_port = self.dst_port if self.protocol in (6, 17) else 0
        if field == "ipv4.src_addr":
            return struct.pack("!I", int(ipaddress.IPv4Address(self.src_ip)))
        if field == "ipv4.dst_addr":
            return struct.pack("!I", int(ipaddress.IPv4Address(self.dst_ip)))
        if field == "ipv4.protocol":
            return struct.pack("!B", self.protocol)
        if field == "l4.src_port":
            return struct.pack("!H", src_port)
        if field == "l4.dst_port":
            return struct.pack("!H", dst_port)
        raise ValueError(f"unsupported hash field: {field}")

    def pack_fields(self, fields: tuple[str, ...]) -> bytes:
        return b"".join(self.field_bytes(field) for field in fields)

    def pack(self) -> bytes:
        return self.pack_fields(STANDARD_HASH_FIELDS)

    def pack_for_crc32(self, salt: int | None = None) -> bytes:
        return self.pack()


@dataclass(frozen=True)
class HashSpec:
    profile: str
    algorithm: str
    key_fields: tuple[str, ...]
    hash_id: int
    table_size: int
    index_mode: str
    p4_tuple: str
    software_packing: str
    crc: Crc32Profile

    @property
    def salt(self) -> int:
        return self.hash_id

    @property
    def hardware_equivalent(self) -> bool:
        return self.index_mode == "low_bits_power_of_two"

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "algorithm": self.algorithm,
            "key_fields": list(self.key_fields),
            "hash_id": self.hash_id,
            "salt": self.hash_id,
            "table_size": self.table_size,
            "index_mode": self.index_mode,
            "p4_tuple": self.p4_tuple,
            "software_packing": self.software_packing,
            "crc": self.crc.as_dict(),
            "hardware_equivalent": self.hardware_equivalent,
        }


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _variant(hash_id: int) -> HashVariant:
    if hash_id < 0 or hash_id >= len(HASH_VARIANTS):
        raise ValueError(f"hash_id must be in [0, {len(HASH_VARIANTS) - 1}]: {hash_id}")
    return HASH_VARIANTS[hash_id]


def _reflect_polynomial(polynomial: int, width: int = 32) -> int:
    reflected = 0
    for bit in range(width):
        if polynomial & (1 << bit):
            reflected |= 1 << (width - bit - 1)
    return reflected


@lru_cache(maxsize=None)
def _crc32_table(polynomial: int) -> tuple[int, ...]:
    reflected = _reflect_polynomial(polynomial)
    table: list[int] = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ reflected if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


def crc32_value(data: bytes, profile: Crc32Profile) -> int:
    if not profile.reversed or profile.msb or profile.extended:
        raise ValueError(f"unsupported software CRC mode: {profile}")
    table = _crc32_table(profile.polynomial)
    crc = profile.init & 0xFFFFFFFF
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (crc ^ profile.xor_out) & 0xFFFFFFFF


def direct_hash_value(value: str, hash_id: int) -> int:
    variant = _variant(hash_id)
    key = FlowKey.parse(value)
    return crc32_value(key.pack_fields(variant.fields), variant.profile)


def direct_hash_value_fields(key: tuple[int, int, int, int, int], hash_id: int) -> int:
    """Hash an integer five-tuple without string conversion."""

    src_addr, dst_addr, src_port, dst_port, protocol = normalize_flow_key(key)
    if not 0 <= src_addr < 1 << 32 or not 0 <= dst_addr < 1 << 32:
        raise ValueError("IPv4 address outside bit<32>")
    if not 0 <= src_port < 1 << 16 or not 0 <= dst_port < 1 << 16:
        raise ValueError("port outside bit<16>")
    if not 0 <= protocol < 1 << 8:
        raise ValueError("protocol outside bit<8>")
    packed = {
        "ipv4.src_addr": struct.pack("!I", src_addr),
        "ipv4.dst_addr": struct.pack("!I", dst_addr),
        "ipv4.protocol": struct.pack("!B", protocol),
        "l4.src_port": struct.pack("!H", src_port),
        "l4.dst_port": struct.pack("!H", dst_port),
    }
    variant = _variant(hash_id)
    return crc32_value(
        b"".join(packed[field] for field in variant.fields), variant.profile
    )


def direct_hash_values_fields(
    key: tuple[int, int, int, int, int], count: int
) -> tuple[int, ...]:
    if count <= 0 or count > len(HASH_VARIANTS):
        raise ValueError("count must be between 1 and {}".format(len(HASH_VARIANTS)))
    src_addr, dst_addr, src_port, dst_port, protocol = normalize_flow_key(key)
    if not 0 <= src_addr < 1 << 32 or not 0 <= dst_addr < 1 << 32:
        raise ValueError("IPv4 address outside bit<32>")
    if not 0 <= src_port < 1 << 16 or not 0 <= dst_port < 1 << 16:
        raise ValueError("port outside bit<16>")
    if not 0 <= protocol < 1 << 8:
        raise ValueError("protocol outside bit<8>")
    packed = {
        "ipv4.src_addr": struct.pack("!I", src_addr),
        "ipv4.dst_addr": struct.pack("!I", dst_addr),
        "ipv4.protocol": struct.pack("!B", protocol),
        "l4.src_port": struct.pack("!H", src_port),
        "l4.dst_port": struct.pack("!H", dst_port),
    }
    return tuple(
        crc32_value(
            b"".join(packed[field] for field in HASH_VARIANTS[hash_id].fields),
            HASH_VARIANTS[hash_id].profile,
        )
        for hash_id in range(count)
    )


def _legacy_hash_value(value: str, hash_id: int) -> int:
    digest = hashlib.blake2b(
        f"{hash_id}:{value}".encode("utf-8"),
        digest_size=8,
        person=b"PLEDSBF2",
    ).digest()
    return int.from_bytes(digest, "big")


def stable_hash_value(value: str, salt: int) -> int:
    try:
        return direct_hash_value(value, salt)
    except ValueError as exc:
        if "expected five-tuple" not in str(exc):
            raise
        return _legacy_hash_value(value, salt)


def tofino_crc32_hash_value(value: str, salt: int) -> int:
    return direct_hash_value(value, salt)


def p4_hash_tuple(hash_id: int) -> str:
    fields = _variant(hash_id).fields
    expressions = {
        "ipv4.src_addr": "hdr.ipv4.src_addr",
        "ipv4.dst_addr": "hdr.ipv4.dst_addr",
        "ipv4.protocol": "hdr.ipv4.protocol",
        "l4.src_port": "ig_md.l4_sport",
        "l4.dst_port": "ig_md.l4_dport",
    }
    return "{" + ", ".join(expressions[field] for field in fields) + "}"


def p4_crc32_tuple(salt: int = 0) -> str:
    return p4_hash_tuple(salt)


def _p4_bool(value: bool) -> str:
    return "true" if value else "false"


def p4_hash_extern(prefix: str, hash_id: int) -> str:
    variant = _variant(hash_id)
    profile = variant.profile
    return f"""    CRCPolynomial<bit<32>>(32w0x{profile.polynomial:08X},
                               {_p4_bool(profile.reversed)},
                               {_p4_bool(profile.msb)},
                               {_p4_bool(profile.extended)},
                               32w0x{profile.init:08X},
                               32w0x{profile.xor_out:08X}) {prefix}_poly{hash_id};
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, {prefix}_poly{hash_id}) {prefix}_hash{hash_id};"""


def p4_hash_externs(prefix: str, count: int) -> str:
    lines: list[str] = []
    for hash_id in range(count):
        lines.append(p4_hash_extern(prefix, hash_id))
    return "\n".join(lines)


def p4_hash_compute_action(
    prefix: str,
    output_prefix: str,
    hash_id: int,
    *,
    extra_lines: str = "",
) -> str:
    _variant(hash_id)
    extra = "\n" + extra_lines if extra_lines else ""
    return f"""    action compute_{output_prefix}{hash_id}() {{
        ig_md.{output_prefix}{hash_id}_value = {prefix}_hash{hash_id}.get({p4_hash_tuple(hash_id)});{extra}
    }}

    table compute_{output_prefix}{hash_id}_table {{
        actions = {{ compute_{output_prefix}{hash_id}; }}
        const default_action = compute_{output_prefix}{hash_id}();
        size = 1;
    }}"""


def hash_spec_for_salt(*, salt: int, table_size: int) -> HashSpec:
    if table_size <= 0:
        raise ValueError("table_size must be positive")
    variant = _variant(salt)
    index_mode = (
        "low_bits_power_of_two" if is_power_of_two(table_size) else "software_modulo"
    )
    return HashSpec(
        profile=DEFAULT_HASH_PROFILE,
        algorithm="CUSTOM_CRC32_FIELD_PERMUTATION",
        key_fields=variant.fields,
        hash_id=salt,
        table_size=table_size,
        index_mode=index_mode,
        p4_tuple=p4_hash_tuple(salt),
        software_packing="network-order concatenation in key_fields order",
        crc=variant.profile,
    )


def hash_specs(*, count: int, table_size: int) -> list[HashSpec]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(HASH_VARIANTS):
        raise ValueError(
            f"at most {len(HASH_VARIANTS)} hardware hash variants are defined"
        )
    return [
        hash_spec_for_salt(salt=hash_id, table_size=table_size)
        for hash_id in range(count)
    ]


def hash_index(value: str, *, salt: int, table_size: int) -> int:
    if table_size <= 0:
        raise ValueError("table_size must be positive")
    hash_value = stable_hash_value(value, salt)
    if is_power_of_two(table_size):
        return hash_value & (table_size - 1)
    return hash_value % table_size


def hash_indexes(value: str, *, table_size: int, count: int) -> list[int]:
    hash_specs(count=count, table_size=table_size)
    return [
        hash_index(value, salt=hash_id, table_size=table_size)
        for hash_id in range(count)
    ]


def hash_indexes_fields(
    key: tuple[int, int, int, int, int], *, table_size: int, count: int
) -> list[int]:
    hash_specs(count=count, table_size=table_size)
    mask = table_size - 1
    if is_power_of_two(table_size):
        return [
            direct_hash_value_fields(key, hash_id) & mask for hash_id in range(count)
        ]
    return [
        direct_hash_value_fields(key, hash_id) % table_size for hash_id in range(count)
    ]


def register_hash_plan(*, table_size: int, count: int) -> dict[str, object]:
    specs = hash_specs(count=count, table_size=table_size)
    return {
        "profile": DEFAULT_HASH_PROFILE,
        "hash_count": count,
        "unique_crc_polynomials": len({spec.crc.polynomial for spec in specs}),
        "table_size": table_size,
        "specs": [spec.as_dict() for spec in specs],
        "notes": [
            "Software and P4 use target-supported custom CRC32 profiles over the normalized five-tuple.",
            "Logical hashes use distinct field permutations and alternate IEEE CRC32 and CRC32C.",
            "Power-of-two tables use low hash bits; non-power-of-two tables are software-only.",
        ],
    }
