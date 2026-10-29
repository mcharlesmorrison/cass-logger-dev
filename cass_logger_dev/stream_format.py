"""
Block stream file format: parsing and decoding.

Firmware variants with FIFO IMUs (``i2c_320g``, ``i2c_320g_2``, and from 0.13
``i2c_45686``, ``i2c_45686_2``) write one file of tagged blocks instead of
back-to-back datrec structs.
Every block is a 5-byte header and a payload, all little-endian and packed::

    [0]     sync     0xA5
    [1]     type     0x01 FILE_HEADER, 0x02 SOURCE_INFO, 0x10 RECORD, 0x20 FIFO
    [2]     source   0 = the logger itself, else a FIFO source id
    [3..4]  len      uint16, payload bytes after the header

The file starts with one FILE_HEADER and one SOURCE_INFO per FIFO source,
then interleaves RECORD blocks (one datrec per 500 Hz tick) and FIFO blocks
(every FIFO word an IMU produced, each with its own time). All times are
``filemicros`` (µs), the same clock as the datrec ``tmicros``.

Format version 2 added each FIFO block's read time and the T_FIT entry flag
(sample times from a fit of the IMU's own clock, for IMUs with no data-ready
pin). Version 1 files still parse.

This module does no serial I/O, so it can be tested without hardware. Full
spec: ``docs/stream-format.md`` in cass-logger-fw.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .firmware_structs import COLUMN_ORDERS, FIRMWARE_DTYPES

SYNC = 0xA5
MAGIC = b"CASS"
FORMAT_VERSIONS = (1, 2)

BLOCK_FILE_HEADER = 0x01
BLOCK_SOURCE_INFO = 0x02
BLOCK_RECORD = 0x10
BLOCK_FIFO = 0x20
KNOWN_BLOCKS = {BLOCK_FILE_HEADER, BLOCK_SOURCE_INFO, BLOCK_RECORD, BLOCK_FIFO}

HEADER_LEN = 5
FILE_HEADER = struct.Struct("<4sH16sHHB")  # magic, format_version, fw_version, record_rate_hz, record_len, n_sources

# FIFO block header: flags, entry count, and (v2) the read time.
FIFO_HEADER_LEN = {1: 2, 2: 6}

# FIFO block flags: events since the source's previous block.
FIFO_BLOCK_FLAGS = {
    0x01: "overrun",      # the IMU FIFO overran: oldest words lost
    0x02: "read_failed",  # a FIFO read failed on the bus: words may be lost
    0x04: "late",         # a read was still running a tick later and was aborted
    0x08: "sync_lost",    # sample timing lost (INT1 pairing or clock fit): times are estimates
    0x10: "resynced",     # sample timing re-established
}

# FIFO entry flags. With neither set, t_us is an estimate (the time of the read).
FIFO_ENTRY_T_EDGE = 0x01  # t_us is the INT1 edge of the sample's ODR instant
FIFO_ENTRY_T_FIT = 0x02   # t_us is from a converged fit of the IMU's sample clock

FIFO_ENTRY_DTYPE = np.dtype(
    [("t_us", "<u4"), ("tag", "u1"), ("flags", "u1"), ("data", "u1", (6,))]
)
"""One FIFO entry: filemicros time, FIFO tag byte, FIFO_ENTRY_* flags, 6 raw data bytes."""

# Device types (DEVICE_* in cass-logger-fw variantConfig.h)
DEVICE_ICM20948 = 1
DEVICE_ICM45686 = 2
DEVICE_ICM45686_MAG = 3
DEVICE_LSM6DSV320X = 4
DEVICE_NAMES = {
    DEVICE_ICM20948: "ICM-20948",
    DEVICE_ICM45686: "ICM-45686",
    DEVICE_ICM45686_MAG: "ICM-45686 + QMC6309",
    DEVICE_LSM6DSV320X: "LSM6DSV320X + QMC6309",
}

# Sensor codes (tag >> 3), the same for every source (the LSM6DSV320X's own FIFO tags).
TAG_GYRO = 0x01
TAG_ACCEL = 0x02       # the LSM6DSV320X's low-g accelerometer
TAG_TIMESTAMP = 0x04
TAG_MAG = 0x0E
TAG_MAG_NACK = 0x19
TAG_ACCEL_HG = 0x1D

# SOURCE_INFO for an IMU: device_type, port, i2c_addr, has_mag, odr_hz, then
# hg / accel / gyro / mag sensitivities (0 for a sensor the IMU doesn't have).
IMU_SOURCE_INFO = struct.Struct("<BcBB5f")


class StreamFormatError(ValueError):
    """The file isn't a block stream file, or its header is unusable."""


@dataclass
class SourceInfo:
    """One FIFO source, from its SOURCE_INFO block."""

    source_id: int
    device_type: int
    port: str
    i2c_addr: int
    has_mag: bool
    odr_hz: float
    hg_mg_per_lsb: float     # 0 if the IMU has no high-g accelerometer
    accel_mg_per_lsb: float
    gyro_mdps_per_lsb: float
    mag_mG_per_lsb: float

    @property
    def device(self) -> str:
        return DEVICE_NAMES.get(self.device_type, f"device {self.device_type}")


@dataclass
class StreamFile:
    """A parsed block stream file, still in raw units.

    ``records`` are the raw datrec structs. ``fifo`` maps source id to its raw
    FIFO entries (``FIFO_ENTRY_DTYPE``) in file order. ``blocks`` maps source id
    to one ``(record_index, flags, read_us, count)`` per FIFO block: the record
    index is how many RECORD blocks came before it, and read_us (format 2+, else
    None) the filemicros time of its FIFO read.
    """

    fw_version: str
    format_version: int
    record_rate_hz: int
    record_len: int
    sources: dict[int, SourceInfo]
    records: np.ndarray
    fifo: dict[int, np.ndarray]
    blocks: dict[int, list[tuple[int, int, Optional[int], int]]]
    truncated: bool = False   # the last block ran past the end of the file
    skipped_bytes: int = 0    # bytes dropped while resyncing after corruption


@dataclass
class ImuData:
    """Decoded data from one FIFO IMU, in physical units.

    Every frame has ``tmicros`` (µs) and ``t`` (s) on the same zero as the
    records DataFrame, and two flags for how good the time is: ``t_edge``, the
    sample's INT1 edge (LSM6DSV320X), and ``t_fit``, from a converged fit of the
    IMU's sample clock (ICM-45686, which has no INT pin wired). With neither,
    the time is an estimate (timing lost; see ``events``). Accelerometers are in
    mg, the gyro in dps and the magnetometer in µT, like the other variants.
    """

    port: str
    device: str
    accel: pd.DataFrame       # accelerometer (LSM6DSV320X: low-g, ±16 g): x, y, z [mg]
    accel_hg: pd.DataFrame    # high-g accelerometer (LSM6DSV320X, ±320 g): x, y, z [mg]; else empty
    gyro: pd.DataFrame        # x, y, z [dps]
    mag: pd.DataFrame         # x, y, z [µT]
    timestamp: pd.DataFrame   # raw IMU timestamp counter: ticks (LSM ~22.2 µs, ICM 1 µs)
    events: pd.DataFrame      # tmicros, t, and one bool column per FIFO_BLOCK_FLAGS name
    reads: pd.DataFrame       # one row per FIFO block: tmicros, t of its read (format 2+), entries
    mag_nacks: int            # LSM6DSV320X sensor hub cycles where the QMC6309 didn't answer
    unknown_words: int        # FIFO words with an unexpected tag


@dataclass
class StreamData:
    """Everything in a block stream file, decoded.

    ``records`` is the same DataFrame ``CassCommands.process_data_file``
    returns. ``imu`` maps port letter ("C", "E") to that IMU's data.
    """

    fw_version: str
    records: pd.DataFrame
    imu: dict[str, ImuData] = field(default_factory=dict)
    truncated: bool = False
    skipped_bytes: int = 0


def is_stream_file(path: Union[str, Path]) -> bool:
    """True if the file starts with a block stream FILE_HEADER."""
    with open(path, "rb") as f:
        head = f.read(HEADER_LEN + len(MAGIC))
    return _is_stream_header(head)


def _is_stream_header(head: bytes) -> bool:
    return (
        len(head) >= HEADER_LEN + len(MAGIC)
        and head[0] == SYNC
        and head[1] == BLOCK_FILE_HEADER
        and head[HEADER_LEN : HEADER_LEN + len(MAGIC)] == MAGIC
    )


def _record_dtype(fw_version: str, record_len: Optional[int] = None) -> tuple[np.dtype, list[str]]:
    # Imported here to avoid a circular import (cass_commands imports this module).
    from .cass_commands import CassCommands

    key = CassCommands.get_dtype_key(fw_version, record_len)
    return FIRMWARE_DTYPES[key]().newbyteorder("<"), COLUMN_ORDERS[key]


def _block_ok(buf: bytes, i: int, btype: int, length: int, record_len: int, fifo_hdr: int) -> bool:
    """Checks a block's length against what its type allows."""
    if btype == BLOCK_RECORD:
        return length == record_len
    if btype == BLOCK_FIFO:
        return length >= fifo_hdr and length == fifo_hdr + FIFO_ENTRY_DTYPE.itemsize * buf[i + HEADER_LEN + 1]
    if btype == BLOCK_FILE_HEADER:
        return length == FILE_HEADER.size
    return length >= 1  # SOURCE_INFO: device-specific


def read_stream(source: Union[str, Path, bytes]) -> StreamFile:
    """Parse a block stream file (path or bytes) into raw arrays.

    A block that runs past the end of the file (power loss, or the download
    skipping a partial last buffer) ends the parse with ``truncated`` set.
    Bytes that don't start a valid block are skipped until the next one, and
    counted in ``skipped_bytes``.

    Raises
    ------
    StreamFormatError
        If the data doesn't start with a FILE_HEADER, or its format version or
        record layout isn't one this parser knows.
    """
    buf = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    buf = bytes(buf)
    mv = memoryview(buf)
    n = len(buf)
    if not _is_stream_header(buf):
        raise StreamFormatError("Not a block stream file (no FILE_HEADER at the start)")

    magic, fmt, fw, rate, record_len, _n_sources = FILE_HEADER.unpack_from(buf, HEADER_LEN)
    if fmt not in FORMAT_VERSIONS:
        raise StreamFormatError(f"Unsupported stream format version {fmt} (parser knows {FORMAT_VERSIONS})")
    fifo_hdr = FIFO_HEADER_LEN[fmt]
    fw_version = fw.split(b"\0", 1)[0].decode("ascii", errors="replace")
    try:
        rec_dtype, _ = _record_dtype(fw_version, record_len)
    except ValueError as exc:
        raise StreamFormatError(str(exc)) from exc
    if rec_dtype.itemsize != record_len:
        raise StreamFormatError(
            f"Record length mismatch: file says {record_len} bytes, firmware '{fw_version}' "
            f"layout is {rec_dtype.itemsize}"
        )

    sources: dict[int, SourceInfo] = {}
    record_slices: list[memoryview] = []
    fifo_slices: dict[int, list[memoryview]] = {}
    blocks: dict[int, list[tuple[int, int, Optional[int], int]]] = {}
    truncated = False
    skipped = 0
    entry_len = FIFO_ENTRY_DTYPE.itemsize

    # A block that runs past the end is either the cut-off last block or a
    # corrupt header. It only counts as truncation if no valid block follows;
    # this holds the skip count from that point, to restore if so.
    past_end_skipped: Optional[int] = None

    i = 0
    while i + HEADER_LEN <= n:
        btype = buf[i + 1]
        length = buf[i + 3] | (buf[i + 4] << 8)
        end = i + HEADER_LEN + length
        if buf[i] == SYNC and btype in KNOWN_BLOCKS:
            if end > n:
                if past_end_skipped is None:
                    past_end_skipped = skipped
            elif _block_ok(buf, i, btype, length, record_len, fifo_hdr):
                past_end_skipped = None
                src = buf[i + 2]
                p = i + HEADER_LEN
                if btype == BLOCK_RECORD:
                    record_slices.append(mv[p:end])
                elif btype == BLOCK_FIFO:
                    flags, count = buf[p], buf[p + 1]
                    read_us = struct.unpack_from("<I", buf, p + 2)[0] if fmt >= 2 else None
                    if count:
                        first = p + fifo_hdr
                        fifo_slices.setdefault(src, []).append(mv[first : first + count * entry_len])
                    blocks.setdefault(src, []).append((len(record_slices), flags, read_us, count))
                elif btype == BLOCK_SOURCE_INFO and buf[p] in DEVICE_NAMES and length >= IMU_SOURCE_INFO.size:
                    dev, port, addr, has_mag, odr, hg, acc, gy, mag = IMU_SOURCE_INFO.unpack_from(buf, p)
                    sources[src] = SourceInfo(src, dev, port.decode("ascii"), addr, bool(has_mag),
                                              odr, hg, acc, gy, mag)
                i = end
                continue
        # Not a valid block here: resync at the next sync byte.
        nxt = buf.find(bytes([SYNC]), i + 1)
        if nxt < 0:
            skipped += n - i
            i = n
            break
        skipped += nxt - i
        i = nxt
    if past_end_skipped is not None:
        truncated = True
        skipped = past_end_skipped
    elif 0 < n - i < HEADER_LEN:
        truncated = True

    records = np.frombuffer(b"".join(record_slices), dtype=rec_dtype)
    fifo = {
        src: np.frombuffer(b"".join(slices), dtype=FIFO_ENTRY_DTYPE)
        for src, slices in fifo_slices.items()
    }
    return StreamFile(fw_version, fmt, rate, record_len, sources, records, fifo, blocks,
                      truncated, skipped)


def unwrap_u32(values: np.ndarray) -> np.ndarray:
    """Unwrap a uint32 µs counter (wraps every ~71.6 min) into int64.

    Steps between neighbours are taken as signed 32-bit differences, so values
    may go slightly backwards (FIFO entries do) but never jump by half a wrap.
    """
    v = np.asarray(values).astype(np.int64) & 0xFFFFFFFF
    if v.size == 0:
        return v
    steps = ((np.diff(v) + 2**31) % 2**32) - 2**31
    return np.concatenate(([v[0]], v[0] + np.cumsum(steps)))


def _time_columns(t_us: np.ndarray, t0: int) -> dict[str, np.ndarray]:
    tmicros = unwrap_u32(t_us) - t0
    return {"tmicros": tmicros, "t": tmicros * 1e-6}


def _time_flags(entries: np.ndarray) -> dict[str, np.ndarray]:
    return {"t_edge": (entries["flags"] & FIFO_ENTRY_T_EDGE) != 0,
            "t_fit": (entries["flags"] & FIFO_ENTRY_T_FIT) != 0}


def _xyz_frame(entries: np.ndarray, scale: float, t0: int) -> pd.DataFrame:
    xyz = entries["data"].copy().view("<i2").reshape(-1, 3).astype(np.float64) * scale
    cols = _time_columns(entries["t_us"], t0)
    cols.update(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], **_time_flags(entries))
    return pd.DataFrame(cols)


def decode_imu(entries: np.ndarray, info: SourceInfo, t0: int,
               blocks: list[tuple[int, int, Optional[int], int]], record_tmicros: np.ndarray) -> ImuData:
    """Decode one IMU source's raw FIFO entries into physical units.

    ``t0`` is the raw (unwrapped) tmicros that becomes zero, and
    ``record_tmicros`` the zero-referenced record times, used to place events
    in format 1 files (which don't record read times).
    """
    sensor = entries["tag"] >> 3
    pick = lambda code: entries[sensor == code]  # noqa: E731

    ts = pick(TAG_TIMESTAMP)
    ts_cols = _time_columns(ts["t_us"], t0)
    ts_cols["ticks"] = ts["data"][:, :4].copy().view("<u4").ravel()
    ts_cols.update(_time_flags(ts))

    # Each block's time: its read time (format 2+), else the record before it
    # (within one 2 ms tick).
    if blocks and blocks[0][2] is not None:
        block_t = unwrap_u32(np.array([b[2] for b in blocks], dtype=np.int64)) - t0
    elif blocks and len(record_tmicros):
        idx = np.clip(np.array([b[0] for b in blocks]) - 1, 0, len(record_tmicros) - 1)
        block_t = record_tmicros[idx]
    else:
        block_t = np.zeros(len(blocks), dtype=np.int64)
    block_flags = np.array([b[1] for b in blocks], dtype=np.int64)

    has_flags = block_flags != 0
    ev = {"tmicros": block_t[has_flags], "t": block_t[has_flags] * 1e-6}
    for bit, name in FIFO_BLOCK_FLAGS.items():
        ev[name] = (block_flags[has_flags] & bit) != 0

    reads = pd.DataFrame({"tmicros": block_t, "t": block_t * 1e-6,
                          "entries": np.array([b[3] for b in blocks], dtype=np.int64)})
    if not blocks or blocks[0][2] is None:
        reads = reads.iloc[0:0]  # format 1: no read times

    known = {TAG_GYRO, TAG_ACCEL, TAG_TIMESTAMP, TAG_MAG, TAG_MAG_NACK, TAG_ACCEL_HG}
    return ImuData(
        port=info.port,
        device=info.device,
        accel=_xyz_frame(pick(TAG_ACCEL), info.accel_mg_per_lsb, t0),
        accel_hg=_xyz_frame(pick(TAG_ACCEL_HG), info.hg_mg_per_lsb, t0),
        gyro=_xyz_frame(pick(TAG_GYRO), info.gyro_mdps_per_lsb / 1000.0, t0),
        mag=_xyz_frame(pick(TAG_MAG), info.mag_mG_per_lsb * 0.1, t0),  # 1 mG = 0.1 µT
        timestamp=pd.DataFrame(ts_cols),
        events=pd.DataFrame(ev),
        reads=reads,
        mag_nacks=int(np.count_nonzero(sensor == TAG_MAG_NACK)),
        unknown_words=int(np.count_nonzero(~np.isin(sensor, list(known)))),
    )


def records_frame(stream: StreamFile) -> tuple[pd.DataFrame, int]:
    """The RECORD blocks as a DataFrame, plus the raw tmicros used as zero.

    Same columns and ``t`` as ``CassCommands.process_data_file`` gives for the
    legacy format, but with tmicros properly unwrapped across the 32-bit wrap.
    """
    _, column_order = _record_dtype(stream.fw_version, stream.record_len)
    df = pd.DataFrame(stream.records)
    tmicros = unwrap_u32(stream.records["tmicros"])
    t0 = int(tmicros[0]) if len(tmicros) else 0
    df["tmicros"] = tmicros - t0
    df.insert(1, "t", df["tmicros"] * 1e-6)
    df = df[[col for col in column_order if col in df.columns]]
    return df, t0


def process_stream(source: Union[str, Path, bytes, StreamFile]) -> StreamData:
    """Parse and decode a block stream file into DataFrames.

    All times (records and every IMU frame) share one zero: the first record.
    """
    stream = source if isinstance(source, StreamFile) else read_stream(source)
    records, t0 = records_frame(stream)
    record_t = records["tmicros"].to_numpy()

    data = StreamData(stream.fw_version, records, truncated=stream.truncated,
                      skipped_bytes=stream.skipped_bytes)
    for src, info in stream.sources.items():
        entries = stream.fifo.get(src, np.zeros(0, dtype=FIFO_ENTRY_DTYPE))
        data.imu[info.port] = decode_imu(entries, info, t0, stream.blocks.get(src, []), record_t)
    return data


# ── Files from before firmware 0.13 ───────────────────────────────────────────
# Back-to-back datrec structs with the IMUs polled into each record (500 Hz).
# legacy_stream_data() presents them the same way as a stream file, so code
# written for process_stream_file() reads both.

# Record column suffix per port in the legacy layouts.
_LEGACY_IMU_SUFFIXES = {"S": [""], "C": ["_i2c", "_i2c_c"], "E": ["_i2c_e"]}


def _empty_frame(*cols: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in ("tmicros", "t") + cols})


def legacy_stream_data(records: pd.DataFrame, fw_version: str) -> StreamData:
    """A legacy (pre-0.13) file's records as a StreamData.

    ``records`` is the DataFrame from ``CassCommands.process_data_file`` and is
    kept as-is (IMU columns included). Each IMU in it also gets an ``ImuData``
    built from its record columns: one sample per record (500 Hz), timed by the
    record (``t_edge`` and ``t_fit`` False), with no timestamp, event or read
    tables.
    """
    variant = fw_version.split("-", 1)[1] if "-" in fw_version else "std"
    data = StreamData(fw_version, records)
    for port, suffixes in _LEGACY_IMU_SUFFIXES.items():
        s = next((s for s in suffixes if f"gx{s}" in records.columns), None)
        if s is None:
            continue
        device = "ICM-45686 + QMC6309" if port != "S" and "45686" in variant else "ICM-20948"

        def frame(prefix: str) -> pd.DataFrame:
            f = pd.DataFrame({"tmicros": records["tmicros"], "t": records["t"]})
            for axis in "xyz":
                f[axis] = records[f"{prefix}{axis}{s}"].astype(np.float64)
            f["t_edge"] = False
            f["t_fit"] = False
            return f

        data.imu[port] = ImuData(
            port=port,
            device=device,
            accel=frame("g"),
            accel_hg=_empty_frame("x", "y", "z", "t_edge", "t_fit"),
            gyro=frame("w"),
            mag=frame("T"),
            timestamp=_empty_frame("ticks", "t_edge", "t_fit"),
            events=_empty_frame(*FIFO_BLOCK_FLAGS.values()),
            reads=_empty_frame("entries"),
            mag_nacks=0,
            unknown_words=0,
        )
    return data
