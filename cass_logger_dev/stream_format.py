"""
Block stream file format: parsing and decoding.

Firmware variants with FIFO sensors (``i2c_320g``, ``i2c_320g_2``; firmware
0.13+) write one file of tagged blocks instead of back-to-back datrec structs.
Every block is a 5-byte header and a payload, all little-endian and packed::

    [0]     sync     0xA5
    [1]     type     0x01 FILE_HEADER, 0x02 SOURCE_INFO, 0x10 RECORD, 0x20 FIFO
    [2]     source   0 = the logger itself, else a FIFO source id
    [3..4]  len      uint16, payload bytes after the header

The file starts with one FILE_HEADER and one SOURCE_INFO per FIFO source,
then interleaves RECORD blocks (one datrec per 500 Hz tick) and FIFO blocks
(every FIFO word an IMU produced, each with its own time). All times are
``filemicros`` (µs), the same clock as the datrec ``tmicros``.

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
FORMAT_VERSION = 1

BLOCK_FILE_HEADER = 0x01
BLOCK_SOURCE_INFO = 0x02
BLOCK_RECORD = 0x10
BLOCK_FIFO = 0x20
KNOWN_BLOCKS = {BLOCK_FILE_HEADER, BLOCK_SOURCE_INFO, BLOCK_RECORD, BLOCK_FIFO}

HEADER_LEN = 5
FILE_HEADER = struct.Struct("<4sH16sHHB")  # magic, format_version, fw_version, record_rate_hz, record_len, n_sources

# FIFO block flags: events since the source's previous block.
FIFO_BLOCK_FLAGS = {
    0x01: "overrun",      # the IMU FIFO overran: oldest words lost
    0x02: "read_failed",  # a FIFO read failed on the bus: words may be lost
    0x04: "late",         # a read was still running a tick later and was aborted
    0x08: "sync_lost",    # FIFO samples and INT1 edges stopped matching: times are estimates
    0x10: "resynced",     # pairing re-established: times are from INT1 edges again
}

# FIFO entry flags
FIFO_ENTRY_T_EDGE = 0x01  # t_us is the INT1 edge of the sample's ODR instant (else an estimate)

FIFO_ENTRY_DTYPE = np.dtype(
    [("t_us", "<u4"), ("tag", "u1"), ("flags", "u1"), ("data", "u1", (6,))]
)
"""One FIFO entry: filemicros time, raw FIFO tag byte, FIFO_ENTRY_* flags, 6 raw data bytes."""

# Device types (DEVICE_* in cass-logger-fw variantConfig.h)
DEVICE_LSM6DSV320X = 4

# LSM6DSV320X FIFO tag sensor codes (tag >> 3)
TAG_GYRO = 0x01
TAG_ACCEL_LG = 0x02
TAG_TIMESTAMP = 0x04
TAG_MAG = 0x0E
TAG_MAG_NACK = 0x19
TAG_ACCEL_HG = 0x1D

LSM6DSV320X_SOURCE_INFO = struct.Struct("<BcBB5f")


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
    hg_mg_per_lsb: float
    lg_mg_per_lsb: float
    gyro_mdps_per_lsb: float
    mag_mG_per_lsb: float


@dataclass
class StreamFile:
    """A parsed block stream file, still in raw units.

    ``records`` are the raw datrec structs. ``fifo`` maps source id to its raw
    FIFO entries (``FIFO_ENTRY_DTYPE``) in file order. ``events`` maps source id
    to ``(record_index, flags)`` pairs, one per FIFO block with flags set; the
    record index is how many RECORD blocks came before that block.
    """

    fw_version: str
    format_version: int
    record_rate_hz: int
    record_len: int
    sources: dict[int, SourceInfo]
    records: np.ndarray
    fifo: dict[int, np.ndarray]
    events: dict[int, list[tuple[int, int]]]
    truncated: bool = False   # the last block ran past the end of the file
    skipped_bytes: int = 0    # bytes dropped while resyncing after corruption


@dataclass
class ImuData:
    """Decoded data from one LSM6DSV320X, in physical units.

    Every frame has ``tmicros`` (µs) and ``t`` (s) on the same zero as the
    records DataFrame, and ``t_edge``: True when the time is the sample's INT1
    edge, False when it's an estimate (pairing lost). Accelerometers are in mg,
    the gyro in dps and the magnetometer in µT, like the other variants.
    """

    port: str
    accel_hg: pd.DataFrame    # high-g accelerometer (±320 g): x, y, z [mg]
    accel_lg: pd.DataFrame    # low-g accelerometer (±16 g): x, y, z [mg]
    gyro: pd.DataFrame        # x, y, z [dps]
    mag: pd.DataFrame         # x, y, z [µT]; ~27% repeat the previous reading
    timestamp: pd.DataFrame   # IMU timestamp counter: ticks (~22.2 µs each)
    events: pd.DataFrame      # tmicros, t, and one bool column per FIFO_BLOCK_FLAGS name
    mag_nacks: int            # sensor hub cycles where the QMC6309 didn't answer
    unknown_words: int        # FIFO words with a tag the firmware doesn't batch


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


def _record_dtype(fw_version: str) -> tuple[np.dtype, list[str]]:
    # Imported here to avoid a circular import (cass_commands imports this module).
    from .cass_commands import CassCommands

    key = CassCommands.get_dtype_key(fw_version)
    return FIRMWARE_DTYPES[key]().newbyteorder("<"), COLUMN_ORDERS[key]


def _block_ok(buf: bytes, i: int, btype: int, length: int, record_len: Optional[int]) -> bool:
    """Checks a block's length against what its type allows."""
    if btype == BLOCK_RECORD:
        return record_len is None or length == record_len
    if btype == BLOCK_FIFO:
        return length >= 2 and length == 2 + FIFO_ENTRY_DTYPE.itemsize * buf[i + HEADER_LEN + 1]
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
    if fmt != FORMAT_VERSION:
        raise StreamFormatError(f"Unsupported stream format version {fmt} (parser knows {FORMAT_VERSION})")
    fw_version = fw.split(b"\0", 1)[0].decode("ascii", errors="replace")
    rec_dtype, _ = _record_dtype(fw_version)
    if rec_dtype.itemsize != record_len:
        raise StreamFormatError(
            f"Record length mismatch: file says {record_len} bytes, firmware '{fw_version}' "
            f"layout is {rec_dtype.itemsize}"
        )

    sources: dict[int, SourceInfo] = {}
    record_slices: list[memoryview] = []
    fifo_slices: dict[int, list[memoryview]] = {}
    events: dict[int, list[tuple[int, int]]] = {}
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
            elif _block_ok(buf, i, btype, length, record_len):
                past_end_skipped = None
                src = buf[i + 2]
                p = i + HEADER_LEN
                if btype == BLOCK_RECORD:
                    record_slices.append(mv[p:end])
                elif btype == BLOCK_FIFO:
                    flags, count = buf[p], buf[p + 1]
                    if count:
                        fifo_slices.setdefault(src, []).append(mv[p + 2 : p + 2 + count * entry_len])
                    if flags:
                        events.setdefault(src, []).append((len(record_slices), flags))
                elif btype == BLOCK_SOURCE_INFO and buf[p] == DEVICE_LSM6DSV320X:
                    dev, port, addr, has_mag, odr, hg, lg, gy, mag = LSM6DSV320X_SOURCE_INFO.unpack_from(buf, p)
                    sources[src] = SourceInfo(src, dev, port.decode("ascii"), addr, bool(has_mag),
                                              odr, hg, lg, gy, mag)
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
    return StreamFile(fw_version, fmt, rate, record_len, sources, records, fifo, events,
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


def _xyz_frame(entries: np.ndarray, scale: float, t0: int) -> pd.DataFrame:
    xyz = entries["data"].copy().view("<i2").reshape(-1, 3).astype(np.float64) * scale
    cols = _time_columns(entries["t_us"], t0)
    cols.update(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                t_edge=(entries["flags"] & FIFO_ENTRY_T_EDGE) != 0)
    return pd.DataFrame(cols)


def decode_imu(entries: np.ndarray, info: SourceInfo, t0: int,
               events: list[tuple[int, int]], record_tmicros: np.ndarray) -> ImuData:
    """Decode one LSM6DSV320X source's raw FIFO entries into physical units.

    ``t0`` is the raw (unwrapped) tmicros that becomes zero, and
    ``record_tmicros`` the zero-referenced record times, used to place events.
    """
    sensor = entries["tag"] >> 3
    pick = lambda code: entries[sensor == code]  # noqa: E731

    ts = pick(TAG_TIMESTAMP)
    ts_cols = _time_columns(ts["t_us"], t0)
    ts_cols["ticks"] = ts["data"][:, :4].copy().view("<u4").ravel()
    ts_cols["t_edge"] = (ts["flags"] & FIFO_ENTRY_T_EDGE) != 0

    # An event's time is the record before its block (within one 2 ms tick).
    if events and len(record_tmicros):
        idx = np.clip(np.array([e[0] for e in events]) - 1, 0, len(record_tmicros) - 1)
        ev_t = record_tmicros[idx]
    else:
        ev_t = np.zeros(len(events), dtype=np.int64)
    ev = {"tmicros": ev_t, "t": ev_t * 1e-6}
    for bit, name in FIFO_BLOCK_FLAGS.items():
        ev[name] = np.array([bool(flags & bit) for _, flags in events], dtype=bool)

    known = {TAG_GYRO, TAG_ACCEL_LG, TAG_TIMESTAMP, TAG_MAG, TAG_MAG_NACK, TAG_ACCEL_HG}
    return ImuData(
        port=info.port,
        accel_hg=_xyz_frame(pick(TAG_ACCEL_HG), info.hg_mg_per_lsb, t0),
        accel_lg=_xyz_frame(pick(TAG_ACCEL_LG), info.lg_mg_per_lsb, t0),
        gyro=_xyz_frame(pick(TAG_GYRO), info.gyro_mdps_per_lsb / 1000.0, t0),
        mag=_xyz_frame(pick(TAG_MAG), info.mag_mG_per_lsb * 0.1, t0),  # 1 mG = 0.1 µT
        timestamp=pd.DataFrame(ts_cols),
        events=pd.DataFrame(ev),
        mag_nacks=int(np.count_nonzero(sensor == TAG_MAG_NACK)),
        unknown_words=int(np.count_nonzero(~np.isin(sensor, list(known)))),
    )


def records_frame(stream: StreamFile) -> tuple[pd.DataFrame, int]:
    """The RECORD blocks as a DataFrame, plus the raw tmicros used as zero.

    Same columns and ``t`` as ``CassCommands.process_data_file`` gives for the
    legacy format, but with tmicros properly unwrapped across the 32-bit wrap.
    """
    _, column_order = _record_dtype(stream.fw_version)
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
        if info.device_type != DEVICE_LSM6DSV320X:
            continue
        entries = stream.fifo.get(src, np.zeros(0, dtype=FIFO_ENTRY_DTYPE))
        data.imu[info.port] = decode_imu(entries, info, t0, stream.events.get(src, []), record_t)
    return data
