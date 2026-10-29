"""
Live view snapshot protocol: frame parsing and payload decoding.

The host sends ``'k'`` on the command port and the device (firmware >= 0.12)
replies on the data port with one snapshot frame::

    [0..1]   sync         0xA5 0x5A
    [2]      frame_type   0x01 = SNAPSHOT (0.12), 0x02 = SNAPSHOT_IMU (0.13+)
    [3]      flags        bit 0 LOGGING, bit 1 SENSOR_INIT_FAILED
    [4..5]   payload_len  uint16 LE; payload bytes, or 0 while recording
    [6..]    payload      SNAPSHOT: raw datrec bytes
                          SNAPSHOT_IMU: uint16 record_len, raw datrec, uint8 n_imu,
                          n_imu × 52-byte IMU readings (see LIVE_IMU)
    [6+N..]  crc16        uint16 LE, CRC-16/CCITT-FALSE over bytes [2, 6+N)

From firmware 0.13 the IMUs aren't in the datrec (they're read through their
FIFOs), so SNAPSHOT_IMU frames carry each FIFO IMU's current reading, in
physical units. They're decoded into the same column names the datrec used:
``gx``/``wx``/``Tx`` for the internal IMU, ``gx_i2c_c``... / ``gx_i2c_e``... for
ports C and E, plus ``hgx_i2c_c``... for the LSM6DSV320X's high-g accelerometer.

This module does no serial I/O, so it can be tested without hardware. The
request/reply call lives in ``CassCommands.snapshot()``. Full spec:
``docs/live-view-plan.md``.
"""

from __future__ import annotations

import binascii
import re
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .firmware_structs import FIRMWARE_DTYPES

SYNC = b"\xA5\x5A"
FRAME_SNAPSHOT = 0x01
FRAME_SNAPSHOT_IMU = 0x02
SNAPSHOT_FRAMES = (FRAME_SNAPSHOT, FRAME_SNAPSHOT_IMU)
FLAG_LOGGING = 0x01
FLAG_SENSOR_INIT_FAILED = 0x02
MAX_PAYLOAD = 512
MIN_LIVE_FW = (0, 12)

HEADER_LEN = 6
CRC_LEN = 2

# One IMU reading in a SNAPSHOT_IMU frame (StreamLiveImu in the firmware):
# source_id, device_type, port, valid flags, then accel [mg], gyro [dps],
# mag [µT], high-g accel [mg], 3 floats each.
LIVE_IMU = struct.Struct("<BBcB12f")
LIVE_IMU_ACCEL = 0x01
LIVE_IMU_GYRO = 0x02
LIVE_IMU_MAG = 0x04
LIVE_IMU_HG = 0x08

# Column suffix per port, as in the record layouts the IMUs used to be in.
IMU_SUFFIX = {"S": "", "C": "_i2c_c", "E": "_i2c_e"}

# FIFO IMUs per variant from firmware 0.13, as (port, has high-g): every
# variant has the internal one; the deprecated I2C ICM-20948 is in the record.
LIVE_FIFO_IMUS = {
    "std":          [("S", False)],
    "std_DEF-dig":  [("S", False)],
    "i2c_1":        [("S", False)],
    "i2c_2":        [("S", False)],
    "i2c_45686":    [("S", False), ("C", False)],
    "i2c_45686_2":  [("S", False), ("C", False), ("E", False)],
    "i2c_320g":     [("S", False), ("C", True)],
    "i2c_320g_2":   [("S", False), ("C", True), ("E", True)],
    "i2c_320x":     [("S", False), ("C", True)],
    "i2c_320":      [("S", False), ("C", True)],
    "i2c_320x_2":   [("S", False), ("C", True), ("E", True)],
}


def imu_columns(port: str, has_hg: bool = False) -> list[str]:
    """Column names for one FIFO IMU's live reading."""
    s = IMU_SUFFIX[port]
    cols = [f"{n}{a}{s}" for n in ("g", "w", "T") for a in "xyz"]
    if has_hg:
        cols += [f"hg{a}{s}" for a in "xyz"]
    return cols


class ProtocolError(ValueError):
    """A frame passed its CRC but breaks the protocol rules."""


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no final XOR)."""
    return binascii.crc_hqx(data, 0xFFFF)


@dataclass
class RawFrame:
    """One frame that passed sync and CRC checks, not yet decoded."""

    frame_type: int
    flags: int
    payload: bytes


@dataclass
class Snapshot:
    """One decoded live view reading.

    ``values`` maps datrec field names to raw values (ADC counts, IMU units as
    written by the firmware), in struct order. It is ``None`` while the device
    is recording.
    """

    host_time: float
    logging: bool
    sensor_init_failed: bool
    values: Optional[dict[str, int | float]] = None


@dataclass
class ParserStats:
    """Running error counts, kept across ``FrameParser.reset()``."""

    frames: int = 0
    crc_errors: int = 0
    false_syncs: int = 0
    skipped_bytes: int = 0


class FrameParser:
    """Incremental snapshot frame parser.

    Feed it bytes as they arrive; it returns every complete, CRC-valid frame.
    Bytes before a sync (e.g. stray debug text) are discarded. A bad CRC or an
    oversized ``payload_len`` is treated as a false sync: one byte is dropped
    and the buffer is rescanned.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pending: list[RawFrame] = []
        self.stats = ParserStats()

    def reset(self) -> None:
        """Drop any buffered bytes and pending frames. Stats are kept."""
        self._buf.clear()
        self._pending.clear()

    def feed(self, chunk: bytes) -> list[RawFrame]:
        """Add bytes and return the frames completed by them.

        Raises
        ------
        ProtocolError
            If a CRC-valid frame has ``payload_len == 0`` without the LOGGING
            flag. The bad frame is consumed; frames already parsed in this call
            are returned by the next ``feed()``.
        """
        self._buf += chunk
        frames, self._pending = self._pending, []
        buf = self._buf

        while True:
            idx = buf.find(SYNC)
            if idx < 0:
                # Keep a trailing 0xA5: it may be the first half of a sync.
                keep = 1 if buf and buf[-1] == SYNC[0] else 0
                self.stats.skipped_bytes += len(buf) - keep
                del buf[: len(buf) - keep]
                break
            if idx > 0:
                self.stats.skipped_bytes += idx
                del buf[:idx]

            if len(buf) < HEADER_LEN:
                break

            frame_type = buf[2]
            flags = buf[3]
            payload_len = buf[4] | (buf[5] << 8)

            if payload_len > MAX_PAYLOAD:
                self.stats.false_syncs += 1
                self._drop_one()
                continue

            total = HEADER_LEN + payload_len + CRC_LEN
            if len(buf) < total:
                break

            end = HEADER_LEN + payload_len
            crc_rx = buf[end] | (buf[end + 1] << 8)
            if crc16(bytes(buf[2:end])) != crc_rx:
                self.stats.crc_errors += 1
                self._drop_one()
                continue

            payload = bytes(buf[HEADER_LEN:end])
            del buf[:total]

            if payload_len == 0 and not flags & FLAG_LOGGING:
                self._pending = frames
                raise ProtocolError(
                    "Snapshot frame has no payload but the LOGGING flag is not set"
                )

            self.stats.frames += 1
            frames.append(RawFrame(frame_type, flags, payload))

        return frames

    def _drop_one(self) -> None:
        self.stats.skipped_bytes += 1
        del self._buf[:1]


def _live_dtype(fw_ver: str, record_len: Optional[int] = None) -> np.dtype:
    # Imported here to avoid a circular import (cass_commands imports this module).
    from .cass_commands import CassCommands

    dt = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver, record_len)]()
    return dt.newbyteorder("<")


def _variant(fw_ver: str) -> tuple[tuple[int, ...], str]:
    match = re.match(r"\s*(\d+)\.(\d+)(?:-(.+))?$", fw_ver or "")
    if not match:
        raise ValueError(f"Invalid firmware version format: '{fw_ver}'")
    return (int(match.group(1)), int(match.group(2))), (match.group(3) or "std").strip()


def _fifo_imus(fw_ver: str) -> list[tuple[str, bool]]:
    """The variant's FIFO IMUs (none before 0.13: they were in the record)."""
    version, variant = _variant(fw_ver)
    if version < (0, 13):
        return []
    return LIVE_FIFO_IMUS.get(variant, [("S", False)])


def _record_values(record: bytes, dt: np.dtype) -> dict[str, int | float]:
    rec = np.frombuffer(record, dtype=dt)[0]
    return {name: rec[name].item() for name in dt.names if not name.startswith("_")}


def decode_imu_payload(payload: bytes, fw_ver: str) -> dict[str, int | float]:
    """Decode a SNAPSHOT_IMU payload into ``{column: value}``: the datrec's
    fields, then every FIFO IMU's reading under ``live_columns()`` names.

    A reading the device couldn't take (e.g. no magnetometer) is NaN; an IMU
    missing from the frame leaves its columns NaN too.

    Raises
    ------
    ValueError
        If the payload is malformed, or its record length doesn't match a
        layout of ``fw_ver``'s variant.
    """
    if len(payload) < 3:
        raise ValueError("SNAPSHOT_IMU payload too short")
    record_len = struct.unpack_from("<H", payload, 0)[0]
    end = 2 + record_len
    if len(payload) < end + 1:
        raise ValueError("SNAPSHOT_IMU payload shorter than its record")
    n_imu = payload[end]
    if len(payload) != end + 1 + n_imu * LIVE_IMU.size:
        raise ValueError(f"SNAPSHOT_IMU payload is {len(payload)} bytes; {n_imu} IMU readings don't fit")
    try:
        dt = _live_dtype(fw_ver, record_len)
    except ValueError as exc:
        raise ValueError(f"Variant mismatch: {exc}. Check the device's firmware variant.") from exc
    values = _record_values(payload[2:end], dt)
    for port, has_hg in _fifo_imus(fw_ver):
        values.update({c: float("nan") for c in imu_columns(port, has_hg)})
    for i in range(n_imu):
        _, _, port, valid, *f = LIVE_IMU.unpack_from(payload, end + 1 + i * LIVE_IMU.size)
        port = port.decode("ascii", errors="replace")
        if port not in IMU_SUFFIX:
            continue
        s = IMU_SUFFIX[port]
        groups = [("g", LIVE_IMU_ACCEL, f[0:3]), ("w", LIVE_IMU_GYRO, f[3:6]),
                  ("T", LIVE_IMU_MAG, f[6:9]), ("hg", LIVE_IMU_HG, f[9:12])]
        for name, bit, xyz in groups:
            if valid & bit:
                for axis, v in zip("xyz", xyz):
                    values[f"{name}{axis}{s}"] = v
    return values


def decode_payload(payload: bytes, fw_ver: str) -> dict[str, int | float]:
    """Decode a raw datrec payload into ``{field: value}`` in struct order.

    Padding fields (names starting with ``_``) are left out.

    Raises
    ------
    ValueError
        If the payload size doesn't match the datrec size for ``fw_ver``
        (the device is running a different variant than expected), or if
        ``fw_ver`` is not a known variant.
    """
    dt = _live_dtype(fw_ver)
    if dt.itemsize != len(payload):
        raise ValueError(
            f"Variant mismatch: snapshot payload is {len(payload)} bytes but "
            f"firmware '{fw_ver}' expects {dt.itemsize}. Check the device's "
            "firmware variant."
        )
    rec = np.frombuffer(payload, dtype=dt)[0]
    return {name: rec[name].item() for name in dt.names if not name.startswith("_")}


def decode_frame(frame: RawFrame, fw_ver: str, host_time: Optional[float] = None) -> Snapshot:
    """Turn a SNAPSHOT ``RawFrame`` into a ``Snapshot``."""
    logging = bool(frame.flags & FLAG_LOGGING)
    if logging:
        values = None
    elif frame.frame_type == FRAME_SNAPSHOT_IMU:
        values = decode_imu_payload(frame.payload, fw_ver)
    else:
        values = decode_payload(frame.payload, fw_ver)
    return Snapshot(
        host_time=time.monotonic() if host_time is None else host_time,
        logging=logging,
        sensor_init_failed=bool(frame.flags & FLAG_SENSOR_INIT_FAILED),
        values=values,
    )


def live_columns(fw_ver: str) -> list[str]:
    """Column names a snapshot for ``fw_ver`` will contain: the datrec's fields
    in struct order, then (0.13+) each FIFO IMU's (see ``imu_columns``)."""
    cols = [n for n in _live_dtype(fw_ver).names if not n.startswith("_")]
    for port, has_hg in _fifo_imus(fw_ver):
        cols += imu_columns(port, has_hg)
    return cols


def supports_live_view(fw_ver: str) -> bool:
    """True if ``fw_ver`` (e.g. ``"0.12-std"``) supports the ``'k'`` snapshot command."""
    match = re.match(r"\s*(\d+)\.(\d+)", fw_ver or "")
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= MIN_LIVE_FW
