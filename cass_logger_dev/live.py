"""
Live view snapshot protocol: frame parsing and payload decoding.

The host sends ``'k'`` on the command port and the device (firmware >= 0.12)
replies on the data port with one snapshot frame::

    [0..1]   sync         0xA5 0x5A
    [2]      frame_type   0x01 = SNAPSHOT
    [3]      flags        bit 0 LOGGING, bit 1 SENSOR_INIT_FAILED
    [4..5]   payload_len  uint16 LE; sizeof(datrec), or 0 while recording
    [6..]    payload      raw datrec bytes
    [6+N..]  crc16        uint16 LE, CRC-16/CCITT-FALSE over bytes [2, 6+N)

This module does no serial I/O, so it can be tested without hardware. The
request/reply call lives in ``CassCommands.snapshot()``. Full spec:
``docs/live-view-plan.md``.
"""

from __future__ import annotations

import binascii
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .firmware_structs import FIRMWARE_DTYPES

SYNC = b"\xA5\x5A"
FRAME_SNAPSHOT = 0x01
FLAG_LOGGING = 0x01
FLAG_SENSOR_INIT_FAILED = 0x02
MAX_PAYLOAD = 512
MIN_LIVE_FW = (0, 12)

HEADER_LEN = 6
CRC_LEN = 2


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


def _live_dtype(fw_ver: str) -> np.dtype:
    # Imported here to avoid a circular import (cass_commands imports this module).
    from .cass_commands import CassCommands

    dt = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]()
    return dt.newbyteorder("<")


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
    return Snapshot(
        host_time=time.monotonic() if host_time is None else host_time,
        logging=logging,
        sensor_init_failed=bool(frame.flags & FLAG_SENSOR_INIT_FAILED),
        values=None if logging else decode_payload(frame.payload, fw_ver),
    )


def live_columns(fw_ver: str) -> list[str]:
    """Field names a snapshot for ``fw_ver`` will contain, in struct order."""
    return [n for n in _live_dtype(fw_ver).names if not n.startswith("_")]


def supports_live_view(fw_ver: str) -> bool:
    """True if ``fw_ver`` (e.g. ``"0.12-std"``) supports the ``'k'`` snapshot command."""
    match = re.match(r"\s*(\d+)\.(\d+)", fw_ver or "")
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= MIN_LIVE_FW
