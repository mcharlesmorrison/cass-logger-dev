"""Tests for the live view frame parser and decoder (no hardware needed)."""

import struct

import numpy as np
import pytest

from cass_logger_dev.cass_commands import CassCommands
from cass_logger_dev.firmware_structs import FIRMWARE_DTYPES
from cass_logger_dev.live import (
    FLAG_LOGGING,
    FLAG_SENSOR_INIT_FAILED,
    FRAME_SNAPSHOT,
    SYNC,
    FrameParser,
    ProtocolError,
    crc16,
    decode_frame,
    decode_payload,
    live_columns,
    supports_live_view,
)


def make_frame(payload: bytes = b"", flags: int = 0, frame_type: int = FRAME_SNAPSHOT) -> bytes:
    """Build a frame exactly as the spec describes."""
    body = bytes([frame_type, flags]) + struct.pack("<H", len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


def make_payload(fw_ver: str, seed: int = 1) -> tuple[bytes, np.ndarray]:
    dt = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]().newbyteorder("<")
    rec = np.zeros(1, dtype=dt)
    for i, name in enumerate(dt.names):
        rec[name] = seed + i
    return rec.tobytes(), rec[0]


STD = "0.12-std"


def test_crc_check_value():
    assert crc16(b"123456789") == 0x29B1


@pytest.mark.parametrize(
    "fw_ver,size",
    [("0.12-std", 76), ("0.12-i2c_1", 108), ("0.12-i2c_45686", 108),
     ("0.12-i2c_2", 136), ("0.12-i2c_45686_2", 136)],
)
def test_payload_sizes_match_firmware(fw_ver, size):
    payload, _ = make_payload(fw_ver)
    assert len(payload) == size


def test_clean_frame():
    payload, _ = make_payload(STD)
    frames = FrameParser().feed(make_frame(payload))
    assert len(frames) == 1
    assert frames[0].frame_type == FRAME_SNAPSHOT
    assert frames[0].flags == 0
    assert frames[0].payload == payload


def test_garbage_before_sync():
    payload, _ = make_payload(STD)
    parser = FrameParser()
    frames = parser.feed(b"didn't get spi data!!!\r\n\xa5\x00junk" + make_frame(payload))
    assert [f.payload for f in frames] == [payload]
    assert parser.stats.skipped_bytes > 0


def test_frame_split_across_feeds():
    payload, _ = make_payload("0.12-i2c_2")
    data = b"noise" + make_frame(payload)
    parser = FrameParser()
    frames = []
    for i in range(len(data)):
        frames += parser.feed(data[i : i + 1])
    assert [f.payload for f in frames] == [payload]


def test_split_inside_sync():
    payload, _ = make_payload(STD)
    data = make_frame(payload)
    parser = FrameParser()
    assert parser.feed(b"text" + data[:1]) == []
    assert [f.payload for f in parser.feed(data[1:])] == [payload]


def test_two_frames_in_one_chunk():
    p1, _ = make_payload(STD, seed=1)
    p2, _ = make_payload(STD, seed=100)
    frames = FrameParser().feed(make_frame(p1) + make_frame(p2))
    assert [f.payload for f in frames] == [p1, p2]


def test_bad_crc_then_good_frame():
    payload, _ = make_payload(STD)
    bad = bytearray(make_frame(payload))
    bad[-1] ^= 0xFF
    parser = FrameParser()
    frames = parser.feed(bytes(bad) + make_frame(payload))
    assert [f.payload for f in frames] == [payload]
    assert parser.stats.crc_errors >= 1


def test_false_sync_inside_payload():
    payload = bytearray(make_payload(STD)[0])
    payload[10:12] = SYNC  # a sync pattern inside the data must not split the frame
    payload[12:14] = b"\x00\x00"  # looks like a zero-length header if mis-parsed
    payload = bytes(payload)
    frames = FrameParser().feed(make_frame(payload) + make_frame(payload))
    assert [f.payload for f in frames] == [payload, payload]


def test_logging_frame():
    parser = FrameParser()
    frames = parser.feed(make_frame(b"", flags=FLAG_LOGGING))
    assert len(frames) == 1
    snap = decode_frame(frames[0], STD)
    assert snap.logging is True
    assert snap.values is None


def test_empty_payload_without_logging_is_protocol_error():
    payload, _ = make_payload(STD)
    parser = FrameParser()
    with pytest.raises(ProtocolError):
        parser.feed(make_frame(payload) + make_frame(b"", flags=0))
    # The good frame parsed before the error isn't lost.
    assert [f.payload for f in parser.feed(b"")] == [payload]


def test_payload_len_over_max_is_false_sync():
    payload, _ = make_payload(STD)
    fake_header = SYNC + bytes([FRAME_SNAPSHOT, 0]) + struct.pack("<H", 600)
    parser = FrameParser()
    frames = parser.feed(fake_header + make_frame(payload))
    assert [f.payload for f in frames] == [payload]
    assert parser.stats.false_syncs == 1


def test_sensor_init_failed_flag():
    payload, _ = make_payload(STD)
    frame = FrameParser().feed(make_frame(payload, flags=FLAG_SENSOR_INIT_FAILED))[0]
    snap = decode_frame(frame, STD)
    assert snap.sensor_init_failed is True
    assert snap.logging is False
    assert snap.values is not None


@pytest.mark.parametrize("fw_ver", ["0.12-std", "0.12-i2c_1", "0.12-i2c_2", "0.12-i2c_45686_2"])
def test_decode_payload_values(fw_ver):
    payload, rec = make_payload(fw_ver)
    values = decode_payload(payload, fw_ver)
    assert list(values) == live_columns(fw_ver)
    assert "_pad" not in values
    for name, value in values.items():
        assert value == rec[name].item()
        assert type(value) in (int, float)


def test_decode_payload_variant_mismatch():
    payload, _ = make_payload("0.12-i2c_2")
    with pytest.raises(ValueError, match="Variant mismatch"):
        decode_payload(payload, "0.12-std")


def test_supports_live_view():
    assert not supports_live_view("0.11-std")
    assert not supports_live_view("0.9-std")
    assert not supports_live_view("")
    assert not supports_live_view("garbage")
    assert supports_live_view("0.12-i2c_2")
    assert supports_live_view("0.13")
    assert supports_live_view("1.0-std")


class FakeSerial:
    """Minimal pyserial stand-in: replies to b'k' with a canned byte string."""

    def __init__(self, reply: bytes = b""):
        self.reply = reply
        self.rx = b""
        self.peer = None
        self.writes = []
        self.resets = 0

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, n):
        out, self.rx = self.rx[:n], self.rx[n:]
        return out

    def write(self, data):
        self.writes.append(data)
        if data == b"k" and self.peer is not None:
            self.peer.rx += self.reply

    def reset_input_buffer(self):
        self.resets += 1
        self.rx = b""


def make_cass(reply: bytes) -> CassCommands:
    cass = CassCommands()
    data, cmd = FakeSerial(), FakeSerial(reply)
    cmd.peer = data
    data.is_open = cmd.is_open = True
    cass._ser_data, cass._ser_command = data, cmd
    return cass


def test_snapshot_round_trip():
    payload, rec = make_payload(STD)
    cass = make_cass(b"stale debug text\r\n" + make_frame(payload))
    snap = cass.snapshot(STD)
    assert cass._ser_command.writes == [b"k"]
    assert snap.values["tmicros"] == rec["tmicros"]


def test_snapshot_timeout_clears_input():
    cass = make_cass(b"")  # old firmware: ignores 'k'
    with pytest.raises(TimeoutError, match="too old"):
        cass.snapshot("0.11-std", timeout=0.05)
    assert cass._ser_data.resets >= 1
