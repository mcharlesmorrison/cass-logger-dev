"""Tests for the block stream file parser and decoder (no hardware needed)."""

import struct
from pathlib import Path

import numpy as np
import pytest

from cass_logger_dev.cass_commands import CassCommands
from cass_logger_dev.firmware_structs import FIRMWARE_DTYPES
from cass_logger_dev.stream_format import (
    BLOCK_FIFO,
    BLOCK_FILE_HEADER,
    BLOCK_RECORD,
    BLOCK_SOURCE_INFO,
    DEVICE_LSM6DSV320X,
    FIFO_ENTRY_T_EDGE,
    SYNC,
    TAG_ACCEL_HG,
    TAG_ACCEL_LG,
    TAG_GYRO,
    TAG_MAG,
    TAG_MAG_NACK,
    TAG_TIMESTAMP,
    StreamFormatError,
    is_stream_file,
    process_stream,
    read_stream,
    unwrap_u32,
)

FW = "0.13-i2c_320g"
REAL_LOG = Path(__file__).parent.parent / "examples" / "data" / "lsm6dsv320x-test_i2c_320g" / "LOG_1793206549.bin"


# ── Builders: blocks exactly as docs/stream-format.md describes ──────────────

def block(btype: int, source: int, payload: bytes) -> bytes:
    return bytes([SYNC, btype, source]) + struct.pack("<H", len(payload)) + payload


def file_header(fw: str = FW, n_sources: int = 1) -> bytes:
    rec_len = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw)]().itemsize
    payload = struct.pack("<4sH16sHHB", b"CASS", 1, fw.encode(), 500, rec_len, n_sources)
    return block(BLOCK_FILE_HEADER, 0, payload)


def source_info(source: int = 1, port: bytes = b"C") -> bytes:
    payload = struct.pack("<BcBB5f", DEVICE_LSM6DSV320X, port, 0x6B, 1, 960.0, 10.417, 0.488, 70.0, 1.0)
    return block(BLOCK_SOURCE_INFO, source, payload)


def record(tmicros: int, fw: str = FW, a0: int = 0) -> bytes:
    dt = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw)]().newbyteorder("<")
    rec = np.zeros(1, dtype=dt)
    rec["tmicros"] = np.int64(tmicros & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    rec["a0"] = a0
    return block(BLOCK_RECORD, 0, rec.tobytes())


def entry(t_us: int, sensor: int, xyz=(0, 0, 0), flags: int = FIFO_ENTRY_T_EDGE, raw: bytes = None) -> bytes:
    data = raw if raw is not None else struct.pack("<3h", *xyz)
    return struct.pack("<IBB", t_us & 0xFFFFFFFF, (sensor << 3) | 0b010, flags) + data


def fifo(entries: list, source: int = 1, flags: int = 0) -> bytes:
    return block(BLOCK_FIFO, source, bytes([flags, len(entries)]) + b"".join(entries))


def tick(t: int, source: int = 1) -> bytes:
    """One collector tick: gyro, low-g, high-g of one ODR instant, then the record."""
    return fifo([entry(t - 500, TAG_GYRO, (1, 2, 3)),
                 entry(t - 500, TAG_ACCEL_LG, (4, 5, 6)),
                 entry(t - 500, TAG_ACCEL_HG, (7, 8, 9))], source) + record(t)


def stream(*parts: bytes, n_sources: int = 1) -> bytes:
    return file_header(n_sources=n_sources) + source_info() + b"".join(parts)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fw_ver,size", [("0.13-i2c_320g", 72), ("0.13-i2c_320g_2", 64)])
def test_record_sizes_match_firmware(fw_ver, size):
    assert FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]().itemsize == size


@pytest.mark.parametrize("fw_ver,key", [
    ("0.13-i2c_320g", "i2c_320g"), ("0.13-i2c_320g_2", "i2c_320g_2"),
    # pre-release names: i2c_320x, and "i2c_320" as cut short by get_fw_ver()
    ("0.13-i2c_320x", "i2c_320g"), ("0.13-i2c_320", "i2c_320g"), ("0.13-i2c_320x_2", "i2c_320g_2"),
])
def test_dtype_keys(fw_ver, key):
    assert CassCommands.get_dtype_key(fw_ver) == key


def test_parse_and_decode():
    data = stream(tick(40000), tick(42000), tick(44000))
    s = read_stream(data)
    assert s.fw_version == FW and s.record_len == 72 and not s.truncated and s.skipped_bytes == 0
    assert s.sources[1].port == "C" and s.sources[1].has_mag
    assert len(s.records) == 3 and len(s.fifo[1]) == 9

    d = process_stream(data)
    assert list(d.records["tmicros"]) == [0, 2000, 4000]
    imu = d.imu["C"]
    # Times share the records' zero (first record at 40000 µs).
    assert list(imu.accel_hg["tmicros"]) == [-500, 1500, 3500]
    assert imu.accel_hg["t_edge"].all()
    assert imu.accel_hg[["x", "y", "z"]].iloc[0].tolist() == pytest.approx([7 * 10.417, 8 * 10.417, 9 * 10.417], rel=1e-6)
    assert imu.accel_lg["x"].iloc[0] == pytest.approx(4 * 0.488, rel=1e-6)
    assert imu.gyro["x"].iloc[0] == pytest.approx(0.070)  # 70 mdps/LSB -> dps


def test_mag_timestamp_and_nack():
    words = [entry(40000, TAG_ACCEL_HG), entry(40000, TAG_MAG, (1000, -2000, 30)),
             entry(40000, TAG_TIMESTAMP, raw=struct.pack("<IH", 123456789, 0)),
             entry(40000, TAG_MAG_NACK), entry(40000, 0x07, flags=0)]
    imu = process_stream(stream(fifo(words), record(41000))).imu["C"]
    assert imu.mag[["x", "y", "z"]].iloc[0].tolist() == pytest.approx([100.0, -200.0, 3.0])  # µT
    assert imu.timestamp["ticks"].iloc[0] == 123456789
    assert imu.mag_nacks == 1 and imu.unknown_words == 1


def test_estimated_times_are_flagged():
    imu = process_stream(stream(fifo([entry(40000, TAG_ACCEL_HG, flags=0)]), record(40000))).imu["C"]
    assert not imu.accel_hg["t_edge"].iloc[0]


def test_events():
    data = stream(tick(40000), fifo([], flags=0x08), record(42000), fifo([], flags=0x10 | 0x01), record(44000))
    ev = process_stream(data).imu["C"].events
    assert len(ev) == 2
    assert ev["sync_lost"].tolist() == [True, False]
    assert ev["resynced"].tolist() == [False, True] and ev["overrun"].tolist() == [False, True]
    assert ev["tmicros"].tolist() == [0, 2000]  # the record before each block


def test_truncated_last_block():
    data = stream(tick(40000), tick(42000))
    s = read_stream(data[:-10])
    assert s.truncated
    assert len(s.records) == 1 and len(s.fifo[1]) == 6


def test_resync_after_garbage():
    data = stream(tick(40000), b"\xa5\x10\x00\xff\xff junk \xa5", tick(42000))
    s = read_stream(data)
    assert s.skipped_bytes > 0 and not s.truncated
    assert len(s.records) == 2 and len(s.fifo[1]) == 6


def test_time_wrap():
    near = 2**32 - 1000
    d = process_stream(stream(tick(near), tick(near + 2000)))
    assert list(d.records["tmicros"]) == [0, 2000]
    assert list(d.imu["C"].accel_hg["tmicros"]) == [-500, 1500]
    assert list(unwrap_u32(np.array([2**32 - 1, 0, 1, 2**32 - 2], dtype=np.uint32))) == [2**32 - 1, 2**32, 2**32 + 1, 2**32 - 2]


def test_not_a_stream_file(tmp_path):
    legacy = tmp_path / "legacy.bin"
    legacy.write_bytes(np.zeros(4, dtype=FIRMWARE_DTYPES["std"]()).tobytes())
    assert not is_stream_file(legacy)
    with pytest.raises(StreamFormatError):
        read_stream(legacy)


def test_process_data_file_detects_stream(tmp_path):
    path = tmp_path / "LOG_1.bin"
    path.write_bytes(stream(tick(40000), tick(42000)))
    df = CassCommands.process_data_file(path, "0.09-std")  # fw_ver ignored: the header names the firmware
    assert list(df["tmicros"]) == [0, 2000]
    assert "e0" in df.columns and "c0" not in df.columns
    assert "C" in CassCommands.process_stream_file(path).imu


@pytest.mark.skipif(not REAL_LOG.exists(), reason="example log not present")
def test_real_log():
    d = CassCommands.process_stream_file(REAL_LOG)
    assert d.fw_version.startswith("0.13-") and d.skipped_bytes == 0
    imu = d.imu["C"]
    hg = imu.accel_hg
    assert len(hg) == len(imu.accel_lg) == len(imu.gyro) > 8000
    assert hg["t_edge"].all() and imu.events.empty
    period = hg["tmicros"].diff().dropna()
    assert period.between(1060, 1075).all()  # IMU clock ~936.7 Hz
    # Low-g and gyro share their high-g sample's time.
    assert set(imu.accel_lg["tmicros"]) == set(hg["tmicros"]) == set(imu.gyro["tmicros"])
