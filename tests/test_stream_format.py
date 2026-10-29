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
    DEVICE_ICM45686_MAG,
    DEVICE_LSM6DSV320X,
    FIFO_ENTRY_T_EDGE,
    FIFO_ENTRY_T_FIT,
    SYNC,
    TAG_ACCEL,
    TAG_ACCEL_HG,
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


def file_header(fw: str = FW, n_sources: int = 1, version: int = 2) -> bytes:
    rec_len = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw)]().itemsize
    payload = struct.pack("<4sH16sHHB", b"CASS", version, fw.encode(), 500, rec_len, n_sources)
    return block(BLOCK_FILE_HEADER, 0, payload)


def source_info(source: int = 1, port: bytes = b"C") -> bytes:
    payload = struct.pack("<BcBB5f", DEVICE_LSM6DSV320X, port, 0x6B, 1, 960.0, 10.417, 0.488, 70.0, 1.0)
    return block(BLOCK_SOURCE_INFO, source, payload)


def icm_source_info(source: int = 1, port: bytes = b"C") -> bytes:
    payload = struct.pack("<BcBB5f", DEVICE_ICM45686_MAG, port, 0x69, 1, 800.0, 0.0,
                          32000 / 32768, 2000000 / 32768, 1.0)
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


def fifo(entries: list, source: int = 1, flags: int = 0, read_us: int = 0, version: int = 2) -> bytes:
    header = bytes([flags, len(entries)]) + (struct.pack("<I", read_us & 0xFFFFFFFF) if version >= 2 else b"")
    return block(BLOCK_FIFO, source, header + b"".join(entries))


def tick(t: int, source: int = 1, version: int = 2) -> bytes:
    """One collector tick: gyro, low-g, high-g of one ODR instant, then the record."""
    return fifo([entry(t - 500, TAG_GYRO, (1, 2, 3)),
                 entry(t - 500, TAG_ACCEL, (4, 5, 6)),
                 entry(t - 500, TAG_ACCEL_HG, (7, 8, 9))], source, read_us=t - 200, version=version) + record(t)


def stream(*parts: bytes, n_sources: int = 1, version: int = 2) -> bytes:
    return file_header(n_sources=n_sources, version=version) + source_info() + b"".join(parts)


# ── Tests ─────────────────────────────────────────────────────────────────────

# Record sizes must match the firmware's static_asserts (deviceConfig.h).
@pytest.mark.parametrize("fw_ver,size", [
    ("0.13-std", 40), ("0.13-i2c_1", 72), ("0.13-i2c_2", 100),
    ("0.13-i2c_320g", 36), ("0.13-i2c_320g_2", 28), ("0.13-i2c_45686", 36), ("0.13-i2c_45686_2", 28),
    ("0.12-std", 76), ("0.12-i2c_45686", 108), ("0.12-i2c_45686_2", 136),
])
def test_record_sizes_match_firmware(fw_ver, size):
    assert FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]().itemsize == size


@pytest.mark.parametrize("fw_ver,key", [
    ("0.12-std", "std"), ("0.13-std", "analog_18"), ("0.14-std", "analog_18"),
    ("0.13-i2c_1", "analog_15_i2c"), ("0.13-i2c_2", "analog_12_i2c2"),
    ("0.13-i2c_320g", "analog_15"), ("0.13-i2c_320g_2", "analog_12"),
    # pre-release names: i2c_320x, and "i2c_320" as cut short by get_fw_ver()
    ("0.13-i2c_320x", "analog_15"), ("0.13-i2c_320", "analog_15"), ("0.13-i2c_320x_2", "analog_12"),
    ("0.12-i2c_45686", "i2c_1"), ("0.13-i2c_45686", "analog_15"),
    ("0.12-i2c_45686_2", "i2c_2"), ("0.13-i2c_45686_2", "analog_12"),
])
def test_dtype_keys(fw_ver, key):
    assert CassCommands.get_dtype_key(fw_ver) == key


# The first 0.13 test builds still had the SPI IMU in the record: the record
# length (from the file header or live frame) picks that layout.
@pytest.mark.parametrize("fw_ver,record_len,key", [
    ("0.13-i2c_320x", 72, "stream_1"), ("0.13-i2c_45686", 72, "stream_1"), ("0.13-i2c_45686_2", 64, "stream_2"),
    ("0.13-i2c_45686", 36, "analog_15"),
])
def test_dtype_key_by_record_len(fw_ver, record_len, key):
    assert CassCommands.get_dtype_key(fw_ver, record_len) == key


def test_dtype_key_unknown_record_len():
    with pytest.raises(ValueError):
        CassCommands.get_dtype_key("0.13-i2c_45686", 50)


def test_parse_and_decode():
    data = stream(tick(40000), tick(42000), tick(44000))
    s = read_stream(data)
    assert s.fw_version == FW and s.record_len == 36 and not s.truncated and s.skipped_bytes == 0
    assert s.sources[1].port == "C" and s.sources[1].has_mag
    assert len(s.records) == 3 and len(s.fifo[1]) == 9

    d = process_stream(data)
    assert list(d.records["tmicros"]) == [0, 2000, 4000]
    imu = d.imu["C"]
    # Times share the records' zero (first record at 40000 µs).
    assert list(imu.accel_hg["tmicros"]) == [-500, 1500, 3500]
    assert imu.accel_hg["t_edge"].all()
    assert imu.accel_hg[["x", "y", "z"]].iloc[0].tolist() == pytest.approx([7 * 10.417, 8 * 10.417, 9 * 10.417], rel=1e-6)
    assert imu.accel["x"].iloc[0] == pytest.approx(4 * 0.488, rel=1e-6)
    assert imu.gyro["x"].iloc[0] == pytest.approx(0.070)  # 70 mdps/LSB -> dps
    # One read per FIFO block, timed by its read_us.
    assert list(imu.reads["tmicros"]) == [-200, 1800, 3800]
    assert list(imu.reads["entries"]) == [3, 3, 3]


def test_format_1_still_parses():
    data = stream(tick(40000, version=1), tick(42000, version=1), version=1)
    s = read_stream(data)
    assert s.format_version == 1 and not s.truncated and s.skipped_bytes == 0
    imu = process_stream(data).imu["C"]
    assert list(imu.accel_hg["tmicros"]) == [-500, 1500]
    assert imu.reads.empty  # format 1 has no read times


def test_icm45686_source():
    # Accel + gyro per packet timed by the clock fit (T_FIT), a raw timestamp,
    # and a mag sample timed by its read.
    words = [entry(40000, TAG_ACCEL, (1024, 0, -1024), flags=FIFO_ENTRY_T_FIT),
             entry(40000, TAG_GYRO, (16384, 0, 0), flags=FIFO_ENTRY_T_FIT),
             entry(40000, TAG_TIMESTAMP, raw=struct.pack("<IH", 70000, 0), flags=FIFO_ENTRY_T_FIT),
             entry(41250, TAG_ACCEL, (0, 0, 1024), flags=0),
             entry(41250, TAG_GYRO, (0, 0, 0), flags=0),
             entry(41500, TAG_MAG, (500, 0, 0), flags=0)]
    data = file_header() + icm_source_info() + fifo(words, read_us=41600) + record(40000)
    imu = process_stream(data).imu["C"]
    assert imu.device == "ICM-45686 + QMC6309"
    assert imu.accel_hg.empty
    assert imu.accel[["x", "z"]].iloc[0].tolist() == pytest.approx([1000.0, -1000.0])  # ±32 g: 32000/32768 mg/LSB
    assert imu.gyro["x"].iloc[0] == pytest.approx(1000.0)                               # ±2000 dps: half scale
    assert imu.accel["t_fit"].tolist() == [True, False] and not imu.accel["t_edge"].any()
    assert list(imu.accel["tmicros"]) == [0, 1250]
    assert imu.timestamp["ticks"].iloc[0] == 70000
    assert imu.mag["x"].iloc[0] == pytest.approx(50.0)  # 1 mG/LSB = 0.1 µT


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
    data = stream(tick(40000), fifo([], flags=0x08, read_us=41800), record(42000),
                  fifo([], flags=0x10 | 0x01, read_us=43800), record(44000))
    ev = process_stream(data).imu["C"].events
    assert len(ev) == 2
    assert ev["sync_lost"].tolist() == [True, False]
    assert ev["resynced"].tolist() == [False, True] and ev["overrun"].tolist() == [False, True]
    assert ev["tmicros"].tolist() == [1800, 3800]  # each block's read time


def test_events_format_1():
    data = stream(tick(40000, version=1), fifo([], flags=0x08, version=1), record(42000), version=1)
    ev = process_stream(data).imu["C"].events
    assert ev["tmicros"].tolist() == [0]  # no read times: the record before the block


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
    assert len(hg) == len(imu.accel) == len(imu.gyro) > 8000
    assert hg["t_edge"].all() and imu.events.empty
    period = hg["tmicros"].diff().dropna()
    assert period.between(1060, 1075).all()  # IMU clock ~936.7 Hz
    # Low-g and gyro share their high-g sample's time.
    assert set(imu.accel["tmicros"]) == set(hg["tmicros"]) == set(imu.gyro["tmicros"])


REAL_LOG_2 = Path(__file__).parent.parent / "examples" / "data" / "icm20948-icm45686-fifo_i2c_45686" / "LOG_1793290048.bin"
LEGACY_LOG = Path(__file__).parent.parent / "examples" / "data" / "icm45686-test-9dof_800" / "LOG_1783595119.bin"


def test_icm20948_source():
    # The internal SPI ICM-20948: source 3, port 'S', ±16 g / ±2000 dps / AK09916.
    payload = struct.pack("<BcBB5f", 1, b"S", 0, 1, 1100.0, 0.0, 1000 / 2048, 1000 / 16.4, 1.5)
    words = [entry(40000, TAG_ACCEL, (2048, 0, 0), flags=FIFO_ENTRY_T_FIT),
             entry(40000, TAG_GYRO, (164, 0, 0), flags=FIFO_ENTRY_T_FIT),
             entry(40000, TAG_MAG, (100, 0, 0), flags=0)]
    data = file_header() + block(BLOCK_SOURCE_INFO, 3, payload) + fifo(words, source=3, read_us=40500) + record(40000)
    imu = process_stream(data).imu["S"]
    assert imu.device == "ICM-20948"
    assert imu.accel["x"].iloc[0] == pytest.approx(1000.0)  # 1 g
    assert imu.gyro["x"].iloc[0] == pytest.approx(10.0)
    assert imu.mag["x"].iloc[0] == pytest.approx(15.0)     # 0.15 µT/LSB


def test_legacy_file_through_process_stream_file():
    if not LEGACY_LOG.exists():
        pytest.skip("example log not present")
    with pytest.raises(ValueError, match="fw_ver"):
        CassCommands.process_stream_file(LEGACY_LOG)
    d = CassCommands.process_stream_file(LEGACY_LOG, "0.10-i2c_45686_9dof")
    legacy = CassCommands.process_data_file(LEGACY_LOG, "0.10-i2c_45686_9dof")
    assert d.records.equals(legacy)
    assert set(d.imu) == {"S", "C"}
    assert d.imu["C"].device == "ICM-45686 + QMC6309"
    assert (d.imu["S"].accel["x"].values == legacy["gx"].values).all()
    assert (d.imu["C"].mag["z"].values == legacy["Tz_i2c"].values).all()
    assert not d.imu["C"].accel["t_fit"].any() and d.imu["C"].accel_hg.empty


@pytest.mark.skipif(not REAL_LOG_2.exists(), reason="example log not present")
def test_real_log_two_fifo_imus():
    d = CassCommands.process_stream_file(REAL_LOG_2)
    assert d.fw_version == "0.13-i2c_45686" and d.skipped_bytes == 0
    assert list(d.records.columns)[:3] == ["tmicros", "t", "a0"] and "gx" not in d.records.columns
    s, c = d.imu["S"], d.imu["C"]
    assert (s.device, c.device) == ("ICM-20948", "ICM-45686 + QMC6309")
    for imu, lo, hi in ((s, 1080, 1180), (c, 780, 860)):
        span = imu.accel["t"].iloc[-1] - imu.accel["t"].iloc[0]
        assert lo < len(imu.accel) / span < hi
        locked = imu.accel[imu.accel["t"] > 2.0]
        assert locked["t_fit"].all()
        assert locked["tmicros"].diff().dropna().std() < 2.0  # µs: regular sample times
        assert imu.events[["overrun", "read_failed", "late", "sync_lost"]].to_numpy().sum() == 0
