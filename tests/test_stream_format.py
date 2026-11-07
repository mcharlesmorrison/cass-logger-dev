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
    PORT_ANALOG,
    PORT_IMU_NO_MAG,
    PORT_WRONG_PORT,
    PORT_IMU_LOST,
    ports_from_bytes,
    PORT_IMU_OK,
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
    read_file_info,
    read_stream,
    unwrap_u32,
)

FW = "0.13-i2c_320g"


# ── Builders: blocks exactly as docs/stream-format.md describes ──────────────

def block(btype: int, source: int, payload: bytes) -> bytes:
    return bytes([SYNC, btype, source]) + struct.pack("<H", len(payload)) + payload


def file_header(fw: str = FW, n_sources: int = 1, version: int = 3, device_id: str = "AB123",
                analog_mask: int = 0x3FFFF, ports=((PORT_ANALOG, 0, 0), (PORT_ANALOG, 0, 0)),
                pullup_mask: int = 0) -> bytes:
    rec_len = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw)]().itemsize
    payload = struct.pack("<4sH16s8sHHB", b"CASS", version, fw.encode(), device_id.encode(),
                          500, rec_len, n_sources)
    if version >= 4:  # analog_mask, pullup_mask, then port C and E (state, device_type, i2c_addr)
        payload += struct.pack("<II3B3B", analog_mask, pullup_mask, *ports[0], *ports[1])
    return block(BLOCK_FILE_HEADER, 0, payload)


def imu_source_info(source: int, device: int, port: bytes, addr: int, sensors: int, pairs: list,
                    kind: int = 1) -> bytes:
    """SOURCE_INFO: common header, IMU part, one (rate, scale) pair per set bit."""
    payload = struct.pack("<BBcBB", kind, device, port, addr, sensors)
    payload += b"".join(struct.pack("<ff", r, s) for r, s in pairs)
    return block(BLOCK_SOURCE_INFO, source, payload)


def source_info(source: int = 1, port: bytes = b"C") -> bytes:
    """An LSM6DSV320X + QMC6309 source."""
    # accel, gyro, mag, high-g
    return imu_source_info(source, DEVICE_LSM6DSV320X, port, 0x6B, 0x0F,
                           [(960.0, 0.488), (960.0, 70.0), (120.0, 1.0), (960.0, 10.417)])


def icm_source_info(source: int = 1, port: bytes = b"C") -> bytes:
    """An ICM-45686 + QMC6309 source: accel, gyro, mag."""
    return imu_source_info(source, DEVICE_ICM45686_MAG, port, 0x69, 0x07,
                           [(800.0, 32000 / 32768), (800.0, 2000000 / 32768), (250.0, 1.0)])


def record(tmicros: int, fw: str = FW, a0: int = 0) -> bytes:
    dt = FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw)]().newbyteorder("<")
    rec = np.zeros(1, dtype=dt)
    rec["tmicros"] = np.int64(tmicros & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    rec["a0"] = a0
    return block(BLOCK_RECORD, 0, rec.tobytes())


def entry(t_us: int, sensor: int, xyz=(0, 0, 0), flags: int = FIFO_ENTRY_T_EDGE, raw: bytes = None) -> bytes:
    data = raw if raw is not None else struct.pack("<3h", *xyz)
    return struct.pack("<IBB", t_us & 0xFFFFFFFF, (sensor << 3) | 0b010, flags) + data


def fifo(entries: list, source: int = 1, flags: int = 0, read_us: int = 0) -> bytes:
    header = struct.pack("<BBI", flags, len(entries), read_us & 0xFFFFFFFF)
    return block(BLOCK_FIFO, source, header + b"".join(entries))


def tick(t: int, source: int = 1) -> bytes:
    """One collector tick: gyro, low-g, high-g of one ODR instant, then the record."""
    return fifo([entry(t - 500, TAG_GYRO, (1, 2, 3)),
                 entry(t - 500, TAG_ACCEL, (4, 5, 6)),
                 entry(t - 500, TAG_ACCEL_HG, (7, 8, 9))], source, read_us=t - 200) + record(t)


def stream(*parts: bytes, n_sources: int = 1) -> bytes:
    return file_header(n_sources=n_sources) + source_info() + b"".join(parts)


# ── Tests ─────────────────────────────────────────────────────────────────────

# Record sizes must match the firmware's static_asserts (deviceConfig.h).
@pytest.mark.parametrize("fw_ver,size", [
    ("0.13-std", 40),
    ("0.13-i2c_320g", 36), ("0.13-i2c_320g_2", 28), ("0.13-i2c_45686", 36), ("0.13-i2c_45686_2", 28),
    ("0.12-std", 76), ("0.12-i2c_45686", 108), ("0.12-i2c_45686_2", 136),
])
def test_record_sizes_match_firmware(fw_ver, size):
    assert FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]().itemsize == size


@pytest.mark.parametrize("fw_ver,key", [
    ("0.12-std", "std"), ("0.13-std", "analog_18"), ("0.14-std", "analog_18"),
    ("0.13-i2c_320g", "analog_15"), ("0.13-i2c_320g_2", "analog_12"),
    ("0.12-i2c_45686", "i2c_1"), ("0.13-i2c_45686", "analog_15"),
    ("0.12-i2c_45686_2", "i2c_2"), ("0.13-i2c_45686_2", "analog_12"),
])
def test_dtype_keys(fw_ver, key):
    assert CassCommands.get_dtype_key(fw_ver) == key


def test_dtype_key_by_record_len():
    assert CassCommands.get_dtype_key("0.13-i2c_45686", 36) == "analog_15"


def test_dtype_key_unknown_record_len():
    with pytest.raises(ValueError):
        CassCommands.get_dtype_key("0.13-i2c_45686", 50)


@pytest.mark.parametrize("fw_ver", ["0.13-i2c_1", "0.13-i2c_2"])
def test_dtype_key_dropped_variants(fw_ver):
    # i2c_1 / i2c_2 have no 0.13 build; their older files still parse.
    with pytest.raises(ValueError, match="No firmware"):
        CassCommands.get_dtype_key(fw_ver)
    assert CassCommands.get_dtype_key(fw_ver.replace("0.13", "0.12")) in ("i2c_1", "i2c_2")


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


def test_device_id():
    s = read_stream(file_header() + source_info() + tick(40000))
    assert s.device_id == "AB123"
    assert process_stream(file_header() + source_info() + tick(40000)).device_id == "AB123"
    # Never set: all NULs.
    assert read_stream(file_header(device_id="") + source_info() + tick(40000)).device_id == ""


@pytest.mark.parametrize("version", [1, 2, 5])
def test_other_format_versions_rejected(version):
    data = file_header(version=version) + source_info() + tick(40000)
    with pytest.raises(StreamFormatError, match=f"version {version}"):
        read_stream(data)


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


LEGACY_LOG = Path(__file__).parent.parent / "examples" / "data" / "icm45686-test-9dof_800" / "LOG_1783595119.bin"


def test_icm20948_source():
    # The internal SPI ICM-20948: source 3, port 'S', ±16 g / ±2000 dps / AK09916.
    info = imu_source_info(3, 1, b"S", 0, 0x07, [(1125.0, 1000 / 2048), (1100.0, 1000 / 16.4), (100.0, 1.5)])
    words = [entry(40000, TAG_ACCEL, (2048, 0, 0), flags=FIFO_ENTRY_T_FIT),
             entry(40000, TAG_GYRO, (164, 0, 0), flags=FIFO_ENTRY_T_FIT),
             entry(40000, TAG_MAG, (100, 0, 0), flags=0)]
    data = file_header() + info + fifo(words, source=3, read_us=40500) + record(40000)
    s = read_stream(data).sources[3]
    assert (s.accel_hz, s.gyro_hz, s.mag_hz, s.hg_hz) == (1125.0, 1100.0, 100.0, 0.0)
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


# ── SOURCE_INFO ────────────────────────────────────────────────────────────

def test_source_info_v3_all_sensors():
    s = read_stream(stream(tick(40000))).sources[1]
    assert s.is_imu and s.device == "LSM6DSV320X + QMC6309" and s.port == "C" and s.i2c_addr == 0x6B
    assert s.sensors == 0x0F and s.has_mag
    assert (s.accel_hz, s.gyro_hz, s.mag_hz, s.hg_hz) == (960.0, 960.0, 120.0, 960.0)
    assert s.hg_mg_per_lsb == pytest.approx(10.417)


def test_accel_only_imu():
    info = imu_source_info(1, 2, b"C", 0x69, 0x01, [(800.0, 0.5)])
    data = file_header() + info + fifo([entry(40000, TAG_ACCEL, (2000, 0, 0))]) + record(40000)
    s = read_stream(data).sources[1]
    assert s.sensors == 0x01 and not s.has_mag and s.gyro_mdps_per_lsb == 0.0
    imu = process_stream(data).imu["C"]
    assert imu.accel["x"].iloc[0] == pytest.approx(1000.0) and imu.gyro.empty


def test_unknown_sensor_bit_is_skipped():
    # A future sensor at bit 4: its pair sits after the four known ones and is skipped.
    info = imu_source_info(1, 2, b"C", 0x69, 0x13, [(800.0, 0.5), (800.0, 61.0), (50.0, 9.9)])
    s = read_stream(file_header() + info + record(40000)).sources[1]
    assert s.sensors == 0x13 and (s.accel_hz, s.gyro_hz) == (800.0, 800.0)
    assert s.accel_mg_per_lsb == pytest.approx(0.5) and s.gyro_mdps_per_lsb == pytest.approx(61.0)


def test_unknown_source_kind_is_listed_not_decoded():
    # e.g. a future GPS: kind 2, with a layout this parser doesn't know.
    gps = block(BLOCK_SOURCE_INFO, 4, struct.pack("<BBc", 2, 7, b"U") + b"\x00" * 6)
    data = file_header(n_sources=2) + source_info() + gps + tick(40000)
    s = read_stream(data)
    assert s.sources[4].source_kind == 2 and not s.sources[4].is_imu and s.sources[4].port == "U"
    assert set(process_stream(data).imu) == {"C"}


def test_source_info_length_must_match_sensors():
    # Bitmask says 3 sensors but only 2 pairs follow: the block is rejected.
    bad = imu_source_info(1, 2, b"C", 0x69, 0x07, [(800.0, 0.5), (800.0, 61.0)])
    assert 1 not in read_stream(file_header() + bad + record(40000)).sources


def test_read_file_info(tmp_path):
    f = tmp_path / "LOG_1.bin"
    f.write_bytes(file_header(device_id="CX7") + source_info() + tick(40000))
    info = read_file_info(f)
    assert (info.fw_version, info.device_id, info.format_version, info.record_len) == (FW, "CX7", 3, 36)

    legacy = tmp_path / "LOG_3.bin"
    legacy.write_bytes(bytes(64))  # raw datrec structs, no header
    assert read_file_info(legacy) is None


# ── Format v4 (firmware 0.14, one "auto" firmware) ───────────────────────────

AUTO = "0.14-auto"
MASK_C_IMU = 0x3FFFF & ~(0x7 << 9)  # everything but c0-c2 (bits 9-11)


def v4_stream(*parts: bytes, analog_mask: int = MASK_C_IMU, pullup_mask: int = 0x38,
              ports=((PORT_IMU_OK, DEVICE_LSM6DSV320X, 0x6B), (PORT_ANALOG, 0, 0))) -> bytes:
    return (file_header(AUTO, version=4, analog_mask=analog_mask, ports=ports, pullup_mask=pullup_mask)
            + source_info()
            + b"".join(parts))


def test_auto_record_layout():
    assert CassCommands.get_dtype_key(AUTO) == "analog_18"
    assert FIRMWARE_DTYPES["analog_18"]().itemsize == 40


def test_v4_header_mask_and_ports(tmp_path):
    data = v4_stream(record(40000, AUTO, a0=7), record(42000, AUTO, a0=8))
    s = read_stream(data)
    assert s.format_version == 4 and s.record_len == 40 and not s.truncated
    assert s.analog_mask == MASK_C_IMU and s.pullup_mask == 0x38  # E0-E2: switches
    assert s.ports["C"].state == PORT_IMU_OK and s.ports["C"].i2c_addr == 0x6B
    assert s.ports["C"].device == "LSM6DSV320X + QMC6309"
    assert s.ports["E"].is_analog and s.ports["E"].device == ""

    d = process_stream(data)
    # Masked channels (port C, used by the IMU) are left out, as 0.13 i2c_320g did.
    assert not {"c0", "c1", "c2"} & set(d.records.columns)
    assert {"d0", "e0", "e2", "f0", "a0", "b2"} <= set(d.records.columns)
    assert d.ports["C"].state == PORT_IMU_OK and d.pullup_mask == 0x38

    path = tmp_path / "LOG.bin"
    path.write_bytes(data)
    info = read_file_info(path)
    assert info.format_version == 4 and info.analog_mask == MASK_C_IMU and info.ports["E"].is_analog
    assert info.pullup_mask == 0x38
    # process_data_file gives the same record columns.
    assert list(CassCommands.process_data_file(path).columns) == list(d.records.columns)


def test_v4_all_analog_keeps_every_channel():
    d = process_stream(v4_stream(record(40000, AUTO), analog_mask=0x3FFFF,
                                 ports=((PORT_ANALOG, 0, 0), (PORT_ANALOG, 0, 0))))
    assert {"c0", "c1", "c2", "e0", "e1", "e2"} <= set(d.records.columns)


def test_v4_header_wrong_length_rejected():
    # A v4 header must be 49 bytes; a 35-byte (v3-sized) one is unusable.
    data = file_header(AUTO, version=3)
    data = data[:5] + data[5:9] + struct.pack("<H", 4) + data[11:]
    with pytest.raises(StreamFormatError, match="expected 49"):
        read_stream(data + record(40000, AUTO))


def test_v3_file_has_no_mask_or_ports():
    s = read_stream(stream(tick(40000)))
    assert s.format_version == 3 and s.analog_mask is None and s.ports == {}


def test_lsm_without_mag_is_named_without_qmc():
    no_mag = imu_source_info(1, DEVICE_LSM6DSV320X, b"C", 0x6B, 0x0B,  # accel, gyro, high-g
                             [(960.0, 0.488), (960.0, 70.0), (960.0, 10.417)])
    data = (file_header(AUTO, version=4, analog_mask=MASK_C_IMU,
                        ports=((PORT_IMU_NO_MAG, DEVICE_LSM6DSV320X, 0x6B), (PORT_ANALOG, 0, 0)))
            + no_mag + record(40000, AUTO))
    s = read_stream(data)
    assert s.sources[1].device == "LSM6DSV320X" and not s.sources[1].has_mag
    assert s.ports["C"].state_name == "imu_no_mag" and s.ports["C"].device == "LSM6DSV320X"


def test_wrong_port_state():
    p = ports_from_bytes(bytes([PORT_WRONG_PORT, DEVICE_LSM6DSV320X, 0x6A, PORT_ANALOG, 0, 0]))
    assert p["C"].state_name == "wrong_port" and p["C"].device == "LSM6DSV320X + QMC6309"
    assert p["C"].i2c_addr == 0x6A and not p["C"].is_analog and p["E"].is_analog


def test_imu_lost_state():
    p = ports_from_bytes(bytes([PORT_ANALOG, 0, 0, PORT_IMU_LOST, DEVICE_LSM6DSV320X, 0x6A]))
    assert p["E"].state_name == "imu_lost" and p["E"].device == "LSM6DSV320X + QMC6309"
    assert not p["E"].is_analog
