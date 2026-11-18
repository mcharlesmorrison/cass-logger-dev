"""Pre-0.13 ICM-45686 gyro units: some 0.10 builds wrote mdps, the rest dps."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from cass_logger_dev.cass_commands import CassCommands
from cass_logger_dev.firmware_structs import FIRMWARE_DTYPES

DPS_PER_LSB = 2000 / 32768  # ±2000 dps
# Recorded 2026-07-09 with a 0.10 build that wrote the port C gyro in mdps.
MDPS_LOG = Path(__file__).parent.parent / "examples" / "data" / "icm45686-test-9dof_800" / "LOG_1783595119.bin"


def write_records(tmp_path, dtype_key, gyro_counts, suffix="_i2c", scale=DPS_PER_LSB):
    """A pre-0.13 file whose port IMU gyro reads ``gyro_counts`` raw LSBs × ``scale``."""
    dt = FIRMWARE_DTYPES[dtype_key]()
    rec = np.zeros(len(gyro_counts), dtype=dt)
    rec["tmicros"] = np.arange(len(gyro_counts)) * 2000
    for i, axis in enumerate("xyz"):
        rec[f"w{axis}{suffix}"] = gyro_counts[:, i] * scale
    path = tmp_path / "LOG_1.bin"
    rec.tofile(path)
    return path


@pytest.fixture
def moving_counts():
    rng = np.random.default_rng(0)
    return np.round(rng.normal(0, 3000, size=(2000, 3)))  # up to a few hundred dps


@pytest.fixture
def still_counts():
    rng = np.random.default_rng(1)
    return np.round(rng.normal(0, 4, size=(2000, 3)))  # gyro noise, well under 1 dps


def test_mdps_build_is_converted(tmp_path, moving_counts):
    path = write_records(tmp_path, "i2c_1", moving_counts, scale=DPS_PER_LSB * 1000)
    with pytest.warns(UserWarning, match="mdps"):
        df = CassCommands.process_data_file(path, "0.10-i2c_45686")
    assert np.allclose(df["wx_i2c"], moving_counts[:, 0] * DPS_PER_LSB, atol=1e-3)


@pytest.mark.parametrize("counts", ["moving_counts", "still_counts"])
def test_dps_build_is_left_alone(tmp_path, counts, request):
    counts = request.getfixturevalue(counts)
    path = write_records(tmp_path, "i2c_1", counts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = CassCommands.process_data_file(path, "0.10-i2c_45686_9dof")
    assert np.allclose(df["wz_i2c"], counts[:, 2] * DPS_PER_LSB, atol=1e-5)


def test_still_mdps_build_is_converted(tmp_path, still_counts):
    path = write_records(tmp_path, "i2c_1", still_counts, scale=DPS_PER_LSB * 1000)
    with pytest.warns(UserWarning, match="mdps"):
        df = CassCommands.process_data_file(path, "0.10-i2c_45686")
    assert df["wy_i2c"].abs().max() < 2


def test_both_ports_of_a_two_imu_variant(tmp_path, moving_counts):
    dt = FIRMWARE_DTYPES["i2c_2"]()
    rec = np.zeros(len(moving_counts), dtype=dt)
    for i, axis in enumerate("xyz"):
        rec[f"w{axis}_i2c_c"] = moving_counts[:, i] * DPS_PER_LSB * 1000  # mdps
        rec[f"w{axis}_i2c_e"] = moving_counts[:, i] * DPS_PER_LSB         # dps
    path = tmp_path / "LOG_1.bin"
    rec.tofile(path)
    with pytest.warns(UserWarning, match="_i2c_c"):
        df = CassCommands.process_data_file(path, "0.10-i2c_45686_2")
    assert np.allclose(df["wx_i2c_c"], df["wx_i2c_e"], atol=1e-3)


def test_zero_gyro_and_other_variants_are_left_alone(tmp_path, moving_counts):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zero = write_records(tmp_path, "i2c_1", np.zeros((100, 3)))
        assert (CassCommands.process_data_file(zero, "0.10-i2c_45686")["wx_i2c"] == 0).all()
        # The I2C ICM-20948 variant always wrote dps; never touched.
        path = write_records(tmp_path, "i2c_1", moving_counts, scale=DPS_PER_LSB * 1000)
        df = CassCommands.process_data_file(path, "0.10-i2c_1")
    assert df["wx_i2c"].abs().max() > 1000


def test_legacy_stream_data_converts_a_raw_record_table(moving_counts):
    # Callers like cass-utils' Run may hand legacy_stream_data a record table
    # that didn't come through process_data_file.
    import pandas as pd

    from cass_logger_dev.stream_format import legacy_stream_data

    n = len(moving_counts)
    cols = {"tmicros": np.arange(n) * 2000, "t": np.arange(n) * 2e-3}
    for i, axis in enumerate("xyz"):
        cols.update({f"g{axis}_i2c": np.zeros(n), f"T{axis}_i2c": np.zeros(n),
                     f"w{axis}_i2c": moving_counts[:, i] * DPS_PER_LSB * 1000})
    records = pd.DataFrame(cols)
    with pytest.warns(UserWarning, match="mdps"):
        data = legacy_stream_data(records, "0.10-i2c_45686")
    assert np.allclose(data.imu["C"].gyro["x"], moving_counts[:, 0] * DPS_PER_LSB, atol=1e-3)
    assert records["wx_i2c"].abs().max() > 1000  # the caller's table isn't modified
    # Already in dps: converting again is a no-op.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        legacy_stream_data(data.records, "0.10-i2c_45686")


def test_example_mdps_log():
    if not MDPS_LOG.exists():
        pytest.skip("example log not present")
    with pytest.warns(UserWarning, match="mdps"):
        data = CassCommands.process_stream_file(MDPS_LOG, "0.10-i2c_45686_9dof")
    gyro = data.imu["C"].gyro
    # The logger sat still for the first second: a few tenths of a dps of bias.
    still = gyro[gyro["t"] < 1.0]
    assert still[["x", "y", "z"]].abs().median().max() < 2
    # Now in the same units as the internal ICM-20948's gyro.
    assert gyro[["x", "y", "z"]].abs().max().max() < 2000
