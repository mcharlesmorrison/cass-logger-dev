"""
Example: Load a block stream file (LSM6DSV320X variants) and plot its IMU data.

Workflow
--------
1. ``import_data`` — loads the pre-bundled ``.bin`` file from
   ``examples/data/`` with ``CassCommands.process_stream_file``.
2. ``print_summary`` — sample counts, rates, and any FIFO events.
3. ``plot_imu_data`` — high-g vs low-g acceleration, gyro, and magnetometer.

Data used
---------
``process_stream_file`` returns a ``StreamData``:

- ``records`` : the 500 Hz record DataFrame (analog channels and the internal
  SPI IMU), the same as ``process_data_file`` returns.
- ``imu["C"]`` (and ``imu["E"]`` on ``i2c_320g_2``) : one LSM6DSV320X, as
  DataFrames ``accel_hg`` / ``accel_lg`` [mg], ``gyro`` [dps], ``mag`` [µT]
  and ``timestamp``, each with ``t`` (s), ``tmicros`` and ``t_edge``.

All times share the records' zero, so the frames can be plotted together.

Notes
-----
- No device connection is required; the example ``.bin`` file is bundled in
  ``examples/data/``.
- High-g, low-g and gyro come at the IMU's own rate (~937 Hz), not 500 Hz,
  each timestamped from its data-ready (INT1) edge.

Usage
-----
Run directly::

    python examples/load_and_plot_320g_ex.py
"""

from pathlib import Path

from matplotlib import pyplot as plt

from cass_logger_dev.cass_commands import CassCommands
from cass_logger_dev.stream_format import StreamData


def import_data() -> StreamData:
    """Load and parse the example block stream file."""
    filepath = Path(__file__).parent / "data" / "lsm6dsv320x-test_i2c_320g" / "LOG_1793206549.bin"
    return CassCommands.process_stream_file(filepath)


def print_summary(data: StreamData):
    """Print sample counts and rates per sensor, and any FIFO events."""
    span = data.records["t"].iloc[-1]
    print(f"Firmware {data.fw_version}: {len(data.records)} records over {span:.2f} s"
          f"{' (last block cut off)' if data.truncated else ''}")
    for port, imu in data.imu.items():
        print(f"IMU on port {port}:")
        for name in ["accel_hg", "accel_lg", "gyro", "mag", "timestamp"]:
            frame = getattr(imu, name)
            edge = frame["t_edge"].mean() * 100 if len(frame) else 0.0
            print(f"  {name:10s} {len(frame):7d} samples, {len(frame) / span:6.1f}/s, "
                  f"{edge:5.1f}% timed by INT1")
        if imu.events.empty:
            print("  no FIFO events (overruns, failed reads, sync losses)")
        else:
            print(imu.events.to_string())


def plot_imu_data(data: StreamData, port: str = "C"):
    """Plot one IMU's acceleration, rotation rate and magnetic field."""
    imu = data.imu[port]
    plt.style.use("ggplot")

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    for ax, axis in zip(axs, ["x", "y", "z"]):
        ax.plot(imu.accel_hg["t"], imu.accel_hg[axis], label="high-g (±320 g)")
        ax.plot(imu.accel_lg["t"], imu.accel_lg[axis], label="low-g (±16 g)")
        ax.set_ylabel(f"a{axis} [mg]")
        ax.tick_params(axis="x", labelbottom=True)
    axs[0].set_title(f"[Example] LSM6DSV320X port {port}: acceleration")
    axs[0].legend(loc="upper right")
    axs[2].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
    for axis in ["x", "y", "z"]:
        axs[0].plot(imu.gyro["t"], imu.gyro[axis], label=axis)
        axs[1].plot(imu.mag["t"], imu.mag[axis], label=axis)
    axs[0].set_title(f"[Example] LSM6DSV320X port {port}: gyro and magnetometer")
    axs[0].set_ylabel("rate [dps]")
    axs[1].set_ylabel("field [µT]")
    axs[1].set_xlabel("time [s]")
    for ax in axs:
        ax.legend(loc="upper right")
        ax.tick_params(axis="x", labelbottom=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_data = import_data()
    print_summary(example_data)
    plot_imu_data(example_data)
