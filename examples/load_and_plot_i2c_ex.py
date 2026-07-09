"""
Example: Load a pre-downloaded binary file and plot I2C IMU data.

Workflow
--------
1. ``import_data`` — loads the pre-bundled ``.bin`` file from the
   ``examples/data/`` directory and parses it into a DataFrame using
   ``CassCommands.process_data_file``.
2. ``plot_pot_data`` — 

Data columns used
-----------------
- ``t``  : elapsed time in seconds (derived from the ``tmicros`` field)
- 

Constants
---------


Notes
-----
- No device connection is required; the example ``.bin`` file is bundled in
  ``examples/data/``.
- To download live data from a connected device use
  ``CassCommands.download_all()`` (see ``download_and_plot_ex.py``).
- ``process_data_file`` defaults to the ``"std"`` firmware dtype; pass the
  ``fw_ver`` keyword if the file was recorded with an I2C firmware variant.

Usage
-----
Run directly::

    python examples/load_and_plot_i2c_ex.py
"""

import cass_logger_dev.cass_commands as cass_commands
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd


def import_data() -> pd.DataFrame:
    """Load and parse the example binary data file.

    Locates the pre-bundled ``.bin`` file in ``examples/data/``,
    parses it with ``CassCommands.process_data_file``, and returns
    the result as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Parsed sensor data with columns including ``t`` (seconds),
    """
    cass_util = cass_commands.CassCommands()
    filepath = str(
        Path(__file__).parent / "data" / "icm45686-test-9dof_800" / "LOG_1783595119.bin"
    )
    fw_ver = "0.10-i2c_45686_9dof"

    return cass_util.process_data_file(filepath, fw_ver)


def plot_pot_data(example_data: pd.DataFrame):
    """Plot IMU data against time.

    NOTE: This is for the existing data in the examples/data dir.

    Parameters
    ----------
    example_data : pd.DataFrame
        DataFrame returned by ``import_data``. Must contain columns
        ``t``, ``a0``, and ``b0``.
    """
    plt.style.use("ggplot")

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    for ax in axs:
        ax.tick_params(axis="x", labelbottom=True)
    axs[0].plot(example_data["t"], example_data["gx_i2c"])
    axs[1].plot(example_data["t"], example_data["gy_i2c"])
    axs[2].plot(example_data["t"], example_data["gz_i2c"])

    # labeling / formatting
    axs[0].set_title("[Example] Accelerometer")
    plt.title("Example data plot")
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("gx")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("gy")
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("gz")

    plt.tight_layout()
    plt.show()

    # New plot for magnetometer data
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    for ax in axs:
        ax.tick_params(axis="x", labelbottom=True)
    axs[0].plot(example_data["t"], example_data["Tx_i2c"])
    axs[1].plot(example_data["t"], example_data["Ty_i2c"])
    axs[2].plot(example_data["t"], example_data["Tz_i2c"])

    # labeling / formatting
    axs[0].set_title("[Example] Magnetometer")
    plt.title("Example data plot")
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("wx")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("wy")
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("wz")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_data = import_data()
    plot_pot_data(example_data)
