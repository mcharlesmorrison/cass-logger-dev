"""
Example: Download data from a connected Cass Logger and plot internal IMU data.

This script demonstrates how to download binary files from a live device and
visualize the on-board accelerometer channels.

Workflow
--------
1. ``download_data`` — connects to a Cass Logger over serial, downloads all
   recorded ``.bin`` files via ``CassCommands.download_all()``, and returns
   the path to the timestamped download directory.
2. ``plot_internal_imu_data`` — iterates over every ``.bin`` file in that
   directory, parses each one with ``CassCommands.process_stream_file``, and
   renders a three-panel time-series plot of the internal IMU's accelerometer.
3. ``test_delete`` — lists files on the device, prompts the user for
   confirmation, then deletes all files from the SD card.

Data used
---------
``process_stream_file`` returns a ``StreamData`` for any firmware version;
``imu["S"]`` is the internal ICM-20948:

- ``imu["S"].accel`` : ``t`` (s) and ``x``, ``y``, ``z`` acceleration [mg]

Works with old and new firmware
-------------------------------
- Firmware 0.13+ files describe themselves (FILE_HEADER), so no ``fw_ver``
  is needed. The internal IMU is read at its own rate (~1.1 kHz).
- Pre-0.13 files are raw records with no header. ``download_all`` writes a
  ``metadata.txt`` next to them, and ``find_and_parse_metadata`` reads the
  ``fw_ver`` from it. Their IMU was polled into each 500 Hz record, and
  ``process_stream_file`` presents it the same way.

Notes
-----
- A Cass Logger must be connected over USB/serial before running this script.

Usage
-----
Run directly::

    python examples/download_and_plot_ex.py
"""

import cass_logger_dev.cass_commands as cass_commands
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

FORK_GAIN = 4.884e-02
SHOCK_GAIN = 2.442e-02

cass_util = cass_commands.CassCommands()


def download_data():
    """Download example data from a connected Cass Logger.

    Returns
    -------
    Path
        Path to the directory where the data was downloaded.
    """
    return cass_util.download_all()


def plot_internal_imu_data(data_dir: str):
    """Plot the internal IMU's acceleration (x, y, z) against time.

    Reads every .bin file in the given directory, from any firmware version,
    and renders a three-panel time-series figure per file.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .bin data files.
    """
    # metadata.txt for pre-0.13 downloads, else the .bin headers. Stream files
    # (0.13+) ignore fw_ver; older files need it to pick their record layout.
    metadata = cass_util.find_and_parse_metadata(data_dir)
    fw_ver = metadata.get("firmware_version") if metadata else None

    for file in sorted(Path(data_dir).glob("*.bin")):
        data = cass_util.process_stream_file(file, fw_ver)
        accel = data.imu["S"].accel
        plt.style.use("ggplot")

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
        for ax, axis in zip(axs, "xyz"):
            ax.plot(accel["t"], accel[axis])
            ax.set_title(f"Internal IMU - {axis.upper()} Acceleration {file.name} ({data.fw_version})")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("accel [mg]")
            ax.tick_params(axis="x", labelbottom=True)

        plt.tight_layout()
        plt.show()


def test_delete():
    print(cass_util.list_files())
    bSuccess = cass_util.delete_all_files(prompt_user=True)
    if bSuccess:
        print("Success!!!")
    print(cass_util.list_files())


if __name__ == "__main__":
    data_dir = download_data()
    print(data_dir)
    if not data_dir:
        raise SystemExit("No files on the device.")
    plot_internal_imu_data(data_dir)
    test_delete()
