# 🚀 cass_logger_dev 🚀

Python package for developers to interface with the Cass Logger. This package allows users to perform simple operations on the logger and download/process data.

## 🛠️ How to Use

1. Create a virtual environment:

   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install the package and its dependencies:

   ```bash
   pip install -e .
   ```

3. Explore usage examples in the `examples/` folder — you’ll find scripts you can run to interact with the logger and test functionality:

   | Script | Description |
   |--------|-------------|
   | `examples/load_and_plot_ex.py` | Load a pre-downloaded `.bin` file from `examples/data/` and plot potentiometer (fork/shock) data |
   | `examples/download_and_plot_ex.py` | Download data from a connected device, then plot internal IMU (accelerometer) data |
   | `examples/load_and_plot_320g_ex.py` | Load a block stream file (LSM6DSV320X variants) from `examples/data/` and plot high-g/low-g acceleration, gyro and magnetometer data |
   | `examples/live_view_ex.py` | Firmware bench tool: poll live snapshots at `--rate` Hz and print every channel plus error counts (needs firmware 0.12+) |

   **macOS / Linux:**
   ```bash
   python3 examples/load_and_plot_ex.py
   python3 examples/download_and_plot_ex.py
   ```

   **Windows:**
   ```cmd
   python examples\load_and_plot_ex.py
   python examples\download_and_plot_ex.py
   ```

## 📋 Logger Operations

All operations are available through `CassCommands` in `cass_logger_dev/cass_commands.py`. Serial ports are opened automatically on first use.

```python
from cass_logger_dev.cass_commands import CassCommands
cass_utils = CassCommands()
```

### File Management

| Operation | Method | Description |
|-----------|--------|-------------|
| List files | `cass_utils.list_files()` | Returns filenames stored on the device SD card |
| List file sizes | `cass_utils.list_file_sizes()` | Returns file sizes in bytes, in the same order as `list_files()` |
| Download all | `cass_utils.download_all()` | Downloads all files to a timestamped local directory and writes a `metadata.txt` |
| Delete all | `cass_utils.delete_all_files()` | Deletes all files from the SD card (pass `prompt_user=True` to confirm first) |

### Data Processing

| Operation | Method | Description |
|-----------|--------|-------------|
| Parse a logger file (default) | `CassCommands.process_stream_file(path, fw_ver=None)` | Any `.bin` file. Returns a `StreamData`: `records` (the 500 Hz DataFrame: analog channels, plus the polled I2C ICM-20948 on `i2c_1` / `i2c_2`) and `imu` with per-sensor DataFrames per IMU: `"S"` the internal ICM-20948, `"C"` / `"E"` the port IMUs. Firmware 0.13+ files describe themselves; older files need `fw_ver` (e.g. from `metadata.txt`), and their IMUs come from the records, one sample each |
| Parse records only | `CassCommands.process_data_file(path)` | The 500 Hz records as a DataFrame with a `t` (seconds) column. Pass `fw_ver` for pre-0.13 files from I2C variants. From 0.13 the IMUs aren't in the records, so this has the analog channels only |
| Parse FIT file | `CassCommands.process_fit_file(dir, filename)` | Parses a `.fit` file into `(df_session, df_record)` DataFrames |
| Find metadata | `CassCommands.find_and_parse_metadata(dir)` | Searches a directory for `metadata.txt` and returns firmware version and device ID |

#### Block stream files

From firmware 0.13 every variant writes a stream of tagged blocks: the 500 Hz record is one block type, and each IMU's FIFO readings come in their own blocks with a time for every sample, at the IMU's own rate (internal ICM-20948 ~1.13 kHz, ICM-45686 ~819 Hz, LSM6DSV320X ~937 Hz). The spec is `docs/stream-format.md` in cass-logger-fw; the parser is `cass_logger_dev/stream_format.py`.

```python
data = CassCommands.process_stream_file("LOG_1793290048.bin")
data.records                 # 500 Hz analog channels
data.imu["S"]                # the internal SPI ICM-20948
imu = data.imu["C"]          # the IMU on port C (imu.device names it)
imu.accel, imu.gyro          # t, tmicros, x, y, z [mg] / [dps], t_edge, t_fit
imu.accel_hg                 # LSM6DSV320X high-g accelerometer [mg] (±320 g); empty for the ICM-45686
imu.mag, imu.timestamp       # [µT], raw IMU timestamp counter
imu.events                   # FIFO overruns, failed/late reads, timing lost/re-established
imu.reads                    # each FIFO read's time and entry count (format 2+)
```

All times share the records' zero. How good a sample's time is: `t_edge` is True when it's the sample's INT1 (data-ready) edge (LSM6DSV320X); `t_fit` when it comes from a converged fit of the IMU's own sample clock (ICM-20948, ICM-45686, which have no INT pin wired; ~1 s after recording starts). With neither, the time is an estimate; `events` shows where timing was lost.

### Device Configuration

| Operation | Method | Description |
|-----------|--------|-------------|
| Set RTC time | `cass_utils.set_RTC_time()` | Syncs the device RTC to the current UTC time |
| Get RTC time | `cass_utils.get_RTC_time()` | Reads the current RTC time from the device |
| Get firmware version | `cass_utils.get_fw_ver()` | Returns the firmware version string (e.g. `"0.13-std"`, `"0.13-i2c_2"`, `"0.13-i2c_320g"`) |
| Get device ID | `cass_utils.get_device_ID()` | Reads the device identifier stored in EEPROM |
| Set device ID | `cass_utils.put_device_ID(id)` | Writes a device identifier string to EEPROM |
| Get RTC install time | `cass_utils.get_rtc_install_timestamp()` | Reads the RTC battery install timestamp from EEPROM |
| Set RTC install time | `cass_utils.put_rtc_install_timestamp()` | Writes the RTC battery install timestamp to EEPROM (defaults to now) |
| Get logger status | `cass_utils.get_status()` | Status report for the most recent recording, as a dict (MB written, duration, write times, `buffer_overflows`). Run after stopping a recording with USB connected; `buffer_overflows` should be 0 (it's `None` on firmware before 0.12) |

### Live View

| Operation | Method | Description |
|-----------|--------|-------------|
| Single snapshot | `cass_utils.get_snapshot()` | Reads every sensor channel once and returns a `Snapshot` (raw ADC counts and IMU values). Needs firmware 0.12+; older firmware raises `TimeoutError`. From 0.13 the IMU readings arrive alongside the record and are decoded into the same column names (`gx`…`Tz`, `gx_i2c_c`…); `live_columns(fw_ver)` lists them |

Live view (continuous polling with a chart) is driven from the Cass Logger app, not from this package. This package provides the protocol: the frame parser and decoder in `cass_logger_dev/live.py`, and `snapshot()` for repeated polling. See `docs/live-view-plan.md` for the spec.

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```
