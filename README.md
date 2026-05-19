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
| Parse binary file | `CassCommands.process_data_file(path)` | Parses a `.bin` file into a pandas DataFrame with a `t` (seconds) column. Pass `fw_ver` if using an I2C firmware variant |
| Parse FIT file | `CassCommands.process_fit_file(dir, filename)` | Parses a `.fit` file into `(df_session, df_record)` DataFrames |
| Find metadata | `CassCommands.find_and_parse_metadata(dir)` | Searches a directory for `metadata.txt` and returns firmware version and device ID |

### Device Configuration

| Operation | Method | Description |
|-----------|--------|-------------|
| Set RTC time | `cass_utils.set_RTC_time()` | Syncs the device RTC to the current UTC time |
| Get RTC time | `cass_utils.get_RTC_time()` | Reads the current RTC time from the device |
| Get firmware version | `cass_utils.get_fw_ver()` | Returns the firmware version string (e.g. `"std"`, `"i2c_1"`, `"i2c_2"`) |
| Get device ID | `cass_utils.get_device_ID()` | Reads the device identifier stored in EEPROM |
| Set device ID | `cass_utils.put_device_ID(id)` | Writes a device identifier string to EEPROM |
| Get RTC install time | `cass_utils.get_rtc_install_timestamp()` | Reads the RTC battery install timestamp from EEPROM |
| Set RTC install time | `cass_utils.put_rtc_install_timestamp()` | Writes the RTC battery install timestamp to EEPROM (defaults to now) |
