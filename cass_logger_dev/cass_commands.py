"""
High-level interface for communicating with the Cass Logger over serial.

Typical usage
-------------
    cass = CassCommands()
    files = cass.list_files()
    dir_path = cass.download_all()
    df = CassCommands.process_data_file(dir_path, files[0])

Notes
-----
- Requires two USB serial ports (data + command). Main testing done on macOS/Linux, limited testing on Windows.
"""

import os
import time
import serial
import fitdecode
import numpy as np
from pathlib import Path
import pandas as pd
import serial.tools.list_ports
import datetime
import warnings
from .firmware_structs import (
    FIRMWARE_DTYPES,
    COLUMN_ORDERS,
)
from typing import Optional, Union, Dict, List
import re
import platform


class CassCommands:
    """
    Interface for communicating with Cass data logger devices over serial.

    Manages dual serial connections (one for data, one for commands),
    file management on the device's SD card, device configuration
    (RTC, device ID, firmware version), and offline data processing.
    
    This class provides cross-platform support for Windows, macOS, and Linux.
    
    Windows Usage (less tested):
        If auto-detection fails, you can manually specify ports:
        1. Create instance: logger = CassCommands()
        2. List ports: logger.list_available_ports()
        3. Set ports manually: logger.set_manual_serial_ports('COM3', 'COM4')
        4. Then use normally: logger.download_all()
    
    If you encounter issue on Windows, use the diagnose_windows_ports()
    method to get detailed information about each COM port, which can help
    identify the correct ports to use.
    
    Note: On Windows, make sure the device drivers are installed and the device
    appears in Device Manager under "Ports (COM & LPT)".
    """
    def __init__(self):
        self._ser_data = None
        self._ser_command = None
        self.reset_buff_used = False
        self._manual_ports = None           # For manual port specification

    # --- Properties ---

    @property
    def ser_data(self):
        """Serial port used for data transfer. Opens lazily on first access."""
        if self._ser_data is None:
            self._establish_serial()
        if not self._ser_data.is_open:
            self._ser_data.open()
        return self._ser_data

    @ser_data.setter
    def ser_data(self, port_name):
        """Open and assign the data serial port by device path.

        Parameters
        ----------
        port_name : str
            OS device path to the serial port (e.g. '/dev/cu.usbmodem1').
        """
        baud_rate = 9600
        ser = serial.Serial(port_name, baud_rate)
        if not ser.is_open:
            ser.open()
        ser.reset_input_buffer()
        ser.flush()
        self._ser_data = ser

    @property
    def ser_command(self):
        """Serial port used for sending commands. Opens lazily on first access."""
        if self._ser_command is None:
            self._establish_serial()
        if not self._ser_command.is_open:
            self._ser_command.open()
        return self._ser_command

    @ser_command.setter
    def ser_command(self, port_name):
        """Open and assign the command serial port by device path.

        Parameters
        ----------
        port_name : str
            OS device path to the serial port (e.g. '/dev/cu.usbmodem2').
        """
        baud_rate = 9600
        ser = serial.Serial(port_name, baud_rate)
        if not ser.is_open:
            ser.open()
        ser.reset_output_buffer()
        ser.flush()
        self._ser_command = ser

    # --- Public Instance Methods ---

    def get_serial_ports(self):
        """Cross-platform method to find the two logger serial ports.
        
        Returns
        -------
        list of str or None
            Two device paths (e.g. ['/dev/cu.usbmodem1', '/dev/cu.usbmodem2']),
            or None if exactly two USB modem ports are not found.
        """
        ports = serial.tools.list_ports.comports()
        logger_ports = []
        
        system = platform.system().lower()
        
        if system == "windows":
            # look for COM ports with specific characteristics
            for port in ports:
                if port.vid is not None:  # USB device
                    # add common USB-to-serial converter VIDs or device descriptions
                    if (
                        port.vid == 0x16C0
                        or "teensy" in port.description.lower()
                        or "usb serial" in port.description.lower()
                        or "ch340" in port.description.lower()
                        or "cp210" in port.description.lower()
                        or "ftdi" in port.description.lower()
                    ):
                        logger_ports.append(port.device)
        else:
            # macOS/Linux
            logger_ports = [port.device for port in ports if "usbmodem" in port.device]
        
        if len(logger_ports) != 2:
            print(f"Expected 2 serial ports, found {len(logger_ports)}: {logger_ports}")
            print("Available ports:")
            for port in ports:
                print(f"  {port.device}: {port.description}")
                if hasattr(port, 'vid') and port.vid:
                    print(f"    VID: 0x{port.vid:04X}, PID: 0x{port.pid:04X}")
            
            # Windows-specific troubleshooting tips
            if system == "windows":
                print("\nWindows troubleshooting:")
                print("1. Ensure device drivers are installed")
                print("2. Check Device Manager > Ports (COM & LPT)")
                print("3. Use list_available_ports() and set_manual_serial_ports() if needed")
                print("4. Make sure no other applications are using the COM ports")
            
            return None
        else:
            return logger_ports

    def set_manual_serial_ports(self, data_port: str, command_port: str):
        """Manually specify the serial ports if auto-detection fails.
        
        Args:
            data_port: Port name for data communication (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux)
            command_port: Port name for command communication (e.g., 'COM4' on Windows, '/dev/ttyACM1' on Linux)
        """
        try:
            # Test that both ports can be opened
            test_data = serial.Serial(data_port, 9600, timeout=1)
            test_command = serial.Serial(command_port, 9600, timeout=1)
            test_data.close()
            test_command.close()
            
            # Store the ports for use by _establish_serial
            self._manual_ports = [data_port, command_port]
            print(f"Manual ports set: data={data_port}, command={command_port}")
            return True
        except serial.SerialException as e:
            print(f"Error accessing ports: {e}")
            return False

    def list_available_ports(self):
        """List all available serial ports for manual selection."""
        ports = serial.tools.list_ports.comports()
        print("Available serial ports:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device}: {port.description}")
            if hasattr(port, 'vid') and port.vid:
                print(f"      VID: 0x{port.vid:04X}, PID: 0x{port.pid:04X}")
        return ports

    def diagnose_windows_ports(self):
        """Windows-specific diagnostic helper to identify potential logger ports."""
        if platform.system().lower() != "windows":
            print("This diagnostic is only for Windows systems.")
            return
            
        ports = serial.tools.list_ports.comports()
        print("=== Windows Serial Port Diagnostics ===")
        print(f"Found {len(ports)} total ports:")
        
        teensy_candidates = []
        usb_candidates = []
        
        for port in ports:
            print(f"\nPort: {port.device}")
            print(f"  Description: {port.description}")
            print(f"  Manufacturer: {port.manufacturer}")
            
            if hasattr(port, 'vid') and port.vid:
                print(f"  VID: 0x{port.vid:04X}")
                print(f"  PID: 0x{port.pid:04X}")
                
                # Check for known Teensy identifiers
                if port.vid == 0x16C0:  # VOTI (Van Ooijen Technische Informatica)
                    teensy_candidates.append(port.device)
                    print("  *** TEENSY DEVICE DETECTED ***")
                elif port.vid == 0x239A:  # Adafruit
                    teensy_candidates.append(port.device)
                    print("  *** POSSIBLE TEENSY/MICROCONTROLLER ***")
                elif "teensy" in port.description.lower():
                    teensy_candidates.append(port.device)
                    print("  *** TEENSY IN DESCRIPTION ***")
                else:
                    usb_candidates.append(port.device)
            else:
                print("  No USB VID/PID (might be built-in port)")
        
        print(f"\n=== Summary ===")
        print(f"Teensy candidates: {teensy_candidates}")
        print(f"Other USB devices: {usb_candidates}")
        
        if len(teensy_candidates) == 2:
            print(f"\nTry: logger.set_manual_serial_ports('{teensy_candidates[0]}', '{teensy_candidates[1]}')")
        elif len(teensy_candidates) > 0:
            print(f"Found {len(teensy_candidates)} potential devices, but need exactly 2.")
        else:
            print("No obvious Teensy devices found. Check Device Manager.")
            
    def set_RTC_time(self):
        """Set the device RTC to the current UTC time.

        Returns
        -------
        bool
            True if the device confirmed the time was set, False otherwise.
        """
        current_time = datetime.datetime.now(datetime.timezone.utc)
        print("Current time: ", current_time)
        # Format time string
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("Time string: ", time_string)
        # Add termination character
        time_string += "x"

        # Send command to set RTC time
        self.ser_command.write(b"e")

        # Send time string
        self.ser_data.write(time_string.encode("utf-8"))

        # Wait for confirmation and unix time
        unix_time = ""
        while True:
            char = self.ser_data.read(1).decode("utf-8")
            if char == "x":
                break
            unix_time += char

        self._close_serial()

        if unix_time:
            print(f"RTC time set successfully. Unix time: {unix_time}")
            return True
        else:
            print("Failed to set RTC time")
            return False

    def get_RTC_time(self):
        """Read the current RTC time from the device.

        Returns
        -------
        str
            Unix timestamp string as returned by the device.
        """
        self._flush_all()

        self.ser_command.write(b"h")
        unix_time = ""
        while True:
            char = self.ser_data.read(1).decode("utf-8")
            if char == "x":
                break
            unix_time += char
        return unix_time

    def list_files(self):
        """List all files stored on the device SD card.

        Returns
        -------
        list of str
            Filenames on the device (one per entry).
        """
        self._open_serial()
        self._flush_all()

        self.ser_command.write(b"l")  # list files

        result = b""
        while b"xxx" not in result:
            if self.ser_data.in_waiting > 0:
                result += self.ser_data.read(self.ser_data.in_waiting)
        result = result.decode("utf-8")

        # self._close_serial()
        return result.splitlines()[:-1]

    def list_file_sizes(self):
        """Return the binary size (in bytes) of each file on the device.

        Returns
        -------
        list of int
            File sizes in the same order as list_files().
        """
        self._open_serial()
        self._flush_all()

        files = self.list_files()  # list files
        num_files = len(files)

        self.ser_command.write(b"z")  # list file sizes

        my_file_sizes = []
        my_file_sizes = [
            int(self.ser_data.read_until(b"\n").decode("utf-8").strip(), 2)
            for i in range(num_files)
        ]

        # self._close_serial()

        return my_file_sizes

    def delete_all_files(self, prompt_user=False):
        """Delete all files from the device SD card.

        Parameters
        ----------
        prompt_user : bool, optional
            If True, ask for confirmation before deleting (default False).

        Returns
        -------
        bool or int
            True on success, False if files remain after deletion,
            0 if the user cancelled.
        """
        if prompt_user:
            user_input = input("Are you sure you want to delete all files? (y/n): ")
            if user_input.lower() != "y":
                print("Operation cancelled by user.")
                return 0
        [self._delete_file(filename) for filename in self.list_files()]

        self._close_serial()

        if len(self.list_files()) == 0:
            return True
        else:
            warnings.warn("Warning: error deleting files.")
            return False

    def read_file(self, filename, file_size):
        """Download a single file from the device as raw bytes.

        Reads the file in 5120-byte SD buffer chunks. Uses _reset_buff to
        recover from stalled transfers.

        Parameters
        ----------
        filename : str
            Name of the file on the device.
        file_size : int
            Size of the file in bytes (as returned by list_file_sizes).

        Returns
        -------
        list of int
            Raw byte values of the file contents.
        """
        filename_term = filename + "x"
        filename_term = bytes(filename_term, "utf-8")

        sd_buff_size = 5120
        num_buffs = file_size / sd_buff_size
        fractional_buffs = num_buffs - int(num_buffs)
        # TODO: add fractional buffer transfer at end
        if file_size % sd_buff_size != 0:
            num_buffs = file_size // sd_buff_size  # skip last incomplete sd_buffer
        num_buffs = int(num_buffs)
        bytes_received = []

        self.ser_command.write(b"o")  # open target file
        self.ser_data.write(filename_term)

        sd_buff = bytes()  # empty byte array for current buffer
        sd_byte_idx = 0  # byte index in current buffer
        retry_loop = False
        i = 0  # current buffer index
        while i < num_buffs:
            # read each buffer
            self.ser_command.write(b"t")  # send command for Teensy to send buffer
            self.ser_command.flush()  # wait until command is sent
            time_in_buffer = time.monotonic()

            sd_byte_idx = 0
            retry_loop = False
            while sd_byte_idx < sd_buff_size:
                # read each byte in buffer
                num_read = min(  # number of bytes to read
                    int(self.ser_data.in_waiting),  # number of bytes in serial buffer
                    sd_buff_size - sd_byte_idx,  # number of bytes remaining in buffer
                )
                if num_read > 0:
                    bytesIn = self.ser_data.read(num_read)  # incoming buffer
                    sd_byte_idx += num_read
                    sd_buff += bytesIn
                    time_in_buffer = time.monotonic()
                elif num_read == 0 and (time.monotonic() - time_in_buffer > 0.1):
                    # NOTE: why does this condition represent a data corruption?
                    # reset the position in the file to (curr_position - sd_byte_idx)
                    # while (self.ser_data.in_waiting) > 0:
                    #     self.ser_data.read(self.ser_data.in_waiting)  # clear serial buffer
                    #     print("Clearing serial buffer...")              # doesn't seem to be doing anything
                    self.ser_data.reset_input_buffer()  # clear serial buffer before initiating reset
                    # self.ser_data.reset_output_buffer()
                    buff_success = self._reset_buff((i) * sd_buff_size, filename)
                    sd_buff = []
                    retry_loop = True
                    self.reset_buff_used = True

                    break

            if retry_loop:
                i -= 1

            i += 1
            bytes_received.extend(sd_buff)
            sd_buff = bytes()
            sd_byte_idx = 0

        # DEBUG
        expected_byte_number = num_buffs * sd_buff_size
        number_buffs_off = expected_byte_number - len(bytes_received)
        print(f"Number of bytes short = {number_buffs_off} ({filename})")

        self.ser_command.write(b"c")  # close target file
        self._close_serial()
        return bytes_received

    def bytes_to_file(
        self, my_bytes, filename, filepath="tmp_{}".format(int(time.time()))
    ):
        """Write raw bytes to a local file, creating the directory if needed.

        Parameters
        ----------
        my_bytes : bytes or list of int
            Data to write.
        filename : str
            Output filename within filepath.
        filepath : str, optional
            Directory to write into. Defaults to a timestamped tmp directory.

        Returns
        -------
        str
            The filepath that was written to.
        """
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        full_filepath = Path(filepath, filename)
        with open(full_filepath, "wb") as f:
            f.write(bytes(my_bytes))
        return filepath

    def download_all(self):
        """Download all files from the device and write a metadata file.

        Files are saved to a timestamped directory (tmp_<unix>). A
        metadata.txt file containing the firmware version and device ID
        is written alongside them.

        Returns
        -------
        str
            Path to the directory containing the downloaded files,
            or an empty list if no files were found on the device.
        """
        my_filenames = self.list_files()
        my_file_sizes = self.list_file_sizes()
        if not len(my_filenames):
            return []
        dir_name = "tmp_{}".format(int(time.time()))

        filepaths = [
            self.bytes_to_file(self.read_file(filename, file_size), filename, dir_name)
            for filename, file_size in zip(my_filenames, my_file_sizes)
        ]

        self._flush_all()
        # write metadata
        fw_ver = self.get_fw_ver()
        device_id = self.get_device_ID()
        md_path = Path(dir_name, "metadata.txt")
        with open(md_path, "w") as meta_file:
            meta_file.write(f"Firmware Ver: {fw_ver}\n")
            meta_file.write(f"Device ID: {device_id}\n")

        return filepaths[-1]

    def put_device_ID(self, device_ID):
        """Write a device identifier string to EEPROM.

        Parameters
        ----------
        device_ID : str
            Identifier to write (must not contain 'x', used as terminator).

        Returns
        -------
        bool
            True if the device echoed back the correct ID, False otherwise.
        """
        self.ser_command.write(b"p")    # eeprom put
        device_ID_orig = device_ID
        device_ID += "x"
        print("device_ID to write = ", device_ID)
        self.ser_data.write(bytes(device_ID, "utf-8"))
        while self.ser_data.in_waiting < len(device_ID) - 1:
            pass

        check_device_ID = self.ser_data.read_all().decode("utf-8")
        print("Device ID is: ", check_device_ID)

        self._close_serial()
        if check_device_ID == device_ID_orig:  # TODO: error handling
            return True
        else:
            return False

    def get_device_ID(self):
        """Read the device identifier from EEPROM.

        Returns
        -------
        str
            The stored device ID string.
        """
        self._flush_all()

        self.ser_command.write(b"g")
        device_ID = b""
        while b"x" not in device_ID:
            if self.ser_data.in_waiting > 0:
                device_ID += self.ser_data.read(self.ser_data.in_waiting)
        device_ID = device_ID.decode("utf-8")

        self._close_serial()

        return device_ID[:-1]

    def put_rtc_install_timestamp(self, unix_install=None):
        """Write the RTC battery install timestamp to EEPROM.

        Parameters
        ----------
        unix_install : int, optional
            Unix timestamp to store. Defaults to the current time.

        Returns
        -------
        bool
            True if the device echoed back the correct timestamp.
        """
        if unix_install is None:
            unix_install = int(time.time())
        self.ser_command.write(b"j")  # eeprom put UNIX timestamp

        unix_install_orig = str(unix_install)  # keep original as string
        unix_install = unix_install_orig + "x"  # add terminator

        print("unix_RTC_batt install time to write = ", unix_install)
        self.ser_data.write(unix_install.encode("utf-8"))

        while self.ser_data.in_waiting < len(unix_install) - 1:
            pass

        check_unix_install = self.ser_data.read_all().decode("utf-8").strip("x\n\r ")
        print("Unix install is: ", check_unix_install)

        try:
            ts_int = int(check_unix_install)
            dt = datetime.datetime.fromtimestamp(ts_int)
            print("Datetime install is:", dt.strftime("%Y-%m-%d %H:%M:%S"))
        except ValueError:
            print("Error: could not parse timestamp")

        self._close_serial()
        return check_unix_install == unix_install_orig

    def get_rtc_install_timestamp(self):
        """Read the RTC battery install timestamp from EEPROM.

        Returns
        -------
        str
            The stored Unix timestamp as a string.
        """
        self._flush_all()
        self._open_serial()

        self.ser_command.write(b"i")
        rtc_install = b""
        while b"x" not in rtc_install:
            if self.ser_data.in_waiting > 0:
                rtc_install += self.ser_data.read(self.ser_data.in_waiting)
        rtc_install = rtc_install.decode("utf-8")
        rtc_install = rtc_install[:-1]
        self._close_serial()

        dt = datetime.datetime.fromtimestamp(int(rtc_install))
        print("Datetime install is:", dt.strftime("%Y-%m-%d %H:%M:%S"))

        return rtc_install

    def get_fw_ver(self):
        """Read the firmware version string from the device.

        Returns
        -------
        str
            Firmware version string (e.g. "std", "i2c_1", "i2c_2").
        """
        self._flush_all()

        self.ser_command.write(b"a")
        fw_ver = b""
        while b"x" not in fw_ver:
            if self.ser_data.in_waiting > 0:
                fw_ver += self.ser_data.read(self.ser_data.in_waiting)
        fw_ver = fw_ver.decode("utf-8")
        self._close_serial()
        return fw_ver[:-1]

    # --- Static and Class Methods

    # Maps firmware variant suffix to dtype key. Variants that share a
    # struct layout map to the same dtype key.
    VARIANT_DTYPE_MAP = {
        "std":               "std",
        "i2c_1":             "i2c_1",
        "i2c_45686":         "i2c_1",   # same 108-byte layout as i2c_1
        "i2c_45686_9dof":    "i2c_1",   # same 108-byte layout as i2c_1 (legacy 9dof naming - it's now just i2c_45686)
        "i2c_2":             "i2c_2",
        "i2c_45686_2":       "i2c_2",   # same 136-byte layout as i2c_2
    }

    @classmethod
    def get_dtype_key(cls, fw_ver: str) -> str:
        """Map a full firmware version string to its dtype/struct-layout key.

        Parameters
        ----------
        fw_ver : str
            Full firmware version string, e.g. "0.10-i2c_2" or "0.09-std".
            Format is "<version>-<variant>" or just "<version>".

        Returns
        -------
        str
            Dtype key into ``FIRMWARE_DTYPES``/``COLUMN_ORDERS``
            (e.g. "std", "i2c_1", "i2c_1_legacy", "i2c_2").

        Raises
        ------
        ValueError
            If fw_ver does not map to a known firmware dtype.
        """
        # Extract version and variant from strings like "0.10-i2c_2" or "0.09"
        match = re.match(r"(\d+\.\d+)(?:-(.+))?$", fw_ver.strip())
        if match:
            version = match.group(1)
            version_tuple = tuple(int(p) for p in version.split("."))
            legacy_version = version_tuple[0] == 0 and version_tuple[1] < 10
            variant = match.group(2) or "std"
        else:
            raise ValueError(f"Invalid firmware version format: '{fw_ver}'")

        # Handle legacy i2c_1 firmware versions (<0.10)
        if variant == "i2c_1" and legacy_version:
            return "i2c_1_legacy"

        dtype_key = cls.VARIANT_DTYPE_MAP.get(variant)
        if dtype_key is None:
            raise ValueError(f"Unsupported firmware variant '{variant}' (from fw_ver='{fw_ver}')")
        return dtype_key

    @classmethod
    def process_data_file(cls, full_filename: Union[str, Path], fw_ver="0.09-std"):
        """Parse a binary sensor data file into a pandas DataFrame.

        The firmware version string determines which NumPy dtype is used for
        parsing. The tmicros column is zero-referenced and a 't' column
        (seconds, float64) is inserted.

        Parameters
        ----------
        full_filename : str
            Path to the binary file.
        fw_ver : str, optional
            Full firmware version string. Must contain firmware version (i.e. 0.09, 0.10)
            and, optionally, firmware variant (e.g. "i2c_2", "i2c_1"); defaults to "std"
            if omitted. Format is "<version>-<variant>" (e.g. "0.10-i2c_2") or just
            "<version>" (e.g. "0.09-std"). Defaults to "0.09-std" (std firmware) if omitted.

        Returns
        -------
        pd.DataFrame
            Parsed sensor data with columns ordered per COLUMN_ORDERS.

        Raises
        ------
        ValueError
            If fw_ver does not map to a known firmware dtype.
        """
        full_filename = Path(full_filename)

        dtype_key = cls.get_dtype_key(fw_ver)

        try:
            dt = FIRMWARE_DTYPES[dtype_key]()
            column_order = COLUMN_ORDERS[dtype_key]
        except KeyError:
            raise ValueError(f"Unsupported firmware version: {fw_ver}")

        data = np.fromfile(full_filename, dtype=dt)
        df = pd.DataFrame(data)

        if (df["tmicros"] < 0).any():
            df["tmicros"] = df["tmicros"].astype(np.float64)
            df["tmicros"] = cls.handle_tmicros_rollover(df["tmicros"])

        df["tmicros"] -= df["tmicros"].iloc[0]
        df.insert(1, "t", df["tmicros"] * 1e-6)

        # Only reorder columns that exist in this firmware's dtype
        df = df[[col for col in column_order if col in df.columns]]

        return df

    @staticmethod
    def find_and_parse_metadata(
        dir_path: str,
        filename: str = "metadata.txt",
        recursive: bool = True,
        first_only: bool = True,
    ) -> Union[Dict[str, Optional[str]], List[Dict[str, Optional[str]]], None]:
        """Search for metadata files and return their parsed contents.

        Parameters
        ----------
        dir_path : str
            Root directory to search.
        filename : str, optional
            Filename to look for (case-insensitive, default "metadata.txt").
        recursive : bool, optional
            Search subdirectories if True (default True).
        first_only : bool, optional
            Return only the first match if True (default True).

        Returns
        -------
        dict, list of dict, or None
            Single parsed dict if first_only=True, list of dicts if False,
            or None if no matching file was found. Each dict contains
            "firmware_version" and "device_id" keys.
        """
        files = CassCommands._find_metadata_files(
            dir_path, filename=filename, recursive=recursive
        )
        if not files:
            return None

        parsed = []
        for f in files:
            try:
                parsed.append(CassCommands._parse_metadata_file(str(f)))
            except Exception as exc:
                # skip files that fail to read/parse; optionally log the error
                parsed.append({"error": f"failed to parse {f}: {exc}"})

        if first_only:
            return parsed[0]
        return parsed

    @staticmethod
    def handle_tmicros_rollover(col):
        """Reconstruct a monotonic timestamp column from a rolled-over microsecond counter.

        Assumes a constant sample interval derived from the first two samples.

        Parameters
        ----------
        col : array-like
            tmicros column values (may contain negative values from rollover).

        Returns
        -------
        np.ndarray
            Monotonically increasing int64 timestamp array starting from 0.
        """
        step_size = col[1] - col[0]
        new_col = np.arange(0, len(col) * step_size, step_size, dtype=np.int64)
        return new_col

    @staticmethod
    def process_fit_file(filepath, filename):
        """Parse a FIT file into session and record DataFrames.

        Parameters
        ----------
        filepath : str
            Directory containing the FIT file.
        filename : str
            Name of the FIT file.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            (df_session, df_record) — one row per session/record frame.
        """
        my_path = str(Path(filepath, filename))
        df_record = pd.DataFrame()
        df_session = pd.DataFrame()
        with fitdecode.FitReader(my_path) as fit:
            for frame in fit:
                if frame.frame_type == fitdecode.FIT_FRAME_DATA:
                    if frame.name == "record":
                        frame_data = {field.name: field.value for field in frame.fields}
                        frame_df = pd.DataFrame([frame_data])
                        df_record = pd.concat([df_record, frame_df], ignore_index=True)
                    elif frame.name == "session":
                        frame_data = {field.name: field.value for field in frame.fields}
                        frame_df = pd.DataFrame([frame_data])
                        df_session = pd.concat(
                            [df_session, frame_df], ignore_index=True
                        )
        return df_session, df_record

    # --- Private Methods ---

    def _establish_serial(self, baud_rate=9600):
        """Open both serial ports and identify which is data vs. command.

        Sends a handshake byte to each port and uses the device response
        to determine the correct assignment. Raises on timeout or no response.

        Parameters
        ----------
        baud_rate : int, optional
            Serial baud rate (default 9600).

        Raises
        ------
        ValueError
            If no logger serial ports are detected.
        TimeoutError
            If the device does not respond within 3 seconds.
        RuntimeError
            If neither port returns the expected handshake response.
        """
        # Use manual ports if set, otherwise auto-detect
        if self._manual_ports:
            serial_ports = self._manual_ports
            print(f"Using manual ports: {serial_ports}")
        else:
            serial_ports = self.get_serial_ports()
            
        if serial_ports is None:
            raise ValueError(
                "No logger detected! Make sure it's plugged in and powered on. "
                "On Windows, you may need to use set_manual_serial_ports() or list_available_ports() "
                "to manually specify the correct COM ports."
            )
        
        # Create serial connections with platform-appropriate settings
        try:
            # On Windows, sometimes need to set additional parameters for stability
            if platform.system().lower() == "windows":
                ser_data = serial.Serial(
                    serial_ports[0], 
                    baud_rate, 
                    timeout=1,
                    write_timeout=1,
                    rtscts=False,
                    dsrdtr=False
                )
                ser_command = serial.Serial(
                    serial_ports[1], 
                    baud_rate, 
                    timeout=1,
                    write_timeout=1,
                    rtscts=False, 
                    dsrdtr=False
                )
            else:
                ser_data = serial.Serial(serial_ports[0], baud_rate)
                ser_command = serial.Serial(serial_ports[1], baud_rate)
        except serial.SerialException as e:
            raise ValueError(f"Failed to open serial ports {serial_ports}: {e}")
            
        if not ser_data.is_open:
            ser_data.open()
        if not ser_command.is_open:
            ser_command.open()

        ser_data.reset_input_buffer()
        ser_command.reset_output_buffer()

        self._flush_ser_port(ser_data)
        self._flush_ser_port(ser_command)

        ser_command.write(b"u")
        ser_data.write(b"u")

        timer = time.monotonic()
        while ser_data.in_waiting < 1 and ser_command.in_waiting < 1:
            if time.monotonic() - timer > 3:
                raise TimeoutError("Timeout waiting for serial response from device.")

        data_response = ser_data.read(ser_data.in_waiting).decode("utf-8")
        command_response = ser_command.read(ser_command.in_waiting).decode("utf-8")

        if command_response == "x":
            ser_data, ser_command = ser_command, ser_data
        elif data_response == "x":
            pass
        else:
            raise RuntimeError("ERROR! No response from teensy!")

        self._flush_ser_port(ser_data)
        self._flush_ser_port(ser_command)

        self._ser_data = ser_data
        self._ser_command = ser_command

    def _flush_ser_port(self, ser_obj):
        """Flush the output buffer and clear the input buffer of a serial port.

        Parameters
        ----------
        ser_obj : serial.Serial
            The serial port instance to flush.
        """
        try:
            ser_obj.reset_input_buffer()
            ser_obj.flush()
        except OSError:
            # macOS can mark the fd stale (ENXIO) while pyserial still reports
            # is_open=True. Close and reopen the same port to recover.
            ser_obj.close()
            ser_obj.open()
            ser_obj.reset_input_buffer()
            ser_obj.flush()

    def _flush_all(self):
        self._flush_ser_port(self.ser_data)
        self._flush_ser_port(self.ser_command)

    def _reset_buff(self, reset_pos, filename):
        START_MARKER = b"\xff\xfe\xfd"
        END_MARKER = b"\xfd\xfe\xff"

        print(f"IN RESET BUFF, file: {filename} at pos: {reset_pos}")

        self.ser_command.write(b"n")  # send reset buffer command
        self._flush_ser_port(self.ser_command)  # wait until command is sent
        # self.ser_command.flush()

        time.sleep(0.05)  # NOTE: not sure if needed

        reset_pos = START_MARKER + str(reset_pos).encode("utf-8") + END_MARKER
        self.ser_data.write(reset_pos)  # send reset idx
        self._flush_ser_port(self.ser_data)  # wait until idx is sent
        # self.ser_data.flush()

        # Validate that the correct position was set
        self.ser_data.read_until(START_MARKER)
        return_position = self.ser_data.read_until(END_MARKER)
        return_position = return_position[: -len(END_MARKER)]
        try:
            return_position = return_position.decode("utf-8")
        except UnicodeDecodeError:
            print("Error decoding return position from device.")
        print(return_position)

        # Make sure there isn't any leftover data in the serial buffers
        # self.ser_data.reset_input_buffer()
        while self.ser_data.in_waiting > 0:
            self.ser_data.read(self.ser_data.in_waiting)
            print("Clearing serial buffer after...")
        self._flush_all()
        return True

    def _delete_file(self, filename):
        """Send a delete command for a single file on the device.

        Parameters
        ----------
        filename : str
            Name of the file to delete.

        Returns
        -------
        bool
            True if the device confirmed deletion, False otherwise.
        """
        self.ser_command.reset_input_buffer()  # TODO: should this be a normal flush?
        self.ser_command.write(b"x")  # delete file

        filename_term = filename + "x"  # append terminator so firmware knows filename is complete
        self.ser_command.write(bytes(filename_term, "utf-8"))
        self.ser_command.flush()  # ensure bytes are transmitted before reading response

        b_success = self.ser_data.read_until(b"x")  # check for success
        b_success = int(b_success.decode("ascii").strip().strip("x"))
        if b_success:
            return True
        else:
            warnings.warn("Warning: error deleting file.")
            return False

    def _close_serial(self):
        """Close both serial port connections."""
        self.ser_data.close()
        self.ser_command.close()

    def _open_serial(self):
        """Open both serial port connections if they are not already open."""
        if not self.ser_data.is_open:
            self.ser_data.open()
        if not self.ser_command.is_open:
            self.ser_command.open()

    @staticmethod
    def _find_metadata_files(
        dir_path: str, filename: str = "metadata.txt", recursive: bool = True
    ) -> List[Path]:
        """Return a list of Path objects matching filename inside dir_path.

        Parameters
        ----------
        dir_path : str
            Root directory to search.
        filename : str, optional
            Target filename (case-insensitive, default "metadata.txt").
        recursive : bool, optional
            Search subdirectories if True (default True).

        Returns
        -------
        list of Path

        Raises
        ------
        FileNotFoundError
            If dir_path does not exist.
        NotADirectoryError
            If dir_path is not a directory.
        """
        root = Path(dir_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        name_lower = filename.lower()
        matches: List[Path] = []
        try:
            if recursive:
                for p in root.rglob("*"):
                    if p.is_file() and p.name.lower() == name_lower:
                        matches.append(p)
            else:
                for p in root.iterdir():
                    if p.is_file() and p.name.lower() == name_lower:
                        matches.append(p)
        except PermissionError:
            # optionally handle permission errors (skip folders you can't access)
            pass

        return matches

    @staticmethod
    def _parse_metadata_file(path: str) -> Dict[str, Optional[str]]:
        """Read a metadata.txt file and extract firmware version and device ID.

        Parameters
        ----------
        path : str
            Path to the metadata file.

        Returns
        -------
        dict
            Keys: "firmware_version" and "device_id" (either may be None if
            not found in the file).
        """
        txt = Path(path).read_text(encoding="utf-8")
        fw_match = re.search(
            r"Firmware\s*(?:Ver(?:\.|sion)?)\s*[:=]\s*([^\r\n]+)", txt, re.I
        )
        id_match = re.search(r"Device\s*ID\s*[:=]\s*([^\r\n]+)", txt, re.I)

        def _clean(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            s = s.strip()
            # remove surrounding quotes if any
            if (s.startswith('"') and s.endswith('"')) or (
                s.startswith("'") and s.endswith("'")
            ):
                s = s[1:-1].strip()
            return s

        return {
            "firmware_version": _clean(fw_match.group(1)) if fw_match else None,
            "device_id": _clean(id_match.group(1)) if id_match else None,
        }
