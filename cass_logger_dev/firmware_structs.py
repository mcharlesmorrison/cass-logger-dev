"""
NumPy dtype definitions and column orderings for each Cass Logger firmware variant.

Exports
-------
FIRMWARE_DTYPES : dict
    Maps record layout keys ("std", "i2c_1", "i2c_2", "stream_1", ...) to their dtype constructor.
COLUMN_ORDERS : dict
    Maps firmware key strings to the preferred DataFrame column order.
"""

import numpy as np


def dtype_std():
    """Return the NumPy dtype for standard firmware.

    Fields
    ------
    tmicros : i4
        Timestamp in microseconds (32-bit signed int, wraps at ~71 min).
    d0-d2, e0-e2, f0-f2, c0-c2 : i2
        Raw ADC channel readings (16-bit signed int).
    a0-a2, b0-b2 : i2
        Processed analog channel readings.
    gx-gz : f4
        Gyroscope X/Y/Z (32-bit float).
    wx-wz : f4
        Angular velocity X/Y/Z.
    Tx-Tz : f4
        Temperature X/Y/Z.
    """
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("c0", "i2"),
            ("c1", "i2"),
            ("c2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            ("gx", "f4"),
            ("gy", "f4"),
            ("gz", "f4"),
            ("wx", "f4"),
            ("wy", "f4"),
            ("wz", "f4"),
            ("Tx", "f4"),
            ("Ty", "f4"),
            ("Tz", "f4"),
        ]
    )

def dtype_i2c_1_legacy():
    """Return the NumPy dtype for I2C v1 (legacy) firmware.

    NOTE: This is for firmware versions <0.10. In v0.10 and later, the
    c0 ADC channel was removed for i2c_1.

    Extends the standard dtype with a single set of I2C sensor channels
    (gx_i2c-gz_i2c, wx_i2c-wz_i2c, Tx_i2c-Tz_i2c). The c1/c2 ADC
    channels present in std are absent here.
    """
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("c0", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            ("gx", "f4"),
            ("gy", "f4"),
            ("gz", "f4"),
            ("wx", "f4"),
            ("wy", "f4"),
            ("wz", "f4"),
            ("Tx", "f4"),
            ("Ty", "f4"),
            ("Tz", "f4"),
            ("gx_i2c", "f4"),
            ("gy_i2c", "f4"),
            ("gz_i2c", "f4"),
            ("wx_i2c", "f4"),
            ("wy_i2c", "f4"),
            ("wz_i2c", "f4"),
            ("Tx_i2c", "f4"),
            ("Ty_i2c", "f4"),
            ("Tz_i2c", "f4"),
        ]
    )

def dtype_i2c_1():
    """Return the NumPy dtype for I2C v1 firmware.

    Extends the standard dtype with a single set of I2C sensor channels
    (gx_i2c-gz_i2c, wx_i2c-wz_i2c, Tx_i2c-Tz_i2c). The c0-c2 ADC
    channels present in std are absent here.
    """
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            # 2-byte alignment pad: 15 uint16_t fields place the struct at offset 34,
            # but float requires 4-byte alignment (offset 36). The C compiler inserts
            # this gap automatically; numpy packed dtypes do not.
            ("_pad", "u2"),
            ("gx", "f4"),
            ("gy", "f4"),
            ("gz", "f4"),
            ("wx", "f4"),
            ("wy", "f4"),
            ("wz", "f4"),
            ("Tx", "f4"),
            ("Ty", "f4"),
            ("Tz", "f4"),
            ("gx_i2c", "f4"),
            ("gy_i2c", "f4"),
            ("gz_i2c", "f4"),
            ("wx_i2c", "f4"),
            ("wy_i2c", "f4"),
            ("wz_i2c", "f4"),
            ("Tx_i2c", "f4"),
            ("Ty_i2c", "f4"),
            ("Tz_i2c", "f4"),
        ]
    )


def dtype_i2c_2():
    """Return the NumPy dtype for I2C v2 firmware.

    Extends the standard dtype with two independent I2C sensor channels
    (_i2c_c and _i2c_e suffixes). The e0-e2 and c0-c2 ADC channels
    present in std/i2c_1 are absent here.
    """
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            ("gx", "f4"),
            ("gy", "f4"),
            ("gz", "f4"),
            ("wx", "f4"),
            ("wy", "f4"),
            ("wz", "f4"),
            ("Tx", "f4"),
            ("Ty", "f4"),
            ("Tz", "f4"),
            ("gx_i2c_c", "f4"),
            ("gy_i2c_c", "f4"),
            ("gz_i2c_c", "f4"),
            ("wx_i2c_c", "f4"),
            ("wy_i2c_c", "f4"),
            ("wz_i2c_c", "f4"),
            ("Tx_i2c_c", "f4"),
            ("Ty_i2c_c", "f4"),
            ("Tz_i2c_c", "f4"),
            ("gx_i2c_e", "f4"),
            ("gy_i2c_e", "f4"),
            ("gz_i2c_e", "f4"),
            ("wx_i2c_e", "f4"),
            ("wy_i2c_e", "f4"),
            ("wz_i2c_e", "f4"),
            ("Tx_i2c_e", "f4"),
            ("Ty_i2c_e", "f4"),
            ("Tz_i2c_e", "f4"),
        ]
    )


def dtype_stream_1():
    """Return the NumPy dtype for the record of the first 0.13 test builds with
    a FIFO IMU on Port C (i2c_320g, i2c_45686), which still had the SPI IMU in
    the record. Later 0.13 builds use dtype_analog_15.

    The RECORD block of the block stream format (see ``stream_format``): the
    i2c_1 layout without the I2C IMU fields, whose data is in FIFO blocks.
    The c0-c2 ADC channels are absent (Port C is the IMU's I2C bus).
    """
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            # 2-byte alignment pad before the floats (same as i2c_1).
            ("_pad", "u2"),
            ("gx", "f4"),
            ("gy", "f4"),
            ("gz", "f4"),
            ("wx", "f4"),
            ("wy", "f4"),
            ("wz", "f4"),
            ("Tx", "f4"),
            ("Ty", "f4"),
            ("Tz", "f4"),
        ]
    )


def dtype_stream_2():
    """Return the NumPy dtype for the record of the first 0.13 test builds with
    FIFO IMUs on Ports C and E (i2c_320g_2, i2c_45686_2), which still had the
    SPI IMU in the record. Later 0.13 builds use dtype_analog_12.

    The RECORD block of the block stream format (see ``stream_format``): the
    i2c_2 layout without the I2C IMU fields, whose data is in FIFO blocks.
    The c0-c2 and e0-e2 ADC channels are absent (both ports are IMUs).
    """
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            ("gx", "f4"),
            ("gy", "f4"),
            ("gz", "f4"),
            ("wx", "f4"),
            ("wy", "f4"),
            ("wz", "f4"),
            ("Tx", "f4"),
            ("Ty", "f4"),
            ("Tz", "f4"),
        ]
    )


# ── Records from firmware 0.13: every variant writes the block stream format ──
# The internal SPI ICM-20948 (and any ICM-45686 / LSM6DSV320X) is in FIFO
# blocks, so the records hold only the analog channels, plus the deprecated
# polled I2C ICM-20948 on i2c_1 / i2c_2.

def dtype_analog_18():
    """Return the NumPy dtype for the std record from firmware 0.13: 18 analog channels."""
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("c0", "i2"),
            ("c1", "i2"),
            ("c2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
        ]
    )


def dtype_analog_15():
    """Return the NumPy dtype for the record of a variant with a FIFO IMU on
    Port C (i2c_320g, i2c_45686), from firmware 0.13: 15 analog channels (no
    c0-c2)."""
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            # Struct padding to a multiple of 4 bytes (the uint32 microtime).
            ("_pad", "u2"),
        ]
    )


def dtype_analog_12():
    """Return the NumPy dtype for the record of a variant with FIFO IMUs on
    Ports C and E (i2c_320g_2, i2c_45686_2), from firmware 0.13: 12 analog
    channels (no c0-c2, e0-e2)."""
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
        ]
    )


def dtype_analog_15_i2c():
    """Return the NumPy dtype for the i2c_1 record from firmware 0.13: 15 analog
    channels and the polled I2C ICM-20948 on Port C."""
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("e0", "i2"),
            ("e1", "i2"),
            ("e2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            # 2-byte alignment pad before the floats (same as i2c_1).
            ("_pad", "u2"),
            ("gx_i2c", "f4"),
            ("gy_i2c", "f4"),
            ("gz_i2c", "f4"),
            ("wx_i2c", "f4"),
            ("wy_i2c", "f4"),
            ("wz_i2c", "f4"),
            ("Tx_i2c", "f4"),
            ("Ty_i2c", "f4"),
            ("Tz_i2c", "f4"),
        ]
    )


def dtype_analog_12_i2c2():
    """Return the NumPy dtype for the i2c_2 record from firmware 0.13: 12 analog
    channels and the polled I2C ICM-20948s on Ports C and E."""
    return np.dtype(
        [
            ("tmicros", "i4"),
            ("d0", "i2"),
            ("d1", "i2"),
            ("d2", "i2"),
            ("f0", "i2"),
            ("f1", "i2"),
            ("f2", "i2"),
            ("a0", "i2"),
            ("a1", "i2"),
            ("a2", "i2"),
            ("b0", "i2"),
            ("b1", "i2"),
            ("b2", "i2"),
            ("gx_i2c_c", "f4"),
            ("gy_i2c_c", "f4"),
            ("gz_i2c_c", "f4"),
            ("wx_i2c_c", "f4"),
            ("wy_i2c_c", "f4"),
            ("wz_i2c_c", "f4"),
            ("Tx_i2c_c", "f4"),
            ("Ty_i2c_c", "f4"),
            ("Tz_i2c_c", "f4"),
            ("gx_i2c_e", "f4"),
            ("gy_i2c_e", "f4"),
            ("gz_i2c_e", "f4"),
            ("wx_i2c_e", "f4"),
            ("wy_i2c_e", "f4"),
            ("wz_i2c_e", "f4"),
            ("Tx_i2c_e", "f4"),
            ("Ty_i2c_e", "f4"),
            ("Tz_i2c_e", "f4"),
        ]
    )


FIRMWARE_DTYPES = {
    "analog_18": dtype_analog_18,
    "analog_15": dtype_analog_15,
    "analog_12": dtype_analog_12,
    "analog_15_i2c": dtype_analog_15_i2c,
    "analog_12_i2c2": dtype_analog_12_i2c2,
    "stream_2": dtype_stream_2,
    "stream_1": dtype_stream_1,
    "i2c_2": dtype_i2c_2,
    "i2c_1": dtype_i2c_1,
    "i2c_1_legacy": dtype_i2c_1_legacy,
    "std": dtype_std,
}
"""Maps firmware key string to its dtype constructor function."""

COLUMN_ORDERS = {
    "analog_18": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "c0",
        "c1",
        "c2",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
    ],
    "analog_15": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
    ],
    "analog_12": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "f0",
        "f1",
        "f2",
    ],
    "analog_15_i2c": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
        "gx_i2c",
        "gy_i2c",
        "gz_i2c",
        "wx_i2c",
        "wy_i2c",
        "wz_i2c",
        "Tx_i2c",
        "Ty_i2c",
        "Tz_i2c",
    ],
    "analog_12_i2c2": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "f0",
        "f1",
        "f2",
        "gx_i2c_c",
        "gy_i2c_c",
        "gz_i2c_c",
        "wx_i2c_c",
        "wy_i2c_c",
        "wz_i2c_c",
        "Tx_i2c_c",
        "Ty_i2c_c",
        "Tz_i2c_c",
        "gx_i2c_e",
        "gy_i2c_e",
        "gz_i2c_e",
        "wx_i2c_e",
        "wy_i2c_e",
        "wz_i2c_e",
        "Tx_i2c_e",
        "Ty_i2c_e",
        "Tz_i2c_e",
    ],
    "stream_2": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "f0",
        "f1",
        "f2",
        "gx",
        "gy",
        "gz",
        "wx",
        "wy",
        "wz",
        "Tx",
        "Ty",
        "Tz",
    ],
    "stream_1": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
        "gx",
        "gy",
        "gz",
        "wx",
        "wy",
        "wz",
        "Tx",
        "Ty",
        "Tz",
    ],
    "i2c_2": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "f0",
        "f1",
        "f2",
        "gx",
        "gy",
        "gz",
        "wx",
        "wy",
        "wz",
        "Tx",
        "Ty",
        "Tz",
        "gx_i2c_c",
        "gy_i2c_c",
        "gz_i2c_c",
        "wx_i2c_c",
        "wy_i2c_c",
        "wz_i2c_c",
        "Tx_i2c_c",
        "Ty_i2c_c",
        "Tz_i2c_c",
        "gx_i2c_e",
        "gy_i2c_e",
        "gz_i2c_e",
        "wx_i2c_e",
        "wy_i2c_e",
        "wz_i2c_e",
        "Tx_i2c_e",
        "Ty_i2c_e",
        "Tz_i2c_e",
    ],
    "i2c_1": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
        "gx",
        "gy",
        "gz",
        "wx",
        "wy",
        "wz",
        "Tx",
        "Ty",
        "Tz",
        "gx_i2c",
        "gy_i2c",
        "gz_i2c",
        "wx_i2c",
        "wy_i2c",
        "wz_i2c",
        "Tx_i2c",
        "Ty_i2c",
        "Tz_i2c",
    ],
    "i2c_1_legacy": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "c0",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
        "gx",
        "gy",
        "gz",
        "wx",
        "wy",
        "wz",
        "Tx",
        "Ty",
        "Tz",
        "gx_i2c",
        "gy_i2c",
        "gz_i2c",
        "wx_i2c",
        "wy_i2c",
        "wz_i2c",
        "Tx_i2c",
        "Ty_i2c",
        "Tz_i2c",
    ],
    "std": [
        "tmicros",
        "t",
        "a0",
        "a1",
        "a2",
        "b0",
        "b1",
        "b2",
        "c0",
        "c1",
        "c2",
        "d0",
        "d1",
        "d2",
        "e0",
        "e1",
        "e2",
        "f0",
        "f1",
        "f2",
        "gx",
        "gy",
        "gz",
        "wx",
        "wy",
        "wz",
        "Tx",
        "Ty",
        "Tz",
    ],
}
"""Maps firmware key string to the preferred DataFrame column order."""
