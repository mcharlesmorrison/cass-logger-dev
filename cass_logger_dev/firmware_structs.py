"""
NumPy dtype definitions and column orderings for each Cass Logger firmware variant.

Exports
-------
FIRMWARE_DTYPES : dict
    Maps firmware key strings ("std", "i2c_1", "i2c_2") to their dtype constructor.
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


FIRMWARE_DTYPES = {
    "i2c_2": dtype_i2c_2,
    "i2c_1": dtype_i2c_1,
    "i2c_1_legacy": dtype_i2c_1_legacy,
    "std": dtype_std,
}
"""Maps firmware key string to its dtype constructor function."""

COLUMN_ORDERS = {
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
