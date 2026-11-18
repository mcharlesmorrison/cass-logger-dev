# Live View Protocol

How the host reads the logger's sensors live: one framed snapshot per request. `cass_logger_dev/live.py` implements the host side (`CassCommands.snapshot()` / `get_snapshot()`), and `examples/live_view_ex.py` polls it.

- **Polling, not streaming.** The host sends `k` and the device replies with one frame. The device never sends anything it wasn't asked for, so live polls and other commands can share the serial ports as long as they take turns: never send a command while a `k` reply is still due.
- **Not while recording.** A recording device answers `k` with the `LOGGING` flag and no sensor data. It doesn't read sensors for live view while the 500 Hz logger is running.
- **Raw values.** Analog channels are ADC counts (0–4095). IMU readings are mg, dps and µT.

Firmware 0.12 added the command; 0.13 and 0.14 added frame types. Firmware 0.11 and older ignore `k`.

## Command

| Direction | Port | Bytes |
|---|---|---|
| host → device | command port | `k` (0x6B), single byte, no terminator |
| device → host | data port | one snapshot frame (below) |

Old firmware (≤ 0.11) silently ignores `k`. The host **must** use a timeout and check the firmware version before using it (see `supports_live_view`).

## Snapshot frame

All multi-byte fields are little-endian.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `sync` | `0xA5 0x5A` |
| 2 | 1 | `frame_type` | `0x01` = SNAPSHOT (firmware 0.12), `0x02` = SNAPSHOT_IMU (0.13), `0x03` = SNAPSHOT_PORTS (0.14+). Other values are reserved. |
| 3 | 1 | `flags` | see below |
| 4 | 2 | `payload_len` | `uint16`. SNAPSHOT: the record size (76 / 108 / 136); SNAPSHOT_IMU / SNAPSHOT_PORTS: see below; `0` when recording |
| 6 | N | `payload` | SNAPSHOT: one raw record, layout per `FIRMWARE_DTYPES` in `cass_logger_dev/firmware_structs.py`. SNAPSHOT_IMU / SNAPSHOT_PORTS: see below |
| 6+N | 2 | `crc16` | CRC-16/CCITT-FALSE over bytes `[2, 6+N)`, i.e. type + flags + len + payload |

**Flags**

| Bit | Name | Meaning |
|---|---|---|
| 0 | `LOGGING` | Device is recording. `payload_len` is 0 and no sensors were read. The host must stop live view. |
| 1 | `SENSOR_INIT_FAILED` | A sensor fault: logging is refused. 0.14+: right now (an unknown I2C device, an IMU that won't set up, I2C pull-ups with nothing answering (an IMU with a broken wire), an IMU wired for the other port, an IMU that stopped answering or an IMU on an analog-only port, or the internal IMU failed at boot); it clears when the port is fixed. 0.12–0.13: a sensor failed at boot. Some IMU fields may be stale or zero. |
| 2 | `PORT_WARNING` | 0.14+. A port IMU is running without its magnetometer (recorded 6-DoF). |
| 3 | `IMU_FAILING` | 0.14+, with `LOGGING`. A port IMU stopped answering during this recording (its FIFO blocks carry `READ_FAILED`). |
| 4–7 | reserved | Must be 0 |

**CRC-16/CCITT-FALSE:** poly `0x1021`, init `0xFFFF`, not reflected, no final XOR. In Python this is `binascii.crc_hqx(data, 0xFFFF)`. Check value: `crc("123456789") == 0x29B1`.

**Parser rules (host):**
- Scan for `A5 5A`, and discard any bytes before it. Debug text can still show up on the data port.
- If `payload_len > 512`, it's a false sync: drop one byte and rescan.
- If the CRC doesn't match: drop one byte and rescan.
- If `payload_len == 0` without `LOGGING`, that's a protocol error.
- SNAPSHOT: if `payload_len != dtype.itemsize` for the connected firmware variant, raise a clear variant-mismatch error. Don't decode it.
- SNAPSHOT_IMU / SNAPSHOT_PORTS: the same check on `record_len`. The payload must also end exactly after the `n_imu` readings (52 × `n_imu` bytes); otherwise it's malformed.

## SNAPSHOT_PORTS payload (firmware 0.14+)

The SNAPSHOT_IMU payload with what is on ports C and E between the record and the readings. From 0.14 one firmware detects the ports at runtime, so the firmware version no longer says which record fields are real.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `record_len` | `uint16`, the record size: 40 (all 18 channels) |
| 2 | `record_len` | `record` | one raw record, the 18-channel layout (`analog_18`) |
| 2+R | 4 | `analog_mask` | `uint32`, bit *n* set: the *n*-th analog field of the record (after `microtime`, in `analog_18` order `d0`…`d2`, `e0`…`e2`, `f0`…`f2`, `c0`…`c2`, `a0`…`a2`, `b0`…`b2`) is a real analog channel. So bit 0 is `d0` and bits 9–11 are `c0`–`c2`. The fields of a port with an I2C device are 0. As in the stream format's `FILE_HEADER` |
| 6+R | 4 | `pullup_mask` | `uint32`, bit *n* set: the *n*-th analog field (same bit order as `analog_mask`) has the logger's internal 22 kΩ pull-up on. As in `FILE_HEADER` (0.14: always 0) |
| 10+R | 3 | `port_c` | `state`, `device_type`, `i2c_addr`, as in the stream format's `FILE_HEADER`: 0 analog, 1 IMU, 2 IMU without its magnetometer, 3 fault (device_type and i2c_addr 0: I2C pull-ups but nothing answering, an IMU with a broken wire), 4 IMU wired for the other port, 5 an IMU found earlier that no longer answers (4 and 5 are faults too) |
| 13+R | 3 | `port_e` | the same for port E |
| 16+R | 1 | `imu_on_analog` | bit *n* set: an I2C IMU is plugged into analog-only port `'A'` + *n* (A, B, D or F; the C and E bits are 0). A fault: `SENSOR_INIT_FAILED` is set |
| 17+R | 1 | `n_imu` | readings that follow (one per running FIFO IMU, at most 3) |
| 18+R | 52 × `n_imu` | readings | as in SNAPSHOT_IMU, below |

A port can change between snapshots (the logger re-checks ports with an I2C device every 2 s while idle), so read `analog_mask`, `pullup_mask` and the port fields from every frame. A port that was analog at boot isn't probed again: sensors are only changed with the logger off. An IMU on an analog-only port is only looked for at boot.

## SNAPSHOT_IMU payload (firmware 0.13)

From 0.13 the IMUs aren't in the record: they're read through their FIFOs when recording (see `cass_logger_dev/stream_format.py`). So the snapshot carries the record and then each FIFO IMU's current reading, read from its registers, in physical units.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `record_len` | `uint16`, the record size |
| 2 | `record_len` | `record` | one raw record. The host picks the layout by variant and `record_len` |
| 2+R | 1 | `n_imu` | readings that follow (one per FIFO IMU, at most 3) |
| 3+R | 52 × `n_imu` | readings | below |

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `source_id` | as in the stream format: 1 = port C, 2 = port E, 3 = internal |
| 1 | 1 | `device_type` | 1 ICM-20948, 2/3 ICM-45686, 4 LSM6DSV320X |
| 2 | 1 | `port` | `'S'` (internal SPI), `'C'` or `'E'` |
| 3 | 1 | `valid` | bit 0 accel, 1 gyro, 2 mag, 3 high-g: which readings were taken |
| 4 | 12 | `accel` | 3 × `float`, mg (LSM6DSV320X: low-g) |
| 16 | 12 | `gyro` | 3 × `float`, dps |
| 28 | 12 | `mag` | 3 × `float`, µT |
| 40 | 12 | `hg` | 3 × `float`, mg, LSM6DSV320X high-g accelerometer |

The host (`cass_logger_dev.live`) decodes the readings into the column names the pre-0.13 records used: `gx gy gz wx wy wz Tx Ty Tz` for the internal IMU, the same with `_i2c_c` / `_i2c_e` for ports C / E, and `hgx hgy hgz` + suffix for high-g. A reading not in `valid`, or an IMU missing from the frame, is NaN. `live_columns(fw_ver)` lists every column a variant's snapshots will have.

## Notes for host code

- **Total frame sizes (SNAPSHOT, 0.12):** 84 / 116 / 144 bytes (std / one port IMU / two port IMUs), or 8 bytes for a `LOGGING` frame.
- **`microtime` in a snapshot** isn't reset when recording stops. Outside a recording it counts µs since boot, or since the start of the most recent recording. It's a `uint32`, so it **wraps every ~71.6 min**. Use it for spacing between frames, and handle the wrap (e.g. a `(b - a) & 0xFFFFFFFF` delta). Don't treat it as an absolute time.
- **A `LOGGING` reply can be dropped.** While recording, the device skips the reply if its USB transmit buffer doesn't have room for it. If the host stops reading the data port during a recording, `k` can time out instead of returning `LOGGING`. Treat repeated timeouts as "not responding".
- **Unwritten fields are 0.** Any record field the firmware doesn't fill reads as 0, not leftover memory.
- If an ICM-20948 has no new sample when a snapshot is taken, its fields repeat the previous reading (same as during recording).
