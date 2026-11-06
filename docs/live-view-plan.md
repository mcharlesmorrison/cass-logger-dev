# Live View — Implementation Plan

Live sensor readout ("Live Stream") at the top of the Data panel of the Cass Logger app, at 20 Hz, via a new snapshot command.

This same file lives in all three repos (`cass-logger-fw`, `cass-logger-dev`, `cass-logger-app`) at `docs/live-view-plan.md`. If you change the protocol, update all three copies.

| Repo | Role | Section |
|---|---|---|
| `cass-logger-fw` | New `'k'` snapshot command, framing, cleanup. Firmware 0.11 → **0.12** | [Phase 1](#phase-1--firmware-cass-logger-fw) |
| `cass-logger-dev` | Frame parser + decoder, `snapshot()` / `get_snapshot()`, bench example. 0.1.0 → **0.2.0** | [Phase 2](#phase-2--host-library-cass-logger-dev) |
| `cass-logger-app` | Device lock, live polling thread, API endpoints, Live Stream section in the Data panel | [Phase 3](#phase-3--app-cass-logger-app) |

`cass-utils` needs no changes for v1. It only matters later, for calibrated units (see [Later](#later-not-in-v1)).

---

## Decisions (v1)

1. **Polling, not continuous streaming.** The host sends `'k'` and the device replies with one framed record. The device never sends anything it wasn't asked for, which is what lets live view share the ports with other commands (decision 5).
2. **The rate is a fixed 20 Hz.** Every frame carries the device `microtime`, so host timing jitter doesn't affect the data, and the app fetches frames in batches whatever the rate, so a lower rate saves nothing worth a setting.
3. **Live view only runs in the app, as the "Live Stream" section at the top of the Data panel.** It stops automatically when the user switches to another panel (Device, Field Test) and resumes on return; disconnect and a firmware flash stop it for good. There's no polling thread in `cass-logger-dev`. That repo only provides the protocol (parser, decoder, single-snapshot call) and a small bench script for firmware development.
4. **No live view while recording.** If the device is recording, it answers `'k'` with a `LOGGING` flag and no sensor data. It never reads sensors from `loop()` while the 500 Hz logger interrupt is running. The app refuses to start live view while recording, and stops it as soon as a frame comes back with `LOGGING` set (e.g. someone pressed the hardware record button).
5. **Other device commands run alongside live view.** A single device I/O lock in the app makes every command and every live poll take turns on the serial ports. While a command such as setting the RTC holds the lock, live view skips its polls and shows "paused". The rate doesn't make this possible: polling plus the lock does, and it would work the same at 20 Hz. See [Why a lock](#why-a-lock-and-not-just-a-lower-rate).
6. **The payload is the raw `datrec` struct.** The host decodes it with the existing `FIRMWARE_DTYPES`, so all five variants work without new per-variant code.
7. **Values are raw.** ADC channels are counts (0–4095). IMU values are mg, dps and mag units, as the firmware writes them today.

### Why a lock and not just a lower rate

Every reply, from both commands and snapshots, comes back on the same data port (`Serial`). Existing replies have no framing: the host reads "everything in the buffer until an `'x'`" (e.g. `get_fw_ver`), or a fixed-size raw block (`'t'` transfers). Several existing commands also close and reopen the ports (`_flush_all`, `_close_serial`).

If a snapshot frame arrived in the middle of another reply, that reply would be corrupted, and a frame can easily contain the byte `'x'`. Lowering the rate only makes a collision less likely, not impossible.

The lock removes the problem entirely: a poll and a command can never be in flight at the same time. Because the device only speaks when spoken to, holding the lock for the length of one request/reply is enough.

---

## Protocol spec (shared contract)

### Command

| Direction | Port | Bytes |
|---|---|---|
| host → device | command port (`SerialUSB1`) | `k` (0x6B), single byte, no terminator |
| device → host | data port (`Serial`) | one snapshot frame (below) |

Old firmware (≤ 0.11) silently ignores `k`. The host **must** use a timeout and check the firmware version before using it (see `supports_live_view`).

### Snapshot frame

All multi-byte fields are little-endian.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `sync` | `0xA5 0x5A` |
| 2 | 1 | `frame_type` | `0x01` = SNAPSHOT (firmware 0.12), `0x02` = SNAPSHOT_IMU (0.13), `0x03` = SNAPSHOT_PORTS (0.14+). Other values are reserved. |
| 3 | 1 | `flags` | see below |
| 4 | 2 | `payload_len` | `uint16`. SNAPSHOT: `sizeof(datrec)` (76 / 108 / 136); SNAPSHOT_IMU / SNAPSHOT_PORTS: see below; `0` when recording |
| 6 | N | `payload` | SNAPSHOT: raw `datrec` bytes, layout per `include/deviceConfig.h`. SNAPSHOT_IMU / SNAPSHOT_PORTS: see below |
| 6+N | 2 | `crc16` | CRC-16/CCITT-FALSE over bytes `[2, 6+N)`, i.e. type + flags + len + payload |

**Flags**

| Bit | Name | Meaning |
|---|---|---|
| 0 | `LOGGING` | Device is recording. `payload_len` is 0 and no sensors were read. The host must stop live view. |
| 1 | `SENSOR_INIT_FAILED` | A sensor fault: logging is refused. 0.14+: right now (an unknown I2C device, an IMU that won't set up, an IMU wired for the other port or an IMU on an analog-only port, or the internal IMU failed at boot); it clears when the port is fixed. 0.12–0.13: `sensorInitOk == false` at boot. Some IMU fields may be stale or zero. |
| 2 | `PORT_WARNING` | 0.14+. A port IMU is running without its magnetometer (recorded 6-DoF). |
| 3 | `IMU_FAILING` | 0.14+, with `LOGGING`. A port IMU stopped answering during this recording (its FIFO blocks carry `READ_FAILED`). |
| 4–7 | reserved | Must be 0 |

**CRC-16/CCITT-FALSE:** poly `0x1021`, init `0xFFFF`, not reflected, no final XOR. In Python this is `binascii.crc_hqx(data, 0xFFFF)`. Check value: `crc("123456789") == 0x29B1`.

**Parser rules (host):**
- Scan for `A5 5A`, and discard any bytes before it. Debug text can still show up on the data port.
- If `payload_len > 512`, it's a false sync: drop one byte and rescan.
- If the CRC doesn't match: drop one byte and rescan.
- If `payload_len == 0` without `LOGGING`, that's a protocol error.
- If `payload_len != dtype.itemsize` for the connected firmware variant, raise a clear variant-mismatch error. Don't decode it.

### SNAPSHOT_PORTS payload (firmware 0.14+)

The SNAPSHOT_IMU payload with what is on ports C and E between the record and the readings. From 0.14 one firmware detects the ports at runtime, so the variant no longer says which `datrec` fields are real.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `record_len` | `uint16`, `sizeof(datrec)`: 40 (all 18 channels) |
| 2 | `record_len` | `record` | raw `datrec` bytes, the 18-channel layout (`dtype_analog_18`) |
| 2+R | 4 | `analog_mask` | `uint32`, bit *n* set: field `a`*n* is a real analog channel; the fields of a port with an I2C device are 0. As in the stream format's `FILE_HEADER` |
| 6+R | 3 | `port_c` | `state`, `device_type`, `i2c_addr` (`StreamPortInfo`, `docs/stream-format.md`): 0 analog, 1 IMU, 2 IMU without its magnetometer, 3 fault, 4 IMU wired for the other port (a fault too) |
| 9+R | 3 | `port_e` | the same for port E |
| 12+R | 1 | `imu_on_analog` | bit *n* set: an I2C IMU is plugged into analog-only port `'A'` + *n* (A, B, D or F; the C and E bits are 0). A fault: `SENSOR_INIT_FAILED` is set |
| 13+R | 1 | `n_imu` | readings that follow (one per running FIFO IMU, at most 3) |
| 14+R | 52 × `n_imu` | readings | as in SNAPSHOT_IMU, below |

A port can change between snapshots (the logger re-checks ports with an I2C device every 2 s while idle), so read `analog_mask` and the port fields from every frame. A port that was analog at boot isn't probed again: sensors are only changed with the logger off. An IMU on an analog-only port is only looked for at boot.

### SNAPSHOT_IMU payload (firmware 0.13)

From 0.13 the IMUs aren't in the `datrec`: they're read through their FIFOs when recording (see `docs/stream-format.md`). So the snapshot carries the record and then each FIFO IMU's current reading, read from its registers, in physical units.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `record_len` | `uint16`, `sizeof(datrec)` |
| 2 | `record_len` | `record` | raw `datrec` bytes. The host picks the layout by variant and `record_len` |
| 2+R | 1 | `n_imu` | readings that follow (one per FIFO IMU, at most 3) |
| 3+R | 52 × `n_imu` | readings | below (`StreamLiveImu` in `lib/StreamFormat/StreamFormat.h`) |

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `source_id` | as in the stream format: 1 = port C, 2 = port E, 3 = internal |
| 1 | 1 | `device_type` | `DEVICE_*`: 1 ICM-20948, 2/3 ICM-45686, 4 LSM6DSV320X |
| 2 | 1 | `port` | `'S'` (internal SPI), `'C'` or `'E'` |
| 3 | 1 | `valid` | bit 0 accel, 1 gyro, 2 mag, 3 high-g: which readings were taken |
| 4 | 12 | `accel` | 3 × `float`, mg (LSM6DSV320X: low-g) |
| 16 | 12 | `gyro` | 3 × `float`, dps |
| 28 | 12 | `mag` | 3 × `float`, µT |
| 40 | 12 | `hg` | 3 × `float`, mg, LSM6DSV320X high-g accelerometer |

The host (`cass_logger_dev.live`) decodes the readings into the column names the `datrec` used, so the app's channel lists keep working: `gx gy gz wx wy wz Tx Ty Tz` for the internal IMU, the same with `_i2c_c` / `_i2c_e` for ports C / E, and `hgx hgy hgz` + suffix for high-g. A reading not in `valid`, or an IMU missing from the frame, is NaN. `live_columns(fw_ver)` lists every column a variant's snapshots will have.

---

## Phase 1 — Firmware (`cass-logger-fw`)

Branch: `live-stream` (used instead of `feature/live-view`). Version: `FW_VERSION "0.12"` in `include/variantConfig.h`, plus a changelog line in the `src/main.cpp` header.

**Status:** 1.1–1.3 are implemented and all five variants build. The 1.4 bench checks are still open, and they need `live_view_ex.py` from Phase 2.

### 1.1 Snapshot command
New `include/live.h` + `src/live.cpp`, following the `commands.cpp` / `fileIO.cpp` pattern:

- `uint16_t crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc = 0xFFFF);`
- `void sendSnapshot();`
  - **If `isLogging`:** send a frame with `flags = LOGGING` and `payload_len = 0`. **Don't touch any sensor or bus.** Only write it if `Serial.availableForWrite() >= 8`, so a host that stopped reading can never block `loop()` and starve `CheckLogger()` (the ring buffer only holds 100 ms).
  - **Else:** `datrec rec; myCollector(&rec);`. Calling the collector directly is safe here because the logger's timer interrupt only runs while recording. `isLogging` only changes in `loop()` context (via `toggleLogging()` / `'r'` / `'q'`), so it can't flip in the middle of a snapshot.
  - If `!sensorInitOk`, set `flags |= SENSOR_INIT_FAILED`.
  - Build the frame in a local buffer (6 + 136 + 2 = 144 bytes max), compute the CRC, then write it once with `Serial.write(buf, n)`.
- `src/main.cpp`: add `if (ch == 'k') sendSnapshot();` to the dispatch list.

### 1.2 Remove serial prints from the collector
`src/collector.cpp` calls `Serial.println("didn't get ... data!!!")`. It does this inside the 500 Hz interrupt while recording, and now also on the snapshot path. The host parser skips stray text, but it's noise on the data port and bad practice in an ISR.
- Replace each one with a `volatile uint32_t` failure counter (`n_fail_spi`, `n_fail_i2c_c`, `n_fail_i2c_e`), declared in `collector.h`.
- Print the counters in `myBinaryDisplay` under `#ifdef TIMING_DEBUG`, next to the existing timing output.

### 1.3 Overflow visibility
`GetStatus()` in `src/commands.cpp`: add `Serial.printf("Buffer overflows: %lu\n", tsp->bufferoverflows);`. The recording test needs it.

### Implementation notes for host-side code
- **Total frame sizes:** 84 / 116 / 144 bytes (std / i2c_1 & i2c_45686 / i2c_2 & i2c_45686_2), or 8 bytes for a `LOGGING` frame.
- **CRC:** the firmware's `crc16_ccitt()` gives `0x29B1` for "123456789". It also matches `binascii.crc_hqx(data, 0xFFFF)` on random data.
- **`microtime` in a snapshot** is `filemicros`, which isn't reset when recording stops. Outside a recording it counts µs since boot, or since the start of the most recent recording. It's a `uint32`, so it **wraps every ~71.6 min**. Use it for spacing between frames, and handle the wrap (e.g. a `(b - a) & 0xFFFFFFFF` delta). Don't treat it as an absolute time.
- **A `LOGGING` reply can be dropped.** While recording, the device skips the reply if its USB TX buffer doesn't have room for it. If the host stops reading `Serial` during a recording, `'k'` can time out instead of returning `LOGGING`. In normal polling this shouldn't happen. The app's "10 timeouts → not responding" path covers it anyway.
- **Unwritten fields are 0.** The payload struct is zero-initialized before the collector fills it, so any field the collector doesn't write reads as 0, not leftover memory.
- If an ICM-20948 has no new sample when a snapshot is taken, its fields repeat the previous reading (same as during recording).
- The `GetStatus()` (`s`) output now includes the line `Buffer overflows: N`, between "Max Collection delay" and "Average Write Time".

### 1.4 Firmware acceptance
- [x] `pio run -e std`, `-e i2c_1`, `-e i2c_2`, `-e i2c_45686`, `-e i2c_45686_2` all build.
- [ ] Bench test with `cass-logger-dev/examples/live_view_ex.py` on at least `std` and one I2C variant. Values respond to touching the inputs, and the IMU responds to motion.
- [ ] While not recording, the bench script runs 10 minutes at 20 Hz without CRC errors.
- [ ] **Recording interaction:** run the bench script, then press the hardware record button. The next frame has `LOGGING` set with `payload_len == 0`. Leave the script sending `'k'` at 20 Hz while recording for 10 minutes, then stop and send `s`. Overflows must be 0. Parse the `.bin` and confirm `diff(tmicros)` is about 2000 µs (500 Hz) everywhere, with no gaps.
- [ ] Existing commands (download, list, RTC, device ID) are unaffected.

---

## Phase 2 — Host library (`cass-logger-dev`)

Branch: `live-stream`. Version: `0.2.0` in `pyproject.toml`. Tag `v0.2.0` after merge, since the app pins to it.

This repo owns the **protocol**: framing, decoding, and a single snapshot request. It doesn't own the polling loop or the locking. Those are app-only (Phase 3).

### 2.1 `cass_logger_dev/live.py` (new)
Pure parsing code with no serial I/O, so it can be unit-tested without hardware.

- Constants: `SYNC = b"\xA5\x5A"`, `FRAME_SNAPSHOT = 0x01`, `FLAG_LOGGING = 0x01`, `FLAG_SENSOR_INIT_FAILED = 0x02`, `MAX_PAYLOAD = 512`, `MIN_LIVE_FW = (0, 12)`.
- `crc16(data: bytes) -> int` returns `binascii.crc_hqx(data, 0xFFFF)`.
- `class FrameParser` with `feed(chunk: bytes) -> list[RawFrame]`. It keeps an internal `bytearray` and applies the parser rules above. `RawFrame` holds `frame_type`, `flags`, `payload`.
- `@dataclass Snapshot` with fields `host_time: float` (`time.monotonic()`), `logging: bool`, `sensor_init_failed: bool`, `values: dict[str, int | float] | None` (`None` when `logging`).
- `decode_payload(payload: bytes, fw_ver: str) -> dict`: look up `FIRMWARE_DTYPES[CassCommands.get_dtype_key(fw_ver)]()`, check that `itemsize == len(payload)` (otherwise raise `ValueError` describing the variant mismatch), `np.frombuffer` it, and return `{name: python_scalar}` in dtype field order.
- `supports_live_view(fw_ver: str) -> bool`: parse the leading `X.Y` from `"0.12-std"` and compare it against `MIN_LIVE_FW`.

### 2.2 `CassCommands` additions (`cass_commands.py`)
- `snapshot(self, fw_ver: str, timeout: float = 0.25) -> Snapshot` is the **low-level call for repeated polling**:
  - send `b"k"` on `ser_command`;
  - feed `ser_data` bytes into a `FrameParser` until one frame arrives or `timeout` passes;
  - decode it;
  - on timeout, call `ser_data.reset_input_buffer()` so a late frame can't leak into the next command's reply, then raise `TimeoutError`.

  It must **not** call `_flush_all()` or `_close_serial()`, so the ports stay open between polls. It goes through the `ser_data` / `ser_command` properties, which reopen the ports if another command closed them.
- `get_snapshot(self, fw_ver: str | None = None, timeout: float = 0.5) -> Snapshot` takes a single snapshot in the existing per-command style: `_flush_all()`, then `snapshot()`, then `_close_serial()`. If `fw_ver` is `None`, call `get_fw_ver()`.

### 2.3 Read timeouts (recommended; can be a separate PR)
`get_fw_ver`, `get_device_ID`, `get_RTC_time` and `get_rtc_install_timestamp` currently wait forever (`while b"x" not in ...`). The app calls these on connect and from the Device panel. While live view shares the ports, a hung command means a frozen live view too. Add a `_read_until(terminator: bytes, timeout: float = 2.0) -> bytes` helper that raises `TimeoutError`, and use it in those four methods.

### 2.4 Tests, example, docs
- `tests/test_live.py` (pytest; add `[project.optional-dependencies] dev = ["pytest>=8"]`). Build synthetic frames from the spec and cover:
  - a clean frame;
  - garbage or ASCII text before the sync bytes;
  - a frame split across many `feed()` calls;
  - two frames in one chunk;
  - a bad CRC followed by a good frame;
  - a false `A5 5A` inside the payload;
  - a `LOGGING` frame with `payload_len == 0`;
  - `payload_len > 512`;
  - a variant mismatch in `decode_payload`;
  - `supports_live_view("0.11-std")` is false and `("0.12-i2c_2")` is true;
  - the CRC check value `0x29B1`.
- `examples/live_view_ex.py` is the **firmware bench tool** for Phase 1.4, not a product feature. It's a plain loop around `snapshot()` at a `--rate` argument that prints a refreshing table of all channels, flags and CRC/timeout counts. Ctrl-C to stop.
- `README.md`: add one row for `get_snapshot()`, and note that live view is driven from the app.

### 2.5 Host library acceptance
- [ ] `pytest` passes.
- [ ] `live_view_ex.py` decodes correctly on every variant available on the bench.
- [ ] Against 0.11 firmware, `get_snapshot()` raises `TimeoutError` within its timeout, with no hang.

---

## Phase 3 — App (`cass-logger-app`)

Branch: `live-stream`. Follow the existing conventions: tabs in `gui/`, type hints, and `ok()`/`err()` results.

### 3.1 Dependency
`pyproject.toml`: pin `cass-logger-dev @ git+https://github.com/mcharlesmorrison/cass-logger-dev@v0.2.0` once that tag exists. It's currently unpinned on `main`. Until then, develop against the local checkout: `pip install -e ../cass-logger-dev`.

### 3.2 Device I/O lock (`gui/services/cass_service.py`)
pywebview runs each JS→Python call on its own thread, so device calls can overlap. The app **already has** this lock: `CassService._op_lock` (an `RLock`), used as `with self._svc.lock():` around the fw / ID / RTC / install-timestamp calls in `main_api.py`. Live view reuses it; there's no second lock and no proxy.

- Wrap the calls that were still unlocked: `list_files`, `start_download` (the whole download), `delete_all_files`.
- `check_alive()`: return `True` while live view runs, and `acquire(blocking=False)`; if the lock is busy, a command is in progress, so the device is alive: return `True` without touching the port. Otherwise run the existing probe and release the lock. This is what makes it safe to hold the lock for a whole download.

### 3.3 Live polling (`gui/services/cass_service.py`)
- State: `_live_thread`, `_live_stop: threading.Event`, `_live_frames: deque[Snapshot]` (`maxlen` 600), `_live_paused_since: Optional[float]`, `_live_missed: int`, `_live_stopped_reason: Optional[str]`.
- `start_live(rate_hz: float) -> tuple[bool, str]`:
  1. Require a connection and `supports_live_view(self._fw_ver)`. Otherwise return "Live view needs firmware 0.12 or newer — update it below."
  2. Take one synchronous `snapshot()` under the lock. If `logging` is set, return "The logger is recording — stop recording to use live view."
  3. Start the daemon thread.
- Each tick of the thread:
  - `self._op_lock.acquire(timeout=0.02)`. If that fails, note `_live_paused_since`, skip the tick and continue. `paused` is only reported once that's over 0.5 s old, so a quick command doesn't flash the chip.
  - Otherwise call `cass.snapshot(fw_ver)`, then release.
  - `TimeoutError`: increment `_live_missed`. After 10 in a row, stop with reason "Device not responding".
  - `snapshot.logging`: stop with reason "Recording started — live view stopped".
  - `SerialException` / `OSError`: stop with reason "Device disconnected", then `disconnect()`.
  - Otherwise append the frame, clear `_live_paused_since`, and sleep the rest of the period.
  - On exit, the thread calls `_close_serial()` under the lock to go back to the normal "ports closed between commands" state.
- `stop_live()` is idempotent: set the event and join the thread (unless called from the thread itself, via `disconnect()`).
- `disconnect()`, `connect()` and `connect_manual()` call `stop_live()` first, **outside** the lock (the exiting thread needs it). `start_firmware_flash` already calls `disconnect()`, so a flash stops live view.
- `check_alive()` while live returns `True` without touching the port: the live thread is the probe. On unplug it either hits a serial error (→ `disconnect()`) or 10 timeouts (→ stops; the next status poll re-probes the closed port).
- Dev aid: `CASS_FAKE_DEVICE=<variant>` connects to a simulated logger (`gui/services/fake_device.py`) that answers snapshots with synthetic signals.

### 3.4 `gui/api/main_api.py`
- `start_live() -> dict` → `ok({"columns": [...], "rate_hz": 20})`. The rate is `LIVE_RATE_HZ` in `cass_service.py`.
- `stop_live() -> dict`.
- `get_live_frames() -> dict` drains the deque and returns a **columnar** batch to keep the JSON small:
  ```json
  {"live": true, "paused": false, "stopped_reason": null, "missed": 0,
   "sensor_init_failed": false,
   "t": [12.31, 12.41, ...],              // host seconds since start_live
   "device_us": [123456, ...],             // datrec microtime
   "values": {"d0": [...], "gx": [...], ...}}
  ```
  The conversion is a pure helper, `frames_to_batch(frames, t0)` in `cass_service.py`, so it can be tested. Floats are rounded to 3 decimals and NaN/inf become `null`.
- `start_firmware_flash`: covered by its existing `disconnect()` call.
- Window close (`gui/app.py`): `service.disconnect()` after `webview.start()` returns, so the ports are released.

### 3.5 Frontend
- `npm install uplot`. It draws on canvas and copes with 10–20 Hz × many series. Recharts (SVG) will stutter at that rate, so keep Recharts for the existing static plots.
- `src/types.ts`: add a `LiveBatch` interface and `start_live` / `stop_live` / `get_live_frames` to `PyApi`.
- `src/App.tsx`: `DataPanel` already gets `active={panel === 'data'}` (panels stay mounted and are only hidden with `display: none`, so the prop is how live view knows to stop). Also pass it the logger's `fw_ver` and a `flashing` state, mirrored from the existing `flashInProgressRef` so live view can't touch the port while a freshly flashed logger boots.
- `src/channels.ts` (new): the port / IMU column names and labels, moved out of `DataPanel.tsx` so both panels label channels the same way.
- `src/components/LiveView.tsx` (new): the "Live Stream" section at the top of `DataPanel.tsx`, above View Files. Logger Files (list/download/delete) is under File Management on the Device panel. The main job is a **bench check** (is each sensor in the right port, and does it respond?):
  - **Toolbar:** Start/Stop and status badges: **Live · 20 Hz**, **Paused — device busy**, **Sensor init failed**, and a missed-poll count when it's above 0.
  - **Collapsed until Start:** while stopped the card is just the toolbar (plus a one-line hint, or the reason it stopped). Stop collapses it again.
  - Everything is built from the `columns` that `start_live` returns, so it adapts to each variant: on I2C variants, ports C / E drop out of the port list and show up as IMU-C / IMU-E.
  - **IMU chart:** Accel / Gyro / Mag buttons (default Accel), plus IMU / IMU-C / IMU-E buttons when the variant has more than one IMU. One plot with x/y/z; the latest x/y/z values sit in the chart header.
  - **Port chart** below it: one-of A–F buttons, then 0 / 1 / 2 toggles for the port's signals (at least one stays on). Each shown signal gets its own plot, stacked in one figure (3 → three short plots, 1 → one tall plot), sharing the time axis (drawn on the bottom plot only) and the hover cursor. Each plot's label shows its latest value. Y is auto-scaled; values are ADC counts.
  - Both charts show the last 10 s (uPlot, legend off).
  - **Lifecycle:** the user's Start is remembered as intent. Live view stops when `active` goes false (another panel) and **resumes when the user comes back**. Disconnect, a firmware flash (tracked in `DevicePanel` through `onFlashingChange`) and any self-stop (`stopped_reason`) clear the intent. The reason is shown inline as a warning.
  - **Polling:** `get_live_frames()` every **200 ms**, whatever the device rate (each call returns every frame since the last one). The rolling 10 s window lives in refs; one state bump per poll re-renders the header values and each uPlot redraws from the refs with `setData()`.
  - Firmware too old: Start is disabled, with a message pointing at the Firmware section above.
  - **Shock potentiometer** checkbox under the signal toggles, per port: plots travel in mm, `(counts − zero) × stroke / 4095`, instead of counts. A **75 mm / 100 mm** toggle under the checkbox (shown only while it's ticked) sets the pot's stroke per port; default 100 mm (0.00002442 m per count). With it ticked, **Zero** sets each of the port's channels' offset to its mean over the last 0.5 s, and **Clear zero** removes it — zero with the bike unloaded, then sit on it to read sag. Display-only: nothing is sent to the logger or saved.
  - Collapsed hint next to Start: "Read sensor data from logger realtime". No footnote. (Known limit, not shown in the UI: at 20 Hz snapshots, short pulses such as reed / wheel-speed switches can be missed.)
- Device commands that can run during live view (anything still in flight when the user switches panels) queue behind the lock, and live view shows "paused" if one takes longer than 0.5 s. A download holds the lock throughout, so live view stays paused until it finishes.

### 3.6 Tests and docs
- `tests/test_live_service.py`, using a fake `CassCommands` whose `snapshot()` returns synthetic frames:
  - while one thread holds `_op_lock`, the live thread doesn't call `snapshot()` and sets `paused`;
  - a `logging` frame stops live view with the recording reason;
  - 10 timeouts in a row stop it with "not responding";
  - `check_alive` doesn't touch the port while live or while the lock is held;
  - `disconnect` stops live view;
  - the shape of `_frames_to_batch`.
- `CLAUDE.md`: amend "Serial ports are opened lazily per command and closed after each operation" to note that live view holds them open between `start_live` and `stop_live`. Document the `_op_lock` rule (all device I/O goes through it), and add `LiveView.tsx` to Key Files.

### 3.7 App acceptance
- [ ] Live view in the Data panel runs at 20 Hz for 10 minutes on macOS and Windows, and the UI stays smooth with all three port signals stacked.
- [ ] Switching to Device or Field Test stops live view (no `'k'` traffic); coming back resumes it. Stop collapses the section.
- [ ] Listing logger files during live view works. Live view shows "paused" if it takes a while and resumes, with no garbled values on either side.
- [ ] Starting live view while recording is refused. Pressing the record button during live view stops it within one poll with the recording message, and the downloaded file has no gaps.
- [ ] Unplugging during live view: the app shows disconnected within about 5 s with no hang. Plugging back in auto-reconnects.
- [ ] A device on 0.11 shows the "update firmware" message.
- [ ] `npm run build` and `pytest tests` pass.

---

## Sequencing

```
Phase 1 firmware ──────────────┐
                               ├─► bench integration (live_view_ex.py) ─► Phase 3 app ─► release
Phase 2.1/2.4 parser + tests ──┘         (needs both)                       (UI can start against a fake snapshot())
```

1. Firmware (Phase 1) and the host parser and its tests (2.1, 2.4) can run **in parallel**: the parser is tested against synthetic frames built from the spec.
2. Build `snapshot()` / `get_snapshot()` and the bench example (2.2, 2.4) against real hardware running the Phase 1 firmware, then pass the Phase 1.4 checks.
3. App lock, polling and UI (Phase 3). The lock (3.2) can land first as its own PR, because it fixes an existing race whether or not live view ships. The UI can be built before step 2 is done by using a fake `snapshot()` that generates sine waves.
4. **Merge and release order:**
   1. merge firmware;
   2. merge `cass-logger-dev` and tag `v0.2.0`;
   3. merge the app (pinned to `v0.2.0`);
   4. publish firmware 0.12 for all variants with `cass-logger-app/scripts/release_firmware.py`;
   5. release the app.

   Firmware has to be on R2 before the app ships, so users can update from the Device panel when live view asks them to.

## Rough effort

| Piece | Estimate |
|---|---|
| Firmware (1.1–1.3) + variant builds + bench/recording checks | 0.5 day |
| `cass-logger-dev` parser, `snapshot()`, tests, bench example | 0.5–1 day (+0.5 for 2.3 timeouts) |
| App lock + polling thread + API + tests | 1 day |
| App Live view card (uPlot charts, selectors, states) | 1–1.5 days |
| Cross-platform testing (Windows port holding) | 0.5 day |

## Later (not in v1)

- **Catch short pulses on digital channels.** Add a live-mode `IntervalTimer` that runs the collector at `SAMPLERATE` into a scratch record and tracks edges and min/max per digital channel, reset on each snapshot. The timer must stop before `StartLogging()`. This would add a new frame type or append fields; `payload_len` and `frame_type` leave room for both.
- **Calibrated units** using sensor calibrations from `cass-utils` (`sensor_db`).
- **Continuous streaming** (device-paced frames) if more than ~50 Hz is ever needed. The cleanest way is to send stream frames on `SerialUSB1` TX, which is unused today, so streaming never mixes with command replies on `Serial`.
- Gate the other non-ISR debug prints (`SetDBPrint(true)`, `enableDebugging()`) behind a build flag.
